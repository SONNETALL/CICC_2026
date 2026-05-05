from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from op_computation_analysis import calculate_linear_macs
from train_mnist_mlp_single_hidden import SingleHiddenMLP


def parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-hidden MNIST MLP with optical fc matmul")
    parser.add_argument("--checkpoint", type=str, default="models/mnist_mlp_single_hidden_H64/best_mnist_mlp_single_hidden_H64.pt")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--bias", type=parse_bool_flag, default=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-type", type=str, default="int4", choices=["int4", "uint4"])
    parser.add_argument("--act-clip-max", type=float, default=4.0)
    parser.add_argument("--act-clip-fc1", type=float, default=None)
    parser.add_argument("--act-clip-fc2", type=float, default=None)
    parser.add_argument("--optical-layers", type=str, default="fc1")
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--print-batch-timing", type=parse_bool_flag, default=False)
    return parser.parse_args()


def parse_optical_layers(raw: str) -> set[str]:
    valid = {"fc1", "fc2"}
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}
    invalid = sorted(items - valid)
    if invalid:
        raise ValueError(f"Invalid optical layer names: {invalid}. Valid names are: {sorted(valid)}")
    return items


def build_layer_clip_map(args: argparse.Namespace) -> dict[str, float]:
    if args.act_clip_max <= 0:
        raise ValueError("--act-clip-max must be positive.")
    clip_map = {
        "fc1": args.act_clip_fc1 if args.act_clip_fc1 is not None else args.act_clip_max,
        "fc2": args.act_clip_fc2 if args.act_clip_fc2 is not None else args.act_clip_max,
    }
    invalid = {name: value for name, value in clip_map.items() if value <= 0}
    if invalid:
        raise ValueError(f"All layer-wise clip values must be positive, got: {invalid}")
    return clip_map


def resolve_path(project_root: Path, user_path: str) -> Path:
    path = Path(user_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def infer_model_config(checkpoint: dict[str, Any], fallback_hidden_size: int, fallback_bias: bool) -> tuple[int, bool]:
    hidden_size = checkpoint.get("hidden_size")
    bias = checkpoint.get("bias")
    args = checkpoint.get("args")
    if hidden_size is None and isinstance(args, dict) and "hidden_size" in args:
        hidden_size = args["hidden_size"]
    if bias is None and isinstance(args, dict) and "bias" in args:
        bias = args["bias"]
    return int(hidden_size if hidden_size is not None else fallback_hidden_size), bool(bias if bias is not None else fallback_bias)


def load_model_weights(checkpoint_path: Path, fallback_hidden_size: int, fallback_bias: bool) -> tuple[SingleHiddenMLP, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format. Expected dict.")

    hidden_size, bias = infer_model_config(checkpoint, fallback_hidden_size, fallback_bias)
    model = SingleHiddenMLP(hidden_size=hidden_size, bias=bias)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    metadata = {
        "epoch": checkpoint.get("epoch"),
        "val_acc": checkpoint.get("val_acc"),
        "hidden_size": hidden_size,
        "bias": bias,
    }
    return model, metadata


def load_optical_model() -> Any:
    try:
        from osimulator.api import load_gazelle_model
    except Exception as exc:
        raise ImportError(
            "Failed to import osimulator.api.load_gazelle_model. "
            "Please ensure osimulator is installed and available in this environment."
        ) from exc
    return load_gazelle_model()


def build_test_loader(data_root: Path, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(root=str(data_root), train=False, transform=transform, download=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_official_macs_profile(
    model: nn.Module,
    optical_layers: set[str],
) -> tuple[int, int, dict[str, int], dict[str, str]]:
    dummy_input = torch.randn(1, 1, 28, 28)
    total_macs_per_sample, macs_dict = calculate_linear_macs(model, dummy_input)
    layer_to_module_name = {"fc1": "fc1", "fc2": "fc2"}
    missing_modules = [name for name in layer_to_module_name.values() if name not in macs_dict]
    if missing_modules:
        raise RuntimeError(
            "Official MACs calculation did not find expected modules: "
            f"{missing_modules}. Found modules: {sorted(macs_dict.keys())}"
        )
    optical_macs_per_sample = sum(int(macs_dict[layer_to_module_name[layer]]) for layer in optical_layers)
    return int(total_macs_per_sample), int(optical_macs_per_sample), macs_dict, layer_to_module_name


def quantize_activation_for_optical(
    x: torch.Tensor,
    *,
    input_type: str,
    clip_max: float,
) -> tuple[np.ndarray, float, int, str]:
    if clip_max <= 0:
        raise ValueError("act-clip-max must be positive.")

    x_cpu = x.detach().to(torch.float32).cpu()
    if input_type == "uint4":
        x_clip = torch.clamp(x_cpu, min=0.0, max=clip_max)
        scale = clip_max / 15.0
        q = torch.round(x_clip / scale).clamp(0, 15).to(torch.int32)
        return q.numpy(), float(scale), 0, "uint4"

    if input_type == "int4":
        x_clip = torch.clamp(x_cpu, min=-clip_max, max=clip_max)
        scale = clip_max / 7.0
        q_signed = torch.round(x_clip / scale).clamp(-8, 7).to(torch.int32)
        zero_point = 8
        return (q_signed + zero_point).numpy(), float(scale), zero_point, "uint4"

    raise ValueError(f"Unsupported input_type: {input_type}")


def quantize_weight_to_int4_symmetric(w: torch.Tensor) -> tuple[np.ndarray, float]:
    w_cpu = w.detach().to(torch.float32).cpu()
    max_abs = torch.max(torch.abs(w_cpu)).item()
    scale = 1.0 if max_abs == 0.0 else max_abs / 7.0
    q = torch.round(w_cpu / scale).clamp(-8, 7).to(torch.int32)
    return q.numpy(), float(scale)


def build_weight_cache(model: SingleHiddenMLP) -> dict[str, tuple[np.ndarray, float]]:
    return {
        "fc1": quantize_weight_to_int4_symmetric(model.fc1.weight.transpose(0, 1).contiguous()),
        "fc2": quantize_weight_to_int4_symmetric(model.fc2.weight.transpose(0, 1).contiguous()),
    }


def run_optical_matmul(
    optical_model: Any,
    a_bmk: torch.Tensor,
    *,
    input_type: str,
    act_clip_max: float,
    weight_q: np.ndarray,
    weight_scale: float,
    timing_stats: dict[str, float],
) -> torch.Tensor:
    if a_bmk.ndim != 3:
        raise ValueError("a_bmk must be 3D with shape [B, M, K].")

    bsz, _, k_dim = a_bmk.shape
    if weight_q.shape[0] != k_dim:
        raise ValueError("K dimension mismatch between activation and weight.")

    a_q, a_scale, a_zero_point, hw_input_type = quantize_activation_for_optical(
        a_bmk,
        input_type=input_type,
        clip_max=act_clip_max,
    )

    if input_type == "int4" and "int4_emulation_notice_printed" not in timing_stats:
        print("[Info] input_type=int4 is emulated via uint4 offset encoding for simulator compatibility.")
        timing_stats["int4_emulation_notice_printed"] = 1.0

    input_tensors = a_q.astype(np.int32, copy=False)
    w_q = weight_q.astype(np.int32, copy=False)
    wght_tensors = np.broadcast_to(w_q[None, :, :], (bsz, w_q.shape[0], w_q.shape[1])).astype(np.int32, copy=True)

    call_start = time.perf_counter()
    result_model = optical_model(input_tensors, wght_tensors, inputType=hw_input_type)
    call_end = time.perf_counter()

    timing_stats["optical_call_time_s"] += call_end - call_start
    timing_stats["optical_call_count"] += 1.0

    if hasattr(result_model, "numpy"):
        out_concat = result_model.numpy()
    else:
        out_concat = np.asarray(result_model)

    if a_zero_point != 0:
        w_sum = np.sum(w_q.astype(np.float32), axis=0, keepdims=True)
        out_concat = out_concat.astype(np.float32) - (float(a_zero_point) * w_sum[None, :, :])

    out_fp32 = out_concat.astype(np.float32) * (a_scale * weight_scale)
    return torch.from_numpy(out_fp32)


def forward_with_optical_matmul(
    model: SingleHiddenMLP,
    images: torch.Tensor,
    optical_model: Any,
    *,
    input_type: str,
    act_clip_by_layer: dict[str, float],
    timing_stats: dict[str, float],
    optical_layers: set[str],
    weight_cache: dict[str, tuple[np.ndarray, float]],
) -> torch.Tensor:
    x = torch.flatten(images, start_dim=1)

    if "fc1" in optical_layers:
        fc1_q, fc1_scale = weight_cache["fc1"]
        x = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x.unsqueeze(1),
            input_type=input_type,
            act_clip_max=act_clip_by_layer["fc1"],
            weight_q=fc1_q,
            weight_scale=fc1_scale,
            timing_stats=timing_stats,
        ).squeeze(1)
        if model.fc1.bias is not None:
            x = x + model.fc1.bias.detach().cpu()
    else:
        x = model.fc1(x)

    x = F.relu(x, inplace=False)

    if "fc2" in optical_layers:
        fc2_q, fc2_scale = weight_cache["fc2"]
        logits = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x.unsqueeze(1),
            input_type=input_type,
            act_clip_max=act_clip_by_layer["fc2"],
            weight_q=fc2_q,
            weight_scale=fc2_scale,
            timing_stats=timing_stats,
        ).squeeze(1)
        if model.fc2.bias is not None:
            logits = logits + model.fc2.bias.detach().cpu()
    else:
        logits = model.fc2(x)

    return logits


@torch.no_grad()
def evaluate(
    model: SingleHiddenMLP,
    loader: DataLoader,
    optical_model: Any,
    *,
    input_type: str,
    act_clip_by_layer: dict[str, float],
    print_batch_timing: bool,
    optical_layers: set[str],
    max_samples: int,
    total_macs_per_sample: int,
    optical_macs_per_sample: int,
    weight_cache: dict[str, tuple[np.ndarray, float]],
) -> tuple[float, float, list[float], float, dict[str, float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    optical_macs_total = 0
    all_macs_total = 0
    timing_stats: dict[str, float] = {"optical_call_time_s": 0.0, "optical_call_count": 0.0}
    batch_forward_times: list[float] = []
    eval_start = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        if total_samples >= max_samples:
            break

        images = images.to("cpu", non_blocking=True)
        labels = labels.to("cpu", non_blocking=True)
        remaining = max_samples - total_samples
        if remaining < labels.size(0):
            images = images[:remaining]
            labels = labels[:remaining]

        forward_start = time.perf_counter()
        logits = forward_with_optical_matmul(
            model=model,
            images=images,
            optical_model=optical_model,
            input_type=input_type,
            act_clip_by_layer=act_clip_by_layer,
            timing_stats=timing_stats,
            optical_layers=optical_layers,
            weight_cache=weight_cache,
        )
        forward_end = time.perf_counter()
        batch_forward_time = forward_end - forward_start
        batch_forward_times.append(batch_forward_time)

        if print_batch_timing:
            print(f"[Timing] batch={batch_idx:03d} forward_time_ms={batch_forward_time * 1000.0:.3f}")

        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        matches = preds.eq(labels)
        total_correct += matches.sum().item()
        total_samples += batch_size

        for class_idx in range(10):
            class_mask = labels.eq(class_idx)
            class_count = class_mask.sum().item()
            if class_count == 0:
                continue
            class_total[class_idx] += class_count
            class_correct[class_idx] += preds[class_mask].eq(labels[class_mask]).sum().item()

        optical_macs_total += optical_macs_per_sample * batch_size
        all_macs_total += total_macs_per_sample * batch_size

    eval_end = time.perf_counter()
    avg_loss = total_loss / total_samples
    overall_acc = total_correct / total_samples
    per_class_acc = [0.0 if class_total[i] == 0 else class_correct[i] / class_total[i] for i in range(10)]
    optical_ratio = 0.0 if all_macs_total == 0 else optical_macs_total / all_macs_total
    timing_summary = {
        "total_eval_time_s": float(eval_end - eval_start),
        "total_forward_time_s": float(np.sum(batch_forward_times)) if batch_forward_times else 0.0,
        "avg_batch_forward_ms": float(np.mean(batch_forward_times) * 1000.0) if batch_forward_times else 0.0,
        "optical_call_time_s": float(timing_stats["optical_call_time_s"]),
        "optical_call_count": float(timing_stats["optical_call_count"]),
        "evaluated_samples": float(total_samples),
    }
    return avg_loss, overall_acc, per_class_acc, optical_ratio, timing_summary


def main() -> None:
    overall_start = time.perf_counter()
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    project_root = Path(__file__).resolve().parent.parent
    data_root = resolve_path(project_root, args.data_root)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    optical_layers = parse_optical_layers(args.optical_layers)
    act_clip_by_layer = build_layer_clip_map(args)

    print(f"Project root: {project_root}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Using device for torch-side ops: cpu")
    print("Optical backend: osimulator (Gazelle model)")
    print(f"Optical offload layers: {sorted(optical_layers)}")
    print(f"Layer-wise clip: {act_clip_by_layer}")

    loader = build_test_loader(data_root=data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    optical_model = load_optical_model()
    model, metadata = load_model_weights(checkpoint_path, args.hidden_size, args.bias)
    model.to("cpu")
    weight_cache = build_weight_cache(model)

    total_macs_per_sample, optical_macs_per_sample, macs_dict, layer_to_module_name = build_official_macs_profile(
        model=model,
        optical_layers=optical_layers,
    )
    expected_optical_ratio = 0.0 if total_macs_per_sample == 0 else optical_macs_per_sample / total_macs_per_sample

    print("Official MAC profile (from op_computation_analysis):")
    print(f"  total_macs_per_sample: {total_macs_per_sample:,}")
    print(f"  optical_macs_per_sample: {optical_macs_per_sample:,}")
    print(f"  expected_optical_compute_ratio: {expected_optical_ratio:.4%}")
    print("  per-module macs:")
    for logical_name in ("fc1", "fc2"):
        module_name = layer_to_module_name[logical_name]
        print(f"    {module_name}: {macs_dict[module_name]:,}")

    expected_batches = math.ceil(args.max_samples / args.batch_size)
    print(f"Max samples: {args.max_samples}")
    print(f"Estimated optical calls: ~{expected_batches * len(optical_layers)}")

    avg_loss, overall_acc, per_class_acc, optical_ratio, timing_summary = evaluate(
        model=model,
        loader=loader,
        optical_model=optical_model,
        input_type=args.input_type,
        act_clip_by_layer=act_clip_by_layer,
        print_batch_timing=args.print_batch_timing,
        optical_layers=optical_layers,
        max_samples=args.max_samples,
        total_macs_per_sample=total_macs_per_sample,
        optical_macs_per_sample=optical_macs_per_sample,
        weight_cache=weight_cache,
    )

    print("Checkpoint metadata:")
    print(f"  epoch: {metadata.get('epoch')}")
    print(f"  val_acc: {metadata.get('val_acc')}")
    print(f"  hidden_size: {metadata.get('hidden_size')}")
    print(f"  bias: {metadata.get('bias')}")

    print("\nEvaluation results on MNIST test set:")
    print(f"  avg_loss: {avg_loss:.6f}")
    print(f"  top1_accuracy: {overall_acc:.4%}")
    print(f"  optical_compute_ratio: {optical_ratio:.4%}")

    print("\nTiming summary:")
    print(f"  total_eval_time_s: {timing_summary['total_eval_time_s']:.6f}")
    print(f"  total_forward_time_s: {timing_summary['total_forward_time_s']:.6f}")
    print(f"  avg_batch_forward_ms: {timing_summary['avg_batch_forward_ms']:.3f}")
    print(f"  optical_call_time_s: {timing_summary['optical_call_time_s']:.6f}")
    print(f"  optical_call_count: {int(timing_summary['optical_call_count'])}")
    print(f"  evaluated_samples: {int(timing_summary['evaluated_samples'])}")

    print("\nPer-class accuracy:")
    for class_idx, acc in enumerate(per_class_acc):
        print(f"  class {class_idx}: {acc:.4%}")

    overall_end = time.perf_counter()
    print(f"\nTotal runtime (end-to-end): {overall_end - overall_start:.6f} s")


if __name__ == "__main__":
    main()
