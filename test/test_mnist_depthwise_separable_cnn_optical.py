from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
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

from mnist_depthwise_separable_cnn import MNISTDepthwiseSeparableCNN, parse_channels
from op_computation_analysis import calculate_linear_macs


@dataclass(frozen=True)
class QuantizedWeight:
    q: np.ndarray
    scale: float
    sum_by_output: np.ndarray


def parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MNIST depthwise separable CNN with optical conv matmul")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mnist_depthwise_separable_cnn/best_mnist_depthwise_separable_cnn.pt",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--input-type", type=str, default="int4", choices=["int4", "uint4"])
    parser.add_argument("--act-clip-max", type=float, default=6.0)
    parser.add_argument("--weight-bits", type=int, default=4, choices=[4])
    parser.add_argument("--act-clip-dw1", type=float, default=None)
    parser.add_argument("--act-clip-pw1", type=float, default=None)
    parser.add_argument("--act-clip-dw2", type=float, default=None)
    parser.add_argument("--act-clip-pw2", type=float, default=None)
    parser.add_argument("--act-clip-dw3", type=float, default=None)
    parser.add_argument("--act-clip-pw3", type=float, default=None)
    parser.add_argument("--optical-layers", type=str, default="pw3")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--print-batch-timing", type=parse_bool_flag, default=False)
    parser.add_argument("--channels", type=str, default="16,64,128")
    return parser.parse_args()


def parse_optical_layers(raw: str) -> set[str]:
    valid = {"dw1", "pw1", "dw2", "pw2", "dw3", "pw3"}
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}
    invalid = sorted(items - valid)
    if invalid:
        raise ValueError(f"Invalid optical layer names: {invalid}. Valid names are: {sorted(valid)}")
    return items


def build_layer_clip_map(args: argparse.Namespace) -> dict[str, float]:
    if args.act_clip_max <= 0:
        raise ValueError("--act-clip-max must be positive.")
    clip_map = {
        "dw1": args.act_clip_dw1 if args.act_clip_dw1 is not None else args.act_clip_max,
        "pw1": args.act_clip_pw1 if args.act_clip_pw1 is not None else args.act_clip_max,
        "dw2": args.act_clip_dw2 if args.act_clip_dw2 is not None else args.act_clip_max,
        "pw2": args.act_clip_pw2 if args.act_clip_pw2 is not None else args.act_clip_max,
        "dw3": args.act_clip_dw3 if args.act_clip_dw3 is not None else args.act_clip_max,
        "pw3": args.act_clip_pw3 if args.act_clip_pw3 is not None else args.act_clip_max,
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


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was set, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_channels_from_checkpoint(checkpoint: dict[str, Any], fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    raw_channels = checkpoint.get("channels")
    if raw_channels is None:
        args = checkpoint.get("args")
        if isinstance(args, dict) and "channels" in args:
            return parse_channels(str(args["channels"]))
        return fallback
    return tuple(int(value) for value in raw_channels)


def load_model_weights(
    checkpoint_path: Path,
    fallback_channels: tuple[int, int, int],
) -> tuple[MNISTDepthwiseSeparableCNN, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format. Expected dict.")

    channels = infer_channels_from_checkpoint(checkpoint, fallback=fallback_channels)
    model = MNISTDepthwiseSeparableCNN(channels=channels)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model, {"epoch": checkpoint.get("epoch"), "val_acc": checkpoint.get("val_acc"), "channels": channels}


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
    layer_to_module_name = {
        "dw1": "block1.depthwise",
        "pw1": "block1.pointwise",
        "dw2": "block2.depthwise",
        "pw2": "block2.pointwise",
        "dw3": "block3.depthwise",
        "pw3": "block3.pointwise",
    }
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


def quantize_weight_symmetric(w: torch.Tensor, weight_bits: int) -> QuantizedWeight:
    w_cpu = w.detach().to(torch.float32).cpu()
    if weight_bits == 4:
        q_min, q_max = -8, 7
    elif weight_bits == 8:
        q_min, q_max = -128, 127
    else:
        raise ValueError(f"Unsupported weight_bits: {weight_bits}")

    max_abs = torch.max(torch.abs(w_cpu)).item()
    scale = 1.0 if max_abs == 0.0 else max_abs / float(q_max)
    q = torch.round(w_cpu / scale).clamp(q_min, q_max).to(torch.int32).numpy()
    sum_axis = 0 if q.ndim == 2 else 1
    return QuantizedWeight(q=q, scale=float(scale), sum_by_output=np.sum(q.astype(np.float32), axis=sum_axis))


def build_weight_cache(model: MNISTDepthwiseSeparableCNN, weight_bits: int) -> dict[str, QuantizedWeight]:
    return {
        "dw1": quantize_weight_symmetric(model.block1.depthwise.weight.view(1, 9, 1), weight_bits=weight_bits),
        "pw1": quantize_weight_symmetric(model.block1.pointwise.weight.view(model.block1.pointwise.out_channels, -1).T, weight_bits=weight_bits),
        "dw2": quantize_weight_symmetric(model.block2.depthwise.weight.view(model.block2.depthwise.out_channels, 9, 1), weight_bits=weight_bits),
        "pw2": quantize_weight_symmetric(model.block2.pointwise.weight.view(model.block2.pointwise.out_channels, -1).T, weight_bits=weight_bits),
        "dw3": quantize_weight_symmetric(model.block3.depthwise.weight.view(model.block3.depthwise.out_channels, 9, 1), weight_bits=weight_bits),
        "pw3": quantize_weight_symmetric(model.block3.pointwise.weight.view(model.block3.pointwise.out_channels, -1).T, weight_bits=weight_bits),
    }


def run_optical_matmul(
    optical_model: Any,
    a_bmk: torch.Tensor,
    quantized_weight: QuantizedWeight,
    *,
    input_type: str,
    act_clip_max: float,
    timing_stats: dict[str, float],
) -> torch.Tensor:
    if a_bmk.ndim != 3:
        raise ValueError("a_bmk must be 3D with shape [B, M, K].")

    bsz, _, k_dim = a_bmk.shape
    w_q = quantized_weight.q
    if w_q.ndim == 2:
        if w_q.shape[0] != k_dim:
            raise ValueError("K dimension mismatch between activation and weight.")
        wght_tensors = np.broadcast_to(w_q[None, :, :], (bsz, w_q.shape[0], w_q.shape[1])).astype(np.int32, copy=True)
    elif w_q.ndim == 3:
        if w_q.shape[0] != bsz or w_q.shape[1] != k_dim:
            raise ValueError("B or K dimension mismatch between activation and batched weight.")
        wght_tensors = w_q.astype(np.int32, copy=False)
    else:
        raise ValueError("quantized weight must be 2D or 3D.")

    a_q, a_scale, a_zero_point, hw_input_type = quantize_activation_for_optical(
        a_bmk,
        input_type=input_type,
        clip_max=act_clip_max,
    )

    if input_type == "int4" and "int4_emulation_notice_printed" not in timing_stats:
        print("[Info] input_type=int4 is emulated via uint4 offset encoding for simulator compatibility.")
        timing_stats["int4_emulation_notice_printed"] = 1.0

    input_tensors = a_q.astype(np.int32, copy=False)
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
        w_sum = quantized_weight.sum_by_output.astype(np.float32)
        if w_sum.ndim == 1:
            out_concat = out_concat.astype(np.float32) - (float(a_zero_point) * w_sum[None, None, :])
        else:
            out_concat = out_concat.astype(np.float32) - (float(a_zero_point) * w_sum[:, None, :])

    out_fp16 = out_concat.astype(np.float16) * np.float16(a_scale * quantized_weight.scale)
    return torch.from_numpy(out_fp16.astype(np.float32))


def optical_pointwise_conv2d(
    x: torch.Tensor,
    conv: nn.Conv2d,
    optical_model: Any,
    quantized_weight: QuantizedWeight,
    *,
    input_type: str,
    act_clip_max: float,
    timing_stats: dict[str, float],
) -> torch.Tensor:
    bsz, _, height, width = x.shape
    a_bmk = x.permute(0, 2, 3, 1).contiguous().view(bsz, height * width, x.size(1))
    out_bmn = run_optical_matmul(
        optical_model=optical_model,
        a_bmk=a_bmk,
        quantized_weight=quantized_weight,
        input_type=input_type,
        act_clip_max=act_clip_max,
        timing_stats=timing_stats,
    )
    out = out_bmn.view(bsz, height, width, conv.out_channels).permute(0, 3, 1, 2).contiguous()
    if conv.bias is not None:
        out = out + conv.bias.detach().cpu().view(1, -1, 1, 1)
    return out


def optical_depthwise_conv2d(
    x: torch.Tensor,
    conv: nn.Conv2d,
    optical_model: Any,
    quantized_weight: QuantizedWeight,
    *,
    input_type: str,
    act_clip_max: float,
    timing_stats: dict[str, float],
) -> torch.Tensor:
    bsz, channels, height, width = x.shape
    x_unfold = F.unfold(x, kernel_size=3, padding=1, stride=1)
    a_bclk = x_unfold.view(bsz, channels, 9, height * width).permute(0, 1, 3, 2).contiguous()
    a_bmk = a_bclk.view(bsz * channels, height * width, 9)

    w_q = quantized_weight.q
    w_batched = np.broadcast_to(w_q[None, :, :, :], (bsz, channels, 9, 1)).reshape(bsz * channels, 9, 1).astype(np.int32, copy=True)
    w_sum = np.sum(w_batched.astype(np.float32), axis=1)
    batched_weight = QuantizedWeight(q=w_batched, scale=quantized_weight.scale, sum_by_output=w_sum)

    out_bmn = run_optical_matmul(
        optical_model=optical_model,
        a_bmk=a_bmk,
        quantized_weight=batched_weight,
        input_type=input_type,
        act_clip_max=act_clip_max,
        timing_stats=timing_stats,
    )
    out = out_bmn.view(bsz, channels, height * width).view(bsz, channels, height, width)
    if conv.bias is not None:
        out = out + conv.bias.detach().cpu().view(1, -1, 1, 1)
    return out


def forward_with_optical_matmul(
    model: MNISTDepthwiseSeparableCNN,
    images: torch.Tensor,
    optical_model: Any,
    *,
    input_type: str,
    act_clip_by_layer: dict[str, float],
    timing_stats: dict[str, float],
    optical_layers: set[str],
    weight_cache: dict[str, QuantizedWeight],
) -> torch.Tensor:
    x = images.to("cpu")

    if "dw1" in optical_layers:
        x = optical_depthwise_conv2d(x, model.block1.depthwise, optical_model, weight_cache["dw1"], input_type=input_type, act_clip_max=act_clip_by_layer["dw1"], timing_stats=timing_stats)
    else:
        x = model.block1.depthwise(x)
    if "pw1" in optical_layers:
        x = optical_pointwise_conv2d(x, model.block1.pointwise, optical_model, weight_cache["pw1"], input_type=input_type, act_clip_max=act_clip_by_layer["pw1"], timing_stats=timing_stats)
    else:
        x = model.block1.pointwise(x)
    x = F.relu(x, inplace=False)
    x = model.pool1(x)

    if "dw2" in optical_layers:
        x = optical_depthwise_conv2d(x, model.block2.depthwise, optical_model, weight_cache["dw2"], input_type=input_type, act_clip_max=act_clip_by_layer["dw2"], timing_stats=timing_stats)
    else:
        x = model.block2.depthwise(x)
    if "pw2" in optical_layers:
        x = optical_pointwise_conv2d(x, model.block2.pointwise, optical_model, weight_cache["pw2"], input_type=input_type, act_clip_max=act_clip_by_layer["pw2"], timing_stats=timing_stats)
    else:
        x = model.block2.pointwise(x)
    x = F.relu(x, inplace=False)
    x = model.pool2(x)

    if "dw3" in optical_layers:
        x = optical_depthwise_conv2d(x, model.block3.depthwise, optical_model, weight_cache["dw3"], input_type=input_type, act_clip_max=act_clip_by_layer["dw3"], timing_stats=timing_stats)
    else:
        x = model.block3.depthwise(x)
    if "pw3" in optical_layers:
        x = optical_pointwise_conv2d(x, model.block3.pointwise, optical_model, weight_cache["pw3"], input_type=input_type, act_clip_max=act_clip_by_layer["pw3"], timing_stats=timing_stats)
    else:
        x = model.block3.pointwise(x)
    x = F.relu(x, inplace=False)

    x = model.avg_pool(x)
    x = model.flatten(x)
    return model.classifier(x.float())


@torch.no_grad()
def evaluate(
    model: MNISTDepthwiseSeparableCNN,
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
    weight_cache: dict[str, QuantizedWeight],
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

        remaining = max_samples - total_samples
        if remaining < labels.size(0):
            images = images[:remaining]
            labels = labels[:remaining]

        labels = labels.to("cpu", non_blocking=True)
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
    return total_loss / total_samples, total_correct / total_samples, per_class_acc, optical_ratio, timing_summary


def main() -> None:
    overall_start = time.perf_counter()
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_root = resolve_path(project_root, args.data_root)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    optical_layers = parse_optical_layers(args.optical_layers)
    act_clip_by_layer = build_layer_clip_map(args)
    _device = select_device(args.device)

    model, metadata = load_model_weights(checkpoint_path, fallback_channels=parse_channels(args.channels))
    model = model.to("cpu")
    loader = build_test_loader(data_root=data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    optical_model = load_optical_model()
    weight_cache = build_weight_cache(model, weight_bits=args.weight_bits)

    total_macs_per_sample, optical_macs_per_sample, macs_dict, layer_to_module_name = build_official_macs_profile(
        model=model,
        optical_layers=optical_layers,
    )
    expected_optical_ratio = 0.0 if total_macs_per_sample == 0 else optical_macs_per_sample / total_macs_per_sample

    print(f"Project root: {project_root}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Using device for torch-side ops: cpu")
    print("Optical backend: osimulator (Gazelle model)")
    print(f"Input type: {args.input_type} (simulator-safe uint4 path for signed int4 emulation)")
    print(f"Weight quantization: int{args.weight_bits}")
    print(f"Channels: {metadata['channels']}")
    print(f"Optical offload layers: {sorted(optical_layers)}")
    print(f"Layer-wise clip: {act_clip_by_layer}")
    print("Official MAC profile (from op_computation_analysis):")
    print(f"  total_macs_per_sample: {total_macs_per_sample:,}")
    print(f"  optical_macs_per_sample: {optical_macs_per_sample:,}")
    print(f"  expected_optical_compute_ratio: {expected_optical_ratio:.4%}")
    if expected_optical_ratio < 0.90:
        print("[Warning] Expected optical MAC ratio is below 90%; pass --optical-layers pw1,pw2,pw3 after confirming quantized accuracy.")
    print("  per-module macs:")
    for logical_layer, module_name in layer_to_module_name.items():
        print(f"    {logical_layer} ({module_name}): {int(macs_dict[module_name]):,}")
    if "classifier" in macs_dict:
        print(f"    classifier: {int(macs_dict['classifier']):,} (CPU fp32)")

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
