"""Evaluate simple MLP (lbx) with optical matmul on MNIST test split.

该脚本参照 test_mnist_bnn_binary_state.py 的流程实现，
针对 models/mnist_mlp_lbx 下的 simplemlp 权重做推理评测，包含：
1) MNIST 测试集推理（支持 10k，全量前先跑 1k）
2) 使用官方 op_computation_analysis 计算光计算占比
3) 输出准确率、光计算占比、计时统计与总耗时

默认行为：
- 读取权重: models/mnist_mlp_lbx/w1.npy, w2.npy
- 默认评测样本: 1000（可改为 10000）
"""

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


# 让测试脚本可以复用 src 下官方 MAC 统计实现。
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from op_computation_analysis import calculate_linear_macs


def parse_bool_flag(value: str) -> bool:
    """将命令行字符串解析为布尔值。"""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate simplemlp lbx on MNIST test set")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="models/mnist_mlp_lbx",
        help="Directory containing w1.npy and w2.npy.",
    )
    parser.add_argument(
        "--w1-file",
        type=str,
        default="w1.npy",
        help="Filename of first layer weight matrix under weights-dir.",
    )
    parser.add_argument(
        "--w2-file",
        type=str,
        default="w2.npy",
        help="Filename of second layer weight matrix under weights-dir.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="MNIST dataset root directory (relative to project root or absolute path).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size used for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (0 is most compatible on Windows).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Execution device for torch-side ops. Optical backend itself is CPU/Numpy based. "
            "auto selects cuda when available, otherwise cpu."
        ),
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default="int4",
        choices=["int4", "uint4"],
        help="Input type passed to osimulator model interface.",
    )
    parser.add_argument(
        "--act-clip-max",
        type=float,
        default=4.0,
        help="Activation clip bound before quantization for optical matmul.",
    )
    parser.add_argument(
        "--print-batch-timing",
        type=parse_bool_flag,
        default=False,
        help="Whether to print per-batch timing details.",
    )
    parser.add_argument(
        "--optical-layers",
        type=str,
        default="fc1,fc2",
        help=(
            "Comma-separated layers offloaded to optical simulator. "
            "Valid names: fc1,fc2. Example: --optical-layers fc1"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1_000,
        help=(
            "Maximum number of test samples to evaluate. "
            "Set to 10000 for full MNIST test split evaluation."
        ),
    )
    return parser.parse_args()


def parse_optical_layers(raw: str) -> set[str]:
    """解析并校验光学卸载层列表。"""
    valid = {"fc1", "fc2"}
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}

    invalid = sorted(items - valid)
    if invalid:
        raise ValueError(
            "Invalid optical layer names: "
            f"{invalid}. Valid names are: {sorted(valid)}"
        )
    return items


def resolve_path(project_root: Path, user_path: str) -> Path:
    """Resolve user path to an absolute path using project root fallback."""
    path = Path(user_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def select_device(device_arg: str) -> torch.device:
    """Resolve runtime device from user argument."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was set, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST: 784 -> 64 -> 10 (no bias)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 64, bias=False)
        self.fc2 = nn.Linear(64, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x, inplace=False)
        x = self.fc2(x)
        return x


def load_simplemlp_weights(model: SimpleMLP, weights_dir: Path, w1_file: str, w2_file: str) -> None:
    """从 npy 文件加载 simplemlp 权重。

    约定：
    - w1 形状为 [784, 64]
    - w2 形状为 [64, 10]
    这与 x @ w 的右乘形式一致；赋值给 nn.Linear 时需要转置。
    """
    w1_path = (weights_dir / w1_file).resolve()
    w2_path = (weights_dir / w2_file).resolve()

    if not w1_path.exists() or not w2_path.exists():
        raise FileNotFoundError(
            "Weight file not found. "
            f"w1: {w1_path.exists()} ({w1_path}), w2: {w2_path.exists()} ({w2_path})"
        )

    w1 = np.asarray(np.load(w1_path), dtype=np.float32)
    w2 = np.asarray(np.load(w2_path), dtype=np.float32)

    if w1.shape != (784, 64):
        raise ValueError(f"Unexpected w1 shape: {w1.shape}, expected (784, 64)")
    if w2.shape != (64, 10):
        raise ValueError(f"Unexpected w2 shape: {w2.shape}, expected (64, 10)")

    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(np.ascontiguousarray(w1.T)))
        model.fc2.weight.copy_(torch.from_numpy(np.ascontiguousarray(w2.T)))


def build_official_macs_profile(
    model: nn.Module,
    optical_layers: set[str],
) -> tuple[int, int, dict[str, int], dict[str, str]]:
    """使用官方 MAC 统计代码构建本模型的计算量画像。"""
    dummy_input = torch.randn(1, 1, 28, 28)
    total_macs_per_sample, macs_dict = calculate_linear_macs(model, dummy_input)

    layer_to_module_name = {
        "fc1": "fc1",
        "fc2": "fc2",
    }

    missing_modules = [
        module_name
        for module_name in layer_to_module_name.values()
        if module_name not in macs_dict
    ]
    if missing_modules:
        raise RuntimeError(
            "Official MACs calculation did not find expected modules: "
            f"{missing_modules}. Found modules: {sorted(macs_dict.keys())}"
        )

    optical_macs_per_sample = 0
    for logical_layer in optical_layers:
        module_name = layer_to_module_name[logical_layer]
        optical_macs_per_sample += int(macs_dict[module_name])

    return int(total_macs_per_sample), int(optical_macs_per_sample), macs_dict, layer_to_module_name


def load_optical_model() -> Any:
    """加载 osimulator 的 Gazelle 光学模型。"""
    try:
        from osimulator.api import load_gazelle_model
    except Exception as exc:
        raise ImportError(
            "Failed to import osimulator.api.load_gazelle_model. "
            "Please ensure osimulator is installed and available in this environment."
        ) from exc

    return load_gazelle_model()


def quantize_activation_for_optical(
    x: torch.Tensor,
    *,
    input_type: str,
    clip_max: float,
) -> tuple[np.ndarray, float, int, str]:
    """将激活量化为光模拟器输入整数表示。"""
    if clip_max <= 0:
        raise ValueError("act-clip-max must be positive.")

    x_cpu = x.detach().to(torch.float32).cpu()

    if input_type == "uint4":
        x_clip = torch.clamp(x_cpu, min=0.0, max=clip_max)
        scale = clip_max / 15.0
        q = torch.round(x_clip / scale).clamp(0, 15).to(torch.int32)
        return q.numpy(), float(scale), 0, "uint4"

    if input_type == "int4":
        # 某些 osimulator 版本在 inputType=int4 时可能触发负索引越界。
        # 这里采用“偏移编码”规避，推理输出端会做对应还原。
        x_clip = torch.clamp(x_cpu, min=-clip_max, max=clip_max)
        scale = clip_max / 7.0
        q_signed = torch.round(x_clip / scale).clamp(-8, 7).to(torch.int32)
        zero_point = 8
        q_uint = q_signed + zero_point
        return q_uint.numpy(), float(scale), zero_point, "uint4"

    raise ValueError(f"Unsupported input_type: {input_type}")


def quantize_weight_to_int4_symmetric(w: torch.Tensor) -> tuple[np.ndarray, float]:
    """将权重量化到对称 int4（[-8, 7]）。"""
    w_cpu = w.detach().to(torch.float32).cpu()
    max_abs = torch.max(torch.abs(w_cpu)).item()
    if max_abs == 0.0:
        scale = 1.0
    else:
        scale = max_abs / 7.0

    q = torch.round(w_cpu / scale).clamp(-8, 7).to(torch.int32)
    return q.numpy(), float(scale)


def run_optical_matmul(
    optical_model: Any,
    a_bmk: torch.Tensor,
    w_kn: torch.Tensor,
    *,
    input_type: str,
    act_clip_max: float,
    timing_stats: dict[str, float],
) -> torch.Tensor:
    """通过 osimulator 执行矩阵乘法并累计光学调用计时。"""
    if a_bmk.ndim != 3:
        raise ValueError("a_bmk must be 3D with shape [B, M, K].")
    if w_kn.ndim != 2:
        raise ValueError("w_kn must be 2D with shape [K, N].")

    bsz, _, k_dim = a_bmk.shape
    if w_kn.shape[0] != k_dim:
        raise ValueError("K dimension mismatch between activation and weight.")

    a_q, a_scale, a_zero_point, hw_input_type = quantize_activation_for_optical(
        a_bmk,
        input_type=input_type,
        clip_max=act_clip_max,
    )
    w_q, w_scale = quantize_weight_to_int4_symmetric(w_kn)

    if input_type == "int4" and "int4_emulation_notice_printed" not in timing_stats:
        print("[Info] input_type=int4 is emulated via uint4 offset encoding for simulator compatibility.")
        timing_stats["int4_emulation_notice_printed"] = 1.0

    input_tensors = a_q.astype(np.int32, copy=False)
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

    out_fp32 = out_concat.astype(np.float32) * (a_scale * w_scale)
    return torch.from_numpy(out_fp32)


def forward_with_optical_matmul(
    model: SimpleMLP,
    images: torch.Tensor,
    optical_model: Any,
    *,
    input_type: str,
    act_clip_max: float,
    timing_stats: dict[str, float],
    optical_layers: set[str],
) -> torch.Tensor:
    """将 simplemlp 中选定线性层替换为光学 matmul 前向。"""
    x = torch.flatten(images, start_dim=1)

    if "fc1" in optical_layers:
        fc1_w_kn = model.fc1.weight.transpose(0, 1).contiguous()
        x = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x.unsqueeze(1),
            w_kn=fc1_w_kn,
            input_type=input_type,
            act_clip_max=act_clip_max,
            timing_stats=timing_stats,
        ).squeeze(1)
    else:
        x = model.fc1(x)

    x = F.relu(x, inplace=False)

    if "fc2" in optical_layers:
        fc2_w_kn = model.fc2.weight.transpose(0, 1).contiguous()
        logits = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x.unsqueeze(1),
            w_kn=fc2_w_kn,
            input_type=input_type,
            act_clip_max=act_clip_max,
            timing_stats=timing_stats,
        ).squeeze(1)
    else:
        logits = model.fc2(x)

    return logits


def build_test_loader(data_root: Path, batch_size: int, num_workers: int) -> DataLoader:
    """构建 MNIST 10k 测试集 DataLoader。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        transform=transform,
        download=False,
    )

    if len(dataset) != 10_000:
        raise RuntimeError(
            f"Expected 10,000 MNIST test samples, but found {len(dataset)}. "
            "Please verify files under data/MNIST/raw."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


@torch.no_grad()
def evaluate(
    model: SimpleMLP,
    loader: DataLoader,
    optical_model: Any,
    *,
    input_type: str,
    act_clip_max: float,
    print_batch_timing: bool,
    optical_layers: set[str],
    max_samples: int,
    total_macs_per_sample: int,
    optical_macs_per_sample: int,
) -> tuple[float, float, list[float], float, dict[str, float]]:
    """执行评测并返回准确率、光学占比与计时统计。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    optical_macs_total = 0
    all_macs_total = 0

    timing_stats: dict[str, float] = {
        "optical_call_time_s": 0.0,
        "optical_call_count": 0.0,
    }
    batch_forward_times: list[float] = []

    eval_start = time.perf_counter()
    evaluated_batches = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        if total_samples >= max_samples:
            break

        images = images.to("cpu", non_blocking=True)
        labels = labels.to("cpu", non_blocking=True)

        remaining = max_samples - total_samples
        if remaining < labels.size(0):
            images = images[:remaining]
            labels = labels[:remaining]

        evaluated_batches += 1

        forward_start = time.perf_counter()
        logits = forward_with_optical_matmul(
            model=model,
            images=images,
            optical_model=optical_model,
            input_type=input_type,
            act_clip_max=act_clip_max,
            timing_stats=timing_stats,
            optical_layers=optical_layers,
        )
        forward_end = time.perf_counter()

        batch_forward_time = forward_end - forward_start
        batch_forward_times.append(batch_forward_time)

        if print_batch_timing:
            print(
                f"[Timing] batch={batch_idx:03d} "
                f"forward_time_ms={batch_forward_time * 1000.0:.3f}"
            )

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

    per_class_acc = []
    for class_idx in range(10):
        if class_total[class_idx] == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(class_correct[class_idx] / class_total[class_idx])

    optical_ratio = 0.0 if all_macs_total == 0 else (optical_macs_total / all_macs_total)

    total_eval_time_s = eval_end - eval_start
    total_forward_time_s = float(np.sum(batch_forward_times)) if batch_forward_times else 0.0
    avg_batch_forward_ms = float(np.mean(batch_forward_times) * 1000.0) if batch_forward_times else 0.0
    p50_batch_forward_ms = float(np.percentile(batch_forward_times, 50) * 1000.0) if batch_forward_times else 0.0
    p95_batch_forward_ms = float(np.percentile(batch_forward_times, 95) * 1000.0) if batch_forward_times else 0.0

    optical_call_time_s = timing_stats["optical_call_time_s"]
    optical_call_count = timing_stats["optical_call_count"]
    avg_optical_call_ms = 0.0 if optical_call_count == 0 else (optical_call_time_s / optical_call_count) * 1000.0

    samples_per_sec_eval = 0.0 if total_eval_time_s == 0 else (total_samples / total_eval_time_s)
    samples_per_sec_forward = 0.0 if total_forward_time_s == 0 else (total_samples / total_forward_time_s)
    optical_time_ratio = 0.0 if total_forward_time_s == 0 else (optical_call_time_s / total_forward_time_s)

    timing_summary = {
        "total_eval_time_s": float(total_eval_time_s),
        "total_forward_time_s": float(total_forward_time_s),
        "avg_batch_forward_ms": float(avg_batch_forward_ms),
        "p50_batch_forward_ms": float(p50_batch_forward_ms),
        "p95_batch_forward_ms": float(p95_batch_forward_ms),
        "optical_call_time_s": float(optical_call_time_s),
        "optical_call_count": float(optical_call_count),
        "avg_optical_call_ms": float(avg_optical_call_ms),
        "samples_per_sec_eval": float(samples_per_sec_eval),
        "samples_per_sec_forward": float(samples_per_sec_forward),
        "optical_time_ratio": float(optical_time_ratio),
        "evaluated_samples": float(total_samples),
        "evaluated_batches": float(evaluated_batches),
    }

    return avg_loss, overall_acc, per_class_acc, optical_ratio, timing_summary


def main() -> None:
    """程序入口。"""
    overall_start = time.perf_counter()

    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    weights_dir = resolve_path(project_root, args.weights_dir)
    data_root = resolve_path(project_root, args.data_root)
    optical_layers = parse_optical_layers(args.optical_layers)

    device = select_device(args.device)

    print(f"Project root: {project_root}")
    print(f"Weights dir: {weights_dir}")
    print(f"Data root: {data_root}")
    print(f"Using device for torch-side ops: {device}")
    print("Optical backend: osimulator (Gazelle model)")
    print(f"Optical offload layers: {sorted(optical_layers)}")

    loader = build_test_loader(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optical_model = load_optical_model()
    model = SimpleMLP().to("cpu")
    load_simplemlp_weights(model=model, weights_dir=weights_dir, w1_file=args.w1_file, w2_file=args.w2_file)

    total_macs_per_sample, optical_macs_per_sample, macs_dict, layer_to_module_name = build_official_macs_profile(
        model=model,
        optical_layers=optical_layers,
    )
    expected_optical_ratio = 0.0 if total_macs_per_sample == 0 else (optical_macs_per_sample / total_macs_per_sample)

    print("Official MAC profile (from op_computation_analysis):")
    print(f"  total_macs_per_sample: {total_macs_per_sample:,}")
    print(f"  optical_macs_per_sample: {optical_macs_per_sample:,}")
    print(f"  expected_optical_compute_ratio: {expected_optical_ratio:.4%}")
    print("  per-module macs:")
    print(f"    {layer_to_module_name['fc1']}: {macs_dict[layer_to_module_name['fc1']]:,}")
    print(f"    {layer_to_module_name['fc2']}: {macs_dict[layer_to_module_name['fc2']]:,}")

    expected_batches = math.ceil(args.max_samples / args.batch_size)
    expected_calls = expected_batches * len(optical_layers)
    print(f"Max samples: {args.max_samples}")
    print(f"Estimated optical calls: ~{expected_calls}")

    avg_loss, overall_acc, per_class_acc, optical_ratio, timing_summary = evaluate(
        model=model,
        loader=loader,
        optical_model=optical_model,
        input_type=args.input_type,
        act_clip_max=args.act_clip_max,
        print_batch_timing=args.print_batch_timing,
        optical_layers=optical_layers,
        max_samples=args.max_samples,
        total_macs_per_sample=total_macs_per_sample,
        optical_macs_per_sample=optical_macs_per_sample,
    )

    print("\nEvaluation results on MNIST test set:")
    print(f"  avg_loss: {avg_loss:.6f}")
    print(f"  top1_accuracy: {overall_acc:.4%}")
    print(f"  optical_compute_ratio: {optical_ratio:.4%}")

    print("\nTiming summary:")
    print(f"  total_eval_time_s: {timing_summary['total_eval_time_s']:.6f}")
    print(f"  total_forward_time_s: {timing_summary['total_forward_time_s']:.6f}")
    print(f"  avg_batch_forward_ms: {timing_summary['avg_batch_forward_ms']:.3f}")
    print(f"  p50_batch_forward_ms: {timing_summary['p50_batch_forward_ms']:.3f}")
    print(f"  p95_batch_forward_ms: {timing_summary['p95_batch_forward_ms']:.3f}")
    print(f"  optical_call_time_s: {timing_summary['optical_call_time_s']:.6f}")
    print(f"  optical_call_count: {int(timing_summary['optical_call_count'])}")
    print(f"  avg_optical_call_ms: {timing_summary['avg_optical_call_ms']:.3f}")
    print(f"  optical_time_ratio: {timing_summary['optical_time_ratio']:.4%}")
    print(f"  samples_per_sec_eval: {timing_summary['samples_per_sec_eval']:.3f}")
    print(f"  samples_per_sec_forward: {timing_summary['samples_per_sec_forward']:.3f}")
    print(f"  evaluated_samples: {int(timing_summary['evaluated_samples'])}")
    print(f"  evaluated_batches: {int(timing_summary['evaluated_batches'])}")

    print("\nPer-class accuracy:")
    for class_idx, acc in enumerate(per_class_acc):
        print(f"  class {class_idx}: {acc:.4%}")

    overall_end = time.perf_counter()
    print(f"\nTotal runtime (end-to-end): {overall_end - overall_start:.6f} s")


if __name__ == "__main__":
    main()
