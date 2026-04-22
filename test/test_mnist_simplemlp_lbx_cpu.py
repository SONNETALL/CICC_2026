"""Evaluate simple MLP (lbx) on CPU for MNIST test split.

该脚本用于 simplemlp 的纯 CPU 推理，不依赖光模拟器：
1) 从 models/mnist_mlp_lbx 加载 w1.npy / w2.npy
2) 在 MNIST 测试集上推理（默认先跑 1k）
3) 输出准确率、分类型准确率与计时统计
4) 最后打印端到端总耗时
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
    parser = argparse.ArgumentParser(description="Evaluate simplemlp lbx on CPU")
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
        "--max-samples",
        type=int,
        default=1_000,
        help=(
            "Maximum number of test samples to evaluate. "
            "Set to 10000 for full MNIST test split evaluation."
        ),
    )
    parser.add_argument(
        "--print-batch-timing",
        type=parse_bool_flag,
        default=False,
        help="Whether to print per-batch timing details.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, user_path: str) -> Path:
    """Resolve user path to an absolute path using project root fallback."""
    path = Path(user_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


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
    """从 npy 文件加载 simplemlp 权重。"""
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
        # npy 权重是右乘形式 [in, out]，Linear.weight 需要 [out, in]。
        model.fc1.weight.copy_(torch.from_numpy(np.ascontiguousarray(w1.T)))
        model.fc2.weight.copy_(torch.from_numpy(np.ascontiguousarray(w2.T)))


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
        pin_memory=False,
    )
    return loader


@torch.no_grad()
def evaluate_cpu(
    model: SimpleMLP,
    loader: DataLoader,
    *,
    max_samples: int,
    print_batch_timing: bool,
) -> tuple[float, float, list[float], dict[str, float]]:
    """执行纯 CPU 推理并返回准确率和计时统计。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    batch_forward_times: list[float] = []
    eval_start = time.perf_counter()
    evaluated_batches = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        if total_samples >= max_samples:
            break

        images = images.to("cpu", non_blocking=False)
        labels = labels.to("cpu", non_blocking=False)

        remaining = max_samples - total_samples
        if remaining < labels.size(0):
            images = images[:remaining]
            labels = labels[:remaining]

        evaluated_batches += 1

        forward_start = time.perf_counter()
        logits = model(images)
        forward_end = time.perf_counter()

        batch_forward_time = forward_end - forward_start
        batch_forward_times.append(batch_forward_time)

        if print_batch_timing:
            print(
                f"[Timing][CPU] batch={batch_idx:03d} "
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

    eval_end = time.perf_counter()

    avg_loss = total_loss / total_samples
    overall_acc = total_correct / total_samples

    per_class_acc = []
    for class_idx in range(10):
        if class_total[class_idx] == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(class_correct[class_idx] / class_total[class_idx])

    total_eval_time_s = eval_end - eval_start
    total_forward_time_s = float(np.sum(batch_forward_times)) if batch_forward_times else 0.0
    avg_batch_forward_ms = float(np.mean(batch_forward_times) * 1000.0) if batch_forward_times else 0.0
    p50_batch_forward_ms = float(np.percentile(batch_forward_times, 50) * 1000.0) if batch_forward_times else 0.0
    p95_batch_forward_ms = float(np.percentile(batch_forward_times, 95) * 1000.0) if batch_forward_times else 0.0

    samples_per_sec_eval = 0.0 if total_eval_time_s == 0 else (total_samples / total_eval_time_s)
    samples_per_sec_forward = 0.0 if total_forward_time_s == 0 else (total_samples / total_forward_time_s)

    timing_summary = {
        "total_eval_time_s": float(total_eval_time_s),
        "total_forward_time_s": float(total_forward_time_s),
        "avg_batch_forward_ms": float(avg_batch_forward_ms),
        "p50_batch_forward_ms": float(p50_batch_forward_ms),
        "p95_batch_forward_ms": float(p95_batch_forward_ms),
        "samples_per_sec_eval": float(samples_per_sec_eval),
        "samples_per_sec_forward": float(samples_per_sec_forward),
        "evaluated_samples": float(total_samples),
        "evaluated_batches": float(evaluated_batches),
    }

    return avg_loss, overall_acc, per_class_acc, timing_summary


def main() -> None:
    """程序入口。"""
    overall_start = time.perf_counter()

    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    weights_dir = resolve_path(project_root, args.weights_dir)
    data_root = resolve_path(project_root, args.data_root)

    print(f"Project root: {project_root}")
    print(f"Weights dir: {weights_dir}")
    print(f"Data root: {data_root}")
    print("Inference backend: pure CPU (PyTorch)")

    loader = build_test_loader(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SimpleMLP().to("cpu")
    load_simplemlp_weights(model=model, weights_dir=weights_dir, w1_file=args.w1_file, w2_file=args.w2_file)

    avg_loss, overall_acc, per_class_acc, timing_summary = evaluate_cpu(
        model=model,
        loader=loader,
        max_samples=args.max_samples,
        print_batch_timing=args.print_batch_timing,
    )

    print("\nEvaluation results on MNIST test set:")
    print(f"  avg_loss: {avg_loss:.6f}")
    print(f"  top1_accuracy: {overall_acc:.4%}")

    print("\nTiming summary:")
    print(f"  total_eval_time_s: {timing_summary['total_eval_time_s']:.6f}")
    print(f"  total_forward_time_s: {timing_summary['total_forward_time_s']:.6f}")
    print(f"  avg_batch_forward_ms: {timing_summary['avg_batch_forward_ms']:.3f}")
    print(f"  p50_batch_forward_ms: {timing_summary['p50_batch_forward_ms']:.3f}")
    print(f"  p95_batch_forward_ms: {timing_summary['p95_batch_forward_ms']:.3f}")
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
