from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mnist_depthwise_separable_cnn import MNISTDepthwiseSeparableCNN, parse_channels
from op_computation_analysis import calculate_linear_macs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MNIST depthwise separable CNN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mnist_depthwise_separable_cnn/best_mnist_depthwise_separable_cnn.pt",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--channels", type=str, default="16,64,128")
    return parser.parse_args()


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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[float, float, list[float], dict[str, float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    batch_forward_times: list[float] = []
    eval_start = time.perf_counter()

    for images, labels in loader:
        if total_samples >= max_samples:
            break

        remaining = max_samples - total_samples
        if remaining < labels.size(0):
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        forward_start = time.perf_counter()
        logits = model(images)
        forward_end = time.perf_counter()
        batch_forward_times.append(forward_end - forward_start)

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
    per_class_acc = [
        0.0 if class_total[class_idx] == 0 else class_correct[class_idx] / class_total[class_idx]
        for class_idx in range(10)
    ]
    timing = {
        "total_eval_time_s": eval_end - eval_start,
        "total_forward_time_s": float(np.sum(batch_forward_times)) if batch_forward_times else 0.0,
        "avg_batch_forward_ms": float(np.mean(batch_forward_times) * 1000.0) if batch_forward_times else 0.0,
        "evaluated_samples": float(total_samples),
    }
    return total_loss / total_samples, total_correct / total_samples, per_class_acc, timing


def main() -> None:
    overall_start = time.perf_counter()
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_root = resolve_path(project_root, args.data_root)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    fallback_channels = parse_channels(args.channels)
    device = select_device(args.device)

    model, metadata = load_model_weights(checkpoint_path, fallback_channels=fallback_channels)
    model = model.to(device)
    loader = build_test_loader(data_root=data_root, batch_size=args.batch_size, num_workers=args.num_workers)

    total_macs_per_sample, macs_dict = calculate_linear_macs(model, torch.randn(1, 1, 28, 28))

    print(f"Project root: {project_root}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Channels: {metadata['channels']}")
    print(f"Parameters: {count_parameters(model):,}")
    print("Official MAC profile (from op_computation_analysis):")
    print(f"  total_macs_per_sample: {int(total_macs_per_sample):,}")
    for name, macs in macs_dict.items():
        print(f"  {name}: {int(macs):,}")

    avg_loss, overall_acc, per_class_acc, timing = evaluate(
        model=model,
        loader=loader,
        device=device,
        max_samples=args.max_samples,
    )

    print("Checkpoint metadata:")
    print(f"  epoch: {metadata.get('epoch')}")
    print(f"  val_acc: {metadata.get('val_acc')}")

    print("\nEvaluation results on MNIST test set:")
    print(f"  avg_loss: {avg_loss:.6f}")
    print(f"  top1_accuracy: {overall_acc:.4%}")

    print("\nTiming summary:")
    print(f"  total_eval_time_s: {timing['total_eval_time_s']:.6f}")
    print(f"  total_forward_time_s: {timing['total_forward_time_s']:.6f}")
    print(f"  avg_batch_forward_ms: {timing['avg_batch_forward_ms']:.3f}")
    print(f"  evaluated_samples: {int(timing['evaluated_samples'])}")

    print("\nPer-class accuracy:")
    for class_idx, acc in enumerate(per_class_acc):
        print(f"  class {class_idx}: {acc:.4%}")

    overall_end = time.perf_counter()
    print(f"\nTotal runtime (end-to-end): {overall_end - overall_start:.6f} s")


if __name__ == "__main__":
    main()
