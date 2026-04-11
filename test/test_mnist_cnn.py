"""Evaluate MNIST CNN with optical matmul acceleration on the 10k test split.

This script keeps the same preprocessing and model architecture as training,
but routes matrix multiplication style computations to the osimulator optical
backend where possible:
1) conv2 is executed as im2col + optical matrix multiplication
2) fc1 and fc2 are executed as optical matrix multiplication

The script prints both:
- top1 accuracy on the full 10k test set
- optical compute ratio (offloaded MACs / total MACs)

Usage examples:
    python test/test_mnist_cnn.py
    python test/test_mnist_cnn.py --batch-size 128
    python test/test_mnist_cnn.py --checkpoint models/mnist_cnn/best_mnist_cnn_int4_dequantized_fp32.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    """Define command-line arguments for flexible evaluation runs.

    Keeping arguments explicit allows this file to be reused in CI, local
    testing, and quick sanity checks without editing the source code.
    """
    parser = argparse.ArgumentParser(description="Evaluate MNIST CNN on 10k test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mnist_cnn/best_mnist_cnn_int4_dequantized_fp32.pt",
        help="Path to checkpoint file (relative to project root or absolute path).",
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
        default=256,
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
        default="uint4",
        choices=["uint4"],
        help="Input type passed to osimulator model interface.",
    )
    parser.add_argument(
        "--act-clip-max",
        type=float,
        default=6.0,
        help="Activation clamp upper bound before uint4 quantization for optical matmul.",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    """Resolve runtime device from the user's argument."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was set, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleMNISTCNN(nn.Module):
    """CNN architecture that matches the training script exactly.

    The shape progression is:
    1x28x28 -> 32x14x14 -> 64x7x7 -> flatten -> 128 -> 10 logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_optical_model() -> Any:
    """Load osimulator optical model using the same API style as LTexample.

    LTexample usage pattern:
    - from osimulator.api import load_gazelle_model
    - model = load_gazelle_model()
    - output = model(input_tensors, wght_tensors, inputType="uint4")
    """
    try:
        from osimulator.api import load_gazelle_model
    except Exception as exc:
        raise ImportError(
            "Failed to import osimulator.api.load_gazelle_model. "
            "Please ensure osimulator is installed and available in this environment."
        ) from exc

    return load_gazelle_model()


def quantize_activation_to_uint4(x: torch.Tensor, clip_max: float) -> tuple[np.ndarray, float]:
    """Quantize non-negative activation to uint4 for optical backend.

    We clamp activations into [0, clip_max], then map to integer range [0, 15].
    """
    if clip_max <= 0:
        raise ValueError("act-clip-max must be positive.")

    x_cpu = x.detach().to(torch.float32).cpu()
    x_clip = torch.clamp(x_cpu, min=0.0, max=clip_max)

    scale = clip_max / 15.0
    q = torch.round(x_clip / scale).clamp(0, 15).to(torch.int32)
    return q.numpy(), float(scale)


def quantize_weight_to_int4_symmetric(w: torch.Tensor) -> tuple[np.ndarray, float]:
    """Symmetric int4 quantization for weights in range [-8, 7]."""
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
    input_type: str,
    act_clip_max: float,
) -> torch.Tensor:
    """通过 osimulator 执行矩阵乘法，调用格式对齐 LTexample。

    Expected shapes:
    - a_bmk: [B, M, K]
    - w_kn:  [K, N]

    The simulator API expects:
    - input_tensors: [B, M, K] int32
    - wght_tensors:  [B, K, N] int32
    """
    if a_bmk.ndim != 3:
        raise ValueError("a_bmk must be 3D with shape [B, M, K].")
    if w_kn.ndim != 2:
        raise ValueError("w_kn must be 2D with shape [K, N].")

    bsz, m_dim, k_dim = a_bmk.shape
    if w_kn.shape[0] != k_dim:
        raise ValueError("K dimension mismatch between activation and weight.")

    # 先将激活和权重量化到光模拟器期望的整数范围。
    a_q, a_scale = quantize_activation_to_uint4(a_bmk, clip_max=act_clip_max)
    w_q, w_scale = quantize_weight_to_int4_symmetric(w_kn)

    # 按 LTexample 的思路逐样本调用：
    # input_tensors: [1, M, K]
    # wght_tensors:  [1, K, N]
    # 这样与示例的 b=1 调用方式保持一致，减少接口差异。
    out_list = []
    for idx in range(bsz):
        input_tensors = a_q[idx : idx + 1].astype(np.int32, copy=False)
        wght_tensors = w_q[None, :, :].astype(np.int32, copy=False)

        # 按示例保留 numpy 参考矩阵乘法路径，便于对齐接口语义与调试。
        _exp = np.matmul(input_tensors.astype(np.float32), wght_tensors.astype(np.float32))

        result_model = optical_model(input_tensors, wght_tensors, inputType=input_type)
        if hasattr(result_model, "numpy"):
            out_np = result_model.numpy()
        else:
            out_np = np.asarray(result_model)
        out_list.append(out_np)

    out_concat = np.concatenate(out_list, axis=0)

    # 将光计算输出反量化回浮点域。
    out_fp32 = out_concat.astype(np.float32) * (a_scale * w_scale)
    return torch.from_numpy(out_fp32)


def forward_with_optical_matmul(
    model: nn.Module,
    images: torch.Tensor,
    optical_model: Any,
    input_type: str,
    act_clip_max: float,
) -> torch.Tensor:
    """Forward pass where matrix multiplication parts are replaced by optical matmul.

    Layer mapping:
    - conv1: torch conv2d (input is normalized and may contain negative values)
    - conv2: im2col + optical matmul
    - fc1/fc2: optical matmul
    """
    # conv1 remains on torch because pre-normalized input includes negative values,
    # while current optical input type here is uint4.
    conv1_w = model.features[0].weight
    conv1_b = model.features[0].bias
    x = F.conv2d(images, conv1_w, conv1_b, stride=1, padding=1)
    x = F.relu(x, inplace=False)
    x = F.max_pool2d(x, kernel_size=2)

    # conv2 rewritten as im2col + batched matmul.
    # x_unfold: [B, Cin*Kh*Kw, L] where L=Hout*Wout.
    x_unfold = F.unfold(x, kernel_size=3, padding=1, stride=1)
    x_bmk = x_unfold.transpose(1, 2).contiguous()

    conv2_w = model.features[3].weight
    conv2_b = model.features[3].bias
    conv2_w_kn = conv2_w.view(conv2_w.size(0), -1).transpose(0, 1).contiguous()

    conv2_out_bmn = run_optical_matmul(
        optical_model=optical_model,
        a_bmk=x_bmk,
        w_kn=conv2_w_kn,
        input_type=input_type,
        act_clip_max=act_clip_max,
    )

    bsz = images.size(0)
    conv2_out = conv2_out_bmn.transpose(1, 2).contiguous().view(bsz, 64, 14, 14)
    conv2_out = conv2_out + conv2_b.detach().cpu().view(1, -1, 1, 1)
    x = F.relu(conv2_out, inplace=False)
    x = F.max_pool2d(x, kernel_size=2)

    # fc1 as optical matmul.
    x = torch.flatten(x, start_dim=1)
    fc1_w_kn = model.classifier[1].weight.transpose(0, 1).contiguous()
    fc1_b = model.classifier[1].bias

    fc1_out = run_optical_matmul(
        optical_model=optical_model,
        a_bmk=x.unsqueeze(1),
        w_kn=fc1_w_kn,
        input_type=input_type,
        act_clip_max=act_clip_max,
    ).squeeze(1)
    fc1_out = fc1_out + fc1_b.detach().cpu()
    x = F.relu(fc1_out, inplace=False)

    # Dropout is disabled in eval mode, so we can skip explicit dropout call.
    fc2_w_kn = model.classifier[4].weight.transpose(0, 1).contiguous()
    fc2_b = model.classifier[4].bias
    logits = run_optical_matmul(
        optical_model=optical_model,
        a_bmk=x.unsqueeze(1),
        w_kn=fc2_w_kn,
        input_type=input_type,
        act_clip_max=act_clip_max,
    ).squeeze(1)
    logits = logits + fc2_b.detach().cpu()
    return logits


def build_test_loader(data_root: Path, batch_size: int, num_workers: int) -> DataLoader:
    """Build DataLoader for MNIST test split (exactly 10,000 samples).

    Important notes:
    - train=False selects the official 10k test set.
    - download=False enforces usage of local files under /data.
    - transform must match training normalization for fair evaluation.
    """
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

    # Guardrail: user explicitly asked for the 10k test set.
    if len(dataset) != 10_000:
        raise RuntimeError(
            f"Expected 10,000 MNIST test samples, but found {len(dataset)}. "
            "Please verify the files under data/MNIST/raw."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def load_model_weights(model: nn.Module, checkpoint_path: Path) -> dict:
    """Load checkpoint robustly and return metadata for logging.

    Supported formats:
    - training checkpoint dict with key: model_state_dict
    - raw state_dict directly saved as a dict of tensors
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    metadata = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata["epoch"] = checkpoint.get("epoch")
        metadata["val_acc"] = checkpoint.get("val_acc")
    elif isinstance(checkpoint, dict):
        # Assume this is already a raw state_dict.
        state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format. Expected dict or checkpoint dict.")

    model.load_state_dict(state_dict)
    return metadata


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    optical_model: Any,
    input_type: str,
    act_clip_max: float,
) -> tuple[float, float, list[float], float]:
    """Run optical-evaluation and report top1 + optical compute ratio.

    Per-class metrics are useful when aggregate accuracy looks good but specific
    digits degrade (for example, confusing 4 and 9).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    optical_macs_total = 0
    all_macs_total = 0

    for images, labels in loader:
        # Optical backend path is CPU/Numpy based, so force data onto CPU.
        images = images.to("cpu", non_blocking=True)
        labels = labels.to("cpu", non_blocking=True)

        logits = forward_with_optical_matmul(
            model=model,
            images=images,
            optical_model=optical_model,
            input_type=input_type,
            act_clip_max=act_clip_max,
        )
        loss = criterion(logits, labels)

        # Accumulate sample-weighted loss so final average is exact.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        matches = preds.eq(labels)

        total_correct += matches.sum().item()
        total_samples += batch_size

        # Track class-wise correctness to reveal category-level behavior.
        for class_idx in range(10):
            class_mask = labels.eq(class_idx)
            class_count = class_mask.sum().item()
            if class_count == 0:
                continue
            class_total[class_idx] += class_count
            class_correct[class_idx] += preds[class_mask].eq(labels[class_mask]).sum().item()

        # MAC accounting for optical compute ratio.
        # conv1: B*28*28*32*(1*3*3)
        # conv2: B*14*14*64*(32*3*3)
        # fc1: B*3136*128
        # fc2: B*128*10
        bsz = labels.size(0)
        conv1_macs = bsz * 28 * 28 * 32 * (1 * 3 * 3)
        conv2_macs = bsz * 14 * 14 * 64 * (32 * 3 * 3)
        fc1_macs = bsz * 3136 * 128
        fc2_macs = bsz * 128 * 10

        optical_macs_total += conv2_macs + fc1_macs + fc2_macs
        all_macs_total += conv1_macs + conv2_macs + fc1_macs + fc2_macs

    avg_loss = total_loss / total_samples
    overall_acc = total_correct / total_samples

    per_class_acc = []
    for class_idx in range(10):
        if class_total[class_idx] == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(class_correct[class_idx] / class_total[class_idx])

    optical_ratio = 0.0 if all_macs_total == 0 else (optical_macs_total / all_macs_total)
    return avg_loss, overall_acc, per_class_acc, optical_ratio


def resolve_path(project_root: Path, user_path: str) -> Path:
    """Resolve user path to an absolute path using project root fallback."""
    path = Path(user_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def main() -> None:
    """Program entry point for model evaluation."""
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_root = resolve_path(project_root, args.data_root)
    checkpoint_path = resolve_path(project_root, args.checkpoint)

    device = select_device(args.device)

    print(f"Project root: {project_root}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Using device for torch-side ops: {device}")
    print("Optical backend: osimulator (Gazelle model)")

    loader = build_test_loader(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optical_model = load_optical_model()
    model = SimpleMNISTCNN().to("cpu")
    metadata = load_model_weights(model=model, checkpoint_path=checkpoint_path)

    avg_loss, overall_acc, per_class_acc, optical_ratio = evaluate(
        model=model,
        loader=loader,
        optical_model=optical_model,
        input_type=args.input_type,
        act_clip_max=args.act_clip_max,
    )

    if metadata:
        print("Checkpoint metadata:")
        print(f"  epoch: {metadata.get('epoch')}")
        print(f"  val_acc: {metadata.get('val_acc')}")

    print("\nEvaluation results on MNIST 10k test set:")
    print(f"  avg_loss: {avg_loss:.6f}")
    print(f"  top1_accuracy: {overall_acc:.4%}")
    print(f"  optical_compute_ratio: {optical_ratio:.4%}")

    print("\nPer-class accuracy:")
    for class_idx, acc in enumerate(per_class_acc):
        print(f"  class {class_idx}: {acc:.4%}")


if __name__ == "__main__":
    main()