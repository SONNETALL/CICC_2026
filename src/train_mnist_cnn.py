"""Train a CNN on the MNIST dataset.

This script is intentionally written as a single, readable training entry point
with detailed comments for beginners:
1) Load data (MNIST)
2) Define a simple CNN model
3) Train + validate for multiple epochs
4) Save the best checkpoint by validation accuracy

Usage examples:
    python src/train_mnist_cnn.py
    python src/train_mnist_cnn.py --epochs 8 --batch-size 128
    python src/train_mnist_cnn.py --device cuda
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments so the script is easy to tune."""
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")

    # Number of full passes over the entire training dataset.
    parser.add_argument("--epochs", type=int, default=8)

    # Number of samples per mini-batch. Larger batch -> faster but higher memory use.
    parser.add_argument("--batch-size", type=int, default=64)

    # Learning rate for the optimizer. Typical MNIST values are 1e-3 to 1e-2.
    parser.add_argument("--lr", type=float, default=1e-3)

    # Optional explicit device selection. If omitted, we auto-select.
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Directory where model checkpoints and logs are saved.
    parser.add_argument("--output-dir", type=str, default="models/mnist_cnn")

    # Random seed for reproducibility.
    parser.add_argument("--seed", type=int, default=42)

    # 每隔多少个 batch 打印一次实时训练指标。
    # 例如 50 表示每 50 个 mini-batch 打印一次当前 loss/acc 和累计平均值。
    parser.add_argument("--log-interval", type=int, default=50)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for Python and PyTorch to improve reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleMNISTCNN(nn.Module):
    """A compact CNN baseline for MNIST (1x28x28 grayscale images).

    Architecture:
    - Conv(1 -> 32) + ReLU + MaxPool
    - Conv(32 -> 64) + ReLU + MaxPool
    - Flatten
    - Linear(64*7*7 -> 128) + ReLU
    - Dropout
    - Linear(128 -> 10)
    """

    def __init__(self) -> None:
        super().__init__()

        # Feature extractor: convert image pixels to higher-level feature maps.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 14x14 -> 7x7
        )

        # Classifier: map extracted features to 10 digit classes.
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


def build_dataloaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create training and test dataloaders for MNIST.

    We normalize MNIST using standard mean/std values for grayscale digits.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # download=True allows this script to bootstrap itself on a new machine.
    train_dataset = datasets.MNIST(root=str(data_root), train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, transform=transform, download=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle train data to improve SGD behavior.
        num_workers=0,  # Keep 0 for maximum compatibility on Windows.
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    log_interval: int,
) -> tuple[float, float]:
    """Run one training epoch and return average loss/accuracy.

    这里会进行“实时打印”：
    - 当前 batch 的 loss/acc
    - 从本轮开始到当前 batch 的累计平均 loss/acc
    """
    model.train()

    # running_loss 记录的是“加权求和后的总 loss”（每个 batch loss * batch_size）
    # 这样最终除以 total 后，得到的是严格意义上的样本级平均 loss。
    running_loss = 0.0
    correct = 0
    total = 0

    # 为了打印进度，需要知道总 batch 数。
    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        # 将输入和标签搬到目标设备（CPU 或 GPU）。
        # non_blocking=True 在使用 pinned memory 时有助于提升数据搬运效率。
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Clear old gradients from previous mini-batch.
        optimizer.zero_grad()

        # Forward pass: compute logits.
        logits = model(images)

        # Compute classification loss.
        loss = criterion(logits, labels)

        # Backward pass: compute gradients.
        loss.backward()

        # Parameter update step.
        optimizer.step()

        # ===== 统计指标 =====
        # loss.item() 是该 batch 的平均 loss，乘上 batch 大小后可累计为全样本总 loss。
        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        batch_correct = (predictions == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)

        # 当前 batch 指标（即时反馈）
        batch_loss = loss.item()
        batch_acc = batch_correct / labels.size(0)

        # 当前累计平均指标（更平滑，代表本轮到目前为止的整体表现）
        avg_loss_so_far = running_loss / total
        avg_acc_so_far = correct / total

        # 按设定间隔打印，或在最后一个 batch 强制打印一次。
        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            print(
                f"[Train] Epoch {epoch_idx:02d}/{total_epochs:02d} "
                f"Batch {batch_idx:04d}/{total_batches:04d} | "
                f"batch_loss={batch_loss:.4f}, batch_acc={batch_acc:.4f} | "
                f"avg_loss={avg_loss_so_far:.4f}, avg_acc={avg_acc_so_far:.4f}",
                flush=True,
            )

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on validation/test set without gradient computation.

    评估阶段不做反向传播，因此速度更快、显存占用更低。
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def select_device(device_arg: str) -> torch.device:
    """Resolve the actual runtime device from user argument."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was set, but CUDA is not available.")
        return torch.device("cuda")

    # Auto mode: choose CUDA when available, otherwise CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    # 1) 读取超参数并设置随机种子。
    args = parse_args()
    set_seed(args.seed)

    # 2) 生成路径：
    # - data_root: MNIST 数据位置
    # - output_dir: 模型权重保存目录
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data" 
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) 选择设备并构建数据加载器。
    device = select_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(data_root=data_root, batch_size=args.batch_size)

    # 4) 初始化模型、损失函数、优化器。
    model = SimpleMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 记录历史最佳验证准确率，用于保存最佳权重。
    best_acc = 0.0
    best_ckpt_path = output_dir / "best_mnist_cnn.pt"

    # 5) 进入训练主循环。
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            log_interval=max(1, args.log_interval),
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        # 这是每个 epoch 结束后的汇总信息。
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Save the best model according to validation accuracy.
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"Saved new best checkpoint to: {best_ckpt_path}")

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()