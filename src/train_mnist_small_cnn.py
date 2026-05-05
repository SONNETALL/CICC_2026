"""Train a small CNN on the MNIST dataset.

Usage examples:
    python src/train_mnist_small_cnn.py
    python src/train_mnist_small_cnn.py --epochs 8 --batch-size 128
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
    parser = argparse.ArgumentParser(description="Train a Small CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="models/mnist_small_cnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallMNISTCNN(nn.Module):
    """A very compact CNN baseline for MNIST.
    Architecture:
    - Conv(1 -> 6) + ReLU + MaxPool(2)
    - Conv(6 -> 12) + ReLU + MaxPool(2)
    - Flatten
    - Linear(12*7*7 -> 24) + ReLU
    - Linear(24 -> 10)
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(12 * 7 * 7, 24)
        self.fc2 = nn.Linear(24, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)  # 28x28 -> 14x14
        
        # Block 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)  # 14x14 -> 7x7
        
        # Classifier
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def build_dataloaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root=str(data_root), train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
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
) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % log_interval == 0 or batch_idx == len(loader):
            avg_loss_so_far = running_loss / total
            avg_acc_so_far = correct / total
            print(f"Epoch [{epoch_idx}/{total_epochs}] "
                  f"Batch [{batch_idx}/{len(loader)}] "
                  f"Avg Loss: {avg_loss_so_far:.4f}, Avg Acc: {avg_acc_so_far:.4f}")

    return correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Directories
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(data_root, args.batch_size)

    model = SmallMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_checkpoint_path = output_dir / "best_mnist_small_cnn.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch} starts ---")
        train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
        )
        
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f"--- Epoch {epoch} summary ---")
        print(f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy! Saving to {best_checkpoint_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, best_checkpoint_path)

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
