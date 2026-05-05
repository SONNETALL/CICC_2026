from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SingleHiddenMLP(nn.Module):
    def __init__(self, hidden_size: int = 64, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, 10, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


def parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-hidden-layer MLP on MNIST")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--bias", type=parse_bool_flag, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was set, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(data_root: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(root=str(data_root), train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, transform=transform, download=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        batch_correct = predictions.eq(labels).sum().item()
        correct += batch_correct
        total += batch_size

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            print(
                f"[Train] Epoch {epoch_idx:02d}/{total_epochs:02d} "
                f"Batch {batch_idx:04d}/{total_batches:04d} | "
                f"batch_loss={loss.item():.4f}, batch_acc={batch_correct / batch_size:.4f} | "
                f"avg_loss={running_loss / total:.4f}, avg_acc={correct / total:.4f}",
                flush=True,
            )

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def main() -> None:
    args = parse_args()
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive.")

    set_seed(args.seed)
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    output_dir = (
        project_root / args.output_dir
        if args.output_dir is not None
        else project_root / "models" / f"mnist_mlp_single_hidden_H{args.hidden_size}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Bias: {args.bias}")
    print(f"Output dir: {output_dir}")

    train_loader, test_loader = build_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = SingleHiddenMLP(hidden_size=args.hidden_size, bias=args.bias).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    best_ckpt_path = output_dir / f"best_mnist_mlp_single_hidden_H{args.hidden_size}.pt"

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
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "hidden_size": args.hidden_size,
                    "bias": args.bias,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"Saved new best checkpoint to: {best_ckpt_path}")

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
