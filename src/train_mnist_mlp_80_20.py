"""Train a 2-hidden-layer MLP on MNIST.

模型结构固定为：
- 输入层: 784 (28x28 展平)
- 隐藏层1: 80
- 隐藏层2: 20
- 输出层: 10

脚本流程：
1) 读取本地 MNIST 数据（data 目录）
2) 训练/评估双层 MLP
3) 按验证准确率保存最佳模型到 models/mnist_mlp_80_20

Usage examples:
    python src/train_mnist_mlp_80_20.py
    python src/train_mnist_mlp_80_20.py --epochs 20 --batch-size 128
    python src/train_mnist_mlp_80_20.py --device cuda
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
    """解析命令行参数，方便快速调参。"""
    parser = argparse.ArgumentParser(description="Train a 2-hidden-layer MLP (80,20) on MNIST")

    # 训练轮次。
    parser.add_argument("--epochs", type=int, default=50)

    # mini-batch 大小。
    parser.add_argument("--batch-size", type=int, default=128)

    # 学习率。
    parser.add_argument("--lr", type=float, default=1e-3)

    # Dropout
    parser.add_argument("--dropout", type=float, default=0.10)

    # 权重衰减。
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    # 设备选择，auto 会优先 CUDA。
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # 模型输出目录。
    parser.add_argument("--output-dir", type=str, default="models/mnist_mlp_80_20")

    # 随机种子。
    parser.add_argument("--seed", type=int, default=42)

    # 每隔多少个 batch 打印一次训练指标。
    parser.add_argument("--log-interval", type=int, default=50)

    # ── QAT (Quantization-Aware Training) ──
    parser.add_argument(
        "--qat",
        action="store_true",
        default=False,
        help="Enable fake-quantization during training to match optical int4 inference.",
    )
    parser.add_argument(
        "--act-clip-fc1",
        type=float,
        default=4.0,
        help="Activation clip max for fc1 fake quant (ignored without --qat).",
    )
    parser.add_argument(
        "--act-clip-fc2",
        type=float,
        default=0.0,
        help="Activation clip max for fc2 fake quant. Default 0 = no fake quant for fc2 (matches optical default where only fc1 is offloaded).",
    )
    parser.add_argument(
        "--act-clip-fc3",
        type=float,
        default=0.0,
        help="Activation clip max for fc3 fake quant. Default 0 = no fake quant for fc3.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，提升实验可复现性。"""
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


def fake_quant_activation(x: torch.Tensor, clip_max: float) -> torch.Tensor:
    """int4 对称伪量化, 前向量化/反量化, 反向用 STE 直传梯度.

    模拟光学推理路径的 int4 量化行为:
        scale = clip_max / 7
        q = round(clamp(x, -clip_max, clip_max) / scale).clamp(-8, 7)
        x_deq = q * scale
    """
    if clip_max <= 0:
        return x
    scale = clip_max / 7.0
    x_clipped = x.clamp(-clip_max, clip_max)
    x_q = (x_clipped / scale).round().clamp(-8, 7)
    x_deq = x_q * scale
    # STE: 前向取量化值, 反向梯度穿过 round
    return x + (x_deq - x).detach()


class MLP80x20(nn.Module):
    """MNIST 双隐藏层 MLP（80, 20），可选 QAT。

    self.net 保持 nn.Sequential 不变，确保 state_dict 与旧 checkpoint
    及光学测试脚本中的索引访问（net.1 / net.4 / net.7）完全兼容。
    QAT 仅在 forward 中手动穿插 fake_quant，不改变模块结构。
    """

    def __init__(
        self,
        dropout: float = 0.2,
        qat: bool = False,
        act_clip_fc1: float = 4.0,
        act_clip_fc2: float = 4.0,
        act_clip_fc3: float = 4.0,
    ) -> None:
        super().__init__()

        self.qat = qat
        self.act_clip_fc1 = act_clip_fc1
        self.act_clip_fc2 = act_clip_fc2
        self.act_clip_fc3 = act_clip_fc3

        self.net = nn.Sequential(
            nn.Flatten(),             # 0
            nn.Linear(28 * 28, 80),  # 1  fc1
            nn.ReLU(inplace=True),    # 2
            nn.Dropout(p=dropout),    # 3
            nn.Linear(80, 20),        # 4  fc2
            nn.ReLU(inplace=True),    # 5
            nn.Dropout(p=dropout),    # 6
            nn.Linear(20, 10),        # 7  fc3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.qat:
            return self.net(x)

        # QAT 路径：逐模块调用，在全连接后插入伪量化
        x = self.net[0](x)  # Flatten
        x = self.net[1](x)  # fc1: Linear(784→80)
        x = fake_quant_activation(x, self.act_clip_fc1)
        x = self.net[2](x)  # ReLU
        x = self.net[3](x)  # Dropout
        x = self.net[4](x)  # fc2: Linear(80→20)
        if self.act_clip_fc2 > 0:
            x = fake_quant_activation(x, self.act_clip_fc2)
        x = self.net[5](x)  # ReLU
        x = self.net[6](x)  # Dropout
        x = self.net[7](x)  # fc3: Linear(20→10)
        if self.act_clip_fc3 > 0:
            x = fake_quant_activation(x, self.act_clip_fc3)
        return x


def build_dataloaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """构建 MNIST 的训练与测试数据加载器。"""
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=8, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # 使用本地数据，不自动下载，避免离线环境失败。
    train_dataset = datasets.MNIST(root=str(data_root), train=True, transform=train_transform, download=False)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, transform=test_transform, download=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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
    """训练一个 epoch，返回平均 loss 与准确率。"""
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

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        batch_correct = (predictions == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            batch_loss = loss.item()
            batch_acc = batch_correct / labels.size(0)
            avg_loss = running_loss / total
            avg_acc = correct / total
            print(
                f"[Train] Epoch {epoch_idx:02d}/{total_epochs:02d} "
                f"Batch {batch_idx:04d}/{total_batches:04d} | "
                f"batch_loss={batch_loss:.4f}, batch_acc={batch_acc:.4f} | "
                f"avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}",
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
    """在测试集上评估模型，返回平均 loss 与准确率。"""
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

    return running_loss / total, correct / total


def main() -> None:
    """程序入口。"""
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(data_root=data_root, batch_size=args.batch_size)

    model = MLP80x20(
        dropout=args.dropout,
        qat=args.qat,
        act_clip_fc1=args.act_clip_fc1,
        act_clip_fc2=args.act_clip_fc2,
        act_clip_fc3=args.act_clip_fc3,
    ).to(device)

    if args.qat:
        print(
            "QAT enabled — fake int4 quantization active during training. "
            f"clip: fc1={args.act_clip_fc1}, fc2={args.act_clip_fc2}, fc3={args.act_clip_fc3}"
        )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_ckpt_path = output_dir / "best_mnist_mlp_80_20.pt"

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

        scheduler.step()

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
                    "args": vars(args),
                    "qat_config": {
                        "qat": args.qat,
                        "act_clip_fc1": args.act_clip_fc1,
                        "act_clip_fc2": args.act_clip_fc2,
                        "act_clip_fc3": args.act_clip_fc3,
                    },
                },
                best_ckpt_path,
            )
            print(f"Saved new best checkpoint to: {best_ckpt_path}")

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
