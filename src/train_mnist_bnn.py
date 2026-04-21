"""Train a Binary Neural Network (BNN) on MNIST.

该脚本基于 Straight-Through Estimator (STE) 训练二值网络：
1) 使用 torchvision 读取本地 MNIST 数据（不下载）
2) 在前向传播中对激活与部分权重做二值化（-1/+1）
3) 使用交叉熵损失训练并在测试集评估
4) 按测试准确率保存最佳模型到 models/mnist_bnn

Usage examples:
    python src/train_mnist_bnn.py
    python src/train_mnist_bnn.py --epochs 20 --batch-size 128 --lr 1e-3
    python src/train_mnist_bnn.py --device cuda --clip-value 1.0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便快速调参。"""
    parser = argparse.ArgumentParser(description="Train a BNN on MNIST")

    # 训练轮次。
    parser.add_argument("--epochs", type=int, default=20)

    # 每个 mini-batch 的样本数。
    parser.add_argument("--batch-size", type=int, default=128)

    # 学习率。
    parser.add_argument("--lr", type=float, default=1e-3)

    # 优化器权重衰减。
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    # 设备选择：auto 优先 CUDA。
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # 输出目录：默认保存到 models/mnist_bnn。
    parser.add_argument("--output-dir", type=str, default="models/mnist_bnn")

    # 随机种子。
    parser.add_argument("--seed", type=int, default=42)

    # 每隔多少个 batch 打印一次训练日志。
    parser.add_argument("--log-interval", type=int, default=50)

    # 对二值层实值权重做裁剪，保持训练稳定。
    parser.add_argument("--clip-value", type=float, default=1.0)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机性，提升实验可复现性。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SignSTE(torch.autograd.Function):
    """符号函数的直通估计器（STE）。

    前向：输出 -1/+1。
    反向：仅在输入绝对值不大于 1 的区域传递梯度，
    这是 BNN 中常见的梯度近似方式。
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        # 将 0 也映射到 +1，避免出现三值 {-1,0,+1}。
        return torch.where(input_tensor >= 0, torch.ones_like(input_tensor), -torch.ones_like(input_tensor))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input_tensor,) = ctx.saved_tensors
        grad_mask = (input_tensor.abs() <= 1.0).to(dtype=grad_output.dtype)
        return grad_output * grad_mask


def binary_sign(x: torch.Tensor) -> torch.Tensor:
    """对输入执行可训练的二值化映射。"""
    return SignSTE.apply(x)


class BinaryConv2d(nn.Conv2d):
    """二值卷积层。

    参数仍以浮点形式存储和更新，前向时将 weight 二值化。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_weight = binary_sign(self.weight)
        return F.conv2d(
            x,
            binary_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BinaryLinear(nn.Linear):
    """二值全连接层。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_weight = binary_sign(self.weight)
        return F.linear(x, binary_weight, self.bias)


class MNISTBNN(nn.Module):
    """用于 MNIST 分类的轻量 BNN。

    设计说明：
    1) 第一层卷积保持浮点，有助于稳健地提取初始特征。
    2) 中间层使用二值卷积/二值全连接，降低乘加复杂度。
    3) 最后一层分类头保持浮点，保证分类边界表达能力。
    """

    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(inplace=True),
        )

        self.block1 = nn.Sequential(
            BinaryConv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )

        self.block2 = nn.Sequential(
            BinaryConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(64 * 7 * 7, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对中间特征做激活二值化，减少后续计算复杂度。
        x = self.stem(x)
        x = binary_sign(x)

        x = self.block1(x)
        x = binary_sign(x)

        x = self.block2(x)
        x = binary_sign(x)

        logits = self.classifier(x)
        return logits


def build_dataloaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """构建 MNIST 的训练/测试数据加载器。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # 这里明确使用本地数据，避免脚本在无网络环境下失败。
    train_dataset = datasets.MNIST(root=str(data_root), train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, transform=transform, download=False)

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


def select_device(device_arg: str) -> torch.device:
    """根据参数选择运行设备。"""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was set, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clamp_binary_weights(model: nn.Module, clip_value: float) -> None:
    """将二值层的实值权重裁剪到 [-clip_value, clip_value]。"""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (BinaryConv2d, BinaryLinear)):
                module.weight.clamp_(min=-clip_value, max=clip_value)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    log_interval: int,
    clip_value: float,
) -> tuple[float, float]:
    """训练一个 epoch，并返回平均 loss/accuracy。"""
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

        # 每次参数更新后裁剪二值层权重，防止权重幅值持续膨胀。
        clamp_binary_weights(model, clip_value=clip_value)

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
    """在测试集上评估模型。"""
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
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(data_root=data_root, batch_size=args.batch_size)

    model = MNISTBNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Adam 在 BNN 训练中通常更稳健；可根据实验再换成 SGD。
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    best_ckpt_path = output_dir / "best_mnist_bnn.pt"

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
            clip_value=args.clip_value,
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
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"Saved new best checkpoint to: {best_ckpt_path}")

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
