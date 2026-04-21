"""Evaluate BNN binary-state checkpoint with optical matmul on MNIST test split.

该脚本参照 test_mnist_cnn.py 的评测流程实现，针对 BNN 的
best_mnist_bnn_binary_state.pt 做推理测试，并增加计时统计。

光学加速映射策略：
1) stem conv 保留 torch 侧执行
2) block1 conv / block2 conv 通过 im2col + 光学矩阵乘法执行
3) fc1 / fc2 通过光学矩阵乘法执行

输出指标包括：
- top1 accuracy（MNIST 10k 测试集）
- optical compute ratio（光学卸载 MAC 占比）
- 计时指标（总耗时、前向耗时、光学调用耗时、吞吐）

Usage examples:
    python test/test_mnist_bnn_binary_state.py
    python test/test_mnist_bnn_binary_state.py --batch-size 128
    python test/test_mnist_bnn_binary_state.py --input-type int4 --act-clip-max 4.0
"""

from __future__ import annotations

import argparse
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


# 让测试脚本可以直接复用 src 下官方 MAC 统计实现。
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from op_computation_analysis import calculate_linear_macs


def parse_bool_flag(value: str) -> bool:
    """将字符串参数解析为布尔值。"""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate MNIST BNN binary-state checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mnist_bnn/best_mnist_bnn_binary_state.pt",
        help="Path to BNN binary-state checkpoint (relative to project root or absolute path).",
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
        default=128,
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
        default="fc2",
        help=(
            "Comma-separated layers offloaded to optical simulator. "
            "Valid names: block1,block2,fc1,fc2. "
            "Example: --optical-layers fc1,fc2"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1_000,
        help=(
            "Maximum number of test samples to evaluate. "
            "Use a smaller value (e.g. 1000) for quick timing runs."
        ),
    )
    return parser.parse_args()


def parse_optical_layers(raw: str) -> set[str]:
    """解析并校验光学卸载层列表。"""
    valid = {"block1", "block2", "fc1", "fc2"}
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}

    invalid = sorted(items - valid)
    if invalid:
        raise ValueError(
            "Invalid optical layer names: "
            f"{invalid}. Valid names are: {sorted(valid)}"
        )
    return items


def select_device(device_arg: str) -> torch.device:
    """Resolve runtime device from user argument."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was set, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_sign_inference(x: torch.Tensor) -> torch.Tensor:
    """推理阶段二值化：x>=0 映射 +1，否则 -1。"""
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


class BinaryConv2d(nn.Conv2d):
    """二值卷积层（推理版）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_weight = binary_sign_inference(self.weight)
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
    """二值全连接层（推理版）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_weight = binary_sign_inference(self.weight)
        return F.linear(x, binary_weight, self.bias)


class MNISTBNN(nn.Module):
    """与训练脚本一致的 BNN 结构定义。"""

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
        x = self.stem(x)
        x = binary_sign_inference(x)

        x = self.block1(x)
        x = binary_sign_inference(x)

        x = self.block2(x)
        x = binary_sign_inference(x)

        logits = self.classifier(x)
        return logits


def build_official_macs_profile(
    model: nn.Module,
    optical_layers: set[str],
) -> tuple[int, int, dict[str, int], dict[str, str]]:
    """使用官方 MAC 统计代码构建本模型的计算量画像。

    Returns:
        total_macs_per_sample: 每个样本的总 MAC
        optical_macs_per_sample: 选中光学层对应的每样本 MAC
        macs_dict: 官方函数返回的逐算子 MAC 明细
        layer_to_module_name: 逻辑层名到模块名映射
    """
    # 官方统计函数约束 batch=1。
    dummy_input = torch.randn(1, 1, 28, 28)
    total_macs_per_sample, macs_dict = calculate_linear_macs(model, dummy_input)

    # 逻辑层名与模型模块名映射。
    layer_to_module_name = {
        "block1": "block1.0",
        "block2": "block2.0",
        "fc1": "classifier.1",
        "fc2": "classifier.4",
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
        # 这里采用“偏移编码”规避：
        # 1) 先量化到 q_signed in [-8, 7]
        # 2) 再映射 q_uint = q_signed + 8，走稳定的 uint4 通路
        # 3) 在输出端减去 zero_point * sum(w_q) 做等价还原
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
    """通过 osimulator 执行矩阵乘法并累计光学调用计时。

    Expected shapes:
    - a_bmk: [B, M, K]
    - w_kn:  [K, N]
    """
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

    # int4 偏移编码模式下，仅首次打印一次提示，避免日志刷屏。
    if input_type == "int4" and "int4_emulation_notice_printed" not in timing_stats:
        print("[Info] input_type=int4 is emulated via uint4 offset encoding for simulator compatibility.")
        timing_stats["int4_emulation_notice_printed"] = 1.0

    # 将逐样本调用改为整批调用，可显著减少模拟器矩阵计算调用次数。
    input_tensors = a_q.astype(np.int32, copy=False)  # [B, M, K]
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

    # 偏移编码还原：
    # q_signed @ w = (q_uint - zp) @ w = q_uint @ w - zp * sum(w)
    if a_zero_point != 0:
        w_sum = np.sum(w_q.astype(np.float32), axis=0, keepdims=True)  # [1, N]
        out_concat = out_concat.astype(np.float32) - (float(a_zero_point) * w_sum[None, :, :])

    out_fp32 = out_concat.astype(np.float32) * (a_scale * w_scale)
    return torch.from_numpy(out_fp32)


def conv2d_output_hw(
    h: int,
    w: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """计算 conv2d 输出空间尺寸。"""
    out_h = (h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_w = (w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return out_h, out_w


def _as_pair(value: Any) -> tuple[int, int]:
    """将 int 或 tuple 统一为二元组。"""
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def optical_conv2d_from_im2col(
    optical_model: Any,
    x: torch.Tensor,
    conv: nn.Conv2d,
    *,
    input_type: str,
    act_clip_max: float,
    timing_stats: dict[str, float],
) -> torch.Tensor:
    """将 conv2d 重写为 im2col + 光学 matmul。"""
    k = _as_pair(conv.kernel_size)
    s = _as_pair(conv.stride)
    p = _as_pair(conv.padding)
    d = _as_pair(conv.dilation)

    x_unfold = F.unfold(x, kernel_size=k, dilation=d, padding=p, stride=s)
    x_bmk = x_unfold.transpose(1, 2).contiguous()

    weight_kn = conv.weight.view(conv.out_channels, -1).transpose(0, 1).contiguous()
    out_bmn = run_optical_matmul(
        optical_model=optical_model,
        a_bmk=x_bmk,
        w_kn=weight_kn,
        input_type=input_type,
        act_clip_max=act_clip_max,
        timing_stats=timing_stats,
    )

    bsz, _, h_in, w_in = x.shape
    out_h, out_w = conv2d_output_hw(h_in, w_in, kernel_size=k, stride=s, padding=p, dilation=d)
    out = out_bmn.transpose(1, 2).contiguous().view(bsz, conv.out_channels, out_h, out_w)

    if conv.bias is not None:
        out = out + conv.bias.detach().cpu().view(1, -1, 1, 1)
    return out


def forward_with_optical_matmul(
    model: nn.Module,
    images: torch.Tensor,
    optical_model: Any,
    *,
    input_type: str,
    act_clip_max: float,
    timing_stats: dict[str, float],
    optical_layers: set[str],
) -> torch.Tensor:
    """将 BNN 中矩阵乘法密集层替换为光学 matmul 前向。"""
    # stem conv 保留 torch 侧执行。
    stem_conv = model.stem[0]
    stem_bn = model.stem[1]

    x = F.conv2d(
        images,
        stem_conv.weight,
        stem_conv.bias,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
        dilation=stem_conv.dilation,
        groups=stem_conv.groups,
    )
    x = stem_bn(x)
    x = F.hardtanh(x, inplace=False)
    x = binary_sign_inference(x)

    # block1 conv -> 光学 matmul。
    block1_conv = model.block1[0]
    block1_bn = model.block1[1]
    if "block1" in optical_layers:
        x = optical_conv2d_from_im2col(
            optical_model=optical_model,
            x=x,
            conv=block1_conv,
            input_type=input_type,
            act_clip_max=act_clip_max,
            timing_stats=timing_stats,
        )
    else:
        x = block1_conv(x)
    x = block1_bn(x)
    x = F.max_pool2d(x, kernel_size=2)
    x = binary_sign_inference(x)

    # block2 conv -> 光学 matmul。
    block2_conv = model.block2[0]
    block2_bn = model.block2[1]
    if "block2" in optical_layers:
        x = optical_conv2d_from_im2col(
            optical_model=optical_model,
            x=x,
            conv=block2_conv,
            input_type=input_type,
            act_clip_max=act_clip_max,
            timing_stats=timing_stats,
        )
    else:
        x = block2_conv(x)
    x = block2_bn(x)
    x = F.max_pool2d(x, kernel_size=2)
    x = binary_sign_inference(x)

    # fc1 -> 光学 matmul。
    x = torch.flatten(x, start_dim=1)
    fc1 = model.classifier[1]
    if "fc1" in optical_layers:
        fc1_w_kn = fc1.weight.transpose(0, 1).contiguous()
        fc1_out = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x.unsqueeze(1),
            w_kn=fc1_w_kn,
            input_type=input_type,
            act_clip_max=act_clip_max,
            timing_stats=timing_stats,
        ).squeeze(1)
        if fc1.bias is not None:
            fc1_out = fc1_out + fc1.bias.detach().cpu()
    else:
        fc1_out = fc1(x)

    fc1_bn = model.classifier[2]
    x = fc1_bn(fc1_out)

    # dropout 在 eval 模式下等价恒等映射。
    fc2 = model.classifier[4]
    if "fc2" in optical_layers:
        fc2_w_kn = fc2.weight.transpose(0, 1).contiguous()
        logits = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x.unsqueeze(1),
            w_kn=fc2_w_kn,
            input_type=input_type,
            act_clip_max=act_clip_max,
            timing_stats=timing_stats,
        ).squeeze(1)
        logits = logits + fc2.bias.detach().cpu()
    else:
        logits = fc2(x)
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


def load_model_weights(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    """加载 binary-state checkpoint 并返回元信息。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata["epoch"] = checkpoint.get("epoch")
        metadata["val_acc"] = checkpoint.get("val_acc")
        metadata["pack_note"] = checkpoint.get("pack_note")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format. Expected dict.")

    model.load_state_dict(state_dict)
    return metadata


@torch.no_grad()
def evaluate(
    model: nn.Module,
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

        # 光学后端是 CPU/Numpy 路径，因此输入固定走 CPU。
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

        # 使用官方 MAC 口径进行占比累计，避免手写公式偏差。
        bsz = labels.size(0)
        optical_macs_total += optical_macs_per_sample * bsz
        all_macs_total += total_macs_per_sample * bsz

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


def resolve_path(project_root: Path, user_path: str) -> Path:
    """Resolve user path to an absolute path using project root fallback."""
    path = Path(user_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def main() -> None:
    """程序入口。"""
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_root = resolve_path(project_root, args.data_root)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    optical_layers = parse_optical_layers(args.optical_layers)

    device = select_device(args.device)

    print(f"Project root: {project_root}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Using device for torch-side ops: {device}")
    print("Optical backend: osimulator (Gazelle model)")
    print(f"Optical offload layers: {sorted(optical_layers)}")

    loader = build_test_loader(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optical_model = load_optical_model()
    model = MNISTBNN().to("cpu")
    metadata = load_model_weights(model=model, checkpoint_path=checkpoint_path)

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
    print(f"    stem.0: {macs_dict.get('stem.0', 0):,}")
    print(f"    {layer_to_module_name['block1']}: {macs_dict[layer_to_module_name['block1']]:,}")
    print(f"    {layer_to_module_name['block2']}: {macs_dict[layer_to_module_name['block2']]:,}")
    print(f"    {layer_to_module_name['fc1']}: {macs_dict[layer_to_module_name['fc1']]:,}")
    print(f"    {layer_to_module_name['fc2']}: {macs_dict[layer_to_module_name['fc2']]:,}")

    expected_batches = (args.max_samples + args.batch_size - 1) // args.batch_size
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

    if metadata:
        print("Checkpoint metadata:")
        print(f"  epoch: {metadata.get('epoch')}")
        print(f"  val_acc: {metadata.get('val_acc')}")
        print(f"  pack_note: {metadata.get('pack_note')}")

    print("\nEvaluation results on MNIST 10k test set:")
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


if __name__ == "__main__":
    main()
