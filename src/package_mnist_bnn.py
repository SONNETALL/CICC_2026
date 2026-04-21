"""Package trained MNIST BNN checkpoint into deployment artifacts.

该脚本用于“最终打包” BNN 模型，生成两类产物：
1) 位压缩打包文件（推荐部署使用）
   - 将二值层权重从 float 映射到 {-1, +1}
   - 再将符号位按 bit 打包到 uint8，显著减小存储体积
2) 二值化 state_dict（便于在 PyTorch 中直接加载做功能验证）

默认输入与输出：
- 输入:  models/mnist_bnn/best_mnist_bnn.pt
- 输出1: models/mnist_bnn/best_mnist_bnn_binary_pack.pt
- 输出2: models/mnist_bnn/best_mnist_bnn_binary_state.pt

Usage examples:
    python src/package_mnist_bnn.py
    python src/package_mnist_bnn.py --threshold 0.0
    python src/package_mnist_bnn.py --strict true
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from train_mnist_bnn import BinaryConv2d, BinaryLinear, MNISTBNN


def parse_bool_flag(value: str) -> bool:
    """将命令行字符串解析为布尔值。"""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="Package trained MNIST BNN for deployment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mnist_bnn/best_mnist_bnn.pt",
        help="训练后保存的最佳 BNN checkpoint 路径。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mnist_bnn/best_mnist_bnn_binary_pack.pt",
        help="位压缩打包文件输出路径。",
    )
    parser.add_argument(
        "--save-binary-state",
        type=parse_bool_flag,
        default=True,
        help="是否同时导出可直接 torch.load 的二值化 state_dict。",
    )
    parser.add_argument(
        "--binary-state-output",
        type=str,
        default="models/mnist_bnn/best_mnist_bnn_binary_state.pt",
        help="二值化 state_dict 输出路径。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="二值化阈值：weight >= threshold 映射为 +1，否则 -1。",
    )
    parser.add_argument(
        "--strict",
        type=parse_bool_flag,
        default=True,
        help="是否严格检查 checkpoint 中必须包含所有二值层权重。",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, user_path: str) -> Path:
    """将用户输入路径转换为绝对路径。"""
    path = Path(user_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def extract_state_dict(checkpoint_obj: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any], bool]:
    """从不同 checkpoint 格式中提取 state_dict。

    Returns:
        state_dict: 模型权重字典
        metadata:   其他辅助元信息
        wrapped:    原始 checkpoint 是否使用 model_state_dict 包裹
    """
    if not isinstance(checkpoint_obj, dict):
        raise TypeError("Checkpoint format is unsupported: expected dict.")

    if "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
        metadata = {k: v for k, v in checkpoint_obj.items() if k != "model_state_dict"}
        return state_dict, metadata, True

    all_tensor_values = all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values())
    if all_tensor_values:
        return checkpoint_obj, {}, False

    raise TypeError("Unsupported checkpoint content. Cannot locate state_dict.")


def collect_binary_weight_names() -> set[str]:
    """从模型定义中自动收集“需要二值打包”的权重名。"""
    model = MNISTBNN()
    binary_weight_names: set[str] = set()

    for module_name, module in model.named_modules():
        if isinstance(module, (BinaryConv2d, BinaryLinear)):
            binary_weight_names.add(f"{module_name}.weight")

    return binary_weight_names


def binarize_weight(weight: torch.Tensor, threshold: float) -> torch.Tensor:
    """将浮点权重映射到 int8 的 {-1, +1}。"""
    return torch.where(
        weight >= threshold,
        torch.ones_like(weight, dtype=torch.int8),
        -torch.ones_like(weight, dtype=torch.int8),
    )


def pack_binary_tensor(binary_weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    """将 {-1, +1} 二值张量按 bit 压缩为 uint8。

    存储约定：
    - +1 映射为 bit 1
    - -1 映射为 bit 0
    - 每 8 个权重打包为 1 个字节
    - bit 顺序采用 little-endian（第 0 个元素放在最低位）

    Returns:
        packed: uint8 打包结果
        valid_numel: 原始二值权重元素总数（用于解包时去除补齐位）
    """
    if binary_weight.dtype != torch.int8:
        raise TypeError("pack_binary_tensor expects int8 input.")

    flat = binary_weight.reshape(-1)
    valid_numel = int(flat.numel())

    if valid_numel == 0:
        return torch.empty(0, dtype=torch.uint8), 0

    # +1 -> 1, -1 -> 0
    bits = (flat > 0).to(torch.uint8)

    # 补齐到 8 的整数倍，方便后续整字节打包。
    pad = (8 - (valid_numel % 8)) % 8
    if pad > 0:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8)], dim=0)

    bits = bits.reshape(-1, 8)
    packed = torch.zeros(bits.size(0), dtype=torch.uint8)

    # little-endian: 列索引就是 bit 位偏移。
    for shift in range(8):
        packed |= bits[:, shift] << shift

    return packed.contiguous(), valid_numel


def build_binary_state_checkpoint(
    *,
    binary_state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    wrapped_source: bool,
) -> dict[str, Any] | dict[str, torch.Tensor]:
    """构建可直接用于 PyTorch 侧验证的二值化 checkpoint。"""
    if not wrapped_source:
        return binary_state_dict

    # 仅保留推理相关关键信息，不再携带优化器状态，减小体积。
    output_obj: dict[str, Any] = {
        "model_state_dict": binary_state_dict,
    }
    if "epoch" in metadata:
        output_obj["epoch"] = metadata["epoch"]
    if "val_acc" in metadata:
        output_obj["val_acc"] = metadata["val_acc"]
    if "args" in metadata:
        output_obj["args"] = metadata["args"]

    output_obj["pack_note"] = "Binary layers are converted to {-1, +1}."
    return output_obj


def main() -> None:
    """程序入口：执行 BNN 最终打包。"""
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = resolve_path(project_root, args.checkpoint)
    output_path = resolve_path(project_root, args.output)
    binary_state_output_path = resolve_path(project_root, args.binary_state_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_state_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint_path}")

    print(f"Project root: {project_root}")
    print(f"Source checkpoint: {checkpoint_path}")
    print(f"Binary package output: {output_path}")
    if args.save_binary_state:
        print(f"Binary state output: {binary_state_output_path}")

    source_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict, metadata, wrapped_source = extract_state_dict(source_obj)

    binary_weight_names = collect_binary_weight_names()
    missing_binary_weights = sorted(name for name in binary_weight_names if name not in state_dict)

    if missing_binary_weights and args.strict:
        missing_lines = "\n".join(f"  - {name}" for name in missing_binary_weights)
        raise KeyError(
            "Checkpoint is missing required binary-layer weights:\n"
            f"{missing_lines}"
        )

    if missing_binary_weights and not args.strict:
        print("[Warning] Missing some expected binary-layer weights (strict=false):")
        for name in missing_binary_weights:
            print(f"  - {name}")

    packed_params: dict[str, Any] = {}
    binary_state_dict: dict[str, torch.Tensor] = {}

    source_total_bytes = 0
    packed_total_bytes = 0
    binary_param_count = 0

    print("\nPacking state_dict...")
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        tensor_cpu = tensor.detach().cpu().contiguous()
        source_total_bytes += tensor_cpu.numel() * tensor_cpu.element_size()

        # 命中二值层权重：执行阈值二值化 + bit 打包。
        if name in binary_weight_names:
            if not tensor_cpu.is_floating_point():
                raise TypeError(f"Binary layer weight must be floating tensor: {name}")

            binary_int8 = binarize_weight(tensor_cpu.to(torch.float32), threshold=args.threshold)
            packed_weight, valid_numel = pack_binary_tensor(binary_int8)

            packed_params[name] = {
                "is_binary_packed": True,
                "shape": list(tensor_cpu.shape),
                "dtype": "int1_as_packed_uint8",
                "packed_weight": packed_weight,
                "valid_numel": valid_numel,
                "positive_bit": 1,
                "negative_bit": 0,
                "bit_order": "little-endian",
                "threshold": args.threshold,
            }

            # 导出 PyTorch 兼容权重：仍保留同形状，但值为 -1/+1。
            binary_state_dict[name] = binary_int8.to(dtype=tensor_cpu.dtype)

            packed_total_bytes += packed_weight.numel()
            binary_param_count += int(tensor_cpu.numel())
        else:
            # 非二值层参数保持原始精度，确保模型行为一致。
            packed_params[name] = {
                "is_binary_packed": False,
                "shape": list(tensor_cpu.shape),
                "dtype": str(tensor_cpu.dtype),
                "tensor": tensor_cpu,
            }
            binary_state_dict[name] = tensor_cpu.clone()
            packed_total_bytes += tensor_cpu.numel() * tensor_cpu.element_size()

    compression_ratio = (source_total_bytes / packed_total_bytes) if packed_total_bytes > 0 else 0.0

    pack_obj = {
        "format": "mnist_bnn_binary_pack_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(checkpoint_path),
        "binary_weight_names": sorted(binary_weight_names),
        "state_dict_packed": packed_params,
        "stats": {
            "source_total_bytes": int(source_total_bytes),
            "packed_total_bytes": int(packed_total_bytes),
            "compression_ratio": float(compression_ratio),
            "binary_param_count": int(binary_param_count),
        },
        "training_summary": {
            "epoch": metadata.get("epoch"),
            "val_acc": metadata.get("val_acc"),
            "args": metadata.get("args"),
        },
    }

    torch.save(pack_obj, output_path)
    print("\nPackaging finished.")
    print(f"Saved binary package: {output_path}")
    print(f"Source bytes: {source_total_bytes}")
    print(f"Packed bytes: {packed_total_bytes}")
    print(f"Compression ratio: {compression_ratio:.4f}x")

    if args.save_binary_state:
        binary_state_obj = build_binary_state_checkpoint(
            binary_state_dict=binary_state_dict,
            metadata=metadata,
            wrapped_source=wrapped_source,
        )
        torch.save(binary_state_obj, binary_state_output_path)
        print(f"Saved binary state checkpoint: {binary_state_output_path}")


if __name__ == "__main__":
    main()
