"""Offline int4 quantization tool for the MNIST CNN checkpoint.

This script converts the trained fp32 checkpoint into a compact int4 package.
The produced file stores packed 4-bit weights and per-tensor/per-channel scales,
which can be consumed by custom inference runtimes or later dequantized.

Default IO:
- input checkpoint:  models/mnist_cnn/best_mnist_cnn.pt
- output package:    models/mnist_cnn/best_mnist_cnn_int4.pt

Usage examples:
    python src/quantize_mnist_cnn_int4.py
    python src/quantize_mnist_cnn_int4.py --per-channel false
    python src/quantize_mnist_cnn_int4.py --output models/mnist_cnn/custom_int4.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


INT4_MIN = -8
INT4_MAX = 7


def parse_bool_flag(value: str) -> bool:
    """Parse a flexible boolean flag from CLI text.

    Accepts: true/false, 1/0, yes/no, y/n (case-insensitive).
    """
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    """Build command-line arguments for int4 quantization."""
    parser = argparse.ArgumentParser(description="Quantize MNIST CNN checkpoint to int4")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mnist_cnn/best_mnist_cnn.pt",
        help="Path to source fp32 checkpoint (absolute or relative to project root).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mnist_cnn/best_mnist_cnn_int4.pt",
        help="Path to save the int4 package.",
    )
    parser.add_argument(
        "--per-channel",
        type=parse_bool_flag,
        default=True,
        help=(
            "Whether to use per-channel quantization for weight tensors (recommended). "
            "Accepted values: true/false"
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Small epsilon to avoid division by zero when tensor values are all zeros.",
    )
    parser.add_argument(
        "--save-dequantized",
        type=parse_bool_flag,
        default=True,
        help=(
            "Whether to also save a dequantized fp32-sim checkpoint for quick accuracy checks. "
            "Accepted values: true/false"
        ),
    )
    return parser.parse_args()


def resolve_path(project_root: Path, user_path: str) -> Path:
    """Resolve path from user input to absolute path."""
    path = Path(user_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def pack_signed_int4(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack signed int4 values into uint8 bytes.

    Input tensor must be int8 in range [-8, 7]. Two int4 values are packed into
    one byte: low nibble stores element[0], high nibble stores element[1].

    Returns:
        packed: uint8 vector of packed nibbles
        valid_numel: original number of int4 values before optional padding
    """
    if values.dtype != torch.int8:
        raise TypeError("pack_signed_int4 expects int8 input.")

    flat = values.reshape(-1)
    valid_numel = int(flat.numel())

    if valid_numel == 0:
        return torch.empty(0, dtype=torch.uint8), 0

    # Move signed range [-8, 7] to unsigned nibble range [0, 15].
    nibble = (flat.to(torch.int16) + 8).to(torch.uint8)

    # Pad one element when odd length so pairing into bytes is safe.
    if nibble.numel() % 2 != 0:
        nibble = torch.cat([nibble, torch.zeros(1, dtype=torch.uint8)])

    low = nibble[0::2]
    high = nibble[1::2] << 4
    packed = (low | high).contiguous()
    return packed, valid_numel


def quantize_tensor_symmetric_int4(
    tensor: torch.Tensor,
    *,
    per_channel: bool,
    channel_axis: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a float tensor into signed int4 with symmetric scaling.

    Quantization formula:
        q = clamp(round(x / s), -8, 7)
        x_hat = q * s

    where s is computed from max absolute value.
    """
    if not tensor.is_floating_point():
        raise TypeError("quantize_tensor_symmetric_int4 expects floating tensor input.")

    tensor_fp32 = tensor.detach().to(torch.float32)

    if per_channel:
        if channel_axis < 0 or channel_axis >= tensor_fp32.ndim:
            raise ValueError("Invalid channel_axis for per-channel quantization.")

        reduce_dims = tuple(dim for dim in range(tensor_fp32.ndim) if dim != channel_axis)
        max_abs = tensor_fp32.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        max_abs = tensor_fp32.abs().amax().reshape([1] * tensor_fp32.ndim)

    scale = (max_abs / float(INT4_MAX)).clamp(min=eps)
    q = torch.round(tensor_fp32 / scale).clamp(INT4_MIN, INT4_MAX).to(torch.int8)
    dequant = q.to(torch.float32) * scale

    # Remove broadcast dimensions in scale so it is compact on disk.
    if per_channel:
        scale_to_store = scale.squeeze()
    else:
        scale_to_store = scale.reshape(1)

    return q, dequant, scale_to_store


def is_weight_like(name: str, tensor: torch.Tensor) -> bool:
    """Heuristic to detect conv/linear weight tensors.

    We only enable per-channel mode on typical multi-dimensional weight tensors.
    Bias tensors and 1D parameters remain per-tensor quantized.
    """
    return name.endswith(".weight") and tensor.ndim >= 2


def extract_state_dict(checkpoint_obj: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any], bool]:
    """Extract model state_dict and metadata from multiple checkpoint formats.

    Returns:
        state_dict: parameter tensors
        metadata: non-weight fields we may want to preserve
        wrapped: True when source format uses key "model_state_dict"
    """
    if not isinstance(checkpoint_obj, dict):
        raise TypeError("Checkpoint format is unsupported: expected dict.")

    if "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
        metadata = {k: v for k, v in checkpoint_obj.items() if k != "model_state_dict"}
        return state_dict, metadata, True

    # Fallback: treat full dict as raw state_dict when all values are tensors.
    all_tensor_values = all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values())
    if all_tensor_values:
        return checkpoint_obj, {}, False

    raise TypeError("Unsupported checkpoint content. Cannot locate state_dict.")


def main() -> None:
    """Program entry point for int4 quantization pipeline."""
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = resolve_path(project_root, args.checkpoint)
    output_path = resolve_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint_path}")

    print(f"Project root: {project_root}")
    print(f"Source checkpoint: {checkpoint_path}")
    print(f"Output int4 package: {output_path}")
    print(f"Per-channel for weights: {args.per_channel}")

    source_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict, metadata, wrapped_checkpoint = extract_state_dict(source_obj)

    int4_params: dict[str, Any] = {}
    dequantized_state_dict: dict[str, torch.Tensor] = {}

    total_fp32_bytes = 0
    total_packed_bytes = 0

    print("\nQuantizing tensors to signed int4...")
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        # Non-floating tensors are copied as-is because int4 quantization is only
        # defined here for floating-point parameters.
        if not tensor.is_floating_point():
            dequantized_state_dict[name] = tensor.clone()
            int4_params[name] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "is_quantized": False,
                "raw_tensor": tensor.clone(),
            }
            continue

        use_per_channel = args.per_channel and is_weight_like(name, tensor)
        channel_axis = 0 if use_per_channel else -1

        q_int4, dequant, scale = quantize_tensor_symmetric_int4(
            tensor,
            per_channel=use_per_channel,
            channel_axis=channel_axis,
            eps=args.eps,
        )

        packed, valid_numel = pack_signed_int4(q_int4)
        total_fp32_bytes += int(tensor.numel()) * 4
        total_packed_bytes += int(packed.numel())

        mse = torch.mean((tensor.detach().to(torch.float32) - dequant) ** 2).item()
        max_abs_err = torch.max(torch.abs(tensor.detach().to(torch.float32) - dequant)).item()

        dequantized_state_dict[name] = dequant.to(dtype=tensor.dtype)
        int4_params[name] = {
            "is_quantized": True,
            "shape": list(tensor.shape),
            "original_dtype": str(tensor.dtype),
            "quant_dtype": "int4_signed_packed_uint8",
            "per_channel": use_per_channel,
            "channel_axis": channel_axis,
            "scale": scale.cpu(),
            "packed": packed.cpu(),
            "valid_numel": valid_numel,
            "mse": mse,
            "max_abs_err": max_abs_err,
        }

        print(
            f"  {name}: per_channel={use_per_channel}, "
            f"shape={tuple(tensor.shape)}, mse={mse:.8f}, max_abs_err={max_abs_err:.8f}"
        )

    compression_ratio = 0.0
    if total_packed_bytes > 0:
        compression_ratio = total_fp32_bytes / float(total_packed_bytes)

    package = {
        "format_version": "mnist_cnn_int4_v1",
        "source_checkpoint": str(checkpoint_path),
        "bit_width": 4,
        "int4_range": [INT4_MIN, INT4_MAX],
        "per_channel_weights": bool(args.per_channel),
        "int4_params": int4_params,
        "metadata": metadata,
        "stats": {
            "fp32_param_bytes": total_fp32_bytes,
            "int4_packed_bytes": total_packed_bytes,
            "compression_ratio_vs_fp32": compression_ratio,
        },
    }
    torch.save(package, output_path)

    print("\nSaved int4 package successfully.")
    print(f"  fp32 bytes: {total_fp32_bytes}")
    print(f"  int4 packed bytes: {total_packed_bytes}")
    print(f"  compression ratio (fp32 / int4): {compression_ratio:.4f}x")

    if args.save_dequantized:
        dequant_path = output_path.with_name(output_path.stem + "_dequantized_fp32.pt")
        if wrapped_checkpoint:
            dequant_obj = dict(metadata)
            dequant_obj["model_state_dict"] = dequantized_state_dict
        else:
            dequant_obj = dequantized_state_dict

        torch.save(dequant_obj, dequant_path)
        print(f"Saved dequantized fp32-sim checkpoint: {dequant_path}")


if __name__ == "__main__":
    main()