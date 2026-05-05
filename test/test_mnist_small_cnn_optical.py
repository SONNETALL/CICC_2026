"""Evaluate Small MNIST CNN with official optical matmul calculating method."""

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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_mnist_small_cnn import SmallMNISTCNN
from op_computation_analysis import calculate_linear_macs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Small MNIST CNN on test set with optical inference")
    parser.add_argument("--checkpoint", type=str, default="models/mnist_small_cnn/best_mnist_small_cnn.pt")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-type", type=str, default="uint4", choices=["uint4", "int4"])
    parser.add_argument("--act-clip-max", type=float, default=6.0)
    # Allow user to specify which layers are computed optically, e.g. "conv2,fc1,fc2"
    parser.add_argument(
        "--optical-layers",
        type=str,
        default="conv2,fc1,fc2",
        help="Comma-separated list of layers mapped to optical matmul.",
    )
    parser.add_argument(
        "--max-samples", 
        type=int,
        default=1000,
        help="Maximum number of test samples to evaluate. Default is 1000.",
    )
    return parser.parse_args()


def load_optical_model() -> Any:
    try:
        from osimulator.api import load_gazelle_model
    except ImportError as exc:
        raise ImportError(
            "Failed to import osimulator.api.load_gazelle_model. "
            "Please ensure osimulator is installed and available in this environment."
        ) from exc
    return load_gazelle_model()


def build_test_loader(data_root: Path, batch_size: int, num_workers: int, max_samples: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(root=str(data_root), train=False, transform=transform, download=False)
    if len(dataset) != 10_000:
        print(f"Warning: Expected 10,000 MNIST test samples, but found {len(dataset)}.")
        
    if max_samples > 0 and max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def quantize_activation_to_uint4(x: torch.Tensor, clip_max: float) -> tuple[np.ndarray, float]:
    x_cpu = x.detach().to(torch.float32).cpu()
    x_clip = torch.clamp(x_cpu, min=0.0, max=clip_max)
    scale = clip_max / 15.0
    q = torch.round(x_clip / scale).clamp(0, 15).to(torch.int32)
    return q.numpy(), float(scale)


def quantize_activation_to_int4(x: torch.Tensor, clip_max: float) -> tuple[np.ndarray, float]:
    x_cpu = x.detach().to(torch.float32).cpu()
    # For int4, range is typically [-clip_max, clip_max] mapped to [-8, 7]
    x_clip = torch.clamp(x_cpu, min=-clip_max, max=clip_max)
    scale = clip_max / 7.0
    q = torch.round(x_clip / scale).clamp(-8, 7).to(torch.int32)
    return q.numpy(), float(scale)


def quantize_weight_to_int4_symmetric(w: torch.Tensor) -> tuple[np.ndarray, float]:
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
    bsz = a_bmk.size(0)
    
    if input_type == "int4":
        a_q, a_scale = quantize_activation_to_int4(a_bmk, clip_max=act_clip_max)
    else:
        a_q, a_scale = quantize_activation_to_uint4(a_bmk, clip_max=act_clip_max)
        
    w_q, w_scale = quantize_weight_to_int4_symmetric(w_kn)

    out_list = []
    for idx in range(bsz):
        input_tensors = a_q[idx : idx + 1].astype(np.int32, copy=False)
        wght_tensors = w_q[None, :, :].astype(np.int32, copy=False)
        result_model = optical_model(input_tensors, wght_tensors, inputType=input_type)
        if hasattr(result_model, "numpy"):
            out_np = result_model.numpy()
        else:
            out_np = np.asarray(result_model)
        out_list.append(out_np)

    out_concat = np.concatenate(out_list, axis=0)
    out_fp32 = out_concat.astype(np.float32) * (a_scale * w_scale)
    return torch.from_numpy(out_fp32)


def forward_with_optical_matmul(
    model: nn.Module,
    images: torch.Tensor,
    optical_model: Any,
    optical_layers_set: set[str],
    input_type: str,
    act_clip_max: float,
) -> torch.Tensor:
    # -- Conv1 --
    if "conv1" in optical_layers_set:
        x_unfold = F.unfold(images, kernel_size=3, padding=1, stride=1)
        x_bmk = x_unfold.transpose(1, 2).contiguous()
        conv1_w_kn = model.conv1.weight.view(model.conv1.out_channels, -1).transpose(0, 1).contiguous()

        conv1_out_bmn = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x_bmk,
            w_kn=conv1_w_kn,
            input_type=input_type,
            act_clip_max=act_clip_max,
        )
        bsz = images.size(0)
        conv1_out = conv1_out_bmn.transpose(1, 2).contiguous().view(bsz, model.conv1.out_channels, 28, 28)
        conv1_out = conv1_out + model.conv1.bias.detach().cpu().view(1, -1, 1, 1)
        x = conv1_out
    else:
        x = F.conv2d(images, model.conv1.weight, model.conv1.bias, stride=1, padding=1)

    x = torch.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # -- Conv2 --
    if "conv2" in optical_layers_set:
        x_unfold = F.unfold(x, kernel_size=3, padding=1, stride=1)
        x_bmk = x_unfold.transpose(1, 2).contiguous()
        conv2_w_kn = model.conv2.weight.view(model.conv2.out_channels, -1).transpose(0, 1).contiguous()

        conv2_out_bmn = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x_bmk,
            w_kn=conv2_w_kn,
            input_type=input_type,
            act_clip_max=act_clip_max,
        )
        bsz = images.size(0)
        conv2_out = conv2_out_bmn.transpose(1, 2).contiguous().view(bsz, model.conv2.out_channels, 14, 14)
        conv2_out = conv2_out + model.conv2.bias.detach().cpu().view(1, -1, 1, 1)
        x = conv2_out
    else:
        x = F.conv2d(x, model.conv2.weight, model.conv2.bias, stride=1, padding=1)

    x = torch.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # -- Flatten --
    x = torch.flatten(x, 1)

    # -- FC1 --
    if "fc1" in optical_layers_set:
        fc1_w_kn = model.fc1.weight.transpose(0, 1).contiguous()
        fc1_out = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x.unsqueeze(1),
            w_kn=fc1_w_kn,
            input_type=input_type,
            act_clip_max=act_clip_max,
        ).squeeze(1)
        fc1_out = fc1_out + model.fc1.bias.detach().cpu()
        x = fc1_out
    else:
        x = F.linear(x, model.fc1.weight, model.fc1.bias)

    x = torch.relu(x)

    # -- FC2 --
    if "fc2" in optical_layers_set:
        fc2_w_kn = model.fc2.weight.transpose(0, 1).contiguous()
        fc2_out = run_optical_matmul(
            optical_model=optical_model,
            a_bmk=x.unsqueeze(1),
            w_kn=fc2_w_kn,
            input_type=input_type,
            act_clip_max=act_clip_max,
        ).squeeze(1)
        logits = fc2_out + model.fc2.bias.detach().cpu()
    else:
        logits = F.linear(x, model.fc2.weight, model.fc2.bias)

    return logits


@torch.no_grad()
def evaluate() -> None:
    args = parse_args()
    
    # Process layers which should run optically
    optical_layers_set = {layer.strip() for layer in args.optical_layers.split(",") if layer.strip()}

    data_root = PROJECT_ROOT / args.data_root
    checkpoint_path = PROJECT_ROOT / args.checkpoint

    # Initialize model
    model = SmallMNISTCNN()
    
    if checkpoint_path.exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Instantiating with random weights.")

    model.eval()

    # Determine MACs using official computation analysis
    dummy_input = torch.randn(1, 1, 28, 28)
    total_macs_sample, macs_dict = calculate_linear_macs(model, dummy_input)
    
    optical_macs_sample = 0
    all_macs_sample = sum(macs_dict.values())
    
    for name, module in model.named_modules():
        if name in macs_dict and name in optical_layers_set:
            optical_macs_sample += macs_dict[name]

    ocr = (optical_macs_sample / all_macs_sample) * 100.0 if all_macs_sample > 0 else 0.0

    print("=" * 40)
    print("MACs calculation from `op_computation_analysis`:")
    print(f"  Total MACs per sample:   {all_macs_sample:,}")
    print(f"  Optical MACs per sample: {optical_macs_sample:,}")
    print(f"  Optical Compute Ratio:   {ocr:.2f}%")
    print("=" * 40)


    # Optical integration
    loader = build_test_loader(data_root, args.batch_size, args.num_workers, args.max_samples)
    optical_model = load_optical_model()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    print(f"Starting test evaluation over {len(loader.dataset)} samples...")
    start_time = time.time()

    for images, labels in loader:
        images = images.to("cpu", non_blocking=True)
        labels = labels.to("cpu", non_blocking=True)

        logits = forward_with_optical_matmul(
            model=model,
            images=images,
            optical_model=optical_model,
            optical_layers_set=optical_layers_set,
            input_type=args.input_type,
            act_clip_max=args.act_clip_max,
        )

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

    end_time = time.time()
    total_time = end_time - start_time

    avg_loss = total_loss / total_samples
    accuracy = (total_correct / total_samples)

    print("\nEvaluation results on MNIST test set:")
    print(f"  avg_loss: {avg_loss:.6f}")
    print(f"  top1_accuracy: {accuracy:.4%}")
    print(f"  total_inference_time: {total_time:.2f} seconds")


if __name__ == "__main__":
    evaluate()
