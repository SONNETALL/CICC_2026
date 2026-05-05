"""Quick uint8 optical inference on a small MNIST subset.

Usage:
    python test/quick_int8_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def quantize_activation_uint8(x: torch.Tensor, clip_max: float) -> tuple[np.ndarray, float]:
    """Unsigned uint8 activation quantization for simulator input."""
    x_cpu = x.detach().to(torch.float32).cpu()
    x_clipped = x_cpu.clamp(0.0, clip_max)
    scale = clip_max / 255.0
    q = (x_clipped / scale).round().clamp(0, 255).to(torch.int32)
    return q.numpy().astype(np.int32), float(scale)


def quantize_weight_int8_symmetric(w: torch.Tensor) -> tuple[np.ndarray, float]:
    """Symmetric int8 weight: q ∈ [-128, 127], scale = max_abs / 127."""
    w_cpu = w.detach().to(torch.float32).cpu()
    max_abs = w_cpu.abs().max().item()
    if max_abs == 0.0:
        return np.zeros(w_cpu.shape, dtype=np.int32), 1.0
    scale = max_abs / 127.0
    q = (w_cpu / scale).round().clamp(-128, 127).to(torch.int32)
    return q.numpy(), float(scale)


def run_uint8_matmul(optical_model, a: torch.Tensor, w: torch.Tensor, act_clip: float) -> torch.Tensor:
    """Execute uint8-input optical matmul with int8 weights, then dequantize."""
    a_q, a_scale = quantize_activation_uint8(a, act_clip)
    w_q, w_scale = quantize_weight_int8_symmetric(w)

    # input shape: [B, M, K], weight shape: [K, N]
    bsz = a_q.shape[0]
    w_broadcast = np.broadcast_to(w_q[None, :, :], (bsz, w_q.shape[0], w_q.shape[1]))

    result = optical_model(a_q.astype(np.int32), w_broadcast.astype(np.int32), inputType="uint8")
    arr = result.numpy() if hasattr(result, "numpy") else np.asarray(result)
    return torch.from_numpy(arr.astype(np.float32) * (a_scale * w_scale))


class MLP80x20(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 80),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(80, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def compare_int4_vs_uint8(
    model: nn.Module,
    loader: DataLoader,
    optical_model,
    num_samples: int = 500,
) -> None:
    """Compare int4 vs uint8 optical accuracy on a subset."""
    criterion = nn.CrossEntropyLoss()

    # --- float32 baseline ---
    model.eval()
    correct_fp = 0
    total = 0
    for images, labels in loader:
        if total >= num_samples:
            break
        logits = model(images * 255.0)
        preds = logits.argmax(dim=1)
        correct_fp += preds.eq(labels).sum().item()
        total += labels.size(0)
    acc_fp = correct_fp / total

    # --- int4 optical (matching test script exactly) ---
    correct_i4 = 0
    total = 0
    clk_i4 = 0.0
    act_clip = 4.0
    for images, labels in loader:
        if total >= num_samples:
            break
        bsz = labels.size(0)

        x = torch.flatten(images * 255.0, 1)
        t0 = time.perf_counter()

        # fc1 weight: [80, 784] → transpose → [784, 80] (K=784, N=80)
        fc1_w = model.net[1].weight  # [80, 784]
        fc1_w_kn = fc1_w.transpose(0, 1).contiguous()  # [784, 80]

        # quantize activation (int4 via uint4 offset)
        a_scale = act_clip / 7.0
        a_q_signed = (x.clamp(-act_clip, act_clip) / a_scale).round().clamp(-8, 7).to(torch.int32)
        a_uint = (a_q_signed + 8).numpy().astype(np.int32)[:, None, :]  # [B, 1, 784]

        # quantize weight (int4 symmetric on transposed weight)
        w_max = fc1_w_kn.abs().max().item()
        w_scale = w_max / 7.0
        w_q = (fc1_w_kn / w_scale).round().clamp(-8, 7).to(torch.int32).numpy()  # [784, 80]

        # call simulator: input [B, M, K]=[1,1,784], weight [B, K, N]=[1,784,80]
        w_brd = np.broadcast_to(w_q[None], (bsz, w_q.shape[0], w_q.shape[1]))
        r = optical_model(a_uint, w_brd.astype(np.int32), inputType="uint4")
        out = np.asarray(r).astype(np.float32)  # [B, 1, 80]

        # undo zero_point offset
        w_sum = np.sum(w_q.astype(np.float32), axis=0, keepdims=True)  # [1, 80]
        out = out - 8.0 * w_sum[None, :, :]
        out = out * (a_scale * w_scale)
        x = torch.from_numpy(out.squeeze(1)) + model.net[1].bias  # [80]
        clk_i4 += time.perf_counter() - t0

        x = F.relu(x)
        x = model.net[4](x)
        x = F.relu(x)
        logits = model.net[7](x)

        preds = logits.argmax(dim=1)
        correct_i4 += preds.eq(labels).sum().item()
        total += bsz
    acc_i4 = correct_i4 / total

    # --- uint8 optical ---
    correct_u8 = 0
    total = 0
    clk_u8 = 0.0
    act_clip = 255.0
    for images, labels in loader:
        if total >= num_samples:
            break
        bsz = labels.size(0)

        x = torch.flatten(images * 255.0, 1)
        t0 = time.perf_counter()

        # fc1 weight: transpose to [K, N]
        fc1_w = model.net[1].weight
        fc1_w_kn = fc1_w.transpose(0, 1).contiguous()  # [784, 80]

        # quantize activation uint8
        a_q, a_s = quantize_activation_uint8(x.unsqueeze(1), act_clip)  # [B, 1, 784]
        # quantize weight int8 (on transposed weight)
        w_q, w_s = quantize_weight_int8_symmetric(fc1_w_kn)  # [784, 80]

        # call simulator with uint8
        w_brd = np.broadcast_to(w_q[None], (bsz, w_q.shape[0], w_q.shape[1]))
        r = optical_model(a_q.astype(np.int32), w_brd.astype(np.int32), inputType="uint8")
        out = np.asarray(r).astype(np.float32)  # [B, 1, 80]

        # no zero-point correction is needed for unsigned uint8 activations
        out = out * (a_s * w_s)

        x = torch.from_numpy(out.squeeze(1)) + model.net[1].bias
        clk_u8 += time.perf_counter() - t0

        x = F.relu(x)
        x = model.net[4](x)
        x = F.relu(x)
        logits = model.net[7](x)

        preds = logits.argmax(dim=1)
        correct_u8 += preds.eq(labels).sum().item()
        total += bsz
    acc_u8 = correct_u8 / total

    print(f"  float32 (fc1-only reference):  {acc_fp:.4%}")
    print(f"  int4 optical fc1:              {acc_i4:.4%}  (fc1 time: {clk_i4:.2f}s)")
    print(f"  uint8 optical fc1:             {acc_u8:.4%}  (fc1 time: {clk_u8:.2f}s)")
    print(f"  int4 → uint8 delta:            {acc_u8 - acc_i4:+.4%}")


def main():
    from osimulator.api import load_gazelle_model

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root=str(data_root), train=False, transform=transform, download=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    checkpoint = torch.load(
        project_root / "models/mnist_mlp_80_20/best_mnist_mlp_80_20.pt",
        map_location="cpu",
    )
    model = MLP80x20()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint val_acc: {checkpoint.get('val_acc', 'N/A')}")
    print(f"QAT config: {checkpoint.get('qat_config', 'N/A')}")
    print()

    print("Loading Gazelle model...")
    optical_model = load_gazelle_model()

    print(f"Testing on {500} samples (batch_size=1, one at a time)...")
    compare_int4_vs_uint8(model, loader, optical_model, num_samples=500)


if __name__ == "__main__":
    main()
