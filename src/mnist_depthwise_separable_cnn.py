from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)


class MNISTDepthwiseSeparableCNN(nn.Module):
    def __init__(self, channels: Sequence[int] = (16, 64, 128)) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("channels must contain exactly three values, for example: 16,64,128")

        c1, c2, c3 = [int(value) for value in channels]
        self.block1 = DepthwiseSeparableConv(1, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.block2 = DepthwiseSeparableConv(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.block3 = DepthwiseSeparableConv(c2, c3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(c3, 10)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.avg_pool(x)
        return self.flatten(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)


def parse_channels(raw: str) -> tuple[int, int, int]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("--channels must be three positive integers, for example: 16,64,128")
    return values
