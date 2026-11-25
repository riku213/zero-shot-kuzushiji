from __future__ import annotations

import torch
from torch import nn


class Stage1SkeletonClassifier(nn.Module):
    """CNN classifier over skeletons observed in the training split."""

    def __init__(
        self,
        num_skeletons: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_skeletons < 1:
            raise ValueError("Stage 1 requires at least one training skeleton class.")
        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_channels, num_skeletons))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images))