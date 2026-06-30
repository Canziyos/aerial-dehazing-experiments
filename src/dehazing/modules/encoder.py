from __future__ import annotations

import torch
from torch import nn

from .blocks import ResidualBlock


class Encoder(nn.Module):
    """
    Encoder used by DMPHN/DMSHN.

    Shape:
        input:  [B, 3, H, W]
        output: [B, 128, H/4, W/4]
    """

    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualBlock(32),
            ResidualBlock(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ResidualBlock(64),
            ResidualBlock(64),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ResidualBlock(128),
            ResidualBlock(128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)