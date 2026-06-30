from __future__ import annotations

import torch
from torch import nn

from .blocks import ResidualBlock


class Decoder(nn.Module):
    """
    Decoder used by DMPHN/DMSHN.

    Shape:
        input:  [B, 128, H/4, W/4]
        output: [B, 3, H, W]
    """

    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            ResidualBlock(64),
            ResidualBlock(64),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            ResidualBlock(32),
            ResidualBlock(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)