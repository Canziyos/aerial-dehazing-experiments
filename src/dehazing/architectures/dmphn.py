from __future__ import annotations

import torch
from torch import nn

from src.dehazing.modules.encoder import Encoder
from src.dehazing.modules.decoder import Decoder


class DMPHN124(nn.Module):
    """
    Deep Multi-Patch Hierarchical Network: 1-2-4 variant.

    Shape:
        input:  [B, 3, H, W]
        output: [B, 3, H, W]

    Constraint:
        H and W should be divisible by 8.
    """
    def __init__(self) -> None:
        super().__init__()

        self.encoder_lv1 = Encoder()
        self.encoder_lv2 = Encoder()
        self.encoder_lv3 = Encoder()

        self.decoder_lv1 = Decoder()
        self.decoder_lv2 = Decoder()
        self.decoder_lv3 = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        _, _, height, width = x.shape

        half_h = height// 2
        half_w = width // 2

        top_half = x[:, :, :half_h, :]
        bottom_half = x[:, :, half_h:, :]

        top_left = top_half[:, :, :, :half_w]
        top_right = top_half[:, :, :, half_w:]
        bottom_left = bottom_half[:, :, :, :half_w]
        bottom_right = bottom_half[:, :, :, half_w:]

        f3_top_left = self.encoder_lv3(top_left)
        f3_top_right = self.encoder_lv3(top_right)
        f3_bottom_left = self.encoder_lv3(bottom_left)
        f3_bottom_right = self.encoder_lv3(bottom_right)

        f3_top = torch.cat([f3_top_left, f3_top_right], dim=3)
        f3_bottom = torch.cat([f3_bottom_left, f3_bottom_right], dim=3)
        f3 = torch.cat([f3_top, f3_bottom], dim=2)

        residual_top = self.decoder_lv3(f3_top)
        residual_bottom = self.decoder_lv3(f3_bottom)

        corrected_top = residual_top + top_half
        corrected_bottom = bottom_half + residual_bottom

        f2_top = self.encoder_lv2(corrected_top)
        f2_bottom = self.encoder_lv2(corrected_bottom)

        f2 = torch.cat([f2_top, f2_bottom], dim=2)
        f2 = f2 + f3

        residual_full = self.decoder_lv2(f2)
        corrected_full = x + residual_full

        f1 = self.encoder_lv1(corrected_full)
        f1 = f1 + f2

        output = self.decoder_lv1(f1)
        return output
    
    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected 4D tensor [B, 3, H, W], got shape {tuple(x.shape)}")

        if x.shape[1] != 3:
            raise ValueError(f"expected 3 RGB channels, got {x.shape[1]}")

        height = x.shape[2]
        width = x.shape[3]

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"DMPHN124 expects H and W divisible by 8, got H={height}, W={width}"
            )


