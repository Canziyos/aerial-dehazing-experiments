from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.dehazing.modules.encoder import Encoder
from src.dehazing.modules.decoder import Decoder


class DMSHN124(nn.Module):
    """
    Deep Multi-Scale Hierarchical Network: 1-2-4 variant.

    Unlike DMPHN, this model does not split the image into patches.
    It processes the same image at full, half, and quarter resolution.

    Shape:
        input:  [B, 3, H, W]
        output: [B, 3, H, W]
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

        x_lv1 = x
        x_lv2 = F.interpolate(
            x_lv1,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
        )
        x_lv3 = F.interpolate(
            x_lv2,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
        )

        f3 = self.encoder_lv3(x_lv3)
        r3 = self.decoder_lv3(f3)

        # Robust size matching.
        # This avoids shape mismatch when H/W are not perfectly divisible
        # through every downsample/upsample step.
        r3 = F.interpolate(
            r3,
            size=x_lv3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        r3_up = F.interpolate(
            r3,
            size=x_lv2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        f2 = self.encoder_lv2(x_lv2 + r3_up)

        f3_up = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        r2 = self.decoder_lv2(f2 + f3_up)

        r2 = F.interpolate(
            r2,
            size=x_lv2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        r2_up = F.interpolate(
            r2,
            size=x_lv1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        f1 = self.encoder_lv1(x_lv1 + r2_up)

        f2_up = F.interpolate(
            f2,
            size=f1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        output = self.decoder_lv1(f1 + f2_up)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(
                output,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return output

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected 4D tensor [B, 3, H, W], got shape {tuple(x.shape)}")

        if x.shape[1] != 3:
            raise ValueError(f"expected 3 RGB channels, got {x.shape[1]}")

        height = x.shape[2]
        width = x.shape[3]

        if height < 16 or width < 16:
            raise ValueError(f"expected H and W >= 16, got H={height}, W={width}")