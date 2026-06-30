from __future__ import annotations

import math

import torch
from torch import nn


def compute_psnr(
    output: torch.Tensor,
    target: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """
    Compute PSNR between output and target tensors.

    Expected tensor range:
        [0, 1]

    Supported shapes:
        [3, H, W] or [B, 3, H, W]
    """
    if output.shape != target.shape:
        raise ValueError(
            f"output and target must have same shape, "
            f"got {tuple(output.shape)} and {tuple(target.shape)}"
        )

    mse = torch.mean((output - target) ** 2).item()

    if mse == 0.0:
        return math.inf

    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    """
    parameters = model.parameters()

    if trainable_only:
        return sum(p.numel() for p in parameters if p.requires_grad)

    return sum(p.numel() for p in parameters)