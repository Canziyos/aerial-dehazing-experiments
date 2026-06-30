from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import VGG16_Weights, vgg16


class DehazingLoss(nn.Module):
    """
    Combined dehazing loss.

    Returns:
        total_loss
        reconstruction_loss
        perceptual_loss
        tv_loss
    """

    def __init__(
        self,
        perceptual_weight: float = 0.006,
        tv_weight: float = 2e-8,
        use_perceptual: bool = True,
    ) -> None:
        super().__init__()

        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight
        self.use_perceptual = use_perceptual

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        if use_perceptual:
            weights = VGG16_Weights.IMAGENET1K_V1
            vgg = vgg16(weights=weights).features[:31]
            vgg.eval()

            for parameter in vgg.parameters():
                parameter.requires_grad = False

            self.vgg_features = vgg
        else:
            self.vgg_features = None

    def train(self, mode: bool = True) -> "DehazingLoss":
        """
        Keep PyTorch train/eval behavior, but force frozen VGG features
        to remain in eval mode.
        """
        super().train(mode)

        if self.vgg_features is not None:
            self.vgg_features.eval()

        return self
    
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction_loss = self._reconstruction_loss(output, target)
        perceptual_loss = self._perceptual_loss(output, target)
        tv_loss = self._tv_loss(output)

        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.tv_weight * tv_loss
        )

        return total_loss, reconstruction_loss, perceptual_loss, tv_loss

    def _reconstruction_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        l1 = self.l1_loss(output, target)
        mse = self.mse_loss(output, target)
        return 0.6 * l1 + 0.4 * mse

    def _perceptual_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.vgg_features is None:
            return output.new_tensor(0.0)

        output_vgg = self._to_vgg_input(output)
        target_vgg = self._to_vgg_input(target)

        output_features = self.vgg_features(output_vgg)
        target_features = self.vgg_features(target_vgg)

        return F.mse_loss(output_features, target_features)

    @staticmethod
    def _tv_loss(output: torch.Tensor) -> torch.Tensor:
        horizontal_difference = output[:, :, 1:, :] - output[:, :, :-1, :]
        vertical_difference = output[:, :, :, 1:] - output[:, :, :, :-1]

        return (
            torch.mean(horizontal_difference * horizontal_difference)
            + torch.mean(vertical_difference * vertical_difference)
        )

    @staticmethod
    def _to_vgg_input(x: torch.Tensor) -> torch.Tensor:
        """
        Convert model-domain tensors from roughly [-0.5, 0.5]
        back to VGG-compatible image tensors in [0, 1].
        """
        return (x + 0.5).clamp(0.0, 1.0)