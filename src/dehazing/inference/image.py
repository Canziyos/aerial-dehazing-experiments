from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch import nn
import torchvision.transforms.functional as TF


def load_image_for_model(
    image_path: str | Path,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Load RGB image and convert it to model input.

    Returns:
        Tensor shape: [1, 3, H, W]
        Tensor range: approximately [-0.5, 0.5]
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image).unsqueeze(0)
    tensor = tensor.to(device)

    return tensor - 0.5


def save_model_output(
    output: torch.Tensor,
    output_path: str | Path,
) -> None:
    """
    Save model output as RGB image.

    Input:
        output shape: [1, 3, H, W] or [3, H, W]
        output range: approximately [-0.5, 0.5]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output.ndim == 4:
        if output.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {output.shape[0]}")
        output = output.squeeze(0)

    if output.ndim != 3:
        raise ValueError(f"expected output shape [3, H, W], got {tuple(output.shape)}")

    image_tensor = (output.detach().cpu() + 0.5).clamp(0.0, 1.0)
    image = TF.to_pil_image(image_tensor)
    image.save(output_path)


def run_single_image_inference(
    model: nn.Module,
    image_path: str | Path,
    output_path: str | Path,
    device: torch.device | str = "cpu",
) -> None:
    """
    Run single-image dehazing inference and save the result.
    """
    model = model.to(device)
    model.eval()

    x = load_image_for_model(image_path=image_path, device=device)

    with torch.no_grad():
        output = model(x)

    save_model_output(output=output, output_path=output_path)