from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dehazing.architectures.factory import create_model, get_supported_model_names
from src.dehazing.training.checkpoints import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained dehazing checkpoint to ONNX."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=get_supported_model_names(),
        help="Model architecture name.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to checkpoint.pt or checkpoint_best.pt.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Input height used for ONNX export.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Input width used for ONNX export.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device. Use cpu unless you have a good reason not to.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )

    return parser.parse_args()


def validate_export_shape(model_name: str, height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")

    if model_name == "DMPHN124":
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "DMPHN124 requires height and width divisible by 8, "
                f"got {height}x{width}"
            )

    if model_name == "DMSHN124":
        if height < 16 or width < 16:
            raise ValueError(
                "DMSHN124 requires height and width >= 16, "
                f"got {height}x{width}"
            )


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


def main() -> None:
    args = parse_args()

    validate_export_shape(
        model_name=args.model,
        height=args.height,
        width=args.width,
    )

    device = torch.device(args.device)

    model = create_model(args.model).to(device)

    checkpoint = load_checkpoint(
        path=args.checkpoint,
        model=model,
        map_location=device,
        expected_model_name=args.model,
    )

    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(
        1,
        3,
        args.height,
        args.width,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        dummy_output = model(dummy_input)

    if dummy_output.shape != dummy_input.shape:
        raise RuntimeError(
            "unexpected model output shape: "
            f"input={tuple(dummy_input.shape)}, output={tuple(dummy_output.shape)}"
        )

    export_to_onnx(
        model=model,
        dummy_input=dummy_input,
        output_path=args.output,
        opset=args.opset,
    )

    epoch = checkpoint.get("epoch", "unknown")

    print(f"model: {args.model}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint_epoch: {epoch}")
    print(f"input_shape: {tuple(dummy_input.shape)}")
    print(f"output_shape: {tuple(dummy_output.shape)}")
    print(f"onnx_exported: {args.output}")


if __name__ == "__main__":
    main()