from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.dehazing.architectures.factory import create_model, get_supported_model_names
from src.dehazing.inference.folder import iter_image_paths
from src.dehazing.inference.image import run_single_image_inference
from src.dehazing.training.checkpoints import load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dehazing inference")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_supported_model_names(),
        help="Model architecture to use",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=str,
        help="Input image file or input image folder",
    )
    source.add_argument(
        "--input-list",
        type=str,
        help="Text file containing image paths relative to --root-dir",
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Dataset root. Required when using --input-list",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output image path for single image, or output directory for folder/list inference",
    )
    parser.add_argument(
        "--preserve-relative-paths",
        action="store_true",
        help=(
            "For --input-list, preserve relative input paths under the output directory. "
            "Useful when different splits contain images with the same filename."
        ),
    )

    return parser.parse_args()


def read_list(path: str | Path) -> list[str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"list file not found: {path}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not lines:
        raise ValueError(f"empty input list: {path}")

    return lines


def resolve_input_list(
    root_dir: str | Path,
    input_list: str | Path,
) -> list[tuple[Path, str]]:
    root = Path(root_dir)
    rel_paths = read_list(input_list)

    resolved: list[tuple[Path, str]] = []
    for rel_path in rel_paths:
        image_path = root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"image listed but not found: {image_path}")
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {image_path}")
        resolved.append((image_path, Path(rel_path).as_posix()))

    return resolved


def output_path_for_list_item(
    output_dir: Path,
    rel_path: str,
    preserve_relative_paths: bool,
) -> Path:
    rel = Path(rel_path)

    if preserve_relative_paths:
        return output_dir / rel

    return output_dir / rel.name


def ensure_unique_output_paths(output_paths: list[Path]) -> None:
    seen: set[Path] = set()
    duplicates: list[Path] = []

    for path in output_paths:
        if path in seen:
            duplicates.append(path)
        seen.add(path)

    if duplicates:
        joined = "\n".join(str(path) for path in duplicates[:10])
        raise ValueError(
            "duplicate output paths detected. "
            "Use --preserve-relative-paths to avoid overwriting files.\n"
            f"Examples:\n{joined}"
        )


def load_model(
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device,
):
    model = create_model(model_name).to(device)

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        map_location=device,
        expected_model_name=model_name,
    )

    print(f"loaded checkpoint: {checkpoint_path}")
    print(f"requested model: {model_name}")
    print(f"checkpoint model_name: {checkpoint.get('model_name')}")
    print(f"epoch: {checkpoint.get('epoch')}")

    config = checkpoint.get("config") or {}
    if "best_metric" in config:
        print(
            "best: "
            f"{config.get('best_metric_name')}={config.get('best_metric')} "
            f"at epoch {config.get('best_epoch')}"
        )

    return model


def infer_one(
    model,
    image_path: Path,
    output_path: Path,
    device: torch.device,
) -> float:
    start_time = time.perf_counter()

    run_single_image_inference(
        model=model,
        image_path=image_path,
        output_path=output_path,
        device=device,
    )

    return time.perf_counter() - start_time


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    model = load_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    output = Path(args.output)

    if args.input_list is not None:
        if args.root_dir is None:
            raise ValueError("--root-dir is required when using --input-list")

        items = resolve_input_list(
            root_dir=args.root_dir,
            input_list=args.input_list,
        )

        output_paths = [
            output_path_for_list_item(
                output_dir=output,
                rel_path=rel_path,
                preserve_relative_paths=args.preserve_relative_paths,
            )
            for _, rel_path in items
        ]
        ensure_unique_output_paths(output_paths)

        total_runtime = 0.0

        for (image_path, rel_path), output_path in zip(items, output_paths):
            elapsed = infer_one(
                model=model,
                image_path=image_path,
                output_path=output_path,
                device=device,
            )
            total_runtime += elapsed
            print(f"{rel_path}: {elapsed:.4f} s")

        print("inference finished")
        print(
            {
                "num_images": len(items),
                "total_runtime_sec": total_runtime,
                "average_runtime_sec": total_runtime / len(items),
            }
        )
        return

    input_path = Path(args.input)

    if input_path.is_file():
        output_path = output

        if output_path.suffix.lower() not in IMAGE_EXTENSIONS:
            output_path = output_path / input_path.name

        elapsed = infer_one(
            model=model,
            image_path=input_path,
            output_path=output_path,
            device=device,
        )

        print(f"{input_path.name}: {elapsed:.4f} s")
        print(f"saved output: {output_path}")
        return

    if input_path.is_dir():
        image_paths = iter_image_paths(input_path)
        output.mkdir(parents=True, exist_ok=True)

        output_paths = [output / image_path.name for image_path in image_paths]
        ensure_unique_output_paths(output_paths)

        total_runtime = 0.0

        for image_path, output_path in zip(image_paths, output_paths):
            elapsed = infer_one(
                model=model,
                image_path=image_path,
                output_path=output_path,
                device=device,
            )
            total_runtime += elapsed
            print(f"{image_path.name}: {elapsed:.4f} s")

        print("inference finished")
        print(
            {
                "num_images": len(image_paths),
                "total_runtime_sec": total_runtime,
                "average_runtime_sec": total_runtime / len(image_paths),
            }
        )
        return

    raise FileNotFoundError(f"input path not found: {input_path}")


if __name__ == "__main__":
    main()