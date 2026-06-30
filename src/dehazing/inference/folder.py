from __future__ import annotations

from pathlib import Path
import time

from torch import nn

from src.dehazing.inference.image import run_single_image_inference


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_paths(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"expected directory, got: {input_dir}")

    image_paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise ValueError(f"no supported images found in: {input_dir}")

    return image_paths


def run_folder_inference(
    model: nn.Module,
    input_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    record_runtime: bool = True,
) -> dict[str, float | int]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(input_dir)

    total_runtime = 0.0

    for image_path in image_paths:
        output_path = output_dir / image_path.name

        start_time = time.perf_counter()

        run_single_image_inference(
            model=model,
            image_path=image_path,
            output_path=output_path,
            device=device,
        )

        elapsed = time.perf_counter() - start_time
        total_runtime += elapsed

        if record_runtime:
            print(f"{image_path.name}: {elapsed:.4f} s")

    average_runtime = total_runtime / len(image_paths)

    return {
        "num_images": len(image_paths),
        "total_runtime_sec": total_runtime,
        "average_runtime_sec": average_runtime,
    }