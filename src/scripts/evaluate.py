from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torchvision.transforms.functional as TF

from src.dehazing.evaluation.metrics import compute_psnr


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dehazing outputs with PSNR")

    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Predicted/dehazed image path for single-image evaluation",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Clean target image path for single-image evaluation",
    )

    parser.add_argument(
        "--pred-dir",
        type=str,
        default=None,
        help="Directory containing predicted/dehazed images",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Dataset root used with --hazy-list and --clean-list",
    )
    parser.add_argument(
        "--hazy-list",
        type=str,
        default=None,
        help="List of hazy input paths used for inference",
    )
    parser.add_argument(
        "--clean-list",
        type=str,
        default=None,
        help="List of clean target paths",
    )
    parser.add_argument(
        "--preserve-relative-paths",
        action="store_true",
        help=(
            "Use this if inference was run with --preserve-relative-paths. "
            "Prediction paths are resolved from hazy-list relative paths."
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
        raise ValueError(f"empty list file: {path}")

    return lines


def load_rgb_tensor(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image)


def output_path_from_hazy_rel(
    pred_dir: Path,
    hazy_rel_path: str,
    preserve_relative_paths: bool,
) -> Path:
    rel_path = Path(hazy_rel_path)

    if preserve_relative_paths:
        return pred_dir / rel_path

    return pred_dir / rel_path.name


def evaluate_pair(pred_path: Path, target_path: Path) -> float:
    pred = load_rgb_tensor(pred_path)
    target = load_rgb_tensor(target_path)

    return compute_psnr(pred, target)


def evaluate_list(
    pred_dir: Path,
    root_dir: Path,
    hazy_list_path: Path,
    clean_list_path: Path,
    preserve_relative_paths: bool,
) -> None:
    hazy_rel_paths = read_list(hazy_list_path)
    clean_rel_paths = read_list(clean_list_path)

    if len(hazy_rel_paths) != len(clean_rel_paths):
        raise ValueError(
            "hazy and clean lists must have the same length, "
            f"got {len(hazy_rel_paths)} and {len(clean_rel_paths)}"
        )

    psnr_values: list[float] = []

    for hazy_rel_path, clean_rel_path in zip(hazy_rel_paths, clean_rel_paths):
        pred_path = output_path_from_hazy_rel(
            pred_dir=pred_dir,
            hazy_rel_path=hazy_rel_path,
            preserve_relative_paths=preserve_relative_paths,
        )
        target_path = root_dir / clean_rel_path

        psnr = evaluate_pair(pred_path=pred_path, target_path=target_path)
        psnr_values.append(psnr)

        print(f"{hazy_rel_path}: PSNR={psnr:.4f} dB")

    average_psnr = sum(psnr_values) / len(psnr_values)

    print("-" * 60)
    print(f"num_images: {len(psnr_values)}")
    print(f"average_PSNR: {average_psnr:.4f} dB")


def validate_args(args: argparse.Namespace) -> None:
    single_mode = args.pred is not None or args.target is not None
    list_mode = (
        args.pred_dir is not None
        or args.root_dir is not None
        or args.hazy_list is not None
        or args.clean_list is not None
    )

    if single_mode and list_mode:
        raise ValueError("use either single-image mode or list mode, not both")

    if single_mode:
        if args.pred is None or args.target is None:
            raise ValueError("single-image mode requires both --pred and --target")
        return

    if list_mode:
        missing = [
            name
            for name, value in [
                ("--pred-dir", args.pred_dir),
                ("--root-dir", args.root_dir),
                ("--hazy-list", args.hazy_list),
                ("--clean-list", args.clean_list),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(f"list mode missing required arguments: {', '.join(missing)}")
        return

    raise ValueError(
        "no evaluation mode selected. Use --pred/--target or "
        "--pred-dir/--root-dir/--hazy-list/--clean-list"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.pred is not None:
        pred_path = Path(args.pred)
        target_path = Path(args.target)

        psnr = evaluate_pair(pred_path=pred_path, target_path=target_path)

        print(f"pred: {pred_path}")
        print(f"target: {target_path}")
        print(f"PSNR: {psnr:.4f} dB")
        return

    evaluate_list(
        pred_dir=Path(args.pred_dir),
        root_dir=Path(args.root_dir),
        hazy_list_path=Path(args.hazy_list),
        clean_list_path=Path(args.clean_list),
        preserve_relative_paths=args.preserve_relative_paths,
    )


if __name__ == "__main__":
    main()