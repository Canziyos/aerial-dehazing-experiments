from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.dehazing.data.paired_dataset import PairedDehazingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect paired dehazing dataset lists")

    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--hazy-list", type=str, required=True)
    parser.add_argument("--clean-list", type=str, required=True)
    parser.add_argument("--max-check", type=int, default=20)
    parser.add_argument("--show-existing", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = PairedDehazingDataset(
        root_dir=args.root_dir,
        hazy_list_path=args.hazy_list,
        clean_list_path=args.clean_list,
    )

    root_dir = Path(args.root_dir)

    print(f"root_dir: {root_dir}")
    print(f"hazy_list: {Path(args.hazy_list)}")
    print(f"clean_list: {Path(args.clean_list)}")
    print(f"pairs: {len(dataset)}")
    print(f"first_hazy: {dataset.hazy_paths[0]}")
    print(f"first_clean: {dataset.clean_paths[0]}")

    max_check = min(args.max_check, len(dataset))
    if max_check <= 0:
        raise ValueError("--max-check must be positive")

    missing: list[tuple[int, str, Path]] = []
    mismatched_sizes: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for index in range(max_check):
        hazy_path = root_dir / dataset.hazy_paths[index]
        clean_path = root_dir / dataset.clean_paths[index]

        if not hazy_path.exists():
            missing.append((index, "hazy", hazy_path))
        if not clean_path.exists():
            missing.append((index, "clean", clean_path))

        if not hazy_path.exists() or not clean_path.exists():
            continue

        with Image.open(hazy_path) as hazy_image:
            hazy_size = hazy_image.size
        with Image.open(clean_path) as clean_image:
            clean_size = clean_image.size

        if hazy_size != clean_size:
            mismatched_sizes.append((index, hazy_size, clean_size))

    print(f"checked_pairs: {max_check}")

    if missing:
        print("missing_files:")
        for index, side, path in missing[:20]:
            print(f"  index={index} side={side}: {path}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more missing files")

        _print_existing_image_sample(root_dir=root_dir, limit=args.show_existing)
        raise FileNotFoundError(
            f"found {len(missing)} missing files among first {max_check} checked pairs"
        )

    if mismatched_sizes:
        print("size_mismatches:")
        for index, hazy_size, clean_size in mismatched_sizes:
            print(f"  index={index}: hazy={hazy_size}, clean={clean_size}")
        raise ValueError(f"found {len(mismatched_sizes)} size mismatches")

    print("dataset inspection passed")


def _print_existing_image_sample(root_dir: Path, limit: int) -> None:
    if limit <= 0:
        return

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    existing = [
        path.relative_to(root_dir)
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    ]

    print(f"existing_image_files_under_root: {len(existing)}")
    for path in existing[:limit]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
