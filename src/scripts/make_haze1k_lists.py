# scripts/make_haze1k_lists.py

from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_pairs(root: Path, split_dir: str) -> list[tuple[Path, Path]]:
    input_dir = root / split_dir / "input"
    target_dir = root / split_dir / "target"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Missing target dir: {target_dir}")

    input_images = image_files(input_dir)
    target_by_name = {path.name: path for path in image_files(target_dir)}

    pairs: list[tuple[Path, Path]] = []

    for input_path in input_images:
        target_path = target_by_name.get(input_path.name)
        if target_path is None:
            raise FileNotFoundError(f"No matching target for: {input_path.name}")
        pairs.append((input_path, target_path))

    return pairs


def write_pair_lists(
    pairs: list[tuple[Path, Path]],
    root: Path,
    hazy_list_path: Path,
    clean_list_path: Path,
) -> None:
    hazy_list_path.parent.mkdir(parents=True, exist_ok=True)

    hazy_lines = [relative_posix(hazy, root) for hazy, _ in pairs]
    clean_lines = [relative_posix(clean, root) for _, clean in pairs]

    hazy_list_path.write_text("\n".join(hazy_lines) + "\n", encoding="utf-8")
    clean_list_path.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        default="data/haze1k/Distributed_haze1k",
        help="Path to Distributed_haze1k",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root_dir)
    lists_dir = root / "lists"

    train_pairs = collect_pairs(root, "train")
    rng = random.Random(args.seed)
    rng.shuffle(train_pairs)

    val_count = int(round(len(train_pairs) * args.val_ratio))
    val_pairs = train_pairs[:val_count]
    train_pairs = train_pairs[val_count:]

    write_pair_lists(
        train_pairs,
        root,
        lists_dir / "train_hazy.txt",
        lists_dir / "train_clean.txt",
    )
    write_pair_lists(
        val_pairs,
        root,
        lists_dir / "val_hazy.txt",
        lists_dir / "val_clean.txt",
    )

    all_test_pairs: list[tuple[Path, Path]] = []

    for split in ["test_thin", "test_moderate", "test_thick"]:
        pairs = collect_pairs(root, split)
        all_test_pairs.extend(pairs)

        split_name = split.replace("test_", "")
        write_pair_lists(
            pairs,
            root,
            lists_dir / f"test_{split_name}_hazy.txt",
            lists_dir / f"test_{split_name}_clean.txt",
        )

    write_pair_lists(
        all_test_pairs,
        root,
        lists_dir / "test_all_hazy.txt",
        lists_dir / "test_all_clean.txt",
    )

    print(f"root_dir: {root}")
    print(f"train_pairs: {len(train_pairs)}")
    print(f"val_pairs: {len(val_pairs)}")
    print(f"test_pairs: {len(all_test_pairs)}")
    print(f"wrote: {lists_dir}")


if __name__ == "__main__":
    main()