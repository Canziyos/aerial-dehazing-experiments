from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dehazing.architectures.dmphn import DMPHN124
from src.dehazing.training.checkpoints import load_legacy_dmphn_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert original six-file DMPHN checkpoint folder to one clean checkpoint.pt"
    )

    parser.add_argument("--legacy-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = DMPHN124().to(device)
    load_legacy_dmphn_checkpoint(
        checkpoint_dir=args.legacy_dir,
        model=model,
        map_location=device,
        strict=True,
    )

    output_path = Path(args.output)
    save_checkpoint(
        path=output_path,
        model=model,
        epoch=0,
        model_name="DMPHN124",
        optimizer=None,
        scheduler=None,
        config={
            "source": "legacy_dmphn_six_file_checkpoint",
            "legacy_dir": str(args.legacy_dir),
        },
    )

    print(f"converted legacy checkpoint: {args.legacy_dir}")
    print(f"saved clean checkpoint: {output_path}")


if __name__ == "__main__":
    main()
