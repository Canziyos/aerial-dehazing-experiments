from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dehazing.architectures.factory import create_model, get_supported_model_names
from src.dehazing.data.paired_dataset import PairedDehazingDataset
from src.dehazing.losses.dehazing_loss import DehazingLoss
from src.dehazing.training.checkpoints import save_checkpoint
from src.dehazing.training.loops import train_one_epoch, validate_one_epoch


def parse_args(default_model: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a rebuilt dehazing model")

    parser.add_argument(
        "--model",
        type=str,
        choices=get_supported_model_names(),
        default=default_model,
        required=default_model is None,
    )

    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--hazy-list", type=str, required=True)
    parser.add_argument("--clean-list", type=str, required=True)

    parser.add_argument("--val-hazy-list", type=str, default=None)
    parser.add_argument("--val-clean-list", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=None)

    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--output-dir", type=str, default="outputs/train")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint.pt")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--use-perceptual", action="store_true")

    return parser.parse_args()


def main(default_model: str | None = None) -> None:
    args = parse_args(default_model=default_model)
    _validate_args(args)

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    device = torch.device(args.device)
    resize = _resolve_resize(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PairedDehazingDataset(
        root_dir=args.root_dir,
        hazy_list_path=args.hazy_list,
        clean_list_path=args.clean_list,
        resize=resize,
        augment=args.augment,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if args.val_hazy_list is not None and args.val_clean_list is not None:
        val_dataset = PairedDehazingDataset(
            root_dir=args.root_dir,
            hazy_list_path=args.val_hazy_list,
            clean_list_path=args.val_clean_list,
            resize=resize,
            augment=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    model = create_model(args.model).to(device)
    loss_fn = DehazingLoss(use_perceptual=args.use_perceptual).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    config = vars(args)

    print(f"model: {args.model}")
    print(f"train_samples: {len(train_dataset)}")
    if val_loader is not None:
        print(f"val_samples: {len(val_loader.dataset)}")
    print(f"device: {device}")

    best_metric: float | None = None
    best_epoch: int | None = None
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}")

        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        print(f"train: {train_losses}")

        val_losses = None
        if val_loader is not None:
            val_losses = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )
            print(f"val:   {val_losses}")

        latest_path = output_dir / args.checkpoint_name
        save_checkpoint(
            path=latest_path,
            model=model,
            epoch=epoch,
            model_name=args.model,
            optimizer=optimizer,
            scheduler=None,
            config=config,
        )
        print(f"saved latest: {latest_path}")

        current_metric = (
            val_losses["total_loss"]
            if val_losses is not None
            else train_losses["total_loss"]
        )

        if best_metric is None or current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch

            best_path = output_dir / "checkpoint_best.pt"
            best_config = {
                **config,
                "best_metric": best_metric,
                "best_metric_name": "val_total_loss" if val_losses is not None else "train_total_loss",
                "best_epoch": best_epoch,
            }

            save_checkpoint(
                path=best_path,
                model=model,
                epoch=epoch,
                model_name=args.model,
                optimizer=optimizer,
                scheduler=None,
                config=best_config,
            )
            print(f"saved best: {best_path} ({current_metric:.6f})")

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            save_checkpoint(
                path=epoch_path,
                model=model,
                epoch=epoch,
                model_name=args.model,
                optimizer=optimizer,
                scheduler=None,
                config=config,
            )
            print(f"saved epoch checkpoint: {epoch_path}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.lr <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")
    if args.torch_threads is not None and args.torch_threads <= 0:
        raise ValueError(f"--torch-threads must be positive, got {args.torch_threads}")
    if args.save_every < 0:
        raise ValueError(f"--save-every must be >= 0, got {args.save_every}")
    if (args.val_hazy_list is None) != (args.val_clean_list is None):
        raise ValueError("validation requires both --val-hazy-list and --val-clean-list")
    if (args.resize_height is None) != (args.resize_width is None):
        raise ValueError("resize requires both --resize-height and --resize-width")


def _resolve_resize(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.resize_height is None and args.resize_width is None:
        return None

    assert args.resize_height is not None
    assert args.resize_width is not None

    if args.resize_height <= 0 or args.resize_width <= 0:
        raise ValueError(
            "resize dimensions must be positive, "
            f"got H={args.resize_height}, W={args.resize_width}"
        )

    return (args.resize_height, args.resize_width)


if __name__ == "__main__":
    main()
