from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
) -> dict[str, float]:
    model.train()

    # Keep loss module in eval mode because perceptual VGG must stay frozen/eval.
    loss_fn.eval()

    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "perceptual_loss": 0.0,
        "tv_loss": 0.0,
    }

    sample_count = 0

    iterator = tqdm(
        dataloader,
        desc="train",
        unit="batch",
        ncols=100,
        dynamic_ncols=False,
        leave=True,
        ascii=True,
    )

    for batch in iterator:
        hazy = batch["hazy"].to(device) - 0.5
        clean = batch["clean"].to(device) - 0.5

        batch_size = hazy.shape[0]

        output = model(hazy)

        total_loss, rec_loss, perc_loss, tv_loss = loss_fn(output, clean)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        totals["total_loss"] += total_loss.item() * batch_size
        totals["reconstruction_loss"] += rec_loss.item() * batch_size
        totals["perceptual_loss"] += perc_loss.item() * batch_size
        totals["tv_loss"] += tv_loss.item() * batch_size

        sample_count += batch_size
        iterator.set_postfix_str(f"L={total_loss.item():.4f}")

    return _average_losses(totals, sample_count)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    model.eval()
    loss_fn.eval()

    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "perceptual_loss": 0.0,
        "tv_loss": 0.0,
    }

    sample_count = 0

    iterator = tqdm(
        dataloader,
        desc="val",
        unit="batch",
        ncols=100,
        dynamic_ncols=False,
        leave=True,
        ascii=True,
    )

    for batch in iterator:
        hazy = batch["hazy"].to(device) - 0.5
        clean = batch["clean"].to(device) - 0.5

        batch_size = hazy.shape[0]

        output = model(hazy)

        total_loss, rec_loss, perc_loss, tv_loss = loss_fn(output, clean)

        totals["total_loss"] += total_loss.item() * batch_size
        totals["reconstruction_loss"] += rec_loss.item() * batch_size
        totals["perceptual_loss"] += perc_loss.item() * batch_size
        totals["tv_loss"] += tv_loss.item() * batch_size

        sample_count += batch_size

        iterator.set_postfix_str(f"L={total_loss.item():.4f}")

    return _average_losses(totals, sample_count)


def _average_losses(
    totals: dict[str, float],
    sample_count: int,
) -> dict[str, float]:
    if sample_count <= 0:
        raise ValueError("cannot average losses over an empty dataloader")

    return {
        name: value / sample_count
        for name, value in totals.items()
    }
