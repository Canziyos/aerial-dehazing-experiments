from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


_ENCODER_LEGACY_KEY_MAP = {
    "layer1.weight": "net.0.weight",
    "layer1.bias": "net.0.bias",
    "layer2.0.weight": "net.1.conv1.weight",
    "layer2.0.bias": "net.1.conv1.bias",
    "layer2.2.weight": "net.1.conv2.weight",
    "layer2.2.bias": "net.1.conv2.bias",
    "layer3.0.weight": "net.2.conv1.weight",
    "layer3.0.bias": "net.2.conv1.bias",
    "layer3.2.weight": "net.2.conv2.weight",
    "layer3.2.bias": "net.2.conv2.bias",
    "layer5.weight": "net.3.weight",
    "layer5.bias": "net.3.bias",
    "layer6.0.weight": "net.4.conv1.weight",
    "layer6.0.bias": "net.4.conv1.bias",
    "layer6.2.weight": "net.4.conv2.weight",
    "layer6.2.bias": "net.4.conv2.bias",
    "layer7.0.weight": "net.5.conv1.weight",
    "layer7.0.bias": "net.5.conv1.bias",
    "layer7.2.weight": "net.5.conv2.weight",
    "layer7.2.bias": "net.5.conv2.bias",
    "layer9.weight": "net.6.weight",
    "layer9.bias": "net.6.bias",
    "layer10.0.weight": "net.7.conv1.weight",
    "layer10.0.bias": "net.7.conv1.bias",
    "layer10.2.weight": "net.7.conv2.weight",
    "layer10.2.bias": "net.7.conv2.bias",
    "layer11.0.weight": "net.8.conv1.weight",
    "layer11.0.bias": "net.8.conv1.bias",
    "layer11.2.weight": "net.8.conv2.weight",
    "layer11.2.bias": "net.8.conv2.bias",
}

_DECODER_LEGACY_KEY_MAP = {
    "layer13.0.weight": "net.0.conv1.weight",
    "layer13.0.bias": "net.0.conv1.bias",
    "layer13.2.weight": "net.0.conv2.weight",
    "layer13.2.bias": "net.0.conv2.bias",
    "layer14.0.weight": "net.1.conv1.weight",
    "layer14.0.bias": "net.1.conv1.bias",
    "layer14.2.weight": "net.1.conv2.weight",
    "layer14.2.bias": "net.1.conv2.bias",
    "layer16.weight": "net.2.weight",
    "layer16.bias": "net.2.bias",
    "layer17.0.weight": "net.3.conv1.weight",
    "layer17.0.bias": "net.3.conv1.bias",
    "layer17.2.weight": "net.3.conv2.weight",
    "layer17.2.bias": "net.3.conv2.bias",
    "layer18.0.weight": "net.4.conv1.weight",
    "layer18.0.bias": "net.4.conv1.bias",
    "layer18.2.weight": "net.4.conv2.weight",
    "layer18.2.bias": "net.4.conv2.bias",
    "layer20.weight": "net.5.weight",
    "layer20.bias": "net.5.bias",
    "layer21.0.weight": "net.6.conv1.weight",
    "layer21.0.bias": "net.6.conv1.bias",
    "layer21.2.weight": "net.6.conv2.weight",
    "layer21.2.bias": "net.6.conv2.bias",
    "layer22.0.weight": "net.7.conv1.weight",
    "layer22.0.bias": "net.7.conv1.bias",
    "layer22.2.weight": "net.7.conv2.weight",
    "layer22.2.bias": "net.7.conv2.bias",
    "layer24.weight": "net.8.weight",
    "layer24.bias": "net.8.bias",
}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    model_name: str,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": model_name,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config or {},
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    expected_model_name: str | None = None,
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    checkpoint_model_name = checkpoint.get("model_name")

    if expected_model_name is not None and checkpoint_model_name != expected_model_name:
        raise ValueError(
            "checkpoint model mismatch: "
            f"expected {expected_model_name!r}, "
            f"got {checkpoint_model_name!r}"
        )

    model.load_state_dict(checkpoint["model_state"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


def load_legacy_dmphn_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> None:
    """
    Load original six-file DMPHN checkpoint folders into the rebuilt DMPHN124.

    Expected files:
        encoder_lv1.pkl
        encoder_lv2.pkl
        encoder_lv3.pkl
        decoder_lv1.pkl
        decoder_lv2.pkl
        decoder_lv3.pkl

    The original implementation used layer names such as ``layer1`` and
    ``layer13``. The rebuilt modules use clean sequential names such as
    ``net.0`` and residual-block names such as ``net.1.conv1``. This loader
    translates the old keys before calling ``load_state_dict``.
    """
    checkpoint_dir = Path(checkpoint_dir)

    mapping = {
        "encoder_lv1": ("encoder_lv1.pkl", _ENCODER_LEGACY_KEY_MAP),
        "encoder_lv2": ("encoder_lv2.pkl", _ENCODER_LEGACY_KEY_MAP),
        "encoder_lv3": ("encoder_lv3.pkl", _ENCODER_LEGACY_KEY_MAP),
        "decoder_lv1": ("decoder_lv1.pkl", _DECODER_LEGACY_KEY_MAP),
        "decoder_lv2": ("decoder_lv2.pkl", _DECODER_LEGACY_KEY_MAP),
        "decoder_lv3": ("decoder_lv3.pkl", _DECODER_LEGACY_KEY_MAP),
    }

    for module_name, (filename, key_map) in mapping.items():
        path = checkpoint_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"legacy checkpoint file not found: {path}")

        module = getattr(model, module_name)
        legacy_state_dict = torch.load(path, map_location=map_location)
        state_dict = _remap_state_dict_keys(legacy_state_dict, key_map)
        module.load_state_dict(state_dict, strict=strict)


def _remap_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
    key_map: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []

    for old_key, value in state_dict.items():
        new_key = key_map.get(old_key)
        if new_key is None:
            unexpected_keys.append(old_key)
            continue
        remapped[new_key] = value

    if unexpected_keys:
        joined = ", ".join(unexpected_keys)
        raise ValueError(f"unexpected legacy checkpoint keys: {joined}")

    missing_old_keys = sorted(set(key_map) - set(state_dict))
    if missing_old_keys:
        joined = ", ".join(missing_old_keys)
        raise ValueError(f"missing legacy checkpoint keys: {joined}")

    return remapped
