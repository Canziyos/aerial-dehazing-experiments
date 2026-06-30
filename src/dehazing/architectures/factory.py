from __future__ import annotations

from torch import nn

from src.dehazing.architectures.dmphn import DMPHN124
from src.dehazing.architectures.dmshn import DMSHN124


def create_model(model_name: str) -> nn.Module:
    normalized_name = _normalize_model_name(model_name)

    if normalized_name == "DMPHN124":
        return DMPHN124()

    if normalized_name == "DMSHN124":
        return DMSHN124()

    supported = ", ".join(get_supported_model_names())
    raise ValueError(
        f"unsupported model_name={model_name!r}. "
        f"Supported models: {supported}"
    )


def get_supported_model_names() -> tuple[str, ...]:
    return (
        "DMPHN124",
        "DMSHN124",
    )


def _normalize_model_name(model_name: str) -> str:
    return (
        model_name
        .strip()
        .upper()
        .replace("-", "")
        .replace("_", "")
    )