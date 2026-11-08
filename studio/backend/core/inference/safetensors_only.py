# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image / video / text-encoder weights load from ``.safetensors`` only: a pickle container (``.pt`` / ``.pth`` /
``.bin`` / ``.ckpt``) is never resolved, downloaded or opened on the load path. Pure string helpers, no torch."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

SAFETENSORS_SUFFIX = ".safetensors"

# Containers torch.load would unpickle. ``.bin`` is listed for weight files only (callers scope it).
PICKLE_WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle")

PICKLE_IGNORE_PATTERNS = tuple(f"*{s}" for s in PICKLE_WEIGHT_SUFFIXES)

SAFETENSORS_ONLY_MESSAGE = (
    "image / video checkpoints are loaded from .safetensors only; {what} is a pickle (.pt / .pth / .bin / "
    ".ckpt) and is refused. Use the .safetensors checkpoint instead "
    "(scripts/convert_prequant_to_safetensors.py converts a pre-quantized .pt)."
)


def _lower(name: Any) -> str:
    return str(name or "").strip().lower()


def is_pickle_weight(name: Optional[Any]) -> bool:
    """True when ``name`` (a file name or path) ends in a pickle container suffix."""
    return _lower(name).endswith(PICKLE_WEIGHT_SUFFIXES)


def is_safetensors_name(name: Optional[Any]) -> bool:
    return _lower(name).endswith(SAFETENSORS_SUFFIX)


def safetensors_only(names: Iterable[Any]) -> list:
    """``names`` in order, minus empties and every name that is not a ``.safetensors`` file."""
    return [n for n in names if n and is_safetensors_name(n)]


def safetensors_twin(name: Optional[str]) -> Optional[str]:
    """``<stem>.safetensors`` for a pickle name, the name itself for a safetensors one, else None."""
    if not name:
        return None
    if is_safetensors_name(name):
        return name
    lower = _lower(name)
    for suffix in PICKLE_WEIGHT_SUFFIXES:
        if lower.endswith(suffix):
            return str(name).strip()[: -len(suffix)] + SAFETENSORS_SUFFIX
    return None


def refuse_pickle_path(path: Any, what: Optional[str] = None) -> None:
    """Raise ValueError unless ``path`` names a ``.safetensors`` file."""
    if not is_safetensors_name(path):
        raise ValueError(SAFETENSORS_ONLY_MESSAGE.format(what = what or repr(str(path))))
