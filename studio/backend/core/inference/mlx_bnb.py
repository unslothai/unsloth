# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mirror Zoo’s MLX BNB repository substitution without importing the ML stack."""

import os
from fnmatch import fnmatchcase
from typing import Iterable, Optional

_BNB_SUFFIXES = ("-unsloth-bnb-4bit", "-bnb-4bit")


def mlx_bnb_base_repo(model_name: Optional[str]) -> Optional[str]:
    """Return the replacement base repository, or None."""
    if not isinstance(model_name, str) or not model_name.startswith("unsloth/"):
        return None
    if os.path.exists(model_name):
        return None
    for suffix in _BNB_SUFFIXES:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return None


def mlx_host_bnb_base_repo(model_name: Optional[str]) -> Optional[str]:
    """Return the MLX replacement, excluding diffusion models."""
    import utils.hardware.hardware as hw
    from core.inference.diffusion_families import detect_family

    if hw.get_device() != hw.DeviceType.MLX:
        return None
    if not isinstance(model_name, str) or detect_family(model_name) is not None:
        return None
    return mlx_bnb_base_repo(model_name)


def mlx_bnb_substitutions(repos: Iterable[str]) -> list[tuple[str, str]]:
    swaps = []
    for repo in repos:
        base = mlx_bnb_base_repo(repo)
        if base:
            swaps.append((repo, base))
    return swaps


# mlx-lm's download filter, plus the weights of an exported adapter.
_MLX_LOAD_PATTERNS = (
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
    "adapters.safetensors",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def mlx_load_siblings(siblings):
    return [
        sibling
        for sibling in siblings
        if any(fnmatchcase(sibling.rfilename, pattern) for pattern in _MLX_LOAD_PATTERNS)
    ]
