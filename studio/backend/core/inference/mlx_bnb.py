# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mirror of the repo substitution ``unsloth_zoo.mlx.loader`` performs.

mlx-lm cannot read bitsandbytes NF4 weights, so ``FastMLXModel.from_pretrained``
rewrites ``unsloth/<model>-bnb-4bit`` to its full-precision base and downloads that
instead. It announces the swap on the worker's stdout only, so Studio has to derive
it: what to recommend on an MLX host, and which repo a load is really fetching.
"""

import os
from typing import Iterable, Optional

_BNB_SUFFIXES = ("-unsloth-bnb-4bit", "-bnb-4bit")


def mlx_bnb_base_repo(model_name: Optional[str]) -> Optional[str]:
    """The repo MLX loads in place of *model_name*, or None when it loads it as given."""
    if not isinstance(model_name, str) or not model_name.startswith("unsloth/"):
        return None
    if os.path.exists(model_name):
        return None
    for suffix in _BNB_SUFFIXES:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return None


def mlx_host_bnb_base_repo(model_name: Optional[str]) -> Optional[str]:
    """``mlx_bnb_base_repo`` for a repo this host really loads through the MLX loader.

    Diffusion is the exception that matters: its bnb repos run on the diffusers/MPS
    path, which reads bitsandbytes weights, so they are loaded exactly as named.
    """
    import utils.hardware.hardware as hw
    from core.inference.diffusion_families import detect_family

    if hw.get_device() != hw.DeviceType.MLX:
        return None
    if not isinstance(model_name, str) or detect_family(model_name) is not None:
        return None
    return mlx_bnb_base_repo(model_name)


def mlx_bnb_substitutions(repos: Iterable[str]) -> list[tuple[str, str]]:
    """``(requested, base)`` for every repo in *repos* MLX will swap out."""
    swaps = []
    for repo in repos:
        base = mlx_bnb_base_repo(repo)
        if base:
            swaps.append((repo, base))
    return swaps
