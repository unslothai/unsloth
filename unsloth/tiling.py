# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Sequence tiling (tiled MLP) for any model: `unsloth.tile_mlp(model)` / `unsloth.untile_mlp(model)`.

Each token-local module (dense MLPs, top-k MoE blocks, VLM projectors) runs over the sequence in row tiles, so only
one tile's intermediate activations are alive at a time. Which modules qualify, and how their outputs are put back
together, is decided by `unsloth_zoo.tiling_registry`.

Parallelism rule (unsloth_zoo.tiling_registry.PARALLEL_RULE): tiling is per rank. It splits only the rows this rank
holds; under Ulysses context parallel the MLP sees the local sequence shard, so the tiles are planned on the shard.
DDP and FSDP2 need nothing extra. Tensor parallel modules are left untiled (a row-parallel all-reduce would run once
per tile) unless UNSLOTH_TILED_MLP_TP=1.
"""

import os

__all__ = ["tile_mlp", "untile_mlp"]

_ENV = "UNSLOTH_TILED_MLP"


def _registry():
    try:
        from unsloth_zoo import tiling_registry
    except ImportError as e:
        raise ImportError(
            "Unsloth: tiled MLP needs a newer unsloth_zoo (unsloth_zoo.tiling_registry); "
            "update with `pip install --upgrade unsloth_zoo`"
        ) from e
    return tiling_registry


def tile_mlp(model, rows = None, target_gb = None, modules = None, mode = None, verbose = True):
    """Tile every eligible MLP / MoE / projector in `model`; idempotent. Returns the coverage report.

    rows: fixed rows per tile. target_gb: activation budget per tile in GB. Neither: `mode`, which defaults to
    UNSLOTH_TILED_MLP or "arctic" (rows per tile = hidden size); "auto" sizes tiles from free memory.
    modules: name or class patterns; only matching modules are tiled.
    """
    if rows is not None and target_gb is not None:
        raise ValueError("Unsloth: tile_mlp takes rows or target_gb, not both")
    if rows is not None:
        mode = f"rows:{int(rows)}"
    elif target_gb is not None:
        mode = f"target:{float(target_gb)}"
    elif mode is None:
        env = os.environ.get(_ENV, "").strip()
        # An explicit call wins over a disabled environment.
        mode = env if env.lower() not in ("", "0", "false", "off") else True
    return _registry().install(model, mode, modules = modules, verbose = verbose)


def untile_mlp(model):
    """Undo `tile_mlp`; returns the number of modules restored."""
    return _registry().uninstall(model)


def resolve_loader_mode(unsloth_tiled_mlp):
    """The tiling mode a loader call asks for, or None. UNSLOTH_TILED_MLP (when set and not "0") gives the mode;
    a truthy `unsloth_tiled_mlp` turns tiling on even when the environment says "0" (the old behaviour)."""
    env = os.environ.get(_ENV, "").strip()
    env_on = env not in ("", "0")
    if env_on:
        return env
    if unsloth_tiled_mlp is None or unsloth_tiled_mlp is False:
        return None
    if isinstance(unsloth_tiled_mlp, str) and unsloth_tiled_mlp.strip() in ("", "0"):
        return None
    return unsloth_tiled_mlp


def loader_tile_mlp(model, unsloth_tiled_mlp):
    """Loader call site: tile `model` when the kwarg or UNSLOTH_TILED_MLP asks for it."""
    mode = resolve_loader_mode(unsloth_tiled_mlp)
    if mode is None:
        return None
    return _registry().install(model, mode)


def revalidate_for_trainer(trainer):
    """Per Trainer instance: re-run the model's tiling plan so modules added since load (PEFT, the fused LoRA MLP
    forward) are covered. No-op for a model that was never tiled."""
    model = getattr(trainer, "model", None)
    if model is None:
        return None
    try:
        return _registry().revalidate(model)
    except ImportError:
        return None
