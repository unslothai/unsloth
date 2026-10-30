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


_OFF = ("", "0", "false", "off", "no", "none")


def _is_off(value):
    """One parse for "tiling off", shared by the env var, the loader kwarg and `tile_mlp`: None, False, 0 and the
    strings "", "0", "false", "off", "no", "none" (any case)."""
    if value is None or value is False:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip().lower() in _OFF
    return False


def _env_mode():
    """UNSLOTH_TILED_MLP, or None when it is unset or any off spelling."""
    env = os.environ.get(_ENV, "").strip()
    return None if _is_off(env) else env


def _registry():
    import importlib

    try:
        tiling_registry = importlib.import_module("unsloth_zoo.tiling_registry")
    except ImportError as e:
        raise ImportError(
            "Unsloth: tiled MLP needs a newer unsloth_zoo (unsloth_zoo.tiling_registry); "
            "update with `pip install --upgrade unsloth_zoo`"
        ) from e
    return tiling_registry


def tile_mlp(
    model,
    rows = None,
    target_gb = None,
    modules = None,
    mode = None,
    verbose = True,
):
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
        # An explicit call wins over a disabled environment.
        env = _env_mode()
        mode = env if env is not None else True
    return _registry().install(model, mode, modules = modules, verbose = verbose)


def untile_mlp(model):
    """Undo `tile_mlp`; returns the number of modules restored."""
    return _registry().uninstall(model)


def resolve_loader_mode(unsloth_tiled_mlp):
    """The tiling mode a loader call asks for, or None. UNSLOTH_TILED_MLP, when set to anything but an off spelling,
    gives the mode; otherwise a `unsloth_tiled_mlp` that is not off turns tiling on, so an off environment never
    overrides an explicit kwarg (the old "0" behaviour, now the same for every off spelling)."""
    env = _env_mode()
    if env is not None:
        return env
    return None if _is_off(unsloth_tiled_mlp) else unsloth_tiled_mlp


def check_tiling_available(unsloth_tiled_mlp):
    """Loader entry, before any weights load: when tiling is requested, make sure this unsloth_zoo has the registry
    and fail fast otherwise. Tiling off never imports it."""
    if resolve_loader_mode(unsloth_tiled_mlp) is not None:
        _registry()


def loader_tile_mlp(model, unsloth_tiled_mlp):
    """Loader call site: tile `model` when the kwarg or UNSLOTH_TILED_MLP asks for it."""
    mode = resolve_loader_mode(unsloth_tiled_mlp)
    if mode is None:
        return None
    return _registry().install(model, mode)


def revalidate_for_trainer(trainer, model = None):
    """Per Trainer instance: re-run the model's tiling plan so modules added since load (PEFT, the fused LoRA MLP
    forward) are covered. No-op for a model that was never tiled. `model` overrides `trainer.model` (the fresh model
    `model_init` returns)."""
    if model is None:
        model = getattr(trainer, "model", None)
    if model is None:
        return None
    try:
        return _registry().revalidate(model)
    except ImportError:
        return None
