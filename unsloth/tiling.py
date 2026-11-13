# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Sequence tiling (tiled MLP): `unsloth.tile_mlp(model)` / `unsloth.untile_mlp(model)`.

Tensor parallel modules stay untiled (a row-parallel all-reduce would run once per tile) unless UNSLOTH_TILED_MLP_TP=1.
"""

import os

__all__ = ["tile_mlp", "untile_mlp"]

_ENV = "UNSLOTH_TILED_MLP"


_OFF = ("", "0", "false", "off", "no", "none")


def _is_off(value):
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


def _refuse_mlx(model):
    if any(
        (getattr(c, "__module__", "") or "").split(".")[0] == "mlx" for c in type(model).__mro__
    ):
        raise NotImplementedError(
            "Unsloth: tiled MLP (tile_mlp / untile_mlp) needs a PyTorch model and is not supported on MLX."
        )


def tile_mlp(
    model,
    rows = None,
    target_gb = None,
    modules = None,
    mode = None,
    verbose = True,
):
    """Tile every eligible MLP / MoE / projector in `model`; idempotent. Returns the coverage report."""
    _refuse_mlx(model)
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
    _refuse_mlx(model)
    return _registry().uninstall(model)


def resolve_loader_mode(unsloth_tiled_mlp):
    """A set UNSLOTH_TILED_MLP wins; an off environment never overrides an explicit kwarg."""
    env = _env_mode()
    if env is not None:
        return env
    return None if _is_off(unsloth_tiled_mlp) else unsloth_tiled_mlp


def check_tiling_available(unsloth_tiled_mlp):
    """Fail fast before weights load; tiling off never imports the registry."""
    if resolve_loader_mode(unsloth_tiled_mlp) is not None:
        _registry()


def loader_tile_mlp(model, unsloth_tiled_mlp):
    mode = resolve_loader_mode(unsloth_tiled_mlp)
    if mode is None:
        return None
    return _registry().install(model, mode)


def revalidate_for_trainer(trainer, model = None):
    if model is None:
        model = getattr(trainer, "model", None)
    if model is None:
        return None
    try:
        return _registry().revalidate(model)
    except ImportError:
        return None
