# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rebuild the loaded scheduler at ComfyUI's static sigma shift (shipped schedulers differ, e.g.
Qwen-Image dynamic ~2.0 + terminal stretch vs 3.1), unless the checkpoint ships its own sampling
grid. No torch/diffusers imports."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

# "0" keeps every shipped diffusers scheduler instead of ComfyUI's static schedule.
COMFY_SIGMAS_ENV = "UNSLOTH_DIFFUSION_COMFY_SIGMAS"


def comfy_sigmas_enabled() -> bool:
    return os.environ.get(COMFY_SIGMAS_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def flux_mu_shift(mu: float) -> float:
    """Static shift equal to ComfyUI's ModelSamplingFlux at a fixed ``mu``:
    e^mu / (e^mu + 1/t - 1) == s*t / (1 + (s - 1)*t) for s = e^mu, at every resolution."""
    return math.exp(mu)


def flow_shift_overrides(config: Any, shift: float) -> Optional[dict]:
    """Scheduler config overrides that sample at static ``shift``, or None when the scheduler has
    no flow shift Studio knows how to set."""
    try:
        keys = set(config.keys())
    except Exception:  # noqa: BLE001 - an exotic config just keeps its scheduler
        return None
    if "flow_shift" in keys and config.get("use_flow_sigmas", True):
        return {"flow_shift": float(shift)}
    if "shift" in keys and "use_dynamic_shifting" in keys:
        # FlowMatchEulerDiscrete: static shift, no resolution-dependent mu, no terminal stretch.
        overrides = {"shift": float(shift), "use_dynamic_shifting": False}
        if "shift_terminal" in keys:
            overrides["shift_terminal"] = None
        return overrides
    return None


def apply_comfy_flow_shift(
    pipe: Any,
    shift: Optional[float],
    logger: Any = None,
) -> bool:
    """Rebuild ``pipe.scheduler`` at ComfyUI's static ``shift``. True when it changed."""
    if shift is None or not comfy_sigmas_enabled():
        return False
    scheduler = getattr(pipe, "scheduler", None)
    config = getattr(scheduler, "config", None)
    if scheduler is None or config is None:
        return False
    overrides = flow_shift_overrides(config, shift)
    if not overrides:
        return False
    if all(config.get(k) == v for k, v in overrides.items()):
        return False
    try:
        pipe.scheduler = type(scheduler).from_config(config, **overrides)
    except Exception as exc:  # noqa: BLE001 - keep the shipped schedule rather than fail the load
        if logger is not None:
            logger.warning("flow shift %s not applied: %s", shift, exc)
        return False
    if logger is not None:
        logger.info("scheduler flow shift set to %s (ComfyUI default)", shift)
    return True


# A pipeline config's fixed sampling grid, terminal sigma excluded (QwenImage21Pipeline, huggingface/diffusers#14950).
SAMPLE_SIGMAS_KEY = "sample_sigmas"


def _valid_sample_sigmas(raw: Any) -> Optional[tuple[float, ...]]:
    """``raw`` as a grid the scheduler can take: finite, in (0, 1], strictly decreasing. None otherwise."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    try:
        grid = tuple(float(s) for s in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(s) and 0.0 < s <= 1.0 for s in grid):
        return None
    if any(b >= a for a, b in zip(grid, grid[1:])):
        return None
    return grid


# Where install_sample_sigmas keeps a grid the pipeline config cannot hold: on the pinned diffusers an extra config key
# fails DiffusionPipeline.components ("Expected [...] to be defined"), which group offload and from_pipe both read.
SAMPLE_SIGMAS_ATTR = "_unsloth_sample_sigmas"


def pipe_sample_sigmas(pipe: Any) -> Optional[tuple[float, ...]]:
    """The grid the loaded pipeline carries (its own config on diffusers with #14950, else installed), or None."""
    config = getattr(pipe, "config", None)
    try:
        raw = config.get(SAMPLE_SIGMAS_KEY) if config is not None else None
    except Exception:  # noqa: BLE001 - an exotic config carries no grid
        raw = None
    return _valid_sample_sigmas(raw if raw is not None else getattr(pipe, SAMPLE_SIGMAS_ATTR, None))


def install_sample_sigmas(
    pipe: Any,
    raw: Any,
    logger: Any = None,
) -> Optional[tuple[float, ...]]:
    """The checkpoint's fixed sampling grid, carried on ``pipe``. ``raw`` is ``sample_sigmas`` from the model_index.json
    the pipeline was built from, which the pinned diffusers drops at load (diffusers with #14950 keeps it in the
    config). None when the checkpoint ships none. A grid replaces the static flow shift: the shipped scheduler is the
    one it was tuned on."""
    grid = pipe_sample_sigmas(pipe)
    if grid is not None or raw is None:
        return grid
    grid = _valid_sample_sigmas(raw)
    if grid is None:
        if logger is not None:
            logger.warning("sample_sigmas ignored: not a decreasing grid in (0, 1]: %r", raw)
        return None
    setattr(pipe, SAMPLE_SIGMAS_ATTR, grid)
    if logger is not None:
        logger.info(
            "checkpoint sampling grid installed: %d steps, shipped scheduler kept", len(grid)
        )
    return grid


def sample_sigmas_for_steps(grid: tuple[float, ...], steps: int) -> list[float]:
    """The grid at ``steps`` steps: itself at its own length, else resampled along it (linear in step index), so another
    step count still follows the checkpoint's curve rather than a linear 1 -> 0 ramp. Studio's own, experimental: the
    Turbo card evaluates only the saved 8-step grid. Measured on Qwen-Image-2.1-Turbo
    at 4 / 6 / 12 / 16 steps: the ramp ghosts at 4-6 and turns grainy at 12-16, the resampled grid stays clean."""
    steps = max(1, int(steps))
    if steps == len(grid):
        return list(grid)
    if len(grid) == 1:
        # No curve to follow: the pipeline's own linear spacing from the grid's start.
        return [grid[0] * (1.0 - i / steps) for i in range(steps)]
    if steps == 1:
        return [grid[0]]
    last = len(grid) - 1
    out = []
    for i in range(steps):
        pos = i * last / (steps - 1)
        lo = min(int(pos), last - 1)
        out.append(grid[lo] + (grid[lo + 1] - grid[lo]) * (pos - lo))
    return out
