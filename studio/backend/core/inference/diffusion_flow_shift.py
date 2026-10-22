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


def valid_sample_sigmas(raw: Any) -> Optional[tuple[float, ...]]:
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


# Not pipe.config: on the pinned diffusers an extra key fails DiffusionPipeline.components (group offload, from_pipe).
SAMPLE_SIGMAS_ATTR = "_unsloth_sample_sigmas"


def pipe_sample_sigmas(pipe: Any) -> Optional[tuple[float, ...]]:
    """The grid the loaded pipeline carries (its own config on diffusers with #14950, else installed), or None."""
    config = getattr(pipe, "config", None)
    try:
        raw = config.get(SAMPLE_SIGMAS_KEY) if config is not None else None
    except Exception:  # noqa: BLE001 - an exotic config carries no grid
        raw = None
    return valid_sample_sigmas(raw if raw is not None else getattr(pipe, SAMPLE_SIGMAS_ATTR, None))


def install_sample_sigmas(
    pipe: Any,
    raw: Any,
    logger: Any = None,
) -> Optional[tuple[float, ...]]:
    """Carry model_index.json ``sample_sigmas`` (dropped by the pinned diffusers) on ``pipe``; None without a grid."""
    grid = pipe_sample_sigmas(pipe)
    if grid is not None or raw is None:
        return grid
    grid = valid_sample_sigmas(raw)
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


def sd_cpp_sample_sigmas(grid: tuple[float, ...], steps: int) -> list[float]:
    """sd.cpp samples custom sigmas verbatim (no shift, steps = len - 1), so it needs the terminal 0 itself."""
    return [*sample_sigmas_for_steps(grid, steps), 0.0]


def sample_sigmas_for_steps(grid: tuple[float, ...], steps: int) -> list[float]:
    """The grid at its own length, else resampled along it (Studio's own: the card evaluates only the 8-step grid)."""
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


# Per-request shift (ComfyUI comfy_extras/nodes_model_advanced.py): one static shift on a per-generation scheduler copy.
# ModelSamplingSD3 / ModelSamplingAuraFlow: sigma = shift * t / (1 + (shift - 1) * t).
SHIFT_NODE_SD3 = "sd3"
# ModelSamplingFlux: mu = linear in latent tokens between (256, base_shift) and (4096, max_shift),
# sigma = e^mu / (e^mu + 1/t - 1), the same curve as the static shift e^mu (``flux_mu_shift``).
SHIFT_NODE_FLUX = "flux"
SHIFT_NODES = (SHIFT_NODE_SD3, SHIFT_NODE_FLUX)
COMFY_FLUX_MAX_SHIFT = 1.15
COMFY_FLUX_BASE_SHIFT = 0.5
_COMFY_FLUX_X1 = 256
_COMFY_FLUX_X2 = 4096
# The nodes' widget bounds (min 0.0, max 100.0); a static shift of 0 collapses every sigma to 0.
SHIFT_MAX = 100.0


def comfy_model_sampling_flux_mu(
    max_shift: float, base_shift: float, width: int, height: int
) -> float:
    """ModelSamplingFlux.patch: the ``shift`` (mu) it sets for an image of ``width`` x ``height``."""
    mm = (max_shift - base_shift) / (_COMFY_FLUX_X2 - _COMFY_FLUX_X1)
    b = base_shift - mm * _COMFY_FLUX_X1
    return (width * height / (8 * 8 * 2 * 2)) * mm + b


def requested_flow_shift(
    node: Optional[str],
    *,
    shift: Optional[float] = None,
    base_shift: Optional[float] = None,
    max_shift: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    defaults: Optional[dict] = None,
) -> Optional[float]:
    """The static flow shift a request asks for, or None when it sets none (today's schedule).

    ``defaults`` (``shift_control``'s) fills the base / max shift a request leaves out, so changing one
    starts from what the load resolved. Raises ValueError for a field this family's node does not take."""
    if shift is None and base_shift is None and max_shift is None:
        return None
    if node not in SHIFT_NODES:
        raise ValueError("Shift is supported for flow-matching image models only.")
    if node == SHIFT_NODE_SD3:
        if base_shift is not None or max_shift is not None:
            raise ValueError(
                "This model takes a single Shift value (ComfyUI ModelSamplingSD3), not base / max shift."
            )
        value = float(shift)
        if not (math.isfinite(value) and 0.0 < value <= SHIFT_MAX):
            raise ValueError(f"Shift must be above 0 and at most {SHIFT_MAX:g}.")
        return value
    if shift is not None:
        raise ValueError(
            "This model takes base / max shift (ComfyUI ModelSamplingFlux), not a single Shift value."
        )
    defaults = defaults or {}
    hi = float(defaults.get("max_shift", COMFY_FLUX_MAX_SHIFT) if max_shift is None else max_shift)
    lo = float(
        defaults.get("base_shift", COMFY_FLUX_BASE_SHIFT) if base_shift is None else base_shift
    )
    for value in (hi, lo):
        if not (math.isfinite(value) and 0.0 <= value <= SHIFT_MAX):
            raise ValueError(f"Base and max shift must be between 0 and {SHIFT_MAX:g}.")
    if not width or not height:
        raise ValueError("Base / max shift needs the output width and height.")
    return flux_mu_shift(comfy_model_sampling_flux_mu(hi, lo, int(width), int(height)))


def scheduler_with_flow_shift(scheduler: Any, shift: float) -> Any:
    """A copy of ``scheduler`` sampling at static ``shift`` (dynamic shifting and terminal stretch off), or
    ``scheduler`` itself when it already samples there (the resolved value: no rebuild, identical render).

    Raises ValueError when the scheduler has no flow shift Studio can set."""
    config = getattr(scheduler, "config", None)
    overrides = flow_shift_overrides(config, shift) if config is not None else None
    if not overrides:
        raise ValueError("The loaded model's scheduler has no flow shift to override.")
    if all(config.get(k) == v for k, v in overrides.items()):
        return scheduler
    return type(scheduler).from_config(config, **overrides)


def shift_control(node: Optional[str], scheduler: Any) -> Optional[dict]:
    """What the Shift control offers on this load, or None to hide it: the node kind plus the
    defaults to seed it with (``shift`` None when the schedule is resolution-dependent)."""
    if node not in SHIFT_NODES:
        return None
    config = getattr(scheduler, "config", None)
    if config is None or not flow_shift_overrides(config, 1.0):
        return None
    if node == SHIFT_NODE_FLUX:
        base, top = COMFY_FLUX_BASE_SHIFT, COMFY_FLUX_MAX_SHIFT
        try:
            if not config.get("use_dynamic_shifting", False) and "flow_shift" not in config.keys():
                # The load resolved a fixed mu (ComfyUI's fixed-mu templates, ``flux_mu_shift``): base = max = mu
                # reproduces it at every size.
                static = float(config.get("shift"))
                if static > 0:
                    base = top = math.log(static)
            else:
                base = float(config.get("base_shift", base))
                top = float(config.get("max_shift", top))
        except (TypeError, ValueError):
            base, top = COMFY_FLUX_BASE_SHIFT, COMFY_FLUX_MAX_SHIFT
        return {"node": node, "base_shift": base, "max_shift": top}
    current: Optional[float] = None
    try:
        if "flow_shift" in config.keys():
            current = float(config.get("flow_shift"))
        elif not config.get("use_dynamic_shifting", False):
            current = float(config.get("shift"))
    except (TypeError, ValueError):
        current = None
    return {"node": node, "shift": current}
