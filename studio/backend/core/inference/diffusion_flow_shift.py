# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rebuild the loaded scheduler at ComfyUI's static sigma shift (shipped schedulers differ, e.g.
Qwen-Image dynamic ~2.0 + terminal stretch vs 3.1). No torch/diffusers imports."""

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
