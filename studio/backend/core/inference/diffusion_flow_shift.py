# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static sigma shift a family samples with in ComfyUI, applied to the loaded scheduler. No
torch/diffusers imports: it only rebuilds ``pipe.scheduler`` from its own config.

Several shipped schedulers shift sigmas differently from ComfyUI's defaults for the same model:
Qwen-Image resolves a resolution-dependent exponential shift (about 2.0 at 1024 px) and stretches
the tail to 0.02 where ComfyUI samples a constant 3.1, Z-Image base ships 6.0 against 3.0, Wan2.2
ships 5.0 / 3.0 against 8 / 5 and HunyuanVideo-1.5 5.0 / 9.0 against 7. A flow-matching shift ``s``
maps sigma to ``s * sigma / (1 + (s - 1) * sigma)`` in both, so setting the static shift (and
dropping the dynamic shift and terminal stretch) reproduces ComfyUI's schedule.
"""

from __future__ import annotations

from typing import Any, Optional


def flow_shift_overrides(config: Any, shift: float) -> Optional[dict]:
    """Scheduler config overrides that sample at static ``shift``, or None when the scheduler has
    no flow shift Studio knows how to set."""
    try:
        keys = set(config.keys())
    except Exception:  # noqa: BLE001 - an exotic config just keeps its scheduler
        return None
    if "flow_shift" in keys and config.get("use_flow_sigmas", True):
        # UniPC / DPM-Solver in flow mode (Wan).
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
    if shift is None:
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
