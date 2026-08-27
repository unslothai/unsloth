# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compact structured progress logs for image and video work."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Mapping

from loggers import get_logger

logger = get_logger(__name__)

_GENERATION_EVENT_NAMES = {
    "image": "image_generation_progress",
    "video": "video_generation_progress",
}
_LOAD_EVENT_NAMES = {
    "image": "diffusion_load_progress",
    "video": "video_load_progress",
}
_LOAD_ACTIVE_PHASES = frozenset({"downloading", "finalizing"})


@dataclass(frozen = True)
class _ProgressState:
    active: bool
    phase: str
    bucket: int
    step: int


@dataclass(frozen = True)
class _LoadProgressState:
    phase: str
    bucket: int


_state_lock = threading.Lock()
_last_state: dict[str, _ProgressState] = {}
_last_load_state: dict[str, _LoadProgressState] = {}


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _clamped_fraction(value: Any) -> float:
    try:
        fraction = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        fraction = 0.0
    if not math.isfinite(fraction):
        fraction = 0.0
    return max(0.0, min(fraction, 1.0))


def _fraction(progress: Mapping[str, Any], step: int, total: int) -> float:
    value = _clamped_fraction(progress.get("fraction", 0.0))
    return _clamped_fraction(step / total) if value <= 0.0 and total > 0 else value


def reset_media_generation_progress(media: str) -> None:
    """Clear progress state for a new job."""
    if media not in _GENERATION_EVENT_NAMES:
        raise ValueError(f"unknown media progress stream: {media}")
    with _state_lock:
        _last_state.pop(media, None)


def reset_media_load_progress(media: str) -> None:
    """Clear progress state for a new load."""
    if media not in _LOAD_EVENT_NAMES:
        raise ValueError(f"unknown media progress stream: {media}")
    with _state_lock:
        _last_load_state.pop(media, None)


def byte_fraction(done: Any, total: Any) -> float:
    """Return a safe byte ratio when the expected size may be unknown."""
    try:
        done_bytes = float(done or 0)
        total_bytes = float(total or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(done_bytes) or not math.isfinite(total_bytes) or total_bytes <= 0:
        return 0.0
    return done_bytes / total_bytes


def log_media_load_progress(media: str, phase: Any, fraction: Any) -> None:
    """Emit load phase changes and 10 percent milestones."""
    event = _LOAD_EVENT_NAMES.get(media)
    if event is None:
        raise ValueError(f"unknown media progress stream: {media}")

    phase_name = str(phase or "")
    if phase_name not in _LOAD_ACTIVE_PHASES:
        reset_media_load_progress(media)
        return

    bucket = min(10, int(_clamped_fraction(fraction) * 10))
    current = _LoadProgressState(phase_name, bucket)
    with _state_lock:
        previous = _last_load_state.get(media)
        if previous is not None and bucket < previous.bucket:
            return
        should_log = bool(
            previous is None or phase_name != previous.phase or bucket > previous.bucket
        )
        _last_load_state[media] = current

    if should_log:
        logger.info(event, phase = phase_name, percent = bucket * 10)


def log_media_generation_progress(media: str, progress: Mapping[str, Any]) -> None:
    """Emit live start, phase, and 10 percent milestones."""
    event = _GENERATION_EVENT_NAMES.get(media)
    if event is None:
        raise ValueError(f"unknown media progress stream: {media}")

    active = bool(progress.get("active", False))
    step = _int(progress.get("step", 0))
    total = _int(progress.get("total_steps", progress.get("total", 0)))
    fraction = _fraction(progress, step, total)
    bucket = min(10, int(fraction * 10))
    phase = str(progress.get("phase") or ("denoise" if active else ""))
    current = _ProgressState(active, phase, bucket, step)

    with _state_lock:
        previous = _last_state.get(media)
        restarted = bool(
            active and previous is not None and previous.active and step < previous.step
        )
        should_log = bool(
            active
            and (
                previous is None
                or not previous.active
                or restarted
                or phase != previous.phase
                or bucket > previous.bucket
            )
        )
        _last_state[media] = current

    if not should_log:
        return

    fields: dict[str, Any] = {
        "phase": phase,
        "percent": bucket * 10,
        "step": step,
        "total_steps": total,
    }
    eta = progress.get("eta_seconds")
    try:
        eta_value = float(eta) if eta is not None else None
    except (TypeError, ValueError, OverflowError):
        eta_value = None
    if eta_value is not None and math.isfinite(eta_value):
        fields["eta_seconds"] = round(max(0.0, eta_value), 1)
    logger.info(event, **fields)
