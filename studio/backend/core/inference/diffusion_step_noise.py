# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pass the call's generator to ``scheduler.step`` when the pipeline omits it (diffusers HiDreamImagePipeline +
FlowMatchLCMScheduler would otherwise draw per-step noise from the global RNG, ignoring the seed)."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Iterator, Optional


def _generator_slot(step: Any) -> Optional[int]:
    """Positional index of ``generator`` in bound ``step``; None if absent."""
    try:
        params = list(inspect.signature(step).parameters.values())
    except (TypeError, ValueError):
        return None
    for index, param in enumerate(params):
        if param.name == "generator":
            return index
    return None


def _sample_rows(args: tuple, kwargs: dict) -> Optional[int]:
    sample = kwargs.get("sample", args[2] if len(args) > 2 else None)
    shape = getattr(sample, "shape", None)
    try:
        return int(shape[0]) if shape is not None and len(shape) else None
    except (TypeError, ValueError, IndexError):
        return None


@contextmanager
def seeded_scheduler_step(pipe: Any, generator: Any) -> Iterator[None]:
    """Fill a missing ``generator`` in ``scheduler.step`` for one call; a list is passed only if it has one per sample row."""
    scheduler = getattr(pipe, "scheduler", None)
    step = getattr(scheduler, "step", None)
    slot = _generator_slot(step) if callable(step) else None
    try:
        own = vars(scheduler)
    except TypeError:
        own = None
    if generator is None or slot is None or own is None:
        yield
        return
    if isinstance(generator, (list, tuple)) and len(generator) == 1:
        generator = generator[0]

    def step_with_generator(*args, **kwargs):
        if kwargs.get("generator") is None and len(args) <= slot:
            chosen = generator
            if isinstance(chosen, (list, tuple)):
                rows = _sample_rows(args, kwargs)
                chosen = list(chosen) if rows == len(chosen) else None
            if chosen is not None:
                kwargs["generator"] = chosen
        return step(*args, **kwargs)

    had_own = "step" in own
    previous = own.get("step")
    scheduler.step = step_with_generator
    try:
        yield
    finally:
        if had_own:
            scheduler.step = previous
        else:
            try:
                del scheduler.step
            except AttributeError:
                pass
