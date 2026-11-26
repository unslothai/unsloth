# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seeds the noise a scheduler draws inside ``step`` when the pipeline does not pass its generator.

diffusers ``HiDreamImagePipeline`` calls ``self.scheduler.step(noise_pred, t, latents, return_dict=False)`` with no
generator, and the ``FlowMatchLCMScheduler`` that HiDream-I1-Fast and -Dev ship draws fresh noise every step from
whatever generator it gets. With none it uses the global RNG, so the same prompt and seed render a different image
each time, and a batch member never matches its own seed's solo render. This hands the call's generator (one per
image in a batch) to ``step`` whenever the pipeline leaves it out."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Iterator, Optional


def _generator_slot(step: Any) -> Optional[int]:
    """Positional index of ``step``'s ``generator`` parameter (bound method, so ``self`` excluded); None if absent."""
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
    """For the duration of one pipeline call, fill ``scheduler.step``'s missing ``generator`` with ``generator``.

    A generator the pipeline passes itself is kept. A per-image list is only handed over when the sample has one row
    per generator, so the noise row i draws comes from image i's own seed, exactly as its solo render draws it."""
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
