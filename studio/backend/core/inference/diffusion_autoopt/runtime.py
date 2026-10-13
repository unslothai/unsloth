# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Call-time checks against the pipeline that actually runs a request.

Load-time eligibility is judged on the base pipe, but a request may run a DIFFERENT pipeline over the same denoiser:
an image workflow built with ``from_pipe`` (FLUX.1 img2img / inpaint), the video I2V class swap, LTX-2.5's condition
pipe. First-Block-Cache hooks live on the shared denoiser and need the RUNNER to open ``cache_context``; FLUX.1's
``FluxPipeline`` does, ``FluxImg2ImgPipeline`` / ``FluxInpaintPipeline`` (diffusers 0.41) do not, so an engaged cache
raised "No cache context is set" on the first img2img step. These helpers answer for the runner.
"""

from __future__ import annotations

from typing import Any, Optional

# Multi-call renders (a request that calls its pipeline more than once, e.g. LTX-2.5's two stages) use the
# family-agnostic helpers of diffusion_multistage; they are the runtime layer's multi-runner primitives, re-exported
# here so the optimizer has one runtime entry point instead of a second copy.
from ..diffusion_multistage import encode_once, hold_denoiser_between_calls  # noqa: F401


def runner_opens_cache_context(runner: Any) -> bool:
    from ..diffusion_cache import _pipeline_opens_cache_context  # noqa: PLC0415
    return _pipeline_opens_cache_context(runner)


def step_cache_runner_reason(base: Any, runner: Any) -> Optional[str]:
    """Why the step cache must stay off for a call that runs ``runner`` over ``base``'s denoiser, or None."""
    if runner is None or runner is base:
        return None
    if not shares_denoiser(base, runner):
        return None
    if runner_opens_cache_context(runner):
        return None
    return (
        f"{type(runner).__name__} opens no cache_context, so the step cache stays off for this call"
    )


def auto_cache_steps(base: Any, runner: Any, steps: int) -> tuple[int, Optional[str]]:
    """The step count an AUTO cache toggle should see for this call: 0 (disengage) when the runner cannot host the
    cache, else ``steps``. Returns ``(steps, reason)``."""
    reason = step_cache_runner_reason(base, runner)
    return (0, reason) if reason else (int(steps), None)


def shares_denoiser(base: Any, runner: Any) -> bool:
    """Whether ``runner`` drives a denoiser module of ``base`` (from_pipe workflows, class swaps and multi-stage
    runners built from ``base``'s components do); the same denoiser resolution diffusion_multistage holds."""
    if runner is None or runner is base:
        return True
    from ..diffusion_multistage import _denoisers  # noqa: PLC0415

    mine = _denoisers(base)
    return bool(mine) and any(m is n for m in mine for n in _denoisers(runner))
