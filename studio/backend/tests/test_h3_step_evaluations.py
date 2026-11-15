# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A requested MiniMax-H3 step is one denoiser evaluation on the real diffusers scheduler."""

from __future__ import annotations

import pytest

from core.inference import video_minimax_h3 as h3


def _scheduler():
    pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "MiniMaxH3Scheduler", None)
    if cls is None:
        pytest.skip("this diffusers has no MiniMaxH3Scheduler")
    return cls(shift = 12.0)


@pytest.mark.parametrize("steps", [1, 2, 8, 20, 30, 50])
def test_requested_steps_are_model_evaluations(steps):
    sched = _scheduler()
    sched.set_timesteps(h3.h3_num_inference_steps(steps))
    assert len(sched.timesteps) == steps
    assert float(sched.sigmas[-1]) == 0.0 and float(sched.sigmas[0]) == 1.0


def test_kill_switch_keeps_the_grid_point_count(monkeypatch):
    monkeypatch.setenv(h3.H3_STEP_EVALUATIONS_ENV, "0")
    sched = _scheduler()
    sched.set_timesteps(h3.h3_num_inference_steps(20))
    assert len(sched.timesteps) == 19
