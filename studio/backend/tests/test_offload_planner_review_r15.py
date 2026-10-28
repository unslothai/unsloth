# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A user tensor split over several devices leaves no --fit on arm to rank a plan against."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.offload_cost_model import HostProfile  # noqa: E402
from core.inference.offload_planner import plan_placement  # noqa: E402
from test_offload_cost_gate import GIB, dense_layout, gated  # noqa: E402
from test_offload_planner_seam import _plan, _Stub  # noqa: E402


def test_a_load_the_fitter_cannot_replace_keeps_the_plan():
    """common/fit.cpp throws "tensor_split already set by user" on more than one device, so the
    --fit on fallback is the launch as typed. Ranked against a fitter that would have moved
    layers, the gate declined the only placement that fits."""
    layout = dense_layout()
    budget = [14848 * 1024 * 1024]
    ranked = plan_placement(
        layout, budget, 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6))
    )
    assert ranked.declined_by_gate and not ranked.spilled_blocks, ranked.reason

    unranked = plan_placement(
        layout,
        budget,
        94 * GIB,
        32768,
        opts = gated(host = HostProfile(threads = 6), rank_against_fitter = False),
    )
    assert unranked.spilled_blocks and not unranked.declined_by_gate, unranked.reason

    # The host-RAM refusal still runs: a spill the host cannot hold is declined either way.
    starved = plan_placement(
        layout,
        budget,
        2 * GIB,
        32768,
        opts = gated(host = HostProfile(threads = 6), rank_against_fitter = False),
    )
    assert not starved.spilled_blocks, starved.reason


@pytest.mark.parametrize(
    "gpus,extra_args,env,ranks",
    [
        ("two", ["--tensor-split", "1,1"], None, False),
        ("two", ["-ts", "3/1"], None, False),
        ("two", None, {"LLAMA_ARG_TENSOR_SPLIT": "1,1"}, False),
        ("two", None, None, True),
        ("one", ["-ts", "1"], None, True),
    ],
    ids = ["argv", "slash", "env", "no_split", "one_card"],
)
def test_the_launcher_says_when_the_fitter_cannot_replace(
    monkeypatch, gpus, extra_args, env, ranks
):
    from core.inference import offload_planner

    seen = []
    real = offload_planner.plan_placement

    def spy(*args, **kwargs):
        seen.append(kwargs["opts"].rank_against_fitter)
        return real(*args, **kwargs)

    monkeypatch.setattr(offload_planner, "plan_placement", spy)
    cards = [(0, 14 * 1024), (1, 14 * 1024)] if gpus == "two" else [(0, 14 * 1024)]
    _plan(_Stub(), gpus = cards, extra_args = extra_args, env = env)
    assert seen and set(seen) == {ranks}, seen
