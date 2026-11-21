# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static step skip with a measured per-step compute mask: the interval form stays identical, a mask replaces it
only at its own step count, and the auto plan carries the family's masks."""

from __future__ import annotations

import itertools
import types

import pytest
import torch

from core.inference import diffusion_cache as dcache
from core.inference import diffusion_step_skip as ss

# 30 steps, 15 computed: steps 0-5 and the last computed, runs of one or two skips between.
MASK_30 = "111111001001010010010100100101"


def test_interval_schedule_is_unchanged_without_a_matching_mask():
    other = {29: "1" * 29}
    for n, every, head, tail in itertools.product(
        range(0, 61), (0, 1, 2, 3, 4), (0.0, 0.2, 0.35), (0.0, 0.1, 0.25)
    ):
        want = ss.static_schedule(n, every = every, head = head, tail = tail)
        assert ss.static_schedule(n, every = every, head = head, tail = tail, masks = None) == want
        assert ss.static_schedule(n, every = every, head = head, tail = tail, masks = {}) == want
        if n != 29:
            assert ss.static_schedule(n, every = every, head = head, tail = tail, masks = other) == want


def test_a_mask_replaces_the_interval_at_its_own_step_count():
    assert len(MASK_30) == 30
    plan = ss.static_schedule(30, masks = {30: MASK_30})
    assert plan == tuple(c == "1" for c in MASK_30)
    assert plan != ss.static_schedule(30)
    # Another step count keeps the interval; the min_steps floor still applies.
    assert ss.static_schedule(31, masks = {30: MASK_30}) == ss.static_schedule(31)
    assert ss.static_schedule(30, min_steps = 40, masks = {30: MASK_30}) == ()
    # A bool tuple works too.
    assert ss.static_schedule(30, masks = {30: plan}) == plan


@pytest.mark.parametrize(
    "bad",
    [
        MASK_30[:-1],  # wrong length for 30 steps
        "0" + MASK_30[1:],  # step 0 skipped: taylor1 has nothing to extrapolate from
        "10" + MASK_30[2:],  # step 1 skipped
        MASK_30[:-1] + "0",  # last step skipped
        MASK_30[:10] + "x" + MASK_30[11:],
        "",
        None,
        12345,
    ],
)
def test_a_malformed_mask_falls_back_to_the_interval(bad):
    assert ss.static_schedule(30, masks = {30: bad}) == ss.static_schedule(30)


def test_auto_settings_carry_masks_unless_an_interval_is_forced_or_masks_are_off():
    plan = {"every": 3, "min_steps": 30, "masks": {30: MASK_30}}
    assert ss.auto_static_settings(plan, env = {})["masks"] == {30: MASK_30}
    assert "masks" not in ss.auto_static_settings(plan, env = {ss.ENV_EVERY: "2"})
    assert "masks" not in ss.auto_static_settings(plan, env = {ss.ENV_MASK: "0"})
    assert "masks" not in ss.auto_static_settings({"every": 3, "min_steps": 30}, env = {})


def test_explicit_head_or_tail_keeps_the_interval():
    # The docstring promise: an explicit UNSLOTH_STATIC_SKIP_* value wins over the measured mask (benchmarking).
    plan = {"every": 3, "min_steps": 30, "masks": {30: MASK_30}}
    for env in ({ss.ENV_HEAD: "0.5"}, {ss.ENV_TAIL: "0.3"}):
        knobs = ss.auto_static_settings(plan, env = env)
        assert "masks" not in knobs
        want = ss.static_schedule(30, every = 3, head = knobs["head"], tail = knobs["tail"])
        assert (
            ss.static_schedule(
                30,
                every = 3,
                head = knobs["head"],
                tail = knobs["tail"],
                min_steps = knobs["min_steps"],
                masks = knobs.get("masks"),
            )
            == want
        )
    # A malformed value is ignored (defaults kept), so the mask still applies.
    assert ss.auto_static_settings(plan, env = {ss.ENV_HEAD: "junk"})["masks"] == {30: MASK_30}


class _DiT:
    def __init__(self):
        self.calls = 0

    def forward(
        self,
        hidden_states = None,
        timestep = None,
        return_dict = True,
    ):
        self.calls += 1
        return (hidden_states * float(timestep.reshape(-1).max()),)


def _run(pipe, steps):
    outs = []
    for i in range(steps):
        t = torch.tensor([1.0 - i / steps])
        outs.append(
            pipe.transformer.forward(hidden_states = torch.ones(1, 4), timestep = t, return_dict = False)
        )
    return outs


def test_the_layer_follows_the_mask_and_reports_it():
    dit = _DiT()
    pipe = types.SimpleNamespace(transformer = dit)
    knobs = ss.auto_static_settings({"every": 3, "min_steps": 30, "masks": {30: MASK_30}}, env = {})
    assert ss.install_static_step_skip(pipe, settings = knobs) == dcache.TC_STATIC
    ss.reset_static_step_skip(pipe, 30)
    outs = _run(pipe, 30)
    assert dit.calls == MASK_30.count("1")
    stats = ss.static_skip_stats(pipe)
    assert stats["stats"] == {
        "calls": 30,
        "computed": MASK_30.count("1"),
        "skipped": MASK_30.count("0"),
    }
    assert stats["masked"] is True and stats["mask_steps"] == [30]
    for i, (out,) in enumerate(outs):
        # Outputs are linear in t, so taylor1 reproduces every skipped step.
        assert torch.allclose(out, torch.full((1, 4), 1.0 - i / 30), atol = 1e-5)
    # Another step count runs the interval and says so.
    ss.reset_static_step_skip(pipe, 31)
    dit.calls = 0
    _run(pipe, 31)
    assert dit.calls == sum(ss.static_schedule(31, every = 3))
    assert ss.static_skip_stats(pipe)["masked"] is False


def test_reinstall_replaces_or_drops_the_masks():
    dit = _DiT()
    pipe = types.SimpleNamespace(transformer = dit)
    ss.install_static_step_skip(pipe, settings = {"every": 3, "masks": {30: MASK_30}})
    ss.install_static_step_skip(pipe, settings = {"every": 3})
    ss.reset_static_step_skip(pipe, 30)
    _run(pipe, 30)
    assert dit.calls == sum(ss.static_schedule(30, every = 3))


@pytest.fixture
def probe_tables(monkeypatch):
    monkeypatch.setattr(
        dcache, "AUTO_STATIC_SKIP", {"probe/model": {"default": 2, "max": 3, "steps": 30}}
    )
    monkeypatch.setattr(dcache, "AUTO_STATIC_SKIP_MASKS", {"probe/model": {"max": {30: MASK_30}}})


def test_the_auto_plan_carries_the_max_tier_masks(probe_tables):
    plan = dcache.auto_static_skip_plan(("probe/model",), "max", 30, env = {})
    assert plan == {"every": 3, "min_steps": 30, "masks": {30: MASK_30}}
    # Default tier still never skips automatically; the kill switch still wins.
    assert dcache.auto_static_skip_plan(("probe/model",), "default", 30, env = {}) is None
    assert (
        dcache.auto_static_skip_plan(
            ("probe/model",), "max", 30, env = {dcache.ENV_AUTO_STEP_SKIP: "0"}
        )
        is None
    )


def test_shipped_masks_are_well_formed_and_match_their_interval_rows():
    for repo, tiers in dcache.AUTO_STATIC_SKIP_MASKS.items():
        entry = dcache.AUTO_STATIC_SKIP.get(repo)
        assert entry is not None, repo
        for tier, masks in tiers.items():
            assert tier in ("default", "max"), (repo, tier)
            for steps, mask in masks.items():
                assert steps == entry["steps"], (repo, tier, steps)
                assert ss.parse_mask(mask), (repo, tier, mask)
                interval = ss.static_schedule(steps, every = entry.get(tier) or entry["max"])
                # A shipped mask never computes more steps than the interval it replaces.
                assert mask.count("1") <= sum(interval), (repo, tier)


def test_wan22_5b_max_tier_runs_the_measured_mask_at_50_steps():
    repo = "wan-ai/wan2.2-ti2v-5b-diffusers"
    masks = dcache.AUTO_STATIC_SKIP_MASKS[repo]["max"]
    mask = masks[50]
    # Same compute budget as the every-3 interval it replaces, so the speedup is unchanged.
    assert mask.count("1") == sum(ss.static_schedule(50, every = 3)) == 27
    plan = dcache.auto_static_skip_plan((repo,), "max", 50, env = {})
    assert plan["masks"] == {50: mask}
    settings = ss.auto_static_settings(plan, env = {})
    assert ss.static_schedule(
        50, every = settings["every"], masks = settings.get("masks")
    ) == ss.parse_mask(mask)
    # Other step counts keep the interval.
    assert ss.static_schedule(40, every = 3, masks = settings.get("masks")) == ss.static_schedule(
        40, every = 3
    )
