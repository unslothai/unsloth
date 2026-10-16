# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A per-device miss is a reason to keep searching, and the MoE context veto needs a fitter."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.inference.offload_cost_model import HostProfile  # noqa: E402
from core.inference.offload_planner import (  # noqa: E402
    ContextPolicy,
    PlanOptions,
    _Knobs,
    _outside_layout_bytes,
    _per_device_shortfall,
    plan_placement,
)
from test_offload_cost_gate import GIB, dense_layout, gated, moe_layout  # noqa: E402


def _long_moe():
    return replace(moe_layout(), kv_bytes_per_token_f16 = int(4.0 * GIB) // 32768, n_ctx_train = 65536)


def test_the_moe_context_veto_stands_down_when_no_fitter_can_replace_the_load():
    """A user -ts across two cards: common/fit.cpp aborts on it, so the veto's '--fit on at the
    context asked for' is the launch as typed. _cost_gate and _knob_only_gate already skip the
    ranking there; the context-search veto has to as well, or the only placement that fits is
    refused."""
    layout = _long_moe()
    devices = [10 * GIB, 10 * GIB]

    def run(**kw):
        return plan_placement(
            layout,
            devices,
            15 * GIB,
            65536,
            opts = gated(host = HostProfile(threads = 8), context_policy = ContextPolicy.FIT_ONLY, **kw),
        )

    ranked = run()
    assert ranked.veto and ranked.declined_by_gate and not ranked.changed, ranked.reason

    unranked = run(rank_against_fitter = False)
    assert unranked.changed and unranked.priced, unranked.reason
    assert not unranked.declined_by_gate and not unranked.veto, unranked.reason
    assert 0 < unranked.n_ctx < 65536 and unranked.spilled_blocks, unranked.reason


def _fits_on_every_card(layout, opts, devices, plan):
    moved = {}
    for i in plan.spilled_blocks:
        moved[i] = layout.blocks[i].spillable_bytes
    return (
        _per_device_shortfall(
            layout,
            opts,
            int(plan.n_ctx),
            moved,
            plan.spilled_lm_head,
            devices,
            quantised = plan.cache_type_k is not None,
            kv_bytes_floor = 0,
            split_weights_per_device = devices,
            kv_layer_weights = (),
            extra_on_device0 = _outside_layout_bytes(opts, _Knobs(max(1, opts.n_parallel))),
            n_seq = max(1, opts.n_parallel),
        )
        is None
    )


def test_a_per_device_miss_at_the_asked_context_still_searches_a_smaller_one():
    """Pooled VRAM holds the load at 32768, card 0's row share does not. FIT_ONLY may shrink, and
    the single-card path does (an unplaceable context is None there), so the multi-card path must
    not end the search on the first per-device miss."""
    layout = dense_layout()
    devices = [4 * GIB, 4 * GIB]
    opts = PlanOptions(context_policy = ContextPolicy.FIT_ONLY)
    plan = plan_placement(layout, devices, 94 * GIB, 32768, opts = opts)
    assert plan.changed and plan.priced, plan.reason
    assert 0 < plan.n_ctx < 32768, plan.reason


def test_a_per_device_miss_in_f16_still_tries_the_quantised_cache():
    layout = dense_layout(kv_gib_at_32k = 4.0)
    devices = [int(4.8 * GIB), int(5.4 * GIB)]
    plan = plan_placement(layout, devices, 94 * GIB, 32768, opts = PlanOptions(allow_kv_quant = True))
    assert plan.changed and plan.priced, plan.reason
    assert plan.cache_type_k == "q8_0" and plan.n_ctx == 32768, plan.reason


def test_a_per_device_miss_on_one_context_rung_still_walks_the_lower_ones():
    """The pooled ceiling is an upper bound, not a per-device fit: the first rung under it can
    overflow card 0 while the next one down places."""
    layout = replace(moe_layout(), kv_bytes_per_token_f16 = int(2.0 * GIB) // 32768)
    devices = [int(2.25 * GIB), int(2.25 * GIB)]
    opts = PlanOptions(context_policy = ContextPolicy.FIT_ONLY)
    plan = plan_placement(layout, devices, 200 * GIB, 32768, opts = opts)
    assert plan.changed and plan.priced, plan.reason
    assert 0 < plan.n_ctx < 7936, plan.reason


def test_the_resident_shortcut_is_checked_card_by_card():
    """PREFER_RESIDENT sized its context on the pooled budget; on two cards the default split can
    still overflow one of them, and a resident plan pins -ngl -1 --fit off."""
    layout = dense_layout(n_blocks = 16)
    devices = [int(4.6 * GIB), int(4.6 * GIB)]
    for extra in (0, 1 * GIB):
        opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT, extra_resident_bytes = extra)
        plan = plan_placement(layout, devices, 94 * GIB, 32768, opts = opts)
        if plan.changed:
            assert _fits_on_every_card(layout, opts, devices, plan), plan.reason


def test_the_resident_shortcut_is_unchanged_on_one_card():
    layout = dense_layout(n_blocks = 16)
    opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT)
    plan = plan_placement(layout, [7 * GIB], 94 * GIB, 32768, opts = opts)
    assert plan.changed and not plan.spilled_blocks and 0 < plan.n_ctx < 32768, plan.reason
    assert "resident" in plan.reason
