# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Plans the cost gate and the seam must not wave through unproved."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.inference.offload_cost_model import HostProfile  # noqa: E402
from core.inference.offload_planner import (  # noqa: E402
    Plan,
    PlanOptions,
    _device_reserve,
    _fit_fallback_placement,
    all_resident_bytes,
    plan_placement,
)
from test_offload_cost_gate import GIB, big_head_layout, dense_layout, gated  # noqa: E402


def test_a_knob_only_plan_with_an_unmodellable_fitter_declines():
    """``_cost_gate`` declines when llama.cpp's own placement cannot be modelled; the knob-only
    gate accepted the same case with both arms priced at zero, so a drafter was dropped on no
    comparison at all, whatever the draft-drop penalty."""
    layout = big_head_layout()
    draft = 3 * GIB
    budget = 5 * GIB
    opts = gated(
        host = HostProfile(threads = 6),
        overhead_bytes_per_device = 0,
        overhead_bytes_per_token = 0,
        draft_bytes = draft,
        draft_droppable = True,
        draft_drop_penalty_frac = 5.0,
    )
    assert (
        _fit_fallback_placement(
            layout,
            opts,
            budget - draft,
            8192,
            quantised = False,
            kv_bytes_floor = 0,
            kv_on_host = False,
        )
        is None
    ), "the fixture must reach the unmodellable branch"
    got = plan_placement(layout, [budget], 94 * GIB, 8192, opts = opts)
    assert got.declined_by_gate and not got.draft_dropped and not got.changed, got.reason
    assert "could not be modelled" in got.reason and "--fit on" in got.reason, got.reason


def test_a_knob_only_plan_with_a_host_cache_is_admitted_against_host_ram():
    """Under -nkvo the cache is host RAM. A plan that only dropped the draft skipped the one
    host-RAM refusal, so 9 GiB of host side was accepted on a box with 4 GiB."""
    layout = dense_layout(kv_gib_at_32k = 8.0)
    draft = 2 * GIB
    needed = all_resident_bytes(layout, 32768, kv_on_host = True)
    card = [needed + _device_reserve(PlanOptions(), 32768) + draft - GIB // 2]
    base = dict(
        host = HostProfile(threads = 6),
        draft_bytes = draft,
        draft_droppable = True,
        draft_drop_penalty_frac = 0.0,
        kv_on_host = True,
    )
    roomy = plan_placement(layout, card, 200 * GIB, 32768, opts = gated(**base))
    assert roomy.draft_dropped and not roomy.spills_anything, roomy.reason
    assert roomy.host_bytes > 8 * GIB

    tight = plan_placement(layout, card, 4 * GIB, 32768, opts = gated(**base))
    assert tight.declined_by_gate and not tight.draft_dropped and not tight.changed, tight.reason
    assert "host RAM" in tight.reason and "--fit on" in tight.reason, tight.reason

    # Without a host cache the host side is the embedding --fit on pays as well: nothing to refuse.
    no_nkvo = {**base, "kv_on_host": False}
    plain_card = [all_resident_bytes(layout, 32768) + _device_reserve(PlanOptions(), 32768)]
    plain_card = [plain_card[0] + draft - GIB // 2]
    plain = plan_placement(layout, plain_card, 4 * GIB, 32768, opts = gated(**no_nkvo))
    assert plain.draft_dropped, plain.reason


def test_an_over_reported_card_keeps_its_reserve(monkeypatch):
    """Free above total is clamped to total, but the usable budget is free less a reserve sized
    off the TOTAL. Taking min(usable, clamped free) gave the whole card to the plan and pinned
    it with --fit off; the reserve has to come off the clamped figure."""
    from core.inference.llama_cpp import LlamaCppBackend
    from test_offload_planner_seam import _Stub, _plan

    LlamaCppBackend._free_over_total_reported.clear()
    # A 12 GiB card's reserve, the case #12138 is: reporting 64 GiB free against that total.
    total = 12 * 1024
    reserve = 1536
    honest = _plan(_Stub(), free_mib = total, usable_mib = total - reserve, gpu_total_mib = {0: total})
    unreserved = _plan(_Stub(), free_mib = total, gpu_total_mib = {0: total})
    assert honest is not None and honest.spills_anything
    assert unreserved is not None
    assert honest.vram_bytes < unreserved.vram_bytes, "the fixture must see the reserve"

    over = _plan(
        _Stub(),
        free_mib = 64 * 1024,
        usable_mib = 64 * 1024 - reserve,
        gpu_total_mib = {0: total},
    )
    assert over is not None
    assert over.ot_patterns == honest.ot_patterns
    assert over.vram_bytes == honest.vram_bytes, "the reserve survives the clamp"


def test_a_discarded_inherited_split_does_not_suppress_the_plans_split(tmp_path, monkeypatch):
    """LLAMA_ARG_TENSOR_SPLIT riding with a non-layer LLAMA_ARG_SPLIT_MODE is popped from the
    child env, and the planner ignores it for the same reason; the launch still treated it as a
    user ratio in force and emitted no --tensor-split, so the child ran --fit off on neither."""
    from core.inference.llama_cpp import LlamaCppBackend
    from test_offload_planner_launch import _launch_with

    monkeypatch.setattr(
        LlamaCppBackend,
        "_pin_visible_gpu_order_for_split",
        staticmethod(lambda env: None),
    )
    plan = Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
        device_layer_counts = (30, 18),
    )
    monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "none")
    monkeypatch.setenv("LLAMA_ARG_TENSOR_SPLIT", "9,1")
    cmd, _b, _s = _launch_with(tmp_path, monkeypatch, plan, extra_args = ["--split-mode", "layer"])
    assert cmd.count("--tensor-split") == 1, cmd
    assert cmd[cmd.index("--tensor-split") + 1] == "30,18", cmd

    # An inherited ratio under a layer split DOES reach the child, so the plan still adds none.
    monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "layer")
    cmd, _b, _s = _launch_with(tmp_path, monkeypatch, plan)
    assert "--tensor-split" not in cmd, cmd
