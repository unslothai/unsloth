# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cost gate's fitter arm is the --fit on launch a decline really makes, at its own -c."""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.offload_planner import (  # noqa: E402
    ContextPolicy,
    HostProfile,
    Placement,
    PlanOptions,
    _fit_fallback_placement,
    _score_arms,
    _usable_vram,
    plan_placement,
)
from test_offload_cost_gate import GIB, dense_layout  # noqa: E402
from test_offload_planner_seam import _captured_opts, _load_model_source, _Stub  # noqa: E402

AUTO_CAP = 8192  # llama_cpp._AUTO_OFFLOAD_CTX
NATIVE = 131072


def _layout():
    return replace(dense_layout(), n_ctx_train = NATIVE)


def _kv_at(ctx: int, slots: int) -> int:
    return int(_layout().kv_bytes_per_token_f16 * ctx * slots)


def _opts(card_gib: float, **kw) -> PlanOptions:
    return PlanOptions(
        require_cost_win = True,
        host = HostProfile(threads = 6),
        context_policy = ContextPolicy.FIT_ONLY,
        kv_bytes_at = _kv_at,
        min_ctx = AUTO_CAP,
        **kw,
    )


def _auto(card_gib: float, **kw):
    """The Auto-context cell: -c capped at 8192 on the coarse fit, the planner asked at native."""
    opts = _opts(
        card_gib,
        fitter_n_ctx = AUTO_CAP,
        fitter_vram_bytes_per_device = (int(card_gib * GIB),),
        **kw,
    )
    return plan_placement(_layout(), [int(card_gib * GIB)], 94 * GIB, NATIVE, opts = opts), opts


def _real_fallback_ms(card_gib: float, opts: PlanOptions) -> float:
    """What --fit on costs at the -c 8192 it launches with: an explicit -c is never shrunk by
    common/fit.cpp (n_ctx_auto is ``cparams->n_ctx == 0``, fit_b11541.cpp:198), so it spills
    whole layers at exactly that context."""
    plain = replace(opts, fitter_n_ctx = 0, fitter_vram_bytes_per_device = ())
    fallback = _fit_fallback_placement(
        _layout(),
        plain,
        _usable_vram([int(card_gib * GIB)], plain, AUTO_CAP),
        AUTO_CAP,
        quantised = False,
        kv_bytes_floor = _kv_at(AUTO_CAP, 1),
        kv_on_host = False,
    )
    assert fallback is not None
    n_prompt = min(opts.workload_prompt_tokens, AUTO_CAP)
    return _score_arms(Placement(host_groups = []), fallback, opts, n_prompt, 1, 1)[1]


@pytest.mark.parametrize("card_gib", [11, 12, 13, 14, 15, 16])
def test_an_accepted_auto_context_plan_beats_the_launch_a_decline_really_makes(card_gib):
    """Priced against a fitter at the context the plan buys, a 16 GiB card took a 36864 plan
    costing 1.37x the -c 8192 --fit on it replaced, through a gate that asks for a 10% win."""
    plan, opts = _auto(card_gib)
    assert plan.spills_anything and not plan.declined_by_gate, plan.reason
    real = _real_fallback_ms(card_gib, opts)
    assert plan.predicted_fit_request_ms == pytest.approx(real)
    assert plan.predicted_request_ms <= real * (1.0 - opts.min_penalty_reduction), (
        plan.n_ctx,
        plan.predicted_request_ms,
        real,
    )
    # The context still comes back above the cap: the ladder walks down, it does not freeze.
    assert AUTO_CAP < plan.n_ctx < NATIVE


@pytest.mark.parametrize("card_gib", [20, 24])
def test_a_resident_fallback_is_not_traded_for_a_spill(card_gib):
    """At -c 8192 the whole model fits, so --fit on moves nothing; a spill bought only for context
    costs generation speed, which the PR ranks above context."""
    plan, _ = _auto(card_gib)
    assert not plan.spills_anything and not plan.declined_by_gate, plan.reason
    assert plan.n_ctx > AUTO_CAP


@pytest.mark.parametrize("card_gib", [12, 16, 20, 30])
def test_an_unset_fitter_context_is_priced_as_before(card_gib):
    """A caller that does not say what -c a decline launches keeps the arm the gate always priced."""
    vram = [int(card_gib * GIB)]
    before = plan_placement(_layout(), vram, 94 * GIB, NATIVE, opts = _opts(card_gib))
    after = plan_placement(
        _layout(),
        vram,
        94 * GIB,
        NATIVE,
        opts = _opts(card_gib, fitter_n_ctx = 0, fitter_vram_bytes_per_device = tuple(vram)),
    )
    assert after == before


@pytest.mark.parametrize("card_gib", [12, 16, 20, 30])
@pytest.mark.parametrize("n_ctx", [8192, 32768, NATIVE])
def test_an_explicit_context_request_is_unchanged(card_gib, n_ctx):
    """An explicit context is never shrunk (the seam asks FIT_ONLY only of the Auto restore) and
    --fit-ctx floors the fallback at it, so the -c a decline emits IS the plan's context."""
    vram = [int(card_gib * GIB)]
    never = dict(context_policy = ContextPolicy.NEVER_REDUCE)
    before = plan_placement(
        _layout(), vram, 94 * GIB, n_ctx, opts = replace(_opts(card_gib), **never)
    )
    after = plan_placement(
        _layout(),
        vram,
        94 * GIB,
        n_ctx,
        opts = replace(
            _opts(card_gib, fitter_n_ctx = n_ctx, fitter_vram_bytes_per_device = tuple(vram)),
            **never,
        ),
    )
    assert after == before


@pytest.mark.parametrize("drop", ["kv_bytes_at", "fitter_vram_bytes_per_device"])
def test_an_unpriceable_fitter_context_declines_rather_than_guesses(drop):
    """Without the cache map or the device budgets the arm cannot be moved to -c 8192, and the
    plan's own context is the wrong one, so no spill above the cap is accepted; the ladder reaches
    the cap, where the two contexts agree and the old arm is the right one."""
    _, opts = _auto(16)
    opts = replace(opts, **{drop: None if drop == "kv_bytes_at" else ()})
    plan = plan_placement(_layout(), [16 * GIB], 94 * GIB, NATIVE, opts = opts)
    assert plan.declined_by_gate or plan.n_ctx <= AUTO_CAP, (plan.n_ctx, plan.reason)


def test_the_seam_hands_the_emitted_context_and_budgets_to_the_planner(monkeypatch):
    opts, seen = _captured_opts(monkeypatch, _Stub(), free_mib = 14 * 1024, fitter_n_ctx = AUTO_CAP)
    assert opts.fitter_n_ctx == AUTO_CAP
    assert opts.fitter_vram_bytes_per_device
    assert len(opts.fitter_vram_bytes_per_device) == 1


def test_a_snapshot_without_it_plans_at_its_own_context(monkeypatch):
    opts, _ = _captured_opts(monkeypatch, _Stub(), free_mib = 14 * 1024)
    assert opts.fitter_n_ctx == 0


def test_load_model_records_the_context_after_it_is_emitted():
    """Taken after ``-c`` is appended (the Auto cap and the Metal floor have both run) and before
    the planner is asked, so it is the context the declined launch actually carries."""
    src = _load_model_source()
    emit = src.index('cmd.extend(["-c", str(effective_ctx)])')
    record = src.index('_spill_inputs["fitter_n_ctx"] = int(max(0, effective_ctx))')
    ask = src.index("_spill = self._planned_tensor_spill(")
    assert emit < record < ask
    between = src[record:ask]
    assert not re.search(r"\beffective_ctx\s*=[^=]", between)
