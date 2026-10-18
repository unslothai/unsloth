# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fitter arm carries the draft the declined launch really carries, and a one-card plan is
held to the same CUDA order rule as a split."""

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
    Plan,
    PlanOptions,
    _fit_fallback_placement,
    _outside_layout_bytes,
    _score_arms,
    _usable_vram,
    plan_placement,
)
from test_offload_cost_gate import GIB, dense_layout  # noqa: E402
from test_offload_planner_seam import _captured_opts, _load_model_source, _Stub  # noqa: E402

MIB = 1024 * 1024
AUTO_CAP = 8192
NATIVE = 131072


def _layout():
    return replace(dense_layout(), n_ctx_train = NATIVE)


def _kv_at(ctx: int, slots: int) -> int:
    return int(_layout().kv_bytes_per_token_f16 * ctx * slots)


def _draft_at(ctx: int) -> int:
    """A drafter whose reserve grows with the context, the way _mtp_bytes does."""
    return 256 * MIB + ctx * 48 * 1024


def _auto_opts(
    card_gib: float,
    *,
    fitter_outside,
    drops = False,
    **kw,
) -> PlanOptions:
    return PlanOptions(
        require_cost_win = True,
        host = HostProfile(threads = 6),
        context_policy = ContextPolicy.FIT_ONLY,
        kv_bytes_at = _kv_at,
        min_ctx = AUTO_CAP,
        draft_bytes = _draft_at(NATIVE),
        draft_droppable = True,
        fitter_n_ctx = AUTO_CAP,
        fitter_vram_bytes_per_device = (int(card_gib * GIB),),
        fitter_outside_layout_bytes = fitter_outside,
        fitter_drops_draft = drops,
        **kw,
    )


def _real_fallback_ms(card_gib: float, opts: PlanOptions, outside: int) -> float:
    """--fit on at -c 8192 with exactly ``outside`` bytes beside the layout."""
    plain = replace(
        opts,
        fitter_n_ctx = 0,
        fitter_vram_bytes_per_device = (),
        fitter_outside_layout_bytes = None,
    )
    fallback = _fit_fallback_placement(
        _layout(),
        plain,
        _usable_vram([int(card_gib * GIB)], plain, AUTO_CAP, outside_layout_bytes = outside),
        AUTO_CAP,
        quantised = False,
        kv_bytes_floor = _kv_at(AUTO_CAP, 1),
        kv_on_host = False,
    )
    assert fallback is not None
    n_prompt = min(opts.workload_prompt_tokens, AUTO_CAP)
    return _score_arms(Placement(host_groups = []), fallback, opts, n_prompt, 1, 1)[1]


# ------------------------------------------------ finding 1: the draft reserve at the fitter's -c


@pytest.mark.parametrize("card_gib", [14, 16, 18])
def test_the_fallback_draft_is_priced_at_the_context_it_launches(card_gib):
    """The cache and reserve followed fitter_n_ctx but the draft stayed at 131072 (6.3 GiB here
    against 0.6 at 8192), so the fallback looked short of VRAM it really has."""
    at_fit = _draft_at(AUTO_CAP)
    probe = _auto_opts(card_gib, fitter_outside = None)
    outside = _outside_layout_bytes(probe) - _draft_at(NATIVE) + at_fit
    opts = _auto_opts(card_gib, fitter_outside = outside)
    plan = plan_placement(_layout(), [int(card_gib * GIB)], 94 * GIB, NATIVE, opts = opts)
    real = _real_fallback_ms(card_gib, opts, outside)
    assert plan.spills_anything and not plan.declined_by_gate, plan.reason
    assert plan.predicted_fit_request_ms == pytest.approx(real)
    assert plan.predicted_request_ms <= real * (1.0 - opts.min_penalty_reduction), (
        plan.n_ctx,
        plan.predicted_request_ms,
        real,
    )


def test_the_stale_draft_really_admitted_a_losing_spill():
    """Non-vacuous: priced with the draft at 131072 the same cell accepts a spill that loses to
    the real fallback, which the repriced arm refuses."""
    card_gib = 16
    stale = _auto_opts(card_gib, fitter_outside = None)
    plan = plan_placement(_layout(), [card_gib * GIB], 94 * GIB, NATIVE, opts = stale)
    outside = _outside_layout_bytes(stale) - _draft_at(NATIVE) + _draft_at(AUTO_CAP)
    real = _real_fallback_ms(card_gib, stale, outside)
    assert plan.spills_anything and not plan.declined_by_gate, plan.reason
    assert plan.predicted_request_ms > real * (1.0 - stale.min_penalty_reduction)


# ------------------------------------------------ findings 3 / 4: a fallback with no draft at all


@pytest.mark.parametrize("card_gib", [14, 16, 18])
def test_a_fallback_that_stands_the_draft_down_is_priced_without_it(card_gib):
    """Auto's embedded Hybrid Mamba MTP under a partial --fit on emits --spec-type none, so the
    fitter places the target with nothing beside it. Priced with the draft, the fallback spilled
    blocks it never spills and the gate accepted a spill against it."""
    probe = _auto_opts(card_gib, fitter_outside = None)
    outside = _outside_layout_bytes(probe) - _draft_at(NATIVE)
    opts = _auto_opts(card_gib, fitter_outside = outside, drops = True)
    plan = plan_placement(_layout(), [int(card_gib * GIB)], 94 * GIB, NATIVE, opts = opts)
    real = _real_fallback_ms(card_gib, opts, outside)
    assert plan.spills_anything and not plan.declined_by_gate, plan.reason
    assert plan.predicted_fit_request_ms == pytest.approx(real)
    assert plan.predicted_request_ms <= real * (1.0 - opts.min_penalty_reduction), (
        plan.n_ctx,
        plan.predicted_request_ms,
        real,
    )
    # Non-vacuous: the same cell priced with the draft on the fallback (finding 1 alone) loses.
    with_draft = _outside_layout_bytes(probe) - _draft_at(NATIVE) + _draft_at(AUTO_CAP)
    stale = plan_placement(
        _layout(),
        [int(card_gib * GIB)],
        94 * GIB,
        NATIVE,
        opts = _auto_opts(card_gib, fitter_outside = with_draft),
    )
    assert stale.spills_anything
    assert stale.predicted_request_ms > real * (1.0 - opts.min_penalty_reduction)


def test_a_resident_draftless_fallback_is_not_traded_for_a_spill():
    """20 GiB: without the draft the whole model fits at 8192, so --fit on moves nothing."""
    card_gib = 20
    probe = _auto_opts(card_gib, fitter_outside = None)
    outside = _outside_layout_bytes(probe) - _draft_at(NATIVE)
    assert _real_fallback_ms(card_gib, probe, outside) == 0.0
    plan = plan_placement(
        _layout(),
        [card_gib * GIB],
        94 * GIB,
        NATIVE,
        opts = _auto_opts(card_gib, fitter_outside = outside, drops = True),
    )
    assert not plan.spills_anything, plan.reason


def test_dropping_a_draft_the_fallback_also_drops_costs_no_penalty():
    """The knob-only gate charged the measured draft gain against a fallback that has none."""
    from core.inference.offload_planner import _Knobs, _knob_only_gate

    layout = _layout()
    # Without the draft the target fits at 8192; with it the fallback spills.
    card = 20 * GIB
    base = dict(
        require_cost_win = True,
        host = HostProfile(threads = 6),
        kv_bytes_at = _kv_at,
        draft_bytes = 2 * GIB,
        draft_droppable = True,
        fitter_vram_bytes_per_device = (card,),
    )
    keeps = PlanOptions(**base)
    outside = _outside_layout_bytes(keeps) - 2 * GIB
    drops = PlanOptions(**base, fitter_outside_layout_bytes = outside, fitter_drops_draft = True)
    knobs = _Knobs(n_parallel = 1, draft_dropped = True)
    args = dict(quantised = False, kv_bytes_floor = _kv_at(8192, 1), knobs = knobs)
    budget = _usable_vram([card], drops, 8192, outside_layout_bytes = outside)
    _, plan_ms, fit_ms = _knob_only_gate(layout, drops, 8192, budget, **args)
    assert plan_ms == 0.0 and fit_ms == 0.0
    # The control: a fallback that keeps the draft still charges the penalty when it spills.
    _, k_plan, k_fit = _knob_only_gate(
        layout, keeps, 8192, _usable_vram([card], keeps, 8192), **args
    )
    assert k_fit > 0
    assert k_plan == pytest.approx(k_fit * keeps.draft_drop_penalty_frac)


# ------------------------------------------------ the seam: what reaches PlanOptions


def _seam(monkeypatch, **kw):
    cell = dict(
        free_mib = 14 * 1024,
        mtp = True,
        n_ctx = 65536,
        fitter_n_ctx = AUTO_CAP,
        draft_bytes = 3 * GIB,
        mtp_reserve_bytes = 1 * GIB,
        extra_gpu = 1 * GIB + 512 * MIB,
    )
    return _captured_opts(monkeypatch, _Stub(), **{**cell, **kw})[0]


def test_the_seam_prices_the_fallback_draft_at_the_emitted_context(monkeypatch):
    opts = _seam(monkeypatch)
    assert not opts.fitter_drops_draft
    assert opts.fitter_outside_layout_bytes == _outside_layout_bytes(opts) - 3 * GIB + 1 * GIB


@pytest.mark.parametrize("droppable", [True, False])
def test_the_seam_prices_a_fallback_without_its_draft(monkeypatch, droppable):
    opts = _seam(
        monkeypatch,
        fitter_drops_draft = True,
        draft_droppable = droppable,
        mtp_draft_compute_bytes = 128 * MIB,
    )
    assert opts.fitter_drops_draft
    every = 3 * GIB + (128 * MIB if droppable else 0)
    assert opts.fitter_outside_layout_bytes == _outside_layout_bytes(opts) - every


@pytest.mark.parametrize(
    "kw",
    [
        dict(mtp = False),  # no draft engages
        dict(mtp_reserve_bytes = 3 * GIB),  # same context, same reserve
    ],
)
def test_the_seam_leaves_the_fallback_alone_when_its_draft_is_the_plans(monkeypatch, kw):
    opts = _seam(monkeypatch, **kw)
    assert opts.fitter_outside_layout_bytes is None and not opts.fitter_drops_draft


def test_load_model_reads_the_fallback_spec_policy_the_flag_builder_reads():
    """The predicate handed to the planner is the embedded_mtp_partial_offload rule, evaluated on
    the declined launch's argv before the planner is asked."""
    src = _load_model_source()
    record = src.index('_spill_inputs["fitter_drops_draft"] = bool(')
    ask = src.index("_spill = self._planned_tensor_spill(")
    assert record < ask
    predicate = src[record : src.index("\n                    )\n", record)]
    for term in (
        '_spill_inputs.get("mtp_will_engage")',
        '_spec_canon == "auto"',
        "self._nextn_predict_layers",
        "self._ssm_inner_size is not None",
        "self._full_attention_interval is not None",
        "not mtp_draft_path",
        "not _extra_args_set_spec_type(extra_args)",
        "self._partially_offloads_layers(",
        "_placement_verdict_partial",
    ):
        assert term in predicate, term
    rule = src[src.index("embedded_mtp_partial_offload = bool(") :]
    for term in (
        '_spec_canon == "auto"',
        "self._nextn_predict_layers",
        "self._ssm_inner_size is not None",
        "self._full_attention_interval is not None",
        "_extra_args_set_spec_type(extra_args)",
        "_placement_verdict_partial",
    ):
        assert term in rule[:4000], term


# ------------------------------------------------ finding 2: one card, the same CUDA order rule


def _one_card_launch(tmp_path, monkeypatch, plan):
    from test_llama_cpp_placement import _backend, _launch
    from test_offload_planner_launch import DENSE

    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    card = 12 * 1024
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, card, card)])

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 20 * GIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    backend._select_gpus = lambda *a, **k: ([0], True)
    backend._select_gpus_split_aware = lambda *a, **k: ([0], True)
    asked = []
    backend._planned_tensor_spill = lambda inputs, **_kw: asked.append(inputs) or plan
    launched = _launch(backend, gguf, speculative_type = "off", n_ctx = 8192, n_parallel = 1)
    assert asked, "the planner was never consulted"
    return launched["cmd"], backend


def _one_card_plan() -> Plan:
    return Plan(changed = True, priced = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))


@pytest.mark.parametrize(
    "order, mask, picked, keeps_plan",
    [
        # The child's CUDA device 0 is the fastest card; the plan's VRAM and link are PCI row 0.
        ("FASTEST_FIRST", "0", None, False),
        ("PCI_BUS_ID", "0", None, True),
        ("FASTEST_FIRST", None, None, True),
        ("FASTEST_FIRST", "0", (0,), True),
    ],
)
def test_an_inherited_cuda_order_nothing_verified_declines_a_one_card_plan(
    tmp_path, monkeypatch, order, mask, picked, keeps_plan
):
    from core.inference import llama_cpp as llama_mod

    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(
        llama_mod.LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: False)
    )
    if picked is not None:
        original = llama_mod.LlamaCppBackend.load_model

        def load(self, intent, *a, **k):
            return original(self, replace(intent, gpu_ids = picked), *a, **k)

        monkeypatch.setattr(llama_mod.LlamaCppBackend, "load_model", load)
    cmd, backend = _one_card_launch(tmp_path, monkeypatch, _one_card_plan())
    if keeps_plan:
        assert "-ot" in cmd and cmd[cmd.index("--fit") + 1] == "off", cmd
    else:
        assert "-ot" not in cmd and cmd[cmd.index("--fit") + 1] == "on", cmd
        assert not backend._spill_plan_flags


def test_a_rocm_host_keeps_its_one_card_plan(tmp_path, monkeypatch):
    """ROCm masks are HIP's, not CUDA_DEVICE_ORDER's: the control the rule exempts."""
    from core.inference import llama_cpp as llama_mod

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        llama_mod.LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: True)
    )
    cmd, _ = _one_card_launch(tmp_path, monkeypatch, _one_card_plan())
    assert "-ot" in cmd, cmd


def test_the_order_clause_is_not_gated_on_a_split():
    src = _load_model_source()
    guard = src[src.index("_cuda_order_unverified = bool(") : src.index("_user_split_in_force")]
    assert re.search(r"_spill is not None\s+and \(\s+_cuda_order_unverified\s+or", guard), guard


# ------------------------------------- an unmasked split budgeted on torch ordinals (r23)


def _two_card_split_launch(
    tmp_path,
    monkeypatch,
    *,
    order,
    ids_are_pci,
    rocm = False,
):
    """Two visible cards, no mask, no pick: a plan-emitted --tensor-split reaches
    _pin_visible_gpu_order_for_split, which switches the child to PCI_BUS_ID."""
    from core.inference import llama_cpp as llama_mod
    from test_llama_cpp_placement import _backend, _launch
    from test_offload_planner_launch import DENSE

    cls = llama_mod.LlamaCppBackend
    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    if order is None:
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    else:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(cls, "_host_torch_is_rocm", staticmethod(lambda: rocm))
    monkeypatch.setattr(cls, "_GPU_IDS_ARE_PCI_INDICES", ids_are_pci)
    pins = []
    monkeypatch.setattr(
        cls,
        "_pin_visible_gpu_order_for_split",
        staticmethod(lambda env: pins.append(dict(env)) or None),
    )
    big, small = 48 * 1024, 24 * 1024
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, big, big), (1, small, small)])

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 90 * GIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
    }
    backend._available_system_memory_mib = lambda: 256 * 1024
    plan = Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
        device_layer_counts = (40, 20),
    )
    asked = []
    backend._planned_tensor_spill = lambda inputs, **_kw: asked.append(inputs) or plan
    launched = _launch(backend, gguf, speculative_type = "off", n_ctx = 8192, n_parallel = 1)
    assert asked, "the planner was never consulted"
    return launched["cmd"], backend, pins


@pytest.mark.parametrize(
    "order, ids_are_pci, rocm, keeps_plan",
    [
        # torch ordinals are FASTEST_FIRST indices; the pin re-reads the rows in PCI order.
        ("FASTEST_FIRST", False, False, False),
        # A user's own unset order is CUDA's default, FASTEST_FIRST, all the same.
        (None, False, False, False),
        # nvidia-smi rows ARE PCI indices, which the pin makes true.
        ("FASTEST_FIRST", True, False, True),
        # torch ordinals under PCI_BUS_ID already are PCI indices.
        ("PCI_BUS_ID", False, False, True),
        # HIP enumeration does not follow CUDA_DEVICE_ORDER.
        ("FASTEST_FIRST", False, True, True),
    ],
)
def test_an_unmasked_split_on_torch_ordinals_is_declined_unless_the_order_is_pci(
    tmp_path, monkeypatch, order, ids_are_pci, rocm, keeps_plan
):
    cmd, backend, pins = _two_card_split_launch(
        tmp_path, monkeypatch, order = order, ids_are_pci = ids_are_pci, rocm = rocm
    )
    if keeps_plan:
        assert "-ot" in cmd and cmd[cmd.index("--fit") + 1] == "off", cmd
        assert cmd[cmd.index("--tensor-split") + 1] == "40,20", cmd
        assert len(pins) == 1, "the plan's split pins the child's enumeration"
    else:
        assert "-ot" not in cmd and cmd[cmd.index("--fit") + 1] == "on", cmd
        assert "--tensor-split" not in cmd, cmd
        assert not backend._spill_plan_flags
