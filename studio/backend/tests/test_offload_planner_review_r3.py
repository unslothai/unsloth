# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-count bounds on the SWA expansions, the reported VRAM of a plan priced at an exact cache,
one request's share of a non-unified window, lazy tables in the prompt-cache clamp, a picker order
under a split plan, and the MLA predicate in the scratch sizing."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.inference import llama_cpp as llama_mod  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.offload_layout import LLAMA_MAX_LAYERS  # noqa: E402
from core.inference.offload_planner import (  # noqa: E402
    Plan,
    PlanOptions,
    _fit_fallback_placement,
    all_resident_bytes,
    plan_placement,
)
from test_offload_cost_gate import GIB, MIB, dense_layout, gated  # noqa: E402
from test_offload_cost_gate import swa_split_layout  # noqa: E402


# ------------------------------------------------ findings 1 / 2: SWA expansion past the cap


def test_the_scalar_swa_period_is_not_expanded_past_llamas_layer_cap():
    """The layout reader and the per-layer vector refuse a block_count above LLAMA_MAX_LAYERS, but
    the header parser expanded the scalar period into a list of that length first."""
    from test_kv_cache_estimation import _backend_from_gguf

    fields = {"attention.sliding_window": 1024, "attention.sliding_window_pattern": 6}
    ok = _backend_from_gguf("gemma3", dict(block_count = LLAMA_MAX_LAYERS, **fields))
    assert ok._sliding_window_pattern is not None
    assert len(ok._sliding_window_pattern) == LLAMA_MAX_LAYERS
    over = _backend_from_gguf("gemma3", dict(block_count = LLAMA_MAX_LAYERS + 1, **fields))
    assert over._sliding_window_pattern is None


def test_the_swa_resolver_does_not_expand_past_llamas_layer_cap():
    assert llama_mod._resolve_swa_pattern("gpt-oss", 24) == [(i + 1) % 2 != 0 for i in range(24)]
    assert len(llama_mod._resolve_swa_pattern("gpt-oss", LLAMA_MAX_LAYERS)) == LLAMA_MAX_LAYERS
    assert llama_mod._resolve_swa_pattern("gpt-oss", LLAMA_MAX_LAYERS + 1) is None


# ------------------------------------------------ finding 3: the reported VRAM of an exact cache


def test_the_reported_vram_uses_the_exact_cache_admission_priced():
    """Admission trusts kv_bytes_at; the reported figure re-ran the product and over-stated the
    resident footprint by the product's excess."""
    layout = dense_layout()
    exact = 512 * MIB
    opts = PlanOptions(kv_bytes_at = lambda _ctx, _slots: exact)
    plan = plan_placement(layout, [12 * GIB], 94 * GIB, 32768, kv_bytes_floor = exact, opts = opts)
    assert plan.spills_anything and not plan.spilled_lm_head, plan.reason
    # The product at 32K is 2 GiB, four times the exact cache.
    assert layout.kv_bytes(plan.n_ctx, 2) == 4 * exact
    spilled = sum(layout.blocks[i].spillable_bytes for i in plan.spilled_blocks)
    admitted = all_resident_bytes(layout, plan.n_ctx, kv_bytes_floor = exact, trust_floor = True)
    assert plan.vram_bytes == admitted - spilled, (
        plan.vram_bytes / GIB,
        (admitted - spilled) / GIB,
    )


# ------------------------------------------------ finding 4: one request's window stream


def test_a_non_unified_window_charges_one_stream_to_the_request():
    """Without --kv-unified every slot has its own stream (llama-kv-cache.cpp: n_stream =
    unified ? 1 : n_seq_max, one [n_embd, kv_size, n_stream] tensor per layer), and one request
    reads only its own. Charging all four windows priced the fitter at four requests' traffic."""
    layout = swa_split_layout()
    n_ctx, floor, windowed = 131072, 5 * GIB, 4 * GIB
    shape = dict(workload_prompt_tokens = 2048, workload_generated_tokens = 0, n_parallel = 4)
    args = dict(quantised = False, kv_bytes_floor = floor, kv_on_host = False, n_seq = 4)

    def placed(**extra):
        placement = _fit_fallback_placement(layout, gated(**shape, **extra), 8 * GIB, n_ctx, **args)
        assert placement is not None
        return placement

    whole = placed()
    private = placed(kv_swa_bytes_floor = windowed)
    shared = placed(kv_swa_bytes_floor = windowed, kv_unified = True)
    assert whole.kv_host_bytes > 0
    # A quarter of the 4 GiB window, plus the 1 GiB full-context part over a 2048-token prefix.
    assert abs(private.kv_host_bytes / whole.kv_host_bytes - (1 + 1 / 64) / 5) < 0.001
    # One shared stream keeps the whole window, and its prefix is over n_ctx.
    assert abs(shared.kv_host_bytes / whole.kv_host_bytes - (4 + 1 / 64) / 5) < 0.001


# ------------------------------------------------ finding 5: lazy tables in the cache clamp


def test_a_lazily_read_table_does_not_clamp_the_prompt_cache(tmp_path, monkeypatch):
    """The load-mode rule and the planner leave a lazily read table out of the host footprint
    (it stays mmap'd); the --cache-ram clamp charged it and zeroed the prompt cache on a host with
    room for the rest."""
    from test_offload_planner_launch import _launch_with

    monkeypatch.setattr(
        LlamaCppBackend, "_lazy_read_host_bytes", lambda self, *a, **k: 20 * 1024 * MIB
    )
    cmd, _b, seen = _launch_with(
        tmp_path, monkeypatch, Plan(reason = "declined"), avail_mib = 28 * 1024, model_mib = 36 * 1024
    )
    assert "--cache-ram" not in cmd, cmd
    assert seen["inputs"]["cache_ram_default_mib"] == 8192


def test_the_cache_clamp_still_binds_without_a_lazy_table(tmp_path, monkeypatch):
    from test_offload_planner_launch import _launch_with

    monkeypatch.setattr(LlamaCppBackend, "_lazy_read_host_bytes", lambda self, *a, **k: 0)
    cmd, _b, _seen = _launch_with(
        tmp_path, monkeypatch, Plan(reason = "declined"), avail_mib = 28 * 1024, model_mib = 36 * 1024
    )
    got = cmd[cmd.index("--cache-ram") + 1] if "--cache-ram" in cmd else None
    assert got is not None and int(got) < 8192, cmd


# ------------------------------------------------ finding 6: a picker order under a split plan


def _split_plan() -> Plan:
    return Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
        device_layer_counts = (30, 18),
    )


def test_a_picker_order_that_reverses_the_cards_declines_a_split_plan(tmp_path, monkeypatch):
    """The plan's rows follow the device order it was budgeted in. A picker order of [1, 0]
    reaches the child as CUDA_VISIBLE_DEVICES=1,0, so --tensor-split 30,18 gave device 1 the
    rows budgeted for device 0."""
    from test_offload_planner_review_r2 import _two_card_launch

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    original = llama_mod.LlamaCppBackend.load_model

    def picked(order):
        def load(self, intent, *a, **k):
            from dataclasses import replace
            return original(self, replace(intent, gpu_ids = order), *a, **k)

        monkeypatch.setattr(llama_mod.LlamaCppBackend, "load_model", load)
        return _two_card_launch(tmp_path, monkeypatch, _split_plan())

    cmd, _backend = picked((0, 1))
    assert cmd[cmd.index("--tensor-split") + 1] == "30,18", "an ascending pick keeps the plan"
    cmd, backend = picked((1, 0))
    assert backend._gpu_ids == [1, 0]
    assert "--tensor-split" not in cmd, cmd
    assert "-ot" not in cmd, cmd
    assert not backend._spill_plan_flags


# ------------------------------------------------ findings 7 / 8: the MLA predicate in scratch


def _legacy_deepseek() -> LlamaCppBackend:
    """unsloth/DeepSeek-R1-GGUF's shape: kv_lora_rank with no MLA head lengths, so llama.cpp runs
    it as MHA over a full per-head K+V cache (deepseek2.cpp: wkv_b, not wk_b/wv_b)."""
    b = LlamaCppBackend()
    for key, value in dict(
        _architecture = "deepseek2",
        _n_layers = 61,
        _n_heads = 128,
        _n_kv_heads = 128,
        _embedding_length = 7168,
        _kv_lora_rank = 512,
        _kv_key_length = 192,
        _kv_value_length = 128,
        _key_length_mla = None,
        _value_length_mla = None,
    ).items():
        setattr(b, key, value)
    return b


def _genuine_mla() -> LlamaCppBackend:
    b = _legacy_deepseek()
    b._n_kv_heads = 1
    b._kv_key_length = 576
    b._kv_value_length = 512
    b._key_length_mla = 192
    b._value_length_mla = 128
    return b


def test_the_dequant_width_follows_the_cache_llama_allocates():
    legacy = _legacy_deepseek()
    assert not legacy._uses_mla_cache()
    assert legacy._attention_dequant_width() == 128 * (192 + 128)
    # Genuine MLA keeps the K-only latent width.
    assert _genuine_mla()._attention_dequant_width() == 576


def test_the_non_flash_latent_copy_is_charged_only_to_an_mla_cache():
    """The float32 latent copy is MLA's; a legacy file runs plain MHA attention."""

    def scratch(b, flash):
        return b._compute_buffer_ctx_bytes(
            4096, flash_attn = flash, cache_type_kv = "f16", n_ubatch = 512
        )

    legacy = _legacy_deepseek()
    mla = _genuine_mla()
    assert (
        scratch(legacy, False) - scratch(legacy, True)
        == scratch(mla, False) - scratch(mla, True) - 576 * 4 * 4096
    )
