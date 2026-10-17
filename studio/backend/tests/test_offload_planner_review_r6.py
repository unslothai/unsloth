# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MoE long-prompt veto under a context search, and the cache a no-flash symmetric launch
really allocates."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend, _kv_bytes_per_elem  # noqa: E402
from core.inference.offload_layout import BlockLayout, ModelLayout  # noqa: E402
from core.inference.offload_planner import ContextPolicy, PlanOptions, plan_placement  # noqa: E402

GIB = 1024**3


def _moe(n_blocks: int = 32) -> ModelLayout:
    """15.04 GiB of experts, a 0.5 GiB output head, 16 KV heads of 128: 0.25 MiB per token."""
    blocks = tuple(BlockLayout(i, int(0.47 * GIB), int(0.02 * GIB)) for i in range(n_blocks))
    return ModelLayout(
        arch = "qwen3moe",
        n_layers = n_blocks,
        n_attention_layers = n_blocks,
        blocks = blocks,
        lm_head_bytes = int(0.5 * GIB),
        token_embd_bytes = int(0.5 * GIB),
        other_resident_bytes = int(0.01 * GIB),
        kv_bytes_per_token_f16 = 16 * 128 * 2 * 2 * n_blocks,
        n_ctx_train = 262144,
        is_moe = True,
        n_expert = 128,
        n_expert_used = 8,
        complete = True,
    )


def _plan(
    card_gib: float,
    n_ctx: int,
    *,
    n_parallel = 1,
    unified = False,
):
    opts = PlanOptions(
        require_cost_win = True,
        context_policy = ContextPolicy.FIT_ONLY,
        n_parallel = n_parallel,
        min_parallel = n_parallel,
        kv_unified = unified,
    )
    return plan_placement(_moe(), [int(card_gib * GIB)], 94 * GIB, n_ctx, opts = opts)


# ------------------------------------------------ finding 1: the veto under a context search


@pytest.mark.parametrize(
    "n_ctx, n_parallel, unified",
    [
        (131072, 1, False),  # the reported Auto request
        (65536, 1, False),
        (32768, 1, False),  # exactly at the threshold
        (131072, 4, True),  # one shared stream: a request may fill all of it
        (65536, 2, True),
        (131072, 4, False),  # 32768 per private slot
    ],
)
def test_a_long_moe_request_that_does_not_place_is_vetoed_not_cut(n_ctx, n_parallel, unified):
    """At the context asked for nothing places, so the search walked under the 32K threshold and
    launched a cut context with every expert and output.weight on the host (-c 13312 --fit off),
    which the veto measured slower than --fit on."""
    plan = _plan(6, n_ctx, n_parallel = n_parallel, unified = unified)
    assert plan.veto and plan.declined_by_gate, (plan.n_ctx, plan.reason)
    assert not plan.spills_anything
    assert plan.n_ctx == n_ctx


def test_a_long_moe_load_that_fits_resident_is_unchanged():
    plan = _plan(64, 131072)
    assert not plan.veto and not plan.spills_anything, plan.reason
    assert plan.n_ctx == 131072


def test_a_short_moe_request_still_searches_and_spills():
    """Below the threshold the veto does not apply, so the context search keeps its say."""
    plan = _plan(6, 16384)
    assert not plan.veto, plan.reason


# ------------------------------------------------ finding 2: the no-flash symmetric cache


def _legacy_deepseek() -> LlamaCppBackend:
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


def test_a_symmetric_model_without_flash_is_priced_at_the_f16_cache_it_launches():
    """The FA-off rewrite sends a quantized V to f16 and, on a K == V model, K with it; the
    estimate priced K at q8_0, which the trusted floor then admitted against."""
    n_ctx = 8192
    for b in (_legacy_deepseek(), _genuine_mla()):
        assert b._target_kv_symmetry()
        q8_off = b._estimate_kv_cache_bytes(n_ctx, "q8_0", flash_attn = False)
        f16_off = b._estimate_kv_cache_bytes(n_ctx, "f16", flash_attn = False)
        assert q8_off == f16_off
        # Flash on keeps the quantized cache.
        assert b._estimate_kv_cache_bytes(n_ctx, "q8_0", flash_attn = True) < f16_off
        # A wider type is kept as typed.
        assert b._estimate_kv_cache_bytes(n_ctx, "f32", flash_attn = False) > f16_off


def test_an_asymmetric_model_keeps_its_quantized_k_without_flash():
    b = LlamaCppBackend()
    for key, value in dict(
        _architecture = "llama",
        _n_layers = 32,
        _n_heads = 32,
        _n_kv_heads = 8,
        _embedding_length = 4096,
        _kv_key_length = 128,
        _kv_value_length = 128,
    ).items():
        setattr(b, key, value)
    assert not b._target_kv_symmetry()
    got = b._estimate_kv_cache_bytes(8192, "q8_0", flash_attn = False)
    cells = 8192
    assert got == int(32 * cells * 8 * 128 * (_kv_bytes_per_elem("q8_0") + 2))
