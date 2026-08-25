# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The -ot spill planner: the ladder, the patterns, and everything it must never do.

The negative half matters most. The planner exists because llama.cpp's own
fitter spills whole layers and drags the KV cache to host RAM with them, so a
plan that emits -ngl or --no-kv-offload, or that charges token_embd to VRAM, is
worse than no plan at all.
"""

from __future__ import annotations

import os
import re

import pytest

from core.inference.offload_layout import (
    LM_HEAD_PATTERN,
    BlockLayout,
    ModelLayout,
    layout_from_gguf,
    spill_pattern_for,
)
from core.inference.offload_cost_model import HostProfile
from core.inference.offload_planner import (
    ContextPolicy,
    Plan,
    PlanOptions,
    SpillOrder,
    all_resident_bytes,
    max_context_for,
    plan_placement,
    plan_to_args,
    resident_floor_bytes,
)

GIB = 1024**3
MIB = 1024**2

_MODELS = "/mnt/disks/unslothai/daniel3/workspace_11/temp/loadmode_sim"
Q2_PATH = f"{_MODELS}/qwen38/Qwen3.8-27B-UD-Q2_K_XL.gguf"
Q4_PATH = f"{_MODELS}/qwen38/Qwen3.8-27B-UD-Q4_K_XL.gguf"
MOE_PATH = f"{_MODELS}/qwen36moe/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

needs_gguf = pytest.mark.skipif(
    not os.path.exists(Q4_PATH), reason = "local GGUF fixtures not present"
)


# ----------------------------------------------------------------- synthetic layouts
# Byte totals are the measured ones for Qwen3.8-27B, so the ladder tests run the
# real arithmetic without needing 27 GB of fixtures on disk.


def _uniform_layout(
    *,
    n_blocks: int = 65,
    spillable_total: int,
    resident_total: int,
    lm_head: int,
    token_embd: int,
    kv_per_token: int = 65536,
    recurrent: int = 156_893_184,
    is_moe: bool = False,
    n_ctx_train: int = 262144,
) -> ModelLayout:
    per_spill, per_res = spillable_total // n_blocks, resident_total // n_blocks
    blocks = tuple(
        BlockLayout(index = i, spillable_bytes = per_spill, resident_bytes = per_res)
        for i in range(n_blocks)
    )
    return ModelLayout(
        arch = "qwen35moe" if is_moe else "qwen35",
        n_layers = 64,
        n_attention_layers = 16,
        blocks = blocks,
        lm_head_bytes = lm_head,
        token_embd_bytes = token_embd,
        kv_bytes_per_token_f16 = kv_per_token,
        recurrent_bytes = recurrent,
        n_ctx_train = n_ctx_train,
        is_moe = is_moe,
        complete = True,
    )


def q4_layout() -> ModelLayout:
    return _uniform_layout(
        spillable_total = 10_836_131_840,
        resident_total = 4_953_923_584,
        lm_head = 1_042_915_328,
        token_embd = 715_128_832,
    )


def q2_layout() -> ModelLayout:
    return _uniform_layout(
        spillable_total = 5_900_615_680,
        resident_total = 2_785_009_664,
        lm_head = 715_128_832,
        token_embd = 417_202_176,
    )


def uneven_layout() -> ModelLayout:
    """Four blocks with deliberately lopsided FFN, for selection policy."""
    sizes = [100 * MIB, 400 * MIB, 200 * MIB, 50 * MIB]
    blocks = tuple(
        BlockLayout(index = i, spillable_bytes = s, resident_bytes = 10 * MIB)
        for i, s in enumerate(sizes)
    )
    return ModelLayout(
        arch = "qwen35",
        n_layers = 4,
        n_attention_layers = 4,
        blocks = blocks,
        lm_head_bytes = 100 * MIB,
        token_embd_bytes = 50 * MIB,
        kv_bytes_per_token_f16 = 1024,
        recurrent_bytes = 0,
        n_ctx_train = 65536,
        complete = True,
    )


# ------------------------------------------------------------------- the ladder


def test_a_load_that_fits_spills_nothing():
    plan = plan_placement(q2_layout(), [24 * GIB], 64 * GIB, 8192)
    assert plan.spilled_blocks == ()
    assert plan.spilled_lm_head is False
    assert plan.ot_patterns == ()
    assert plan.load_mode_none is True
    assert "fits in VRAM" in plan.reason


def test_a_load_that_does_not_fit_spills_ffn_and_keeps_the_cache():
    """The whole point: weights move, the cache does not."""
    plan = plan_placement(q4_layout(), [12 * GIB], 64 * GIB, 32768)
    assert plan.spilled_blocks, "should have spilled something"
    assert plan.spilled_lm_head is False, "lm_head is the LAST rung, not the first"
    args = plan_to_args(plan)
    assert "-ngl" not in args and "--n-gpu-layers" not in args
    assert "--no-kv-offload" not in args and "-nkvo" not in args


def test_lm_head_is_only_spilled_after_every_block():
    """43% of generation alone, 16% after FFN. Never take it first."""
    layout = q4_layout()
    # Tight enough that the whole FFN is not enough on its own.
    plan = plan_placement(layout, [7 * GIB], 64 * GIB, 8192)
    assert plan.spilled_lm_head is True
    assert len(plan.spilled_blocks) == len(layout.blocks), "all blocks go first"


def test_partial_spill_takes_only_what_is_needed():
    layout = q4_layout()
    plan = plan_placement(layout, [16 * GIB], 64 * GIB, 8192)
    assert 0 < len(plan.spilled_blocks) < len(layout.blocks)


def test_a_load_that_cannot_fit_keeps_mmap_and_says_so():
    """Below the resident floor, -ot has nothing left to give. mmap must stay:
    it is the only thing that makes an over-commit pageable instead of killed."""
    plan = plan_placement(q4_layout(), [4 * GIB], 64 * GIB, 131072)
    assert plan.insufficient is True
    assert plan.changed is False
    assert plan.load_mode_none is False
    assert plan_to_args(plan) == []
    assert "smaller quant" in plan.reason


# ------------------------------------------------------------ what must never happen


@pytest.mark.parametrize("vram", [4 * GIB, 8 * GIB, 12 * GIB, 16 * GIB, 24 * GIB])
@pytest.mark.parametrize("ctx", [4096, 32768, 131072])
def test_no_plan_ever_moves_the_kv_cache(vram, ctx):
    """-ngl and --no-kv-offload both cost 22x to 73x. Neither may ever appear."""
    for layout in (q2_layout(), q4_layout()):
        args = plan_to_args(plan_placement(layout, [vram], 64 * GIB, ctx))
        for banned in ("-ngl", "--gpu-layers", "--n-gpu-layers", "-nkvo", "--no-kv-offload"):
            assert banned not in args


def test_token_embd_is_never_charged_to_vram():
    """llama-model.cpp pins dev_input to the CPU unconditionally, so the
    embedding is host RAM the plan must pay for, never VRAM it may spend.

    Proved by varying ONLY token_embd: a 4 GiB embedding must not change the
    VRAM figure or the spill decision by a single byte, and must show up in the
    host figure in full.
    """
    small = q4_layout()
    big = ModelLayout(**{**small.__dict__, "token_embd_bytes": 4 * GIB})

    tight = [12 * GIB]
    a = plan_placement(small, tight, 64 * GIB, 8192)
    b = plan_placement(big, tight, 64 * GIB, 8192)

    assert a.vram_bytes == b.vram_bytes
    assert a.spilled_blocks == b.spilled_blocks
    assert b.host_bytes - a.host_bytes == 4 * GIB - small.token_embd_bytes


def test_load_mode_none_is_withheld_when_host_ram_cannot_hold_the_spill():
    """Turning mmap off on a host that cannot hold the spill turns a pageable
    load into an OOM kill."""
    layout = q4_layout()
    roomy = plan_placement(layout, [12 * GIB], 64 * GIB, 8192)
    cramped = plan_placement(layout, [12 * GIB], 3 * GIB, 8192)
    assert roomy.load_mode_none is True
    assert cramped.load_mode_none is False


def test_unreadable_host_ram_keeps_mmap():
    plan = plan_placement(q4_layout(), [12 * GIB], None, 8192)
    assert plan.load_mode_none is False


# ------------------------------------------------------------------- patterns


def test_every_emitted_pattern_is_anchored():
    """Unanchored is a live trap: 'output\\.weight' also matches every
    blk.N.attn_output.weight, which silently moved 16 attention projections."""
    for layout in (q2_layout(), q4_layout()):
        for vram in (6 * GIB, 12 * GIB, 16 * GIB):
            plan = plan_placement(layout, [vram], 64 * GIB, 8192)
            for pattern in plan.ot_patterns:
                assert pattern.startswith("^") and pattern.endswith("$"), pattern


def test_the_lm_head_pattern_does_not_catch_attention_output():
    assert re.search(LM_HEAD_PATTERN, "output.weight")
    assert not re.search(LM_HEAD_PATTERN, "blk.3.attn_output.weight")
    # The unanchored form is what went wrong; pin the difference.
    assert re.search(r"output\.weight", "blk.3.attn_output.weight")


def test_the_dense_pattern_does_not_catch_the_router():
    pattern = spill_pattern_for(q4_layout())
    assert re.search(pattern, "blk.7.ffn_up.weight")
    assert re.search(pattern, "blk.7.ffn_gate.weight")
    assert not re.search(pattern, "blk.7.ffn_gate_inp.weight")
    assert not re.search(pattern, "blk.7.ffn_norm.weight")
    assert not re.search(pattern, "blk.7.attn_q.weight")


def test_the_moe_pattern_spills_experts_but_not_shared_experts():
    """Shared experts run on EVERY token, like a dense FFN, for 0.6% of the
    model. Spilling them buys nothing and costs dense-like bandwidth."""
    moe = _uniform_layout(
        n_blocks = 40, spillable_total = 19_671_285_760, resident_total = 1_597_483_520,
        lm_head = 540_000_000, token_embd = 540_000_000, kv_per_token = 20480, is_moe = True,
    )
    pattern = spill_pattern_for(moe)
    assert re.search(pattern, "blk.5.ffn_up_exps.weight")
    assert re.search(pattern, "blk.5.ffn_down_exps.weight")
    assert not re.search(pattern, "blk.5.ffn_up_shexp.weight")
    assert not re.search(pattern, "blk.5.ffn_gate_inp.weight")


def test_partial_spill_names_only_the_chosen_blocks():
    pattern = spill_pattern_for(q4_layout(), [3, 7, 11])
    assert re.search(pattern, "blk.3.ffn_up.weight")
    assert re.search(pattern, "blk.11.ffn_down.weight")
    assert not re.search(pattern, "blk.4.ffn_up.weight")
    # blk.1 must not sneak in through blk.11's alternation.
    assert not re.search(pattern, "blk.1.ffn_up.weight")


def test_a_full_spill_uses_the_compact_global_pattern():
    layout = q4_layout()
    plan = plan_placement(layout, [7 * GIB], 64 * GIB, 8192)
    assert plan.ot_patterns[0] == spill_pattern_for(layout, None)
    assert r"\d+" in plan.ot_patterns[0]


def test_plan_to_args_shape():
    layout = q4_layout()
    args = plan_to_args(plan_placement(layout, [12 * GIB], 64 * GIB, 8192))
    assert args.count("-ot") == 1
    assert args[args.index("-ot") + 1].endswith("=CPU")
    assert "--load-mode" in args and args[args.index("--load-mode") + 1] == "none"


# ---------------------------------------------------------------- spill selection


def test_largest_first_minimises_overshoot():
    """Best-fit-decreasing: cover a small residual with a small block, not by
    dragging a 400 MiB block across the bus on every token."""
    layout = uneven_layout()
    # Need ~50 MiB freed: the 50 MiB block alone should do it.
    floor = resident_floor_bytes(layout, 4096)
    budget = floor + layout.spillable_bytes - 40 * MIB + 1 * GIB  # +overhead
    plan = plan_placement(layout, [budget], 64 * GIB, 4096, opts = PlanOptions())
    spilled = sum(b.spillable_bytes for b in layout.blocks if b.index in plan.spilled_blocks)
    assert spilled == 50 * MIB, "should take the 50 MiB block, not a bigger one"


def test_front_and_back_orders_pick_opposite_ends():
    layout = uneven_layout()
    floor = resident_floor_bytes(layout, 4096)
    budget = floor + layout.spillable_bytes - 40 * MIB + 1 * GIB
    front = plan_placement(
        layout, [budget], 64 * GIB, 4096,
        opts = PlanOptions(spill_order = SpillOrder.FRONT_FIRST),
    )
    back = plan_placement(
        layout, [budget], 64 * GIB, 4096,
        opts = PlanOptions(spill_order = SpillOrder.BACK_FIRST),
    )
    assert front.spilled_blocks == (0,)
    assert back.spilled_blocks == (3,)


# ------------------------------------------------------------------ abstention


def test_an_incomplete_layout_abstains():
    plan = plan_placement(ModelLayout(), [24 * GIB], 64 * GIB, 8192)
    assert plan.changed is False
    assert plan_to_args(plan) == []


def test_no_devices_abstains():
    assert plan_placement(q4_layout(), [], 64 * GIB, 8192).changed is False


def test_a_device_smaller_than_its_own_overhead_abstains():
    assert plan_placement(q4_layout(), [512 * MIB], 64 * GIB, 8192).changed is False


# -------------------------------------------------------------------- multi GPU


def test_every_device_pays_the_fixed_overhead():
    """A layer split puts a CUDA context and scratch on each card, so two 8 GiB
    cards are not one 16 GiB card."""
    layout = q4_layout()
    one_big = plan_placement(layout, [16 * GIB], 64 * GIB, 8192)
    two_small = plan_placement(layout, [8 * GIB, 8 * GIB], 64 * GIB, 8192)
    assert len(two_small.spilled_blocks) > len(one_big.spilled_blocks)


def test_multi_gpu_credit_sums():
    layout = q2_layout()
    assert plan_placement(layout, [6 * GIB, 6 * GIB], 64 * GIB, 8192).spilled_blocks == ()


# --------------------------------------------------------------- context policy


def test_never_reduce_keeps_the_requested_context():
    layout = q4_layout()
    plan = plan_placement(layout, [12 * GIB], 64 * GIB, 65536)
    assert plan.n_ctx == 65536
    assert plan.spilled_blocks, "it pays with spill, not with the user's context"


def test_prefer_resident_shrinks_instead_of_spilling():
    layout = q4_layout()
    plan = plan_placement(
        layout, [18 * GIB], 64 * GIB, 65536,
        opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT),
    )
    assert plan.spilled_blocks == ()
    assert plan.n_ctx < 65536
    assert "shrank context" in plan.reason


def test_prefer_resident_still_spills_when_even_min_ctx_will_not_fit():
    layout = q4_layout()
    plan = plan_placement(
        layout, [10 * GIB], 64 * GIB, 65536,
        opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT),
    )
    assert plan.spilled_blocks, "shrinking cannot save this one, so spill"


def test_context_is_clamped_to_what_the_model_was_trained_on():
    layout = q4_layout()
    plan = plan_placement(layout, [24 * GIB], 128 * GIB, 999_999)
    assert plan.n_ctx == layout.n_ctx_train


# ------------------------------------------------------------------- KV quant


def test_kv_quantisation_is_off_by_default():
    """35% slower generation, and only matched pairs are compiled."""
    plan = plan_placement(q4_layout(), [12 * GIB], 64 * GIB, 65536)
    assert plan.cache_type_k is None and plan.cache_type_v is None


def test_kv_quantisation_rescues_a_load_f16_cannot_fit():
    """9 GiB with 64K of context: the f16 cache alone puts the resident floor
    at 9.73 GiB against 8 GiB usable, so no rung of the ladder fits. Halving the
    cache brings the floor to 7.73 and it does."""
    layout = q4_layout()
    without = plan_placement(layout, [9 * GIB], 64 * GIB, 65536)
    assert without.insufficient is True

    with_quant = plan_placement(
        layout, [9 * GIB], 64 * GIB, 65536,
        opts = PlanOptions(allow_kv_quant = True),
    )
    assert with_quant.insufficient is False
    assert with_quant.cache_type_k == "q8_0"
    # Matched pair, always: an unmatched K/V combination is not compiled without
    # GGML_CUDA_FA_ALL_QUANTS and silently falls back to CPU.
    assert with_quant.cache_type_k == with_quant.cache_type_v


def test_f16_is_preferred_when_it_fits_even_with_quant_allowed():
    """q8_0 costs 35% of generation, so it is a rescue, not a default."""
    plan = plan_placement(
        q2_layout(), [24 * GIB], 64 * GIB, 8192,
        opts = PlanOptions(allow_kv_quant = True),
    )
    assert plan.cache_type_k is None


# ------------------------------------------------- the budget ladder, end to end


@pytest.mark.parametrize(
    "budget_gib,expected_k",
    # From the independently computed placement table: max context with the FFN
    # spilled, f16 cache, 1 GiB overhead.
    [(8, 20), (10, 52), (12, 84), (16, 148), (20, 212)],
)
def test_q4_ffn_spilled_context_ladder(budget_gib, expected_k):
    got = max_context_for(q4_layout(), [budget_gib * GIB], spill_all_ffn = True)
    assert expected_k * 1024 <= got < (expected_k + 1) * 1024, got


@pytest.mark.parametrize("budget_gib,expected_k", [(6, 25), (8, 57), (12, 121), (20, 249)])
def test_q2_ffn_spilled_context_ladder(budget_gib, expected_k):
    got = max_context_for(q2_layout(), [budget_gib * GIB], spill_all_ffn = True)
    assert expected_k * 1024 <= got < (expected_k + 1) * 1024, got


@pytest.mark.parametrize("budget_gib,expected_k", [(12, 33), (16, 97), (24, 225)])
def test_q2_fully_resident_context_ladder(budget_gib, expected_k):
    got = max_context_for(q2_layout(), [budget_gib * GIB])
    assert expected_k * 1024 <= got < (expected_k + 1) * 1024, got


@pytest.mark.parametrize("budget_gib", [4, 6, 8, 10])
def test_q4_cannot_be_fully_resident_below_18_gib(budget_gib):
    assert max_context_for(q4_layout(), [budget_gib * GIB]) == 0


def test_the_ladder_never_regresses_as_vram_grows():
    """More VRAM must never mean more spill."""
    layout = q4_layout()
    counts = [
        len(plan_placement(layout, [g * GIB], 64 * GIB, 32768).spilled_blocks)
        for g in (8, 10, 12, 14, 16, 18, 20, 22, 24)
    ]
    assert counts == sorted(counts, reverse = True), counts


# ------------------------------------------------------ against the real GGUFs


@needs_gguf
@pytest.mark.parametrize(
    "path,spillable_gib,resident_gib,lm_head_gib,embd_gib,moe",
    [
        (Q2_PATH, 5.291, 2.471, 0.666, 0.389, False),
        (Q4_PATH, 9.888, 4.491, 0.971, 0.666, False),
        (MOE_PATH, 18.320, 1.488, 0.503, 0.503, True),
    ],
)
def test_layout_matches_the_measured_buckets(
    path, spillable_gib, resident_gib, lm_head_gib, embd_gib, moe
):
    layout = layout_from_gguf(path)
    assert layout.complete
    assert layout.is_moe is moe
    assert layout.spillable_bytes / GIB == pytest.approx(spillable_gib, abs = 0.01)
    assert layout.block_resident_bytes / GIB == pytest.approx(resident_gib, abs = 0.01)
    assert layout.lm_head_bytes / GIB == pytest.approx(lm_head_gib, abs = 0.01)
    assert layout.token_embd_bytes / GIB == pytest.approx(embd_gib, abs = 0.01)


@needs_gguf
def test_the_hybrid_cache_is_priced_on_attention_layers_only():
    """qwen35 is 1-in-4 attention, so a naive all-layers cache would be 4x too
    big and would refuse fits that are really there."""
    layout = layout_from_gguf(Q4_PATH)
    assert layout.n_layers == 64
    assert layout.n_attention_layers == 16
    assert layout.kv_bytes_per_token_f16 == 16 * 4 * (256 + 256) * 2
    assert layout.recurrent_bytes / MIB == pytest.approx(149.6, abs = 1.0)


@needs_gguf
def test_a_bad_path_abstains_rather_than_raising():
    assert layout_from_gguf("/nonexistent/model.gguf").complete is False


# ------------------------------------------------- host profile and cost integration


def _dense_q4() -> ModelLayout:
    """Measured Qwen3.8-27B UD-Q4_K_XL buckets."""
    return _uniform_layout(
        spillable_total = int(10.092 * GIB),
        resident_total = int(4.614 * GIB),
        lm_head = int(0.971 * GIB),
        token_embd = int(0.666 * GIB),
    )


def test_a_unified_memory_host_never_spills():
    """Apple Silicon, an AMD APU and a Vulkan iGPU all report host RAM as VRAM,
    so moving a tensor to "host" frees nothing on the device. The planner must
    abstain rather than emit -ot flags that buy no memory and cost real speed."""
    plan = plan_placement(
        _dense_q4(), [8 * GIB], 64 * GIB, 32768,
        opts = PlanOptions(host = HostProfile(unified_memory = True)),
    )
    assert plan.changed is False
    assert plan.ot_patterns == ()
    assert "unified memory" in plan.reason


def test_a_spilling_plan_reports_what_it_will_cost():
    """A plan that spills is not free, and the number has to travel with it."""
    tight = plan_placement(_dense_q4(), [8 * GIB], 64 * GIB, 32768)
    roomy = plan_placement(_dense_q4(), [48 * GIB], 64 * GIB, 32768)
    assert tight.spills_anything and tight.predicted_gen_penalty_ms > 0.0
    assert not roomy.spills_anything and roomy.predicted_gen_penalty_ms == 0.0


def test_a_small_host_is_predicted_to_suffer_more_for_the_same_spill():
    """Spilled decode runs on the CPU backend (ggml migrates an op only at
    batch >= 32, and decode is batch 1), so the penalty tracks core count. A
    desktop must not be told a server's story."""
    layout, vram, ram, ctx = _dense_q4(), [8 * GIB], 64 * GIB, 32768
    big = plan_placement(layout, vram, ram, ctx, opts = PlanOptions(host = HostProfile(threads = 192)))
    small = plan_placement(layout, vram, ram, ctx, opts = PlanOptions(host = HostProfile(threads = 8)))
    assert big.spilled_blocks == small.spilled_blocks, "same placement, different host"
    assert small.predicted_gen_penalty_ms > big.predicted_gen_penalty_ms * 2


def test_routed_experts_are_charged_less_than_a_dense_ffn_of_equal_size():
    """Only n_expert_used of n_expert are read per token, so an offloaded MoE
    moves a fraction of its bytes while a dense FFN moves all of them. This is
    the real reason MoE tolerates spilling (2.5x) and dense does not (5.5x) --
    NOT the mmap penalty ratio, which points the other way."""
    dense = _dense_q4()
    moe = _uniform_layout(
        spillable_total = int(10.092 * GIB),
        resident_total = int(4.614 * GIB),
        lm_head = int(0.971 * GIB),
        token_embd = int(0.666 * GIB),
        is_moe = True,
    )
    moe = ModelLayout(**{**moe.__dict__, "n_expert": 256, "n_expert_used": 8})
    opts = PlanOptions(host = HostProfile(threads = 192))
    d = plan_placement(dense, [8 * GIB], 64 * GIB, 32768, opts = opts)
    m = plan_placement(moe, [8 * GIB], 64 * GIB, 32768, opts = opts)
    assert d.spilled_blocks == m.spilled_blocks, "same bytes spilled either way"
    assert m.predicted_gen_penalty_ms < d.predicted_gen_penalty_ms


@needs_gguf
def test_the_mtp_block_is_not_counted_as_spillable():
    """blk.<nextn> is not part of the target model and llama.cpp does not load it
    unless a draft is engaged, so an -ot pattern naming it moves nothing.

    Measured against the real binary: spilling ONLY that block leaves the host
    buffer at exactly token_embd (682.03 MiB) and the device buffer unchanged at
    15718.48 MiB. Counting it would credit the plan 209 MiB it can never free,
    which is the optimistic direction that claims a fit that is not there.
    """
    layout = layout_from_gguf(Q4_PATH)
    assert layout.n_layers == 64
    assert [b.index for b in layout.blocks] == list(range(64)), "no nextn block"
    assert all(b.index < layout.n_layers for b in layout.blocks)
