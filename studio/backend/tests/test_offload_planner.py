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
from dataclasses import replace

import pytest

from core.inference.offload_layout import (
    LM_HEAD_PATTERN,
    BlockLayout,
    ModelLayout,
    _layout_from_reader,
    layout_from_gguf,
    spill_pattern_for,
)
from core.inference.offload_cost_model import HostProfile
from core.inference.offload_planner import (
    _device_slots,
    _per_device_shortfall,
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
        n_blocks = 40,
        spillable_total = 19_671_285_760,
        resident_total = 1_597_483_520,
        lm_head = 540_000_000,
        token_embd = 540_000_000,
        kv_per_token = 20480,
        is_moe = True,
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
    plan = plan_placement(
        layout,
        [budget],
        64 * GIB,
        4096,
        opts = PlanOptions(
            overhead_bytes_per_device = GIB,
        ),
    )
    spilled = sum(b.spillable_bytes for b in layout.blocks if b.index in plan.spilled_blocks)
    assert spilled == 50 * MIB, "should take the 50 MiB block, not a bigger one"


def test_front_and_back_orders_pick_opposite_ends():
    layout = uneven_layout()
    floor = resident_floor_bytes(layout, 4096)
    budget = floor + layout.spillable_bytes - 40 * MIB + 1 * GIB
    front = plan_placement(
        layout,
        [budget],
        64 * GIB,
        4096,
        opts = PlanOptions(overhead_bytes_per_device = GIB, spill_order = SpillOrder.FRONT_FIRST),
    )
    back = plan_placement(
        layout,
        [budget],
        64 * GIB,
        4096,
        opts = PlanOptions(overhead_bytes_per_device = GIB, spill_order = SpillOrder.BACK_FIRST),
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
    cards are not one 16 GiB card. Read through max_context_for, which is the
    budget arithmetic without the ladder: the same 16 GiB split in two credits
    one overhead less, so it holds strictly less cache."""
    layout = q4_layout()
    one_big = max_context_for(layout, [16 * GIB], spill_all_ffn = True)
    two_small = max_context_for(layout, [8 * GIB, 8 * GIB], spill_all_ffn = True)
    assert one_big > 0 and two_small > 0
    assert two_small < one_big


def test_a_partial_spill_across_two_gpus_abstains():
    """A pooled budget is not a per-device fit test for a partial spill.

    llama.cpp fixes the layer split from free memory BEFORE any override exists
    (llama-model.cpp:1416-1447) and hands each device a contiguous layer-index
    range (:1457), while -ot only swaps one tensor's buffer type inside
    create_tensor (llama-model-loader.cpp:1177-1203) and never touches
    dev_layer(il) (:1467-1474). So a subset of block indices can relieve one
    card while another keeps its whole share: the aggregate deficit reads
    covered, one device is still over, and --fit off means nothing rebalances
    (a per-device shortfall throws, llama-model.cpp:1731-1733). The same layout
    on ONE card of the same pooled size still plans, which is what makes this
    about device COUNT and not about the budget.
    """
    layout = q4_layout()
    one_card = plan_placement(layout, [16 * GIB], 64 * GIB, 8192)
    assert one_card.spilled_blocks
    assert len(one_card.spilled_blocks) < len(layout.blocks), "partial, not everything"

    two_cards = plan_placement(layout, [8 * GIB, 8 * GIB], 64 * GIB, 8192)
    assert two_cards.spilled_blocks == ()
    assert two_cards.ot_patterns == ()
    assert two_cards.changed is False


def test_a_full_spill_is_checked_per_device_not_assumed():
    """A full spill used to be waved through on the theory that "every device
    keeps its layer share". It does keep its ROW share -- llama.cpp splits rows
    in proportion to free VRAM (llama-model.cpp:1439-1457) -- but rows are
    integers and bytes are not: 65 rows over two equal cards is 33/32, so at a
    budget sized to the pooled total device 0 is over by half a row's worth. The
    pooled arithmetic says it fits, the per-device check says it does not, and
    abstaining hands the load to --fit on, which is per-device aware.

    The identical pooled budget on ONE card has no split to be uneven about and
    still plans, which is what makes this about the SPLIT and not the budget.
    """
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    half = _ALL_SPILL_VRAM // 2
    plan = plan_placement(layout, [half, half], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert plan.changed is False
    assert plan.spilled_blocks == ()
    assert "device 0" in plan.reason

    one = plan_placement(layout, [_ALL_SPILL_VRAM], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert len(one.spilled_blocks) == len(layout.blocks)


def test_a_full_spill_abstains_when_the_cache_layout_is_unknown():
    """A hybrid keeps a recurrent state on some layers only, and the layout does
    not record WHICH -- so there is no per-row byte model to validate against.
    Abstain rather than guess uniform. Again scoped to the multi-device split:
    the same layout on one card is unaffected."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    hybrid = replace(layout, recurrent_bytes = 8 * MIB)
    half = _ALL_SPILL_VRAM // 2
    plan = plan_placement(hybrid, [half, half], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert plan.changed is False
    assert "recurrent state" in plan.reason

    one = plan_placement(hybrid, [_ALL_SPILL_VRAM], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert len(one.spilled_blocks) == len(layout.blocks)


def test_the_row_split_matches_llama_cpp():
    """_device_slots is a transcription of llama-model.cpp:1439-1457, so pin the
    two properties that matter: contiguous ranges, and sizes in proportion to
    free VRAM with the remainder landing on the EARLIER device."""
    assert [len(r) for r in _device_slots(65, [8 * GIB, 8 * GIB])] == [33, 32]
    assert [len(r) for r in _device_slots(65, [24 * GIB, 8 * GIB])] == [49, 16]
    assert [len(r) for r in _device_slots(65, [8 * GIB, 8 * GIB, 8 * GIB])] == [22, 22, 21]
    # Contiguous, in device order, covering every row exactly once.
    rows = _device_slots(65, [24 * GIB, 8 * GIB])
    assert rows[0] == list(range(0, 49)) and rows[1] == list(range(49, 65))
    # One device takes everything, and a zero-sized pool does not divide by zero.
    assert _device_slots(65, [8 * GIB]) == [list(range(65))]
    assert _device_slots(4, [0, 0]) == [[0, 1, 2, 3], []]


def test_the_per_device_check_passes_when_the_shares_really_fit():
    """The check must not be a disguised "never plan on two GPUs". Cards sized so
    that each one's row share fits with room to spare return None -- no abstain
    reason -- for the same full spill the tight case rejects."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    spilled = {b.index for b in layout.blocks}
    tight = _ALL_SPILL_VRAM // 2
    assert (
        _per_device_shortfall(
            layout,
            _NO_OVERHEAD,
            4096,
            spilled,
            False,
            [tight, tight],
            quantised = False,
            kv_bytes_floor = 0,
        )
        is not None
    )
    roomy = _ALL_SPILL_VRAM
    assert (
        _per_device_shortfall(
            layout,
            _NO_OVERHEAD,
            4096,
            spilled,
            False,
            [roomy, roomy],
            quantised = False,
            kv_bytes_floor = 0,
        )
        is None
    )


def test_multi_gpu_credit_sums():
    layout = q2_layout()
    assert (
        plan_placement(
            layout, [6 * GIB, 6 * GIB], 64 * GIB, 8192, opts = FIXED_OVERHEAD_OPTS
        ).spilled_blocks
        == ()
    )


# --------------------------------------------------------------- context policy


def test_never_reduce_keeps_the_requested_context():
    layout = q4_layout()
    plan = plan_placement(layout, [12 * GIB], 64 * GIB, 65536)
    assert plan.n_ctx == 65536
    assert plan.spilled_blocks, "it pays with spill, not with the user's context"


def test_prefer_resident_shrinks_instead_of_spilling():
    layout = q4_layout()
    plan = plan_placement(
        layout,
        [18 * GIB],
        64 * GIB,
        65536,
        opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT),
    )
    assert plan.spilled_blocks == ()
    assert plan.n_ctx < 65536
    assert "shrank context" in plan.reason


def test_prefer_resident_still_spills_when_even_min_ctx_will_not_fit():
    layout = q4_layout()
    plan = plan_placement(
        layout,
        [10 * GIB],
        64 * GIB,
        65536,
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
        layout,
        [9 * GIB],
        64 * GIB,
        65536,
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
        q2_layout(),
        [24 * GIB],
        64 * GIB,
        8192,
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
    got = max_context_for(
        q4_layout(), [budget_gib * GIB], spill_all_ffn = True, opts = FIXED_OVERHEAD_OPTS
    )
    assert expected_k * 1024 <= got < (expected_k + 1) * 1024, got


# Budgets and expected contexts here are ARITHMETIC against a stated overhead
# reserve, not measurements, so moving the constant (1 GiB -> 1.5 GiB, to cover
# the prefill compute buffer that was OOMing at depth) does not silently
# invalidate them. The constant itself is pinned by
# test_the_overhead_reserve_covers_the_measured_prefill_buffer.
FIXED_OVERHEAD_OPTS = PlanOptions(overhead_bytes_per_device = GIB)


@pytest.mark.parametrize("budget_gib,expected_k", [(6, 25), (8, 57), (12, 121), (20, 249)])
def test_q2_ffn_spilled_context_ladder(budget_gib, expected_k):
    got = max_context_for(
        q2_layout(), [budget_gib * GIB], spill_all_ffn = True, opts = FIXED_OVERHEAD_OPTS
    )
    assert expected_k * 1024 <= got < (expected_k + 1) * 1024, got


@pytest.mark.parametrize("budget_gib,expected_k", [(12, 33), (16, 97), (24, 225)])
def test_q2_fully_resident_context_ladder(budget_gib, expected_k):
    got = max_context_for(q2_layout(), [budget_gib * GIB], opts = FIXED_OVERHEAD_OPTS)
    assert expected_k * 1024 <= got < (expected_k + 1) * 1024, got


@pytest.mark.parametrize("budget_gib", [4, 6, 8, 10])
def test_q4_cannot_be_fully_resident_below_18_gib(budget_gib):
    assert max_context_for(q4_layout(), [budget_gib * GIB]) == 0


def test_the_ladder_never_regresses_as_vram_grows():
    """More VRAM must never mean more spill."""
    layout = q4_layout()
    counts = [
        len(
            plan_placement(
                layout, [g * GIB], 64 * GIB, 32768, opts = FIXED_OVERHEAD_OPTS
            ).spilled_blocks
        )
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


class _StubField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _StubTensor:
    def __init__(self, name, n_bytes):
        self.name = name
        self.n_bytes = n_bytes


class _StubReader:
    """The handful of attributes ``_layout_from_reader`` touches."""

    def __init__(self, fields, tensors):
        self.fields = {k: _StubField(v) for k, v in fields.items()}
        self.tensors = tensors


def _shard_fields(**extra):
    base = {
        "general.architecture": "llama",
        "llama.block_count": 64,
        "llama.attention.head_count_kv": 8,
        "llama.attention.head_count": 64,
        "llama.embedding_length": 8192,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
        "llama.context_length": 262144,
    }
    base.update(extra)
    return base


def _shard_tensors(indices):
    out = []
    for i in indices:
        out.append(_StubTensor(f"blk.{i}.ffn_up.weight", GIB // 8))
        out.append(_StubTensor(f"blk.{i}.attn_q.weight", MIB * 32))
    return out


def test_a_single_file_gguf_is_still_read():
    """The guard below must not catch an ordinary one-file model."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    assert layout.complete
    assert len(layout.blocks) == 64


def test_a_split_gguf_abstains_instead_of_planning_on_one_shard():
    """GGUFReader memmaps the ONE path it is given, but llama.cpp reads
    split.count off the first shard and loads every sibling
    (llama-model-loader.cpp:590-618). Shard 1 carries the model metadata, so
    without the guard the layout looks complete while holding a fraction of the
    tensors. Undercounting the model is the OPTIMISTIC direction: the plan claims
    a fit that is not there, emits too few -ot patterns, and the launch path
    follows it with --fit off."""
    partial = _StubReader(_shard_fields(**{"split.count": 4}), _shard_tensors(range(16)))
    assert _layout_from_reader(partial).complete is False

    # The undercount it guards against is real: the same shard read as the whole
    # model reports a quarter of the blocks and a quarter of the spillable bytes.
    whole = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    as_if_whole = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(16))))
    assert as_if_whole.spillable_bytes * 4 == whole.spillable_bytes


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
        _dense_q4(),
        [8 * GIB],
        64 * GIB,
        32768,
        opts = PlanOptions(host = HostProfile(unified_memory = True)),
    )
    assert plan.changed is False
    assert plan.ot_patterns == ()
    assert "unified memory" in plan.reason


def test_a_spilling_plan_reports_what_it_will_cost():
    """A plan that spills is not free, and the number has to travel with it."""
    tight = plan_placement(_dense_q4(), [8 * GIB], 64 * GIB, 32768, opts = FIXED_OVERHEAD_OPTS)
    roomy = plan_placement(_dense_q4(), [48 * GIB], 64 * GIB, 32768, opts = FIXED_OVERHEAD_OPTS)
    assert tight.spills_anything and tight.predicted_gen_penalty_ms > 0.0
    assert not roomy.spills_anything and roomy.predicted_gen_penalty_ms == 0.0


def test_a_small_host_is_predicted_to_suffer_more_for_the_same_spill():
    """Spilled decode runs on the CPU backend (ggml migrates an op only at
    batch >= 32, and decode is batch 1), so the penalty tracks core count. A
    desktop must not be told a server's story."""
    layout, vram, ram, ctx = _dense_q4(), [8 * GIB], 64 * GIB, 32768
    big = plan_placement(
        layout,
        vram,
        ram,
        ctx,
        opts = PlanOptions(overhead_bytes_per_device = GIB, host = HostProfile(threads = 192)),
    )
    small = plan_placement(
        layout,
        vram,
        ram,
        ctx,
        opts = PlanOptions(overhead_bytes_per_device = GIB, host = HostProfile(threads = 8)),
    )
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
    opts = PlanOptions(overhead_bytes_per_device = GIB, host = HostProfile(threads = 192))
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


def test_the_overhead_reserve_covers_the_measured_prefill_buffer():
    """The reserve is what the planner leaves free on every device, and it has
    to cover the child's prefill compute buffer plus its CUDA primary context.

    1 GiB did not, and failed CONSISTENTLY rather than randomly, which is what
    made it easy to miss: the planner fills to budget minus this reserve, so
    whatever the budget it leaves exactly this much, and the dense 27B at depth
    32768 died with the identical shortfall at 6, 7, 8 and 10 GiB budgets:

        allocating 594.16 MiB on device 0: cudaMalloc failed: out of memory
        llama_init_from_model: failed to allocate compute pp buffers

    Not fragmentation from the benchmark's VRAM pinning: the same case at 16, 64
    and 1024 MiB hog blocks reproduced the identical 594.16 MiB failure.
    """
    reserve = PlanOptions().overhead_bytes_per_device
    measured_prefill_buffer = int(594.16 * 1024 * 1024)
    # The context consumed the remainder of the old 1 GiB, since 594 MiB could
    # not be allocated inside it.
    inferred_cuda_context = GIB - measured_prefill_buffer
    assert reserve >= measured_prefill_buffer + inferred_cuda_context
    assert reserve > GIB, "1 GiB is the value that OOMed"
    # Bounded: erring high costs spill at 5.544 ms/GiB, so it is not free.
    assert reserve <= 2 * GIB


# ------------------------------------------- excluded blocks and the pool budget


# Just too little VRAM for the 64-block stub at 4096 ctx, so every block spills
# and the planner reaches the all-of-them branch that emits the compact pattern.
_NO_OVERHEAD = PlanOptions(overhead_bytes_per_device = 0)
_ALL_SPILL_VRAM = 3 * GIB + 64 * MIB


def _nextn_reader(nextn: int, total_blocks: int = 66):
    """A GGUF whose block_count includes trailing nextn/MTP blocks.

    llama.cpp reads block_count straight into n_layer_all
    (llama-model.cpp:1206) and n_layer() subtracts n_layer_nextn
    (llama-hparams.cpp:301-303), so the last `nextn` blk.<N> are the MTP head.
    They carry real ffn_* weights, loaded when a draft is engaged
    (models/qwen35moe.cpp, load_block_mtp).
    """
    fields = _shard_fields(
        **{"llama.block_count": total_blocks, "llama.nextn_predict_layers": nextn}
    )
    return _StubReader(fields, _shard_tensors(range(total_blocks)))


def _tied_reader(*, with_output: bool):
    """The 64-block stub plus a vocabulary matrix, tied or untied."""
    tensors = list(_shard_tensors(range(64)))
    tensors.append(_StubTensor("token_embd.weight", GIB))
    if with_output:
        tensors.append(_StubTensor("output.weight", GIB))
    return _StubReader(_shard_fields(), tensors)


def test_a_tied_embedding_gguf_still_charges_a_vocabulary_matrix_to_vram():
    """Omitting output.weight does not save the matrix, it duplicates it.

    Every tying architecture re-creates the output tensor from token_embd with
    TENSOR_DUPLICATED (models/llama.cpp:41-45, models/qwen3.cpp:22-25,
    models/gemma3.cpp:43-47), and the loader routes a duplicated TOKEN_EMBD
    through the OUTPUT buffer list (llama-model-loader.cpp:1113-1114). dev_input
    is pinned to the CPU while dev_output follows the layer split
    (llama-model.cpp:1465, 1474), so the same-context reuse check misses
    (llama-model-loader.cpp:1309-1314), ggml_dup_tensor allocates a second full
    matrix (:1318) and load_all_data fills it by name over PCIe (:1542, :1583).
    Charging it to host RAM only understated VRAM by a whole vocabulary, which
    is the optimistic direction: too few blocks spill, and --fit off pins that.
    """
    tied = _layout_from_reader(_tied_reader(with_output = False))
    untied = _layout_from_reader(_tied_reader(with_output = True))
    assert tied.complete and untied.complete
    assert tied.lm_head_bytes == 0, "there is no output.weight to spill"

    # The VRAM floor is the same either way: a vocabulary matrix is resident in
    # both, it just arrives as a duplicate in the tied case.
    assert resident_floor_bytes(tied, 4096) == resident_floor_bytes(untied, 4096)
    assert all_resident_bytes(tied, 4096) == all_resident_bytes(untied, 4096)

    # And it reaches the decision: the same card spills the same blocks.
    opts = PlanOptions(overhead_bytes_per_device = 0)
    tied_plan = plan_placement(tied, [10 * GIB], 256 * GIB, 4096, opts = opts)
    untied_plan = plan_placement(untied, [10 * GIB], 256 * GIB, 4096, opts = opts)
    assert tied_plan.spilled_blocks, "a partial spill, so lm_head is not in play"
    assert len(tied_plan.spilled_blocks) == len(untied_plan.spilled_blocks)

    # Never as lm_head: the duplicate keeps the name token_embd.weight, so
    # LM_HEAD_PATTERN cannot match it and spilling it would move nothing.
    assert tied_plan.spilled_lm_head is False
    assert LM_HEAD_PATTERN not in tied_plan.ot_patterns

    # token_embd itself is still host RAM the plan pays for, counted once.
    assert tied_plan.host_bytes - untied_plan.host_bytes == 0


def test_excluded_mtp_block_bytes_are_kept_so_a_draft_can_be_charged():
    """Dropping the trailing blocks is right for an ordinary load -- llama.cpp
    gives them TENSOR_SKIP unless load_mtp is set (models/glm4-moe.cpp:42-44)
    and TENSOR_SKIP returns before a tensor exists
    (llama-model-loader.cpp:1123-1131). But --spec-type draft-mtp sets load_mtp
    on the TARGET's model params (common/common.cpp:1713), so the whole trailing
    block becomes resident, and i_gpu_start counting backwards from n_layer_all
    (llama-model.cpp:1449) puts it on a GPU first. The seam charges it through
    extra_resident_bytes, so the total has to survive the drop."""
    layout = _layout_from_reader(_nextn_reader(2))
    assert layout.has_excluded_blocks is True

    per_block = layout.blocks[0].spillable_bytes + layout.blocks[0].resident_bytes
    assert layout.excluded_block_bytes == 2 * per_block

    plain = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    assert plain.excluded_block_bytes == 0

    # Charging it shrinks the budget, so more blocks spill.
    base = PlanOptions(overhead_bytes_per_device = 0)
    charged = PlanOptions(
        overhead_bytes_per_device = 0, extra_resident_bytes = layout.excluded_block_bytes
    )
    without = plan_placement(layout, [10 * GIB], 256 * GIB, 4096, opts = base)
    with_mtp = plan_placement(layout, [10 * GIB], 256 * GIB, 4096, opts = charged)
    assert len(with_mtp.spilled_blocks) > len(without.spilled_blocks)


def test_a_nextn_gguf_is_marked_as_having_excluded_blocks():
    with_mtp = _layout_from_reader(_nextn_reader(2))
    assert with_mtp.complete
    assert with_mtp.has_excluded_blocks is True
    assert [b.index for b in with_mtp.blocks] == list(range(64))

    plain = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    assert plain.has_excluded_blocks is False


def test_the_spill_pattern_never_reaches_an_excluded_mtp_block():
    """Spilling every block used to emit the unbounded ^blk\\.\\d+\\. form, which
    llama.cpp applies with std::regex_search (llama-model-loader.cpp:1182) --
    so it also matched the trailing nextn blocks the layout deliberately
    dropped. Those are real weights once a draft is loaded, and moving them
    spills bytes that neither host_bytes nor the deficit ever counted, then
    runs the draft FFN on the CPU backend.
    """
    layout = _layout_from_reader(_nextn_reader(2))
    plan = plan_placement(layout, [_ALL_SPILL_VRAM], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert len(plan.spilled_blocks) == len(layout.blocks), "every block goes"
    assert len(plan.ot_patterns) >= 1

    pattern = re.compile(plan.ot_patterns[0])
    assert pattern.search("blk.0.ffn_up.weight"), "a target block still spills"
    assert pattern.search("blk.63.ffn_up.weight")
    for excluded in ("blk.64.ffn_up.weight", "blk.65.ffn_down.weight"):
        assert pattern.search(excluded) is None, excluded


def test_a_gguf_without_excluded_blocks_keeps_the_compact_pattern():
    """The bound is only paid where it buys something: with nothing excluded the
    global form is still used, which is the shape the benchmarks measured."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    plan = plan_placement(layout, [_ALL_SPILL_VRAM], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert len(plan.spilled_blocks) == len(layout.blocks)
    assert re.compile(plan.ot_patterns[0]).search("blk.999.ffn_up.weight")


def test_extra_resident_bytes_are_charged_against_the_pooled_budget():
    """GPU-resident bytes outside the layout -- a vision projector, an MTP draft
    reserve -- have to shrink the budget, or the deficit comes out too small on
    a load the caller already judged not to fit."""
    layout = q4_layout()
    base = PlanOptions(overhead_bytes_per_device = GIB)
    charged = PlanOptions(overhead_bytes_per_device = GIB, extra_resident_bytes = 3 * GIB)

    without = plan_placement(layout, [16 * GIB], 128 * GIB, 8192, opts = base)
    with_extra = plan_placement(layout, [16 * GIB], 128 * GIB, 8192, opts = charged)

    assert without.spills_anything and with_extra.spills_anything
    assert len(with_extra.spilled_blocks) > len(without.spilled_blocks)
    # And it reaches the context ladder too, not just the deficit.
    assert max_context_for(layout, [16 * GIB], spill_all_ffn = True, opts = charged) < max_context_for(
        layout, [16 * GIB], spill_all_ffn = True, opts = base
    )


def test_row_ownership_is_modelled_on_raw_free_not_on_the_budget():
    """llama.cpp reads free VRAM straight from the driver for its split
    (llama-model.cpp:1433); the budget is that minus a reserve sized on each
    card's TOTAL, so the two agree only when every card has the same free/total.
    Feeding the budget in as the split weight silently moves the boundary."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    spilled = {b.index for b in layout.blocks}
    budgets = [2 * GIB, 2 * GIB]

    # Same budgets, different RAW free: the rows move, so the verdict may too.
    even = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        4096,
        spilled,
        False,
        budgets,
        quantised = False,
        kv_bytes_floor = 0,
        split_weights_per_device = [8 * GIB, 8 * GIB],
    )
    lopsided = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        4096,
        spilled,
        False,
        budgets,
        quantised = False,
        kv_bytes_floor = 0,
        split_weights_per_device = [1 * GIB, 15 * GIB],
    )
    assert even != lopsided, "the split weights have to reach _device_slots"
    assert "device 1" in (lopsided or ""), "the card drawing 60 of 65 rows is the one over"


def test_a_sliding_window_model_abstains_on_a_multi_gpu_split():
    """Gemma3 and friends interleave window and full-context layers, and EVERY
    layer is an attention layer, so the n_attention_layers guard passes. Spreading
    the cache evenly then under-books whichever card drew the full-context rows,
    which is the optimistic direction."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    swa = replace(layout, has_swa = True)
    assert swa.n_attention_layers == swa.n_layers, "the earlier guard does NOT cover this"

    half = _ALL_SPILL_VRAM // 2
    plan = plan_placement(swa, [half, half], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert plan.changed is False
    assert "sliding-window" in plan.reason

    # One card has no split to mislocate the caches across.
    one = plan_placement(swa, [_ALL_SPILL_VRAM], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert len(one.spilled_blocks) == len(layout.blocks)
