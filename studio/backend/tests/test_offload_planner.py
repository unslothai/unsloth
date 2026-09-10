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
    LLAMA_MAX_LAYERS,
    _layout_from_reader,
    hybrid_layer_split,
    _layout_from_readers,
    layout_from_gguf,
    split_shard_paths,
    spill_pattern_for,
)
from core.inference.offload_cost_model import HostProfile
from core.inference.offload_planner import (  # noqa: F401
    _device_reserve,
    _device_slots,
    _kv_floor_at,
    _per_device_shortfall,
    _per_device_usage,
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


def test_a_multi_device_plan_pins_the_split_it_was_budgeted_on():
    """--fit off means llama.cpp never re-fits, so the split has to be pinned.

    Without -ts the child falls back to the default split, which is the free VRAM
    ggml_backend_dev_memory reads IN THE CHILD (llama-model.cpp:1462-1477). That
    is not the pre-launch snapshot this plan was budgeted on: it is short by at
    least a CUDA primary context per card, and on a 24 GiB plus 10 GiB pair the
    normalised ratio moves far enough to shift a layer boundary the plan assumed.

    Integer layer counts per device, the way common/fit.cpp:555 writes them
    (tensor_split[id] = ngl_per_device[id].n_layer).
    """
    layout = q4_layout()
    # A hybrid layout needs the per-layer cache vector before the per-device
    # arithmetic will run at all; 0 marks the rows that hold recurrent state.
    weights = [1 if i % 4 == 3 else 0 for i in range(layout.n_layers)]
    opts = PlanOptions(trust_device_row_model = True)
    rows = layout.n_layers + 1

    plan = plan_placement(
        layout, [24 * GIB, 10 * GIB], 64 * GIB, 8192, opts = opts, kv_layer_weights = weights
    )
    assert plan.priced and plan.changed
    assert sum(plan.device_layer_counts) == rows, "every row is owned by exactly one device"
    assert plan_to_args(plan)[-2:] == [
        "--tensor-split",
        ",".join(str(n) for n in plan.device_layer_counts),
    ]

    # The counts REPRODUCE the assignment they were read off, rather than merely
    # resembling it: llama.cpp prefix-sums and normalises whatever it is given, so
    # a device whose share is c owns exactly c rows.
    assert _device_slots(rows, plan.device_layer_counts) == _device_slots(
        rows, [24 * GIB, 10 * GIB]
    )

    # One device: no split to pin, and #28218 measured a 20x slowdown from -ts on
    # a single GPU.
    single = plan_placement(layout, [24 * GIB], 64 * GIB, 8192, opts = opts)
    assert single.device_layer_counts == ()
    assert "--tensor-split" not in plan_to_args(single)


def test_a_plan_that_is_never_emitted_pins_no_split():
    """An abstention leaves llama.cpp's own placement alone. Pinning a split onto
    it would be a launch of its own, and plan_to_args must stay empty."""
    layout = q4_layout()
    starved = plan_placement(layout, [4 * GIB, 4 * GIB], 64 * GIB, 131072)
    assert starved.insufficient is True
    assert starved.device_layer_counts == ()
    assert plan_to_args(starved) == []


def test_plan_to_args_shape():
    layout = q4_layout()
    args = plan_to_args(plan_placement(layout, [12 * GIB], 64 * GIB, 8192))
    assert args.count("-ot") == 1
    assert args[args.index("-ot") + 1].endswith("=CPU")
    assert "--load-mode" in args and args[args.index("--load-mode") + 1] == "none"


# ---------------------------------------------------------------- spill selection


def test_largest_first_minimises_overshoot():
    """Best-fit-decreasing: cover a small residual with a small block, not by dragging a 400 MiB
    block across the bus on every token.
    """
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
            spill_order = SpillOrder.LARGEST_FIRST,
        ),
    )
    spilled = sum(b.spillable_bytes for b in layout.blocks if b.index in plan.spilled_blocks)
    assert spilled == 50 * MIB, "should take the 50 MiB block, not a bigger one"


def test_the_default_order_is_back_first():
    """On a fixture where the two orders DISAGREE: the smallest block is at the front and
    the last block is twice its size. The default takes the last block; LARGEST_FIRST,
    asked for by name, takes the small one."""
    sizes = [50 * MIB, 200 * MIB, 400 * MIB, 100 * MIB]
    layout = ModelLayout(
        **{
            **uneven_layout().__dict__,
            "blocks": tuple(
                BlockLayout(index = i, spillable_bytes = s, resident_bytes = 10 * MIB)
                for i, s in enumerate(sizes)
            ),
        }
    )
    floor = resident_floor_bytes(layout, 4096)
    budget = floor + layout.spillable_bytes - 40 * MIB + 1 * GIB
    default = plan_placement(
        layout, [budget], 64 * GIB, 4096, opts = PlanOptions(overhead_bytes_per_device = GIB)
    )
    assert default.spilled_blocks == (3,), default.reason
    minimal = plan_placement(
        layout,
        [budget],
        64 * GIB,
        4096,
        opts = PlanOptions(overhead_bytes_per_device = GIB, spill_order = SpillOrder.LARGEST_FIRST),
    )
    assert minimal.spilled_blocks == (0,), minimal.reason


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


def test_a_safe_partial_spill_across_two_gpus_is_planned():
    sizes = [GIB // 2, GIB // 2, GIB // 2, 2 * GIB]
    layout = ModelLayout(
        arch = "qwen35",
        n_layers = 4,
        n_attention_layers = 4,
        blocks = tuple(
            BlockLayout(index = i, spillable_bytes = size, resident_bytes = GIB // 10)
            for i, size in enumerate(sizes)
        ),
        lm_head_bytes = GIB // 10,
        token_embd_bytes = 0,
        kv_bytes_per_token_f16 = 0,
        recurrent_bytes = 0,
        n_ctx_train = 4096,
        complete = True,
    )
    opts = PlanOptions(
        overhead_bytes_per_device = GIB,
        pipeline_overhead_bytes = GIB,
        host_ram_headroom_bytes = 0,
    )

    plan = plan_placement(
        layout,
        [23 * GIB // 10, 22 * GIB // 10],
        64 * GIB,
        4096,
        opts = opts,
    )

    assert plan.changed is True
    assert plan.spilled_blocks == (2, 3)


def test_partial_spill_selection_covers_each_device_shortfall():
    sizes = [1200 * MIB, 0, 0, 600 * MIB, 600 * MIB]
    layout = ModelLayout(
        arch = "qwen35",
        n_layers = 5,
        n_attention_layers = 5,
        blocks = tuple(
            BlockLayout(index = i, spillable_bytes = size, resident_bytes = 0)
            for i, size in enumerate(sizes)
        ),
        lm_head_bytes = 0,
        token_embd_bytes = 0,
        kv_bytes_per_token_f16 = 0,
        recurrent_bytes = 0,
        n_ctx_train = 4096,
        complete = True,
    )
    plan = plan_placement(
        layout,
        [1300 * MIB, 100 * MIB],
        64 * GIB,
        4096,
        opts = PlanOptions(overhead_bytes_per_device = 0, host_ram_headroom_bytes = 0),
        split_weights_per_device = [1, 1],
    )

    assert plan.spilled_blocks == (3, 4)


def test_per_device_selection_cannot_drop_cache_remainder_from_the_pool():
    spillable = [17, 4, 14, 20, 3, 5, 17, 17]
    resident = [1, 5, 5, 1, 6, 6, 5, 0]
    layout = ModelLayout(
        arch = "qwen35",
        n_layers = 8,
        n_attention_layers = 8,
        blocks = tuple(
            BlockLayout(index = i, spillable_bytes = spill, resident_bytes = keep)
            for i, (spill, keep) in enumerate(zip(spillable, resident))
        ),
        lm_head_bytes = 0,
        token_embd_bytes = 0,
        other_resident_bytes = 8,
        kv_bytes_per_token_f16 = 3,
        recurrent_bytes = 0,
        n_ctx_train = 4096,
        complete = True,
    )
    opts = PlanOptions(
        overhead_bytes_per_device = 8,
        pipeline_overhead_bytes = 5,
        extra_resident_bytes = 4,
        host_ram_headroom_bytes = 0,
    )
    plan = plan_placement(
        layout,
        [90, 25],
        1024,
        5,
        opts = opts,
        split_weights_per_device = [56, 27],
    )

    assert not plan.changed or plan.vram_bytes <= 90


def test_output_device_shortfall_can_reach_the_lm_head_rung():
    layout = ModelLayout(
        arch = "qwen35",
        n_layers = 3,
        n_attention_layers = 3,
        blocks = tuple(
            BlockLayout(index = i, spillable_bytes = size, resident_bytes = 0)
            for i, size in enumerate([100, 100, 10])
        ),
        lm_head_bytes = 100,
        token_embd_bytes = 0,
        kv_bytes_per_token_f16 = 0,
        recurrent_bytes = 0,
        n_ctx_train = 4096,
        complete = True,
    )
    plan = plan_placement(
        layout,
        [100, 20],
        1024,
        1,
        opts = PlanOptions(overhead_bytes_per_device = 0, host_ram_headroom_bytes = 0),
        split_weights_per_device = [1, 1],
    )

    assert plan.spilled_blocks == (1,)
    assert plan.spilled_lm_head is True
    assert plan.vram_bytes <= 120


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
    assert _device_slots(65, [8 * GIB]) == [list(range(65))]
    # An all-zero split is not a placement llama.cpp produces, since it prefix-sums the
    # shares and divides by the total, so it is refused rather than modelled.
    with pytest.raises(ValueError):
        _device_slots(4, [0, 0])


def test_the_row_split_uses_llama_cpp_float32_boundaries():
    rows = _device_slots(353, [39407 * MIB, 12114 * MIB])
    assert len(rows[0]) == 270
    assert 270 in rows[1]


def test_a_float32_split_boundary_cannot_approve_an_oom():
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = 2 * GIB,
            resident_bytes = 2 * GIB if i == 270 else 0,
        )
        for i in range(352)
    )
    layout = ModelLayout(
        arch = "qwen35",
        n_layers = 352,
        n_attention_layers = 352,
        blocks = blocks,
        lm_head_bytes = 0,
        token_embd_bytes = 0,
        kv_bytes_per_token_f16 = 0,
        recurrent_bytes = 0,
        n_ctx_train = 4096,
        complete = True,
    )
    plan = plan_placement(
        layout,
        [2 * GIB, GIB],
        1024 * GIB,
        4096,
        opts = PlanOptions(overhead_bytes_per_device = 0, host_ram_headroom_bytes = 0),
        split_weights_per_device = [39407 * MIB, 12114 * MIB],
    )

    assert plan.changed is False
    assert "device 1" in plan.reason


def test_the_per_device_check_passes_when_the_shares_really_fit():
    """The check must not be a disguised "never plan on two GPUs". Cards sized so
    that each one's row share fits with room to spare return None -- no abstain
    reason -- for the same full spill the tight case rejects."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    spilled = {b.index: b.spillable_bytes for b in layout.blocks}
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


def test_the_per_device_check_charges_each_secondary_pipeline_reserve():
    layout = uneven_layout()
    spilled = {b.index for b in layout.blocks}
    cache_per_layer = layout.kv_bytes(4096, 2) // layout.n_layers
    device_one_used = layout.blocks[3].resident_bytes + cache_per_layer + layout.lm_head_bytes
    pipeline_reserve = GIB
    opts = PlanOptions(overhead_bytes_per_device = 0, pipeline_overhead_bytes = pipeline_reserve)

    below = _per_device_shortfall(
        layout,
        opts,
        4096,
        spilled,
        False,
        [4 * GIB, device_one_used + pipeline_reserve - 1],
        quantised = False,
        kv_bytes_floor = 0,
        split_weights_per_device = [1, 1],
    )
    exact = _per_device_shortfall(
        layout,
        opts,
        4096,
        spilled,
        False,
        [4 * GIB, device_one_used + pipeline_reserve],
        quantised = False,
        kv_bytes_floor = 0,
        split_weights_per_device = [1, 1],
    )
    above = _per_device_shortfall(
        layout,
        opts,
        4096,
        spilled,
        False,
        [4 * GIB, device_one_used + pipeline_reserve + 1],
        quantised = False,
        kv_bytes_floor = 0,
        split_weights_per_device = [1, 1],
    )

    assert below is not None and "device 1" in below
    assert exact is None
    assert above is None


def test_an_empty_secondary_still_has_to_fit_its_fixed_reserves():
    layout = uneven_layout()
    spilled = {b.index for b in layout.blocks}
    pipeline_reserve = GIB
    opts = PlanOptions(overhead_bytes_per_device = 0, pipeline_overhead_bytes = pipeline_reserve)

    below = _per_device_shortfall(
        layout,
        opts,
        4096,
        spilled,
        True,
        [4 * GIB, pipeline_reserve - 1],
        quantised = False,
        kv_bytes_floor = 0,
        split_weights_per_device = [1000, 1],
    )
    exact = _per_device_shortfall(
        layout,
        opts,
        4096,
        spilled,
        True,
        [4 * GIB, pipeline_reserve],
        quantised = False,
        kv_bytes_floor = 0,
        split_weights_per_device = [1000, 1],
    )

    assert _device_slots(layout.n_layers + 1, [1000, 1])[1] == []
    assert below is not None and "device 1" in below
    assert exact is None


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
        opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT, overhead_bytes_per_token = 0),
    )
    assert plan.spilled_blocks, "shrinking cannot save this one, so spill"


def test_only_the_default_context_reads_the_training_window():
    """MOVED: this pinned a clamp of every request to n_ctx_train."""
    layout = q4_layout()
    assert plan_placement(layout, [24 * GIB], 128 * GIB, 999_999).n_ctx == 999_999
    assert plan_placement(layout, [24 * GIB], 128 * GIB, 0).n_ctx == layout.n_ctx_train


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
    pinned = PlanOptions(overhead_bytes_per_token = 0)
    without = plan_placement(layout, [9 * GIB], 64 * GIB, 65536, opts = pinned)
    assert without.insufficient is True

    with_quant = plan_placement(
        layout,
        [9 * GIB],
        64 * GIB,
        65536,
        opts = PlanOptions(allow_kv_quant = True, overhead_bytes_per_token = 0),
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
FIXED_OVERHEAD_OPTS = PlanOptions(overhead_bytes_per_device = GIB, overhead_bytes_per_token = 0)
# The context-linear part of the reserve is pinned to zero here for the same reason the flat
# part is pinned to 1 GiB: these ladders are arithmetic about the CACHE, and the term itself
# is pinned by test_the_reserve_grows_with_context_and_max_context_agrees.


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


def test_a_block_count_above_llama_cpp_s_layer_cap_abstains(monkeypatch):
    """llama.cpp asserts n_layer_all <= LLAMA_MAX_LAYERS (512) in load_hparams, so a file
    declaring more is one the child will refuse. The readers used to allocate per-layer lists
    off the declared count first: a public GGUF naming 2**40 blocks beside a two-entry
    head_count_kv list would have had the backend building a 2**40-element list before
    llama.cpp saw the file. Both readers abstain above the cap instead."""
    # A plain dense file one block over the cap, every block present, so the ONLY thing
    # stopping a complete layout is the cap itself.
    over = LLAMA_MAX_LAYERS + 1
    fields = _shard_fields(**{"llama.block_count": over})
    layout = _layout_from_reader(_StubReader(fields, _shard_tensors(range(over))))
    assert not layout.complete
    fields = _shard_fields(**{"llama.block_count": LLAMA_MAX_LAYERS})
    layout = _layout_from_reader(_StubReader(fields, _shard_tensors(range(LLAMA_MAX_LAYERS))))
    assert layout.complete and len(layout.blocks) == LLAMA_MAX_LAYERS
    # The hybrid split, which the KV estimator also calls, abstains on its own.
    assert hybrid_layer_split("qwen35", 100_000, n_kv_head = [8, 0]) == (0, 0, False)
    # And the reader refuses BEFORE asking the split: the per-layer padding after the
    # split call is sized off the declared count too, so the reader's own cap is the
    # only thing keeping a 2**40 file from being expanded there.
    import core.inference.offload_layout as _layout_mod

    def _never(*a, **k):
        raise AssertionError("the layout asked the split about a file above the cap")

    monkeypatch.setattr(_layout_mod, "hybrid_layer_split", _never)
    fields = _shard_fields(**{"llama.block_count": over})
    assert not _layout_from_reader(_StubReader(fields, _shard_tensors(range(64)))).complete
    # The cap itself is still readable: one layer under it is a normal file.
    assert hybrid_layer_split("qwen35", LLAMA_MAX_LAYERS, n_kv_head = [8, 0])[2] is True


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


def _qwen3next_fields(**extra):
    """Qwen3-Next-80B-A3B's header, whose GGUF carries ssm.* and NO interval."""
    base = {
        "general.architecture": "qwen3next",
        "qwen3next.block_count": 48,
        "qwen3next.attention.head_count_kv": 2,
        "qwen3next.attention.head_count": 16,
        "qwen3next.embedding_length": 2048,
        "qwen3next.attention.key_length": 256,
        "qwen3next.attention.value_length": 256,
        "qwen3next.context_length": 262144,
        "qwen3next.expert_count": 512,
        "qwen3next.expert_used_count": 10,
        "qwen3next.ssm.inner_size": 4096,
        "qwen3next.ssm.state_size": 128,
        "qwen3next.ssm.conv_kernel": 4,
        "qwen3next.ssm.group_count": 16,
    }
    base.update(extra)
    return base


def _hybrid_tensors(n = 48):
    return [_StubTensor(f"blk.{i}.attn_q.weight", MIB) for i in range(n)]


def test_a_hybrid_without_the_interval_key_uses_the_architecture_default():
    """llama.cpp defaults full_attention_interval per ARCHITECTURE and only then treats every
    layer as attention (models/qwen3next.cpp:24). Reading the absent key as 0 made Qwen3-Next
    look like 48 GQA layers with no state, a 3.6x over-count of the cache."""
    layout = _layout_from_reader(_StubReader(_qwen3next_fields(), _hybrid_tensors()))
    assert layout.complete
    assert layout.n_attention_layers == 12
    assert layout.kv_bytes(131072) == 3072 * MIB
    assert layout.recurrent_bytes > 0

    spelled = _layout_from_reader(
        _StubReader(
            _qwen3next_fields(**{"qwen3next.full_attention_interval": 4}), _hybrid_tensors()
        )
    )
    assert spelled.n_attention_layers == layout.n_attention_layers
    assert spelled.recurrent_bytes == layout.recurrent_bytes


def test_an_explicit_recurrent_layer_mask_beats_the_interval():
    """attention.recurrent_layers is what llama.cpp reads FIRST, and it wins over
    both the key and the default (models/qwen3next.cpp:21)."""
    mask = [0] * 48
    for i in (7, 15, 23, 31):
        mask[i] = 1
    layout = _layout_from_reader(
        _StubReader(
            _qwen3next_fields(
                **{
                    "qwen3next.attention.recurrent_layers": mask,
                    "qwen3next.full_attention_interval": 4,
                }
            ),
            _hybrid_tensors(),
        )
    )
    assert layout.n_attention_layers == 44


def test_ssm_keys_with_no_recurrent_map_abstain_instead_of_reading_all_attention():
    """An architecture this file has no default for, whose GGUF states ssm.* and
    nothing else: the layout cannot say which rows are recurrent, and calling them
    all attention is the 3.6x over-count above. Abstain, as the sharded case does."""
    fields = {k.replace("qwen3next", "mysteryhybrid"): v for k, v in _qwen3next_fields().items()}
    fields["general.architecture"] = "mysteryhybrid"
    assert _layout_from_reader(_StubReader(fields, _hybrid_tensors())).complete is False

    plain = {k: v for k, v in fields.items() if ".ssm." not in k}
    assert _layout_from_reader(_StubReader(plain, _hybrid_tensors())).complete is True


def test_nemotron_h_mlp_only_rows_are_not_charged_a_recurrent_state():
    """A row is recurrent on nemotron_h only when its KV heads AND its FFN width
    are both 0 (models/nemotron-h.cpp:17). Its MLP-only rows have zero heads and a
    real FFN, so counting every zero-head row charged 6 states where 4 exist."""
    heads = [0, 0, 0, 8, 0, 0, 0, 8]
    fields = {
        "general.architecture": "nemotron_h",
        "nemotron_h.block_count": 8,
        "nemotron_h.attention.head_count_kv": heads,
        "nemotron_h.attention.head_count": 32,
        "nemotron_h.feed_forward_length": [0, 0, 4096, 0, 0, 0, 4096, 0],
        "nemotron_h.embedding_length": 4096,
        "nemotron_h.attention.key_length": 128,
        "nemotron_h.attention.value_length": 128,
        "nemotron_h.ssm.inner_size": 4096,
        "nemotron_h.ssm.state_size": 128,
        "nemotron_h.ssm.conv_kernel": 4,
        "nemotron_h.ssm.group_count": 8,
    }
    layout = _layout_from_reader(_StubReader(fields, _hybrid_tensors(8)))
    one_state = (3 * (4096 + 2 * 8 * 128) + 128 * 4096) * 4
    assert layout.complete
    assert layout.n_attention_layers == 2
    assert layout.recurrent_bytes == 4 * one_state

    other = {k.replace("nemotron_h", "jamba"): v for k, v in fields.items()}
    other["general.architecture"] = "jamba"
    assert _layout_from_reader(_StubReader(other, _hybrid_tensors(8))).recurrent_bytes == (
        6 * one_state
    )


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
        opts = PlanOptions(
            overhead_bytes_per_device = GIB,
            overhead_bytes_per_token = 0,
            host = HostProfile(threads = 192),
        ),
    )
    small = plan_placement(
        layout,
        vram,
        ram,
        ctx,
        opts = PlanOptions(
            overhead_bytes_per_device = GIB,
            overhead_bytes_per_token = 0,
            host = HostProfile(threads = 8),
        ),
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
    opts = PlanOptions(
        overhead_bytes_per_device = GIB, overhead_bytes_per_token = 0, host = HostProfile(threads = 192)
    )
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


def test_the_context_reserve_matches_what_was_measured_on_hardware():
    """The flat term was right and incomplete: the reserve also grows with context."""
    opts = PlanOptions()
    mib = 1024 * 1024

    assert _device_reserve(opts, 8192) == opts.overhead_bytes_per_device
    assert _device_reserve(opts, 32768) == opts.overhead_bytes_per_device
    assert _device_reserve(opts, opts.overhead_free_ctx) == opts.overhead_bytes_per_device

    assert _device_reserve(opts, 33792) / mib > 1184
    assert _device_reserve(opts, 66560) / mib > 1932.5
    assert _device_reserve(opts, 132096) / mib > 1958.1

    assert _device_reserve(opts, 66560) / mib < 1932.5 + 1024
    assert _device_reserve(opts, 132096) / mib < 1958.1 + 3 * 1024

    kib = 1024
    assert 18.8 * kib <= opts.overhead_bytes_per_token <= 23.4 * kib


def test_the_context_reserve_declines_a_load_the_flat_one_accepted():
    """The behaviour change, stated once and directly: on an 8.5 GiB card at 64K the flat
    reserve leaves room for a q8_0 cache and the load is planned, while the context-aware one
    correctly finds it does not fit."""
    layout = q4_layout()
    flat = plan_placement(
        layout,
        [8704 * 1024 * 1024],
        64 * GIB,
        65536,
        opts = PlanOptions(allow_kv_quant = True, overhead_bytes_per_token = 0),
    )
    aware = plan_placement(
        layout, [8704 * 1024 * 1024], 64 * GIB, 65536, opts = PlanOptions(allow_kv_quant = True)
    )
    assert flat.insufficient is False, "the flat reserve accepted this cell"
    assert aware.insufficient is True, "the context reserve must not"

    a = plan_placement(
        layout, [8704 * 1024 * 1024], 64 * GIB, 8192, opts = PlanOptions(overhead_bytes_per_token = 0)
    )
    b = plan_placement(layout, [8704 * 1024 * 1024], 64 * GIB, 8192)
    assert a.ot_patterns == b.ot_patterns and a.n_ctx == b.n_ctx


def test_max_context_for_is_consistent_with_the_reserve_it_charges():
    """The reserve depends on the context and the context is what is being solved for, so a
    single pass answers with the reserve for some OTHER context and hands back a context whose
    own reserve no longer leaves room for its own cache. Feed the answer back in and it fits."""
    layout = q4_layout()
    for budget_gib in (12, 16, 24, 48):
        vram = [budget_gib * GIB]
        ctx = max_context_for(layout, vram, spill_all_ffn = True)
        if ctx <= 0:
            continue
        assert max_context_for(layout, vram, spill_all_ffn = True) == ctx

        flat_only = max_context_for(
            layout, vram, spill_all_ffn = True, opts = PlanOptions(overhead_bytes_per_token = 0)
        )
        assert ctx <= flat_only, "the context term can only reduce the answer, never raise it"
        if ctx > PlanOptions().overhead_free_ctx and flat_only < layout.n_ctx_train:
            assert ctx < flat_only, "and above the free context it must actually bite"


# ------------------------------------------- excluded blocks and the pool budget


# Just too little VRAM for the 64-block stub at 4096 ctx, so every block spills
# and the planner reaches the all-of-them branch that emits the compact pattern.
_NO_OVERHEAD = PlanOptions(overhead_bytes_per_device = 0, overhead_bytes_per_token = 0)
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

    # The UNION, not patterns[0]: boundary grading emits the last block taken as its own
    # pattern, so a single-pattern assertion reads that block as missing.
    def spills(tensor: str) -> bool:
        return any(re.compile(p).search(tensor) for p in plan.ot_patterns)

    assert spills("blk.0.ffn_up.weight"), "a target block still spills"
    assert spills("blk.63.ffn_up.weight")
    for excluded in ("blk.64.ffn_up.weight", "blk.65.ffn_down.weight"):
        assert not spills(excluded), excluded


def test_every_spill_pattern_is_bounded_to_blocks_the_layout_knows():
    """The unbounded ``^blk\\.\\d+\\.`` form is gone, including where nothing is excluded."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64))))
    plan = plan_placement(layout, [_ALL_SPILL_VRAM], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert len(plan.spilled_blocks) == len(layout.blocks)

    def spills(tensor: str) -> bool:
        return any(re.compile(p).search(tensor) for p in plan.ot_patterns)

    assert spills("blk.0.ffn_up.weight")
    assert spills("blk.63.ffn_up.weight"), "the boundary block is covered too"
    assert not spills("blk.999.ffn_up.weight"), "no block outside the layout"


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
    spilled = {b.index: b.spillable_bytes for b in layout.blocks}
    budgets = [2 * GIB, 2 * GIB]

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

    one = plan_placement(swa, [_ALL_SPILL_VRAM], 256 * GIB, 4096, opts = _NO_OVERHEAD)
    assert one.changed is False
    assert one.spilled_blocks == ()
    assert "sliding-window" in one.reason

    measured = plan_placement(
        swa,
        [_ALL_SPILL_VRAM],
        256 * GIB,
        4096,
        opts = _NO_OVERHEAD,
        kv_bytes_floor = 48 * MIB,
    )
    assert measured.changed is True
    assert "sliding-window" not in measured.reason
    assert 0 < len(measured.spilled_blocks) <= len(layout.blocks)


def _moe_reader(names):
    """A 2-block MoE GGUF whose expert tensors use ``names``."""
    fields = dict(_shard_fields())
    fields["llama.block_count"] = 2
    fields["llama.expert_count"] = 64
    fields["llama.expert_used_count"] = 8
    tensors = []
    for i in range(2):
        for n in names:
            tensors.append(_StubTensor(f"blk.{i}.{n}.weight", 400 * MIB))
        tensors.append(_StubTensor(f"blk.{i}.attn_q.weight", 10 * MIB))
    tensors.append(_StubTensor("token_embd.weight", 100 * MIB))
    tensors.append(_StubTensor("output.weight", 100 * MIB))
    return _StubReader(fields, tensors)


@pytest.mark.parametrize(
    "label,names,expect_mib",
    [
        ("split", ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"], 1200),
        # Fused gate+up: the same two matrices under one tensor name.
        ("fused", ["ffn_gate_up_exps", "ffn_down_exps"], 800),
        # grovemoe's chunked experts, one tensor per chunk-expert.
        ("chunked", ["ffn_up_chexps", "ffn_gate_chexps", "ffn_down_chexps"], 1200),
    ],
)
def test_every_expert_spelling_is_spillable(label, names, expect_mib):
    """ffn_gate_up_exps and ffn_*_chexps are experts under another name: created
    per expert and dispatched with GGML_OP_MUL_MAT_ID, so just as cheap to spill.
    Matching only the split form left every fused-expert GGUF with nothing the
    planner was allowed to move."""
    layout = _layout_from_reader(_moe_reader(names))
    assert layout.is_moe
    assert layout.blocks[0].spillable_bytes == expect_mib * MIB

    # The pattern must move exactly what the layout counted.
    pattern = spill_pattern_for(layout, None)
    for n in names:
        assert re.search(pattern, f"blk.0.{n}.weight"), f"{label}: {n} not matched"


def test_routed_latent_projections_are_never_spilled():
    """kimi-k3's ffn_routed_up/down read like experts and are not: no expert axis
    and plain GGML_OP_MUL_MAT, so every token crosses them. Spilling one puts a
    hot tensor on the host at the rate the cost model reserves for cold ones."""
    layout = _layout_from_reader(_moe_reader(["ffn_routed_up", "ffn_routed_down"]))
    assert layout.blocks[0].spillable_bytes == 0
    pattern = spill_pattern_for(layout, None)
    assert not re.search(pattern, "blk.0.ffn_routed_up.weight")
    assert not re.search(pattern, "blk.0.ffn_routed_down.weight")


def _swa_layout(n_blocks = 64):
    """Every layer is attention AND the cache is per-layer uneven -- the shape
    n_attention_layers cannot describe."""
    layout = _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(n_blocks))))
    return replace(layout, has_swa = True)


def test_a_per_layer_vector_replaces_the_sliding_window_abstain():
    """Gemma3 interleaves 5:1, so a window layer's cache is a fraction of a
    full-context one. Without a vector the planner would have to spread the cache
    evenly, so it refuses; with one it knows where the big caches land."""
    layout = _swa_layout()
    half = _ALL_SPILL_VRAM // 2
    spilled = {b.index: b.spillable_bytes for b in layout.blocks}

    without = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        4096,
        spilled,
        False,
        [half, half],
        quantised = False,
        kv_bytes_floor = 0,
    )
    assert without is not None and "sliding-window" in without

    # 1 full-attention layer in every 5, the rest windowed at a 64th of the cost.
    weights = [64 if (i % 5 == 0) else 1 for i in range(layout.n_layers)]
    with_vector = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        4096,
        spilled,
        False,
        [half, half],
        quantised = False,
        kv_bytes_floor = 0,
        kv_layer_weights = weights,
    )
    assert with_vector is None or "sliding-window" not in with_vector


def test_the_vector_places_the_cache_it_does_not_resize_it():
    """Scaled to the total the caller already priced, so changing only its SHAPE
    moves the cache between devices without changing how much cache there is."""
    layout = _swa_layout()
    spilled = {b.index: b.spillable_bytes for b in layout.blocks}
    n = layout.n_layers
    budgets = [_ALL_SPILL_VRAM // 2, _ALL_SPILL_VRAM // 2]

    # All the cache on the rows device 0 owns, then all on device 1's.
    front = [1] * (n // 2) + [0] * (n - n // 2)
    back = [0] * (n // 2) + [1] * (n - n // 2)
    a = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        32768,
        spilled,
        False,
        budgets,
        quantised = False,
        kv_bytes_floor = 8 * GIB,
        kv_layer_weights = front,
    )
    b = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        32768,
        spilled,
        False,
        budgets,
        quantised = False,
        kv_bytes_floor = 8 * GIB,
        kv_layer_weights = back,
    )
    assert a is not None and b is not None
    assert "device 0" in a, "the front-loaded cache overflows the first card"
    assert "device 1" in b, "the back-loaded cache overflows the second"


def test_a_wrong_length_vector_is_ignored_rather_than_trusted():
    """A vector of the wrong length is not evidence: abstain, do not stretch it."""
    layout = _swa_layout()
    spilled = {b.index: b.spillable_bytes for b in layout.blocks}
    half = _ALL_SPILL_VRAM // 2
    got = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        4096,
        spilled,
        False,
        [half, half],
        quantised = False,
        kv_bytes_floor = 0,
        kv_layer_weights = [1, 2, 3],
    )
    assert got is not None and "sliding-window" in got


def test_every_shard_handed_over_prices_the_whole_split():
    """Two readers summing to one layout, token_embd in its bucket whichever shard holds it."""
    first = _StubReader(
        _shard_fields(**{"split.count": 2, "split.no": 0}),
        _shard_tensors(range(32)) + [_StubTensor("output.weight", GIB)],
    )
    second = _StubReader(
        _shard_fields(**{"split.count": 2, "split.no": 1}),
        _shard_tensors(range(32, 64)) + [_StubTensor("token_embd.weight", 3 * GIB)],
    )
    layout = _layout_from_readers([first, second])
    assert layout.complete
    assert len(layout.blocks) == 64
    assert layout.token_embd_bytes == 3 * GIB
    assert layout.lm_head_bytes == GIB


def test_a_partial_shard_set_still_abstains():
    readers = [
        _StubReader(_shard_fields(**{"split.count": 3}), _shard_tensors(range(32))),
        _StubReader(_shard_fields(**{"split.count": 3}), _shard_tensors(range(32, 64))),
    ]
    assert not _layout_from_readers(readers).complete


def test_split_shard_paths_follow_llama_cpp_naming():
    paths = split_shard_paths("/m/model-Q4-00002-of-00003.gguf")
    assert paths == [
        "/m/model-Q4-00001-of-00003.gguf",
        "/m/model-Q4-00002-of-00003.gguf",
        "/m/model-Q4-00003-of-00003.gguf",
    ]
    assert split_shard_paths("/m/model-Q4.gguf") is None


def test_layout_from_gguf_reads_the_sibling_shards_only_when_asked(tmp_path, monkeypatch):
    import gguf

    shards = [tmp_path / f"m-{i:05d}-of-00002.gguf" for i in (1, 2)]
    for shard in shards:
        shard.write_bytes(b"GGUF")
    by_path = {
        str(shards[0]): _StubReader(_shard_fields(**{"split.count": 2}), _shard_tensors(range(32))),
        str(shards[1]): _StubReader(
            _shard_fields(**{"split.count": 2}), _shard_tensors(range(32, 64))
        ),
    }
    monkeypatch.setattr(gguf, "GGUFReader", lambda path: by_path[str(path)])

    assert not layout_from_gguf(str(shards[0])).complete
    whole = layout_from_gguf(str(shards[0]), all_shards = True)
    assert whole.complete and len(whole.blocks) == 64

    shards[1].unlink()
    assert not layout_from_gguf(str(shards[0]), all_shards = True).complete


# The owner's priority order for what a launch KEEPS: generation speed, context, the prompt
# cache, the MTP / speculative draft, --parallel, the projector resident.

_R_CTX = 4096
_R_OPTS = dict(overhead_bytes_per_device = GIB, overhead_bytes_per_token = 0)


def _card_short_by(
    layout: ModelLayout,
    short: int,
    *,
    floor: int = 0,
    n_seq: int = 1,
) -> int:
    """A single card whose usable budget is ``short`` bytes below the resident load."""
    needed = all_resident_bytes(layout, _R_CTX, kv_bytes_floor = floor, n_seq = n_seq)
    return needed + GIB - short


def test_rung0_pins_the_projector_before_touching_a_block():
    layout = q4_layout()
    mmproj = 600 * MIB
    card = _card_short_by(layout, 100 * MIB) + mmproj
    kept = plan_placement(
        layout,
        [card],
        64 * GIB,
        _R_CTX,
        opts = PlanOptions(**_R_OPTS, mmproj_bytes = mmproj, mmproj_movable = False),
    )
    assert kept.spills_anything and not kept.mmproj_to_host
    moved = plan_placement(
        layout,
        [card],
        64 * GIB,
        _R_CTX,
        opts = PlanOptions(**_R_OPTS, mmproj_bytes = mmproj, mmproj_movable = True),
    )
    assert moved.mmproj_to_host
    assert not moved.spills_anything, moved.reason
    assert moved.changed and moved.reshapes_launch
    assert "--no-mmproj-offload" in plan_to_args(moved)
    assert "projector" in moved.reason


def test_rung1_steps_the_slot_count_down_one_slot_at_a_time():
    """Short by a quarter of the cache plus a hair: three slots suffice, so three it is, not one."""
    layout = q4_layout()
    floor = GIB
    table = {4: floor, 3: 3 * floor // 4, 2: floor // 2, 1: floor // 4}
    card = _card_short_by(layout, floor // 4 + MIB, floor = floor, n_seq = 4)
    plan = plan_placement(
        layout,
        [card],
        64 * GIB,
        _R_CTX,
        kv_bytes_floor = floor,
        opts = PlanOptions(**_R_OPTS, n_parallel = 4, kv_bytes_floor_by_parallel = table),
    )
    assert plan.n_parallel == 3, plan.reason
    assert not plan.spills_anything
    assert plan_to_args(plan)[-2:] == ["--parallel", "3"]


def test_rung1_respects_min_parallel():
    layout = q4_layout()
    floor = GIB
    table = {4: floor, 3: 3 * floor // 4, 2: floor // 2, 1: floor // 4}
    card = _card_short_by(layout, floor // 2 + MIB, floor = floor, n_seq = 4)
    plan = plan_placement(
        layout,
        [card],
        64 * GIB,
        _R_CTX,
        kv_bytes_floor = floor,
        opts = PlanOptions(**_R_OPTS, n_parallel = 4, min_parallel = 3, kv_bytes_floor_by_parallel = table),
    )
    assert plan.n_parallel == 3
    assert plan.spills_anything, plan.reason


def test_rung1_is_skipped_on_a_sliding_window_cache_without_the_map():
    """The layout's product has no window term, so nothing in the planner can
    price an SWA cache at a different slot count. Without the caller's map the
    rung is skipped and the ladder moves blocks instead of guessing."""
    layout = replace(q4_layout(), has_swa = True)
    floor = GIB
    card = _card_short_by(layout, floor // 4 + MIB, floor = floor, n_seq = 4)
    plan = plan_placement(
        layout,
        [card],
        64 * GIB,
        _R_CTX,
        kv_bytes_floor = floor,
        opts = PlanOptions(**_R_OPTS, n_parallel = 4),
    )
    assert plan.n_parallel == 0
    assert plan.spills_anything, plan.reason
    with_map = plan_placement(
        layout,
        [card],
        64 * GIB,
        _R_CTX,
        kv_bytes_floor = floor,
        opts = PlanOptions(
            **_R_OPTS,
            n_parallel = 4,
            kv_bytes_floor_by_parallel = {3: 3 * floor // 4, 2: floor // 2, 1: floor // 4},
        ),
    )
    assert with_map.n_parallel == 3 and not with_map.spills_anything


def test_the_linear_slot_fallback_never_undercuts_the_layout_product():
    """Without a map a non-SWA floor scales linearly in slots; cache_bytes still
    takes the max against the layout's own product, so the guess can only
    over-reserve."""
    from core.inference.offload_planner import cache_bytes

    layout = q4_layout()
    opts = PlanOptions(n_parallel = 4)
    for slots in (1, 2, 3):
        scaled = _kv_floor_at(layout, opts, 64 * MIB, _R_CTX, _R_CTX, slots)
        assert scaled is not None
        assert cache_bytes(layout, _R_CTX, kv_bytes_floor = scaled) >= layout.kv_bytes(_R_CTX)
    assert _kv_floor_at(replace(layout, has_swa = True), opts, 64 * MIB, _R_CTX, _R_CTX, 2) is None
    assert (
        _kv_floor_at(replace(layout, has_swa = True), opts, 64 * MIB, _R_CTX, _R_CTX, 4) == 64 * MIB
    )


def test_the_recurrent_state_is_charged_per_slot():
    layout = q4_layout()
    one = all_resident_bytes(layout, _R_CTX, n_seq = 1)
    four = all_resident_bytes(layout, _R_CTX, n_seq = 4)
    assert four - one == 3 * layout.recurrent_bytes


def test_rung2_drops_the_draft_only_after_the_slots_are_exhausted():
    layout = q4_layout()
    floor = GIB
    draft = 700 * MIB
    table = {2: floor, 1: floor // 2}
    card = _card_short_by(layout, floor // 2 - MIB, floor = floor, n_seq = 2) + draft
    common = dict(
        **_R_OPTS,
        n_parallel = 2,
        kv_bytes_floor_by_parallel = table,
        draft_bytes = draft,
        draft_droppable = True,
    )
    slot_first = plan_placement(
        layout, [card], 64 * GIB, _R_CTX, kv_bytes_floor = floor, opts = PlanOptions(**common)
    )
    assert slot_first.n_parallel == 1 and not slot_first.draft_dropped, slot_first.reason
    assert not slot_first.spills_anything
    pinned = plan_placement(
        layout,
        [card],
        64 * GIB,
        _R_CTX,
        kv_bytes_floor = floor,
        opts = PlanOptions(**{**common, "min_parallel": 2}),
    )
    assert pinned.n_parallel == 0 and pinned.draft_dropped, pinned.reason
    assert not pinned.spills_anything
    assert "--parallel" not in plan_to_args(pinned)


def test_rungs_0_to_2_alone_are_ranked_against_the_launch_the_caller_typed():
    """These used to skip the gate outright, on the argument that a plan with no spill has no
    fallback arm. It has one: the launch as the caller typed it, fitted by llama.cpp, so both
    figures are computed and reported rather than left at zero."""
    layout = q4_layout()
    mmproj = 600 * MIB
    card = _card_short_by(layout, 100 * MIB) + mmproj
    plan = plan_placement(
        layout,
        [card],
        64 * GIB,
        _R_CTX,
        opts = PlanOptions(
            **_R_OPTS, mmproj_bytes = mmproj, mmproj_movable = True, require_cost_win = True
        ),
    )
    assert plan.mmproj_to_host and plan.changed
    assert plan.predicted_request_ms == 0.0, "nothing is on the host to charge for"
    assert plan.predicted_fit_request_ms > 0.0, "the fitter's arm is priced"
    assert not plan.declined_by_gate


def test_the_prompt_cache_bound_is_reported_only_when_it_binds():
    layout = q4_layout()
    card = _card_short_by(layout, -GIB)  # roomy: nothing spilled
    roomy = plan_placement(layout, [card], 64 * GIB, _R_CTX, opts = PlanOptions(**_R_OPTS))
    assert roomy.load_mode_none and roomy.cache_ram_mib == -1
    assert "--cache-ram" not in plan_to_args(roomy)
    opts = PlanOptions(**_R_OPTS)
    tight_host = opts.host_ram_headroom_bytes + roomy.host_bytes + GIB
    tight = plan_placement(layout, [card], tight_host, _R_CTX, opts = opts)
    assert tight.load_mode_none
    assert tight.cache_ram_mib == 1024, tight.cache_ram_mib
    assert plan_to_args(tight)[-2:] == ["--cache-ram", "1024"]
    cramped = plan_placement(layout, [card], opts.host_ram_headroom_bytes, _R_CTX, opts = opts)
    assert not cramped.load_mode_none and cramped.cache_ram_mib == -1


def test_max_context_for_honours_the_measured_floor():
    layout = q4_layout()
    bare = max_context_for(layout, [16 * GIB], spill_all_ffn = True, opts = FIXED_OVERHEAD_OPTS)
    product_at = layout.kv_bytes(32768)
    doubled = max_context_for(
        layout,
        [16 * GIB],
        spill_all_ffn = True,
        opts = FIXED_OVERHEAD_OPTS,
        kv_bytes_floor = 2 * product_at,
        floor_ctx = 32768,
    )
    assert 0 < doubled < bare
    assert abs(doubled - bare // 2) <= 1024, (bare, doubled)
    swa = replace(layout, has_swa = True)
    flat_a = max_context_for(
        swa,
        [16 * GIB],
        spill_all_ffn = True,
        opts = FIXED_OVERHEAD_OPTS,
        kv_bytes_floor = GIB,
        floor_ctx = 32768,
    )
    flat_b = max_context_for(
        swa,
        [16 * GIB],
        spill_all_ffn = True,
        opts = FIXED_OVERHEAD_OPTS,
        kv_bytes_floor = GIB,
        floor_ctx = 4096,
    )
    assert flat_a == flat_b
    assert flat_a >= doubled


def test_the_shrink_prices_the_budget_at_the_shrunk_context():
    """At 131072 the reserve's context term alone eats the card; at a shorter context the same card
    holds the load with every FFN spilled.
    """
    layout = q4_layout()
    opts = PlanOptions(context_policy = ContextPolicy.FIT_ONLY)
    plan = plan_placement(layout, [10 * GIB], 64 * GIB, 131072, opts = opts)
    assert plan.changed and not plan.insufficient, plan.reason
    assert 4096 <= plan.n_ctx < 131072, plan.n_ctx


def _mixed_card_vision_layout():
    """A tail-heavy dense model: the pooled budget fits, one card's rows do not."""
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = (700 * MIB if i < 25 else 800 * MIB),
            resident_bytes = (217 * MIB if i < 25 else 224 * MIB),
        )
        for i in range(32)
    )
    return ModelLayout(
        arch = "qwen35",
        n_layers = 32,
        n_attention_layers = 32,
        blocks = blocks,
        lm_head_bytes = 512 * MIB,
        token_embd_bytes = 256 * MIB,
        kv_bytes_per_token_f16 = 4096,
        recurrent_bytes = 0,
        n_ctx_train = 262144,
        complete = True,
    )


def test_a_knob_only_fit_is_checked_device_by_device():
    """Moving the projector can make the POOLED budget fit while the small card stays over."""
    layout = _mixed_card_vision_layout()
    opts = PlanOptions(
        overhead_bytes_per_device = 1 * GIB,
        pipeline_overhead_bytes = 0,
        mmproj_bytes = 3 * GIB,
        mmproj_movable = True,
        n_parallel = 1,
    )
    vram = [24 * GIB, 8 * GIB]
    assert all_resident_bytes(layout, 8192) > 27 * GIB
    assert all_resident_bytes(layout, 8192) <= 30 * GIB
    assert (
        _per_device_shortfall(
            layout,
            opts,
            8192,
            {},
            False,
            vram,
            quantised = False,
            kv_bytes_floor = 0,
            split_weights_per_device = vram,
            extra_on_device0 = 0,
        )
        is not None
    )

    plan = plan_placement(layout, vram, 128 * GIB, 8192, opts = opts)
    assert plan.mmproj_to_host and plan.spilled_blocks, plan.reason
    assert "device 1" in plan.reason
    assert all(index >= 25 for index in plan.spilled_blocks), plan.spilled_blocks
    assert (
        _per_device_shortfall(
            layout,
            opts,
            8192,
            {i: layout.blocks[i].spillable_bytes for i in plan.spilled_blocks},
            False,
            vram,
            quantised = False,
            kv_bytes_floor = 0,
            split_weights_per_device = vram,
            extra_on_device0 = 0,
        )
        is None
    )


def _bound_layout(*, resident_per_block: int, has_swa: bool = False):
    blocks = tuple(
        BlockLayout(index = i, spillable_bytes = 180 * MIB, resident_bytes = resident_per_block)
        for i in range(48)
    )
    return ModelLayout(
        arch = "gemma3" if has_swa else "qwen35",
        n_layers = 48,
        n_attention_layers = 48,
        blocks = blocks,
        lm_head_bytes = 600 * MIB,
        token_embd_bytes = 600 * MIB,
        kv_bytes_per_token_f16 = 98304,
        recurrent_bytes = 0,
        n_ctx_train = 131072,
        has_swa = has_swa,
        complete = True,
    )


def test_the_context_bound_leaves_a_host_held_cache_in_host_ram():
    """-nkvo holds both caches in host RAM, so no context charges them to VRAM."""
    layout = _bound_layout(resident_per_block = 130 * MIB)
    opts = PlanOptions(
        overhead_bytes_per_device = 1 * GIB,
        context_policy = ContextPolicy.FIT_ONLY,
        kv_on_host = True,
        min_ctx = 4096,
        ctx_step = 1024,
    )
    vram = [8 * GIB]

    bound = max_context_for(layout, vram, spill_all_ffn = True, opts = opts, kv_on_host = True)
    assert resident_floor_bytes(layout, bound, kv_on_host = True) <= _usable_vram_for(
        vram, opts, bound
    )
    assert bound > 40960

    plan = plan_placement(layout, vram, 256 * GIB, 131072, opts = opts)
    assert plan.n_ctx > 65536
    assert not plan.insufficient


def test_a_windowed_cache_does_not_cap_the_context_bound_at_the_naive_product():
    """``cache_at`` charges a measured SWA floor context-FLAT, so the layout's full-context product
    is not an upper bound on the answer.
    """
    layout = _bound_layout(resident_per_block = 100 * MIB, has_swa = True)
    opts = PlanOptions(
        overhead_bytes_per_device = 1 * GIB,
        context_policy = ContextPolicy.FIT_ONLY,
        min_ctx = 4096,
        ctx_step = 1024,
    )
    vram = [8 * GIB]
    floor = 400 * MIB

    bound = max_context_for(
        layout, vram, spill_all_ffn = True, opts = opts, kv_bytes_floor = floor, floor_ctx = 131072
    )
    assert bound > 65536
    assert resident_floor_bytes(layout, bound, kv_bytes_floor = floor) <= _usable_vram_for(
        vram, opts, bound
    )

    plan = plan_placement(layout, vram, 128 * GIB, 131072, opts = opts, kv_bytes_floor = floor)
    assert plan.n_ctx > 65536
    assert not plan.insufficient


def _usable_vram_for(vram, opts, n_ctx):
    from core.inference.offload_planner import _usable_vram
    return _usable_vram(vram, opts, n_ctx)


def test_a_resident_fit_below_the_requested_context_is_a_change():
    """The context ladder can settle on a context where every tensor stays resident."""
    blocks = tuple(
        BlockLayout(index = i, spillable_bytes = 0, resident_bytes = 128 * MIB) for i in range(32)
    )
    layout = ModelLayout(
        arch = "qwen3",
        n_layers = 32,
        n_attention_layers = 32,
        blocks = blocks,
        lm_head_bytes = 256 * MIB,
        token_embd_bytes = 256 * MIB,
        kv_bytes_per_token_f16 = 64 * 1024,
        n_ctx_train = 131072,
        complete = True,
    )
    opts = PlanOptions(context_policy = ContextPolicy.FIT_ONLY, allow_lm_head_spill = False)
    resident = max_context_for(layout, [8 * GIB], opts = opts)
    assert opts.min_ctx <= resident < 131072, resident
    plan = plan_placement(layout, [8 * GIB], None, 131072, opts = opts)
    assert not plan.insufficient, plan.reason
    assert not plan.ot_patterns and not plan.load_mode_none
    assert plan.n_ctx == resident
    assert plan.changed


def test_the_context_bound_assumes_the_rungs_above_the_first_spill():
    """The ladder's upper bound charged the projector and the draft in full and priced the
    cache at the full slot count, so a large projector made the bound zero and the ladder never
    looked. The bound now assumes those rungs, which is what the retry will do."""
    blocks = tuple(
        BlockLayout(index = i, spillable_bytes = 0, resident_bytes = 128 * MIB) for i in range(32)
    )
    layout = ModelLayout(
        arch = "qwen3",
        n_layers = 32,
        n_attention_layers = 32,
        blocks = blocks,
        lm_head_bytes = 256 * MIB,
        token_embd_bytes = 256 * MIB,
        kv_bytes_per_token_f16 = 64 * 1024,
        n_ctx_train = 131072,
        complete = True,
    )
    opts = PlanOptions(
        context_policy = ContextPolicy.FIT_ONLY,
        allow_lm_head_spill = False,
        mmproj_bytes = 3 * GIB,
        mmproj_movable = True,
    )
    assert max_context_for(layout, [8 * GIB], opts = opts) == 0
    plan = plan_placement(layout, [8 * GIB], 64 * GIB, 131072, opts = opts)
    assert plan.changed and not plan.insufficient, plan.reason
    assert plan.mmproj_to_host
    assert opts.min_ctx <= plan.n_ctx < 131072, plan.n_ctx


def test_a_host_held_recurrent_state_is_charged_once_per_slot():
    """-nkvo puts the recurrent state in host RAM, one copy per slot; the plan's
    host bytes carried one copy, so the load-mode rule and the prompt-cache
    clamp were short by the other slots' state."""
    import dataclasses

    layout = dataclasses.replace(q4_layout(), recurrent_bytes = GIB)

    def at(slots):
        return plan_placement(
            layout,
            [16 * GIB],
            64 * GIB,
            8192,
            opts = PlanOptions(kv_on_host = True, n_parallel = slots, min_parallel = slots),
        )

    one, four = at(1), at(4)
    assert one.ot_patterns == four.ot_patterns
    assert four.host_bytes - one.host_bytes == 3 * GIB


def test_prefer_resident_tests_the_footprint_the_slots_really_serve():
    """The guard priced one copy of the recurrent state while the launch serves
    one per slot, so a load whose single copy fit skipped the resident search
    and spilled weights where a shorter context would have kept everything on
    the card."""
    import dataclasses

    ctx = 32768
    layout = dataclasses.replace(
        q4_layout(), recurrent_bytes = GIB, kv_bytes_per_token_f16 = 4 * GIB / ctx
    )
    o = PlanOptions(
        context_policy = ContextPolicy.PREFER_RESIDENT,
        n_parallel = 4,
        min_parallel = 4,
        overhead_bytes_per_device = GIB,
        overhead_bytes_per_token = 0,
    )
    single = all_resident_bytes(layout, ctx, n_seq = 1)
    card = single + GIB + 512 * MIB
    assert all_resident_bytes(layout, ctx, n_seq = 4) > card - GIB
    plan = plan_placement(layout, [card], 64 * GIB, ctx, opts = o)
    assert plan.changed and not plan.insufficient, plan.reason
    assert not plan.ot_patterns, plan.ot_patterns
    assert 0 < plan.n_ctx < ctx, plan.n_ctx
    assert all_resident_bytes(layout, plan.n_ctx, n_seq = 4) <= card - GIB


def test_the_context_bound_releases_the_dense_ffn_rung_too():
    """Shared and dense FFN inside a MoE block are a rung of their own and sit in
    resident_bytes, so a bound that released only the expert bytes sat below the
    context the ladder could reach once that rung fired."""
    with_shared = graded_moe_with_shared(0.2)
    without = graded_moe_with_shared(0.0)
    bound = max_context_for(with_shared, [11 * GIB], spill_all_ffn = True)
    assert bound > 0
    assert bound == max_context_for(without, [11 * GIB], spill_all_ffn = True)


def graded_moe_with_shared(shexp_gib: float) -> ModelLayout:
    d, u, g = int(0.16 * GIB), int(0.16 * GIB), int(0.15 * GIB)
    attn, shexp = int(0.025 * GIB), int(shexp_gib * GIB)
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = d + u + g,
            resident_bytes = attn + shexp,
            ffn_down_bytes = d,
            ffn_up_bytes = u,
            ffn_gate_bytes = g,
            dense_ffn_bytes = shexp,
            attn_bytes = attn,
        )
        for i in range(40)
    )
    return ModelLayout(
        arch = "qwen3moe",
        n_layers = 40,
        n_attention_layers = 40,
        blocks = blocks,
        lm_head_bytes = int(0.5 * GIB),
        token_embd_bytes = int(0.5 * GIB),
        other_resident_bytes = int(0.01 * GIB),
        kv_bytes_per_token_f16 = 0.62 * GIB / 32768,
        n_ctx_train = 32768,
        is_moe = True,
        n_expert = 128,
        n_expert_used = 8,
        complete = True,
    )


def test_the_per_device_check_places_the_recurrent_state_on_the_rows_that_hold_it():
    """The pooled fit charges the recurrent state in full, one copy per slot; the
    per-device check charged only block weights and the attention cache, so a
    hybrid split across cards could pass device by device while the card holding
    the recurrent rows was over, and llama.cpp threw on that card's allocation."""
    import dataclasses

    layout = dataclasses.replace(
        _swa_layout(), arch = "qwen35", has_swa = False, recurrent_bytes = 2 * GIB, n_attention_layers = 4
    )
    n = layout.n_layers
    weights = [1 if i % (n // 4) == 0 else 0 for i in range(n)]
    spilled = {b.index: b.spillable_bytes for b in layout.blocks}

    def usage(
        lay,
        n_seq = 1,
        vector = weights,
    ):
        _err, used, _slots = _per_device_usage(
            lay,
            _NO_OVERHEAD,
            4096,
            spilled,
            False,
            [GIB, GIB],
            quantised = False,
            kv_bytes_floor = 0,
            kv_layer_weights = vector,
            n_seq = n_seq,
        )
        return used

    one, four = usage(layout), usage(layout, n_seq = 4)
    assert abs(sum(four) - sum(one) - 3 * 2 * GIB) <= n
    bare = usage(dataclasses.replace(layout, recurrent_bytes = 0))
    budgets = [b + 64 * MIB for b in bare]
    short = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        4096,
        spilled,
        False,
        budgets,
        quantised = False,
        kv_bytes_floor = 0,
        kv_layer_weights = weights,
    )
    assert short is not None and "device" in short
    uniform = _per_device_shortfall(
        layout,
        _NO_OVERHEAD,
        4096,
        spilled,
        False,
        budgets,
        quantised = False,
        kv_bytes_floor = 0,
        kv_layer_weights = [1] * n,
    )
    assert uniform is not None and "recurrent state" in uniform


def test_the_context_ladder_tests_the_minimum_context_before_giving_up(monkeypatch):
    """The lattice steps down from the refused context and lands on min_ctx only
    by coincidence: 8960 refused, minus a 1024 step, is 7936, below an 8192
    minimum, so the ladder returned the refusal without ever asking about 8192."""
    from core.inference import offload_planner as planner
    from core.inference.offload_planner import ContextPolicy, Plan

    layout = _bound_layout(resident_per_block = 100 * MIB)
    seen = []

    def fake(_layout, _opts, ctx, *args, **kwargs):
        seen.append(ctx)
        if ctx == 8192:
            return Plan(changed = True, n_ctx = ctx, ot_patterns = ("x",), spilled_blocks = (1,))
        return Plan(n_ctx = ctx, declined_by_gate = True, reason = "gate")

    monkeypatch.setattr(planner, "_plan_at", fake)
    opts = PlanOptions(
        overhead_bytes_per_device = GIB,
        context_policy = ContextPolicy.FIT_ONLY,
        min_ctx = 8192,
        ctx_step = 1024,
    )
    plan = plan_placement(layout, [8 * GIB], 256 * GIB, 8960, opts = opts)
    assert seen[0] == 8960
    assert seen[-1] == 8192
    assert plan.n_ctx == 8192 and not plan.declined_by_gate


def test_the_boundary_block_is_left_whole_when_a_later_rung_closed_the_deficit():
    """Once the expert rung runs out and dense units follow, grading the last
    expert block against a deficit the dense units already cover keeps cheap
    routed-expert bytes resident so dearer dense bytes can stay spilled. The
    rung that closed the gap is the only one that may be graded."""
    from core.inference.offload_cost_model import HostProfile

    layout = graded_moe_with_shared(0.4)
    opts = PlanOptions(host = HostProfile(threads = 6))
    plan = plan_placement(layout, [5 * GIB], 94 * GIB, 8192, opts = opts)
    assert plan.spills_anything, plan.reason
    dense = [p for p in plan.ot_patterns if "_shexp" in p]
    assert dense, plan.ot_patterns
    graded_expert = [p for p in plan.ot_patterns if "_shexp" not in p and r"\d+" not in p]
    assert not graded_expert, plan.ot_patterns


def test_an_explicit_context_above_the_training_window_is_priced_as_asked():
    """llama-server serves a -c above n_ctx_train, so a plan clamped to the window
    prices a cache the child does not run at; the seam then rewrote the user's -c
    or, with the context in the extras, launched --fit off under a spill sized for
    the smaller cache. Explicit means as asked; only the default reads the window."""
    layout = _bound_layout(resident_per_block = 100 * MIB)
    opts = PlanOptions(overhead_bytes_per_device = GIB)
    asked = 2 * layout.n_ctx_train
    plan = plan_placement(layout, [40 * GIB], 256 * GIB, asked, opts = opts)
    assert plan.n_ctx == asked, plan
    at_window = plan_placement(layout, [40 * GIB], 256 * GIB, layout.n_ctx_train, opts = opts)
    assert at_window.n_ctx == layout.n_ctx_train
    assert plan.host_bytes > at_window.host_bytes or (
        plan.insufficient and not at_window.insufficient
    )
    assert plan_placement(layout, [40 * GIB], 256 * GIB, 0, opts = opts).n_ctx == layout.n_ctx_train


def test_a_resident_context_plan_is_priced_at_the_slots_it_was_checked_at():
    """PREFER_RESIDENT charged one recurrent state per slot when it checked that
    the requested context does not fit, then assembled the plan with no slot
    state at all, so vram_bytes under-reported a hybrid by (slots - 1) states."""
    from core.inference.offload_planner import all_resident_bytes

    layout = q4_layout()
    assert layout.recurrent_bytes > 0
    plan = plan_placement(
        layout,
        [22 * GIB],
        64 * GIB,
        65536,
        opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT, n_parallel = 4),
    )
    assert "shrank context" in plan.reason and plan.priced and plan.n_ctx < 65536
    assert plan.vram_bytes == all_resident_bytes(layout, plan.n_ctx, n_seq = 4)
    assert (
        plan.vram_bytes - all_resident_bytes(layout, plan.n_ctx, n_seq = 1)
        == 3 * layout.recurrent_bytes
    )


def test_only_a_priced_plan_says_the_load_fits():
    """A fit and an abstain both carry an n_ctx and a reason; only the fit was priced."""
    fits = plan_placement(q4_layout(), [200 * GIB], 64 * GIB, 65536)
    assert fits.priced and not fits.spills_anything and "fits in VRAM" in fits.reason
    abstained = plan_placement(
        q4_layout(),
        [200 * GIB],
        64 * GIB,
        65536,
        opts = PlanOptions(host = HostProfile(threads = 8, unified_memory = True)),
    )
    assert not abstained.priced and "unified memory" in abstained.reason


def test_a_cache_already_quantised_at_launch_is_priced_as_one():
    """A launch with --cache-type-kv q8_0 handed the planner only the measured
    byte floor, and its first, normally only, mode was f16: cache_bytes took the
    larger f16 product over the floor, and a model whose weights plus q8 cache
    are fully resident was several GiB over budget and spilled for nothing."""
    from core.inference.offload_planner import cache_bytes

    layout = q4_layout()
    n_ctx = 65536
    q8_floor = cache_bytes(layout, n_ctx, kv_quantised = True)
    f16 = cache_bytes(layout, n_ctx)
    assert q8_floor < f16
    budget = all_resident_bytes(layout, n_ctx, kv_quantised = True) + 1024 * MIB
    assert all_resident_bytes(layout, n_ctx) > budget
    flat = dict(overhead_bytes_per_device = 0, overhead_bytes_per_token = 0)
    as_floor_only = plan_placement(
        layout, [budget], 64 * GIB, n_ctx, kv_bytes_floor = q8_floor, opts = PlanOptions(**flat)
    )
    assert as_floor_only.spills_anything
    as_quantised = plan_placement(
        layout,
        [budget],
        64 * GIB,
        n_ctx,
        kv_bytes_floor = q8_floor,
        opts = PlanOptions(cache_quantised = True, kv_quant_type = "q8_0", **flat),
    )
    assert not as_quantised.spills_anything and as_quantised.priced, as_quantised.reason
    assert as_quantised.cache_type_k is None and as_quantised.cache_type_v is None


def test_prefer_resident_prices_a_quantised_launch_cache_as_one():
    """The PREFER_RESIDENT branch runs ahead of _kv_modes and priced the cache as
    f16 over a q8 floor, shrinking a context that fit fully resident with the
    cache type the child runs."""
    from core.inference.offload_planner import cache_bytes

    layout = q4_layout()
    n_ctx = 65536
    q8_floor = cache_bytes(layout, n_ctx, kv_quantised = True)
    budget = all_resident_bytes(layout, n_ctx, kv_quantised = True) + 1024 * MIB
    assert all_resident_bytes(layout, n_ctx) > budget
    flat = dict(
        context_policy = ContextPolicy.PREFER_RESIDENT,
        overhead_bytes_per_device = 0,
        overhead_bytes_per_token = 0,
    )
    shrunk = plan_placement(
        layout, [budget], 64 * GIB, n_ctx, kv_bytes_floor = q8_floor, opts = PlanOptions(**flat)
    )
    assert shrunk.n_ctx < n_ctx and "shrank context" in shrunk.reason
    kept = plan_placement(
        layout,
        [budget],
        64 * GIB,
        n_ctx,
        kv_bytes_floor = q8_floor,
        opts = PlanOptions(cache_quantised = True, kv_quant_type = "q8_0", **flat),
    )
    assert kept.n_ctx == n_ctx and not kept.spills_anything, kept.reason


def test_an_unbounded_prompt_cache_keeps_the_plan_pageable():
    """--cache-ram -1 bounds the prompt cache by nothing, so no figure charged for
    it is a ceiling: the RAM proof behind --load-mode none abstains and the clamp
    is never derived, whatever host RAM reads."""
    layout = q4_layout()
    bounded = plan_placement(layout, [12 * GIB], 256 * GIB, 32768)
    assert bounded.spills_anything and bounded.load_mode_none
    unbounded = plan_placement(
        layout, [12 * GIB], 256 * GIB, 32768, opts = PlanOptions(prompt_cache_unbounded = True)
    )
    assert unbounded.spills_anything and not unbounded.load_mode_none
    assert unbounded.cache_ram_mib == -1
    assert not unbounded.declined_by_gate


def test_an_mla_cache_trusts_the_measured_floor():
    """MLA keeps one compressed K-only latent per token; the layout's per-head
    K+V product over-counts it by up to two orders of magnitude, and taking the
    maximum let that product override the seam's byte-accurate floor, so a fully
    resident full-context load read as over budget and was shrunk or spilled."""
    from core.inference.offload_planner import cache_bytes

    n_ctx = 65536
    gqa = q4_layout()
    mla = replace(gqa, has_mla = True)
    product = gqa.kv_bytes(n_ctx)
    latent = product // 64
    assert cache_bytes(gqa, n_ctx, kv_bytes_floor = latent) == product
    assert cache_bytes(mla, n_ctx, kv_bytes_floor = latent) == latent
    assert cache_bytes(mla, n_ctx) == product
    flat = PlanOptions(overhead_bytes_per_device = 0, overhead_bytes_per_token = 0)
    budget = all_resident_bytes(mla, n_ctx) - product + latent + 256 * MIB
    assert plan_placement(
        gqa, [budget], 64 * GIB, n_ctx, kv_bytes_floor = latent, opts = flat
    ).spills_anything
    got = plan_placement(mla, [budget], 64 * GIB, n_ctx, kv_bytes_floor = latent, opts = flat)
    assert not got.spills_anything and got.n_ctx == n_ctx, got.reason


def test_layout_from_gguf_marks_a_latent_attention_cache():
    # Both MLA head lengths, as llama_hparams::is_mla keys it; the LoRA rank alone
    # is the DeepSeek-R1 shape and gets the full per-head K+V cache instead.
    fields = _shard_fields(
        **{
            "llama.attention.kv_lora_rank": 512,
            "llama.attention.key_length_mla": 192,
            "llama.attention.value_length_mla": 128,
        }
    )
    assert _layout_from_reader(_StubReader(fields, _shard_tensors(range(64)))).has_mla is True
    assert (
        _layout_from_reader(_StubReader(_shard_fields(), _shard_tensors(range(64)))).has_mla
        is False
    )


def test_a_fit_across_a_split_is_checked_device_by_device_even_with_nothing_given_up():
    """A no-spill plan is emitted as ``-ngl -1 --fit off`` whenever it reshapes the launch, and
    a context the ladder shrank reshapes it as much as a knob does; the per-device check ran
    only when a knob had been given up, so a pooled fit could pin a row split one card cannot
    hold. What this pins is that the check RUNS and the split that comes out of it holds."""
    layout = _mixed_card_vision_layout()
    opts = PlanOptions(overhead_bytes_per_device = 1 * GIB, pipeline_overhead_bytes = 0)
    vram = [24 * GIB, 8 * GIB]
    assert all_resident_bytes(layout, 8192) <= 30 * GIB  # the pool says yes
    split = plan_placement(layout, vram, 128 * GIB, 8192, opts = opts)
    assert split.priced and split.spilled_blocks, split.reason
    assert "device 1" in split.reason, split.reason
    assert all(index >= 25 for index in split.spilled_blocks), split.spilled_blocks
    whole = plan_placement(layout, [32 * GIB], 128 * GIB, 8192, opts = opts)
    assert whole.priced and "fits in VRAM" in whole.reason


# One scalar floor plus a scaling rule cannot describe a hybrid's fixed state, a windowed
# cache's two halves or an MLA latent at the same time.


def _small_layout(**kw) -> ModelLayout:
    """A tiny complete layout: the cache, not the weights, is what these move."""
    fields = dict(
        arch = "qwen3",
        n_layers = 8,
        n_attention_layers = 8,
        blocks = tuple(BlockLayout(i, int(0.25 * GIB), int(0.0125 * GIB)) for i in range(8)),
        lm_head_bytes = int(0.1 * GIB),
        token_embd_bytes = int(0.1 * GIB),
        other_resident_bytes = int(0.01 * GIB),
        kv_bytes_per_token_f16 = 1024,
        n_ctx_train = 32768,
        complete = True,
    )
    fields.update(kw)
    return ModelLayout(**fields)


_FLAT = dict(overhead_bytes_per_device = 0, overhead_bytes_per_token = 0)


def test_a_fixed_recurrent_state_is_re_priced_not_scaled_with_the_context():
    """A hybrid's floor is part fixed state, and the state does not shrink with the context, so
    the context rule scaled the WHOLE thing: a floor that is mostly state read a fraction of
    its true size at a shorter context, and a plan built on that under-reserves and OOMs."""
    layout = _small_layout()

    def kv_at(ctx: int, slots: int) -> int:
        return (4 * GIB + GIB * ctx // 32768) * slots

    opts = PlanOptions(kv_bytes_at = kv_at, **_FLAT)
    assert _kv_floor_at(layout, opts, 5 * GIB, 32768, 8192, 1) == 4 * GIB + GIB // 4
    assert _kv_floor_at(layout, PlanOptions(), 5 * GIB, 32768, 8192, 1) == 5 * GIB // 4
    hybrid = _small_layout(recurrent_bytes = 2 * GIB)
    assert _kv_floor_at(hybrid, PlanOptions(), 5 * GIB, 32768, 8192, 1) == 2 * GIB + 3 * GIB // 4


def test_a_shrunk_context_reserves_the_state_the_child_still_allocates():
    """End to end: the context PREFER_RESIDENT settles on has to fit the card
    under the caller's own estimator, not under a scaled scalar."""
    layout = _small_layout()

    def kv_at(ctx: int, slots: int) -> int:
        return (4 * GIB + GIB * ctx // 32768) * slots

    card = int(6.5 * GIB)
    opts = PlanOptions(kv_bytes_at = kv_at, context_policy = ContextPolicy.PREFER_RESIDENT, **_FLAT)
    plan = plan_placement(layout, [card], 64 * GIB, 32768, kv_bytes_floor = 5 * GIB, opts = opts)
    assert plan.n_ctx < 32768 and not plan.spills_anything, plan.reason
    weights = all_resident_bytes(layout, 0)  # no cache at ctx 0
    assert weights + kv_at(plan.n_ctx, 1) <= card, plan.reason


def test_the_context_search_prices_a_latent_cache_the_way_the_planner_does():
    """``cache_bytes`` trusts a measured floor on MLA; the search did not."""
    mla = _small_layout(
        has_mla = True,
        blocks = tuple(BlockLayout(i, int(0.00625 * GIB), int(0.15 * GIB)) for i in range(8)),
        lm_head_bytes = int(0.05 * GIB),
        kv_bytes_per_token_f16 = 2 * MIB,
        n_ctx_train = 65536,
    )
    opts = PlanOptions(context_policy = ContextPolicy.FIT_ONLY, **_FLAT)
    floor = dict(kv_bytes_floor = 2 * GIB, floor_ctx = 65536)
    assert max_context_for(mla, [3 * GIB], opts = opts, **floor) >= 32768
    plan = plan_placement(mla, [3 * GIB], 64 * GIB, 65536, kv_bytes_floor = 2 * GIB, opts = opts)
    assert plan.priced and not plan.insufficient, plan.reason
    assert plan.n_ctx >= 32768, plan.reason


def test_a_windowed_cache_shrinks_when_the_caller_can_split_its_halves():
    """iSWA is flat only because one scalar cannot say which half is which."""
    swa = _small_layout(
        has_swa = True,
        blocks = tuple(BlockLayout(i, int(0.00625 * GIB), int(0.0125 * GIB)) for i in range(8)),
    )

    def kv_at(ctx: int, slots: int) -> int:
        return (GIB + 3 * GIB * ctx // 32768) * slots

    budget = [3 * GIB]
    flat = PlanOptions(**_FLAT)
    priced = PlanOptions(kv_bytes_at = kv_at, **_FLAT)
    floor = dict(kv_bytes_floor = 4 * GIB, floor_ctx = 32768)
    assert max_context_for(swa, budget, opts = flat, **floor) == 0
    assert max_context_for(swa, budget, opts = priced, **floor) >= 8192
    assert _kv_floor_at(swa, priced, 4 * GIB, 32768, 8192, 1) == GIB + 3 * GIB // 4


def test_an_exact_cache_is_charged_as_given_at_any_architecture():
    """The max against the per-head product is a guard against a floor that might be short."""
    from core.inference.offload_planner import cache_bytes as _cache_bytes

    layout = _small_layout(kv_bytes_per_token_f16 = 64 * 1024)
    product = layout.kv_bytes(32768)
    assert _cache_bytes(layout, 32768, kv_bytes_floor = product // 4) == product
    assert _cache_bytes(layout, 32768, kv_bytes_floor = product // 4, trust_floor = True) == (
        product // 4
    )


def test_a_card_the_requested_context_leaves_nothing_of_still_reaches_the_ladder():
    """The budget was priced once, at the context the caller asked for."""
    layout = _small_layout(
        blocks = tuple(BlockLayout(i, 0, int(0.175 * GIB)) for i in range(8)),
        lm_head_bytes = int(0.05 * GIB),
        kv_bytes_per_token_f16 = 4096,
        n_ctx_train = 262144,
    )
    shape = dict(
        overhead_bytes_per_device = 1536 * MIB,
        overhead_bytes_per_token = 23961,
        overhead_free_ctx = 32768,
        min_ctx = 8192,
        allow_lm_head_spill = False,
    )
    card = [4 * GIB]
    fit_only = PlanOptions(context_policy = ContextPolicy.FIT_ONLY, **shape)
    plan = plan_placement(layout, card, 64 * GIB, 262144, opts = fit_only)
    assert "no creditable VRAM" not in plan.reason, plan.reason
    assert plan.priced and 32768 <= plan.n_ctx < 262144, plan.reason
    assert not plan.spills_anything, plan.reason
    pinned = plan_placement(layout, card, 64 * GIB, 262144, opts = PlanOptions(**shape))
    assert "no creditable VRAM" in pinned.reason, pinned.reason


def _lopsided_pair_layout(n_blocks: int = 16, per_block: int = 384 * MIB) -> ModelLayout:
    """Uniform blocks and no output-row weight, so the row split is the only
    thing that decides which card is over."""
    blocks = tuple(
        BlockLayout(index = i, spillable_bytes = per_block, resident_bytes = 0) for i in range(n_blocks)
    )
    return ModelLayout(
        arch = "qwen35",
        n_layers = n_blocks,
        n_attention_layers = n_blocks,
        blocks = blocks,
        lm_head_bytes = 0,
        token_embd_bytes = 64 * MIB,
        kv_bytes_per_token_f16 = 1,
        recurrent_bytes = 0,
        n_ctx_train = 65536,
        complete = True,
    )


def _lopsided_pair_args(**kwargs):
    layout = _lopsided_pair_layout()
    vram = [3 * GIB, 5 * GIB]
    options = PlanOptions(
        overhead_bytes_per_device = 0,
        overhead_bytes_per_token = 0,
        pipeline_overhead_bytes = 0,
        n_parallel = 4,
        kv_bytes_floor_by_parallel = {4: 2 * GIB, 3: 3 * GIB // 2, 2: GIB, 1: GIB // 2},
        **kwargs,
    )
    return dict(
        layout = layout,
        vram_bytes_per_device = vram,
        host_ram_bytes = 64 * GIB,
        requested_ctx = 4096,
        opts = options,
        kv_bytes_floor = 2 * GIB,
        split_weights_per_device = vram,
        kv_layer_weights = [1] * layout.n_layers,
    )


def _plan_lopsided(**kwargs) -> Plan:
    args = _lopsided_pair_args(**kwargs)
    return plan_placement(
        args["layout"],
        args["vram_bytes_per_device"],
        args["host_ram_bytes"],
        args["requested_ctx"],
        opts = args["opts"],
        kv_bytes_floor = args["kv_bytes_floor"],
        split_weights_per_device = args["split_weights_per_device"],
        kv_layer_weights = args["kv_layer_weights"],
    )


def test_a_pooled_fit_with_one_card_short_tries_the_rungs_before_it_gives_up():
    """8 GiB of model against 3 GiB + 5 GiB: the pool fits exactly and the row split puts 3.5 GiB
    on the 3 GiB card.
    """
    args = _lopsided_pair_args()
    layout, vram = args["layout"], args["vram_bytes_per_device"]
    assert all_resident_bytes(layout, 4096, kv_bytes_floor = 2 * GIB) == 8 * GIB
    assert (
        _per_device_shortfall(
            layout,
            args["opts"],
            4096,
            {},
            False,
            vram,
            quantised = False,
            kv_bytes_floor = 2 * GIB,
            split_weights_per_device = vram,
            kv_layer_weights = args["kv_layer_weights"],
            extra_on_device0 = 0,
            n_seq = 4,
        )
        is not None
    )

    plan = _plan_lopsided()
    assert plan.priced and plan.changed, plan.reason
    assert 0 < plan.n_parallel < 4, plan.reason
    assert not plan.spills_anything, plan.reason


def test_a_pooled_fit_with_one_card_short_spills_that_card_when_no_rung_is_left():
    """With the slot count pinned there is nothing above the weights to give, so
    the short card's own rows pay for it. A pooled pick would have relieved
    whichever card its rows happened to sit on."""
    plan = _plan_lopsided(min_parallel = 4)
    assert plan.priced and plan.spilled_blocks, plan.reason
    args = _lopsided_pair_args(min_parallel = 4)
    layout = args["layout"]
    assert all(index <= 6 for index in plan.spilled_blocks), plan.spilled_blocks
    assert (
        _per_device_shortfall(
            layout,
            args["opts"],
            4096,
            {i: layout.blocks[i].spillable_bytes for i in plan.spilled_blocks},
            False,
            args["vram_bytes_per_device"],
            quantised = False,
            kv_bytes_floor = 2 * GIB,
            split_weights_per_device = args["vram_bytes_per_device"],
            kv_layer_weights = args["kv_layer_weights"],
            extra_on_device0 = 0,
            n_seq = 4,
        )
        is None
    )


def test_a_pooled_fit_with_a_card_no_rung_can_reach_still_abstains():
    """The remedies are not unlimited: a card whose own rows cannot cover its
    shortfall is still handed back to llama.cpp's own fitter."""
    plan = _plan_lopsided(min_parallel = 4, extra_resident_bytes = 3 * GIB)
    assert not plan.changed and not plan.spills_anything, plan.reason
    assert "fitter" in plan.reason


def test_an_all_zero_tensor_split_abstains_instead_of_being_modelled():
    """A -ts whose active prefix truncates to zeros reaches the planner as an all-zero split."""
    args = _lopsided_pair_args()
    layout = args["layout"]
    plan = plan_placement(
        layout,
        args["vram_bytes_per_device"],
        args["host_ram_bytes"],
        args["requested_ctx"],
        opts = args["opts"],
        kv_bytes_floor = args["kv_bytes_floor"],
        split_weights_per_device = [0, 0],
        kv_layer_weights = args["kv_layer_weights"],
    )
    assert not plan.changed and not plan.priced and not plan.spills_anything, plan.reason
    assert "all zero" in plan.reason, plan.reason
    assert plan_to_args(plan) == []


def _tail_heavy_layout(
    tail: int,
    rest: int,
    n_blocks: int = 8,
) -> ModelLayout:
    sizes = [rest] * (n_blocks - 1) + [tail]
    return ModelLayout(
        **{
            **uneven_layout().__dict__,
            "n_layers": n_blocks,
            "n_attention_layers": n_blocks,
            "blocks": tuple(
                BlockLayout(index = i, spillable_bytes = s, resident_bytes = 10 * MIB)
                for i, s in enumerate(sizes)
            ),
        }
    )


def _plan_for_deficit(layout: ModelLayout, deficit: int, **kwargs) -> Plan:
    floor = resident_floor_bytes(layout, 4096)
    budget = floor + layout.spillable_bytes - deficit + GIB
    return plan_placement(
        layout,
        [budget],
        64 * GIB,
        4096,
        opts = PlanOptions(overhead_bytes_per_device = GIB, **kwargs),
    )


def test_the_trailing_pick_gives_way_when_the_tail_block_is_a_size_accident():
    """BACK_FIRST is a POSITION rule and was blind to size: a 100 MiB deficit on a layout whose
    last block carries an 8 GiB FFN moved 8 GiB of it while a 128 MiB block sat one row down.
    The byte-minimal walk takes over past twice the bytes."""
    layout = _tail_heavy_layout(tail = 8 * GIB, rest = 128 * MIB)
    plan = _plan_for_deficit(layout, 100 * MIB)
    assert plan.spilled_blocks and 7 not in plan.spilled_blocks, plan.spilled_blocks
    moved = sum(layout.blocks[i].spillable_bytes for i in plan.spilled_blocks)
    assert moved < 256 * MIB, moved


def test_a_uniform_layout_still_takes_the_contiguous_tail():
    """The guard must not cost the order the 98 cells it was measured on, every
    one of which has blocks of one size: there the two walks free the same bytes
    and the tie goes to the tail."""
    layout = _tail_heavy_layout(tail = 200 * MIB, rest = 200 * MIB, n_blocks = 4)
    plan = _plan_for_deficit(layout, 40 * MIB)
    assert plan.spilled_blocks == (3,), plan.spilled_blocks


def test_a_cost_ranked_caller_takes_the_smaller_pick_outright():
    """Under require_cost_win the caller is choosing between two placements of ONE rung, where
    rank() is monotone in bytes, so the smaller pick simply wins and only a tie keeps the tail.
    """
    from core.inference.offload_planner import SpillUnit, _select_units

    units = [
        SpillUnit(0, None, 100 * MIB),
        SpillUnit(1, None, 100 * MIB),
        SpillUnit(2, None, 150 * MIB),
    ]
    tail, freed = _select_units(units, 90 * MIB, SpillOrder.BACK_FIRST)
    assert [u.index for u in tail] == [2] and freed == 150 * MIB
    ranked, least = _select_units(units, 90 * MIB, SpillOrder.BACK_FIRST, cost_ranked = True)
    assert [u.index for u in ranked] == [0] and least == 100 * MIB
    even = [SpillUnit(i, None, 100 * MIB) for i in range(3)]
    kept, _ = _select_units(even, 90 * MIB, SpillOrder.BACK_FIRST, cost_ranked = True)
    assert [u.index for u in kept] == [2]


def test_the_per_device_check_sizes_the_cache_the_caller_measured():
    """The pooled deficit trusted ``kv_bytes_at`` while the per-device check went
    back through the product, so a q4_0 cache was spread over the cards at 1.78x
    its bytes and FFN moved for cache that is not there."""
    layout = ModelLayout(
        arch = "dense",
        blocks = tuple(BlockLayout(i, 64 * MIB, 64 * MIB) for i in range(16)),
        complete = True,
        n_layers = 16,
        n_attention_layers = 16,
        kv_bytes_per_token_f16 = 4 * GIB // 4096,
        n_ctx_train = 131072,
    )

    def kv_at(n_ctx: int, slots: int) -> int:
        return layout.kv_bytes(n_ctx, 1) * 9 // 32  # q4_0: 4.5 bits per element

    exact = kv_at(4096, 4)
    vram = [6 * GIB // 5, 11 * GIB // 5]  # each card holds its rows with the exact cache only
    base = PlanOptions(
        overhead_bytes_per_device = 0,
        overhead_bytes_per_token = 0,
        pipeline_overhead_bytes = 0,
        n_parallel = 4,
        min_parallel = 4,
        cache_quantised = True,
    )
    plans = []
    for o in (replace(base, kv_bytes_at = kv_at), base):
        plans.append(
            plan_placement(
                layout,
                vram,
                64 * GIB,
                4096,
                opts = o,
                kv_bytes_floor = exact,
                split_weights_per_device = vram,
                kv_layer_weights = [1] * 16,
            )
        )
    trusted, product = plans
    assert not trusted.spills_anything, trusted.reason
    assert product.spills_anything, product.reason


def test_a_context_priced_reserve_lets_the_ladder_find_the_context_that_fits():
    """The seam's compute buffer is context-linear and was folded into the flat
    per-device term at the requested context, so every rung the ladder tried
    below it still paid the requested context's buffer. A callable re-prices it
    per rung; frozen, the same load abstains at every context."""
    layout = _small_layout(
        blocks = tuple(BlockLayout(i, 0, int(0.175 * GIB)) for i in range(8)),
        lm_head_bytes = int(0.05 * GIB),
        kv_bytes_per_token_f16 = 4096,
        n_ctx_train = 262144,
    )
    per_token = 8 * 1024  # 2 GiB of compute buffer at 262144, 64 MiB at 8192
    shape = dict(
        overhead_bytes_per_token = 0,
        overhead_free_ctx = 32768,
        min_ctx = 8192,
        allow_lm_head_spill = False,
        context_policy = ContextPolicy.FIT_ONLY,
    )
    card = [3 * GIB]
    frozen = PlanOptions(overhead_bytes_per_device = 512 * MIB + per_token * 262144, **shape)
    stuck = plan_placement(layout, card, 64 * GIB, 262144, opts = frozen)
    assert not stuck.priced or stuck.spills_anything, stuck.reason
    priced = PlanOptions(
        overhead_bytes_per_device = 512 * MIB + per_token * 262144,
        overhead_bytes_at = lambda ctx: 512 * MIB + per_token * ctx,
        **shape,
    )
    plan = plan_placement(layout, card, 64 * GIB, 262144, opts = priced)
    assert plan.priced and 8192 <= plan.n_ctx < 262144 and not plan.spills_anything, plan.reason


def test_a_windowed_cache_across_devices_is_not_shrunk_on_a_stale_layer_vector():
    """The per-layer cache vector is measured at the requested context."""
    layout = replace(
        _small_layout(
            blocks = tuple(BlockLayout(i, int(0.05 * GIB), int(0.2 * GIB)) for i in range(8)),
            kv_bytes_per_token_f16 = 64 * 1024,
            n_ctx_train = 65536,
        ),
        has_swa = True,
    )
    shape = dict(
        overhead_bytes_per_device = 0,
        overhead_bytes_per_token = 0,
        pipeline_overhead_bytes = 0,
        min_ctx = 8192,
        context_policy = ContextPolicy.FIT_ONLY,
    )
    floor = 3 * GIB  # at 65536; more than the pair can hold with the weights
    shape["kv_bytes_at"] = lambda ctx, slots: floor * ctx // 65536
    vector = [1, 0, 1, 0, 1, 0, 1, 0]
    pair = [2 * GIB, 2 * GIB]
    split = plan_placement(
        layout,
        pair,
        64 * GIB,
        65536,
        opts = PlanOptions(**shape),
        kv_bytes_floor = floor,
        split_weights_per_device = pair,
        kv_layer_weights = vector,
    )
    assert not split.changed and not split.priced, split.reason
    assert "windowed cache" in split.reason or "sliding-window" in split.reason, split.reason
    one = plan_placement(
        layout,
        [4 * GIB],
        64 * GIB,
        65536,
        opts = PlanOptions(**shape),
        kv_bytes_floor = floor,
        kv_layer_weights = vector,
    )
    assert one.priced and one.n_ctx < 65536, one.reason


# ----------------------------------------------- lazily-read per-layer embeddings


def test_the_per_layer_embeddings_are_their_own_bucket_inside_token_embd():
    """A slice, not a second charge: the seam has to be able to take the PLE back out of
    the host side without the tied-output duplicate going with it."""
    fields = {k.replace("llama.", "gemma4."): v for k, v in _shard_fields().items()}
    fields["general.architecture"] = "gemma4"
    reader = _StubReader(
        fields,
        _shard_tensors(range(64))
        + [
            _StubTensor("token_embd.weight", GIB),
            _StubTensor("per_layer_token_embd.weight", 5 * GIB),
        ],
    )
    layout = _layout_from_reader(reader)
    assert layout.complete
    assert layout.per_layer_embd_bytes == 5 * GIB
    assert layout.token_embd_bytes == 6 * GIB


def _ple_layout(ple_bytes):
    """``q4_layout`` with ``ple_bytes`` of per-layer embeddings inside token_embd."""
    base = q4_layout()
    return ModelLayout(
        **{
            **base.__dict__,
            "arch": "gemma4",
            "token_embd_bytes": base.token_embd_bytes + ple_bytes,
            "per_layer_embd_bytes": ple_bytes,
        }
    )


def test_a_lazily_read_per_layer_embedding_leaves_the_mmap_branch_host_side():
    """gemma4 and qwen4exp create per_layer_token_embd TENSOR_READ_LAZY, so under mmap
    llama.cpp serves it out of the mapping: page cache the OS can evict, not resident bytes
    the plan has to buy. Charging Qwen3.8-Flash-Next's 26.82 GiB in full flipped this plan
    to pageable on machines that had the room.

    RAM is one byte short of holding the charged plan unmapped, so the charged plan stays
    pageable while the lazy one, 5 GiB lighter, takes ``none``; the host delta IS the tensor,
    and VRAM must not move by a byte.
    """
    from core.inference.offload_planner import PlanOptions

    layout = _ple_layout(5 * GIB)
    charged_opts, lazy_opts = PlanOptions(), PlanOptions(ple_read_lazily = True)
    full = plan_placement(layout, [12 * GIB], 200 * GIB, 8192, opts = charged_opts).host_bytes
    ram = full + charged_opts.host_ram_headroom_bytes - 1

    charged = plan_placement(layout, [12 * GIB], ram, 8192, opts = charged_opts)
    lazy = plan_placement(layout, [12 * GIB], ram, 8192, opts = lazy_opts)

    assert charged.load_mode_none is False and lazy.load_mode_none is True
    assert charged.vram_bytes == lazy.vram_bytes
    assert charged.spilled_blocks == lazy.spilled_blocks
    assert charged.host_bytes - lazy.host_bytes == 5 * GIB
    assert charged.ple_charged_to_host is True and lazy.ple_charged_to_host is False


def test_the_none_branch_does_not_pay_for_a_table_llama_cpp_keeps_mapped():
    """llama.cpp maps a lazy context whatever the load mode (llama-model-loader.cpp:
    llama_model_loader::init_mappings maps whenever lazy.any()), so --load-mode none does not
    fault the table in and the none branch is sized without it. Charging it there put
    Qwen3.8-Flash-Next's 26.82 GiB table on the mmap branch on hosts that had the room for
    the spill, at 2x on prefill."""
    from core.inference.offload_planner import PlanOptions

    layout = _ple_layout(5 * GIB)
    charged_opts, lazy_opts = PlanOptions(), PlanOptions(ple_read_lazily = True)
    charged = plan_placement(layout, [12 * GIB], 200 * GIB, 8192, opts = charged_opts)
    lazy = plan_placement(layout, [12 * GIB], 200 * GIB, 8192, opts = lazy_opts)

    assert charged.load_mode_none is True and lazy.load_mode_none is True
    assert lazy.host_bytes == charged.host_bytes - 5 * GIB
    assert lazy.ple_charged_to_host is False

    # RAM that holds the spill but not the table: the plan still takes none.
    ram = lazy.host_bytes + lazy_opts.host_ram_headroom_bytes
    tight = plan_placement(layout, [12 * GIB], ram, 8192, opts = lazy_opts)
    assert tight.load_mode_none is True
    assert plan_placement(layout, [12 * GIB], ram, 8192, opts = charged_opts).load_mode_none is False


def test_a_host_that_holds_only_the_mapped_plan_takes_it_unmapped_rather_than_refusing():
    """The two decisions the over-charge moved, on one host: the spill is admitted, and it
    takes --load-mode none, because the bytes that made it look unaffordable are the ones
    llama.cpp never faults in under either load mode."""
    from core.inference.offload_planner import PlanOptions

    layout = _ple_layout(20 * GIB)
    charged_opts = PlanOptions(require_cost_win = True)
    lazy_opts = PlanOptions(require_cost_win = True, ple_read_lazily = True)
    full = plan_placement(layout, [12 * GIB], 200 * GIB, 8192, opts = charged_opts).host_bytes
    # Room for everything but the per-layer embeddings.
    ram = full - 20 * GIB + charged_opts.host_ram_headroom_bytes

    charged = plan_placement(layout, [12 * GIB], ram, 8192, opts = charged_opts)
    lazy = plan_placement(layout, [12 * GIB], ram, 8192, opts = lazy_opts)

    assert charged.declined_by_gate is True and "host RAM" in charged.reason
    assert lazy.declined_by_gate is False and lazy.spilled_blocks
    assert lazy.load_mode_none is True, "the lazy table is paged under none as well"
    assert lazy.ple_charged_to_host is False


def _deepseek2_fields(**extra):
    """unsloth/DeepSeek-R1-GGUF's header: the LoRA rank, and no MLA head lengths."""
    base = {
        "general.architecture": "deepseek2",
        "deepseek2.block_count": 61,
        "deepseek2.attention.head_count_kv": 128,
        "deepseek2.attention.head_count": 128,
        "deepseek2.embedding_length": 7168,
        "deepseek2.attention.key_length": 192,
        "deepseek2.attention.value_length": 128,
        "deepseek2.attention.kv_lora_rank": 512,
        "deepseek2.context_length": 163840,
    }
    base.update(extra)
    return base


def test_the_lora_rank_alone_is_not_the_latent_cache():
    """llama_hparams::is_mla (llama-hparams.cpp) is true only when BOTH MLA head
    lengths are present and non-zero, because deepseek2.cpp reads them optionally.
    unsloth/DeepSeek-R1-GGUF and unsloth/DeepSeek-V3-0324-GGUF predate the keys and
    carry kv_lora_rank 512 with head_count_kv 128, so llama.cpp allocates the full
    128-head K+V cache -- 39040 MiB at 8192 -- and the product below is exact for
    them. Keying has_mla on the rank made the planner throw that exact number away
    for the estimator's latent-only floor, which is 40% short."""
    layout = _layout_from_reader(_StubReader(_deepseek2_fields(), _shard_tensors(range(61))))
    assert layout.complete
    assert layout.has_mla is False
    assert layout.kv_bytes(8192) == 39040 * MIB

    latent = _layout_from_reader(
        _StubReader(
            _deepseek2_fields(
                **{
                    "deepseek2.attention.key_length_mla": 192,
                    "deepseek2.attention.value_length_mla": 128,
                }
            ),
            _shard_tensors(range(61)),
        )
    )
    assert latent.has_mla is True

    # One of the two alone is not is_mla() either.
    half = _layout_from_reader(
        _StubReader(
            _deepseek2_fields(**{"deepseek2.attention.key_length_mla": 192}),
            _shard_tensors(range(61)),
        )
    )
    assert half.has_mla is False


def _kimi_k3_fields(**extra):
    """unsloth/Kimi-K3-GGUF's header: MLA attention on 24 rows, KDA on the other 69."""
    heads = [1 if (i % 4) == 3 else 0 for i in range(93)]
    heads[92] = 1
    base = {
        "general.architecture": "kimi-k3",
        "kimi-k3.block_count": 93,
        "kimi-k3.attention.head_count_kv": heads,
        "kimi-k3.attention.head_count": 96,
        "kimi-k3.embedding_length": 7168,
        "kimi-k3.attention.key_length": 576,
        "kimi-k3.attention.value_length": 74,
        "kimi-k3.attention.kv_lora_rank": 512,
        "kimi-k3.attention.key_length_mla": 192,
        "kimi-k3.attention.value_length_mla": 128,
        "kimi-k3.ssm.conv_kernel": 4,
        "kimi-k3.kda.head_dim": 128,
        "kimi-k3.context_length": 1048576,
    }
    base.update(extra)
    return base


def test_a_kda_hybrid_is_charged_its_recurrent_state():
    """A Kimi-Delta-Attention row carries kda.head_dim and no ssm.inner_size, so the
    Mamba branch sized it at zero while llama.cpp allocates 443 MiB per slot
    (llama-hparams.cpp:n_embd_r / n_embd_s: 3*(d_conv-1)*n_head*head_dim conv rows
    plus head_dim^2*n_head state, f32). recurrent_bytes is added per slot by
    resident_floor_bytes, max_context_for's fixed term and the multi-device guard,
    separately from the cache, so a zero there is an allocation no context shrink
    can recover."""
    layout = _layout_from_reader(_StubReader(_kimi_k3_fields(), _shard_tensors(range(93))))
    assert layout.complete
    assert layout.n_attention_layers == 24
    # 69 KDA rows x (3*3*96*128 + 128*128*96) x 4 B.
    assert layout.recurrent_bytes == 69 * (110592 + 1572864) * 4
    assert round(layout.recurrent_bytes / MIB) == 443

    # GLM-5.3-Flash: 46 blocks, one of them nextn, 11 of the remaining 45 attention.
    glm_heads = [1 if (i % 4) == 3 else 0 for i in range(46)]
    glm_heads[45] = 1
    glm = _layout_from_reader(
        _StubReader(
            {
                "general.architecture": "glm5next",
                "glm5next.block_count": 46,
                "glm5next.nextn_predict_layers": 1,
                "glm5next.attention.head_count_kv": glm_heads,
                "glm5next.attention.head_count": 64,
                "glm5next.embedding_length": 4096,
                "glm5next.attention.key_length": 512,
                "glm5next.attention.value_length": 512,
                "glm5next.attention.key_length_mla": 256,
                "glm5next.attention.value_length_mla": 256,
                "glm5next.ssm.conv_kernel": 4,
                "glm5next.kda.head_dim": 128,
                "glm5next.context_length": 1048576,
            },
            _shard_tensors(range(46)),
        )
    )
    assert round(glm.recurrent_bytes / MIB) == 146


def test_a_kda_header_with_no_recurrent_map_abstains():
    """Same rule the ssm.* branch follows: kda.head_dim says the model HAS recurrent
    rows, and calling every row attention would charge a cache none of them hold."""
    fields = _kimi_k3_fields(**{"kimi-k3.attention.head_count_kv": 1})
    assert _layout_from_reader(_StubReader(fields, _shard_tensors(range(93)))).complete is False
