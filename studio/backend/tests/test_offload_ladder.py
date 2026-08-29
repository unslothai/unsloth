# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ladder: sub-FFN rungs, and the minimal number of blocks at each one.

The planner used to have one weight rung. A block's whole spillable FFN moved or
none of it did, so covering a 0.96 GiB deficit cost 1.41 GiB of moved weights and
every byte of the overshoot was paid again on every token that touched it. These
pin the finer ladder that replaces it:

    1  ffn_down            (MoE: ffn_down_exps / _chexps)
    2  + ffn_up / gate_up
    3  + ffn_gate
    4  + shared and dense FFN
    5  + lm_head
    6  + attention projections     (off by default)
    7  + the KV cache              (off by default)

and, at every rung, the fewest blocks that close the gap.

The two bottom rungs are off because the module's own numbers put them off the
scale (13.63 t/s for FFN, ~1.03 for anything touching attention or the cache).
They are implemented so the ladder is complete and a benchmark can reach the
bottom of it, not because anything should turn them on.
"""

from __future__ import annotations

import re

import pytest

from core.inference.offload_cost_model import HostProfile
from core.inference.offload_layout import (
    BlockLayout,
    ModelLayout,
    SpillClass,
    spill_pattern_for_class,
)
from core.inference.offload_planner import (
    FfnGranularity,
    PlanOptions,
    plan_placement,
    plan_to_args,
)

GIB = 1024**3


def graded_moe(
    n_blocks: int = 40,
    down: float = 0.16,
    up: float = 0.16,
    gate: float = 0.15,
    attn: float = 0.025,
    shexp: float = 0.0,
) -> ModelLayout:
    """A 35B-A3B-shaped MoE with its FFN broken out per matrix.

    ``spillable_bytes`` is the SUM of the three, never a rounded constant: a
    block whose parts do not add up to its whole is exactly what
    :meth:`BlockLayout.graded` refuses, and building the fixture that way would
    silently test the coarse path instead of the ladder.
    """
    d, u, g = int(down * GIB), int(up * GIB), int(gate * GIB)
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = d + u + g,
            resident_bytes = int(attn * GIB) + int(shexp * GIB),
            ffn_down_bytes = d,
            ffn_up_bytes = u,
            ffn_gate_bytes = g,
            dense_ffn_bytes = int(shexp * GIB),
            attn_bytes = int(attn * GIB),
        )
        for i in range(n_blocks)
    )
    return ModelLayout(
        arch = "qwen3moe",
        n_layers = n_blocks,
        n_attention_layers = n_blocks,
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


def ungraded_moe(n_blocks: int = 40) -> ModelLayout:
    """The same model as the planner saw it before the ladder: one FFN scalar."""
    layout = graded_moe(n_blocks)
    return ModelLayout(
        **{
            **layout.__dict__,
            "blocks": tuple(
                BlockLayout(b.index, b.spillable_bytes, b.resident_bytes) for b in layout.blocks
            ),
        }
    )


def opts(**kwargs) -> PlanOptions:
    """Default options for these tests, at ALL granularity.

    ALL rather than the shipped BOUNDARY default because these tests are about
    the LADDER -- which rung fires, in what order, over how few blocks -- and
    BOUNDARY only ever grades one block, so it exercises the rung machinery
    barely at all. The granularity DECISION, and the measurements behind
    defaulting it to BOUNDARY, are pinned separately below.
    """
    kwargs.setdefault("ffn_granularity", FfnGranularity.ALL)
    return PlanOptions(host = HostProfile(threads = 6), **kwargs)


def shipped(**kwargs) -> PlanOptions:
    """The options a real launch uses."""
    return PlanOptions(host = HostProfile(threads = 6), **kwargs)


def moved_bytes(plan, layout: ModelLayout) -> int:
    """Weight bytes this plan puts on the host, less the always-host embedding."""
    return plan.host_bytes - layout.token_embd_bytes


def test_the_first_rung_is_ffn_down_alone():
    layout = graded_moe()
    plan = plan_placement(layout, [21 * GIB], 94 * GIB, 8192, opts = opts())
    assert plan.spilled_blocks
    assert len(plan.ot_patterns) == 1
    assert "ffn_down_(exps|chexps)" in plan.ot_patterns[0]
    assert "ffn_gate" not in plan.ot_patterns[0]
    assert "ffn_up" not in plan.ot_patterns[0]


def test_the_rungs_arrive_in_order_as_the_card_shrinks():
    """down, then down+up, then down+up+gate. Never a later rung before an
    earlier one is exhausted, which is what makes "minimal" mean anything."""
    layout = graded_moe()
    seen = []
    for vram in (21, 19, 15, 11, 9):
        plan = plan_placement(layout, [vram * GIB], 94 * GIB, 8192, opts = opts())
        classes = []
        for pattern in plan.ot_patterns:
            for cls in (SpillClass.FFN_DOWN, SpillClass.FFN_UP, SpillClass.FFN_GATE):
                body = spill_pattern_for_class(layout, cls, [0]).split(".", 2)[-1]
                if body in pattern:
                    classes.append(cls)
        seen.append(tuple(classes))
    assert seen[0] == (SpillClass.FFN_DOWN,)
    assert seen[-1] == (SpillClass.FFN_DOWN, SpillClass.FFN_UP, SpillClass.FFN_GATE)
    # Monotone: a rung once entered is never given back on a smaller card.
    for earlier, later in zip(seen, seen[1:]):
        assert set(earlier) <= set(later), (earlier, later)


def test_a_later_rung_is_only_entered_once_the_earlier_one_is_exhausted():
    layout = graded_moe()
    plan = plan_placement(layout, [15 * GIB], 94 * GIB, 8192, opts = opts())
    down, up = plan.ot_patterns[0], plan.ot_patterns[1]
    # The down rung took every block -- collapsed to the unbounded form -- while
    # the up rung names only the handful it needed.
    assert r"^blk\.\d+\." in down
    assert re.search(r"\(\d+(\|\d+)*\)", up), up


def deficit_of(layout: ModelLayout, vram_gib: float) -> int:
    """What the plan had to cover, from the planner's own arithmetic."""
    from core.inference.offload_planner import all_resident_bytes

    budget = int(vram_gib * GIB) - PlanOptions().overhead_bytes_per_device
    return all_resident_bytes(layout, 8192) - budget


def test_the_ladders_overshoot_is_bounded_by_one_rung_not_by_one_block():
    """The guarantee the change actually buys, stated as the bound it is.

    NOT "always moves fewer bytes". A greedy best-fit over coarse units can get
    lucky -- on the real Qwen3.6-35B-A3B at 20 GiB the coarse planner's 4 blocks
    came to 1.97 GiB against a 1.83 GiB deficit while the ladder's 11 ffn_down
    units came to 1.99, because four 0.457 GiB blocks happen to land almost
    exactly on that deficit. Asserting strict improvement would pin a
    coincidence.

    What IS guaranteed is the worst case. Both selections stop at the first unit
    that covers the deficit, so each overshoots by less than ONE UNIT -- and the
    ladder's unit is a third of the coarse one. That is the whole claim, and it
    is the claim that matters for #9861's two worst cells, where an 8B with
    11008 MiB free had 22 and 29 of 36 whole FFNs moved for a far smaller
    deficit and landed at 0.19x and 0.21x of ``--fit on``.
    """
    fine, coarse = graded_moe(), ungraded_moe()
    unit_fine = max(b.ffn_down_bytes for b in fine.blocks)
    unit_coarse = max(b.spillable_bytes for b in coarse.blocks)
    assert unit_fine < unit_coarse / 2

    overshoots = []
    for vram in (21, 19, 17, 15):
        a = plan_placement(fine, [vram * GIB], 94 * GIB, 8192, opts = opts())
        b = plan_placement(coarse, [vram * GIB], 94 * GIB, 8192, opts = opts())
        assert a.spills_anything and b.spills_anything
        need = deficit_of(fine, vram)
        over_fine = moved_bytes(a, fine) - need
        over_coarse = moved_bytes(b, coarse) - need
        assert 0 <= over_fine < unit_fine, (vram, over_fine)
        assert 0 <= over_coarse < unit_coarse, (vram, over_coarse)
        overshoots.append((over_fine, over_coarse))

    # And in aggregate it does pay, which is the reason to prefer it even though
    # any single cell can go the other way.
    assert sum(f for f, _ in overshoots) < sum(c for _, c in overshoots)


def test_an_ungraded_layout_still_gets_the_old_whole_ffn_answer():
    """Backwards compatibility, and the safe direction for an unknown model.

    A layout whose per-rung bytes do not add up to its FFN cannot be spilled by
    rung without crediting bytes the pattern would not move. Rather than guess,
    the ladder collapses to the single unit it always had -- so an architecture
    whose tensor names this module has never seen degrades to the previous
    behaviour instead of to a spill that does not cover its own deficit.
    """
    layout = ungraded_moe()
    assert not any(b.graded for b in layout.blocks)
    plan = plan_placement(layout, [19 * GIB], 94 * GIB, 8192, opts = opts())
    assert len(plan.ot_patterns) == 1
    assert "ffn_(up|gate|down|gate_up)_(exps|chexps)" in plan.ot_patterns[0]


def test_the_emitted_patterns_move_exactly_the_bytes_the_plan_charged_itself():
    """The invariant that a 20 GiB miscount already got past once.

    A plan is only as good as the ``-ot`` it emits: if the patterns move fewer
    bytes than the deficit assumed, VRAM is filled against a gap that was never
    closed and the load OOMs. So this rebuilds the GGUF's tensor table, runs
    llama.cpp's own matching rule over it, and checks the total against what the
    plan says it put on the host.

    ``re.search`` rather than ``re.match``, because that is what llama.cpp uses
    (``llama-model-loader.cpp``) and it is the reason every pattern this module
    emits is anchored.
    """
    layout = graded_moe(n_blocks = 12, shexp = 0.02)
    table: list[tuple[str, int]] = []
    for b in layout.blocks:
        table += [
            (f"blk.{b.index}.ffn_down_exps.weight", b.ffn_down_bytes),
            (f"blk.{b.index}.ffn_up_exps.weight", b.ffn_up_bytes),
            (f"blk.{b.index}.ffn_gate_exps.weight", b.ffn_gate_bytes),
            (f"blk.{b.index}.ffn_down_shexp.weight", b.dense_ffn_bytes),
            (f"blk.{b.index}.attn_q.weight", b.attn_bytes),
            (f"blk.{b.index}.attn_norm.weight", 4096),
            (f"blk.{b.index}.ffn_gate_inp.weight", 4096),
        ]
    table.append(("output.weight", layout.lm_head_bytes))

    for vram in (7, 6, 5, 4, 3):
        plan = plan_placement(layout, [vram * GIB], 94 * GIB, 8192, opts = opts())
        if not plan.spills_anything:
            continue
        matched = sum(
            n for name, n in table if any(re.search(p, name) for p in plan.ot_patterns)
        )
        assert matched == moved_bytes(plan, layout), (vram, plan.reason)


def test_the_down_rung_never_drags_the_shared_experts_with_it():
    """``ffn_down`` unanchored also matches ``ffn_down_exps`` and
    ``ffn_down_shexp``. On a MoE model that is most of the file, moved for a
    deficit that asked for a third of one rung."""
    layout = graded_moe(shexp = 0.02)
    pattern = spill_pattern_for_class(layout, SpillClass.FFN_DOWN, [3])
    assert re.search(pattern, "blk.3.ffn_down_exps.weight")
    assert not re.search(pattern, "blk.3.ffn_down_shexp.weight")
    assert not re.search(pattern, "blk.3.ffn_down.weight")
    assert not re.search(pattern, "blk.13.ffn_down_exps.weight")


def test_lm_head_is_the_rung_after_every_ffn_rung_and_not_before():
    """The user's question, and the answer the measurements give: lm_head stays
    in VRAM as long as there is any expert byte left to move instead.

    16% here against 43% if taken first, because by the time it is reached FFN
    offload has already made generation host-bandwidth-bound.
    """
    layout = graded_moe(n_blocks = 8)
    spillable = layout.spillable_bytes
    # Roomy enough that some FFN still fits: lm_head must not be touched.
    plan = plan_placement(layout, [5 * GIB], 94 * GIB, 8192, opts = opts())
    assert plan.spills_anything and not plan.spilled_lm_head
    assert moved_bytes(plan, layout) < spillable

    # Tight enough that every rung above lm_head is exhausted: 3.76 GiB of FFN
    # against a deficit larger than that, so there is nothing else left to give.
    tight = plan_placement(layout, [2 * GIB], 94 * GIB, 8192, opts = opts())
    assert tight.spills_anything
    assert tight.spilled_lm_head
    assert moved_bytes(tight, layout) == spillable + layout.lm_head_bytes


def test_embed_tokens_is_never_a_rung_because_it_is_never_on_the_card():
    """It is not a placement decision at all. ``llama-model.cpp`` pins
    ``dev_input`` to the CPU unconditionally, so ``token_embd`` is host-resident
    on every launch, spilled or not. It appears here only as host RAM the plan
    must be able to pay for."""
    layout = graded_moe()
    plan = plan_placement(layout, [64 * GIB], 94 * GIB, 8192, opts = opts())
    assert not plan.spills_anything, "this card holds the whole model"
    assert plan.host_bytes == layout.token_embd_bytes
    assert not any("token_embd" in p for p in plan.ot_patterns)


def test_attention_and_the_cache_are_off_the_ladder_by_default():
    """A load that needs them is a load ``--fit on`` should place.

    Measured: 13.63 t/s with the FFN on the host, ~1.03 once attention or the
    cache follows it. Abstaining costs 0.93x to 1.16x. So the planner stops.
    """
    layout = graded_moe(n_blocks = 8, attn = 0.4)
    plan = plan_placement(layout, [2 * GIB], 94 * GIB, 8192, opts = opts())
    assert not plan.spills_anything
    assert not plan.kv_spilled_to_host


def test_the_attention_rung_exists_and_sits_below_lm_head():
    """Implemented so the ladder is complete and a benchmark can reach it."""
    layout = graded_moe(n_blocks = 8, attn = 0.4)
    plan = plan_placement(
        layout, [2 * GIB], 94 * GIB, 8192, opts = opts(allow_attention_spill = True)
    )
    assert plan.spills_anything
    assert plan.spilled_lm_head, "lm_head goes before any attention weight does"
    assert any("attn_" in p for p in plan.ot_patterns)


def test_the_cache_is_the_last_rung_of_all_and_emits_nkvo():
    layout = graded_moe(n_blocks = 8, attn = 0.4)
    plan = plan_placement(
        layout,
        [1 * GIB],
        94 * GIB,
        8192,
        opts = opts(allow_attention_spill = True, allow_kv_host_fallback = True),
    )
    if plan.kv_spilled_to_host:
        assert "--no-kv-offload" in plan_to_args(plan)
        assert any("attn_" in p for p in plan.ot_patterns), "attention goes first"


def test_load_mode_is_none_when_the_host_side_fits_and_mmap_when_it_does_not():
    """The rule the user stated, and it is orthogonal to which rung fired.

    ``--load-mode none`` is worth 2.09x to 2.35x on prefill against mmap on
    host-resident weights, but only while those bytes really are in RAM. Past
    that, mmap is the only thing making an over-commit pageable rather than
    OOM-killed.
    """
    layout = graded_moe()
    roomy = plan_placement(layout, [15 * GIB], 94 * GIB, 8192, opts = opts())
    assert roomy.spills_anything and roomy.load_mode_none
    assert "--load-mode" in plan_to_args(roomy)

    cramped = plan_placement(layout, [15 * GIB], 8 * GIB, 8192, opts = opts())
    assert not cramped.load_mode_none
    assert "--load-mode" not in plan_to_args(cramped)


def test_a_partial_rung_is_charged_to_the_right_card_not_to_the_pool():
    """Two devices and a rung that moved a third of some blocks.

    ``-ot`` does not move a layer, so a partial spill relieves only the card the
    chosen indices already sat on. Crediting the whole block for a partial move
    is the optimistic direction, and a per-device shortfall is a hard throw
    rather than a slow load, so the planner abstains instead of guessing.
    """
    layout = graded_moe()
    plan = plan_placement(layout, [11 * GIB, 11 * GIB], 94 * GIB, 8192, opts = opts())
    if plan.spills_anything:
        # Only a FULL spill is checkable; anything less must have abstained.
        assert plan.host_bytes - layout.token_embd_bytes >= layout.spillable_bytes
    else:
        assert "partial spill" in plan.reason or "device" in plan.reason


def test_a_dense_model_gets_the_same_gradation():
    """The three matrices are ``ffn_down`` / ``ffn_up`` / ``ffn_gate`` with no
    ``_exps`` suffix, and the ladder does not care which it is looking at."""
    d = u = g = int(0.07 * GIB)
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = d + u + g,
            resident_bytes = int(0.045 * GIB),
            ffn_down_bytes = d,
            ffn_up_bytes = u,
            ffn_gate_bytes = g,
            attn_bytes = int(0.045 * GIB),
        )
        for i in range(64)
    )
    layout = ModelLayout(
        arch = "qwen3",
        n_layers = 64,
        n_attention_layers = 64,
        blocks = blocks,
        lm_head_bytes = int(1.0 * GIB),
        token_embd_bytes = int(1.0 * GIB),
        other_resident_bytes = int(0.01 * GIB),
        kv_bytes_per_token_f16 = 2.0 * GIB / 32768,
        n_ctx_train = 32768,
        complete = True,
    )
    plan = plan_placement(layout, [16 * GIB], 94 * GIB, 8192, opts = opts())
    assert plan.spills_anything
    assert any(r"ffn_down\.weight" in p for p in plan.ot_patterns)
    assert not any("_exps" in p for p in plan.ot_patterns)


@pytest.mark.parametrize(
    "order",
    [
        (SpillClass.FFN_DOWN, SpillClass.FFN_UP, SpillClass.FFN_GATE),
        (SpillClass.FFN_GATE, SpillClass.FFN_UP, SpillClass.FFN_DOWN),
        (SpillClass.FFN_UP, SpillClass.FFN_DOWN, SpillClass.FFN_GATE),
    ],
)
def test_the_rung_order_is_configurable_and_costs_the_same_either_way(order):
    """Which matrix goes first is llama.cpp's choice and UNMEASURED by us.

    The cost model prices all three identically -- same ``Access.SCATTERED``,
    same routed fraction -- and they are within a few percent of the same size,
    so nothing here can prefer one order to another. That is worth pinning
    rather than hiding: it says the ladder's win is GRANULARITY, and leaves the
    ordering as an open question a benchmark can answer.
    """
    layout = graded_moe()
    plan = plan_placement(
        layout, [19 * GIB], 94 * GIB, 8192, opts = opts(ffn_rung_order = order)
    )
    assert plan.spills_anything
    first = spill_pattern_for_class(layout, order[0], [0]).split(".", 2)[-1]
    assert first in plan.ot_patterns[0]
    assert abs(moved_bytes(plan, layout) - 3.04 * GIB) < 0.35 * GIB


def overshoot(layout: ModelLayout, vram: float, options: PlanOptions) -> float:
    plan = plan_placement(layout, [int(vram * GIB)], 94 * GIB, 8192, opts = options)
    if not plan.spills_anything:
        return 0.0
    return moved_bytes(plan, layout) - deficit_of(layout, vram)


def test_the_shipped_default_grades_only_the_boundary_block():
    """BOUNDARY: coarse blocks, then the last one trimmed to the rungs needed.

    The reason is measured, not aesthetic. A partly-spilled block puts part of
    its FFN on each side of the backend boundary, so the decode graph crosses
    once more per block than a wholly-spilled one does. On Qwen3.6-35B-A3B Q4 at
    4 slots, ggml's own ``graph splits (with bs=1)`` went 10 / 20 / 28 / 36 for
    whole blocks against 24 / 46 / 70 / 82 for the every-block ladder, and
    generation came out 0.88x / 0.95x / 0.92x / 0.93x -- 5 to 12% paid to save
    0.02 to 0.36 GiB of overshoot.

    Grading one block costs exactly one extra split and recovers most of the
    overshoot, which is also what llama.cpp settled on (fit.cpp:490 applies its
    graded fraction to il0 and LAYER_FRACTION_MOE to every layer past it).
    """
    layout = graded_moe()
    plan = plan_placement(layout, [19 * GIB], 94 * GIB, 8192, opts = shipped())
    assert plan.spills_anything
    # One coarse pattern covering whole FFNs, plus graded patterns naming a
    # single block between them.
    coarse = [p for p in plan.ot_patterns if "ffn_(up|gate|down|gate_up)_" in p]
    graded = [p for p in plan.ot_patterns if p not in coarse]
    assert len(coarse) == 1, plan.ot_patterns
    assert graded, "the boundary block should have been trimmed"
    blocks_named = {b for p in graded for b in re.findall(r"\d+", p.split(".ffn")[0])}
    assert len(blocks_named) == 1, blocks_named


def test_boundary_grading_beats_both_extremes_on_overshoot():
    """Less overshoot than WHOLE, and no worse than ALL, at one extra split.

    Better than ALL rather than merely close to it, because BOUNDARY can combine
    whole blocks with a partial one: its step size is the sub-FFN matrix on top
    of a coarse base, while ALL is quantised to the matrix everywhere.
    """
    layout = graded_moe()
    for vram in (21, 19, 17):
        whole = overshoot(layout, vram, shipped(ffn_granularity = FfnGranularity.WHOLE))
        bound = overshoot(layout, vram, shipped(ffn_granularity = FfnGranularity.BOUNDARY))
        every = overshoot(layout, vram, shipped(ffn_granularity = FfnGranularity.ALL))
        assert bound < whole, (vram, bound, whole)
        assert bound <= every, (vram, bound, every)


def test_boundary_falls_back_to_the_whole_block_when_trimming_cannot_help():
    """A deficit that needs the entire last block leaves it whole.

    Emitting a graded pattern that adds up to the same bytes would pay the extra
    split for nothing, so the trim is skipped and one pattern says it all.
    """
    layout = graded_moe()
    # Every spillable byte is needed here, so there is no last block to trim.
    plan = plan_placement(layout, [4 * GIB], 94 * GIB, 8192, opts = shipped())
    if plan.spills_anything:
        assert len(plan.ot_patterns) <= 2, plan.ot_patterns
        assert any("ffn_(up|gate|down|gate_up)_" in p for p in plan.ot_patterns)


# --------------------------------------------------------------------------
# Two accounting bugs that made the planner spill a model which already fit.
# Both were found by benchmarking gemma-4-E2B-it UD-Q4_K_XL on a 4.15 GiB
# budget, where --fit on measured 447.8 t/s (identical to its 6 GiB control,
# i.e. it had nothing to do) and the planner measured 187.7 -- 0.42x, which is
# #9861's headline failure reproduced on hardware we control.
# --------------------------------------------------------------------------


def tied_embedding_layout(vocab: int, per_layer: int) -> ModelLayout:
    """A gemma-shaped layout: tied embeddings, plus per-layer input embeddings.

    Tied means no ``output.weight``, so llama.cpp re-creates the output tensor
    from ``token_embd`` as TENSOR_DUPLICATED and a second vocabulary matrix is
    really allocated. ``per_layer_token_embd`` is NOT part of that duplicate.
    """
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = int(0.025 * GIB),
            resident_bytes = int(0.008 * GIB),
            ffn_down_bytes = int(0.009 * GIB),
            ffn_up_bytes = int(0.008 * GIB),
            ffn_gate_bytes = int(0.008 * GIB),
            attn_bytes = int(0.008 * GIB),
        )
        for i in range(35)
    )
    return ModelLayout(
        arch = "gemma4",
        n_layers = 35,
        n_attention_layers = 35,
        blocks = blocks,
        lm_head_bytes = 0,
        token_embd_bytes = vocab + per_layer,
        # What the loader charges to VRAM for the tied duplicate.
        other_resident_bytes = vocab,
        kv_bytes_per_token_f16 = 0.615 * GIB / 9216,
        n_ctx_train = 32768,
        has_swa = True,
        complete = True,
    )


def test_the_tied_embedding_duplicate_is_the_vocabulary_not_the_per_layer_embeddings():
    """1540 MiB charged to VRAM for a 264 MiB duplicate.

    ``per_layer_token_embd`` lands in the same host-resident bucket as
    ``token_embd`` -- correctly, both are host-pinned -- and the tied-embedding
    branch then added that WHOLE bucket to ``other_resident_bytes`` to account
    for the duplicate. On gemma-4-E2B that charged VRAM 1804 MiB instead of 264,
    and ``all_resident_bytes`` came to 3.57 GiB against the 1.45 GiB llama.cpp
    actually placed.

    Checked here as arithmetic rather than through the GGUF reader, so the
    property survives a reader that buckets the tensors differently.
    """
    vocab, per_layer = 264 * 1024**2, 1540 * 1024**2
    layout = tied_embedding_layout(vocab, per_layer)
    # The duplicate is charged, once, at the vocabulary's size.
    assert layout.other_resident_bytes == vocab
    # And the per-layer embeddings are host bytes, never VRAM ones.
    assert layout.token_embd_bytes == vocab + per_layer
    from core.inference.offload_planner import all_resident_bytes

    resident = all_resident_bytes(layout, 9216, kv_bytes_floor = 48 * 1024**2)
    assert resident < vocab + per_layer, "the PLE must not be charged to VRAM"


def test_a_measured_cache_beats_the_product_under_sliding_window_attention():
    """The product has no SWA term, so its error is one-sided and large.

    On gemma-4-E2B at n_ctx 9216 it says 0.615 GiB where llama.cpp allocated
    48 MiB: the window caps most layers at 512 tokens, those layers use narrower
    heads (key_length_swa 256 against 512), and 20 of them share one cache. Under
    ``max(product, measurement)`` a 13x over-estimate overrode a real number, and
    that difference is deficit the planner spills real blocks to cover.

    The max is KEPT without SWA, where the product's failure mode is the other
    one (MLA under-counts) and the measurement is what might be short.
    """
    from core.inference.offload_planner import cache_bytes

    measured = 48 * 1024**2
    swa = tied_embedding_layout(264 * 1024**2, 1540 * 1024**2)
    assert cache_bytes(swa, 9216, kv_bytes_floor = measured) == measured

    dense = ModelLayout(**{**swa.__dict__, "has_swa": False})
    assert cache_bytes(dense, 9216, kv_bytes_floor = measured) > measured


def test_sliding_window_without_a_measurement_abstains_rather_than_guessing():
    """No measurement, no spill. The estimate is only wrong upwards here, and
    upwards is the direction that invents a deficit and costs 0.42x."""
    layout = tied_embedding_layout(264 * 1024**2, 1540 * 1024**2)
    plan = plan_placement(layout, [4 * GIB], 200 * GIB, 9216, opts = shipped())
    assert not plan.spills_anything
    assert "sliding-window" in plan.reason

    # With the cache priced, it plans normally again -- and on this budget the
    # right answer is that nothing needs to move at all.
    measured = plan_placement(
        layout, [4 * GIB], 200 * GIB, 9216,
        kv_bytes_floor = 48 * 1024**2, opts = shipped(),
    )
    assert not measured.spills_anything
    assert "fits in VRAM" in measured.reason


# ---------------------------------------------------------------------------
# Per-layer attention.head_count_kv
# ---------------------------------------------------------------------------

def _reader_with_kv_heads(value, n_layers=6):
    """A stand-in GGUFReader exposing just the fields the layout reads."""
    fields = {
        "general.architecture": "gemma4",
        "gemma4.block_count": n_layers,
        "gemma4.attention.head_count_kv": value,
        "gemma4.attention.head_count": 8,
        "gemma4.embedding_length": 256,
        "gemma4.attention.key_length": 32,
        "gemma4.attention.value_length": 32,
    }

    class _F:
        def __init__(self, v): self.v = v

    class _R:
        tensors = ()
        def __init__(self): self.fields = {k: _F(v) for k, v in fields.items()}

    return _R(), fields


def test_a_per_layer_kv_head_list_does_not_abstain():
    """gemma-4-26B ships head_count_kv as a LIST, and int(list) raises.

    layout_from_gguf swallows that to a debug log, so the planner abstained on
    every quant of gemma-4-26B-A4B and gemma-4-31B -- six of the thirteen models
    in the sweep -- with nothing visibly failing, because an abstain falls
    through to --fit on. Found only when a Kaggle cell reported "layout or
    device inventory is incomplete".
    """
    from core.inference import offload_layout as OL

    scalar = OL._kv_heads_total(4, 6)
    listed = OL._kv_heads_total([4, 4, 4, 4, 4, 4], 6)
    assert scalar == listed == 24


def test_a_mixed_kv_head_list_is_summed_not_multiplied():
    """The real lists mix widths, so one head count times a layer count is wrong.

    gemma-4-26B is [8, 8, 8, 8, 8, 2, ...] and gemma-4-31B [16, 16, 16, 16, 16,
    4, ...]. Multiplying the first entry by the layer count would over-count the
    cache; multiplying the last would under-count it.
    """
    from core.inference import offload_layout as OL

    heads = [8, 8, 8, 8, 8, 2]
    assert OL._kv_heads_total(heads, 6) == 42
    assert OL._kv_heads_total(heads, 6) != 8 * 6
    assert OL._kv_heads_total(heads, 6) != 2 * 6


def test_a_short_kv_head_list_pads_with_its_last_value():
    """Trailing layers beyond the list reuse its last width rather than crash."""
    from core.inference import offload_layout as OL

    assert OL._kv_heads_total([8, 2], 4) == 8 + 2 + 2 + 2


def test_no_ladder_rung_can_reach_a_unified_memory_host():
    """Every granularity must abstain identically on a unified pool.

    On Strix Halo (gfx1151) and Apple silicon the device and the host are the
    same chips, so moving a tensor "to RAM" renames bytes and frees nothing. The
    abstain that says so is the SECOND check in plan_placement, ahead of the
    budget, the context and all rung selection, which is what makes the whole
    ladder unreachable there.

    Verified on real hardware before the ladder existed: the AMD CI GPU
    measurement job declined on the gfx1151 unified pool while spilling 51
    blocks for the identical layout and budget with the flag off. This test is
    what keeps that true as rungs are added, since a rung wired in above the
    abstain would start spilling on an APU and nobody would see it in a CUDA
    matrix.
    """
    from dataclasses import replace

    from core.inference.offload_planner import (
        FfnGranularity,
        PlanOptions,
        plan_placement,
    )

    layout = graded_moe()
    gib = 1024 ** 3
    base = PlanOptions()

    reasons = set()
    for granularity in FfnGranularity:
        unified = replace(
            base,
            ffn_granularity = granularity,
            host = replace(base.host, unified_memory = True),
        )
        discrete = replace(
            base,
            ffn_granularity = granularity,
            host = replace(base.host, unified_memory = False),
        )
        on_apu = plan_placement(layout, [8 * gib], 64 * gib, 8192, opts = unified)
        on_gpu = plan_placement(layout, [8 * gib], 64 * gib, 8192, opts = discrete)

        assert not on_apu.ot_patterns, granularity
        assert "unified memory" in on_apu.reason, granularity
        reasons.add(on_apu.reason)
        # The same budget on a discrete card must actually spill, otherwise this
        # test would pass on a layout that simply fits and prove nothing.
        assert on_gpu.ot_patterns, granularity

    assert len(reasons) == 1, f"granularity leaked into the abstain: {reasons}"


def test_the_selection_matches_an_independent_minimal_walk():
    """MINIMALITY, checked against a reimplementation rather than a comment.

    The request this ladder answers asks for "the MINIMAL number of layers to
    offload", and the selection code asserts it gets one by construction: take
    the fewest units of the cheapest rung, step down only when a rung is
    exhausted. That is sound reasoning, but it is reasoning, and a greedy walk
    across several rungs is the shape of code where an off-by-one leaves one
    unit too many on the host without breaking any test that only checks the
    plan FITS.

    So this recomputes the answer independently -- walk the rungs in order,
    take units largest-first, stop at the first unit that closes the gap -- and
    demands the planner move exactly that many bytes. An overshoot bound cannot
    catch an off-by-one that stays inside one unit; an exact comparison can.

    NOTE ON THE BOUND, because the first version of this test was wrong. It
    compared the overshoot against the smallest unit ANYWHERE in the layout and
    failed at 21 GiB, where the plan overshot by 166.4 MB while a 161 MB
    ffn_gate unit existed. The plan had stopped inside the ffn_down rung (unit
    171.8 MB), so it was minimal; the test was asking it to substitute a unit
    from a rung the ladder had not entered, which is precisely what rung ORDER
    forbids. Minimality here means minimal SUBJECT TO the rung discipline, and
    a check that ignores the discipline reports a violation that is not one.
    """
    layout = graded_moe()
    checked = 0
    for vram in (23, 21, 19, 17, 15, 13, 11):
        plan = plan_placement(layout, [vram * GIB], 94 * GIB, 8192, opts = opts())
        if not plan.spills_anything:
            continue
        need = deficit_of(layout, vram)

        # Independent walk: ffn_down over every block, then ffn_up, then
        # ffn_gate, each rung largest-first, stopping the moment the gap closes.
        expected = 0
        for attr in ("ffn_down_bytes", "ffn_up_bytes", "ffn_gate_bytes"):
            if expected >= need:
                break
            sizes = sorted(
                (getattr(b, attr) for b in layout.blocks if getattr(b, attr)),
                reverse = True,
            )
            for size in sizes:
                if expected >= need:
                    break
                expected += size

        assert expected >= need, (vram, expected, need)
        assert moved_bytes(plan, layout) == expected, (
            f"at {vram} GiB the planner moved {moved_bytes(plan, layout)} bytes "
            f"but the minimal rung-ordered walk needs {expected} for a {need} "
            f"byte deficit"
        )
        checked += 1

    assert checked >= 5, f"only {checked} budgets spilled; the sweep is vacuous"
