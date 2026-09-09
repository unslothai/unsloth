# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The planner may now decline a spill it is perfectly capable of making.

Before this, every abstain in the planner was a FEASIBILITY abstain: layout
incomplete, unified memory, no creditable VRAM, no usable context, a per-device
shortfall. If a spill covered the deficit it was taken, however badly it paid.
``_spill_penalty_ms`` was computed and stored on the Plan, and read by nothing
but tests.

These pin the gate that replaces that: what it declines, what it must not
decline, and that it stays out of the way of every caller who is asking the
planner what it CAN place rather than whether it should.
"""

from __future__ import annotations

from core.inference.offload_cost_model import HostProfile
from core.inference.offload_layout import BlockLayout, ModelLayout
from core.inference.offload_planner import (
    ContextPolicy,
    PlanOptions,
    _fit_fallback_placement,
    plan_placement,
)
from dataclasses import replace

GIB = 1024**3
MIB = 1024**2


def dense_layout(
    n_blocks: int = 64,
    ffn_gib: float = 0.20,
    attn_gib: float = 0.045,
    lm_head_gib: float = 1.0,
    kv_gib_at_32k: float = 2.0,
) -> ModelLayout:
    """A dense 27B-ish model, the shape #9861 spilled every block of."""
    blocks = tuple(BlockLayout(i, int(ffn_gib * GIB), int(attn_gib * GIB)) for i in range(n_blocks))
    return ModelLayout(
        arch = "qwen3",
        n_layers = n_blocks,
        n_attention_layers = n_blocks,
        blocks = blocks,
        lm_head_bytes = int(lm_head_gib * GIB),
        token_embd_bytes = int(lm_head_gib * GIB),
        other_resident_bytes = int(0.01 * GIB),
        kv_bytes_per_token_f16 = kv_gib_at_32k * GIB / 32768,
        n_ctx_train = 32768,
        complete = True,
    )


def gated(**kwargs) -> PlanOptions:
    return PlanOptions(require_cost_win = True, **kwargs)


def test_the_gate_is_off_unless_the_caller_has_an_alternative():
    """A bare PlanOptions still answers the old question, unchanged.

    Only the launch seam has somewhere to fall through to, so only the launch
    seam asks. Defaulting this on would have silently changed the answer for
    every other caller of a module that advertises itself as pure arithmetic.
    """
    assert PlanOptions().require_cost_win is False
    layout = dense_layout()
    plan = plan_placement(layout, [14848 * 1024 * 1024], 94 * GIB, 32768)
    assert plan.spilled_blocks, "ungated planning should still spill"


def test_a_spill_that_barely_beats_the_fitter_is_declined():
    """A near-tie is not worth taking, because the two errors are not symmetric.

    #9861 measured all 33 cells where the planner declined between 0.93x and
    1.16x, and cells it planned wrongly at up to 8x slower. Coin-flips therefore
    go to ``--fit on``.
    """
    layout = dense_layout()
    plan = plan_placement(
        layout, [14848 * 1024 * 1024], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6))
    )
    assert not plan.spilled_blocks
    assert not plan.changed
    assert "not worth it" in plan.reason
    assert "--fit on" in plan.reason


def test_declining_says_what_it_compared():
    """A silent abstain is unactionable; both sides of the trade go in the reason."""
    layout = dense_layout()
    plan = plan_placement(
        layout, [14848 * 1024 * 1024], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6))
    )
    assert "ms" in plan.reason
    assert "2048 prompt" in plan.reason
    assert "256 generated" in plan.reason


def test_a_load_the_fitter_cannot_place_either_is_never_declined():
    """The gate only ever chooses between two viable placements.

    When even moving every layer to the host does not fit, there is no fallback
    to lose to, and declining would trade a slow launch for a failed one.
    """
    layout = dense_layout()
    tiny = 3 * GIB
    plan = plan_placement(layout, [tiny], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6)))
    assert "not worth it" not in plan.reason


def test_the_workload_shape_moves_the_trade():
    """rank() needs a request shape and the answer genuinely depends on it.

    A spill is at its worst during prefill, where the moved weights stream at
    FULL size once per ubatch -- a 512-token ubatch selects essentially every
    expert, so MoE sparsity buys nothing there -- and at its best during a long
    decode. So the same placement on the same card is worth more for one
    workload than another, and a planner that scores only generation, as this one
    did, cannot see the difference at all.

    The claim here is the DIRECTION, not a crossing: on this fixture the spill
    stays inside the margin at every shape.

    Every shape below carries the same TOTAL, because the fallback's live-cache
    term is sized from ``prompt + generated`` -- a shape that reads more of the
    cache charges the fitter more for it, which is a second effect and not the
    one under test. Holding the total fixed isolates the split.
    """
    layout = dense_layout()
    card = [14848 * 1024 * 1024]
    ratios = []
    for n_prompt, n_generated in ((8192, 16), (6000, 2208), (2208, 6000), (16, 8192)):
        plan = plan_placement(
            layout,
            card,
            94 * GIB,
            32768,
            opts = gated(
                host = HostProfile(threads = 6),
                workload_prompt_tokens = n_prompt,
                workload_generated_tokens = n_generated,
            ),
        )
        assert plan.predicted_fit_request_ms > 0.0
        ratios.append(plan.predicted_request_ms / plan.predicted_fit_request_ms)
    assert ratios == sorted(ratios, reverse = True), ratios
    assert ratios[0] > ratios[-1], "the longer the decode, the better a spill looks"


def test_the_margin_is_what_decides_a_near_tie():
    """Set the margin to zero and the same cell is planned again.

    Which confirms the decline above is the MARGIN talking and not an accident
    of the arithmetic: the spill really is cheaper here, just not by enough.

    17.2 GiB, RE-ANCHORED from 18.8. The band did not disappear, it moved: the
    fitter's moved cache is now priced at Access.KV_CACHE's calibrated rate
    instead of the contiguous weight rate, which is worth 5 to 12 points of the
    ratio on a dense cell, so 18.8 GiB is now a clear planner win (1.19x) and a
    strict margin plans it too. That is the gate working, not the near-tie
    property lapsing, and re-anchoring is the honest response -- deleting the
    test would drop the only check that the margin is load-bearing rather than
    decorative.
    ---
    This budget is chosen for the SAME reason 18.8 was: it is the band. At 17.2
    the spill is 0.95x the fitter -- genuinely cheaper, by less than the 10%
    margin -- so strict declines and lenient plans, and neither passes for the
    wrong reason. Do not re-anchor this to a budget where the spill is outright
    more expensive: a zero margin would decline there too and the test would go
    green while asserting nothing.
    ---
    The per-device reserve's context-linear term once started at 16384 and charged 343 MiB at
    n_ctx 32768, and every budget in this file was shifted by exactly that to keep its cell at
    the same distance from the band's edge. The term now starts at 32768, where the measured
    requirement stops being flat, so the budgets are back at their original values.
    """
    layout = dense_layout()
    card = [17584 * 1024 * 1024]
    strict = plan_placement(layout, card, 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6)))
    lenient = plan_placement(
        layout,
        card,
        94 * GIB,
        32768,
        opts = gated(host = HostProfile(threads = 6), min_penalty_reduction = 0.0),
    )
    assert not strict.spilled_blocks
    assert lenient.spilled_blocks


def moe_layout(n_blocks: int = 40) -> ModelLayout:
    """A 35B-A3B-shaped MoE: most of the weight is in routed experts."""
    blocks = tuple(BlockLayout(i, int(0.47 * GIB), int(0.025 * GIB)) for i in range(n_blocks))
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


def test_the_moe_fallback_places_about_what_the_planner_places():
    """Measured from the fitter's own trace, not from a buffer report.

    n_part came back as 14, 23 and 31 partial layers on a 16 GiB A100, a 12 GiB
    L4 and an 8 GiB T4, against 13, 22 and 31 spilled blocks for the planner on
    the same cards. The two arrive at nearly the same placement, so on MoE the
    gate should usually find them equivalent and decline, which is cheap.

    A previous version of this test asserted the opposite -- that the fitter
    moves EVERY expert regardless of card size -- on a misreading of CPU_Mapped,
    which only counts host-resident bytes when mmap is off. The fitter arm keeps
    mmap, so that figure was the size of the mapped file, and it was identical on
    all three cards because a file size does not vary with VRAM.
    """
    layout = moe_layout()
    small = _fit_fallback_placement(
        layout,
        gated(),
        11 * GIB,
        8192,
        quantised = False,
        kv_bytes_floor = 0,
        kv_on_host = False,
    )
    large = _fit_fallback_placement(
        layout,
        gated(),
        15 * GIB,
        8192,
        quantised = False,
        kv_bytes_floor = 0,
        kv_on_host = False,
    )
    assert small is not None and large is not None
    moved = [sum(g.bytes_total for g in p.host_groups) for p in (small, large)]
    # A bigger card moves LESS, which is the property the discarded model lacked.
    assert moved[0] > moved[1] > 0
    assert all(m < layout.spillable_bytes for m in moved)
    # The cache is never among what it moves: on MoE both arms keep it resident,
    # and generation duly measured 33.65 against 34.77 t/s, a 1.03x tie.
    assert all(p.kv_host_bytes == 0 for p in (small, large))


def test_the_dense_fallback_does_move_whole_layers_and_the_cache_with_them():
    """The other half, also measured: n_part=0, no overrides, cache off the GPU.

    ``--fit on`` chose 54 of 65 layers on a 16 GiB card and 38 of 65 on a 12 GiB
    one, both with no tensor overrides at all. So the planner's original premise
    is right for dense models and wrong for MoE, and the fallback has to branch.
    """
    layout = dense_layout()
    placement = _fit_fallback_placement(
        layout, gated(), 12 * GIB, 32768, quantised = False, kv_bytes_floor = 0, kv_on_host = False
    )
    assert placement is not None
    names = {g.name for g in placement.host_groups}
    assert "layers" in names, "a dense fit moves attention weights too"
    # ``kv_host_bytes``, the same field the MoE case above asserts is ZERO, and
    # not a host group named "kv" as this used to look for. The moved cache was
    # re-expressed in that field so it is charged at Access.KV_CACHE's calibrated
    # rate rather than the contiguous weight rate; asserting it here keeps the two
    # halves of the branch stated in one vocabulary, and makes this test fail if
    # the cache is ever demoted back to a plain host group, which is precisely the
    # regression it exists to catch.
    assert (
        placement.kv_host_bytes > 0
    ), "a dense fit drags the moved layers' cache to host with them"


def test_a_moved_layer_takes_its_share_of_the_recurrent_state_with_it():
    """The state follows the layer, so the fallback has to free it AND pay for it.

    llama.cpp allocates layer ``i``'s recurrent state in
    ``ggml_backend_dev_buffer_type(model.dev_layer(i))`` when offload is on
    (llama-memory-recurrent.cpp:85-89) -- the same branch that puts the attention
    cache with its layer at llama-kv-cache.cpp:214-225 -- which is what
    ``ModelLayout.recurrent_bytes`` documents.

    ``all_resident_bytes`` counts the whole state, and the dense fallback loop
    used to free none of it and price none of it. It is not caught upstream
    either: the uneven-cache abstain returns early on a single device, so a dense
    hybrid on one card walks straight into this. The result was a fallback built
    from more layers than llama.cpp would move, scored on that heavier placement,
    which biases the gate towards ACCEPTING -- at 4 GiB of state on this cell it
    is the whole difference between a spill and an abstain.
    """
    import dataclasses

    plain = dense_layout()
    hybrid = dataclasses.replace(plain, recurrent_bytes = 4 * GIB)
    card, ram = [12 * GIB], 94 * GIB
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)

    placement = _fit_fallback_placement(hybrid, gated(), 12 * GIB, 32768, **args)
    assert placement is not None
    priced = [g for g in placement.host_groups if g.name.startswith("recurrent")]
    assert priced and priced[0].bytes_total > 0, "the moved state is never charged to the host"

    # Freed as well as charged: the state is 4 GiB of VRAM that moving layers
    # really does release, so the fallback reaches the budget on fewer of them
    # than a loop that only ever frees weights and cache. 39 layers before, 32
    # now, on a 64-block layout.
    def moved_layers(layout):
        p = _fit_fallback_placement(layout, gated(), 12 * GIB, 32768, **args)
        ffn = [g for g in p.host_groups if g.name == "ffn"][0]
        return round(ffn.bytes_total / int(0.20 * GIB))

    assert moved_layers(plain) < moved_layers(hybrid) < 39, (
        "a bigger state still needs more layers moved, but not as many as a loop "
        "that never frees it"
    )

    # And it reaches the verdict. Pre-fix this cell planned a spill at 1.13x;
    # the fallback it was beating was one llama.cpp would not have chosen.
    gate = plan_placement(hybrid, card, ram, 32768, opts = gated(host = HostProfile(threads = 6)))
    assert not gate.spilled_blocks
    assert "not worth it" in gate.reason


def test_the_recurrent_state_is_not_freed_twice_under_no_kv_offload():
    """``-nkvo`` already put the whole state on the host, so no layer frees any.

    ``llama_memory_hybrid`` hands ONE ``offload`` flag to both the attention cache
    and the recurrent memory (llama-memory-hybrid.cpp:28,40,58) and that flag is
    ``cparams.offload_kqv`` (llama-model.cpp:2445-2453), so with the cache off the
    device the recurrent state is off it too: ``ggml_backend_cpu_buffer_type()``
    unless ``offload`` (llama-memory-recurrent.cpp:85-91).

    ``resident_floor_bytes`` says the same and leaves ``recurrent_bytes`` out of
    ``resident`` on that branch, so a per-layer share freed here is bytes that
    were never counted: the modeled fitter satisfies the budget on fewer layers
    than llama.cpp must actually move, and is then billed a host recurrent group
    for state BOTH arms carry.
    """
    import dataclasses

    hybrid = dataclasses.replace(dense_layout(), recurrent_bytes = 4 * GIB)
    nkvo = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = True)

    placement = _fit_fallback_placement(hybrid, gated(), 12 * GIB, 32768, **nkvo)
    assert placement is not None
    assert not [g for g in placement.host_groups if g.name.startswith("recurrent")], (
        "state that -nkvo already moved is common to both placements and must not "
        "be charged to the fitter alone"
    )

    # Nothing about the state is freed either, so the fitter has to move exactly
    # as many layers as it would with no state at all.
    def moved_layers(layout):
        p = _fit_fallback_placement(layout, gated(), 12 * GIB, 32768, **nkvo)
        assert p is not None
        ffn = [g for g in p.host_groups if g.name == "ffn"][0]
        return round(ffn.bytes_total / int(0.20 * GIB))

    assert moved_layers(hybrid) == moved_layers(dense_layout())

    # ``kv_on_host`` is the only thing switched off: with the cache on the device
    # the state still follows its layer, so the test above keeps its teeth.
    on_device = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    still = _fit_fallback_placement(hybrid, gated(), 12 * GIB, 32768, **on_device)
    assert [g for g in still.host_groups if g.name.startswith("recurrent")]


def test_a_spill_larger_than_host_ram_is_refused_outright():
    """The one configuration measured to be unambiguously worse than the fitter.

    Refused before any cost comparison, because the cost model cannot see it: it
    prices host bytes at host bandwidth, which holds only while they are IN host
    memory. Past that they come from disk, and the planner also loses
    ``--load-mode none`` -- mmap can page, a no-mmap load cannot -- which is
    where most of its measured advantage came from.

    On a 12.67 GiB host the planner ran at 0.31x and 0.23x of ``--fit on`` on
    generation, and its dense placement failed to load at all, twice, while the
    fitter completed both times on the same box.
    """
    layout = dense_layout()
    # The near-tie band, so the cost comparison alone would plan this cell and
    # the refusal below is demonstrably the RAM test rather than the cost one.
    card = [18688 * 1024 * 1024]
    roomy = plan_placement(
        layout,
        card,
        94 * GIB,
        32768,
        opts = gated(host = HostProfile(threads = 6), min_penalty_reduction = 0.0),
    )
    assert roomy.spills_anything, "with RAM to spare this cell is planned"

    cramped = plan_placement(
        layout,
        card,
        4 * GIB,
        32768,
        opts = gated(host = HostProfile(threads = 6), min_penalty_reduction = 0.0),
    )
    assert not cramped.spills_anything
    assert "host RAM" in cramped.reason and "--fit on" in cramped.reason


def test_the_ram_refusal_counts_the_bytes_the_load_really_puts_on_the_host():
    """token_embd is host-resident on every launch, spilled or not.

    Leaving it out would let a plan sit just under the limit on paper and go over
    it in practice, which is the direction that costs a load rather than a rung.
    """
    layout = dense_layout()
    opts = gated(host = HostProfile(threads = 6), min_penalty_reduction = 0.0)
    # Sized so the spill alone fits but the spill plus token_embd does not.
    plan = plan_placement(layout, [18688 * 1024 * 1024], 4608 * MIB, 32768, opts = opts)
    if not plan.spills_anything:
        assert "host RAM" in plan.reason
        needed = float(plan.reason.split("needs ")[1].split(" GiB")[0])
        assert needed >= layout.token_embd_bytes / GIB


# ------------------------------------------------- the MoE long-prompt gate


def _moe_cell(**opts):
    return plan_placement(
        moe_layout(), [12 * GIB], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6), **opts)
    )


def test_a_moe_at_a_long_prompt_per_slot_is_left_to_the_fitter_before_it_is_ranked():
    """-ot measured 0.94 to 0.97x of llama.cpp's layerwise fit at a 32K prompt on
    5 cells, 2 models, 3 hosts. The ranking cannot see it (both arms spill through
    -ot), so the gate declines on the measurement and never ranks."""
    got = _moe_cell()
    assert not got.spills_anything and got.declined_by_gate
    assert "tokens per slot" in got.reason, got.reason
    assert got.predicted_request_ms == 0.0 and got.predicted_fit_request_ms == 0.0
    ungated = _moe_cell(moe_long_prompt_ctx = 0)
    # With the gate off the ranking runs and reports its numbers, whatever it decides.
    assert ungated.predicted_fit_request_ms > 0.0, ungated.reason


def test_the_moe_gate_is_per_slot_not_total():
    layout = replace(moe_layout(), n_ctx_train = 131072)
    # min_parallel pins the slots: rung 1 would otherwise trade them for cache
    # before any weight moved, and one slot at 65536 IS the long-prompt point.
    opts = gated(host = HostProfile(threads = 6), n_parallel = 4, min_parallel = 4)
    got = plan_placement(layout, [12 * GIB], 94 * GIB, 65536, opts = opts)
    # 65536 over four slots is 16384 per slot: not the long-prompt point.
    assert "tokens per slot" not in got.reason, got.reason
    one_slot = plan_placement(
        layout, [12 * GIB], 94 * GIB, 65536, opts = gated(host = HostProfile(threads = 6))
    )
    assert "tokens per slot" in one_slot.reason, one_slot.reason


def test_the_moe_gate_boundary_is_inclusive():
    exactly = _moe_cell(moe_long_prompt_ctx = 32768)
    assert "tokens per slot" in exactly.reason
    above = _moe_cell(moe_long_prompt_ctx = 32769)
    assert "tokens per slot" not in above.reason


def test_a_declined_moe_keeps_its_first_refusal_when_no_context_is_accepted():
    """What comes back is the refusal at the REQUESTED context, not the last one tried.

    Re-anchored: this cell is now refused by the long-prompt veto, which is a
    measurement the context does not move, so FIT_ONLY never walks down at all and
    the answer is the requested context by construction. It used to reach the same
    answer the long way -- the ladder ran, every context below the long-prompt point
    was an exact tie with the fitter, nothing was accepted, and the original refusal
    was kept -- and both routes have to end here."""
    got = _moe_cell(context_policy = ContextPolicy.FIT_ONLY)
    assert got.declined_by_gate and got.veto and got.n_ctx == 32768
    assert "tokens per slot" in got.reason, got.reason


def test_the_draft_drop_penalty_can_turn_a_win_into_a_decline():
    """Rung 2 drops the draft to make room; a fitter keeps it. With the penalty at
    0 the spill is scored as if the draft were free to lose; at 2.0 the same
    cell is refused."""
    layout = dense_layout()
    draft = GIB
    card = [14848 * 1024 * 1024 + draft]
    base = dict(
        host = HostProfile(threads = 6),
        min_penalty_reduction = 0.0,
        draft_bytes = draft,
        draft_droppable = True,
    )
    free = plan_placement(layout, card, 94 * GIB, 32768, opts = gated(**base))
    assert free.draft_dropped and free.spills_anything, free.reason
    priced = plan_placement(
        layout, card, 94 * GIB, 32768, opts = gated(**base, draft_drop_penalty_frac = 2.0)
    )
    assert priced.declined_by_gate and not priced.spills_anything, priced.reason


def test_fit_only_shrinks_to_the_largest_context_the_gate_accepts():
    """The near-tie cell is refused at 32768. FIT_ONLY walks down and accepts a
    smaller context; the step above it is still refused, so it is the largest."""
    layout = dense_layout()
    card = [14848 * 1024 * 1024]
    strict = plan_placement(layout, card, 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6)))
    assert strict.declined_by_gate and not strict.spills_anything
    shrunk = plan_placement(
        layout,
        card,
        94 * GIB,
        32768,
        opts = gated(host = HostProfile(threads = 6), context_policy = ContextPolicy.FIT_ONLY),
    )
    assert shrunk.changed and shrunk.spills_anything, shrunk.reason
    assert 4096 <= shrunk.n_ctx < 32768, shrunk.n_ctx
    above = plan_placement(
        layout, card, 94 * GIB, shrunk.n_ctx + 1024, opts = gated(host = HostProfile(threads = 6))
    )
    assert above.declined_by_gate, above.reason


def test_the_fitter_keeps_lm_head_on_the_device():
    """A partial ``--fit on`` never moves the output tensor, so it is never charged.

    llama.cpp keeps rows ``[i_gpu_start, i_gpu_start + act_gpu_layers)`` with
    ``i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)``, and takes the
    output row's device from ``get_layer_buft_list(n_layer_all)``
    (llama-model.cpp:1467-1492). Substituting any ``n_gpu_layers >= 1`` leaves
    row ``n_layer_all`` inside the window, so lm_head is resident for EVERY
    partial fit -- it is the last row to leave, not the first.

    Charging it did two wrong things at once: it credited the fallback with
    freeing bytes ``-ngl`` cannot free, so the fallback appeared to fit a layer
    or two early, and then billed it a host lm_head at SINGLE_MATVEC rates that
    llama.cpp never pays. Both inflate the fallback's score, which is the
    direction that lets a spill through this gate.
    """
    layout = dense_layout()
    opts = gated(host = HostProfile(threads = 6))
    placement = _fit_fallback_placement(
        layout,
        opts,
        13 * GIB,
        32768,
        quantised = False,
        kv_bytes_floor = 0,
        kv_on_host = False,
    )
    assert placement is not None
    assert "lm_head" not in {group.name for group in placement.host_groups}
    assert placement.host_groups, "the fallback still moves layers, just not the output row"


def test_a_spill_the_real_fitter_beats_is_declined_at_the_margin():
    """The decision, not just the placement: this cell used to be planned.

    A 27B on a 18.8 GiB card is the '#9861 nearly fitted anyway' shape -- the
    fitter has one or two layers to move and the spill has 22 blocks. Billing
    the fallback a host lm_head made it look 1.6x more expensive than it is and
    the gate took the spill; without that charge the fallback wins and the
    planner correctly stands down.
    ---
    The reserve's context term once started at 16384 and this budget carried +343 MiB for
    it; the term now starts at 32768 and the budget is back at its original value.
    """
    layout = dense_layout()
    plan = plan_placement(
        layout, [18800 * 1024 * 1024], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6))
    )
    assert not plan.spills_anything
    assert "not worth it" in plan.reason


def test_the_draft_drop_penalty_default_is_the_measured_generation_cost():
    """Rung 2 (dropping the MTP / draft) was priced free until it was measured.
    Four fully-resident A100 cells on Qwen3.6-35B-A3B Q4 put the draft at +2.3 to
    +7.6 percent generation (one 3-repeat outlier at +14.3 that its 5-repeat
    re-run did not reproduce), so the default charges 5 percent. Pinned so a
    change is deliberate and re-measured, not drifted."""
    assert PlanOptions().draft_drop_penalty_frac == 0.05


def graded_moe_layout(n_blocks: int = 8, ffn_gib: float = 3.0) -> ModelLayout:
    """A MoE whose blocks carry the per-rung breakdown a real GGUF gives them.

    ``moe_layout`` above leaves the three FFN rungs at 0, which makes every block
    ungraded and sends the ladder down its coarse whole-FFN path -- where the
    planner's placement and the modelled fitter's are the same bytes and ``rank``
    can only tie. A graded layout is what the seam actually builds, and it is the
    one where the ladder moves LESS than the fitter would and the gate accepts.
    """
    third = int(ffn_gib * GIB) // 3
    blocks = tuple(
        BlockLayout(
            i,
            third * 3,
            int(0.025 * GIB),
            ffn_down_bytes = third,
            ffn_up_bytes = third,
            ffn_gate_bytes = third,
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
        n_ctx_train = 131072,
        is_moe = True,
        n_expert = 128,
        n_expert_used = 8,
        complete = True,
    )


def test_a_unified_cache_gives_every_slot_the_whole_prompt_window():
    """--kv-unified is one shared stream, so n_ctx / slots is not the window.

    llama-server reports n_ctx_slot = n_ctx to every slot under a unified cache
    and n_ctx / n_parallel without one, and Studio appends --kv-unified on every
    multi-slot launch the binary supports it on. Dividing regardless made a 32K
    shared window with four slots look like 8K, skipped the measured long-prompt
    veto, and ACCEPTED -ot at exactly the 32K point where it measured 0.94 to
    0.97x of llama.cpp's own fit.
    """
    layout = graded_moe_layout()
    host = HostProfile(threads = 12)
    divided = plan_placement(
        layout,
        [20 * GIB],
        200 * GIB,
        32768,
        opts = gated(host = host, n_parallel = 4, min_parallel = 4),
    )
    # Four private 8K windows really are below the long-prompt point.
    assert divided.spills_anything and "tokens per slot" not in divided.reason

    unified = plan_placement(
        layout,
        [20 * GIB],
        200 * GIB,
        32768,
        opts = gated(host = host, n_parallel = 4, min_parallel = 4, kv_unified = True),
    )
    assert not unified.spills_anything and unified.declined_by_gate
    assert "32768 tokens per slot" in unified.reason, unified.reason

    # And one slot is the same answer either way: nothing to divide.
    one_slot = plan_placement(layout, [20 * GIB], 200 * GIB, 32768, opts = gated(host = host))
    assert "tokens per slot" in one_slot.reason, one_slot.reason


def test_the_measured_moe_veto_still_applies_when_the_fallback_cannot_be_modelled():
    """An unmodellable fallback is not a licence to skip a MEASUREMENT.

    Where moving every expert is still short of the budget,
    ``_fit_fallback_placement`` returns None. llama.cpp does not fail there: with
    ``global_surplus_cpu_moe <= 0`` common/fit.cpp returns its step-3 placement,
    which simply assigns fewer dense-only layers to the device. So None means
    "nothing to RANK against", not "the fitter cannot place this" -- and the
    long-prompt veto, which is a measurement rather than a comparison, must not
    be skipped with it. It was: this was the only path by which a gated MoE ever
    accepted -ot at a 32K prompt.

    The modelled fitter has since gained that step-3 stage, so the unmodellable
    case no longer arises here; the veto must still fire FIRST, before any
    ranking, and with it disabled the plan is ranked rather than waved through.
    """
    layout = graded_moe_layout(n_blocks = 64, ffn_gib = 0.5)
    got = plan_placement(
        layout, [4 * GIB], 200 * GIB, 32768, opts = gated(host = HostProfile(threads = 12))
    )
    assert not got.spills_anything and got.declined_by_gate
    assert "tokens per slot" in got.reason, got.reason
    assert got.predicted_fit_request_ms == 0.0, "the veto fires before any ranking"
    # ...and with the veto disabled the fitter's layer-lowering stage gives the
    # gate an arm to rank against, so the verdict is a comparison, not a pass.
    ungated = plan_placement(
        layout,
        [4 * GIB],
        200 * GIB,
        32768,
        opts = gated(host = HostProfile(threads = 12), moe_long_prompt_ctx = 0),
    )
    assert "tokens per slot" not in ungated.reason
    assert ungated.predicted_fit_request_ms > 0, ungated.reason


def mixed_quant_layout(n_blocks: int = 64) -> ModelLayout:
    """A dense layout whose blocks are NOT all the same size.

    Real GGUFs are like this: a dynamic/UD quant keeps the first blocks at a
    denser type than the middle ones, and a dense recurrent hybrid interleaves
    attention and SSM blocks that differ by several times. The uniform layouts
    above cannot see which END of the block list a placement takes.
    """
    blocks = tuple(
        BlockLayout(i, int((0.30 if i < 8 else 0.20) * GIB), int(0.045 * GIB))
        for i in range(n_blocks)
    )
    return ModelLayout(
        arch = "qwen3",
        n_layers = n_blocks,
        n_attention_layers = n_blocks,
        blocks = blocks,
        lm_head_bytes = int(1.0 * GIB),
        token_embd_bytes = int(1.0 * GIB),
        other_resident_bytes = int(0.01 * GIB),
        kv_bytes_per_token_f16 = 2.0 * GIB / 32768,
        n_ctx_train = 32768,
        complete = True,
    )


def test_the_dense_fallback_moves_the_fitters_leading_block_prefix():
    """llama.cpp offloads the LEADING blocks, so the model has to price those.

    On a dense model the fitter only lowers ``n_gpu_layers``
    (common/fit.cpp:551-559 sets it from the per-device layer counts, with
    ``n_part`` 0), and llama-model.cpp then computes
    ``i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`` and hands every row
    with ``il < i_gpu_start`` to the CPU device (llama-model.cpp:1479-1484). The
    host therefore takes a PREFIX of the block list.

    Walking from the other end is equivalent only when every block is the same
    size, and the loop's terms are per-block BYTES. On a mixed-quant or hybrid
    layout the two directions disagree on the host weights and on the layer
    count, so the fitter arm of the cost gate gets scored on a placement
    llama.cpp would not produce.

    A leading prefix of WHOLE layers, plus at most the FFN of the next one: the
    fitter's step 4 keeps that layer resident and overrides part of its FFN
    rather than lowering ngl again, which is the boundary grading this models.
    """
    layout = mixed_quant_layout()
    placement = _fit_fallback_placement(
        layout, gated(), 12 * GIB, 32768, quantised = False, kv_bytes_floor = 0, kv_on_host = False
    )
    assert placement is not None
    weights = sum(g.bytes_total for g in placement.host_groups if g.name in ("ffn", "layers"))

    sizes = [b.spillable_bytes + b.resident_bytes for b in layout.blocks]
    whole = [sum(sizes[:k]) for k in range(len(sizes) + 1)]
    k = max(i for i, total in enumerate(whole) if total <= weights)
    overflow = weights - whole[k]
    assert k < len(sizes), "the moved weights are not a leading prefix of the block list"
    assert (
        overflow <= layout.blocks[k].spillable_bytes
    ), "the boundary layer gave up more than its FFN"
    moved = k + (1 if overflow else 0)
    assert 0 < moved < len(sizes), "a partial fit, or the two ends cannot be told apart"
    assert weights != sum(sizes[-moved:]), (
        "the trailing blocks of this layout weigh the same as the leading ones, "
        "so the assertion above proves nothing"
    )


def test_the_fallback_charges_the_recurrent_state_once_per_slot():
    """The state is one copy per sequence, resident and moved alike; a fitter
    modelled at one copy on a four-slot hybrid frees a quarter of what moving a
    layer really frees and pays a quarter of the host work, so it stops at the
    wrong layer count and the gate scores an arm the child never runs."""
    import dataclasses

    hybrid = dataclasses.replace(dense_layout(), recurrent_bytes = GIB)
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    one = _fit_fallback_placement(hybrid, gated(), 12 * GIB, 32768, n_seq = 1, **args)
    four = _fit_fallback_placement(hybrid, gated(), 12 * GIB, 32768, n_seq = 4, **args)
    assert one is not None and four is not None

    def per_moved_layer(placement):
        layers = next(g for g in placement.host_groups if g.name == "layers").bytes_total
        moved = layers / hybrid.blocks[0].resident_bytes
        state = next(g for g in placement.host_groups if g.name.startswith("recurrent"))
        return state.bytes_total / moved

    assert abs(per_moved_layer(four) / per_moved_layer(one) - 4) < 0.05


def test_the_moe_fallback_lowers_layers_once_every_expert_is_on_the_host():
    """When the deficit outruns every expert, common/fit.cpp does not fail: it
    lowers n_gpu_layers and moves whole leading layers with their cache. Modelled
    as None, the planner's lm_head rung went through with no comparison at all."""
    layout = moe_layout()
    tight = int(1.5 * GIB)
    placement = _fit_fallback_placement(
        layout, gated(), tight, 8192, quantised = False, kv_bytes_floor = 0, kv_on_host = False
    )
    assert placement is not None
    names = [g.name for g in placement.host_groups]
    assert "layers" in names, names
    experts = next(g for g in placement.host_groups if g is placement.host_groups[0])
    assert experts.bytes_total == layout.spillable_bytes
    assert placement.kv_host_bytes > 0

    # And the gate now has an arm to rank the lm_head rung against.
    from core.inference.offload_planner import plan_placement

    ranked = None
    for tenths in range(15, 40):
        vram = tenths * GIB // 10
        plan = plan_placement(layout, [vram], 64 * GIB, 8192, opts = gated(allow_lm_head_spill = True))
        if any("output" in p for p in plan.ot_patterns) or (
            plan.declined_by_gate and plan.predicted_fit_request_ms > 0
        ):
            ranked = plan
            break
    assert ranked is not None, "no budget reached the lm_head rung"
    assert ranked.predicted_fit_request_ms > 0, ranked.reason


def test_a_saturated_windowed_cache_is_charged_flat_when_the_fitter_moves_it():
    """A windowed model's measured floor is context-flat once its window is
    saturated; scaling it by the live fraction charged the fitter a fraction of a
    cache it reads in full, and the gate declined spills the hardware wins."""
    layout = replace(dense_layout(), arch = "gemma3", has_swa = True)
    floor = 4 * GIB
    n_ctx = 32768
    live_fraction = (2048 + 128) / n_ctx
    placement = _fit_fallback_placement(
        layout,
        gated(workload_prompt_tokens = 2048, workload_generated_tokens = 128),
        8 * GIB,
        n_ctx,
        quantised = False,
        kv_bytes_floor = floor,
        kv_on_host = False,
    )
    assert placement is not None
    per_block = layout.blocks[0].spillable_bytes + layout.blocks[0].resident_bytes
    # Whole layers only: the graded boundary layer stays on the device with its
    # cache, so the tail of that division is FFN bytes and not a moved row.
    moved = int(sum(g.bytes_total for g in placement.host_groups) // per_block)
    assert moved > 0
    flat = floor * moved / len(layout.blocks)
    assert placement.kv_host_bytes >= flat * 0.98, (placement.kv_host_bytes, flat)
    assert placement.kv_host_bytes > flat * live_fraction * 2


def test_an_unbounded_prompt_cache_declines_a_weight_spill():
    """--cache-ram -1 keeps an accepted spill pageable, and the gate scored it with
    host-side numbers measured unmapped, where mapped reads run 2 to 4.6x slower;
    a spill that won on paper could lose on the launch it got."""
    layout = dense_layout()
    budget = 14848 * 1024 * 1024
    lenient = dict(host = HostProfile(threads = 6), min_penalty_reduction = 0.0)
    accepted = plan_placement(layout, [budget], 94 * GIB, 32768, opts = gated(**lenient))
    assert accepted.spilled_blocks, accepted.reason
    declined = plan_placement(
        layout, [budget], 94 * GIB, 32768, opts = gated(**lenient, prompt_cache_unbounded = True)
    )
    assert declined.declined_by_gate and not declined.spilled_blocks
    assert "unbounded" in declined.reason and "--fit on" in declined.reason
    # Ungated callers still get the pageable plan the seam asked for.
    assert plan_placement(
        layout, [budget], 94 * GIB, 32768, opts = PlanOptions(prompt_cache_unbounded = True)
    ).spilled_blocks


def test_the_fallbacks_live_cache_is_capped_at_the_slot_window():
    """A request lives in one slot's window. The fitter's moved-cache term was
    capped at the whole context, so once the prompt filled its slot the generated
    tokens priced live cache the slot cannot hold, on the fitter's arm alone."""
    import dataclasses

    layout = dense_layout()
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    shape = dict(workload_prompt_tokens = 32768, workload_generated_tokens = 256)

    def live_per_moved_layer(placement):
        layers = next(g for g in placement.host_groups if g.name == "layers").bytes_total
        return placement.kv_host_bytes / (layers / layout.blocks[0].resident_bytes)

    one = _fit_fallback_placement(layout, gated(**shape), 12 * GIB, 32768, n_seq = 1, **args)
    four = _fit_fallback_placement(
        layout, gated(**shape, n_parallel = 4), 12 * GIB, 32768, n_seq = 4, **args
    )
    unified = _fit_fallback_placement(
        layout, gated(**shape, n_parallel = 4, kv_unified = True), 12 * GIB, 32768, n_seq = 4, **args
    )
    assert one is not None and four is not None and unified is not None
    assert abs(live_per_moved_layer(one) / live_per_moved_layer(four) - 4) < 0.05
    assert abs(live_per_moved_layer(unified) / live_per_moved_layer(one) - 1) < 0.05


def test_the_moe_long_prompt_veto_falls_through_at_the_context_asked_for():
    """A veto is a measurement, and a smaller context does not change it.

    ``declined_by_gate`` invites FIT_ONLY to try a shorter context, so the ladder
    walked this cell down to the first per-slot context below the long-prompt
    point and accepted the SAME spill at 31744. Nothing measured says the spill
    is better there: the 5 cells behind the veto say -ot loses to llama.cpp's own
    layerwise fit on MoE at a long prompt, so the fall-through is that fit, at the
    context the caller asked for and not at one the planner picked instead.
    """
    opts = gated(host = HostProfile(threads = 12), context_policy = ContextPolicy.FIT_ONLY)
    got = plan_placement(graded_moe_layout(), [20 * GIB], 200 * GIB, 65536, opts = opts)
    assert got.declined_by_gate and got.veto
    assert not got.spills_anything and not got.changed, got.reason
    assert got.n_ctx == 65536, got.reason
    assert "tokens per slot" in got.reason and "--fit on" in got.reason, got.reason


def big_head_layout(n_blocks: int = 8) -> ModelLayout:
    """A layout whose output head outweighs everything the fitter can move.

    ``_fit_fallback_placement`` walks whole layers, and llama.cpp keeps the output
    row on the device for any ``n_gpu_layers >= 1``, so with every layer moved the
    head and the output norms are still resident. When those alone are over budget
    the loop runs out of layers and answers None. The planner has a rung the fitter
    does not -- lm_head through ``-ot`` -- so it can place what the fitter cannot,
    which is exactly the shape that reached the gate with nothing to rank against.
    """
    blocks = tuple(BlockLayout(i, int(0.1 * GIB), int(0.05 * GIB)) for i in range(n_blocks))
    return ModelLayout(
        arch = "qwen3",
        n_layers = n_blocks,
        n_attention_layers = n_blocks,
        blocks = blocks,
        lm_head_bytes = 3 * GIB,
        token_embd_bytes = int(0.1 * GIB),
        other_resident_bytes = int(0.01 * GIB),
        kv_bytes_per_token_f16 = 1024,
        n_ctx_train = 32768,
        complete = True,
    )


def test_a_fallback_that_cannot_be_modelled_declines_rather_than_waves_the_spill_through():
    """An unranked spill is what the gate exists to stop.

    ``fallback is None`` was an ACCEPT, so the one class of layout this loop
    cannot walk got its spill taken with no comparison at all -- the layouts whose
    arithmetic is least trustworthy, passed through the check meant to catch it.
    llama.cpp still places these loads; it is only the MODEL of that placement
    that is missing, so the honest answer is to leave it to --fit on.
    """
    layout = big_head_layout()
    opts = gated(
        host = HostProfile(threads = 6),
        overhead_bytes_per_device = 0,
        overhead_bytes_per_token = 0,
    )
    assert (
        _fit_fallback_placement(
            layout, opts, 3 * GIB, 8192, quantised = False, kv_bytes_floor = 0, kv_on_host = False
        )
        is None
    ), "the fixture must reach the unmodellable branch"
    got = plan_placement(layout, [3 * GIB], 94 * GIB, 8192, opts = opts)
    assert got.declined_by_gate and not got.spills_anything, got.reason
    assert "could not be modelled" in got.reason and "--fit on" in got.reason, got.reason
    # Ungated, the planner still answers what it CAN place: lm_head is its own rung.
    ungated = plan_placement(
        layout,
        [3 * GIB],
        94 * GIB,
        8192,
        opts = PlanOptions(overhead_bytes_per_device = 0, overhead_bytes_per_token = 0),
    )
    assert ungated.spilled_lm_head, ungated.reason


def test_a_knob_only_plan_is_ranked_against_the_launch_the_caller_typed():
    """A plan that spills nothing but drops the draft was never scored at all.

    ``_cost_gate`` ran only for a plan with host bytes, so rung 2 could give the
    draft away for free: ``draft_drop_penalty_frac`` multiplied a spill cost
    that did not exist. The fallback arm of a no-spill plan is the caller's own
    launch as llama.cpp would fit it, which is priceable, so it is priced.
    """
    from core.inference.offload_planner import _device_reserve, all_resident_bytes

    layout = dense_layout()
    draft = 2 * GIB
    # Short by half the draft, so rung 2 closes it and no weight moves.
    needed = all_resident_bytes(layout, 32768)
    card = [needed + _device_reserve(PlanOptions(), 32768) + draft - GIB // 2]
    base = dict(host = HostProfile(threads = 6), draft_bytes = draft, draft_droppable = True)
    free = plan_placement(
        layout, card, 94 * GIB, 32768, opts = gated(**base, draft_drop_penalty_frac = 0.0)
    )
    assert free.draft_dropped and not free.spills_anything, free.reason
    assert free.predicted_fit_request_ms > 0.0, "the fitter's arm has to be priced"
    assert free.predicted_request_ms == 0.0, "nothing on the host, so nothing to charge"

    priced = plan_placement(
        layout, card, 94 * GIB, 32768, opts = gated(**base, draft_drop_penalty_frac = 5.0)
    )
    assert priced.declined_by_gate and not priced.draft_dropped, priced.reason
    assert priced.predicted_request_ms > priced.predicted_fit_request_ms
    assert "not worth it" in priced.reason and "--fit on" in priced.reason


def test_a_caller_nkvo_cache_is_host_ram_the_refusal_has_to_see():
    """-nkvo puts the WHOLE cache in host RAM, and the refusal counted only the
    spilled weights, the embedding and a pinned projector. A spill that fits the
    box with the cache left out was admitted onto a box the cache had already
    filled, which is the one configuration measured to be worse than --fit on.
    """
    from core.inference.offload_planner import _device_reserve, all_resident_bytes, cache_bytes

    layout = dense_layout()
    base = dict(host = HostProfile(threads = 6), kv_on_host = True, min_penalty_reduction = 0.0)
    needed = all_resident_bytes(layout, 32768, kv_on_host = True)
    card = [needed + _device_reserve(PlanOptions(), 32768) - 2 * GIB]
    roomy = plan_placement(layout, card, 200 * GIB, 32768, opts = gated(**base))
    assert roomy.spills_anything, roomy.reason

    cache = cache_bytes(layout, 32768)
    # What the refusal used to count: the embedding and the spilled weights.
    weights = sum(layout.blocks[i].spillable_bytes for i in roomy.spilled_blocks)
    without = layout.token_embd_bytes + weights
    assert cache > 0 and roomy.host_bytes == without + cache

    headroom = PlanOptions().host_ram_headroom_bytes
    # Room for all of that and not for the cache.
    refused = plan_placement(layout, card, headroom + without + MIB, 32768, opts = gated(**base))
    assert refused.declined_by_gate and not refused.spills_anything, refused.reason
    assert "host RAM" in refused.reason
    # Room for the cache as well, and the same spill is taken.
    fits = plan_placement(layout, card, headroom + without + cache, 32768, opts = gated(**base))
    assert fits.spilled_blocks == roomy.spilled_blocks, fits.reason


def test_the_fallback_places_the_cache_by_the_per_layer_vector():
    """A moved layer takes ITS OWN cache with it, not an even share of the total.

    The fitter spread the cache uniformly over every block, which charges a
    recurrent or MLP-only row a share it does not hold. On a hybrid that
    over-states what the leading rows free, so the modelled fitter reaches the
    budget early and is scored on a placement llama.cpp would not produce -- the
    direction that makes the fallback look slower than it is and lets the gate
    take spills it should decline.
    """
    layout = dense_layout()
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    uniform = _fit_fallback_placement(layout, gated(), 12 * GIB, 32768, **args)
    # One attention row in five, so the fitter's prefix does not carry a whole
    # number of periods and the two answers cannot coincide by construction.
    hybrid = [1 if i % 5 == 4 else 0 for i in range(layout.n_layers)]
    spread = _fit_fallback_placement(
        layout, gated(), 12 * GIB, 32768, kv_layer_weights = hybrid, **args
    )
    assert uniform is not None and spread is not None
    assert spread.kv_host_bytes < uniform.kv_host_bytes

    # And rows that hold no cache at all take none with them: here every
    # attention row is behind the prefix the fitter moves, so it frees no cache
    # and has to move more weights instead.
    recurrent_first = [1 if i >= layout.n_layers // 2 else 0 for i in range(layout.n_layers)]
    none = _fit_fallback_placement(
        layout, gated(), 12 * GIB, 32768, kv_layer_weights = recurrent_first, **args
    )
    assert none is not None and none.kv_host_bytes == 0
    assert sum(g.bytes_total for g in none.host_groups) > sum(
        g.bytes_total for g in uniform.host_groups
    )


def graded_dense_layout(n_blocks: int = 64) -> ModelLayout:
    """A dense model with its FFN broken out per matrix, so the boundary layer
    has fractions to give."""
    d, u, g, a = int(0.08 * GIB), int(0.07 * GIB), int(0.05 * GIB), int(0.045 * GIB)
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = d + u + g,
            resident_bytes = a,
            ffn_down_bytes = d,
            ffn_up_bytes = u,
            ffn_gate_bytes = g,
            attn_bytes = a,
        )
        for i in range(n_blocks)
    )
    return replace(dense_layout(n_blocks), blocks = blocks)


def test_the_fallback_grades_its_boundary_layer_the_way_fit_cpp_does():
    """common/fit.cpp does not lower ngl one more time when a fraction of a layer
    would do: step 4 keeps that layer on the device and overrides part of its FFN
    (LAYER_FRACTION_UP, narrowed to _GATE, widened to _ATTN), so the layer's
    attention and cache stay resident. Walking whole layers charged the fallback
    a boundary layer it only partly needed."""
    graded = graded_dense_layout()
    whole = replace(
        graded,
        blocks = tuple(
            BlockLayout(
                index = b.index, spillable_bytes = b.spillable_bytes, resident_bytes = b.resident_bytes
            )
            for b in graded.blocks
        ),
    )
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    fine = _fit_fallback_placement(graded, gated(), 12 * GIB, 32768, **args)
    coarse = _fit_fallback_placement(whole, gated(), 12 * GIB, 32768, **args)
    assert fine is not None and coarse is not None
    fine_bytes = sum(g.bytes_total for g in fine.host_groups)
    coarse_bytes = sum(g.bytes_total for g in coarse.host_groups)
    assert fine_bytes < coarse_bytes, (fine_bytes, coarse_bytes)

    # The boundary really is a FRACTION of one block, and the whole layers below
    # it are a leading prefix.
    per_block = graded.blocks[0].spillable_bytes + graded.blocks[0].resident_bytes
    assert fine_bytes % per_block == graded.blocks[0].ffn_down_bytes


def test_the_per_layer_vector_reaches_the_gate_from_plan_placement():
    """The vector is a plan_placement argument and the fallback is built two
    calls below it, so the end-to-end path is worth pinning: without the
    threading the gate scores an arm that ignores what the caller measured."""
    layout = dense_layout()
    card = [14848 * 1024 * 1024]
    hybrid = [1 if i % 5 == 4 else 0 for i in range(layout.n_layers)]
    opts = gated(host = HostProfile(threads = 6), min_penalty_reduction = 0.0)
    plain = plan_placement(layout, card, 94 * GIB, 32768, opts = opts)
    with_vector = plan_placement(layout, card, 94 * GIB, 32768, opts = opts, kv_layer_weights = hybrid)
    assert plain.spilled_blocks == with_vector.spilled_blocks
    assert plain.predicted_fit_request_ms != with_vector.predicted_fit_request_ms


def test_the_fitter_is_modelled_on_the_cache_size_the_caller_measured():
    """Both arms have to describe ONE cache, and only one of them did.

    ``PlanOptions.kv_bytes_at`` prices the exact (context, slots) the child will
    run, and every feasibility site takes it as given (``trust_floor``). The
    fitter model did not: it went back through ``cache_bytes`` without the flag,
    which takes the max against the layout's product, and that product charges a
    quantised cache ONE byte per element. A q4_0 cache is 4.5 bits
    (``_kv_bytes_per_elem`` in llama_cpp.py), so the fitter was modelled on a
    cache 1.78x its real size, appeared to move layers to carry cache it does not
    hold, and the gate approved a spill the real fitter beats.
    """
    layout = dense_layout(kv_gib_at_32k = 6.0)
    exact = int(layout.kv_bytes(32768, 1) * 0.5625)

    def kv_at(n_ctx: int, slots: int) -> int:
        return int(layout.kv_bytes(n_ctx, 1) * 0.5625)

    opts = gated(host = HostProfile(threads = 6), cache_quantised = True, kv_bytes_at = kv_at)
    plan = plan_placement(
        layout, [14848 * 1024 * 1024], 94 * GIB, 32768, kv_bytes_floor = exact, opts = opts
    )
    assert plan.declined_by_gate, plan.reason
    assert not plan.spilled_blocks, plan.reason
    # The fitter arm has to move on the measured cache, not the product's.
    untrusted = plan_placement(
        layout,
        [14848 * 1024 * 1024],
        94 * GIB,
        32768,
        kv_bytes_floor = exact,
        opts = replace(opts, kv_bytes_at = None),
    )
    assert untrusted.predicted_fit_request_ms > plan.predicted_fit_request_ms
