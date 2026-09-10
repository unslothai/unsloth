# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The planner may now decline a spill it is perfectly capable of making."""

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
    """A bare PlanOptions still answers the old question, unchanged: only the launch seam
    has somewhere to fall through to, so only the launch seam asks."""
    assert PlanOptions().require_cost_win is False
    layout = dense_layout()
    plan = plan_placement(layout, [14848 * 1024 * 1024], 94 * GIB, 32768)
    assert plan.spilled_blocks, "ungated planning should still spill"


def test_a_spill_that_barely_beats_the_fitter_is_declined():
    """A near-tie is not worth taking, because the two errors are not symmetric: a decline
    measured nearly free and a wrong plan measured up to 8x slower."""
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
    """The gate only ever chooses between two viable placements: with no fallback to lose
    to, declining would trade a slow launch for a failed one."""
    layout = dense_layout()
    tiny = 3 * GIB
    plan = plan_placement(layout, [tiny], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6)))
    assert "not worth it" not in plan.reason


def test_the_workload_shape_moves_the_trade():
    """rank() needs a request shape and the answer depends on it: a spill is worst during
    prefill and best during a long decode. Every shape below carries the same TOTAL, so
    the split is isolated from the fallback's live-cache term."""
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
    """Set the margin to zero and the same cell is planned again, which confirms the
    decline above is the MARGIN talking: at this budget the spill really is cheaper, by
    less than the margin. Do not re-anchor to a budget where the spill is dearer."""
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
    """Measured from the fitter's own trace, not from a buffer report: the fitter and the
    planner arrive at nearly the same placement on MoE, so the gate should usually find
    them equivalent and decline."""
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
    assert moved[0] > moved[1] > 0
    assert all(m < layout.spillable_bytes for m in moved)
    assert all(p.kv_host_bytes == 0 for p in (small, large))


def test_the_dense_fallback_does_move_whole_layers_and_the_cache_with_them():
    """The other half, also measured: n_part=0, no overrides, cache off the GPU."""
    layout = dense_layout()
    placement = _fit_fallback_placement(
        layout, gated(), 12 * GIB, 32768, quantised = False, kv_bytes_floor = 0, kv_on_host = False
    )
    assert placement is not None
    names = {g.name for g in placement.host_groups}
    assert "layers" in names, "a dense fit moves attention weights too"
    # ``kv_host_bytes``, not a host group named "kv": the moved cache is charged at
    # Access.KV_CACHE's calibrated rate, and this fails if it is ever demoted back.
    assert (
        placement.kv_host_bytes > 0
    ), "a dense fit drags the moved layers' cache to host with them"


def test_a_moved_layer_takes_its_share_of_the_recurrent_state_with_it():
    """The state follows the layer, so the fallback has to free it AND pay for it."""
    import dataclasses

    plain = dense_layout()
    hybrid = dataclasses.replace(plain, recurrent_bytes = 4 * GIB)
    card, ram = [12 * GIB], 94 * GIB
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)

    placement = _fit_fallback_placement(hybrid, gated(), 12 * GIB, 32768, **args)
    assert placement is not None
    priced = [g for g in placement.host_groups if g.name.startswith("recurrent")]
    assert priced and priced[0].bytes_total > 0, "the moved state is never charged to the host"

    def moved_layers(layout):
        p = _fit_fallback_placement(layout, gated(), 12 * GIB, 32768, **args)
        ffn = [g for g in p.host_groups if g.name == "ffn"][0]
        return round(ffn.bytes_total / int(0.20 * GIB))

    assert moved_layers(plain) < moved_layers(hybrid) < 39, (
        "a bigger state still needs more layers moved, but not as many as a loop "
        "that never frees it"
    )

    gate = plan_placement(hybrid, card, ram, 32768, opts = gated(host = HostProfile(threads = 6)))
    assert not gate.spilled_blocks
    assert "not worth it" in gate.reason


def test_the_recurrent_state_is_not_freed_twice_under_no_kv_offload():
    """``-nkvo`` already put the whole state on the host, so no layer frees any."""
    import dataclasses

    hybrid = dataclasses.replace(dense_layout(), recurrent_bytes = 4 * GIB)
    nkvo = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = True)

    placement = _fit_fallback_placement(hybrid, gated(), 12 * GIB, 32768, **nkvo)
    assert placement is not None
    assert not [g for g in placement.host_groups if g.name.startswith("recurrent")], (
        "state that -nkvo already moved is common to both placements and must not "
        "be charged to the fitter alone"
    )

    def moved_layers(layout):
        p = _fit_fallback_placement(layout, gated(), 12 * GIB, 32768, **nkvo)
        assert p is not None
        ffn = [g for g in p.host_groups if g.name == "ffn"][0]
        return round(ffn.bytes_total / int(0.20 * GIB))

    assert moved_layers(hybrid) == moved_layers(dense_layout())

    on_device = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    still = _fit_fallback_placement(hybrid, gated(), 12 * GIB, 32768, **on_device)
    assert [g for g in still.host_groups if g.name.startswith("recurrent")]


def test_a_spill_larger_than_host_ram_is_refused_outright():
    """The one configuration measured to be unambiguously worse than the fitter, refused
    before any cost comparison: the cost model prices host bytes at host bandwidth, which
    holds only while they are IN host memory, and the plan also loses ``--load-mode none``."""
    layout = dense_layout()
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
    """token_embd is host-resident on every launch, spilled or not."""
    layout = dense_layout()
    opts = gated(host = HostProfile(threads = 6), min_penalty_reduction = 0.0)
    plan = plan_placement(layout, [18688 * 1024 * 1024], 4608 * MIB, 32768, opts = opts)
    if not plan.spills_anything:
        assert "host RAM" in plan.reason
        needed = float(plan.reason.split("needs ")[1].split(" GiB")[0])
        assert needed >= layout.token_embd_bytes / GIB


def _moe_cell(**opts):
    return plan_placement(
        moe_layout(), [12 * GIB], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6), **opts)
    )


def test_a_moe_at_a_long_prompt_per_slot_is_left_to_the_fitter_before_it_is_ranked():
    """-ot measured worse than llama.cpp's layerwise fit at a 32K prompt."""
    got = _moe_cell()
    assert not got.spills_anything and got.declined_by_gate
    assert "tokens per slot" in got.reason, got.reason
    assert got.predicted_request_ms == 0.0 and got.predicted_fit_request_ms == 0.0
    ungated = _moe_cell(moe_long_prompt_ctx = 0)
    assert ungated.predicted_fit_request_ms > 0.0, ungated.reason


def test_the_moe_gate_is_per_slot_not_total():
    layout = replace(moe_layout(), n_ctx_train = 131072)
    opts = gated(host = HostProfile(threads = 6), n_parallel = 4, min_parallel = 4)
    got = plan_placement(layout, [12 * GIB], 94 * GIB, 65536, opts = opts)
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
    """What comes back is the refusal at the REQUESTED context, not the last one tried:
    the long-prompt veto is a measurement the context does not move, so FIT_ONLY never
    walks down at all."""
    got = _moe_cell(context_policy = ContextPolicy.FIT_ONLY)
    assert got.declined_by_gate and got.veto and got.n_ctx == 32768
    assert "tokens per slot" in got.reason, got.reason


def test_the_draft_drop_penalty_can_turn_a_win_into_a_decline():
    """Rung 2 drops the draft to make room; a fitter keeps it."""
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
    """The near-tie cell is refused at 32768."""
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
    """A partial ``--fit on`` never moves the output tensor, so it is never charged."""
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
    """The decision, not just the placement: this cell used to be planned."""
    layout = dense_layout()
    plan = plan_placement(
        layout, [18800 * 1024 * 1024], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6))
    )
    assert not plan.spills_anything
    assert "not worth it" in plan.reason


def test_the_draft_drop_penalty_default_is_the_measured_generation_cost():
    """Rung 2 (dropping the MTP / draft) was priced free until it was measured."""
    assert PlanOptions().draft_drop_penalty_frac == 0.05


def graded_moe_layout(n_blocks: int = 8, ffn_gib: float = 3.0) -> ModelLayout:
    """A MoE whose blocks carry the per-rung breakdown a real GGUF gives them."""
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

    Re-anchored at ``min_penalty_reduction = 0``: now that the fitter grades its boundary
    block the same way the planner does, the two arms place the same bytes on this layout
    and the default margin declines the tie, which would hide the window this pins.
    """
    layout = graded_moe_layout()
    host = HostProfile(threads = 12)
    tie = dict(host = host, min_penalty_reduction = 0.0)
    divided = plan_placement(
        layout,
        [20 * GIB],
        200 * GIB,
        32768,
        opts = gated(n_parallel = 4, min_parallel = 4, **tie),
    )
    assert divided.spills_anything and "tokens per slot" not in divided.reason

    unified = plan_placement(
        layout,
        [20 * GIB],
        200 * GIB,
        32768,
        opts = gated(n_parallel = 4, min_parallel = 4, kv_unified = True, **tie),
    )
    assert not unified.spills_anything and unified.declined_by_gate
    assert "32768 tokens per slot" in unified.reason, unified.reason

    one_slot = plan_placement(layout, [20 * GIB], 200 * GIB, 32768, opts = gated(**tie))
    assert "tokens per slot" in one_slot.reason, one_slot.reason


def test_the_measured_moe_veto_still_applies_when_the_fallback_cannot_be_modelled():
    """An unmodellable fallback is not a licence to skip a MEASUREMENT."""
    layout = graded_moe_layout(n_blocks = 64, ffn_gib = 0.5)
    got = plan_placement(
        layout, [4 * GIB], 200 * GIB, 32768, opts = gated(host = HostProfile(threads = 12))
    )
    assert not got.spills_anything and got.declined_by_gate
    assert "tokens per slot" in got.reason, got.reason
    assert got.predicted_fit_request_ms == 0.0, "the veto fires before any ranking"
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
    """A dense layout whose blocks are NOT all the same size, as real GGUFs are: the uniform
    layouts above cannot see which END of the block list a placement takes."""
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
    """llama.cpp offloads the LEADING blocks, so the model has to price those."""
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
    """The state is one copy per sequence, resident and moved alike, so a fitter modelled at
    one copy on a four-slot hybrid stops at the wrong layer count and the gate scores an
    arm the child never runs."""
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
    assert plan_placement(
        layout, [budget], 94 * GIB, 32768, opts = PlanOptions(prompt_cache_unbounded = True)
    ).spilled_blocks


def test_the_fallbacks_live_cache_is_capped_at_the_slot_window():
    """A request lives in one slot's window."""
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
    """A veto is a measurement, and a smaller context does not change it: ``declined_by_gate``
    would invite FIT_ONLY to walk down and accept the SAME spill at a slightly shorter
    context, so the fall-through is llama.cpp's fit at the context the caller asked for."""
    opts = gated(host = HostProfile(threads = 12), context_policy = ContextPolicy.FIT_ONLY)
    got = plan_placement(graded_moe_layout(), [20 * GIB], 200 * GIB, 65536, opts = opts)
    assert got.declined_by_gate and got.veto
    assert not got.spills_anything and not got.changed, got.reason
    assert got.n_ctx == 65536, got.reason
    assert "tokens per slot" in got.reason and "--fit on" in got.reason, got.reason


def big_head_layout(n_blocks: int = 8) -> ModelLayout:
    """A layout whose output head outweighs everything the fitter can move: with every layer
    moved the head is still resident, so the loop answers None, while the planner has a rung
    the fitter does not.
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
    """An unranked spill is what the gate exists to stop: ``fallback is None`` was an ACCEPT,
    so the layouts whose arithmetic is least trustworthy were passed through the check meant
    to catch them."""
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
    ungated = plan_placement(
        layout,
        [3 * GIB],
        94 * GIB,
        8192,
        opts = PlanOptions(overhead_bytes_per_device = 0, overhead_bytes_per_token = 0),
    )
    assert ungated.spilled_lm_head, ungated.reason


def test_a_knob_only_plan_is_ranked_against_the_launch_the_caller_typed():
    """A plan that spills nothing but drops the draft was never scored at all: ``_cost_gate``
    ran only for a plan with host bytes, so ``draft_drop_penalty_frac`` multiplied a spill
    cost that did not exist."""
    from core.inference.offload_planner import _device_reserve, all_resident_bytes

    layout = dense_layout()
    draft = 2 * GIB
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
    """-nkvo puts the WHOLE cache in host RAM, and the refusal counted only the spilled
    weights, the embedding and a pinned projector, so a spill was admitted onto a box the
    cache had already filled."""
    from core.inference.offload_planner import _device_reserve, all_resident_bytes, cache_bytes

    layout = dense_layout()
    base = dict(host = HostProfile(threads = 6), kv_on_host = True, min_penalty_reduction = 0.0)
    needed = all_resident_bytes(layout, 32768, kv_on_host = True)
    card = [needed + _device_reserve(PlanOptions(), 32768) - 2 * GIB]
    roomy = plan_placement(layout, card, 200 * GIB, 32768, opts = gated(**base))
    assert roomy.spills_anything, roomy.reason

    cache = cache_bytes(layout, 32768)
    weights = sum(layout.blocks[i].spillable_bytes for i in roomy.spilled_blocks)
    without = layout.token_embd_bytes + weights
    assert cache > 0 and roomy.host_bytes == without + cache

    headroom = PlanOptions().host_ram_headroom_bytes
    refused = plan_placement(layout, card, headroom + without + MIB, 32768, opts = gated(**base))
    assert refused.declined_by_gate and not refused.spills_anything, refused.reason
    assert "host RAM" in refused.reason
    fits = plan_placement(layout, card, headroom + without + cache, 32768, opts = gated(**base))
    assert fits.spilled_blocks == roomy.spilled_blocks, fits.reason


def test_the_fallback_places_the_cache_by_the_per_layer_vector():
    layout = dense_layout()
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    uniform = _fit_fallback_placement(layout, gated(), 12 * GIB, 32768, **args)
    hybrid = [1 if i % 5 == 4 else 0 for i in range(layout.n_layers)]
    spread = _fit_fallback_placement(
        layout, gated(), 12 * GIB, 32768, kv_layer_weights = hybrid, **args
    )
    assert uniform is not None and spread is not None
    assert spread.kv_host_bytes < uniform.kv_host_bytes

    recurrent_first = [1 if i >= layout.n_layers // 2 else 0 for i in range(layout.n_layers)]
    none = _fit_fallback_placement(
        layout, gated(), 12 * GIB, 32768, kv_layer_weights = recurrent_first, **args
    )
    assert none is not None and none.kv_host_bytes == 0
    assert sum(g.bytes_total for g in none.host_groups) > sum(
        g.bytes_total for g in uniform.host_groups
    )


def graded_dense_layout(n_blocks: int = 64) -> ModelLayout:
    """A dense model with its FFN broken out per matrix, so the boundary layer has fractions to give."""
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
    """common/fit.cpp does not lower ngl one more time when a fraction of a layer would do:
    step 4 keeps that layer on the device and overrides part of its FFN, so its attention
    and cache stay resident."""
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
    """Both arms have to describe ONE cache, and only one of them did: the fitter model went
    back through ``cache_bytes`` without ``trust_floor``, and that product charges a
    quantised cache ONE byte per element, so it was modelled on a cache 1.78x its real size."""
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
    untrusted = plan_placement(
        layout,
        [14848 * 1024 * 1024],
        94 * GIB,
        32768,
        kv_bytes_floor = exact,
        opts = replace(opts, kv_bytes_at = None),
    )
    assert untrusted.predicted_fit_request_ms > plan.predicted_fit_request_ms


def test_a_load_that_already_fits_gives_the_fitter_nothing_to_move():
    """--fit on only moves what does not fit, so a load inside the budget is placed whole.
    Modelling it as a spill invented a cost for the fitter and let the gate accept a spill
    it should have ranked against a free launch."""
    from core.inference.offload_planner import (
        SpillClass,
        SpillUnit,
        _cost_gate,
        _knob_only_gate,
        _Knobs,
    )

    roomy = 200 * GIB
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    for layout in (dense_layout(), moe_layout()):
        placement = _fit_fallback_placement(layout, gated(), roomy, 8192, **args)
        assert placement is not None, "a load that fits is still a placement, not an unknown"
        assert placement.host_groups == [], [g.name for g in placement.host_groups]
        assert placement.kv_host_bytes == 0

    layout = dense_layout()
    units = [SpillUnit(b.index, SpillClass.FFN_DOWN, b.spillable_bytes) for b in layout.blocks[:4]]
    declined, plan_ms, fit_ms = _cost_gate(
        layout,
        gated(),
        8192,
        units,
        False,
        roomy,
        quantised = False,
        kv_bytes_floor = 0,
        host_ram_bytes = 512 * GIB,
    )
    assert fit_ms == 0.0, fit_ms
    assert plan_ms > 0.0 and declined is not None and declined.declined_by_gate

    knob_declined, knob_ms, knob_fit_ms = _knob_only_gate(
        layout, gated(), 8192, roomy, quantised = False, kv_bytes_floor = 0, knobs = _Knobs(n_parallel = 1)
    )
    assert knob_fit_ms == 0.0 and knob_ms == 0.0
    assert knob_declined is None, "a free plan against a free fit is a tie, not a decline"


def per_matrix_moe_layout(n_blocks: int = 40) -> ModelLayout:
    """An MoE with its expert matrices broken out, so the boundary block has rungs to give."""
    d, u, g, a = int(0.20 * GIB), int(0.15 * GIB), int(0.12 * GIB), int(0.025 * GIB)
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
    return replace(moe_layout(n_blocks), blocks = blocks)


def test_the_moe_fallback_grades_its_boundary_block_like_the_dense_arm():
    """fit.cpp grades the first partial layer it reaches and moves LAYER_FRACTION_MOE of
    every layer past it. Adding the boundary block's whole expert set over-moved by up to
    one block, made the fitter look costlier than it is, and biased the gate toward
    accepting the planner's spill."""
    from core.inference.offload_planner import all_resident_bytes

    layout = per_matrix_moe_layout()
    down = layout.blocks[0].ffn_down_bytes
    gate = layout.blocks[0].ffn_gate_bytes
    whole = layout.blocks[0].spillable_bytes
    args = dict(quantised = False, kv_bytes_floor = 0, kv_on_host = False)
    resident = all_resident_bytes(
        layout, 8192, kv_quantised = False, kv_bytes_floor = 0, kv_on_host = False, n_seq = 1
    )

    def moved(deficit_past_three: int) -> int:
        budget = resident - 3 * whole - deficit_past_three
        placement = _fit_fallback_placement(layout, gated(), budget, 8192, **args)
        assert placement is not None
        return sum(g.bytes_total for g in placement.host_groups)

    assert moved(down // 2) == 3 * whole + down
    assert moved(down + gate // 2) == 3 * whole + down + gate
    assert moved(down + gate + 1) == 4 * whole
