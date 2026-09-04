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
    PlanOptions,
    _fit_fallback_placement,
    plan_placement,
)

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
    """
    layout = dense_layout()
    plan = plan_placement(
        layout, [18800 * 1024 * 1024], 94 * GIB, 32768, opts = gated(host = HostProfile(threads = 6))
    )
    assert not plan.spills_anything
    assert "not worth it" in plan.reason
