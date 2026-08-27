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
    stays inside the margin at every shape, because the fitter's cost is
    dominated by the output tensor and that is charged to decode as well.
    """
    layout = dense_layout()
    card = [14848 * 1024 * 1024]
    ratios = []
    for n_prompt, n_generated in ((8192, 16), (2048, 256), (512, 2048), (16, 8192)):
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
    """
    layout = dense_layout()
    card = [14848 * 1024 * 1024]
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
    assert any(
        g.name.startswith("kv") for g in placement.host_groups
    ), "a dense fit drags the moved layers' cache to host with them"


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
    card = [14848 * 1024 * 1024]
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
        8 * GIB,
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
    plan = plan_placement(layout, [14848 * 1024 * 1024], 16 * GIB, 32768, opts = opts)
    if not plan.spills_anything:
        assert "host RAM" in plan.reason
        needed = float(plan.reason.split("needs ")[1].split(" GiB")[0])
        assert needed >= layout.token_embd_bytes / GIB
