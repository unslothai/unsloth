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
from core.inference.offload_planner import PlanOptions, plan_placement

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
