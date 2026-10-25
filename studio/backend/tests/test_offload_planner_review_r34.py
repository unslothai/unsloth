# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which spills count as "every block" on a multi-GPU plan.

Only a spill that takes the same tensors off every block keeps each device's share independent of
where llama.cpp cuts the layer ranges. A byte test on the routed experts alone called a MoE spill
with the shared/dense FFN of only some blocks "every block", so the pooled back-first pick skipped
the per-device selection, and grading the boundary block after that test turned an every-block
pick partial, so the planner vetoed its own selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference import offload_planner as OP  # noqa: E402
from core.inference.offload_layout import BlockLayout, ModelLayout, SpillClass  # noqa: E402
from core.inference.offload_planner import ContextPolicy, PlanOptions, plan_placement  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB


def _moe(
    lm_head_mib = 500,
    n = 32,
    routed_mib = 300,
    shared_mib = 150,
    leading_dense = 0,
):
    """``n`` MoE blocks (routed experts plus a shared FFN), after ``leading_dense`` dense-FFN
    blocks that hold no experts (DeepSeek-V3 / GLM-4.5)."""
    third = routed_mib * MIB // 3
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = 0 if i < leading_dense else 3 * third,
            resident_bytes = shared_mib * MIB,
            ffn_down_bytes = 0 if i < leading_dense else third,
            ffn_up_bytes = 0 if i < leading_dense else third,
            ffn_gate_bytes = 0 if i < leading_dense else third,
            dense_ffn_bytes = shared_mib * MIB,
        )
        for i in range(n)
    )
    return ModelLayout(
        arch = "qwen3moe",
        n_layers = n,
        n_attention_layers = n,
        blocks = blocks,
        lm_head_bytes = lm_head_mib * MIB,
        token_embd_bytes = 500 * MIB,
        other_resident_bytes = 10 * MIB,
        kv_bytes_per_token_f16 = 128 * 1024,
        n_ctx_train = 32768,
        is_moe = True,
        n_expert = 128,
        n_expert_used = 8,
        complete = True,
    )


def _dense(n = 32):
    third = 100 * MIB
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = 3 * third,
            resident_bytes = 100 * MIB,
            ffn_down_bytes = third,
            ffn_up_bytes = third,
            ffn_gate_bytes = third,
            attn_bytes = 100 * MIB,
        )
        for i in range(n)
    )
    return ModelLayout(
        arch = "qwen3",
        n_layers = n,
        n_attention_layers = n,
        blocks = blocks,
        lm_head_bytes = 500 * MIB,
        token_embd_bytes = 500 * MIB,
        other_resident_bytes = 10 * MIB,
        kv_bytes_per_token_f16 = 128 * 1024,
        n_ctx_train = 32768,
        complete = True,
    )


def _plan(
    layout,
    vram_mib,
    policy = ContextPolicy.NEVER_REDUCE,
):
    return plan_placement(
        layout,
        [v * MIB for v in vram_mib],
        256 * GIB,
        32768,
        opts = PlanOptions(context_policy = policy),
        split_weights_per_device = [1, 1],
    )


def test_a_partial_shared_ffn_spill_is_not_every_block():
    layout = _moe()
    experts = OP._units_of(layout.blocks, None)
    shared = OP._units_of(layout.blocks, SpillClass.DENSE_FFN)
    assert OP._spills_every_block(layout, experts)
    assert OP._spills_every_block(layout, experts + shared)
    assert not OP._spills_every_block(layout, experts + shared[:4])


@pytest.mark.parametrize("policy", [ContextPolicy.NEVER_REDUCE, ContextPolicy.FIT_ONLY])
def test_a_partial_shared_ffn_spill_is_picked_device_by_device(policy):
    """Untied: the pooled pick took the shared FFN off the last card, so the first card's shortfall
    was covered by spilling the output head. The per-device pick needs no head spill."""
    plan = _plan(_moe(), [5024, 6500], policy)
    assert plan.changed and plan.n_ctx == 32768, plan.reason
    assert not plan.spilled_lm_head, plan.reason
    assert "device by device" in plan.reason, plan.reason


@pytest.mark.parametrize("policy", [ContextPolicy.NEVER_REDUCE, ContextPolicy.FIT_ONLY])
def test_a_tied_moe_with_a_partial_shared_ffn_spill_still_places(policy):
    """Tied: no head to rescue with, so the pooled pick abstained (or shrank the context)."""
    plan = _plan(_moe(lm_head_mib = 0), [5024, 6500], policy)
    assert plan.changed and plan.n_ctx == 32768, plan.reason


def test_grading_the_boundary_block_does_not_veto_an_every_block_spill():
    """Every block was needed, grading trimmed the last one against the pooled deficit, and the
    now-partial spill was refused as uncheckable against a pooled budget."""
    plan = _plan(_dense(), [5500, 5500])
    assert "partial spill" not in plan.reason, plan.reason
    assert plan.changed and plan.n_ctx == 32768, plan.reason


def test_the_description_counts_each_rung_against_its_own_blocks():
    """3 leading dense blocks hold a dense FFN and no experts: 58 of 61, not "every block"."""
    layout = _moe(n = 61, leading_dense = 3)
    shared = OP._units_of(layout.blocks, SpillClass.DENSE_FFN)
    experts = OP._units_of(layout.blocks, None)
    assert OP._rung_description(shared[3:], layout) == "the shared/dense FFN of 58 of 61 blocks"
    assert OP._rung_description(shared, layout) == "the shared/dense FFN of every block"
    assert OP._rung_description(experts[:10], layout) == "the FFN of 10 of 58 blocks"
