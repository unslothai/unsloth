# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the two step-1 VRAM planners together: the nf4 dequant cache (sized in the optimizer hook) and
budgeted gradient checkpointing (sized right after). Both read the same free memory, so their sum must stay inside one
planner's budget, the gc plan must be the one it makes alone, and a pinned cache must come out of the gc plan."""

from __future__ import annotations

import pytest
import torch

import core.training.diffusion_dit_trainer as dit
import core.training.diffusion_gc_budget as gb
import core.training.diffusion_lora_fast as fl

GIB = 1 << 30
MIB = 1 << 20


def _budget(per_block, blocks):
    b = gb.BudgetedCheckpoint()
    b.class_cost = {"B": 1.0}
    for idx in range(blocks):
        b.block_class[idx] = "B"
        b.block_proxy[idx] = int(per_block)
    return b


def _step_one(
    monkeypatch,
    *,
    total,
    free,
    reserved,
    peak,
    pre,
    post,
    per_block,
    blocks,
    pin_gb = None,
):
    """Run step 1's optimizer step through the real hooks with simulated CUDA memory, then the trainer's step-1 plan
    (main: the gc plan alone). Returns (gc budget, cache, cache plan events)."""
    if pin_gb is None:
        monkeypatch.delenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", str(pin_gb))
    mem = {"alloc": pre}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (free, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: mem["alloc"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: peak)

    p = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.AdamW([p], lr = 1e-3)
    # Registered first, so the cache's hooks see the pre-step and post-step allocations (the optimizer state).
    opt.register_step_pre_hook(lambda *_: mem.update(alloc = pre))
    opt.register_step_post_hook(lambda *_: mem.update(alloc = post))
    cache = fl.Nf4DequantCache()
    planned = []
    budget = _budget(per_block, blocks)
    hook_cb = planned.append
    joint = getattr(dit, "_plan_after_first_step", None)
    if joint is not None:
        hook_cb = None  # the trainer reports the plan after the gc reservation
    handles = fl.attach_training_hooks(opt, cache, 1.0, hook_cb)
    p.grad = torch.ones(4)
    fb_peak, alloc_before = peak, pre
    opt.step()
    if joint is not None:
        joint(budget, cache, fb_peak, alloc_before, None, planned.append)
    else:
        dit._plan_gc_budget(budget, fb_peak, alloc_before)
    for h in handles:
        h.remove()
    return budget, cache, planned


def _one_budget(total, free, reserved, peak, pre, post):
    """What either planner may take alone: 0.85 x (free + reserved - projected peak - margin)."""
    return fl.dequant_budget_bytes(free, reserved, peak + (post - pre), total)


# Colab T4 fp16 klein (15 GiB card, the observed case): the cache planned 7.44 GiB and the gc plan kept all 25
# blocks (7.1 GiB); L4 klein at 1024 px, where the gc plan alone can fill the whole headroom; A100-40 Wan.
CASES = {
    "t4_klein_512": dict(
        total = 15 * GIB,
        free = 10 * GIB,
        reserved = 4 * GIB,
        peak = int(3.3 * GIB),
        pre = int(2.5 * GIB),
        post = int(2.6 * GIB),
        per_block = 290 * MIB,
        blocks = 25,
    ),
    "l4_klein_1024": dict(
        total = 22 * GIB,
        free = 15 * GIB,
        reserved = 5 * GIB,
        peak = int(5.5 * GIB),
        pre = int(2.5 * GIB),
        post = int(2.6 * GIB),
        per_block = 900 * MIB,
        blocks = 25,
    ),
    "a100_wan_704": dict(
        total = 40 * GIB,
        free = 33 * GIB,
        reserved = 5 * GIB,
        peak = int(4.3 * GIB),
        pre = int(3.0 * GIB),
        post = int(3.2 * GIB),
        per_block = 95 * MIB,
        blocks = 30,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_both_planners_sum_within_one_budget(monkeypatch, case):
    c = CASES[case]
    budget, cache, planned = _step_one(monkeypatch, **c)
    one = _one_budget(c["total"], c["free"], c["reserved"], c["peak"], c["pre"], c["post"])
    assert budget.kept_bytes > 0
    # Step 2 holds kept activations and cache entries on top of step 1's peak: together inside one planner's budget,
    # which leaves the 15% share and the max(2 GiB, 10%) margin each planner was designed with.
    assert budget.kept_bytes + cache.budget <= one
    margin = max(2 * GIB, int(c["total"] * 0.10))
    step2 = c["peak"] + (c["post"] - c["pre"]) + budget.kept_bytes + cache.budget
    assert step2 <= c["free"] + c["reserved"] - margin
    # Rebalancing from step 2 on also leaves the kept activations alone.
    assert cache.extra_reserved == budget.kept_bytes
    assert len(planned) == 1 and planned[0]["budget_gib"] == round(cache.budget / GIB, 3)


@pytest.mark.parametrize("case", sorted(CASES))
def test_gc_plan_is_unchanged_and_cache_keeps_the_rest(monkeypatch, case):
    c = CASES[case]
    alone = _budget(c["per_block"], c["blocks"])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (c["free"], c["total"]))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: c["reserved"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: c["post"])
    dit._plan_gc_budget(alone, c["peak"], c["pre"])
    budget, cache, _ = _step_one(monkeypatch, **c)
    assert budget.keep == alone.keep and budget.kept_bytes == alone.kept_bytes
    one = _one_budget(c["total"], c["free"], c["reserved"], c["peak"], c["pre"], c["post"])
    assert cache.budget == max(0, one - budget.kept_bytes) == cache.cap


def test_pinned_cache_is_held_out_of_the_gc_plan(monkeypatch):
    c = dict(CASES["l4_klein_1024"])
    budget, cache, _ = _step_one(monkeypatch, pin_gb = 4, **c)
    assert cache.budget == 4 * GIB  # a pinned budget is the user's: never shrunk
    held = dict(c, peak = c["peak"] + 4 * GIB)
    assert budget.kept_bytes <= _one_budget(
        held["total"], held["free"], held["reserved"], held["peak"], held["pre"], held["post"]
    )
    margin = max(2 * GIB, int(c["total"] * 0.10))
    step2 = c["peak"] + (c["post"] - c["pre"]) + budget.kept_bytes + cache.budget
    assert step2 <= c["free"] + c["reserved"] - margin


def test_without_a_gc_budget_the_cache_plan_is_unchanged(monkeypatch):
    # UNSLOTH_DIFFUSION_GC_MODE=plain (or ROCm): no gc budget, so no reservation and the hook reports the plan.
    c = CASES["t4_klein_512"]
    monkeypatch.delenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", raising = False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (c["free"], c["total"]))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: c["reserved"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: c["post"])
    cache = fl.Nf4DequantCache()
    fl.plan_dequant_cache(cache, c["peak"], c["pre"])
    assert cache.budget == _one_budget(
        c["total"], c["free"], c["reserved"], c["peak"], c["pre"], c["post"]
    )
    assert cache.extra_reserved == 0 and cache.stats().get("reserved_gib", 0.0) == 0.0


def test_reserve_cuts_the_plan_evicts_and_hands_back_on_fallback():
    cache = fl.Nf4DequantCache()
    cache.plan(1000 * MIB)
    for i in range(6):
        cache.entries[i] = torch.empty(0)
        cache.sizes[i] = 100 * MIB
    cache.used = 600 * MIB
    # A 700 MiB reservation leaves 300 MiB of the plan: the newest entries go.
    assert cache.reserve(700 * MIB, from_plan = True) == 300 * MIB
    assert cache.budget == cache.cap == 300 * MIB and cache.used == 300 * MIB and cache.full
    assert sorted(cache.entries) == [0, 1, 2]
    # Without from_plan a reservation only narrows rebalancing, as before.
    other = fl.Nf4DequantCache()
    other.plan(1000 * MIB)
    assert other.reserve(700 * MIB) == 0 and other.budget == other.cap == 1000 * MIB
    # The OOM fallback checkpoints every block again: the cap returns to the plan; live memory decides the regrowth.
    assert cache.reserve(0, from_plan = True) == 0
    assert cache.cap == 1000 * MIB and cache.budget == 300 * MIB
    cache.margin = 0
    cache.rebalance(free = 4 * GIB, reserved = 0, allocated = 0)
    assert cache.budget == 1000 * MIB and not cache.full


def test_reserve_never_shrinks_a_pinned_budget():
    cache = fl.Nf4DequantCache()
    cache.plan(2 * GIB, pinned = True)
    assert cache.reserve(5 * GIB, from_plan = True) == 0
    assert cache.budget == cache.cap == 2 * GIB and cache.extra_reserved == 5 * GIB
