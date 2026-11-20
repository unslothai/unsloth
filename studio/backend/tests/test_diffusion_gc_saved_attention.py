# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: the budgeted checkpoint keeps long attention outputs of the blocks it checkpoints, so the backward
recompute does not rerun those kernels; gradients are unchanged, short (text cross-attention) calls are recomputed,
the env switch turns it off and an out-of-memory fall back drops the kept outputs."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import core.training.diffusion_gc_budget as gb

_SDPA_FORWARDS = {
    getattr(torch.ops.aten, n).default
    for n in (
        "_scaled_dot_product_flash_attention_for_cpu",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
    )
    if hasattr(torch.ops.aten, n)
}


class _CountAttention(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func in _SDPA_FORWARDS:
            self.calls += 1
        return func(*args, **(kwargs or {}))


class _Block(torch.nn.Module):
    def __init__(
        self,
        dim = 16,
        heads = 2,
    ):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.out = torch.nn.Linear(dim, dim)

    def forward(self, x, context):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(h.transpose(1, 2).reshape(b, n, d))
        # Cross-attention over a short context: cheap, recomputed.
        c = context.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, c, c)
        return x + h.transpose(1, 2).reshape(b, n, d)


def _run(checkpoint, tokens = 1024):
    torch.manual_seed(0)
    block = _Block()
    x = torch.randn(1, tokens, 16, requires_grad = True)
    context = torch.randn(1, 8, 16)
    out = checkpoint(block, x, context = context)
    counter = _CountAttention()
    with counter:
        out.float().square().mean().backward()
    grads = [x.grad] + [p.grad for p in block.parameters()]
    return counter.calls, grads


def _supported():
    return gb.save_attention_context_fn() is not None


def test_budget_checkpoint_reuses_long_attention_outputs(monkeypatch):
    monkeypatch.delenv(gb.SAVE_ATTENTION_ENV, raising = False)
    if not _supported():
        pytest.skip("this torch has no compile-safe selective checkpointing")
    budget = gb.BudgetedCheckpoint()
    budget.probing = False  # planned, nothing kept: every block checkpointed
    budget.planned = True
    reruns, grads = _run(budget)
    # Only the 8-key cross-attention reruns in the backward; the 1024-key self-attention comes from the first pass.
    assert reruns == 1
    assert budget.base.saved_attention == 1
    # Output [1, 2, 1024, 8] fp32 plus an fp32 logsumexp per query row.
    assert budget.base.saved_attention_bytes == 1 * 2 * 1024 * 8 * 4 + 1 * 2 * 1024 * 4
    plain_reruns, plain_grads = _run(gb._Checkpoint())
    assert plain_reruns == 2
    assert all(torch.equal(a, b) for a, b in zip(grads, plain_grads))
    assert budget.can_retry


def test_short_sequences_are_recomputed(monkeypatch):
    monkeypatch.delenv(gb.SAVE_ATTENTION_ENV, raising = False)
    ckpt = gb._Checkpoint(save_attention = True)
    reruns, _ = _run(ckpt, tokens = 64)
    assert reruns == 2 and ckpt.saved_attention == 0


def test_env_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv(gb.SAVE_ATTENTION_ENV, "0")
    budget = gb.BudgetedCheckpoint()
    assert budget.base.context_fn is None
    budget.probing, budget.planned = False, True
    reruns, _ = _run(budget)
    assert reruns == 2
    assert not budget.can_retry


def test_out_of_memory_fall_back_drops_kept_attention(monkeypatch):
    monkeypatch.delenv(gb.SAVE_ATTENTION_ENV, raising = False)
    if not _supported():
        pytest.skip("this torch has no compile-safe selective checkpointing")
    budget = gb.BudgetedCheckpoint()
    budget.probing, budget.planned = False, True
    _run(budget)
    # Nothing kept, but attention outputs are: a rerun without them uses less memory.
    assert budget.can_retry
    assert budget.step_down() is False
    assert budget.disabled and budget.base.context_fn is None and not budget.can_retry
    reruns, _ = _run(budget)
    assert reruns == 2


def test_torch_without_compile_safe_modes_is_left_alone(monkeypatch):
    import torch.utils.checkpoint as ck

    monkeypatch.delenv(gb.SAVE_ATTENTION_ENV, raising = False)
    mode = getattr(ck, "_CachingTorchDispatchMode", None)
    if mode is None:
        assert gb.save_attention_context_fn() is None
        return
    # Older torch: the selective checkpoint mode would make dynamo skip the compiled blocks.
    monkeypatch.setattr(
        mode, "ignore_compile_internals", classmethod(lambda cls: False), raising = False
    )
    assert gb.save_attention_context_fn() is None


def test_plan_credits_the_attention_a_kept_block_gives_back(monkeypatch):
    monkeypatch.delenv(gb.SAVE_ATTENTION_ENV, raising = False)
    if not _supported():
        pytest.skip("this torch has no compile-safe selective checkpointing")
    budget = gb.BudgetedCheckpoint()
    torch.manual_seed(0)
    block = _Block()
    x = torch.randn(1, 1024, 16, requires_grad = True)
    budget.block_class[1] = "_Block"
    budget.block_proxy[1] = x.numel()
    out = budget._measure_attention("_Block", 1, block, (x,), {"context": torch.randn(1, 8, 16)})
    out.sum().backward()
    per_elem = budget.class_attention["_Block"]
    assert per_elem * x.numel() == budget.base.saved_attention_bytes > 0
    # Step 1's peak holds the attention outputs of every checkpointed block: keeping a block costs the rest.
    budget.class_cost["_Block"] = 10.0 * per_elem
    assert budget.estimate(1) == int(9.0 * per_elem * x.numel())
    budget.class_attention.clear()
    assert budget.estimate(1) == int(10.0 * per_elem * x.numel())
