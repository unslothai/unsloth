# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MoE helpers on an empty shard or an all-masked micro-batch: the router aux loss is 0 (not NaN) and the remote
MoE training dispatch returns an empty output instead of raising. CPU only."""

import pytest
import torch
from torch import nn

import unsloth  # noqa: F401
from unsloth import context_parallel
from unsloth.models.pp_native import _load_balancing_from_statistics, _router_statistics
from unsloth.models.remote_moe_shims import _moe_train_dispatch


def test_load_balancing_from_statistics_with_no_rows_is_zero():
    num_experts, top_k = 4, 2
    logits = (torch.randn(6, num_experts, requires_grad = True),)
    mask = torch.zeros(1, 6, dtype = torch.bool)  # every token masked (an all-padding micro-batch)
    stats = _router_statistics(logits, num_experts, top_k, mask)
    assert float(stats[-1]) == 0.0
    loss = _load_balancing_from_statistics(stats, num_experts)
    assert torch.isfinite(loss) and float(loss) == 0.0, loss
    loss.backward()
    assert torch.isfinite(logits[0].grad).all()


def test_load_balancing_from_statistics_with_rows_is_unchanged():
    num_experts, top_k = 4, 2
    logits = (torch.randn(6, num_experts),)
    mask = torch.tensor([[1, 1, 0, 1, 0, 1]], dtype = torch.bool)
    stats = _router_statistics(logits, num_experts, top_k, mask)
    counts, probs, rows = stats[:num_experts], stats[num_experts:-1], stats[-1]
    expected = torch.sum((counts / rows) * (probs / rows)) * num_experts
    assert torch.equal(_load_balancing_from_statistics(stats, num_experts), expected)


@pytest.mark.parametrize("cp", [False, True], ids = ["no_cp", "cp_shard"])
def test_remote_moe_train_dispatch_with_no_tokens_returns_empty(monkeypatch, cp):
    """A shard that routes no tokens (cp: every expert still runs on zero rows for its zero gradient)."""
    block = nn.Module()
    block.experts = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
    if cp:
        monkeypatch.setattr(context_parallel, "_ACTIVE_MANAGER", object())
    x = torch.randn(0, 4, requires_grad = True)
    topk_idx = torch.zeros(0, 2, dtype = torch.long)
    topk_weight = torch.zeros(0, 2)
    out = _moe_train_dispatch(block, x, topk_idx, topk_weight)
    assert out.shape == (0, 4) and out.dtype == x.dtype
    if cp:
        out.sum().backward()
        assert all(e.weight.grad is not None and not e.weight.grad.any() for e in block.experts)


def test_remote_moe_train_dispatch_with_tokens_is_unchanged():
    torch.manual_seed(0)
    block = nn.Module()
    block.experts = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
    x = torch.randn(5, 4)
    topk_idx = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]])
    topk_weight = torch.rand(5, 2)
    out = _moe_train_dispatch(block, x, topk_idx, topk_weight)
    expected = sum(
        topk_weight[:, j : j + 1]
        * torch.stack([block.experts[int(e)](x[i]) for i, e in enumerate(topk_idx[:, j])])
        for j in range(2)
    )
    torch.testing.assert_close(out, expected)
