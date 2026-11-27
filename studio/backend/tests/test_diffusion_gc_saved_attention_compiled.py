# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Saved-attention checkpoint around a compiled block matches a plain checkpoint (pytorch #201158)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import core.training.diffusion_gc_budget as gb


def _supported() -> bool:
    import torch.utils.checkpoint as ck

    ignores = getattr(
        getattr(ck, "_CachingTorchDispatchMode", None), "ignore_compile_internals", None
    )
    if not (callable(ignores) and bool(ignores())):
        return False
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler
        get_cpp_compiler()
    except Exception:  # noqa: BLE001 -- no host C++ compiler: inductor cannot build CPU kernels here
        return False
    return True


class _Lora(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        rank = 4,
    ):
        super().__init__()
        self.base = torch.nn.Linear(dim_in, dim_out)
        self.base.requires_grad_(False)
        self.a = torch.nn.Parameter(torch.randn(rank, dim_in) * 0.1)
        self.b = torch.nn.Parameter(torch.randn(dim_out, rank) * 0.1)

    def forward(self, x):
        return self.base(x) + (x @ self.a.t()) @ self.b.t()


class _Block(torch.nn.Module):
    """No grad reaches q / k / v, so the attention output is a compiled-graph intermediate."""

    def __init__(
        self,
        dim = 64,
        heads = 4,
    ):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.out = torch.nn.Linear(dim, dim)
        self.qkv.requires_grad_(False)
        self.out.requires_grad_(False)
        self.up = _Lora(dim, 4 * dim)
        self.down = _Lora(4 * dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = (
            self.qkv(F.layer_norm(x, (d,)))
            .view(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        h = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(h.transpose(1, 2).reshape(b, n, d))
        return x + self.down(F.gelu(self.up(F.layer_norm(x, (d,))), approximate = "tanh"))


def _grads(checkpoint):
    torch.manual_seed(0)
    block = _Block()
    torch._dynamo.reset()
    block.compile(fullgraph = True, dynamic = True)
    torch.manual_seed(1)
    x = torch.randn(1, 1024, 64)
    target = torch.randn(1, 1024, 64)
    F.mse_loss(checkpoint(block, x), target).backward()
    return [p.grad.clone() for p in block.parameters() if p.requires_grad]


@pytest.mark.skipif(
    not _supported(), reason = "needs compile-safe selective checkpointing and a C++ compiler"
)
def test_saved_attention_around_a_compiled_block_keeps_gradients(monkeypatch):
    monkeypatch.delenv(gb.SAVE_ATTENTION_ENV, raising = False)
    saving = gb._Checkpoint(save_attention = True)
    assert saving.context_fn is not None
    plain = gb._Checkpoint(save_attention = False)
    kept = _grads(saving)
    assert saving.saved_attention == 1  # the 1024-key attention really was kept
    expected = _grads(plain)
    for got, want in zip(kept, expected):
        torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-6)
