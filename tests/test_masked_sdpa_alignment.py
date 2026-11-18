# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Compiled masked training SDPA on torch < 2.14 pads the call to a multiple of 8: Inductor otherwise
hands the memory-efficient backward an unaligned bias for lengths that are not a multiple of 8
("attn_bias is not correctly aligned" or a misaligned address fault). The padded call is exact."""

import pytest
import torch
import torch.nn.functional as F
from packaging.version import Version
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

from unsloth.utils import attention_dispatch as ad


def _left_padded_mask(
    bsz,
    q_len,
    device,
    additive = False,
    dtype = torch.float32,
):
    pad = torch.tensor([0, 3, q_len // 2])[:bsz]
    keep = torch.arange(q_len)[None, :] >= pad[:, None]
    causal = torch.ones(q_len, q_len, dtype = torch.bool).tril()
    mask = causal[None, None] & keep[:, None, None, :]
    mask = (mask | ~mask.any(dim = -1, keepdim = True)).to(device)
    if additive:
        return torch.zeros(mask.shape, dtype = dtype, device = device).masked_fill(
            ~mask, torch.finfo(dtype).min
        )
    return mask


def _qkv(
    bsz,
    n_heads,
    n_kv,
    q_len,
    head_dim,
    device,
    dtype,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    mk = lambda h: torch.randn(bsz, h, q_len, head_dim, generator = g).to(device, dtype)
    return [t.requires_grad_() for t in (mk(n_heads), mk(n_kv), mk(n_kv))]


def _out_and_grads(fn, q, k, v, grad, **kw):
    for t in (q, k, v):
        t.grad = None
    out = fn(q, k, v, **kw)
    out.backward(grad)
    return out.detach(), [t.grad.detach().clone() for t in (q, k, v)]


def test_gate_matches_affected_torch_versions():
    affected = Version(torch.__version__.split("+")[0]) < Version("2.14.0")
    assert ad._MASKED_SDPA_NEEDS_ALIGNMENT is affected
    q = torch.zeros(1, 1, 4, 8)
    assert not ad._masked_sdpa_needs_alignment(q, q, torch.ones(1, 1, 4, 4, dtype = torch.bool))


@pytest.mark.parametrize("q_len", [5, 8, 9, 13, 16, 17])
@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("gqa", [False, True])
def test_padded_masked_call_is_exact(monkeypatch, q_len, additive, gqa):
    """CPU fp32: the 2..9 row pad (keys masked, queries sliced) leaves outputs and grads unchanged."""
    seen = []
    real = ad.scaled_dot_product_attention

    def record(Q, K, V, **kwargs):
        seen.append((Q.shape[-2], K.shape[-2], kwargs["attn_mask"].shape[-1]))
        return real(Q, K, V, **kwargs)

    kw = dict(enable_gqa = True) if gqa else {}
    q, k, v = _qkv(3, 4, 2 if gqa else 4, q_len, 16, "cpu", torch.float32, seed = q_len)
    mask = _left_padded_mask(3, q_len, "cpu", additive = additive)
    grad = torch.randn(3, 4, q_len, 16, generator = torch.Generator().manual_seed(1))
    ref, ref_g = _out_and_grads(F.scaled_dot_product_attention, q, k, v, grad, attn_mask = mask, **kw)

    monkeypatch.setattr(ad, "scaled_dot_product_attention", record)
    monkeypatch.setattr(ad, "_masked_sdpa_needs_alignment", lambda *a: True)
    got, got_g = _out_and_grads(ad._training_sdpa, q, k, v, grad, attn_mask = mask, **kw)

    padded = q_len + 2 + (-q_len - 2) % 8
    assert padded % 8 == 0 and 2 <= padded - q_len <= 9
    assert seen == [(padded, padded, padded)]
    assert got.shape == ref.shape
    for a, b in [(got, ref)] + list(zip(got_g, ref_g)):
        torch.testing.assert_close(a, b, atol = 1e-5, rtol = 1e-5)


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.parametrize("gqa", [False, True])
def test_compiled_masked_training_sdpa_backward_at_unaligned_lengths(gqa):
    """Compiled bf16 backward at lengths 9..17: no fault, matches eager, one dynamic graph."""
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    kw = dict(enable_gqa = True) if gqa else {}

    def f(q, k, v, m):
        return ad._training_sdpa(q, k, v, attn_mask = m, **kw)

    compiled = torch.compile(f, dynamic = True, fullgraph = True)
    for q_len in range(9, 18):
        q, k, v = _qkv(2, 4, 2 if gqa else 4, q_len, 128, "cuda", torch.bfloat16, seed = q_len)
        mask = _left_padded_mask(2, q_len, "cuda")
        grad = torch.randn(2, 4, q_len, 128, device = "cuda", dtype = torch.bfloat16)
        got, got_g = _out_and_grads(compiled, q, k, v, grad, m = mask)
        ref, ref_g = _out_and_grads(f, q, k, v, grad, m = mask)
        torch.cuda.synchronize()
        for a, b in [(got, ref)] + list(zip(got_g, ref_g)):
            torch.testing.assert_close(a.float(), b.float(), atol = 2e-2, rtol = 2e-2)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    assert counters["stats"]["unique_graphs"] <= 2, dict(counters["stats"])
