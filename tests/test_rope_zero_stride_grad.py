# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""RoPE backward must be exact for expanded (stride-0) grads, e.g. from .sum() (#3781)."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "Triton RoPE kernel needs CUDA"),
]


def _reference(x, cos, sin, idx):
    c, s = (cos[idx].unsqueeze(1), sin[idx].unsqueeze(1)) if idx is not None else (cos, sin)
    x1, x2 = x.chunk(2, -1)
    return x * c + torch.cat((-x2, x1), -1) * s


@pytest.mark.parametrize("use_indices", [False, True])
@pytest.mark.parametrize("grad_kind", ["sum", "expanded_row", "dense"])
def test_rope_backward_matches_reference(use_indices, grad_kind):
    from unsloth.kernels.rope_embedding import fast_rope_embedding

    torch.manual_seed(0)
    b, h, s, d = 2, 4, 16, 64
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device = "cuda").float() / d))
    freqs = torch.outer(torch.arange(s, device = "cuda").float(), inv)
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos(), emb.sin()
    idx = torch.arange(s, device = "cuda").expand(b, s).contiguous() if use_indices else None

    x = torch.randn(b, h, s, d, device = "cuda", requires_grad = True)
    x_ref = x.detach().clone().requires_grad_(True)
    q, k = fast_rope_embedding(x.clone(), x.clone(), cos, sin, idx)
    rq, rk = _reference(x_ref, cos, sin, idx), _reference(x_ref, cos, sin, idx)

    if grad_kind == "sum":
        (rq.sum() + rk.sum()).backward()
        (q.sum() + k.sum()).backward()
    else:
        shape = (1, 1, 1, d) if grad_kind == "expanded_row" else (b, h, s, d)
        g = torch.randn(shape, device = "cuda").expand(b, h, s, d)
        torch.autograd.backward([rq, rk], [g, g])
        # The kernel rotates contiguous grads in place, so give it its own copies.
        torch.autograd.backward([q, k], [g.clone(), g.clone()])

    torch.testing.assert_close(q, rq, atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(x.grad, x_ref.grad, atol = 1e-4, rtol = 1e-4)


@pytest.mark.parametrize("implementation", ["single", "indexed_qk", "slow"])
def test_rope_backward_preserves_shared_dense_gradient(implementation):
    from unsloth.kernels.rope_embedding import (
        Fast_RoPE_Embedding,
        Fast_RoPE_Embedding_QK,
        Slow_RoPE_Embedding,
    )

    torch.manual_seed(1)
    b, h, s, d = 2, 4, 16, 64
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device = "cuda").float() / d))
    freqs = torch.outer(torch.arange(s, device = "cuda").float(), inv)
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos(), emb.sin()
    x = torch.randn(b, h, s, d, device = "cuda", requires_grad = True)
    z = torch.randn_like(x, requires_grad = True)

    if implementation == "single":
        q = Fast_RoPE_Embedding.apply(x.transpose(1, 2).contiguous(), cos, sin)
        k = Fast_RoPE_Embedding.apply(z.transpose(1, 2).contiguous(), cos, sin)
    elif implementation == "indexed_qk":
        indices = torch.arange(s, device = "cuda").expand(b, s).contiguous()
        q, k = Fast_RoPE_Embedding_QK.apply(x, z, cos, sin, indices)
    else:
        q = Slow_RoPE_Embedding.apply(x, cos[None, None], sin[None, None], None)
        k = Slow_RoPE_Embedding.apply(z, cos[None, None], sin[None, None], None)

    grad = torch.randn_like(q)
    grad_before = grad.clone()
    (q + k).backward(grad)
    assert torch.equal(grad, grad_before)


@pytest.mark.parametrize("implementation", ["single", "indexed_qk"])
def test_rope_rows_past_2_pow_31_elements(implementation):
    from unsloth.kernels.rope_embedding import Fast_RoPE_Embedding, Fast_RoPE_Embedding_QK

    b, h, s, d = 513, 8, 4096, 128  # b * h * s * d > 2^31
    need = b * (h + 1) * s * d * 2 + (1 << 30)
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(reason = f"needs {need / 2**30:.1f} GiB free GPU memory")
    torch.manual_seed(0)
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device = "cuda").float() / d))
    freqs = torch.outer(torch.arange(s, device = "cuda").float(), inv)
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos(), emb.sin()

    with torch.no_grad():
        if implementation == "single":
            Q = torch.randn(b, s, h, d, device = "cuda", dtype = torch.float16)
            ref = _reference(Q[-1, -4:].transpose(0, 1).float(), cos[-4:], sin[-4:], None)
            Fast_RoPE_Embedding.apply(Q, cos, sin)
            out = Q[-1, -4:].transpose(0, 1)
        else:
            Q = torch.randn(b, h, s, d, device = "cuda", dtype = torch.float16)
            K = torch.zeros(b, 1, s, d, device = "cuda", dtype = torch.float16)
            idx = torch.arange(s, device = "cuda", dtype = torch.int32).expand(b, s)
            ref = _reference(Q[-1, :, -4:].float(), cos[-4:], sin[-4:], None)
            Fast_RoPE_Embedding_QK.apply(Q, K, cos, sin, idx)
            out = Q[-1, :, -4:]
    torch.testing.assert_close(out, ref.half())
