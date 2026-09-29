# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""fast_rope_embedding backward must be exact when the incoming grad is expanded (#3781).

`(q.sum() + k.sum()).backward()` hands the kernel an all-zero-stride grad; the in-place
Triton kernel then rotated one shared element instead of every position.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "Triton RoPE kernel needs CUDA"
)


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
