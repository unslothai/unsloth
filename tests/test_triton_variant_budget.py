# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

# Row counts and sequence lengths are runtime arguments that Triton would otherwise specialize
# (== 1, divisible by 16): every new token count could compile another copy of the same kernel.

import pytest
import torch

from real_accelerator import has_real_cuda

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
triton = pytest.importorskip("triton")

import unsloth  # noqa: F401  (patches first)
from unsloth.kernels import qk_norm_rope, rms_layernorm, rope_embedding
from unsloth.kernels.rms_layernorm import fast_rms_layernorm
from unsloth.kernels.rope_embedding import fast_rope_embedding

# Varied lengths: odd, divisible by 16, a single token.
_SHAPES = ((1, 100), (1, 511), (1, 512), (2, 2048), (4, 333), (3, 1))


def _variants(kernel):
    while not isinstance(kernel, triton.runtime.jit.JITFunction):
        kernel = kernel.fn
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        return sum(len(v[0]) for v in caches.values())
    return sum(len(v) for v in kernel.cache.values())


class _Norm(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, device = "cuda", dtype = torch.bfloat16))
        self.variance_epsilon = 1e-6


def _reset(*kernels):
    for kernel in kernels:
        while not isinstance(kernel, triton.runtime.jit.JITFunction):
            kernel = kernel.fn
        caches = getattr(kernel, "device_caches", None)
        if caches is not None:
            for v in caches.values():
                v[0].clear()
        else:
            for v in kernel.cache.values():
                v.clear()
    from unsloth.kernels import triton_launch

    triton_launch._CACHE.clear()


def test_rms_layernorm_backward_one_variant_across_token_counts():
    kernels = (rms_layernorm._rms_layernorm_backward, rms_layernorm._rms_sum_dw_partials)
    _reset(*kernels)
    norm = _Norm(1024)
    for batch, seq in _SHAPES:
        X = torch.randn(batch, seq, 1024, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        fast_rms_layernorm(norm, X).sum().backward()
    torch.cuda.synchronize()
    assert [_variants(k) for k in kernels] == [1, 1]


def test_rope_one_variant_per_direction_across_lengths():
    kernel = rope_embedding._rope_embedding_QK
    _reset(kernel)
    cos = torch.randn(4096, 128, device = "cuda", dtype = torch.bfloat16)
    sin = torch.randn(4096, 128, device = "cuda", dtype = torch.bfloat16)
    for batch, seq in _SHAPES[:-1]:
        q = torch.randn(batch, 16, seq, 128, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        k = torch.randn(batch, 8, seq, 128, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        Q, K = fast_rope_embedding(q, k, cos[:seq], sin[:seq])
        (Q.float().sum() + K.float().sum()).backward()
    torch.cuda.synchronize()
    assert _variants(kernel) == 2  # forward + backward


def test_qk_norm_rope_one_variant_across_lengths():
    kernels = (qk_norm_rope._qk_norm_rope_forward, qk_norm_rope._qk_norm_rope_backward)
    _reset(*kernels)
    Wq = torch.randn(128, device = "cuda", dtype = torch.bfloat16)
    Wk = torch.randn(128, device = "cuda", dtype = torch.bfloat16)
    cos = torch.randn(4096, 128, device = "cuda", dtype = torch.bfloat16)
    sin = torch.randn(4096, 128, device = "cuda", dtype = torch.bfloat16)
    eager = lambda kernel: kernel
    for batch, seq in _SHAPES:
        Q = torch.randn(batch, seq, 16, 128, device = "cuda", dtype = torch.bfloat16)
        K = torch.randn(batch, seq, 8, 128, device = "cuda", dtype = torch.bfloat16)
        Qo, Ko, rq, rk = qk_norm_rope._launch_forward(
            Q, K, Wq, Wk, cos[:seq], sin[:seq], cos, 1e-6, 1e-6, False, eager
        )
        qk_norm_rope._launch_backward(
            Qo, Ko, Q, K, Wq, Wk, rq, rk, cos[:seq], sin[:seq], cos, False, eager
        )
    torch.cuda.synchronize()
    assert [_variants(k) for k in kernels] == [1, 1]
