# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

# Row counts are runtime arguments that Triton would otherwise specialize
# (== 1, divisible by 16): every new token count could compile another copy of the same kernel.

import pytest
import torch

from real_accelerator import has_real_cuda

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
triton = pytest.importorskip("triton")

import unsloth  # noqa: F401  (patches first)
from unsloth.kernels import rms_layernorm
from unsloth.kernels.rms_layernorm import fast_rms_layernorm

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
