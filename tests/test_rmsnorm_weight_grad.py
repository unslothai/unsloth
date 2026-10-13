# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from real_accelerator import (
    has_real_cuda,
)  # tests/_shared, on sys.path via tests/conftest.py
import torch

pytestmark = pytest.mark.gpu


def _reference(
    inputs,
    weights,
    grad,
    gemma,
    eps = 1e-6,
):
    x = inputs.clone().requires_grad_()
    w = weights.clone().requires_grad_()
    normed = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim = True) + eps)
    if gemma:
        y = (normed * (w.float() + 1.0)).to(x.dtype)
    else:
        y = w * normed.to(x.dtype)
    y.backward(grad)
    return x.grad, w.grad


def _fast_eager(
    inputs,
    weights,
    grad,
    gemma,
    eps = 1e-6,
):
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    x = inputs.clone().requires_grad_()
    w = weights.clone().requires_grad_()
    Fast_RMS_Layernorm.apply(x, w, eps, gemma).backward(grad.clone())
    return x.grad, w.grad


def _fast_traced(
    inputs,
    weights,
    grad,
    gemma,
    eps = 1e-6,
):
    from unsloth.kernels import rms_layernorm

    if not rms_layernorm._TRACEABLE:
        pytest.skip("torch.library.triton_op unavailable")
    x = inputs.clone().requires_grad_()
    w = weights.clone().requires_grad_()
    y, _ = torch.ops.unsloth.rms_layernorm(x.reshape(-1, x.shape[-1]), w, eps, gemma)
    y.view(x.shape).backward(grad.clone())
    return x.grad, w.grad


# Issue #3014: a trainable norm weight (full finetuning without full_finetuning=True) got no grad.
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("path", ["eager", "traced"])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_trainable_weight_gets_grad(path, gemma, dtype):
    torch.manual_seed(3407)
    shape = (2, 37, 256)
    inputs = torch.randn(shape, device = "cuda", dtype = dtype)
    weights = torch.rand(shape[-1], device = "cuda", dtype = dtype)
    grad = torch.randn(shape, device = "cuda", dtype = dtype)
    ref_dx, ref_dw = _reference(inputs, weights, grad, gemma)
    fast = _fast_eager if path == "eager" else _fast_traced
    dx, dw = fast(inputs, weights, grad, gemma)
    assert dw is not None and dw.dtype == weights.dtype
    torch.testing.assert_close(dx, ref_dx, rtol = 2e-2, atol = 1e-2)
    cos = torch.nn.functional.cosine_similarity(
        dw.double().flatten(), ref_dw.double().flatten(), dim = 0
    )
    assert cos > 0.999, cos
    torch.testing.assert_close(dw.float(), ref_dw.float(), rtol = 2e-2, atol = 5e-2)


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
def test_rmsnorm_frozen_weight_stays_gradless():
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    x = torch.randn(4, 64, device = "cuda", requires_grad = True)
    w = torch.rand(64, device = "cuda")
    Fast_RMS_Layernorm.apply(x, w, 1e-6, False).sum().backward()
    assert x.grad is not None and w.grad is None


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("chunk_rows", [1, 7, 64, 1000])
def test_rmsnorm_weight_grad_chunked_matches_unchunked(chunk_rows):
    from unsloth.kernels.rms_layernorm import _rms_weight_grad

    torch.manual_seed(3407)
    dY = torch.randn(203, 96, device = "cuda", dtype = torch.bfloat16)
    X = torch.randn(203, 96, device = "cuda", dtype = torch.bfloat16)
    W = torch.rand(96, device = "cuda", dtype = torch.bfloat16)
    r = torch.rsqrt(X.float().square().mean(-1) + 1e-6)
    expected = _rms_weight_grad(dY, X, W, r)
    actual = _rms_weight_grad(dY, X, W, r, chunk_rows = chunk_rows)
    torch.testing.assert_close(actual, expected)
