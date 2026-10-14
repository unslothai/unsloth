# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from real_accelerator import (
    has_real_cuda,
)
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required"),
]


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


def _fast(
    path,
    inputs,
    weights,
    grad,
    gemma,
    eps = 1e-6,
):
    from unsloth.kernels import rms_layernorm

    x = inputs.clone().requires_grad_()
    w = weights.clone().requires_grad_()
    if path == "eager":
        y = rms_layernorm.Fast_RMS_Layernorm.apply(x, w, eps, gemma)
    else:
        if not rms_layernorm._TRACEABLE:
            pytest.skip("torch.library.triton_op unavailable")
        y, _ = torch.ops.unsloth.rms_layernorm(x.reshape(-1, x.shape[-1]), w, eps, gemma)
    y.view(x.shape).backward(grad.clone())
    return x.grad, w.grad


# Issue #3014: a trainable norm weight (full finetuning without full_finetuning=True) got no grad.
@pytest.mark.parametrize("path", ["eager", "traced"])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,cols", [(3, 128), (9000, 128), (3, 384), (9000, 384)])
def test_rmsnorm_trainable_weight_gets_grad(path, gemma, dtype, rows, cols):
    torch.manual_seed(3407)
    inputs = torch.randn(rows, cols, device = "cuda", dtype = dtype)
    weights = torch.rand(cols, device = "cuda", dtype = dtype)
    grad = torch.randn(rows, cols, device = "cuda", dtype = dtype)
    ref_dx, ref_dw = _reference(inputs, weights, grad, gemma)
    dx, dw = _fast(path, inputs, weights, grad, gemma)
    assert dw is not None and dw.dtype == weights.dtype and dw.shape == weights.shape
    torch.testing.assert_close(dx, ref_dx, rtol = 2e-2, atol = 1e-2)
    exact = (
        ref_dw.double()
        if dtype == torch.float32
        else _reference(inputs.float(), weights.float(), grad.float(), gemma)[1].double()
    )
    cos = torch.nn.functional.cosine_similarity(dw.double(), exact, dim = 0)
    assert cos > 0.9999, cos
    torch.testing.assert_close(dw.double(), exact, rtol = 1e-2, atol = 1e-2 * exact.abs().max().item())


@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("cols", [128, 384])
def test_rmsnorm_trainable_weight_dx_matches_frozen(gemma, cols):
    """The trainable row loop may contract differently (Gemma's (W + 1), and plain rows on Blackwell): at most
    a bf16 ulp apart."""
    torch.manual_seed(3407)
    inputs = torch.randn(777, cols, device = "cuda", dtype = torch.bfloat16)
    weights = torch.rand(cols, device = "cuda", dtype = torch.bfloat16)
    grad = torch.randn(777, cols, device = "cuda", dtype = torch.bfloat16)
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    frozen = inputs.clone().requires_grad_()
    Fast_RMS_Layernorm.apply(frozen, weights, 1e-6, gemma).backward(grad.clone())
    dx, _ = _fast("eager", inputs, weights, grad, gemma)
    torch.testing.assert_close(dx.float(), frozen.grad.float(), rtol = 2**-7, atol = 1e-6)


@pytest.mark.parametrize("gemma", [False, True])
def test_rmsnorm_weight_grad_eager_matches_traced_bytes(gemma):
    torch.manual_seed(3407)
    inputs = torch.randn(5000, 256, device = "cuda", dtype = torch.bfloat16)
    weights = torch.rand(256, device = "cuda", dtype = torch.bfloat16)
    grad = torch.randn(5000, 256, device = "cuda", dtype = torch.bfloat16)
    _, eager = _fast("eager", inputs, weights, grad, gemma)
    _, traced = _fast("traced", inputs, weights, grad, gemma)
    assert torch.equal(eager.view(torch.int16), traced.view(torch.int16))


def test_rmsnorm_frozen_weight_stays_gradless():
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    x = torch.randn(4, 64, device = "cuda", requires_grad = True)
    w = torch.rand(64, device = "cuda")
    Fast_RMS_Layernorm.apply(x, w, 1e-6, False).sum().backward()
    assert x.grad is not None and w.grad is None


@pytest.mark.parametrize("cols", [128, 384])
def test_rmsnorm_trainable_weight_empty_input(cols):
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    x = torch.randn(0, cols, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    w = torch.rand(cols, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    Fast_RMS_Layernorm.apply(x, w, 1e-6, False).sum().backward()
    assert x.grad.shape == (0, cols)
    assert torch.equal(w.grad, torch.zeros_like(w))
