# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# The incoming grad of a norm backward can also belong to another node (y + s hands one
# tensor to both branches) or to the caller (autograd.grad(..., grad_outputs)), so the
# backward must not write dX over it.

import pytest
from real_accelerator import has_real_cuda
import torch


def _rms_fp64(x, W, gemma):
    x = x.double()
    W = W.double() + (1.0 if gemma else 0.0)
    return x * torch.rsqrt(x.square().mean(-1, keepdim = True) + 1e-6) * W


def _ln_fp64(x, W, b):
    x, W, b = x.double(), W.double(), b.double()
    return torch.nn.functional.layer_norm(x, x.shape[-1:], W, b, 1e-6)


def _graphs(norm, shape, device, dtype):
    # Each returns (dX from the Unsloth norm, dX from the fp64 reference, grads to check unmodified).
    def user_grad(x, go, reference):
        out = torch.autograd.grad(norm(x), x, go)[0]
        x64 = x.detach().double().requires_grad_()
        return out, torch.autograd.grad(reference(x64), x64, go.double())[0]

    def add_self(x, go, reference):
        out = torch.autograd.grad(norm(x) + x, x, go)[0]
        x64 = x.detach().double().requires_grad_()
        return out, torch.autograd.grad(reference(x64) + x64, x64, go.double())[0]

    def two_norms_sum(x, go, reference):
        out = torch.autograd.grad(norm(x) + norm(2 * x), x, go)[0]
        x64 = x.detach().double().requires_grad_()
        return out, torch.autograd.grad(reference(x64) + reference(2 * x64), x64, go.double())[0]

    return {"user_grad": user_grad, "add_self": add_self, "two_norms_sum": two_norms_sum}


_TOLERANCE = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 1e-1}


def _check(norm, reference, dtype):
    torch.manual_seed(0)
    shape = (3, 5, 256)
    for name, graph in _graphs(norm, shape, "cuda", dtype).items():
        x = torch.randn(shape, device = "cuda", dtype = dtype).requires_grad_()
        go = torch.randn(shape, device = "cuda", dtype = dtype)
        go_before = go.clone()
        dX, dX64 = graph(x, go, reference)
        assert torch.equal(go, go_before), f"{name}: incoming grad was overwritten"
        error = (dX.double() - dX64).abs().max().item()
        assert error <= _TOLERANCE[dtype], f"{name}: max error vs fp64 {error}"


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_layernorm_backward_leaves_shared_grad_alone(gemma, dtype):
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    W = (1.0 + 0.1 * torch.randn(256, device = "cuda")).to(dtype)
    norm = lambda x: Fast_RMS_Layernorm.apply(x, W, 1e-6, gemma)
    _check(norm, lambda x: _rms_fp64(x, W, gemma), dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layernorm_backward_leaves_shared_grad_alone(dtype):
    from unsloth.kernels.layernorm import Fast_Layernorm

    W = (1.0 + 0.1 * torch.randn(256, device = "cuda")).to(dtype)
    b = (0.1 * torch.randn(256, device = "cuda")).to(dtype)
    norm = lambda x: Fast_Layernorm.apply(x, W, b, 1e-6)
    _check(norm, lambda x: _ln_fp64(x, W, b), dtype)
