# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "columns", "rows", "expanded"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_backward_gradient_layout(gemma, layout, dtype):
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    torch.manual_seed(42)
    shape = (1, 4, 64)
    inputs = torch.randn(shape, device = "cuda", dtype = dtype)
    weights = torch.rand(shape[-1], device = "cuda", dtype = dtype)
    if layout == "columns":
        grad = torch.randn(1, 4, 128, device = "cuda", dtype = dtype)[..., ::2]
    elif layout == "rows":
        grad = torch.randn(1, 8, 64, device = "cuda", dtype = dtype)[:, ::2]
    elif layout == "expanded":
        grad = torch.ones(1, device = "cuda", dtype = dtype).expand(shape)
    else:
        grad = torch.randn(shape, device = "cuda", dtype = dtype)
    assert grad.is_contiguous() == (layout == "contiguous")
    reference_inputs = inputs.clone().requires_grad_()
    normalized = reference_inputs.float() * torch.rsqrt(
        reference_inputs.float().square().mean(-1, keepdim = True) + 1e-6
    )
    if gemma:
        expected = (normalized * (weights.float() + 1.0)).to(dtype)
    else:
        expected = normalized.to(dtype) * weights
    expected.backward(grad.clone())

    actual_inputs = inputs.clone().requires_grad_()
    actual = Fast_RMS_Layernorm.apply(actual_inputs, weights, 1e-6, gemma)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol = 1e-2, atol = 1e-3)
    torch.testing.assert_close(actual_inputs.grad, reference_inputs.grad, rtol = 2e-2, atol = 1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_forward_column_strided_input(gemma, dtype):
    """`x[..., ::2]` reshapes to a stride-2 view, not a copy, so the forward reads it wrong too."""
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    torch.manual_seed(42)
    source = torch.randn(1, 4, 128, device = "cuda", dtype = dtype, requires_grad = True)
    strided = source[..., ::2]
    assert not strided.is_contiguous()
    weights = torch.rand(strided.shape[-1], device = "cuda", dtype = dtype)

    reference_source = source.detach().clone().requires_grad_()
    reference_inputs = reference_source[..., ::2]
    normalized = reference_inputs.float() * torch.rsqrt(
        reference_inputs.float().square().mean(-1, keepdim = True) + 1e-6
    )
    if gemma:
        expected = (normalized * (weights.float() + 1.0)).to(dtype)
    else:
        expected = normalized.to(dtype) * weights
    expected.backward(torch.ones_like(expected))

    actual = Fast_RMS_Layernorm.apply(strided, weights, 1e-6, gemma)
    actual.backward(torch.ones_like(actual))
    torch.testing.assert_close(actual, expected, rtol = 1e-2, atol = 1e-3)
    torch.testing.assert_close(source.grad, reference_source.grad, rtol = 2e-2, atol = 1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "columns", "expanded"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_weight_layout(gemma, layout, dtype):
    """The kernels read the norm weight with unit stride, so it needs materializing as well."""
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    torch.manual_seed(42)
    shape = (1, 4, 64)
    dim = shape[-1]
    inputs = torch.randn(shape, device = "cuda", dtype = dtype)
    if layout == "columns":
        weights = torch.rand(2 * dim, device = "cuda", dtype = dtype)[::2]
    elif layout == "expanded":
        weights = torch.rand(1, device = "cuda", dtype = dtype).expand(dim)
    else:
        weights = torch.rand(dim, device = "cuda", dtype = dtype)
    assert weights.is_contiguous() == (layout == "contiguous")
    grad = torch.randn(shape, device = "cuda", dtype = dtype)

    reference_inputs = inputs.clone().requires_grad_()
    normalized = reference_inputs.float() * torch.rsqrt(
        reference_inputs.float().square().mean(-1, keepdim = True) + 1e-6
    )
    if gemma:
        expected = (normalized * (weights.float() + 1.0)).to(dtype)
    else:
        expected = normalized.to(dtype) * weights
    expected.backward(grad.clone())

    actual_inputs = inputs.clone().requires_grad_()
    actual = Fast_RMS_Layernorm.apply(actual_inputs, weights, 1e-6, gemma)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol = 1e-2, atol = 1e-3)
    torch.testing.assert_close(actual_inputs.grad, reference_inputs.grad, rtol = 2e-2, atol = 1e-2)
