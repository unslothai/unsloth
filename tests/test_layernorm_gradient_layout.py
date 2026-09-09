# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("layout", ["contiguous", "columns", "rows", "transposed", "expanded"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids = ["float32", "float16", "bfloat16"]
)
@pytest.mark.parametrize(
    "input_layout", ["contiguous", "columns", "rows", "transposed", "expanded"]
)
def test_layernorm_backward_gradient_layout(layout, dtype, input_layout):
    from unsloth.kernels.layernorm import fast_layernorm

    torch.manual_seed(42)
    shape = (2, 4, 80)
    if input_layout == "columns":
        inputs = torch.randn(2, 4, 160, device = "cuda", dtype = dtype)[..., ::2]
    elif input_layout == "rows":
        inputs = torch.randn(2, 8, 80, device = "cuda", dtype = dtype)[:, ::2]
    elif input_layout == "transposed":
        inputs = torch.randn(4, 2, 80, device = "cuda", dtype = dtype).transpose(0, 1)
    elif input_layout == "expanded":
        inputs = torch.randn(1, 1, 1, device = "cuda", dtype = dtype).expand(shape)
    else:
        inputs = torch.randn(shape, device = "cuda", dtype = dtype)
    layer = torch.nn.LayerNorm(shape[-1], device = "cuda", dtype = dtype)
    layer.requires_grad_(False)
    layer.weight.uniform_(0.5, 1.5)
    layer.bias.uniform_(-0.5, 0.5)
    if layout == "columns":
        grad = torch.randn(2, 4, 160, device = "cuda", dtype = dtype)[..., ::2]
    elif layout == "rows":
        grad = torch.randn(2, 8, 80, device = "cuda", dtype = dtype)[:, ::2]
    elif layout == "transposed":
        grad = torch.randn(4, 2, 80, device = "cuda", dtype = dtype).transpose(0, 1)
    elif layout == "expanded":
        grad = torch.ones(1, device = "cuda", dtype = dtype).expand(shape)
    else:
        grad = torch.randn(shape, device = "cuda", dtype = dtype)
    reference_inputs = inputs.clone().requires_grad_()
    expected = torch.nn.functional.layer_norm(
        reference_inputs.float(), (shape[-1],), layer.weight.float(), layer.bias.float(), layer.eps
    ).to(dtype)
    expected.backward(grad.clone())

    actual_inputs = inputs.detach().requires_grad_()
    actual = fast_layernorm(layer, actual_inputs)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol = 1e-2, atol = 1e-3)
    torch.testing.assert_close(actual_inputs.grad, reference_inputs.grad, rtol = 2e-2, atol = 1e-2)
