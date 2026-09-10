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


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("source_layout", ["columns", "rows", "transposed", "expanded"])
def test_layernorm_gradient_reaches_a_live_strided_source(source_layout):
    # The matrix above detaches its strided view into a leaf, so it never checks that
    # the gradient is scattered back through the slice/transpose/expand that produced it.
    from unsloth.kernels.layernorm import fast_layernorm

    torch.manual_seed(42)
    if source_layout == "columns":
        source = torch.randn(2, 4, 160, device = "cuda")
        view = lambda tensor: tensor[..., ::2]
    elif source_layout == "rows":
        source = torch.randn(2, 8, 80, device = "cuda")
        view = lambda tensor: tensor[:, ::2]
    elif source_layout == "transposed":
        source = torch.randn(4, 2, 80, device = "cuda")
        view = lambda tensor: tensor.transpose(0, 1)
    else:
        source = torch.randn(1, 1, 80, device = "cuda")
        view = lambda tensor: tensor.expand(2, 4, 80)
    layer = torch.nn.LayerNorm(80, device = "cuda")
    layer.requires_grad_(False)
    layer.weight.uniform_(0.5, 1.5)
    layer.bias.uniform_(-0.5, 0.5)

    grad = torch.randn(2, 4, 80, device = "cuda")
    # Snapshot up front: backward writes its result into the buffer it is handed.
    reference_grad = grad.clone()

    actual_source = source.detach().requires_grad_()
    fast_layernorm(layer, view(actual_source)).backward(grad)

    reference_source = source.detach().requires_grad_()
    torch.nn.functional.layer_norm(
        view(reference_source), (80,), layer.weight, layer.bias, layer.eps
    ).backward(reference_grad)

    assert actual_source.grad.shape == source.shape
    torch.testing.assert_close(actual_source.grad, reference_source.grad, rtol = 2e-2, atol = 1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("layout", ["columns", "rows", "transposed", "expanded"])
@pytest.mark.parametrize("no_grad", [torch.no_grad, torch.inference_mode], ids = ["no_grad", "inference_mode"])
def test_layernorm_forward_layout_without_autograd(layout, no_grad):
    # Inference never enters backward, so the forward's own layout handling needs
    # its own guard rather than riding along on the gradient assertions.
    from unsloth.kernels.layernorm import fast_layernorm

    torch.manual_seed(42)
    layer = torch.nn.LayerNorm(80, device = "cuda")
    layer.requires_grad_(False)
    layer.weight.uniform_(0.5, 1.5)
    layer.bias.uniform_(-0.5, 0.5)

    with no_grad():
        if layout == "columns":
            inputs = torch.randn(2, 4, 160, device = "cuda")[..., ::2]
        elif layout == "rows":
            inputs = torch.randn(2, 8, 80, device = "cuda")[:, ::2]
        elif layout == "transposed":
            inputs = torch.randn(4, 2, 80, device = "cuda").transpose(0, 1)
        else:
            inputs = torch.randn(1, 1, 80, device = "cuda").expand(2, 4, 80)
        actual = fast_layernorm(layer, inputs)
        expected = torch.nn.functional.layer_norm(
            inputs, (80,), layer.weight, layer.bias, layer.eps
        )
    torch.testing.assert_close(actual, expected, rtol = 1e-2, atol = 1e-3)
