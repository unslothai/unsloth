# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("kernel", ["qkv", "swiglu", "geglu", "geglu_approx"])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "columns"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
def test_lora_backward_gradient_layout(kernel, layout, dtype, inplace):
    from unsloth.kernels.fast_lora import LoRA_MLP, LoRA_QKV
    from unsloth.kernels.geglu import (
        geglu_approx_backward_kernel,
        geglu_approx_forward_kernel,
        geglu_exact_backward_kernel,
        geglu_exact_forward_kernel,
    )
    from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel

    torch.manual_seed(42)
    inputs = torch.randn(2, 4, 64, device = "cuda", dtype = dtype)
    widths = [(64, 64)] * 3 if kernel == "qkv" else [(64, 96), (64, 96), (96, 64)]
    weights = []
    adapters = []
    scales = [0.25, 0.5, 0.75]
    for in_features, out_features in widths:
        weights.append(torch.randn(out_features, in_features, device = "cuda", dtype = dtype) * 0.05)
        adapters.extend(
            [
                (torch.randn(8, in_features, device = "cuda", dtype = dtype) * 0.05).requires_grad_(),
                (torch.randn(out_features, 8, device = "cuda", dtype = dtype) * 0.05).requires_grad_(),
            ]
        )
    reference_adapters = [value.detach().clone().requires_grad_() for value in adapters]
    reference_inputs = inputs.clone().requires_grad_()

    def project(value, index):
        a, b = reference_adapters[2 * index : 2 * index + 2]
        return F.linear(value, weights[index]) + F.linear(F.linear(value, a), b) * scales[index]

    if kernel == "qkv":
        expected = tuple(project(reference_inputs, i) for i in range(3))
    else:
        gate, up = project(reference_inputs, 0), project(reference_inputs, 1)
        if kernel == "swiglu":
            activated = F.silu(gate.float()).to(dtype)
            forward, backward = swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
        elif kernel == "geglu":
            activated = F.gelu(gate.float()).to(dtype)
            forward, backward = geglu_exact_forward_kernel, geglu_exact_backward_kernel
        else:
            activated = F.gelu(gate.float(), approximate = "tanh").to(dtype)
            forward, backward = geglu_approx_forward_kernel, geglu_approx_backward_kernel
        expected = (project(activated * up, 2),)

    gradients = []
    for output in expected:
        if layout == "transposed":
            gradient = torch.randn(4, 2, output.shape[-1], device = "cuda", dtype = dtype).transpose(
                0, 1
            )
        elif layout == "columns":
            gradient = torch.randn(2, 4, output.shape[-1] * 2, device = "cuda", dtype = dtype)[..., ::2]
        else:
            gradient = torch.randn_like(output)
        gradients.append(gradient)
    torch.autograd.backward(expected, [gradient.clone() for gradient in gradients])

    actual_inputs = inputs.clone().requires_grad_()
    arguments = []
    for i in range(3):
        arguments.extend([weights[i], None, *adapters[2 * i : 2 * i + 2], scales[i]])
    if kernel == "qkv":
        actual = LoRA_QKV.apply(actual_inputs, *arguments, inplace)
    else:
        actual = (LoRA_MLP.apply(actual_inputs, *arguments, forward, backward, inplace),)
    torch.autograd.backward(actual, gradients)

    tolerance = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (1e-2, 1e-3),
        torch.bfloat16: (5e-2, 1e-2),
    }
    rtol, atol = tolerance[dtype]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol = rtol, atol = atol)
    for result, reference in zip(
        [actual_inputs, *adapters], [reference_inputs, *reference_adapters]
    ):
        torch.testing.assert_close(result.grad, reference.grad, rtol = rtol, atol = atol)
