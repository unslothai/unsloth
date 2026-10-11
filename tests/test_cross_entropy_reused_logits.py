# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from real_accelerator import has_real_cuda
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required"),
]


def _reference_loss(logits, labels, softcap, scaling):
    transformed = logits.float()
    if scaling:
        transformed = scaling * transformed
    if softcap:
        transformed = softcap * torch.tanh(transformed / softcap)
    return F.cross_entropy(transformed.flatten(0, 1), labels.flatten())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [37, 65537])
@pytest.mark.parametrize("softcap,scaling", [(0.0, 0.0), (2.0, 0.0), (0.0, 0.5), (2.0, 0.5)])
def test_two_cross_entropy_losses_share_logits(dtype, vocab_size, softcap, scaling):
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    torch.manual_seed(42)
    hidden = torch.randn(2, 3, 8, device = "cuda")
    weight = torch.randn(vocab_size, 8, device = "cuda", requires_grad = True)
    reference_weight = weight.detach().clone().requires_grad_()
    logits = F.linear(hidden, weight).to(dtype)
    reference_logits = F.linear(hidden, reference_weight).to(dtype)
    original_logits = logits.detach().clone()
    labels_a = torch.tensor([[0, 1, -100], [2, vocab_size - 1, 3]], device = "cuda")
    labels_b = torch.tensor([[vocab_size - 1, -100, 2], [3, 1, 0]], device = "cuda")
    assert logits.is_contiguous()

    actual = fast_cross_entropy_loss(
        logits, labels_a, logit_softcapping = softcap, logit_scaling = scaling
    ) + 0.7 * fast_cross_entropy_loss(
        logits, labels_b, logit_softcapping = softcap, logit_scaling = scaling
    )
    expected = _reference_loss(
        reference_logits, labels_a, softcap, scaling
    ) + 0.7 * _reference_loss(reference_logits, labels_b, softcap, scaling)
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(
        weight.grad,
        reference_weight.grad,
        rtol = 1e-4 if dtype == torch.float32 else 2e-2,
        atol = 2e-6 if dtype == torch.float32 else 2e-3,
    )
    torch.testing.assert_close(logits, original_logits, rtol = 0, atol = 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [37, 65537])
@pytest.mark.parametrize("softcap,scaling", [(0.0, 0.0), (2.0, 0.5)])
@pytest.mark.parametrize("use_autograd_grad", [False, True], ids = ["backward", "autograd_grad"])
def test_cross_entropy_retained_graph(dtype, vocab_size, softcap, scaling, use_autograd_grad):
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    torch.manual_seed(42)
    logits = torch.randn(2, 3, vocab_size, device = "cuda", dtype = dtype, requires_grad = True)
    original_logits = logits.detach().clone()
    reference_logits = logits.detach().clone().requires_grad_()
    labels = torch.tensor([[0, 1, -100], [2, vocab_size - 1, 3]], device = "cuda")
    actual = fast_cross_entropy_loss(
        logits, labels, logit_softcapping = softcap, logit_scaling = scaling
    )
    expected = _reference_loss(reference_logits, labels, softcap, scaling)
    (reference_grad,) = torch.autograd.grad(expected, reference_logits)

    gradients = []
    for multiplier in (1.0, 0.3):
        upstream_grad = torch.full_like(actual, multiplier)
        if use_autograd_grad:
            (gradient,) = torch.autograd.grad(actual, logits, upstream_grad, retain_graph = True)
        else:
            logits.grad = None
            actual.backward(upstream_grad, retain_graph = True)
            gradient = logits.grad
        gradients.append(gradient)
        torch.testing.assert_close(
            gradient,
            multiplier * reference_grad,
            rtol = 1e-4 if dtype == torch.float32 else 2e-2,
            atol = 1e-7 if dtype == torch.float32 else 2e-3,
        )

    # A later backward must also leave previously returned gradients intact.
    torch.testing.assert_close(
        gradients[0],
        reference_grad,
        rtol = 1e-4 if dtype == torch.float32 else 2e-2,
        atol = 1e-7 if dtype == torch.float32 else 2e-3,
    )
    torch.testing.assert_close(actual, expected, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(logits, original_logits, rtol = 0, atol = 0)
