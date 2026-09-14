# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "vocab_size,softcap,offset",
    [
        (32000, 1.0, 0.0),
        (65537, 1.0, 0.0),
        (256000, 30.0, -100.0),
        (262208, 30.0, -100.0),
        (32768, 1.0, 0.0),
        (32000, 0.0, 0.0),
        (65537, 0.0, 0.0),
    ],
)
def test_cross_entropy_softcap_padding(dtype, vocab_size, softcap, offset):
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    torch.manual_seed(42)
    inputs = (torch.randn(1, 3, vocab_size, device = "cuda") + offset).to(dtype)
    labels = torch.tensor([[0, vocab_size - 1, -100]], device = "cuda")
    logits = inputs.clone().requires_grad_()
    reference_logits = inputs.clone().requires_grad_()
    transformed = reference_logits.float()
    if softcap:
        transformed = softcap * torch.tanh(transformed / softcap)
    expected = F.cross_entropy(transformed.flatten(0, 1), labels.flatten())
    expected.backward()

    actual = fast_cross_entropy_loss(logits, labels, logit_softcapping = softcap)
    actual.backward()

    torch.testing.assert_close(actual, expected, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol = 1e-2, atol = 1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("vocab_size", [32000, 65537, 256000])
@pytest.mark.parametrize("logit_scaling", [0.0625, 0.5, 2.0])
def test_cross_entropy_softcap_padding_with_logit_scaling(vocab_size, logit_scaling):
    """Cohere-style logit scaling runs before the softcap, so the mask has to survive both."""
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    softcap = 30.0
    torch.manual_seed(42)
    inputs = (torch.randn(1, 3, vocab_size, device = "cuda") - 100.0).float()
    labels = torch.tensor([[0, vocab_size - 1, -100]], device = "cuda")
    logits = inputs.clone().requires_grad_()
    reference_logits = inputs.clone().requires_grad_()
    transformed = logit_scaling * reference_logits.float()
    transformed = softcap * torch.tanh(transformed / softcap)
    expected = F.cross_entropy(transformed.flatten(0, 1), labels.flatten())
    expected.backward()

    actual = fast_cross_entropy_loss(
        logits,
        labels,
        logit_softcapping = softcap,
        logit_scaling = logit_scaling,
    )
    actual.backward()

    torch.testing.assert_close(actual, expected, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol = 1e-2, atol = 1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("vocab_size", [32000, 65537])
@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_negative_logit_scaling_does_not_nan(vocab_size, softcap):
    """A negative scale maps the -inf padding to +inf, which used to poison the row maximum."""
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    torch.manual_seed(42)
    inputs = torch.randn(1, 2, vocab_size, device = "cuda").float()
    labels = torch.tensor([[0, vocab_size - 1]], device = "cuda")
    logits = inputs.clone().requires_grad_()
    reference_logits = inputs.clone().requires_grad_()
    transformed = -1.0 * reference_logits.float()
    if softcap:
        transformed = softcap * torch.tanh(transformed / softcap)
    expected = F.cross_entropy(transformed.flatten(0, 1), labels.flatten())

    actual = fast_cross_entropy_loss(
        logits,
        labels,
        logit_softcapping = softcap,
        logit_scaling = -1.0,
    )

    assert torch.isfinite(actual), f"loss is {actual}"
    torch.testing.assert_close(actual, expected, rtol = 1e-5, atol = 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("vocab_size", [32000, 65537, 256000, 262208])
def test_softcapped_probabilities_sum_to_one(vocab_size):
    """Guards the denominator without leaning on a loss tolerance."""
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    softcap = 30.0
    torch.manual_seed(42)
    inputs = (torch.randn(1, 1, vocab_size, device = "cuda") - 100.0).float()
    labels = torch.tensor([[0]], device = "cuda")

    loss = fast_cross_entropy_loss(
        inputs.clone().requires_grad_(), labels, logit_softcapping = softcap
    )
    transformed = softcap * torch.tanh(inputs.double() / softcap)
    # A single supervised token means loss == logsumexp - transformed[label].
    logsumexp = loss.double() + transformed[0, 0, 0]
    mass = torch.exp(transformed - logsumexp).sum()

    torch.testing.assert_close(mass, torch.ones_like(mass), rtol = 0, atol = 1e-5)
