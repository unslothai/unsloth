# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Once the label logit leads the logsumexp of the others by ~16.6 nats, softmax(label) rounds to
1 in fp32: logsumexp - x and softmax - 1 both become exactly 0 while the other classes keep their
gradients, so the loss reads 0 and the gradient no longer sums to zero (arXiv 2610.31642)."""

import math

import pytest
from real_accelerator import (
    has_real_cuda,
)  # tests/_shared, on sys.path via tests/conftest.py
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu


def _reference(
    logits,
    labels,
    softcap = 0.0,
):
    """fp64 loss and gradient written as softplus(logsumexp(others) - x)."""
    z = logits.double().clone().requires_grad_()
    t = softcap * torch.tanh(z / softcap) if softcap else z
    target = t.gather(-1, labels.clamp(min = 0)[..., None])
    others = (t - target).masked_fill(
        F.one_hot(labels.clamp(min = 0), t.shape[-1]).bool(), float("-inf")
    )
    loss = F.softplus(torch.logsumexp(others, -1)) * (labels != -100)
    loss.sum().backward()
    return loss.detach(), z.grad


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("vocab_size", [32000, 65537, 262144])
@pytest.mark.parametrize("margin", [5.0, 10.0, 17.0, 20.0, 40.0, 80.0])
def test_confident_label_keeps_loss_and_gradient(vocab_size, margin):
    from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss

    # margin = x - logsumexp(others) with the vocab_size - 1 others at 0
    inputs = torch.zeros(3, vocab_size, device = "cuda")
    labels = torch.tensor([3, vocab_size - 1, -100], device = "cuda")
    inputs[0, 3] = inputs[1, vocab_size - 1] = margin + math.log(vocab_size - 1)
    logits = inputs.clone().requires_grad_()

    loss = Fast_CrossEntropyLoss.apply(logits, labels, 0, 0)
    loss.sum().backward()
    expected_loss, expected_grad = _reference(inputs, labels)

    rows = torch.arange(2, device = "cuda")
    torch.testing.assert_close(loss.double(), expected_loss, rtol = 1e-5, atol = 0)
    torch.testing.assert_close(
        logits.grad[rows, labels[:2]].double(), expected_grad[rows, labels[:2]], rtol = 1e-5, atol = 0
    )
    # Used to be +exp(-margin): the label's share was rounded away.
    assert logits.grad[:2].double().sum(-1).abs().max() <= 1e-4 * math.exp(-margin)


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("vocab_size", [32000, 65537, 262144])
@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_random_logits_match_fp64(vocab_size, softcap):
    from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss

    torch.manual_seed(0)
    inputs = torch.randn(4, vocab_size, device = "cuda") * 4
    labels = torch.tensor([0, 17, -100, vocab_size - 1], device = "cuda")
    logits = inputs.clone().requires_grad_()

    loss = Fast_CrossEntropyLoss.apply(logits, labels, softcap, 0)
    loss.sum().backward()
    expected_loss, expected_grad = _reference(inputs, labels, softcap)

    torch.testing.assert_close(loss.double(), expected_loss, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(logits.grad.double(), expected_grad, rtol = 1e-4, atol = 1e-9)


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
def test_confident_bf16_label():
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    inputs = torch.zeros(1, 1, 32000, device = "cuda", dtype = torch.bfloat16)
    inputs[0, 0, 0] = 32.0  # ~21.6 nats over the others
    labels = torch.tensor([[0]], device = "cuda")
    logits = inputs.clone().requires_grad_()

    loss = fast_cross_entropy_loss(logits, labels)
    loss.backward()
    expected_loss, expected_grad = _reference(inputs[0].float(), labels[0])

    torch.testing.assert_close(loss.double(), expected_loss[0], rtol = 1e-5, atol = 0)
    torch.testing.assert_close(
        logits.grad[0, 0, 0].double(), expected_grad[0, 0], rtol = 1e-2, atol = 0
    )


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("vocab_size, label_logit, other", [(2, 0.0, 103.0), (262144, -91.0, 0.0)])
def test_hopeless_label_gradient_stays_at_minus_one(vocab_size, label_logit, other):
    from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss

    # Loss ~100: exp(-loss) is subnormal, the label gradient must be -1 and never past it.
    inputs = torch.full((1, vocab_size), other, device = "cuda")
    inputs[0, 0] = label_logit
    labels = torch.tensor([0], device = "cuda")
    logits = inputs.clone().requires_grad_()

    Fast_CrossEntropyLoss.apply(logits, labels, 0, 0).sum().backward()
    _, expected_grad = _reference(inputs, labels)
    torch.testing.assert_close(logits.grad[0, 0].double(), expected_grad[0, 0], rtol = 1e-6, atol = 0)
    assert logits.grad[0, 0].item() >= -1.0
