# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Compare fused and chunked softcapped CE with an unpadded PyTorch reference."""

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
