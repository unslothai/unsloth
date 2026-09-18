# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [80, 65540])
@pytest.mark.parametrize(
    "layout",
    [
        "contiguous",
        "logits_columns",
        "logits_slice",
        "logits_transpose",
        "logits_expand",
        "labels_columns",
        "labels_transpose",
    ],
)
def test_cross_entropy_input_layout(layout, vocab_size, dtype):
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    torch.manual_seed(123)
    logits = torch.randn(2, 4, vocab_size, device = "cuda", dtype = dtype)
    labels = torch.randint(vocab_size, (2, 4), device = "cuda")
    if layout == "logits_columns":
        logits = torch.randn(2, 4, 2 * vocab_size, device = "cuda", dtype = dtype)[..., ::2]
    elif layout == "logits_slice":
        logits = torch.randn(2, 8, vocab_size, device = "cuda", dtype = dtype)[:, :4]
    elif layout == "logits_transpose":
        logits = torch.randn(4, 2, vocab_size, device = "cuda", dtype = dtype).transpose(0, 1)
    elif layout == "logits_expand":
        logits = torch.randn(1, 1, vocab_size, device = "cuda", dtype = dtype).expand(2, 4, vocab_size)
    elif layout == "labels_columns":
        labels = torch.randint(vocab_size, (2, 8), device = "cuda")[:, ::2]
    elif layout == "labels_transpose":
        labels = torch.randint(vocab_size, (4, 2), device = "cuda").t()
    labels[0, -1] = -100
    logits.requires_grad_(True)
    reference = logits.detach().clone().requires_grad_(True)
    expected = F.cross_entropy(reference.float().reshape(-1, vocab_size), labels.reshape(-1))
    expected.backward()
    actual = fast_cross_entropy_loss(logits, labels)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(logits.grad, reference.grad, rtol = 2e-3, atol = 1e-5)
