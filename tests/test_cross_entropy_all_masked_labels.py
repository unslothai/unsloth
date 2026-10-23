# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from real_accelerator import (
    has_real_cuda,
)  # tests/_shared, on sys.path via tests/conftest.py
import torch

pytestmark = pytest.mark.gpu


# A batch whose labels are all -100 (truncation cut every response away) used to return 0 / 0 = NaN.
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("n_items", [None, 0, "tensor"])
def test_fast_cross_entropy_all_masked_labels(n_items):
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    logits = torch.randn(2, 5, 256, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    labels = torch.full((2, 5), -100, device = "cuda")
    if n_items == "tensor":
        n_items = torch.tensor(0, device = "cuda")
    loss = fast_cross_entropy_loss(logits, labels, n_items = n_items)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(logits.grad).all() and not logits.grad.any()


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
def test_fast_cross_entropy_partly_masked_keeps_the_mean():
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

    torch.manual_seed(0)
    logits = torch.randn(1, 4, 256, device = "cuda")
    labels = torch.tensor([[3, -100, 7, -100]], device = "cuda")
    expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
    torch.testing.assert_close(
        fast_cross_entropy_loss(logits, labels), expected, rtol = 1e-5, atol = 1e-5
    )
