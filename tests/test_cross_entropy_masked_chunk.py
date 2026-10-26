# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Chunked CE forward: a fully -inf vocab chunk must not turn the loss or grads into NaN (CUDA and ROCm)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA or ROCm device", allow_module_level = True)

from unsloth.kernels import cross_entropy_loss as ce_mod
from unsloth.kernels.utils import MAX_FUSED_SIZE


@pytest.mark.parametrize("config", [(8192, 4), (16384, 8), (MAX_FUSED_SIZE, 32)])
@pytest.mark.parametrize("masked", [(16384, 32768), (65536, 131072)])
def test_an_all_masked_chunk_matches_the_fp64_reference(config, masked, monkeypatch):
    monkeypatch.setattr(ce_mod, "_chunked_forward_config", lambda device, vocab_size: config)
    vocab = 131073
    generator = torch.Generator(device = "cuda").manual_seed(0)
    logits = torch.randn(4, vocab, device = "cuda", generator = generator)
    logits[:, masked[0] : masked[1]] = -float("inf")
    labels = torch.tensor([0, 5, vocab - 1, -100], device = "cuda")
    reference = logits.double().requires_grad_(True)
    expected = torch.nn.functional.cross_entropy(reference, labels, reduction = "sum") / 3
    expected.backward()
    logits.requires_grad_(True)
    loss = ce_mod.fast_cross_entropy_loss(logits.view(1, 4, vocab), labels.view(1, 4))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    torch.testing.assert_close(loss.double(), expected.detach(), rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(logits.grad.double(), reference.grad, rtol = 1e-4, atol = 1e-6)
