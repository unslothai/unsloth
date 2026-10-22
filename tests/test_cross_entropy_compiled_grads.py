# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""torch.compile of fast_cross_entropy_loss must give the eager gradients for chunked vocabs
(> 65536): mutating the forward output in place after the kernel made compiled grads zero."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("needs an NVIDIA CUDA device", allow_module_level = True)

from unsloth.kernels import cross_entropy_loss as ce_mod


@pytest.mark.parametrize("vocab", [32768, 65537, 151936])
@pytest.mark.parametrize("fullgraph", [False, True])
def test_compiled_grads_match_the_fp64_reference(vocab, fullgraph):
    torch._dynamo.reset()
    generator = torch.Generator(device = "cuda").manual_seed(vocab)
    logits = torch.randn(2, 5, vocab, device = "cuda", generator = generator).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (2, 5), device = "cuda", generator = generator)
    labels[0, 1] = -100
    reference = logits.double().requires_grad_(True)
    n_items = (labels != -100).sum()
    expected = torch.nn.functional.cross_entropy(
        reference.flatten(0, 1), labels.flatten(), reduction = "sum"
    ) / n_items
    expected.backward()
    x = logits.clone().requires_grad_(True)
    loss = torch.compile(ce_mod.fast_cross_entropy_loss, fullgraph = fullgraph)(x, labels)
    loss.backward()
    torch.testing.assert_close(loss.double(), expected.detach(), rtol = 1e-4, atol = 1e-4)
    rel = (x.grad.double() - reference.grad).norm() / reference.grad.norm()
    assert rel < 1e-2, rel
