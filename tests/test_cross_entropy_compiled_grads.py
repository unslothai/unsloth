# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Chunked vocabs (> 65536): in-place ops on the forward output zeroed compiled grads (torch 2.11)."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip(
        "Triton cross entropy kernels under torch.compile need an NVIDIA CUDA device",
        allow_module_level = True,
    )

from unsloth.kernels import cross_entropy_loss as ce_mod

_TORCH = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


@pytest.mark.parametrize("vocab", [32768, 65537, 151936])
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("dynamic", [None, True])
def test_compiled_grads_match_the_fp64_reference(vocab, fullgraph, dynamic):
    if dynamic and _TORCH < (2, 10):
        # torch 2.6 Dynamo also rejects bool() of the symbolic softcap int; checked from 2.10 on.
        pytest.skip("dynamic=True Triton cross entropy is checked on torch 2.10+")
    torch._dynamo.reset()
    generator = torch.Generator(device = "cuda").manual_seed(vocab)
    # sm75 (T4) has no bf16 Triton under torch.compile.
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logits = torch.randn(2, 5, vocab, device = "cuda", generator = generator).to(dtype)
    labels = torch.randint(0, vocab, (2, 5), device = "cuda", generator = generator)
    labels[0, 1] = -100
    reference = logits.double().requires_grad_(True)
    n_items = (labels != -100).sum()
    expected = (
        torch.nn.functional.cross_entropy(
            reference.flatten(0, 1), labels.flatten(), reduction = "sum"
        )
        / n_items
    )
    expected.backward()
    x = logits.clone().requires_grad_(True)
    # dynamic=True: the vocab size is symbolic, so the host-side chunk counts must trace (no divmod).
    compiled = torch.compile(ce_mod.fast_cross_entropy_loss, fullgraph = fullgraph, dynamic = dynamic)
    loss = compiled(x, labels)
    loss.backward()
    torch.testing.assert_close(loss.double(), expected.detach(), rtol = 1e-4, atol = 1e-4)
    rel = (x.grad.double() - reference.grad).norm() / reference.grad.norm()
    assert rel < 1e-2, rel
