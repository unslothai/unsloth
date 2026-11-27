# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Chunked vocabs (> 65536): in-place ops on the forward output zeroed compiled grads (torch 2.11)."""

import pytest
from real_accelerator import (
    accelerator_device_type,
    has_real_cuda_or_xpu,
)  # tests/_shared, on sys.path via tests/conftest.py
import torch

if not has_real_cuda_or_xpu() or torch.version.hip is not None:
    pytest.skip(
        "Triton cross entropy kernels under torch.compile need an NVIDIA CUDA or Intel XPU device",
        allow_module_level = True,
    )

from unsloth.kernels import cross_entropy_loss as ce_mod

DEVICE = accelerator_device_type()

_TORCH = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


@pytest.mark.parametrize("vocab", [32768, 65537, 151936])
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("dynamic", [None, True])
def test_compiled_grads_match_the_fp64_reference(vocab, fullgraph, dynamic):
    if dynamic and _TORCH < (2, 10):
        # torch 2.6 Dynamo also rejects bool() of the symbolic softcap int; checked from 2.10 on.
        pytest.skip("dynamic=True Triton cross entropy is checked on torch 2.10+")
    torch._dynamo.reset()
    generator = torch.Generator(device = DEVICE).manual_seed(vocab)
    # sm75 (T4) has no bf16 Triton under torch.compile; XPU has bf16.
    dtype = (
        torch.bfloat16
        if DEVICE != "cuda" or torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    logits = torch.randn(2, 5, vocab, device = DEVICE, generator = generator).to(dtype)
    labels = torch.randint(0, vocab, (2, 5), device = DEVICE, generator = generator)
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
