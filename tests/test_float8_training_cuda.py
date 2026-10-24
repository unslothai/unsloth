# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""float8_training on a real fp8 GPU: token counts not divisible by 16 train in eager and compiled mode,
and an offloaded (CPU) embedding does not block conversion of the CUDA linears."""

from __future__ import annotations

import importlib.util

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("torchao") is None, reason = "require torchao"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
        reason = "require an fp8 GPU",
    ),
]

from unsloth.models import float8_training as f8t  # noqa: E402


def _config(recipe):
    from torchao.float8 import Float8LinearConfig
    return Float8LinearConfig.from_recipe_name(recipe)


def _mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(256, 512, bias = False),
        torch.nn.GELU(),
        torch.nn.Linear(512, 256, bias = False),
    ).to("cuda", torch.bfloat16)


@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tokens", [1, 7, 17, 33])
def test_unaligned_token_counts_backward(recipe, compiled, tokens):
    if recipe == "tensorwise" and torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("tensorwise is refused on 8.9")
    model = _mlp()
    names = f8t.convert_model_to_float8_training(model, _config(recipe), size_filter = False)
    assert len(names) == 2
    fn = torch.compile(model) if compiled else model
    x = torch.randn(1, tokens, 256, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    fn(x).float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_offloaded_embedding_does_not_block_conversion():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(
                64, 256
            )  # stays on the CPU, like offload_embedding = True
            self.body = _mlp()

    model = M()
    assert next(model.parameters()).device.type == "cpu"
    names = f8t.convert_model_to_float8_training(model, _config("rowwise"), size_filter = False)
    assert names == ["body.0", "body.2"]
