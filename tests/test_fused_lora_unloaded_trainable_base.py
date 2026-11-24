# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An unloaded (plain nn.Linear) projection with a trainable weight and frozen X must skip the fused
Functions, whose backward crashed on absent LoRA weights."""

from __future__ import annotations

import os
import sys

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels import fast_lora

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fused_lora_dropout_bias import H, PATHS, _peft_block  # noqa: E402

TRAINABLE = {"qkv": "k_proj", "o": "o_proj"}


def _grads(fn, block, X, name):
    block.zero_grad(set_to_none = True)
    out = fn(X)
    outs = out if isinstance(out, tuple) else (out,)
    gen = torch.Generator().manual_seed(11)
    sum((o * torch.randn(o.shape, generator = gen, dtype = o.dtype)).sum() for o in outs).backward()
    return [o.detach() for o in outs], block.get_submodule(name).weight.grad.clone()


@pytest.mark.parametrize("path", ["qkv", "o"])
@pytest.mark.parametrize("unload", ["merge_and_unload", "unload"])
def test_unloaded_projection_with_trainable_weight_gets_its_gradient(path, unload):
    model, _ = _peft_block("cpu", torch.float64, 0.0)
    block = getattr(model, unload)()
    name = TRAINABLE[path]
    assert type(block.get_submodule(name)) is torch.nn.Linear
    block.get_submodule(name).weight.requires_grad_(True)
    X = torch.randn(2, 5, H, dtype = torch.float64)  # frozen input
    got, got_grad = _grads(PATHS[path][0](block), block, X, name)
    want, want_grad = _grads(PATHS[path][1](block), block, X, name)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol = 0, atol = 0)
    torch.testing.assert_close(got_grad, want_grad)


def test_unloaded_projection_routing():
    model, _ = _peft_block("cpu", torch.float64, 0.0)
    block = model.merge_and_unload()
    X = torch.randn(2, 5, H, dtype = torch.float64)
    projs = (block.q_proj, block.k_proj, block.v_proj)
    assert fast_lora._lora_dropout_and_bias(projs, X) is not None
    block.k_proj.weight.requires_grad_(True)
    assert fast_lora._base_weight_requires_grad(block.k_proj)
    with torch.no_grad():
        assert fast_lora._lora_dropout_and_bias(projs, X) is not None
    assert fast_lora._lora_dropout_and_bias(projs, X) is None
