# SPDX-License-Identifier: AGPL-3.0-only
"""Cold fullgraph regression with static tiny experts and isolated backend probes."""

from __future__ import annotations
import ast
import functools
import logging
import math
import os
from pathlib import Path
import sys
import types
from typing import Iterable, Optional

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def _load_experts():
    path = Path(__file__).resolve().parents[1] / "unsloth/kernels/multi_lora.py"
    names = {
        "MultiLoraBatch",
        "MultiLoraExpertParam",
        "MultiLoraExperts",
        "_expert_stack_shape",
        "_adapter_groups",
        "_token_adapters",
        "_grouped_disabled",
        "_warn_once",
        "_is_quantized",
    }
    tree = ast.parse(path.read_text(), filename = str(path))
    tree.body = [
        n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    mod = types.ModuleType("_multi_lora_compile_source")
    from dataclasses import dataclass

    mod.__dict__.update(
        torch = torch,
        nn = nn,
        F = F,
        math = math,
        Optional = Optional,
        Iterable = Iterable,
        functools = functools,
        dataclass = dataclass,
        logging = logging,
        os = os,
        _HALF = (torch.float16, torch.bfloat16),
        _FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64),
        _WARNED = set(),
        _peft_expert_in_axis = lambda transposed: 1 if transposed else 2,
        _zoo_stash_read_store = lambda base: None,
        MULTI_LORA_MOE_STATS = {"per_adapter_calls": 0, "grouped_fallbacks": 0, "grouped_calls": 0},
    )
    sys.modules[mod.__name__] = mod
    exec(compile(tree, str(path), "exec"), mod.__dict__)
    return mod


class _StaticExperts(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.full((1, 16, 8), 0.03, dtype = dtype))
        self.down_proj = nn.Parameter(torch.full((1, 8, 8), 0.02, dtype = dtype))

    def forward(self, hidden, index, weights):
        gate, up = (hidden @ self.gate_up_proj[0].t()).chunk(2, -1)
        return ((F.silu(gate) * up) @ self.down_proj[0].t()) * weights.sum(-1, keepdim = True)


@torch.compiler.disable
def _disabled_support_probe(self, hidden, args):
    return None


def _parity_grouped(self, hidden, index, weights, ids):
    y = torch.zeros_like(hidden)
    for u in range(self.num_adapters):
        y = y + torch.where((ids == u).unsqueeze(-1), self._run(u, (hidden, index, weights)), 0)
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("probe", ["disabled_support", "data_dependent_parity"])
def test_cold_fullgraph_does_not_run_grouped_probes(monkeypatch, dtype, probe):
    ml = _load_experts()
    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    monkeypatch.setattr(
        ml.MultiLoraExperts,
        "_grouped_unsupported",
        _disabled_support_probe if probe == "disabled_support" else lambda *a: None,
    )
    monkeypatch.setattr(ml.MultiLoraExperts, "_grouped", _parity_grouped)
    experts = ml.MultiLoraExperts(
        _StaticExperts(dtype), ["gate_up_proj", "down_proj"], 2, r = 1, lora_alpha = 16
    )
    with torch.no_grad():
        for adapter in experts.adapters.values():
            for A, B in zip(adapter.lora_A, adapter.lora_B):
                A.fill_(0.01)
                B.fill_(0.02)
    batch = ml.MultiLoraBatch.from_adapter_ids(torch.tensor([0, 1, 0, 1]), 2)
    for adapter in experts.adapters.values():
        adapter.batch = batch
    ml._adapter_groups(batch)  # batch preparation is intentionally eager in both arms
    hidden = torch.full((4, 8), 0.1, dtype = dtype, requires_grad = True)
    index = torch.zeros(4, 1, dtype = torch.long)
    weights = torch.ones(4, 1, dtype = dtype, requires_grad = True)
    assert experts.mode is None  # no eager module call before compilation
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(capture_scalar_outputs = True):
            out = torch.compile(experts, backend = "eager", fullgraph = True, dynamic = True)(
                hidden, index, weights
            )
        leaves = [hidden, weights, *experts.adapters.parameters()]
        gradients = torch.autograd.grad(out.sum(), leaves)
        ref = experts._per_adapter(hidden, (index, weights), ml._adapter_groups(batch))
        ref_gradients = torch.autograd.grad(ref.sum(), leaves)
        torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
        for actual, expected in zip(gradients, ref_gradients):
            assert actual.isfinite().all()
            torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)
    finally:
        torch._dynamo.reset()
