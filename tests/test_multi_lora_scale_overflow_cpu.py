# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only projection regression; no model download or package startup required."""

from __future__ import annotations
import ast
import functools
import math
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
    tree = ast.parse(path.read_text(), filename = str(path))
    names = {"MultiLoraExpertParam", "MultiLoraExperts", "_expert_stack_shape"}
    tree.body = [
        n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    mod = types.ModuleType("_multi_lora_overflow_source")
    mod.__dict__.update(
        torch = torch,
        nn = nn,
        F = F,
        math = math,
        Optional = Optional,
        Iterable = Iterable,
        functools = functools,
        _peft_expert_in_axis = lambda transposed: 1 if transposed else 2,
    )
    sys.modules[mod.__name__] = mod
    exec(compile(tree, str(path), "exec"), mod.__dict__)
    return mod


def _grouped_mm(x, weight, offsets):
    bounds = [0, *offsets.tolist()]
    return torch.cat([x[start:end] @ w for w, start, end in zip(weight, bounds, bounds[1:])])


@pytest.mark.parametrize("reversed_io", [False, True])
@pytest.mark.parametrize("scale", [2.0, 16.0, 100000.0])
def test_grouped_projection_scales_output_without_fp16_factor_overflow(reversed_io, scale):
    ml = _load_experts()
    base = nn.Module()
    base.gate_up_proj = nn.Parameter(torch.zeros(1, 16, 8, dtype = torch.float16))
    base.down_proj = nn.Parameter(torch.zeros(1, 8, 8, dtype = torch.float16))
    experts = ml.MultiLoraExperts(base, ["gate_up_proj"], 2, r = 1, lora_alpha = [scale, 1.0])
    experts.reversed_io = reversed_io
    adapter = experts.adapters["gate_up_proj"]
    # A large but finite second factor, a small first factor and a finite final projection.
    # Reversed orientation also swaps the actual in/out dimensions.
    first_value, second_value = (
        (0.000001, 40000.0) if scale == 100000 else (0.01, 40000.0 if scale == 2 else 5000.0)
    )
    with torch.no_grad():
        for A, B in zip(adapter.lora_A, adapter.lora_B):
            A.fill_(second_value if reversed_io else first_value)
            B.fill_(first_value if reversed_io else second_value)
    in_dim, out_dim = (16, 8) if reversed_io else (8, 16)
    x = torch.full((3, in_dim), 0.01, dtype = torch.float16, requires_grad = True)
    offsets = torch.tensor([1, 2, 3], dtype = torch.int32)
    slots = torch.tensor([0, 1, 2])  # unequal scales, plus no-adapter row
    backend = types.SimpleNamespace(
        _get_base_weight = lambda *a: torch.zeros(1, in_dim, out_dim, dtype = torch.float16),
        preprocess_weight = lambda w, *a, **kw: w,
        _moe_recompute_enabled = lambda *a, **kw: False,
        _base_grouped_mm = lambda x, offs, provider, *a: _grouped_mm(x, provider(), offs),
        _grouped_mm_with_backward_fix = _grouped_mm,
    )
    args = (
        backend,
        "gate_up_proj",
        x,
        torch.tensor([3]),
        offsets,
        torch.tensor([3]),
        None,
        torch.float16,
    )
    # Before the fix _project has no slot argument; run that implementation too.
    if "pair_slots" in ml.MultiLoraExperts._project.__code__.co_varnames:
        got = experts._project(*args, slots)
    else:
        got = experts._project(*args)
    refs = []
    for u in range(2):
        a, b, scaling, _ = adapter.grouped_mm_factors(u, reversed_io)
        refs.append(((x[u : u + 1] @ a[0].half()) @ b[0].half()) * scaling)
    ref = torch.cat([*refs, x[2:3] @ torch.zeros(in_dim, out_dim, dtype = x.dtype)])
    grad = torch.full_like(got, 1e-6)
    leaves = [x, *adapter.lora_A, *adapter.lora_B]
    got_grads = torch.autograd.grad(got, leaves, grad, retain_graph = True)
    ref_grads = torch.autograd.grad(ref, leaves, grad)
    assert (
        got.isfinite().all()
    ), "finite LoRA projection overflowed while folding scaling into its factor"
    torch.testing.assert_close(got, ref, rtol = 0, atol = 0)
    for actual, expected in zip(got_grads, ref_grads):
        assert actual.isfinite().all()
        torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)
