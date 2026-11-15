# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Stats counters must not recompile a compiled base layer on the small-row 16-bit fallback."""

from __future__ import annotations

import pytest
import torch
from torch._dynamo.utils import counters

from unsloth.kernels import fp8_lora as FL

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason = "needs torch float8 dtypes"
)


def _frozen_linear(bias):
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64, bias = bias, dtype = torch.bfloat16)
    return lin.requires_grad_(False)


@pytest.mark.parametrize("bias", [False, True])
def test_small_row_fallback_compiles_once_under_fullgraph(bias):
    torch._dynamo.reset()
    counters.clear()
    FL.reset_fp8_lora_stats()
    lin = _frozen_linear(bias)
    FL.attach_fp8_lora_base(lin, offload = False)
    compiled = torch.compile(lin, backend = "aot_eager", fullgraph = True)
    x = torch.randn(8, 64, dtype = torch.bfloat16, requires_grad = True)
    assert x.shape[0] < FL._MIN_ROWS
    ref = torch.nn.functional.linear(x, lin.weight, lin.bias)
    for _ in range(12):
        out = compiled(x)
        out.float().sum().backward()
        x.grad = None
    torch.testing.assert_close(out, ref)
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])
    assert not counters["graph_break"], dict(counters["graph_break"])

    FL.reset_fp8_lora_stats()
    for _ in range(3):
        lin(x)
    assert FL.FP8_LORA_STATS["fwd_16bit"] == 3, FL.FP8_LORA_STATS


def test_state_matmul_small_rows_compiles_once_under_fullgraph():
    torch._dynamo.reset()
    counters.clear()
    lin = _frozen_linear(False)
    state = FL.Fp8LoRABaseState(lin.weight)
    W = lin.weight

    def run(x, dy):
        return state.matmul(x, W), state.dgrad(dy, W)

    compiled = torch.compile(run, backend = "aot_eager", fullgraph = True)
    x, dy = torch.randn(4, 64).bfloat16(), torch.randn(4, 64).bfloat16()
    for _ in range(12):
        compiled(x, dy)
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])
