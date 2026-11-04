# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The FP8 LoRA base forward must follow autocast like F.linear."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from unsloth.kernels import fp8_lora as FL


def _cpu_scaled_mm_works():
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        return False
    try:
        a = torch.zeros(16, 16).to(torch.float8_e4m3fn)
        one = torch.ones((), dtype = torch.float32)
        torch._scaled_mm(a, a.t(), scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
        return True
    except (NotImplementedError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _cpu_scaled_mm_works(), reason = "this torch build has no CPU torch._scaled_mm kernel"
)

FP8_FWD_TOL = 0.06


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _fp8_linear(bias = False, fill = None):
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64, bias = bias, dtype = torch.bfloat16).requires_grad_(False)
    if fill is not None:
        with torch.no_grad():
            lin.weight.fill_(fill)
    weight = lin.weight.detach().clone()
    bias_ref = None if lin.bias is None else lin.bias.detach().clone()
    FL.attach_fp8_lora_base(lin, offload = False)
    return lin, weight, bias_ref


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_fp8_base_follows_bf16_autocast(input_dtype, bias, compiled):
    torch._dynamo.reset()
    lin, weight, bias_ref = _fp8_linear(bias)
    call = torch.compile(lin, backend = "aot_eager", fullgraph = True) if compiled else lin
    x = torch.randn(32, 64, dtype = input_dtype, requires_grad = True)
    assert x.shape[0] >= FL._MIN_ROWS
    FL.reset_fp8_lora_stats()
    with torch.autocast("cpu", dtype = torch.bfloat16):
        ref = F.linear(x, weight, bias_ref)
        got = call(x)
    assert FL.FP8_LORA_STATS["quantize"] == 0
    assert ref.dtype == torch.bfloat16
    assert got.dtype == ref.dtype
    assert _rel(got, ref) < FP8_FWD_TOL
    got.float().sum().backward()
    assert x.grad.dtype == input_dtype


@pytest.mark.parametrize("compiled", [False, True])
def test_fp16_input_under_bf16_autocast_does_not_overflow(compiled):
    torch._dynamo.reset()
    lin, weight, _ = _fp8_linear(fill = 128)
    call = torch.compile(lin, backend = "aot_eager", fullgraph = True) if compiled else lin
    x = torch.full((32, 64), 16, dtype = torch.float16)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        ref = F.linear(x, weight)
        got = call(x)
    assert ref[0, 0].item() == 131072.0
    assert got.dtype == torch.bfloat16 and torch.isfinite(got).all()
    torch.testing.assert_close(got, ref)
