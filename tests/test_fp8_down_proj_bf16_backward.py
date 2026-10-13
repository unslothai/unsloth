# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""LoRA_MLP.backward must take down_proj's grad through the dequantized FP8 weight in bf16, like gate / up.

Runs on CPU: the triton dequant is replaced by its torch twin and fp8_linear by a recording stand-in.
"""

import types

import pytest
import torch

from unsloth.kernels import fast_lora, fp8
from unsloth.kernels import utils as kernel_utils

BLOCK = 128
HIDDEN, INTER, RANK, TOKENS = 256, 384, 4, 8


def _dequant(W, scale):
    # Handles the transposed views LoRA_MLP.backward passes (W.t() of the stored layout).
    if W.stride(0) == 1 and W.stride(1) != 1:
        return _dequant(W.t(), scale).t()
    return fp8._torch_blockwise_dequant(W, scale, [BLOCK, BLOCK], torch.float32)


def _fp8_weight(out_f, in_f, seed):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(out_f, in_f, generator = g) * 0.5).to(torch.float8_e4m3fn)
    return W, torch.rand(out_f // BLOCK, in_f // BLOCK, generator = g) + 0.5


@pytest.fixture
def fp8_calls(monkeypatch):
    calls = []

    def fake_fp8_linear(
        X,
        W,
        W_quant,
        bias = None,
    ):
        calls.append(tuple(W.shape))
        return X @ _dequant(W, W_quant).t()

    monkeypatch.setattr(kernel_utils, "fp8_linear", fake_fp8_linear)
    monkeypatch.setattr(
        fast_lora,
        "fast_dequantize",
        lambda W, q, *a, **k: W if q is None else _dequant(W, q),
    )
    return calls


def _backward(downW, downW_quant):
    g = torch.Generator().manual_seed(1)
    rnd = lambda *s: torch.randn(*s, generator = g) * 0.1
    gateW, gate_q = _fp8_weight(INTER, HIDDEN, 2)
    upW, up_q = _fp8_weight(INTER, HIDDEN, 3)
    gateA, upA, downA = rnd(RANK, HIDDEN), rnd(RANK, HIDDEN), rnd(RANK, INTER)
    gateB, upB, downB = rnd(INTER, RANK), rnd(INTER, RANK), rnd(HIDDEN, RANK)
    X, e, gg, dY = (
        rnd(1, TOKENS, HIDDEN),
        rnd(TOKENS, INTER),
        rnd(TOKENS, INTER),
        rnd(1, TOKENS, HIDDEN),
    )
    seen = {}

    def backward_function(DW, e, g):
        seen["DW"] = DW.clone()
        return DW, e, g

    ctx = types.SimpleNamespace(
        custom_saved_tensors = (
            gateW,
            gate_q,
            1.0,
            upW,
            up_q,
            1.0,
            downW,
            downW_quant,
            2.0,
            backward_function,
        ),
        saved_tensors = (gateA, gateB, upA, upB, downA, downB, X, e, gg),
        inplace = False,
        _fwd_used_autocast = False,
        _dtype = torch.float32,
    )
    fast_lora.LoRA_MLP.backward(ctx, dY)
    dY2 = dY.view(-1, HIDDEN)
    return seen["DW"], dY2, downA, downB


def test_fp8_down_proj_backward_is_dequant_matmul(fp8_calls):
    downW, down_q = _fp8_weight(HIDDEN, INTER, 4)
    DW, dY, downA, downB = _backward(downW, down_q)
    assert fp8_calls == []
    expected = dY @ _dequant(downW, down_q) + 2.0 * (dY @ downB) @ downA
    torch.testing.assert_close(DW, expected, rtol = 1.6e-2, atol = 1e-3)


def test_bf16_down_proj_backward_unchanged(fp8_calls):
    downW = torch.randn(HIDDEN, INTER, generator = torch.Generator().manual_seed(5))
    DW, dY, downA, downB = _backward(downW, None)
    assert fp8_calls == []
    old = kernel_utils.matmul_lora(dY, downW.t(), None, downB.t(), downA.t(), 2.0)
    assert torch.equal(DW, old)
