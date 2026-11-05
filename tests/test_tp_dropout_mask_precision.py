# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel dropout masks round as one process's F.dropout: each tp rank's slice of the output and of
the input gradient is bitwise the matching slice of F.dropout on the whole tensor under the same RNG state, for
bf16 / fp16 / fp32 activations. On CUDA F.dropout is ATen's fused native_dropout, which applies 1 / (1 - p) in
accumulator precision and rounds once; on CPU it multiplies by a noise tensor holding 1 / (1 - p) rounded to the
activation dtype. Ranks are emulated in one process: a mask depends only on the RNG state and the shape."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_accelerator import has_real_cuda  # noqa: E402
from test_tp_native_oracle import tpn  # noqa: E402  (tp_native without the GPU-only unsloth import)

TP = 2
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _assert_bitwise(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape, what
    bad = (_bits(a) != _bits(b)).sum().item()
    assert bad == 0, f"{what}: {bad} of {a.numel()} elements differ"


def _reference(x, g, p, seed):
    torch.manual_seed(seed)
    x = x.clone().requires_grad_()
    y = F.dropout(x, p, True)
    y.backward(g)
    return y.detach(), x.grad


def _one_rank(drop, x, g, rank, p, seed):
    width = x.shape[-1] // TP
    xr = x[..., rank * width : (rank + 1) * width].clone().requires_grad_()
    torch.manual_seed(seed)
    y = drop(xr, rank)
    y.backward(g[..., rank * width : (rank + 1) * width])
    return y.detach(), xr.grad


def _tp_dropout(p, width):
    def drop(xr, rank):
        state = SimpleNamespace(
            kind = "mlp",
            rank = rank,
            tp = TP,
            local_heads = None,
            local_width = width,
            row_bias = False,
            phase = 1,
            bypass = 0,
            path = "mlp",
        )
        return tpn._tp_dropout(state, xr, p, True)

    return drop


def _lora_dropout(p):
    def drop(xr, rank):
        dropout = nn.Dropout(p).train()
        proj = SimpleNamespace(
            lora_dropout = {"default": dropout},
            _unsloth_tp_style = tpn.ROWWISE,
            _unsloth_tp_coord = (rank, TP),
        )
        return tpn._lora_dropout(proj, "default", xr)

    return drop


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("p", [0.1, 0.15, 0.25, 0.3])
@pytest.mark.parametrize("path", ["tp_dropout", "row_lora_dropout"])
def test_tp_dropout_slices_round_as_one_process(device, dtype, p, path):
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(4, 16, 256, generator = gen) * 3).to(device, dtype)
    g = torch.randn(4, 16, 256, generator = gen).to(device, dtype)
    seed = 1234
    y_ref, dx_ref = _reference(x, g, p, seed)
    width = x.shape[-1] // TP
    drop = _tp_dropout(p, width) if path == "tp_dropout" else _lora_dropout(p)
    for rank in range(TP):
        y, dx = _one_rank(drop, x, g, rank, p, seed)
        cols = slice(rank * width, (rank + 1) * width)
        _assert_bitwise(y, y_ref[..., cols], f"{path} rank {rank} output")
        _assert_bitwise(dx, dx_ref[..., cols], f"{path} rank {rank} input grad")


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA native_dropout rounding")
def test_bf16_p025_example_rounds_in_accumulator_precision():
    # 1.015625 / 0.75 in fp32 rounds to 1.3515625 in bf16; with 1 / 0.75 first rounded to bf16 it is 1.359375.
    x = torch.full((1, 64), 1.015625, dtype = torch.bfloat16, device = "cuda")
    mask = torch.ones(1, 64, dtype = torch.bool, device = "cuda")
    y = tpn._apply_mask(x, mask, 0.25)
    assert y.unique().tolist() == [1.3515625]
