# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frozen W8A8 int8 LoRA base (prototype, UNSLOTH_DIFFUSION_INT8_TRAIN=w8a8): numerics against the dense layer,
quantize-once, streamed conversion, and the trainer gates."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_frozen_int8 as fi8


def _cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim = 0).item()


@pytest.mark.parametrize("rows", [40, 8])
def test_forward_and_input_grad_track_the_dense_layer(rows):
    torch.manual_seed(0)
    lin = nn.Linear(64, 96).to(torch.bfloat16).requires_grad_(False)
    m = fi8.FrozenInt8Linear(lin)
    x = torch.randn(2, rows, 64, dtype = torch.bfloat16, requires_grad = True)
    m(x).float().pow(2).sum().backward()
    ref_x = x.detach().clone().requires_grad_(True)
    F.linear(ref_x.float(), lin.weight.float(), lin.bias.float()).pow(2).sum().backward()
    assert _cos(m(x), F.linear(x.float(), lin.weight.float(), lin.bias.float())) > 0.9995
    assert _cos(x.grad, ref_x.grad) > 0.999
    assert x.grad.dtype == torch.bfloat16
    # Frozen: no parameter of the int8 layer takes a gradient, and the weight is int8 codes + fp32 row scales.
    assert m.weight.dtype == torch.int8 and m.weight_scale.dtype == torch.float32
    assert all(not p.requires_grad for p in m.parameters())


def test_weight_is_quantized_once_not_per_call(monkeypatch):
    lin = nn.Linear(32, 32, bias = False).to(torch.bfloat16)
    m = fi8.FrozenInt8Linear(lin)
    calls = []
    monkeypatch.setattr(fi8, "quantize_weight", lambda *a, **k: calls.append(1))
    x = torch.randn(24, 32, dtype = torch.bfloat16, requires_grad = True)
    for _ in range(3):
        m(x).sum().backward()
    assert calls == []


def test_rowwise_codes_round_trip():
    w = torch.randn(48, 64)
    q, s = fi8.quantize_weight(w, rows = 16)
    assert q.abs().max() <= 127
    assert (q.float() * s - w).abs().max() <= s.max() * 0.5 + 1e-6


def test_convert_streams_and_skips_unaligned():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)
            self.b = nn.Linear(30, 32)
            self.to(torch.bfloat16).requires_grad_(False)

    m = _M()
    done, failed = fi8.convert_frozen_int8(m, {"a", "b"}, device = "meta")
    assert (done, failed) == (1, 1)
    assert isinstance(m.a, fi8.FrozenInt8Linear) and m.a.weight.device.type == "meta"
    # An unaligned layer keeps its bf16 weight (already moved; the final move would take it anyway).
    assert type(m.b) is nn.Linear and m.b.weight.dtype == torch.bfloat16


def test_compiles_without_graph_breaks():
    torch._dynamo.reset()
    lin = nn.Linear(64, 64).to(torch.bfloat16).requires_grad_(False)
    m = fi8.FrozenInt8Linear(lin)
    from torch._dynamo.utils import counters

    counters.clear()
    f = torch.compile(m, backend = "aot_eager", fullgraph = True)
    x = torch.randn(32, 64, dtype = torch.bfloat16, requires_grad = True)
    f(x).float().sum().backward()
    assert x.grad is not None and sum(counters["graph_break"].values()) == 0


def test_trainer_gates(monkeypatch):
    cfg = type("C", (), {"compile_transformer": "auto", "train_steps": 10})()
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
    assert dit._should_compile(cfg, False, "cuda", "int8") is False
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "w8a8")
    assert dit._int8_w8a8() is True
    assert dit._should_compile(cfg, False, "cuda", "int8") is True
    cfg.compile_transformer = "off"
    assert dit._should_compile(cfg, False, "cuda", "int8") is False


def test_apply_lands_the_base_and_reports(monkeypatch):
    class _B(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(32, 32, bias = False)
            self.norm = nn.LayerNorm(32)
            self.to(torch.bfloat16).requires_grad_(False)

    model = nn.Module()
    model.blocks = nn.ModuleList([_B(), _B()])
    events = []
    assert dit._apply_int8_w8a8(model, events.append, None, device = "meta") is True
    assert all(p.device.type == "meta" for p in model.parameters())
    assert isinstance(model.blocks[0].to_q, fi8.FrozenInt8Linear)
    assert "W8A8" in events[-1]["message"]
