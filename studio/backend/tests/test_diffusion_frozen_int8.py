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


def test_fp16_preflight_allows_int8_only_with_the_w8a8_gate(monkeypatch):
    import core.training.diffusion_train_common as tc

    monkeypatch.setattr(tc, "bf16_unsupported_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_accelerator_missing_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_fp16_fallback", lambda fam: True)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
    assert "fp16" in (tc.training_precision_preflight_error("flux.2-klein", "int8") or "")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "w8a8")
    reason = tc.training_precision_preflight_error("flux.2-klein", "int8")
    assert reason is None or "fp16" not in reason
    assert "fp16" in (tc.training_precision_preflight_error("flux.2-klein", "bf16") or "")


@pytest.mark.parametrize("cap, refused", [((7, 0), True), ((7, 5), False), ((8, 9), False)])
def test_fp16_preflight_refuses_w8a8_without_int8_tensor_cores(monkeypatch, cap, refused):
    """sm70 has no int8 GEMM: the trainer's probe would raise after residents were evicted, so the preflight refuses."""
    import core.training.diffusion_train_common as tc

    monkeypatch.setattr(tc, "bf16_unsupported_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_accelerator_missing_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_fp16_fallback", lambda fam: True)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "w8a8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cap)
    reason = tc.training_precision_preflight_error("flux.2-klein", "int8")
    assert ("7.5+" in (reason or "")) is refused


def test_conversion_keeps_the_fp32_island_on_an_untargeted_layer():
    """fp16 path: islands are installed before the W8A8 conversion and live on the module's forward. A converted
    Z-Image feed_forward.w2 (not a LoRA target) must still run fp32 with autocast off, not fp16 / bf16."""
    from core.training.diffusion_dit_fp16 import install_fp32_islands

    class _FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.w2 = nn.Linear(64, 32)

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(32, 32)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.feed_forward = _FF()
            self.attention = _Attn()

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    torch.manual_seed(0)
    model = _M().to(torch.bfloat16).requires_grad_(False)
    islands = install_fp32_islands(model, "z-image", "cpu")
    assert islands == ["layers.0.feed_forward.w2"]
    done, failed = fi8.convert_frozen_int8(
        model, {"layers.0.feed_forward.w2", "layers.0.attention.to_q"}
    )
    assert (done, failed) == (2, 0)
    w2, q = model.layers[0].feed_forward.w2, model.layers[0].attention.to_q
    assert isinstance(w2, fi8.FrozenInt8Linear) and isinstance(q, fi8.FrozenInt8Linear)
    assert getattr(w2, "_unsloth_fp32_island", False)
    assert not getattr(q, "_unsloth_fp32_island", False)
    x64 = torch.randn(24, 64, dtype = torch.bfloat16, requires_grad = True)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        y = w2(x64)
        yq = q(torch.randn(24, 32, dtype = torch.bfloat16))
    assert y.dtype == torch.float32 and yq.dtype == torch.bfloat16
    y.sum().backward()
    assert x64.grad is not None and torch.isfinite(x64.grad).all()


@pytest.mark.parametrize("fp16, refused", [(True, False), (False, True)])
def test_preflight_needs_torchao_for_int8_only_where_the_trainer_does(monkeypatch, fp16, refused):
    """The fp16 path's W8A8 base runs torch._int_mm and raises instead of falling back to torchao, so a missing
    torchao must not refuse it; the bf16 path keeps torchao weight-only int8 as its fallback and still needs it."""
    import core.training.diffusion_train_common as tc

    monkeypatch.setattr(tc, "bf16_unsupported_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_accelerator_missing_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_fp16_fallback", lambda fam: fp16)
    monkeypatch.setattr(tc, "has_functional_torchao", lambda: False)
    monkeypatch.setattr(tc, "torch_is_rocm", lambda: False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "w8a8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    reason = tc.training_precision_preflight_error("flux.2-klein", "int8")
    assert ("torchao" in (reason or "")) is refused
    if not refused:
        assert reason is None


def test_a_partial_w8a8_conversion_does_not_fall_back_to_a_mixed_base(monkeypatch):
    """A device move that fails after some layers converted must not hand the half-W8A8 base to the torchao
    weight-only fallback (the trainer would compile, cache and record it as weight-only int8)."""

    class _B(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(32, 32, bias = False)
            self.to_k = nn.Linear(32, 32, bias = False)
            self.to(torch.bfloat16).requires_grad_(False)

    model = nn.Module()
    model.blocks = nn.ModuleList([_B()])
    moves = []
    real_to = nn.Module.to

    def to(self, *a, **k):
        if type(self) is nn.Linear:
            moves.append(1)
            if len(moves) == 2:
                raise torch.OutOfMemoryError("injected OOM on the second move")
        return real_to(self, *a, **k)

    monkeypatch.setattr(nn.Module, "to", to)
    events = []
    with pytest.raises(ValueError, match = "partway|part of the base"):
        dit._apply_int8_w8a8(model, events.append, None, device = "meta")
    kinds = {type(m) for m in model.blocks[0].children()}
    assert kinds == {
        fi8.FrozenInt8Linear,
        nn.Linear,
    }  # the mixed state the fallback would have trained


def test_a_w8a8_failure_before_any_layer_still_falls_back(monkeypatch):
    model = nn.Module()
    model.lin = nn.Linear(32, 32, bias = False).to(torch.bfloat16).requires_grad_(False)

    def boom(*a, **k):
        raise RuntimeError("no int8 GEMM")

    monkeypatch.setattr(fi8, "convert_frozen_int8", boom)
    events = []
    assert dit._apply_int8_w8a8(model, events.append, None, device = "meta") is False
    assert "weight-only" in events[-1]["message"]
