# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for transformer quantisation (``diffusion_transformer_quant.py``).

Hermetic: torch + torchao are stubbed via ``sys.modules``, and the per-scheme smoke
probe (``_scheme_supported`` / ``_smoke_probe``) is monkeypatched where the test cares
about the selection ladder rather than the GPU probe, so everything runs CPU-only.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_transformer_quant import (
    TQ_FP8,
    TQ_INT8,
    TQ_MXFP8,
    TQ_NVFP4,
    dense_transformer_supported,
    make_filter_fn,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)


def _target(*, device = "cuda", dtype = "bfloat16"):
    return types.SimpleNamespace(device = device, dtype = dtype)


def _stub_torch(
    monkeypatch,
    *,
    cc = (10, 0),
    with_fp8 = True,
    cuda_available = True,
):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    if with_fp8:
        torch.float8_e4m3fn = "float8_e4m3fn"
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_available,
        get_device_capability = lambda *a: cc,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# ── normalisation ─────────────────────────────────────────────────────────────


def test_normalize_transformer_quant():
    assert normalize_transformer_quant(None) is None
    assert normalize_transformer_quant("") is None
    assert normalize_transformer_quant("none") is None
    assert normalize_transformer_quant("off") is None
    assert normalize_transformer_quant("AUTO") == "auto"
    assert normalize_transformer_quant("INT8") == TQ_INT8
    assert normalize_transformer_quant("fp8") == TQ_FP8
    with pytest.raises(ValueError):
        normalize_transformer_quant("int2")


# ── dense-source gate ───────────────────────────────────────────────────────────


def test_dense_transformer_supported_requires_cuda_bf16(monkeypatch):
    _stub_torch(monkeypatch)
    assert dense_transformer_supported(_target()) is True
    assert dense_transformer_supported(_target(device = "cpu")) is False
    assert dense_transformer_supported(_target(dtype = "float16")) is False


# ── scheme selection ladder ─────────────────────────────────────────────────────


def _allow(monkeypatch, allowed):
    """Force ``_scheme_supported`` to accept only ``allowed`` (simulates smoke results)."""
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device: scheme in allowed)


def test_auto_blackwell_prefers_nvfp4_then_falls_back(monkeypatch):
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_NVFP4
    # nvfp4 unavailable: fp8 is preferred over mxfp8 (measured faster + a touch more
    # accurate on B200), even though mxfp8 is also supported.
    _allow(monkeypatch, {TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    # Only mxfp8 + int8 left -> mxfp8 (still above int8).
    _allow(monkeypatch, {TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_MXFP8
    # Only int8 usable -> int8.
    _allow(monkeypatch, {TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_ada_hopper_prefers_fp8(monkeypatch):
    _stub_torch(monkeypatch, cc = (8, 9))
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    _stub_torch(monkeypatch, cc = (9, 0))  # Hopper
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8


def test_auto_ampere_prefers_int8(monkeypatch):
    _stub_torch(monkeypatch, cc = (8, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})  # fp8 cores absent on Ampere -> int8 only in ladder
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8
    _stub_torch(monkeypatch, cc = (8, 6))
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_pre_ampere_unsupported(monkeypatch):
    _stub_torch(monkeypatch, cc = (7, 5))  # Turing: below the int8-dynamic floor
    _allow(monkeypatch, {TQ_INT8, TQ_FP8})
    assert select_transformer_quant_scheme(_target(), "auto") is None


def test_explicit_scheme_honored_or_none(monkeypatch):
    _stub_torch(monkeypatch, cc = (8, 0))
    _allow(monkeypatch, {TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "int8") == TQ_INT8
    # Explicit unsupported scheme is NOT silently downgraded -> None (-> GGUF fallback).
    assert select_transformer_quant_scheme(_target(), "fp8") is None
    assert select_transformer_quant_scheme(_target(), "nvfp4") is None


def test_select_none_when_disabled_or_non_cuda(monkeypatch):
    _stub_torch(monkeypatch)
    _allow(monkeypatch, {TQ_INT8, TQ_FP8, TQ_NVFP4})
    assert select_transformer_quant_scheme(_target(), None) is None
    assert select_transformer_quant_scheme(_target(device = "cpu"), "auto") is None


# ── _scheme_supported / _smoke_probe ────────────────────────────────────────────


def test_scheme_supported_shortcircuits(monkeypatch):
    # No CUDA -> False without running the smoke probe.
    _stub_torch(monkeypatch, cuda_available = False)
    monkeypatch.setattr(tq, "_smoke_probe", lambda *a: pytest.fail("probe should not run"))
    assert tq._scheme_supported(TQ_INT8, "cuda") is False
    # fp8 requested but the fp8 dtype is missing -> False before the probe.
    _stub_torch(monkeypatch, with_fp8 = False)
    monkeypatch.setattr(tq, "_smoke_probe", lambda *a: pytest.fail("probe should not run"))
    assert tq._scheme_supported(TQ_FP8, "cuda") is False


def test_smoke_probe_caches_and_tolerates_failure(monkeypatch):
    tq._SMOKE_CACHE.clear()
    calls = {"n": 0}

    class _Lin:
        def __init__(self, *a, **k):
            pass

        def to(self, **k):
            return self

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    torch.randn = lambda *a, **k: object()
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.cuda = types.SimpleNamespace(is_available = lambda: True, synchronize = lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    tqz = types.ModuleType("torchao.quantization")

    def _quantize_ok(
        module,
        config,
        filter_fn = None,
    ):
        calls["n"] += 1

    tqz.quantize_ = _quantize_ok
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda: "fp8cfg"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    # _Lin is callable? No -> the forward lin(x) would fail. Make instances callable.
    _Lin.__call__ = lambda self, x: x

    assert tq._smoke_probe(TQ_INT8, "cuda") is True
    assert tq._smoke_probe(TQ_INT8, "cuda") is True  # cached, no second quantize_
    assert calls["n"] == 1

    # A scheme whose quantize_ raises -> probe False (and cached).
    tq._SMOKE_CACHE.clear()

    def _quantize_boom(
        module,
        config,
        filter_fn = None,
    ):
        raise RuntimeError("kernel unavailable")

    tqz.quantize_ = _quantize_boom
    assert tq._smoke_probe(TQ_FP8, "cuda") is False


# ── filter ──────────────────────────────────────────────────────────────────────


def test_make_filter_fn(monkeypatch):
    class _Lin:
        def __init__(self, i, o):
            self.in_features, self.out_features = i, o

    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    monkeypatch.setitem(sys.modules, "torch", torch)

    keep = make_filter_fn(512)
    assert keep(_Lin(1024, 4096), "blocks.0.attn.to_q") is True
    assert keep(_Lin(256, 4096), "time_proj") is False  # small in_features -> skip
    assert keep(_Lin(4096, 256), "out_proj") is False  # small out_features -> skip
    assert keep(object(), "not_linear") is False  # non-Linear -> skip
    assert keep(types.SimpleNamespace(), "no_attrs") is False


# ── apply ───────────────────────────────────────────────────────────────────────


def test_quantize_transformer_applies_and_marks(monkeypatch):
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda target, mode: TQ_FP8)
    monkeypatch.setattr(tq, "_make_quant_config", lambda scheme: f"{scheme}cfg")
    recorder: list = []
    tqz = types.ModuleType("torchao.quantization")
    tqz.quantize_ = lambda module, config, filter_fn = None: recorder.append(
        (module, config, filter_fn)
    )
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)

    transformer = types.SimpleNamespace()
    pipe = types.SimpleNamespace(transformer = transformer)
    assert quantize_transformer(pipe, _target(), mode = "fp8") == TQ_FP8
    assert len(recorder) == 1 and recorder[0][0] is transformer and recorder[0][1] == "fp8cfg"
    assert callable(recorder[0][2])  # a filter_fn was passed
    assert transformer._unsloth_runtime_quant == TQ_FP8  # diagnostic marker set


def test_quantize_transformer_none_when_unsupported(monkeypatch):
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda target, mode: None)
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert quantize_transformer(pipe, _target(), mode = "auto") is None


def test_quantize_transformer_tolerates_failure(monkeypatch):
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda target, mode: TQ_INT8)
    monkeypatch.setattr(tq, "_make_quant_config", lambda scheme: "cfg")
    tqz = types.ModuleType("torchao.quantization")

    def _boom(
        module,
        config,
        filter_fn = None,
    ):
        raise RuntimeError("partial quant failure")

    tqz.quantize_ = _boom
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    # A quantise failure returns None (caller falls back to GGUF), never raises.
    assert quantize_transformer(pipe, _target(), mode = "int8") is None
