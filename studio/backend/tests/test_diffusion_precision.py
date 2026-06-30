# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for text-encoder quantisation (``diffusion_precision.py``).

Hermetic: torch + the diffusers / torchao casters are stubbed via ``sys.modules`` so
gating and the apply path run without a GPU, real diffusers, or real torchao.
"""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.diffusion_precision import (
    TE_QUANT_FP8,
    TE_QUANT_NVFP4,
    normalize_te_quant,
    quantize_text_encoders,
    te_quant_supported,
)


def _target(
    *,
    device = "cuda",
    dtype = "bfloat16",
    cc = (10, 0),
):
    return types.SimpleNamespace(device = device, dtype = dtype, _cc = cc)


def _stub_torch(
    monkeypatch,
    *,
    with_fp8 = True,
    cc = (10, 0),
):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    if with_fp8:
        torch.float8_e4m3fn = "float8_e4m3fn"
    torch.cuda = types.SimpleNamespace(get_device_capability = lambda *a: cc)
    # _cast_fp8 skips nn.Embedding modules from layerwise casting.
    torch.nn = types.SimpleNamespace(Embedding = type("Embedding", (), {}))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _stub_casters(monkeypatch, recorder):
    # diffusers fp8 layerwise casting
    hooks = types.ModuleType("diffusers.hooks")
    casting = types.ModuleType("diffusers.hooks.layerwise_casting")
    casting.DEFAULT_SKIP_MODULES_PATTERN = ("norm",)
    hooks.apply_layerwise_casting = lambda module, **kw: recorder.append(("fp8", module))
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.layerwise_casting", casting)
    # torchao nvfp4
    tq = types.ModuleType("torchao.quantization")
    tq.quantize_ = lambda module, config: recorder.append(("nvfp4", module))
    mx = types.ModuleType("torchao.prototype.mx_formats")
    mx.NVFP4WeightOnlyConfig = lambda: "nvfp4cfg"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tq)
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", mx)


# ── normalisation ─────────────────────────────────────────────────────────────


def test_normalize_te_quant():
    assert normalize_te_quant(None) is None
    assert normalize_te_quant("") is None
    assert normalize_te_quant("none") is None
    assert normalize_te_quant("FP8") == TE_QUANT_FP8
    assert normalize_te_quant("NVFP4") == TE_QUANT_NVFP4
    with pytest.raises(ValueError):
        normalize_te_quant("int2")


# ── gating ────────────────────────────────────────────────────────────────────


def test_fp8_supported_requires_cuda_bf16_and_fp8(monkeypatch):
    _stub_torch(monkeypatch, with_fp8 = True)
    assert te_quant_supported(_target(), TE_QUANT_FP8) is True
    assert te_quant_supported(_target(device = "cpu"), TE_QUANT_FP8) is False
    assert te_quant_supported(_target(dtype = "float16"), TE_QUANT_FP8) is False


def test_nvfp4_supported_requires_blackwell(monkeypatch):
    _stub_torch(monkeypatch, cc = (10, 0))
    assert te_quant_supported(_target(), TE_QUANT_NVFP4) is True
    # Hopper (cc 9.0) has no NVFP4 tensor cores.
    _stub_torch(monkeypatch, cc = (9, 0))
    assert te_quant_supported(_target(), TE_QUANT_NVFP4) is False


# ── apply ─────────────────────────────────────────────────────────────────────


def test_quantize_disabled_returns_none(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(), mode = None) is None
    assert quantize_text_encoders(pipe, _target(), mode = "none") is None


def test_quantize_fp8_casts_all_encoders(monkeypatch):
    _stub_torch(monkeypatch)
    recorder: list = []
    _stub_casters(monkeypatch, recorder)
    te1, te3 = object(), object()
    pipe = types.SimpleNamespace(text_encoder = te1, text_encoder_2 = None, text_encoder_3 = te3)
    mode = quantize_text_encoders(pipe, _target(), mode = "fp8")
    assert mode == TE_QUANT_FP8
    assert recorder == [("fp8", te1), ("fp8", te3)]


def test_quantize_nvfp4_uses_torchao(monkeypatch):
    _stub_torch(monkeypatch, cc = (10, 0))
    recorder: list = []
    _stub_casters(monkeypatch, recorder)
    te = object()
    pipe = types.SimpleNamespace(text_encoder = te)
    mode = quantize_text_encoders(pipe, _target(), mode = "nvfp4")
    assert mode == TE_QUANT_NVFP4
    assert recorder == [("nvfp4", te)]


def test_quantize_nvfp4_unsupported_on_hopper_is_noop(monkeypatch):
    _stub_torch(monkeypatch, cc = (9, 0))
    recorder: list = []
    _stub_casters(monkeypatch, recorder)
    pipe = types.SimpleNamespace(text_encoder = object())
    assert quantize_text_encoders(pipe, _target(cc = (9, 0)), mode = "nvfp4") is None
    assert recorder == []


def test_quantize_tolerates_caster_failure(monkeypatch):
    _stub_torch(monkeypatch)
    hooks = types.ModuleType("diffusers.hooks")
    casting = types.ModuleType("diffusers.hooks.layerwise_casting")
    casting.DEFAULT_SKIP_MODULES_PATTERN = ("norm",)

    def _boom(module, **kwargs):
        raise RuntimeError("fp8 unsupported for this layer")

    hooks.apply_layerwise_casting = _boom
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.layerwise_casting", casting)
    pipe = types.SimpleNamespace(text_encoder = object())
    # The only encoder fails to cast -> nothing applied -> None.
    assert quantize_text_encoders(pipe, _target(), mode = "fp8") is None
