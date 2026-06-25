# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for fp8 text-encoder casting (``diffusion_precision.py``).

Hermetic: torch + diffusers.hooks are stubbed via ``sys.modules`` so the gating and
the apply path run without a GPU or real diffusers.
"""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.diffusion_precision import (
    apply_fp8_text_encoder,
    fp8_text_encoder_supported,
)


def _target(*, device = "cuda", dtype = "bfloat16"):
    return types.SimpleNamespace(device = device, dtype = dtype)


def _stub_torch(monkeypatch, *, with_fp8 = True):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    if with_fp8:
        torch.float8_e4m3fn = "float8_e4m3fn"
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _stub_diffusers_hooks(monkeypatch, recorder):
    hooks = types.ModuleType("diffusers.hooks")
    casting = types.ModuleType("diffusers.hooks.layerwise_casting")
    casting.DEFAULT_SKIP_MODULES_PATTERN = ("norm",)

    def _apply(module, *, storage_dtype, compute_dtype, skip_modules_pattern):
        recorder.append((module, storage_dtype, compute_dtype, skip_modules_pattern))

    hooks.apply_layerwise_casting = _apply
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.layerwise_casting", casting)


# ── gating ────────────────────────────────────────────────────────────────────


def test_fp8_supported_requires_cuda_bf16_and_fp8_dtype(monkeypatch):
    _stub_torch(monkeypatch, with_fp8 = True)
    assert fp8_text_encoder_supported(_target()) is True
    assert fp8_text_encoder_supported(_target(device = "cpu")) is False
    assert fp8_text_encoder_supported(_target(dtype = "float16")) is False


def test_fp8_unsupported_without_fp8_dtype(monkeypatch):
    _stub_torch(monkeypatch, with_fp8 = False)
    assert fp8_text_encoder_supported(_target()) is False


# ── apply ─────────────────────────────────────────────────────────────────────


def test_apply_disabled_returns_empty(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = types.SimpleNamespace(text_encoder = object())
    assert apply_fp8_text_encoder(pipe, _target(), enable = False) == []


def test_apply_casts_all_present_text_encoders(monkeypatch):
    _stub_torch(monkeypatch)
    recorder: list = []
    _stub_diffusers_hooks(monkeypatch, recorder)
    te1, te3 = object(), object()
    # text_encoder + text_encoder_3 present, text_encoder_2 absent.
    pipe = types.SimpleNamespace(text_encoder = te1, text_encoder_2 = None, text_encoder_3 = te3)
    cast = apply_fp8_text_encoder(pipe, _target(), enable = True)
    assert cast == ["text_encoder", "text_encoder_3"]
    # Casts to fp8 storage with bf16 compute and skips norms.
    assert {r[0] for r in recorder} == {te1, te3}
    assert all(r[1] == "float8_e4m3fn" and r[2] == "bfloat16" for r in recorder)


def test_apply_unsupported_target_is_noop(monkeypatch):
    _stub_torch(monkeypatch)
    recorder: list = []
    _stub_diffusers_hooks(monkeypatch, recorder)
    pipe = types.SimpleNamespace(text_encoder = object())
    assert apply_fp8_text_encoder(pipe, _target(device = "cpu"), enable = True) == []
    assert recorder == []


def test_apply_tolerates_casting_failure(monkeypatch):
    _stub_torch(monkeypatch)
    hooks = types.ModuleType("diffusers.hooks")
    casting = types.ModuleType("diffusers.hooks.layerwise_casting")
    casting.DEFAULT_SKIP_MODULES_PATTERN = ("norm",)

    def _boom(module, **kwargs):
        raise RuntimeError("fp8 not supported for this layer")

    hooks.apply_layerwise_casting = _boom
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.layerwise_casting", casting)
    pipe = types.SimpleNamespace(text_encoder = object())
    # A casting failure leaves the encoder dense and reports nothing cast.
    assert apply_fp8_text_encoder(pipe, _target(), enable = True) == []
