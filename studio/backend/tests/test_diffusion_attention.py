# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for attention-backend selection. No torch/diffusers needed:
``_is_cuda_nvidia`` is monkeypatched for the policy tests, and the apply path uses a fake
transformer that records / raises on ``set_attention_backend``.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import (
    ATTN_AUTO,
    apply_attention_backend,
    normalize_attention_backend,
    select_attention_backend,
)


def _target(device = "cuda"):
    return types.SimpleNamespace(device = device)


# ── normalize ────────────────────────────────────────────────────────────────────
def test_normalize_defaults_and_aliases():
    assert normalize_attention_backend(None) == ATTN_AUTO
    assert normalize_attention_backend("") == ATTN_AUTO
    assert normalize_attention_backend("auto") == ATTN_AUTO
    assert normalize_attention_backend("CuDNN") == "cudnn"
    assert normalize_attention_backend("FLASH3") == "flash3"


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_attention_backend("bogus")


# ── select policy ─────────────────────────────────────────────────────────────────
def test_auto_upgrades_to_cudnn_on_nvidia_when_speed_active(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    assert select_attention_backend(_target(), "auto", speed_active = True) == "_native_cudnn"


def test_auto_stays_native_when_speed_off(monkeypatch):
    # off must stay bit-identical -> no backend change even on NVIDIA.
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    assert select_attention_backend(_target(), "auto", speed_active = False) is None


def test_auto_stays_native_off_nvidia(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(device = "mps"), "auto", speed_active = True) is None


def test_explicit_backend_honored_regardless_of_speed(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target(), "sage", speed_active = False) == "sage"
    assert select_attention_backend(_target(), "flash4", speed_active = False) == "flash_4_hub"
    assert select_attention_backend(_target(), "cudnn", speed_active = False) == "_native_cudnn"


def test_explicit_native_returns_none():
    # native is the default -> nothing to set.
    assert select_attention_backend(_target(), "native", speed_active = True) is None


# ── apply ─────────────────────────────────────────────────────────────────────────
class _FakeTransformer:
    def __init__(self, *, fail = False):
        self.fail = fail
        self.set_to = None

    def set_attention_backend(self, name):
        if self.fail:
            raise RuntimeError(f"{name} kernel unavailable")
        self.set_to = name


def _pipe(transformer):
    return types.SimpleNamespace(transformer = transformer)


def test_apply_none_is_noop():
    assert apply_attention_backend(_pipe(_FakeTransformer()), None) is None


def test_apply_sets_backend():
    t = _FakeTransformer()
    engaged = apply_attention_backend(_pipe(t), "_native_cudnn")
    assert engaged == "_native_cudnn" and t.set_to == "_native_cudnn"


def test_apply_falls_back_on_unavailable_kernel():
    # an unavailable kernel must not fail the load -> returns None (diffusers default).
    t = _FakeTransformer(fail = True)
    assert apply_attention_backend(_pipe(t), "sage") is None


def test_apply_handles_missing_method():
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert apply_attention_backend(pipe, "_native_cudnn") is None
