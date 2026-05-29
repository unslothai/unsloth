# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGML_CUDA_ENABLE_UNIFIED_MEMORY must be set only for AMD unified-memory APUs
(gfx1150/gfx1151), never for discrete AMD, NVIDIA, CPU or macOS."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _fake_torch(hip, archs, *, cuda_ok = True):
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = hip)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_ok,
        device_count = lambda: len(archs),
        get_device_properties = lambda i: types.SimpleNamespace(gcnArchName = archs[i]),
    )
    return t


@pytest.mark.parametrize(
    "hip,archs,expected",
    [
        ("6.2.0", ["gfx1151:xnack-"], True),  # Strix Halo APU (suffix stripped)
        ("6.2.0", ["gfx1150"], True),  # Strix Point APU
        ("6.2.0", ["gfx1100"], False),  # discrete RDNA3
        ("6.2.0", ["gfx1201"], False),  # discrete RDNA4
        ("6.2.0", ["gfx942"], False),  # MI300X (data center)
        (None, ["sm_90"], False),  # NVIDIA (no torch.version.hip)
        ("6.2.0", ["gfx1100", "gfx1151"], True),  # mixed dGPU + APU
    ],
)
def test_apu_unified_memory_gating(monkeypatch, hip, archs, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip, archs))
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is expected


def test_cpu_no_cuda_returns_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", [], cuda_ok = False))
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is False


def test_missing_torch_returns_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is False
