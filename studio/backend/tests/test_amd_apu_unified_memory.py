# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGML_CUDA_ENABLE_UNIFIED_MEMORY must be set only for AMD unified-memory APUs
(gfx1150/gfx1151), never for discrete AMD, NVIDIA, CPU or macOS."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _fake_torch(
    hip,
    archs,
    *,
    cuda_ok = True,
):
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


def test_apu_guard_scopes_to_selected_gpu(monkeypatch):
    # Mixed host: physical id 0 = discrete gfx1100, 1 = gfx1151 APU.
    for _m in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_m, raising = False)
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch("6.2.0", ["gfx1100", "gfx1151"])
    )
    # Selecting only the dGPU, or an empty selection, must not be unified-memory.
    assert LlamaCppBackend._amd_apu_wants_unified_memory([0]) is False
    assert LlamaCppBackend._amd_apu_wants_unified_memory([]) is False
    # Selecting the APU, or no selection, does.
    assert LlamaCppBackend._amd_apu_wants_unified_memory([1]) is True
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is True


def test_apu_guard_honors_hip_visible_devices_mask(monkeypatch):
    # ROCm resolves ids via HIP first: the mask exposes only the APU as ordinal 0
    # but physical id 1, so the selection [1] must still match.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", ["gfx1151"]))
    assert LlamaCppBackend._amd_apu_wants_unified_memory([1]) is True
    assert LlamaCppBackend._amd_apu_wants_unified_memory([0]) is False


def test_cpu_no_cuda_returns_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", [], cuda_ok = False))
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is False


def test_missing_torch_returns_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is False


_GB = 1024**3
_MIB_PER_GB = 1024
# Module-level (not a class attr) so it stays a plain function, not a bound method.
_shortfall = LlamaCppBackend._apu_ram_shortfall_message


class TestApuRamShortfall:
    """On a unified-memory APU the weights load into system RAM, so a model
    larger than available RAM (the field case: a 64.6 GB GGUF on a WSL VM capped
    well below the ROCm-reported APU budget) must be refused before spawning,
    not left to OOM-kill the Unsloth process."""

    def test_field_case_wsl_cap_refuses(self):
        # 64.6 GB weights, ~46 GB available (WSL VM): refuse with guidance.
        msg = _shortfall(int(64.6 * _GB), 46 * _MIB_PER_GB)
        assert msg is not None
        assert "65 GB" in msg and "46 GB" in msg
        assert ".wslconfig" in msg

    def test_bare_metal_fits_allows(self):
        # Same model, ~92 GB available (no WSL cap): allow.
        assert _shortfall(int(64.6 * _GB), 92 * _MIB_PER_GB) is None

    def test_unknown_available_never_refuses(self):
        assert _shortfall(int(64.6 * _GB), None) is None

    def test_boundary_at_headroom(self):
        # 20 GB weights, headroom 2 GB. avail 23 GB -> fits; 21 GB -> refuse.
        assert _shortfall(20 * _GB, 23 * _MIB_PER_GB) is None
        assert _shortfall(20 * _GB, 21 * _MIB_PER_GB) is not None

    def test_available_system_memory_is_int_or_none(self):
        v = LlamaCppBackend._available_system_memory_mib()
        assert v is None or (isinstance(v, int) and v > 0)
