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
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", ["gfx1100", "gfx1151"]))
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
    not left to OOM-kill the Studio process."""

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


class TestApuMemoryCeiling:
    """#6834: on bare metal the ROCm-reported free VRAM, not the OS's notion of
    "available" system RAM, is the real ceiling for a unified-memory APU -- the
    BIOS/driver can carve out far more unified memory as VRAM than the OS
    considers general-purpose RAM. WSL keeps the old system-RAM ceiling, since
    there the WSL2 VM's RAM cap is what actually limits the load."""

    @staticmethod
    def _patch_wsl(monkeypatch, is_wsl):
        import utils.paths.path_utils as path_utils
        monkeypatch.setattr(path_utils, "_is_wsl", lambda: is_wsl)

    def test_field_case_6834_strix_halo_bare_metal_allows(self, monkeypatch):
        # The exact reported case: 110 GB free VRAM, but only ~19 GB "available"
        # system RAM. Bare metal must use the GPU figure and allow the load.
        self._patch_wsl(monkeypatch, False)
        monkeypatch.setattr(
            LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda: [(0, 110 * _MIB_PER_GB)])
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 19 * _MIB_PER_GB)
        )
        avail_mib, source = LlamaCppBackend._apu_memory_ceiling_mib()
        assert source == "GPU memory"
        msg = _shortfall(int(21.3 * _GB), avail_mib, source = source)
        assert msg is None

    def test_bare_metal_insufficient_gpu_memory_refuses(self, monkeypatch):
        self._patch_wsl(monkeypatch, False)
        monkeypatch.setattr(
            LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda: [(0, 4 * _MIB_PER_GB)])
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 30 * _MIB_PER_GB)
        )
        avail_mib, source = LlamaCppBackend._apu_memory_ceiling_mib()
        assert source == "GPU memory"
        msg = _shortfall(int(21.3 * _GB), avail_mib, source = source)
        assert msg is not None
        assert "GPU memory" in msg
        assert ".wslconfig" not in msg  # WSL hint doesn't apply to a GPU-memory refusal

    def test_wsl_still_uses_system_ram_ceiling(self, monkeypatch):
        # Regression guard: WSL must keep refusing on the VM RAM cap even when
        # the GPU driver reports plenty of free VRAM.
        self._patch_wsl(monkeypatch, True)
        monkeypatch.setattr(
            LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda: [(0, 110 * _MIB_PER_GB)])
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 19 * _MIB_PER_GB)
        )
        avail_mib, source = LlamaCppBackend._apu_memory_ceiling_mib()
        assert source == "system RAM"
        assert avail_mib == 19 * _MIB_PER_GB
        msg = _shortfall(int(21.3 * _GB), avail_mib, source = source)
        assert msg is not None and ".wslconfig" in msg

    def test_gpu_probe_empty_falls_back_to_system_ram(self, monkeypatch):
        # Unsupported/misdetected GPU: the refusal must not silently vanish,
        # it should keep gating on system RAM like before #6834.
        self._patch_wsl(monkeypatch, False)
        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda: []))
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 19 * _MIB_PER_GB)
        )
        avail_mib, source = LlamaCppBackend._apu_memory_ceiling_mib()
        assert source == "system RAM"
        assert avail_mib == 19 * _MIB_PER_GB

    def test_gpu_indices_scope_the_free_memory_sum(self, monkeypatch):
        # A mixed multi-GPU host: only the selected GPU's free memory counts,
        # matching how _amd_apu_wants_unified_memory itself scopes to a selection.
        self._patch_wsl(monkeypatch, False)
        monkeypatch.setattr(
            LlamaCppBackend,
            "_get_gpu_free_memory",
            staticmethod(lambda: [(0, 4 * _MIB_PER_GB), (1, 110 * _MIB_PER_GB)]),
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 30 * _MIB_PER_GB)
        )
        avail_mib, source = LlamaCppBackend._apu_memory_ceiling_mib([0])
        assert source == "GPU memory"
        assert avail_mib == 4 * _MIB_PER_GB

    def test_gpu_probe_exception_falls_back_to_system_ram(self, monkeypatch):
        self._patch_wsl(monkeypatch, False)

        def _raise():
            raise RuntimeError("boom")

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(_raise))
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 19 * _MIB_PER_GB)
        )
        avail_mib, source = LlamaCppBackend._apu_memory_ceiling_mib()
        assert source == "system RAM"
        assert avail_mib == 19 * _MIB_PER_GB
