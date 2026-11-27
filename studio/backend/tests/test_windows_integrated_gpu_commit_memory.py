# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows RTX Spark iGPU: system commit bounds the credit. Fixtures are real RTX Spark readings."""

from __future__ import annotations

import ctypes
import sys
import types

import core.inference.diffusion_memory as diffusion_memory
import pytest
from utils import host_memory

GIB = 1 << 30
MIB = 1 << 20

CUDA_FREE_B = int(109.52 * GIB)
CUDA_TOTAL_B = 117_808_041_984
CUDA_FREE_MIB = CUDA_FREE_B // MIB
CUDA_TOTAL_MIB = CUDA_TOTAL_B // MIB
OS_AVAILABLE_MIB = int(50.67 * 1024)
COMMIT_AVAIL_MIB = int(58.14 * 1024)
COMMIT_LIMIT_MIB = 74_733


class _RtxSparkProps:
    name = "NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)"
    total_memory = CUDA_TOTAL_B
    is_integrated = 1  # `.integrated` is absent on this torch
    gcnArchName = ""


class _DiscreteProps:
    name = "NVIDIA GeForce RTX 5090"
    total_memory = 32 * GIB
    is_integrated = 0
    gcnArchName = ""


def _install(
    monkeypatch,
    *,
    platform,
    props,
    hip = None,
    free_b = CUDA_FREE_B,
    total_b = CUDA_TOTAL_B,
    commit = "ok",
):
    # Import before faking sys.platform so _torch_is_rocm's lazy import is not re-run under win32.
    from core.inference import llama_cpp  # noqa: F401

    torch_stub = types.SimpleNamespace(
        __version__ = "2.16.0+cu135" if hip is None else "2.9.0+rocm6.4",
        version = types.SimpleNamespace(hip = hip, cuda = None if hip else "13.5"),
        cuda = types.SimpleNamespace(
            current_device = lambda: 0,
            get_device_properties = lambda ordinal: props,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    hardware_stub = types.ModuleType("utils.hardware")
    hardware_stub.trusted_mem_get_info = lambda: (free_b, total_b)
    monkeypatch.setitem(sys.modules, "utils.hardware", hardware_stub)
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: OS_AVAILABLE_MIB)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: None)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)
    monkeypatch.setattr(diffusion_memory, "_device_user_budget_reserve_mib", lambda *a, **k: 0)
    calls = []

    def _commit():
        calls.append(1)
        if commit == "raise":
            raise OSError("GlobalMemoryStatusEx unavailable")
        if commit is None:
            return None
        return COMMIT_AVAIL_MIB, COMMIT_LIMIT_MIB

    monkeypatch.setattr(host_memory, "windows_commit_mib", _commit)
    monkeypatch.setattr(sys, "platform", platform)
    return calls


def _snapshot():
    return diffusion_memory.snapshot_device_memory(
        types.SimpleNamespace(device = "cuda", backend = "cuda")
    )


def test_windows_integrated_gpu_is_bounded_by_commit(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_WIN_IGPU_COMMIT_CAP", raising = False)
    _install(monkeypatch, platform = "win32", props = _RtxSparkProps())
    memory = _snapshot()

    assert memory.memory_kind == "unified_memory"
    # Before: max(cuda_free, min(os_available, cuda_total)) = 109.5 GiB credited.
    assert memory.free_mib == COMMIT_AVAIL_MIB
    assert memory.total_mib == COMMIT_LIMIT_MIB
    assert memory.credit_basis == diffusion_memory.WINDOWS_COMMIT_CREDIT
    assert memory.as_public_dict()["credit_basis"] == "windows_commit"

    reserve = diffusion_memory._reserve_mib("unified_memory", COMMIT_LIMIT_MIB)
    budget = diffusion_memory._safe_device_budget_mib(memory)
    assert budget == COMMIT_AVAIL_MIB - reserve
    assert 40 * 1024 < budget < 50 * 1024


@pytest.mark.parametrize("switch", ["0", "false", "off"])
def test_kill_switch_restores_the_previous_credit(monkeypatch, switch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_WIN_IGPU_COMMIT_CAP", switch)
    calls = _install(monkeypatch, platform = "win32", props = _RtxSparkProps())
    memory = _snapshot()

    assert (memory.free_mib, memory.total_mib) == (CUDA_FREE_MIB, CUDA_TOTAL_MIB)
    assert memory.credit_basis is None
    assert "credit_basis" not in memory.as_public_dict()
    assert calls == []


@pytest.mark.parametrize("commit", [None, "raise"])
def test_unreadable_commit_falls_back_to_the_previous_credit(monkeypatch, commit):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_WIN_IGPU_COMMIT_CAP", raising = False)
    _install(monkeypatch, platform = "win32", props = _RtxSparkProps(), commit = commit)
    memory = _snapshot()

    assert memory.memory_kind == "unified_memory"
    assert (memory.free_mib, memory.total_mib) == (CUDA_FREE_MIB, CUDA_TOTAL_MIB)
    assert memory.credit_basis is None


def test_linux_integrated_gpu_is_unchanged(monkeypatch):
    """DGX Spark / Jetson keep max(cuda_free, min(os_available, cuda_total)); the commit reader is never asked."""
    monkeypatch.delenv("UNSLOTH_DIFFUSION_WIN_IGPU_COMMIT_CAP", raising = False)
    calls = _install(monkeypatch, platform = "linux", props = _RtxSparkProps())
    memory = _snapshot()

    assert memory.memory_kind == "unified_memory"
    assert (memory.free_mib, memory.total_mib) == (CUDA_FREE_MIB, CUDA_TOTAL_MIB)
    assert memory.credit_basis is None
    assert set(memory.as_public_dict()) == {
        "backend",
        "device",
        "memory_kind",
        "free_mib",
        "total_mib",
    }
    assert calls == []


def test_windows_discrete_gpu_is_unchanged(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_WIN_IGPU_COMMIT_CAP", raising = False)
    calls = _install(
        monkeypatch, platform = "win32", props = _DiscreteProps(), free_b = 30 * GIB, total_b = 32 * GIB
    )
    memory = _snapshot()

    assert memory.memory_kind == "discrete_vram"
    assert (memory.free_mib, memory.total_mib) == (30 * 1024, 32 * 1024)
    assert memory.credit_basis is None
    assert calls == []


def test_windows_rocm_apu_is_unchanged(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_WIN_IGPU_COMMIT_CAP", raising = False)

    class _ApuProps:
        name = "AMD Radeon 8060S Graphics"
        total_memory = 96 * GIB
        is_integrated = 1
        gcnArchName = "gfx1151"

    calls = _install(
        monkeypatch,
        platform = "win32",
        props = _ApuProps(),
        hip = "6.4.0",
        free_b = 90 * GIB,
        total_b = 96 * GIB,
    )
    memory = _snapshot()

    assert memory.memory_kind == "unified_memory"
    assert (memory.free_mib, memory.total_mib) == (90 * 1024, 96 * 1024)
    assert memory.credit_basis is None
    assert calls == []


def test_commit_credit_never_exceeds_the_driver_reading(monkeypatch):
    """Commit headroom above cuda_free (a near-full carve-out) still credits only cuda_free."""
    monkeypatch.delenv("UNSLOTH_DIFFUSION_WIN_IGPU_COMMIT_CAP", raising = False)
    _install(monkeypatch, platform = "win32", props = _RtxSparkProps(), free_b = 20 * GIB)
    memory = _snapshot()

    assert memory.free_mib == 20 * 1024
    assert memory.total_mib == COMMIT_LIMIT_MIB


def test_windows_commit_mib_reads_global_memory_status(monkeypatch):
    def _global_memory_status_ex(ref):
        status = ref._obj
        assert status.dwLength == ctypes.sizeof(status)
        status.ullTotalPageFile = COMMIT_LIMIT_MIB * MIB
        status.ullAvailPageFile = COMMIT_AVAIL_MIB * MIB
        status.ullAvailPhys = int(50.75 * GIB)
        return 1

    kernel32 = types.SimpleNamespace(GlobalMemoryStatusEx = _global_memory_status_ex)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: kernel32, raising = False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert host_memory.windows_commit_mib() == (COMMIT_AVAIL_MIB, COMMIT_LIMIT_MIB)

    kernel32.GlobalMemoryStatusEx = lambda ref: 0
    assert host_memory.windows_commit_mib() is None

    def _boom(*a, **k):
        raise OSError("no kernel32")

    monkeypatch.setattr(ctypes, "WinDLL", _boom, raising = False)
    assert host_memory.windows_commit_mib() is None


def test_windows_commit_mib_is_none_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert host_memory.windows_commit_mib() is None
