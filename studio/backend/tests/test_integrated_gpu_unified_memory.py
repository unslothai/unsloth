# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On a unified-memory NVIDIA host (integrated GPU: CPU and GPU share one RAM
pool -- GB10 / DGX Spark, Jetson) nvidia-smi reports [N/A] free, so the probe
falls through to torch.cuda.mem_get_info, whose "free" tracks raw MemFree and
ignores reclaimable page cache. That misleading ~1.5 GB floored the context fit
at min_ctx=4096 even with ~60 GB really available (#6757). The torch fallback
must lift the free figure to system MemAvailable for integrated GPUs only, never
for discrete cards.
"""

from __future__ import annotations

import subprocess
import sys
import types
from unittest import mock

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _fake_torch(*, integrated, free_mib, total_mib, hip = None):
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = hip)
    props = types.SimpleNamespace(is_integrated = 1 if integrated else 0)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        mem_get_info = lambda i: (free_mib * 1024 * 1024, total_mib * 1024 * 1024),
        get_device_properties = lambda i: props,
    )
    return t


def _mock_nvidia_smi_run(fake_output: str, returncode: int = 0):
    """Patch subprocess.run so only the nvidia-smi probe is faked; on a unified
    GPU it returns [N/A] columns, which the probe skips -> torch fallback."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and "nvidia-smi" in cmd[0]:
            return subprocess.CompletedProcess(
                args = cmd, returncode = returncode, stdout = fake_output, stderr = ""
            )
        return real_run(cmd, *args, **kwargs)

    return mock.patch("subprocess.run", side_effect = fake_run)


@pytest.fixture(autouse = True)
def _clear_visibility_masks(monkeypatch):
    for _m in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(_m, raising = False)


def _fixed_avail(monkeypatch, mib):
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: mib)
    )


# ── _get_gpu_memory: unified-memory ceiling ──


def test_integrated_gpu_uses_system_available_ram(monkeypatch):
    # GB10 reproducer: nvidia-smi -> [N/A], mem_get_info free a misleading ~1.5 GB,
    # ~60 GB actually available. Free must be lifted to system-available; total,
    # already correct from mem_get_info on a unified device, stays put.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(integrated = True, free_mib = 1590, total_mib = 124610))
    _fixed_avail(monkeypatch, 61850)
    with _mock_nvidia_smi_run("0, [N/A], [N/A]\n"):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 61850, 124610)]


def test_discrete_gpu_keeps_mem_get_info_free(monkeypatch):
    # Discrete card: dedicated VRAM is the real ceiling, system RAM is irrelevant,
    # so the reported free must be left exactly as mem_get_info gives it.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(integrated = False, free_mib = 20000, total_mib = 24576))
    _fixed_avail(monkeypatch, 61850)
    with _mock_nvidia_smi_run("", returncode = 1):  # force the torch fallback
        assert LlamaCppBackend._get_gpu_memory() == [(0, 20000, 24576)]


def test_integrated_gpu_unknown_available_keeps_free(monkeypatch):
    # System RAM unreadable (psutil + /proc both fail): fail safe to mem_get_info.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(integrated = True, free_mib = 1590, total_mib = 124610))
    _fixed_avail(monkeypatch, None)
    with _mock_nvidia_smi_run("", returncode = 1):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 1590, 124610)]


def test_integrated_gpu_override_never_lowers_free(monkeypatch):
    # The override only ever raises the ceiling: if mem_get_info free already
    # exceeds system-available, keep the larger figure.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(integrated = True, free_mib = 8000, total_mib = 124610))
    _fixed_avail(monkeypatch, 2000)
    with _mock_nvidia_smi_run("", returncode = 1):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 8000, 124610)]


# ── _gpu_is_integrated flag ──


def test_gpu_is_integrated_true_and_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(integrated = True, free_mib = 1, total_mib = 1))
    assert LlamaCppBackend._gpu_is_integrated(0) is True
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(integrated = False, free_mib = 1, total_mib = 1))
    assert LlamaCppBackend._gpu_is_integrated(0) is False


def test_gpu_is_integrated_missing_attr_is_false(monkeypatch):
    # Older torch without cudaDeviceProp.integrated must fail closed, not raise.
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = None)
    t.cuda = types.SimpleNamespace(get_device_properties = lambda i: types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", t)
    assert LlamaCppBackend._gpu_is_integrated(0) is False
