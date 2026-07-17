# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The System tab's multi-GPU view must show system-wide VRAM on ROCm (#7072).

When amd-smi is unavailable, get_visible_gpu_utilization fell back to torch,
whose readings are process-local: on Windows WDDM hands each process its own
budget, so a model held by the separate llama-server process read as ~0 VRAM
used even with the GPU full. The primary-GPU endpoint already overlays
system-wide sources (Windows Performance Counters, Linux DRM sysfs); the
multi-device endpoint now applies the same per-GPU overlay.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

import utils.hardware.hardware as hw  # noqa: E402


def _device(index, used, total):
    return {
        "index": index,
        "index_kind": "physical",
        "visible_ordinal": index,
        "gpu_utilization_pct": None,
        "temperature_c": None,
        "vram_used_gb": used,
        "vram_total_gb": total,
        "vram_utilization_pct": round((used / total) * 100, 1) if total > 0 else None,
        "power_draw_w": None,
        "power_limit_w": None,
        "power_utilization_pct": None,
    }


# ── Windows per-adapter perf counters ──


def _mock_powershell(stdout, returncode = 0):
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and "powershell" in str(cmd[0]).lower():
            return subprocess.CompletedProcess(
                args = cmd, returncode = returncode, stdout = stdout, stderr = ""
            )
        return real_run(cmd, *args, **kwargs)

    return mock.patch.object(hw.subprocess, "run", side_effect = fake_run)


def test_windows_per_adapter_parses_and_groups(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    out = (
        "luid_0x00000000_0x0000d3ec_phys_0|21474836480\n"  # 20 GiB
        "luid_0x00000000_0x0000d3ed_phys_1|1073741824\n"  # 1 GiB
        "luid_0x00000000_0x0000d3ee_phys_1|1073741824\n"  # +1 GiB same adapter
        "garbage-line-without-separator\n"
        "luid_0x0_no_phys_index|123\n"
    )
    with _mock_powershell(out):
        per_adapter = hw._rocm_windows_perf_counter_vram_per_adapter_gb()
    assert per_adapter == {0: 20.0, 1: 2.0}


def test_windows_per_adapter_failure_is_empty(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    with _mock_powershell("", returncode = 1):
        assert hw._rocm_windows_perf_counter_vram_per_adapter_gb() == {}


def test_windows_per_adapter_off_windows_is_empty(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    assert hw._rocm_windows_perf_counter_vram_per_adapter_gb() == {}


# ── Linux per-card sysfs ──


def test_linux_per_card_reads_sorted_cards(monkeypatch, tmp_path):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    # card10 before card2 in glob order: numeric sort must fix it.
    files = {}
    for card, used, total in ((10, 1, 8), (2, 40, 48)):
        d = tmp_path / f"card{card}" / "device"
        d.mkdir(parents = True)
        (d / "mem_info_vram_used").write_text(str(used * 1024**3))
        (d / "mem_info_vram_total").write_text(str(total * 1024**3))
        files[card] = str(d / "mem_info_vram_used")
    monkeypatch.setattr(hw.glob, "glob", lambda pattern: [files[10], files[2]])
    assert hw._rocm_linux_sysfs_vram_per_card_gb() == [(40.0, 48.0), (1.0, 8.0)]


def test_linux_per_card_skips_zero_total_and_bad_files(monkeypatch, tmp_path):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    good = tmp_path / "card0" / "device"
    zero = tmp_path / "card1" / "device"
    good.mkdir(parents = True)
    zero.mkdir(parents = True)
    (good / "mem_info_vram_used").write_text(str(2 * 1024**3))
    (good / "mem_info_vram_total").write_text(str(16 * 1024**3))
    (zero / "mem_info_vram_used").write_text("0")
    (zero / "mem_info_vram_total").write_text("0")  # e.g. headless iGPU node
    monkeypatch.setattr(
        hw.glob,
        "glob",
        lambda pattern: [
            str(good / "mem_info_vram_used"),
            str(zero / "mem_info_vram_used"),
            str(tmp_path / "card2" / "device" / "mem_info_vram_used"),  # missing
        ],
    )
    assert hw._rocm_linux_sysfs_vram_per_card_gb() == [(2.0, 16.0)]


# ── overlay ──


def test_overlay_windows_replaces_matched_indices(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_per_adapter_gb", lambda: {0: 21.5})
    devices = [_device(0, used = 0.02, total = 45.0), _device(1, used = 0.01, total = 8.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 21.5  # llama-server's usage now visible
    assert devices[0]["vram_utilization_pct"] == 47.8
    assert devices[1]["vram_used_gb"] == 0.01  # unmatched adapter untouched


def test_overlay_windows_clamps_to_total(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_per_adapter_gb", lambda: {0: 99.0})
    devices = [_device(0, used = 0.0, total = 8.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 8.0


def test_overlay_linux_replaces_in_order(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw, "_rocm_linux_sysfs_vram_per_card_gb", lambda: [(30.0, 45.0), (0.5, 8.0)]
    )
    devices = [_device(0, used = 0.02, total = 45.0), _device(1, used = 0.01, total = 8.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 30.0
    assert devices[1]["vram_used_gb"] == 0.5
    assert devices[0]["vram_utilization_pct"] == 66.7


def test_overlay_linux_count_mismatch_keeps_torch(monkeypatch):
    # Extra iGPU card vs two visible devices: ambiguous mapping, keep torch data.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_per_card_gb",
        lambda: [(30.0, 45.0), (0.5, 8.0), (0.1, 1.0)],
    )
    devices = [_device(0, used = 0.02, total = 45.0), _device(1, used = 0.01, total = 8.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02
    assert devices[1]["vram_used_gb"] == 0.01


def test_overlay_empty_devices_is_noop(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    hw._overlay_system_wide_vram([])  # must not raise


# ── integration: the ROCm torch fallback applies the overlay ──


def test_visible_utilization_rocm_fallback_overlays(monkeypatch):
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)  # amd-smi unavailable
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids: [
            {"index": 0, "visible_ordinal": 0, "used_gb": 0.02, "total_gb": 45.0},
            {"index": 1, "visible_ordinal": 1, "used_gb": 0.01, "total_gb": 8.0},
        ],
    )
    overlaid = []
    monkeypatch.setattr(
        hw, "_overlay_system_wide_vram", lambda devices: overlaid.append(len(devices))
    )
    result = hw.get_visible_gpu_utilization()
    assert result["available"] is True
    assert overlaid == [2]  # overlay ran over both devices


def test_visible_utilization_nvidia_fallback_skips_overlay(monkeypatch):
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids: [{"index": 0, "visible_ordinal": 0, "used_gb": 1.0, "total_gb": 24.0}],
    )
    called = []
    monkeypatch.setattr(hw, "_overlay_system_wide_vram", lambda devices: called.append(1))
    result = hw.get_visible_gpu_utilization()
    assert result["available"] is True
    assert called == []  # NVIDIA keeps the plain torch fallback
