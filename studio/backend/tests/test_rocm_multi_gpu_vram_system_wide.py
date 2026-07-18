# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The System tab's multi-GPU view must show system-wide VRAM on ROCm (#7072).

When amd-smi is unavailable, get_visible_gpu_utilization fell back to torch,
whose readings are process-local: on Windows WDDM hands each process its own
budget, so a model held by the separate llama-server process read as ~0 VRAM
used even with the GPU full. The primary-GPU endpoint already overlays
system-wide sources (Windows Performance Counters, Linux DRM sysfs); the
multi-device endpoint now applies the same per-GPU overlay, matched by
physical device index so reordering visibility masks stay correct.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _maybe_stub(name: str, builder):
    # Stub only if the real module is unavailable, so this file never shadows
    # real packages for later tests in the same pytest process.
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


def _build_structlog_stub():
    m = types.ModuleType("structlog")
    m.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", _build_structlog_stub)

import utils.hardware.hardware as hw  # noqa: E402


def _device(
    index,
    used,
    total,
    *,
    ordinal = None,
):
    return {
        "index": index,
        "index_kind": "physical",
        "visible_ordinal": index if ordinal is None else ordinal,
        "gpu_utilization_pct": None,
        "temperature_c": None,
        "vram_used_gb": used,
        "vram_total_gb": total,
        "vram_utilization_pct": round((used / total) * 100, 1) if total > 0 else None,
        "power_draw_w": None,
        "power_limit_w": None,
        "power_utilization_pct": None,
    }


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


def test_overlay_windows_is_noop_keeps_torch(monkeypatch):
    # Windows multi-GPU is intentionally not overlaid (perf counters can't be
    # mapped to ROCm ordinals and miss WDDM shared memory): keep torch figures.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_per_card_gb",
        lambda: (_ for _ in ()).throw(AssertionError("sysfs must not run on Windows")),
    )
    devices = [_device(0, used = 0.02, total = 8.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # untouched


def test_overlay_linux_matches_by_physical_index(monkeypatch):
    # HIP_VISIBLE_DEVICES=1,0: devices arrive as [index 1, index 0]. Each must
    # get ITS OWN card's figures, not the other's (positional zip would swap).
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_per_card_gb",
        lambda: [(30.0, 45.0), (0.5, 8.0)],  # position 0 = big card, 1 = small
    )
    devices = [_device(1, used = 0.01, total = 8.0), _device(0, used = 0.02, total = 45.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.5  # index 1 -> small card
    assert devices[0]["vram_total_gb"] == 8.0
    assert devices[1]["vram_used_gb"] == 30.0  # index 0 -> big card
    assert devices[1]["vram_total_gb"] == 45.0


def test_overlay_linux_skips_unified_memory_card(monkeypatch):
    # Strix Halo: sysfs reports only the small dedicated slice while torch sees
    # the GTT-backed pool; the smaller sysfs total must NOT shrink the device
    # (mirrors _apply_unified_memory_correction's larger-total-wins rule).
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_per_card_gb", lambda: [(0.4, 1.0)])
    devices = [_device(0, used = 12.0, total = 96.0)]  # torch's unified pool
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 12.0
    assert devices[0]["vram_total_gb"] == 96.0


def test_overlay_linux_out_of_range_index_untouched(monkeypatch):
    # A masked host exposing physical index 5 with only 2 DRM cards: no
    # unambiguous mapping for it, keep torch data.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw, "_rocm_linux_sysfs_vram_per_card_gb", lambda: [(30.0, 45.0), (0.5, 8.0)]
    )
    devices = [_device(5, used = 0.02, total = 45.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02


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


def test_visible_utilization_relative_index_skips_overlay(monkeypatch):
    # UUID/MIG mask -> no numeric ids -> torch ordinals with index_kind
    # "relative". The overlay matches by PHYSICAL index, so it must not run here
    # (relative 0 is not physical card/adapter 0).
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "GPU-uuid-a", "numeric_ids": None, "supports_explicit_gpu_ids": False},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [])  # UUID mask
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 1)
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids: [{"index": 0, "visible_ordinal": 0, "used_gb": 0.02, "total_gb": 8.0}],
    )
    called = []
    monkeypatch.setattr(hw, "_overlay_system_wide_vram", lambda devices: called.append(1))
    result = hw.get_visible_gpu_utilization()
    assert result["index_kind"] == "relative"
    assert called == []  # overlay skipped for relative indices


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
