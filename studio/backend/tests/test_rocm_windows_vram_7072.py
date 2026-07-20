# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #7072 -- "VRAM Usage in System Tab is wrong".

Reporter: dual AMD (Radeon PRO W7900 ~48GB + W7500 8GB), Windows 10, ROCm 7.13,
torch 2.11.0+rocm7.13. On Windows without a HIP SDK, amd-smi is permanently
disabled (avoids a UAC/DiskPart prompt) and hipMemGetInfo returns free==total
(used 0). Two symptoms followed:

  * System tab (/api/system -> get_visible_gpu_utilization) showed ~0 VRAM used
    on every GPU (torch mem_get_info free==total quirk; ROCm/ROCm#1909).
  * get_gpu_utilization()'s Windows fallback SUMMED "GPU Adapter Memory\\Dedicated
    Usage" across all adapters into ONE fake device with only GPU 0's total, so
    the second GPU never appeared.

The fix reads the per-adapter (LUID-instanced) Dedicated Usage performance
counter -- Task Manager's source -- for per-GPU used, takes per-GPU total from
torch device properties, and guards the free==total mem_get_info quirk. CI has no
AMD GPU/Windows, so torch, the performance counter, and platform are all mocked.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from utils.hardware import hardware as hw

GB = 1024**3
MiB = 1024**2


# ----------------------------------------------------------------------------- #
# Fakes
# ----------------------------------------------------------------------------- #
def _fake_torch(
    devices,
    *,
    free_equals_total = False,
    used_per_device = None,
):
    """Build a fake `torch` module. devices: list of (name, total_bytes)."""
    dev = list(devices)

    class _Props:
        def __init__(self, name, total):
            self.name = name
            self.total_memory = total

    def get_device_properties(i):
        name, total = dev[i]
        return _Props(name, total)

    def mem_get_info(i):
        _, total = dev[i]
        if free_equals_total:
            return (total, total)
        used = used_per_device[i] if used_per_device is not None else 0
        return (total - used, total)

    t = types.ModuleType("torch")
    t.__version__ = "2.11.0+rocm7.13"
    t.version = types.SimpleNamespace(hip = "7.13", cuda = None)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: len(dev) > 0,
        device_count = lambda: len(dev),
        current_device = lambda: 0,
        get_device_properties = get_device_properties,
        mem_get_info = mem_get_info,
        memory_allocated = lambda i: 0,
        memory_reserved = lambda i: 0,
    )
    return t


def _adapter_output(adapters):
    if not adapters:
        return "__NONE__\n"
    return "".join(f"{name}|{int(used)}\n" for name, used in adapters)


def _subprocess_run(*, adapter_output = "__NONE__\n", util_output = "12.0\n"):
    def fake_run(cmd, *a, **k):
        joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "GPU Adapter Memory" in joined and "InstanceName" in joined:
            out = adapter_output
        elif "engtype_3D" in joined or "GPU Engine" in joined:
            out = util_output
        else:
            out = "-1\n"
        return subprocess.CompletedProcess(args = cmd, returncode = 0, stdout = out, stderr = "")

    return fake_run


@pytest.fixture
def win_rocm(monkeypatch):
    """Configure the hardware module as a Windows ROCm host with 2 visible GPUs."""
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw.sys, "platform", "win32")
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)  # amd-smi disabled
    # Visible set via HIP mask so we don't shell out to amd-smi for the count.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    return monkeypatch


REPORTER_ADAPTERS = [
    ("luid_0x00000000_0x0000d1e2_phys_0", 40.0 * GB),  # W7900, model loaded
    ("luid_0x00000000_0x0000e34a_phys_0", 0.5 * GB),  # W7500, idle
    ("luid_0x00000000_0x0000f001_phys_0", 3 * MiB),  # Basic Render Driver
]
DEVICES = [("AMD Radeon PRO W7900", 48 * GB), ("AMD Radeon PRO W7500", 8 * GB)]


# ----------------------------------------------------------------------------- #
# System tab (get_visible_gpu_utilization) -- the reporter's screenshot
# ----------------------------------------------------------------------------- #
def test_system_tab_shows_per_gpu_used(win_rocm, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(DEVICES, free_equals_total = True))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )

    devices = hw.get_visible_gpu_utilization()["devices"]
    by_idx = {d["index"]: d for d in devices}
    assert len(devices) == 2
    assert by_idx[0]["vram_total_gb"] == 48.0
    assert by_idx[0]["vram_used_gb"] == pytest.approx(40.0, abs = 0.01)  # not 0
    assert by_idx[1]["vram_total_gb"] == 8.0  # own total
    # The 3 MiB "Basic Render Driver" counter makes this a hidden-adapter case, so
    # the idle card's 0.5 GiB is not capacity-forced (it also fits a hidden adapter).
    # Only the 40 GiB is forced onto the 48 GiB card; the idle card reads Unknown.
    assert by_idx[1]["vram_used_gb"] is None
    assert by_idx[1]["vram_utilization_pct"] is None
    assert all(
        d["vram_used_gb"] <= d["vram_total_gb"] for d in devices if d["vram_used_gb"] is not None
    )


def test_gpu_utilization_does_not_collapse(win_rocm, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(DEVICES, free_equals_total = True))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )

    result = hw.get_gpu_utilization()
    devices = result["devices"]
    assert sorted(d["index"] for d in devices) == [0, 1]  # both GPUs, no collapse
    assert {d["vram_total_gb"] for d in devices} == {48.0, 8.0}
    assert result["vram_total_gb"] == 48.0  # legacy primary mirror preserved


def test_localized_counter_reports_unknown_not_zero(win_rocm, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(DEVICES, free_equals_total = True))
    monkeypatch.setattr(hw.subprocess, "run", _subprocess_run(adapter_output = "__NONE__\n"))

    devices = hw.get_visible_gpu_utilization()["devices"]
    assert len(devices) == 2  # both still shown with correct totals
    assert {d["vram_total_gb"] for d in devices} == {48.0, 8.0}
    assert all(d["vram_used_gb"] is None for d in devices)  # unknown, not fake 0
    assert all(d["vram_utilization_pct"] is None for d in devices)


# ----------------------------------------------------------------------------- #
# mem_get_info free==total guard scoping
# ----------------------------------------------------------------------------- #
def test_mem_get_info_guard_scopes_to_windows_rocm(monkeypatch):
    torch_mod = _fake_torch(DEVICES, free_equals_total = True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    # Windows ROCm -> used unknown (None), total kept.
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.sys, "platform", "win32")
    win = hw._torch_get_per_device_info([0, 1])
    assert [d["used_gb"] for d in win] == [None, None]
    assert [d["total_gb"] for d in win] == [48.0, 8.0]

    # Linux ROCm -> unchanged numeric used.
    monkeypatch.setattr(hw.sys, "platform", "linux")
    assert [d["used_gb"] for d in hw._torch_get_per_device_info([0, 1])] == [0.0, 0.0]

    # Windows NVIDIA -> guard must not fire.
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.sys, "platform", "win32")
    assert [d["used_gb"] for d in hw._torch_get_per_device_info([0, 1])] == [0.0, 0.0]


# ----------------------------------------------------------------------------- #
# Per-adapter attribution helpers (pure unit)
# ----------------------------------------------------------------------------- #
def test_match_adapter_pairs_and_clamps():
    assert hw._match_adapter_used_to_devices([40 * GB, 0.5 * GB], [48 * GB, 8 * GB]) == [
        40 * GB,
        0.5 * GB,
    ]
    assert hw._match_adapter_used_to_devices([100 * GB], [48 * GB]) == [48 * GB]  # clamp
    assert hw._match_adapter_used_to_devices([40 * GB], [48 * GB, 8 * GB]) == [40 * GB, None]


def test_match_adapter_reports_unknown_when_more_active_than_visible():
    # More adapters actively using VRAM than are visible (a GPU outside the mask):
    # attribution would fabricate a value, so report unknown for every device.
    assert hw._match_adapter_used_to_devices([40 * GB, 0.5 * GB], [8 * GB]) == [None]


def test_match_adapter_reports_unknown_when_hidden_high_use_adapter_survives_filter():
    # Idle visible 8 GiB card (10 MiB, filtered as noise) beside a hidden 48 GiB
    # card at 40 GiB. The 40 GiB survivor can't fit the 8 GiB device, so clamping
    # it there would fabricate a fully-used reading. Report unknown.
    assert hw._match_adapter_used_to_devices([40 * GB, 10 * MiB], [8 * GB]) == [None]
    # Order of the counters must not matter.
    assert hw._match_adapter_used_to_devices([10 * MiB, 40 * GB], [8 * GB]) == [None]


def test_match_adapter_reports_unknown_for_placeholder_fallback():
    # Idle GPUs (every counter below the 64 MiB floor) plus a placeholder counter.
    # No LUID-to-ordinal mapping distinguishes the placeholder from an idle GPU, so
    # report unknown rather than fabricate.
    # Single visible 8 GiB card idle (10 MiB) beside a 50 MiB placeholder counter.
    assert hw._match_adapter_used_to_devices([50 * MiB, 10 * MiB], [8 * GB]) == [None]
    # Order of the counters must not matter.
    assert hw._match_adapter_used_to_devices([10 * MiB, 50 * MiB], [8 * GB]) == [None]
    # Two idle visible GPUs plus a placeholder: all three counters below the floor.
    assert hw._match_adapter_used_to_devices([50 * MiB, 10 * MiB, 5 * MiB], [48 * GB, 8 * GB]) == [
        None,
        None,
    ]


def test_match_adapter_reports_unknown_when_usage_not_capacity_ordered():
    # 8 GiB card near full (7 GiB) beside a lightly used 48 GiB card (5 GiB). The
    # bigger usage still fits the smaller card, so both [8<-7, 48<-5] and
    # [8<-5, 48<-7] are feasible; without a verified mapping, report unknown.
    assert hw._match_adapter_used_to_devices([7 * GB, 5 * GB], [8 * GB, 48 * GB]) == [None, None]
    # Device order must not matter (same physical situation, ordinals flipped).
    assert hw._match_adapter_used_to_devices([7 * GB, 5 * GB], [48 * GB, 8 * GB]) == [None, None]
    # Same-capacity cards with unequal usage are equally unattributable.
    assert hw._match_adapter_used_to_devices([12 * GB, 8 * GB], [24 * GB, 24 * GB]) == [None, None]
    # A single usage that fits both cards can sit on either -> unknown.
    assert hw._match_adapter_used_to_devices([5 * GB], [48 * GB, 8 * GB]) == [None, None]
    # But a capacity-forced assignment (usage exceeds the smaller card) is kept:
    # 40 GiB can only be the 48 GiB card, so it is not fabrication.
    assert hw._match_adapter_used_to_devices([40 * GB], [48 * GB, 8 * GB]) == [40 * GB, None]


def test_match_adapter_reports_unknown_when_hidden_usage_fits_visible_card():
    # With a hidden adapter present, a survivor that merely *fits* a visible card
    # must NOT be pinned onto it. Two visible cards (48/8 GiB) at 40 GiB / 10 MiB
    # beside a hidden 6 GiB adapter: the 6 GiB fits the idle 8 GiB card but isn't
    # capacity-forced, so that card reads Unknown; only 40 GiB is forced.
    assert hw._match_adapter_used_to_devices([40 * GB, 10 * MiB, 6 * GB], [48 * GB, 8 * GB]) == [
        40 * GB,
        None,
    ]
    # Counter order must not matter.
    assert hw._match_adapter_used_to_devices([6 * GB, 40 * GB, 10 * MiB], [48 * GB, 8 * GB]) == [
        40 * GB,
        None,
    ]
    # A single visible card with a hidden adapter is never attributable: a fitting
    # survivor could be the hidden GPU's while the visible card is idle.
    assert hw._match_adapter_used_to_devices([6 * GB, 10 * MiB], [8 * GB]) == [None]


def test_match_adapter_capacity_forced_matrix():
    """Exhaustive hidden-adapter matrix for the capacity-forced rule.

    A value is emitted only when the supra-threshold counters number exactly the
    visible devices AND a device's ranked usage strictly exceeds every smaller
    card's capacity. Otherwise (a visible card idle, a merely-fitting usage, or the
    smallest card) every device reports unknown.
    """
    m = hw._match_adapter_used_to_devices
    # -- exactly-n supra-threshold counters, capacity-forced survivors are kept - #
    # Both visible cards have a real reading (the 3 MiB is a placeholder): 40 GiB
    # forced onto the 48 GiB card, 0.5 GiB not forced -> None.
    assert m([40 * GB, 0.5 * GB, 3 * MiB], [48 * GB, 8 * GB]) == [40 * GB, None]
    # Three visible cards all active (supra-threshold) + placeholder: 40 > 24 and
    # 20 > 8, both forced; the 8 GiB card is not forced -> None.
    assert m([40 * GB, 20 * GB, 5 * GB, 3 * MiB], [48 * GB, 24 * GB, 8 * GB]) == [
        40 * GB,
        20 * GB,
        None,
    ]
    # -- fewer supra-threshold counters than visible cards -> all unknown ------ #
    # A visible card is idle, so even a "forced" 40 could be the hidden GPU's.
    assert m([40 * GB, 3 * MiB, 3 * MiB], [48 * GB, 8 * GB]) == [None, None]
    assert m([40 * GB, 10 * MiB, 10 * MiB], [48 * GB, 8 * GB]) == [None, None]
    assert m([40 * GB, 20 * GB, 3 * MiB, 3 * MiB], [48 * GB, 24 * GB, 8 * GB]) == [
        None,
        None,
        None,
    ]
    # Middle usage (6 GiB) fits both the 24 and 8 GiB cards, and only two cards are
    # active for three visible -> not a bijection -> all unknown.
    assert m([40 * GB, 6 * GB, 3 * MiB, 3 * MiB], [48 * GB, 24 * GB, 8 * GB]) == [
        None,
        None,
        None,
    ]
    # -- hidden larger than every visible card -> all unknown ----------------- #
    assert m([40 * GB, 10 * MiB], [8 * GB]) == [None]
    assert m([48 * GB, 3 * MiB, 3 * MiB], [24 * GB, 8 * GB]) == [None, None]
    # -- more active adapters than visible cards -> all unknown --------------- #
    assert m([40 * GB, 7 * GB, 6 * GB, 3 * MiB], [48 * GB, 8 * GB]) == [None, None]
    assert m([40 * GB, 7 * GB, 6 * GB, 3 * MiB, 3 * MiB], [48 * GB, 8 * GB]) == [None, None]
    # -- every counter below the noise floor (placeholder fallback) -> unknown - #
    assert m([50 * MiB, 10 * MiB], [8 * GB]) == [None]
    assert m([50 * MiB, 10 * MiB, 5 * MiB], [48 * GB, 8 * GB]) == [None, None]
    # -- equal-capacity cards with a hidden adapter: nothing is forced -------- #
    assert m([40 * GB, 40 * GB, 3 * MiB], [48 * GB, 48 * GB]) == [None, None]
    assert m([40 * GB, 30 * GB, 3 * MiB], [48 * GB, 48 * GB]) == [None, None]


def test_perf_counter_parser_and_sentinel(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )
    parsed = hw._rocm_windows_perf_counter_vram_by_adapter()
    assert parsed is not None and len(parsed) == 3
    assert parsed[0][0].startswith("luid_")
    monkeypatch.setattr(hw.subprocess, "run", _subprocess_run(adapter_output = "__NONE__\n"))
    assert hw._rocm_windows_perf_counter_vram_by_adapter() is None


# ----------------------------------------------------------------------------- #
# Unified-memory (Strix Halo APU) total reconciliation (Codex #7238)
# ----------------------------------------------------------------------------- #
def test_unified_memory_adopts_torch_total_even_when_used_unknown():
    """Windows ROCm unified-memory APU: torch's used is None but its total (the full
    GTT pool) is authoritative. The correction must still adopt the larger total;
    used stays at amd-smi's figure when torch's is unknown."""
    metrics = {"vram_total_gb": 8.0, "vram_used_gb": 2.0, "vram_utilization_pct": 25.0}
    hw._apply_unified_memory_correction(metrics, {"total_gb": 124.0, "used_gb": None, "index": 0})
    assert metrics["vram_total_gb"] == 124.0  # full unified pool, not the 8 GB carve-out
    assert metrics["vram_used_gb"] == 2.0  # amd-smi used preserved (torch's was None)
    assert metrics["vram_utilization_pct"] == pytest.approx(round(2.0 / 124.0 * 100, 1))


def test_unified_memory_overwrites_used_when_torch_used_known():
    """When torch reports both a larger total and a known used, both are adopted
    and utilization is recomputed against the corrected total (unchanged path)."""
    metrics = {"vram_total_gb": 8.0, "vram_used_gb": 2.0, "vram_utilization_pct": 25.0}
    hw._apply_unified_memory_correction(metrics, {"total_gb": 124.0, "used_gb": 40.0, "index": 0})
    assert metrics["vram_total_gb"] == 124.0
    assert metrics["vram_used_gb"] == 40.0
    assert metrics["vram_utilization_pct"] == pytest.approx(round(40.0 / 124.0 * 100, 1))


def test_unified_memory_no_op_when_torch_total_not_larger():
    """A discrete GPU where torch total does not exceed amd-smi's is left untouched."""
    metrics = {"vram_total_gb": 48.0, "vram_used_gb": 10.0, "vram_utilization_pct": 20.8}
    hw._apply_unified_memory_correction(metrics, {"total_gb": 48.0, "used_gb": None, "index": 0})
    assert metrics["vram_total_gb"] == 48.0
    assert metrics["vram_used_gb"] == 10.0
    assert metrics["vram_utilization_pct"] == 20.8
