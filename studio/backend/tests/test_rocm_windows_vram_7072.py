# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #7072 -- "VRAM Usage in System Tab is wrong".

Reporter: dual AMD (Radeon PRO W7900 ~48GB + W7500 8GB), Windows 10, ROCm 7.13,
torch 2.11.0+rocm7.13. On Windows without a HIP SDK, amd-smi is permanently
disabled (avoids a UAC/DiskPart prompt) and hipMemGetInfo returns free==total
(used 0). Two symptoms followed:

  * System tab (/api/system -> get_visible_gpu_utilization) showed ~0 VRAM used on
    every GPU (mem_get_info free==total; see rocm_windows_free_is_untrusted for why
    the reading is optimistic).
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
    # The 3 MiB Basic Render Driver counter makes this a hidden-adapter case: only
    # the 40 GiB is forced onto the 48 GiB card; the idle card reads Unknown.
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


def test_match_adapter_pairs_single_device_past_placeholder_adapters():
    # Exactly the shape a gfx1151 Strix Halo host emits: three counter instances,
    # two of them placeholders at EXACTLY 0, one visible device. An adapter at zero
    # has nothing committed and so cannot hold the survivor's bytes, which is what
    # makes this attributable. Before this, the capacity test needed a next-smaller
    # device to compare against and discarded the survivor as unknown.
    assert hw._match_adapter_used_to_devices([1.2 * GB, 0.0, 0.0], [89 * GB]) == [1.2 * GB]
    # Order of the counters must not matter.
    assert hw._match_adapter_used_to_devices([0.0, 1.2 * GB, 0.0], [89 * GB]) == [1.2 * GB]
    # Still clamped: a hidden larger adapter must not report a fully-used card.
    assert hw._match_adapter_used_to_devices([100 * GB, 0.0, 0.0], [48 * GB]) == [None]
    assert hw._match_adapter_used_to_devices([40 * GB, 0.0], [48 * GB]) == [40 * GB]
    # Zero is what makes it attributable, not the cardinality. A merely sub-floor
    # counter can be the visible card idle, making the survivor a hidden GPU's --
    # that shape must stay unknown (see the [6 GiB, 10 MiB] case below).
    assert hw._match_adapter_used_to_devices([1.2 * GB, 10 * MiB, 0.0], [89 * GB]) == [None]


def test_match_adapter_reports_unknown_when_more_active_than_visible():
    # More adapters actively using VRAM than are visible (a GPU outside the mask):
    # attribution would fabricate a value, so report unknown for every device.
    assert hw._match_adapter_used_to_devices([40 * GB, 0.5 * GB], [8 * GB]) == [None]


def test_unified_used_sums_dedicated_and_shared_for_the_compute_adapter(monkeypatch):
    # On an APU, Dedicated saturates at the carve-out and the overflow lands in
    # Shared, so only the sum tracks the allocation (measured: 48 GiB held reports
    # +29.19 dedicated, +19.02 shared on a gfx1151 host).
    def fake(counter = "Dedicated Usage"):
        if counter == "Dedicated Usage":
            return [("luid_compute", 30.5 * GB), ("luid_placeholder", 0.0)]
        return [("luid_compute", 19.0 * GB), ("luid_placeholder", 1.3 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() == 30.5 * GB + 19.0 * GB


def test_unified_used_selects_on_dedicated_not_the_sum(monkeypatch):
    # A display adapter reports 0 dedicated while holding gigabytes of shared.
    # Selecting on the sum would see two candidates and either bail or add a
    # foreign adapter's bytes; selecting on dedicated isolates the compute device.
    def fake(counter = "Dedicated Usage"):
        if counter == "Dedicated Usage":
            return [("luid_compute", 1.2 * GB), ("luid_display", 0.0)]
        return [("luid_compute", 0.15 * GB), ("luid_display", 1.3 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() == 1.2 * GB + 0.15 * GB


def test_unified_used_declines_when_the_compute_adapter_is_ambiguous(monkeypatch):
    # Two adapters above the floor: no key says which is visible, so report None
    # rather than pick one, matching _rocm_windows_aggregate_used_bytes.
    def two(counter = "Dedicated Usage"):
        return [("luid_a", 1.2 * GB), ("luid_b", 2.4 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", two)
    assert hw._rocm_windows_unified_used_bytes() is None

    # Every adapter below the floor: nothing to stand on.
    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": [("luid_a", 10 * MiB)],
    )
    assert hw._rocm_windows_unified_used_bytes() is None

    # Counter unavailable entirely.
    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": None,
    )
    assert hw._rocm_windows_unified_used_bytes() is None


def test_unified_used_declines_rather_than_falling_back_to_dedicated_only(monkeypatch):
    # Dedicated-only is correct BELOW the carve-out and wrong above it, and nothing
    # here knows which side a reading is on: on the measured host 30.5 GiB is both a
    # legitimate below-carve-out figure and a saturated one. So a failed Shared query
    # declines instead of degrading to dedicated, which would overstate free.
    def fake(counter = "Dedicated Usage"):
        return [("luid_compute", 1.2 * GB)] if counter == "Dedicated Usage" else None

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() is None


def test_match_adapter_reports_unknown_when_hidden_high_use_adapter_survives_filter():
    # Idle 8 GiB card (10 MiB noise) beside a hidden 48 GiB card at 40 GiB: the
    # 40 GiB can't fit the 8 GiB device, so clamping there would fabricate. Unknown.
    assert hw._match_adapter_used_to_devices([40 * GB, 10 * MiB], [8 * GB]) == [None]
    # Order of the counters must not matter.
    assert hw._match_adapter_used_to_devices([10 * MiB, 40 * GB], [8 * GB]) == [None]


def test_match_adapter_reports_unknown_for_placeholder_fallback():
    # Every counter below the 64 MiB floor plus a placeholder: no LUID-to-ordinal
    # mapping tells placeholder from idle GPU, so report unknown, not fabricate.
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
    # 8 GiB card at 7 GiB beside a 48 GiB card at 5 GiB: the bigger usage still fits
    # the smaller card, so both pairings are feasible -> unknown.
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
    # A survivor that merely *fits* a visible card must not be pinned onto it. Two
    # cards (48/8 GiB) at 40 GiB / 10 MiB beside a hidden 6 GiB adapter: the 6 GiB
    # fits the idle 8 GiB card but isn't forced -> Unknown; only 40 GiB is forced.
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


def test_unified_used_declines_when_the_shared_query_fails(monkeypatch):
    # Past the carve-out the overflow lives entirely in Shared, so treating a
    # FAILED shared query as zero reports the measured 48 GiB case as 30.5 and
    # overstates free by 19 GiB. Nothing here knows where the carve-out sits, so
    # a failed query has to decline rather than guess.
    def fake(counter = "Dedicated Usage"):
        return [("luid_compute", 30.5 * GB)] if counter == "Dedicated Usage" else None

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() is None


def test_unified_used_keeps_a_successful_query_that_omits_the_luid(monkeypatch):
    # A query that SUCCEEDS but has no row for this adapter is a real zero, not a
    # failure, and must stay usable.
    def fake(counter = "Dedicated Usage"):
        if counter == "Dedicated Usage":
            return [("luid_compute", 1.2 * GB)]
        return [("luid_other", 3.0 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() == 1.2 * GB


def test_per_device_vram_uses_the_snapshot_it_is_given(monkeypatch):
    # The caller validates a snapshot's cardinality before trusting the mapping.
    # Re-sampling here would apply that check to a different sample than
    # attribution runs on, and costs a second ~1.3 s PowerShell call.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": (_ for _ in ()).throw(
            AssertionError("must not re-sample when a snapshot was passed in")
        ),
    )
    monkeypatch.setattr(
        hw,
        "_torch_get_device_module",
        lambda: (
            types.SimpleNamespace(
                get_device_properties = lambda _o: types.SimpleNamespace(
                    name = "APU", total_memory = 89 * GB
                )
            ),
            None,
        ),
    )

    devices, _agg = hw._rocm_windows_per_device_vram([0], [("a", 1.2 * GB)])

    assert devices[0]["used_gb"] == round(1.2 * GB / (1024**3), 2)


# ----------------------------------------------------------------------------- #
# The per-device total on a unified-memory APU
#
# _rocm_windows_per_device_vram took props.total_memory verbatim, which on an APU
# is the dedicated carve-out rather than what torch can use, so a 128 GiB Strix
# Halo budgeted as roughly 8. Every other path already corrects this through
# _rocm_props_total_is_carve_out; this one did not.
# ----------------------------------------------------------------------------- #
APU = ("AMD Radeon(TM) 8060S Graphics", 8 * GB)
APU_ADAPTERS = [("luid_0x00000000_0x0000c001_phys_0", 2 * GB)]


def test_windows_apu_total_is_the_driver_pool_not_the_carve_out(win_rocm, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    torch = _fake_torch([APU])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (96 * GB, 128 * GB))
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(APU_ADAPTERS))
    )

    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda props: True)
    monkeypatch.setattr(hw, "_rocm_windows_unified_used_bytes", lambda d = None: 12.0 * GB)
    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 128.0
    # Dedicated Usage alone saturates at the carve-out, so a widened total takes
    # the Dedicated+Shared sum instead. Pairing the carve-out reading with a pool
    # total would hand back the shared pool as free while a model sits in it.
    assert devices[0]["used_gb"] == 12.0
    assert aggregate == 12.0


def test_an_unsettled_classifier_does_not_cost_a_card_its_occupancy(win_rocm, monkeypatch):
    """The classifier answers "carve-out" for a discrete card too whenever the
    runtime leaves it unsettled (no integrated flag, HIP below 6.2). The driver
    total comes back equal there, so nothing widened and nothing is unknown."""
    torch = _fake_torch(DEVICES, free_equals_total = True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]
    assert devices[0]["used_gb"] == pytest.approx(40.0, abs = 0.01)


def test_a_discrete_card_must_not_pay_a_context_for_its_total(win_rocm, monkeypatch):
    """Only an APU pays mem_get_info for its total; a poll that attaches a
    context on a discrete card never gives the memory back."""
    torch = _fake_torch(DEVICES, free_equals_total = True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i: pytest.fail("a discrete card must not be asked")
    )
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: False)
    monkeypatch.setattr(hw.subprocess, "run", _subprocess_run(adapter_output = "__NONE__\n"))

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]


# ----------------------------------------------------------------------------- #
# A widened APU total must not follow the APU into the capacity matching
#
# _match_adapter_used_to_devices ranks the Dedicated Usage counters against
# device capacity, and that counter measures the DEDICATED segment, whose ceiling
# is the carve-out and never the pool. Ranked against a widened total the APU
# outranks every discrete card, the threshold that forces a pairing rises, and so
# does the ceiling of the impossible-counter check. The total_is_pool guard runs
# after the matching and nulls only the widened device, so it cannot reach that.
# ----------------------------------------------------------------------------- #
MIXED = [("AMD Radeon(TM) 8060S Graphics", 8 * GB), ("AMD Radeon RX 7900 XTX", 24 * GB)]
MIXED_ADAPTERS = [
    ("luid_0x00000000_0x0000c001_phys_0", 2.0 * GB),  # APU carve-out
    ("luid_0x00000000_0x0000d1e2_phys_0", 10.0 * GB),  # 7900 XTX, model loaded
    ("luid_0x00000000_0x0000f001_phys_0", 3 * MiB),  # Basic Render Driver
]


def _mixed_host(monkeypatch):
    """APU (8 GiB carve-out, 128 GiB pool) at ordinal 0, discrete 24 GiB at 1."""
    torch = _fake_torch(MIXED, free_equals_total = True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    # _Props carries no is_integrated, so the classifier stays patched; keying it
    # on the total is what lets one fake host hold an APU and a discrete card.
    monkeypatch.setattr(
        hw, "_rocm_props_total_is_carve_out", lambda props: props.total_memory == 8 * GB
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda i: (96 * GB, 128 * GB)
        if i == 0
        else pytest.fail("a discrete card must not be asked"),
    )
    return torch


def test_widened_apu_total_does_not_cost_the_discrete_card_its_usage(win_rocm, monkeypatch):
    """Hidden-adapter branch. Against carve-out totals [8, 24] the 10 GiB counter
    exceeds 8 GiB, so capacity forces it onto the 24 GiB card. Against a 128 GiB
    pool the APU takes rank 0, 10 GiB no longer exceeds the next capacity, and
    the discrete card reads Unknown."""
    _mixed_host(monkeypatch)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(MIXED_ADAPTERS))
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["total_gb"] for d in devices] == [128.0, 24.0]  # displayed total still widened
    assert devices[0]["used_gb"] is None  # Dedicated Usage is not the pool's numerator
    assert devices[1]["used_gb"] == pytest.approx(10.0, abs = 0.01)
    assert aggregate is None  # one member of the visible set is unknown


def test_widened_apu_total_does_not_make_every_pairing_ambiguous(win_rocm, monkeypatch):
    """Equal-length branch, no placeholder counter. The ambiguity check asks
    whether the two usages could be swapped without breaking capacity: 10 GiB
    does not fit the 8 GiB carve-out, so they could not, but it fits a 128 GiB
    slot, so every ranking on a mixed host becomes swappable."""
    _mixed_host(monkeypatch)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(MIXED_ADAPTERS[:2]))
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert devices[1]["used_gb"] == pytest.approx(10.0, abs = 0.01)
    assert devices[0]["used_gb"] is None


def test_widened_apu_total_does_not_admit_an_impossible_counter(win_rocm, monkeypatch):
    """A 50 GiB counter fits no visible card, so the list is not the visible set
    and every device must read Unknown. Against a 128 GiB slot it fits, shifts
    the others down a rank and fabricates 30 GiB on the 48 GiB card: a wrong
    reading rather than an unknown one, which the guard does not cover."""
    torch = _fake_torch(
        [
            ("AMD Radeon(TM) 8060S Graphics", 8 * GB),
            ("AMD Radeon PRO W7900", 48 * GB),
            ("AMD Radeon PRO W7800", 24 * GB),
        ],
        free_equals_total = True,
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        hw, "_rocm_props_total_is_carve_out", lambda props: props.total_memory == 8 * GB
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (96 * GB, 128 * GB))
    adapters = [
        ("luid_0x00000000_0x0000a001_phys_0", 50.0 * GB),  # hidden card, no visible home
        ("luid_0x00000000_0x0000d1e2_phys_0", 30.0 * GB),
        ("luid_0x00000000_0x0000e34a_phys_0", 1.0 * GB),
        ("luid_0x00000000_0x0000f001_phys_0", 3 * MiB),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1, 2])
    assert [d["used_gb"] for d in devices] == [None, None, None]
    assert [d["total_gb"] for d in devices] == [128.0, 48.0, 24.0]
    assert aggregate is None


def test_a_driver_total_below_the_carve_out_is_not_adopted(win_rocm, monkeypatch):
    """The classifier says carve-out for a discrete card on an unsettled runtime
    too, and this path carries a used alongside the total. A driver total below
    props.total_memory there reports past 100% utilization and, through
    free = max(total - used, 0), zero free on a card that is mostly empty."""
    torch = _fake_torch(DEVICES, free_equals_total = True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    # Driver under-reports the 48 GiB card.
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i: (0, 36 * GB) if i == 0 else (0, 8 * GB)
    )
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]
    assert devices[0]["used_gb"] == pytest.approx(40.0, abs = 0.01)
    assert all(d["used_gb"] <= d["total_gb"] for d in devices if d["used_gb"] is not None)


def test_a_failing_carve_out_probe_keeps_the_device(win_rocm, monkeypatch):
    """The correction is a probe, and it is the first thing this path asks the
    classifier. A probe that throws must cost the device its correction, not its
    place in the visible set: dropping it shows the System tab no GPU at all."""
    torch = _fake_torch(DEVICES, free_equals_total = True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def _boom(props):
        raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", _boom)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]
    assert devices[0]["used_gb"] == pytest.approx(40.0, abs = 0.01)


def test_the_inventory_path_also_refuses_to_shrink_a_total(monkeypatch):
    """The sibling correction this one is modelled on. It publishes no used, so a
    shrink cannot break the used <= total invariant there, but an understated
    total still hides models the device can hold, which is the failure the
    classifier exists to prevent."""
    monkeypatch.setattr(hw, "IS_ROCM", True)
    torch = _fake_torch(DEVICES)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    # Under-reports the 48 GiB card, agrees on the 8 GiB one.
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i: (0, 36 * GB) if i == 0 else (0, 8 * GB)
    )

    devices = hw._torch_get_device_inventory([0, 1])
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]
    assert all(d["used_gb"] is None for d in devices)


def _apu_host(monkeypatch, unified_used = 12.0 * GB):
    """A lone 8 GiB carve-out / 128 GiB pool APU.

    ``unified_used`` stands in for _rocm_windows_unified_used_bytes, which sums
    Dedicated and Shared. Patched rather than driven through the counters because
    what is under test here is that a widened device takes that figure, not the
    counter plumbing, which #9362's own tests already cover. ``None`` simulates it
    declining.
    """
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    torch = _fake_torch([APU])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (96 * GB, 128 * GB))
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda props: True)
    monkeypatch.setattr(hw, "_rocm_windows_unified_used_bytes", lambda d = None: unified_used)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(APU_ADAPTERS))
    )


def test_widened_total_survives_get_gpu_utilization(win_rocm, monkeypatch):
    """The helper is not what the user sees. The corrected total has to reach the
    payload, and the unknown occupancy has to stay unknown through the legacy
    primary-device mirror rather than being filled in with a zero."""
    _apu_host(monkeypatch)

    result = hw.get_gpu_utilization()
    assert result["devices"][0]["vram_total_gb"] == 128.0
    assert result["devices"][0]["vram_used_gb"] == 12.0
    assert result["devices"][0]["vram_utilization_pct"] == pytest.approx(9.4, abs = 0.1)
    assert result["vram_total_gb"] == 128.0  # legacy mirror carries it too
    assert result["vram_used_gb"] == 12.0


def test_the_system_tab_does_not_invent_free_vram_on_an_apu(win_rocm, monkeypatch):
    """The failure this guards is a 128 GiB card reporting 128 GiB free while a
    model sits in the shared pool: free is total minus used, and used is unknown,
    so free must stay unknown rather than defaulting used to zero."""
    _apu_host(monkeypatch)

    devices = hw.get_visible_gpu_utilization()["devices"]
    assert devices[0]["vram_total_gb"] == 128.0
    assert devices[0]["vram_used_gb"] == 12.0
    assert devices[0]["vram_utilization_pct"] == pytest.approx(9.4, abs = 0.1)


def test_a_declining_unified_read_leaves_the_apu_unknown(win_rocm, monkeypatch):
    """When the sum cannot be established the carve-out reading must not be left
    standing under a pool-sized total: that is the reading that reports a loaded
    128 GiB card as almost entirely free."""
    _apu_host(monkeypatch, unified_used = None)

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 128.0
    assert devices[0]["used_gb"] is None
    assert aggregate is None


def test_shared_usage_needs_a_positively_unified_part(win_rocm, monkeypatch):
    """_rocm_props_total_is_carve_out fails open, so a discrete card on an
    unsettled runtime can widen. Shared Usage is host memory that its
    props.total_memory never counted, so the sum must not become its numerator:
    the stricter classifier has to gate it, exactly as get_gpu_memory_info does."""
    _apu_host(monkeypatch, unified_used = 90.0 * GB)
    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda props: False)

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    # Not merely "the sum is not used": without positive UMA the driver total is
    # never asked for, so there is no widening to pair a numerator with either.
    assert devices[0]["total_gb"] == 8.0
    assert devices[0]["used_gb"] == 2.0  # the Dedicated counter, not the sum
    assert aggregate == 2.0


def test_a_nonzero_sub_threshold_row_declines_the_unified_sum(win_rocm, monkeypatch):
    """The helper picks the lone counter above the noise floor. With only the APU
    visible but a hidden GPU active, that counter is the hidden one and the APU is
    the small row, so its usage would be published as the APU's. The matcher
    declines this shape by requiring every dropped counter to be an exact zero."""
    _apu_host(monkeypatch, unified_used = 30.0 * GB)
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        _subprocess_run(
            adapter_output = _adapter_output(
                [
                    ("luid_0x00000000_0x0000c001_phys_0", 10 * MiB),  # the visible APU, idle
                    ("luid_0x00000000_0x0000d1e2_phys_0", 30.0 * GB),  # a hidden GPU, loaded
                ]
            )
        ),
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["used_gb"] is None
    assert aggregate is None


def test_a_negative_counter_never_publishes_negative_usage(win_rocm, monkeypatch):
    """The helper declines a negative reading at source; this is the consumer end
    of the same guarantee, driven with the helper stubbed so the clamp that
    publishes the number is what gets tested rather than the decline above it."""
    _apu_host(monkeypatch, unified_used = -5.0 * GB)

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["used_gb"] == 0.0
    assert aggregate == 0.0


def test_an_apu_beside_a_discrete_card_probes_only_the_apu(win_rocm, monkeypatch):
    """The mixed host, which is the one configuration nobody has hardware for.

    An APU whose props.total_memory IS the 32 GiB carve-out, beside a 48 GiB
    discrete card holding 40 GiB. Three things have to hold at once, and only
    the middle one is covered elsewhere:

      * the APU's total widens to the pool, 32 -> 89.46
      * the discrete card KEEPS its capacity-forced 40 GiB; widening one device
        must not rerank the others out of a reading
      * exactly one mem_get_info call, on the APU. The discrete card must not be
        asked, because that call attaches a permanent ~612 MiB context.
    """
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    torch = _fake_torch(
        [
            ("AMD Radeon(TM) 8060S Graphics", 32 * GB),  # carve-out, not the pool
            ("AMD Radeon PRO W7900", 48 * GB),
        ]
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    probed = []
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda i: (probed.append(i), (0, int(89.465 * GB) if i == 0 else 48 * GB))[1],
    )
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    monkeypatch.setattr(
        hw,
        "_rocm_props_are_positively_unified",
        lambda props: props.total_memory == 32 * GB,
    )
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        _subprocess_run(
            adapter_output = _adapter_output(
                [
                    ("luid_0x00000000_0x0001532a_phys_0", 31.58 * GB),  # APU, saturated
                    ("luid_0x00000000_0x0000d1e2_phys_0", 40.0 * GB),  # the discrete card
                    ("luid_0x00000000_0x00017034_phys_0", 0.0),  # placeholder
                ]
            )
        ),
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert devices[0]["total_gb"] == 89.46  # widened
    assert devices[1]["used_gb"] == 40.0  # and the discrete card kept its reading
    assert probed == [0], f"expected only the APU to be probed, got {probed}"


def test_the_poll_does_not_probe_an_unclassified_discrete_card(win_rocm, monkeypatch):
    """mem_get_info attaches a primary HIP context worth ~612 MiB that is never
    released, and main reaches this function without ever calling it. The
    carve-out classifier fails open for an unclassified discrete card, so gating
    the probe on it would take that off every discrete GPU on the host on every
    telemetry poll, for a total the device then does not use."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    torch = _fake_torch(DEVICES)
    monkeypatch.setitem(sys.modules, "torch", torch)
    probed = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (probed.append(i), (0, 99 * GB))[1])
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)  # fails open
    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda props: False)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert probed == [], "a discrete card was asked for a context it does not need"
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]  # props totals, unwidened


def test_a_failed_driver_probe_keeps_the_apu_on_the_dedicated_counter(win_rocm, monkeypatch):
    """Being a UMA part says what the device is, not what scope its total has. A
    confirmed APU whose mem_get_info probe fails keeps a carve-out-sized total,
    and Dedicated Usage is the correct numerator for that one. Summing Shared
    into it would clamp to a fabricated 100%."""
    _apu_host(monkeypatch, unified_used = 40.0 * GB)
    torch = sys.modules["torch"]
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda i: (_ for _ in ()).throw(RuntimeError("hipMemGetInfo failed")),
    )

    devices, _ = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 8.0  # the carve-out, unwidened
    assert devices[0]["used_gb"] == 2.0  # the Dedicated counter, not the sum
    assert devices[0]["used_gb"] <= devices[0]["total_gb"]


def test_the_measured_strix_halo_at_64_gib_held(win_rocm, monkeypatch):
    """Real numbers off a Windows gfx1151, driver 32.0.21041.1000, 32 GiB carve-out.

    Holding 64 GiB, Dedicated Usage reads 31.637 GB and Shared 33.887: Dedicated
    tracks the allocation one for one until it pins at the carve-out, then stays
    flat while Shared absorbs every further GiB. props.total_memory is already
    89.465 GB, so nothing widens and total_is_pool never fires.

    Pre-PR this reported 31.64 used against an 89.46 total, i.e. 57.82 GB free
    with 64 GiB resident. Overstating free by 33.88 GB is the direction that
    OOMs, and it is the case this pairing exists to fix.
    """
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    torch = _fake_torch([("AMD Radeon(TM) 8060S Graphics", int(89.465 * GB))])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (0, int(89.465 * GB)))
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda props: True)
    monkeypatch.setattr(
        hw,
        "_rocm_windows_unified_used_bytes",
        lambda dedicated = None: (31.637 + 33.887) * GB,
    )
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        _subprocess_run(
            adapter_output = _adapter_output(
                [
                    (
                        "luid_0x00000000_0x0001532a_phys_0",
                        31.637 * GB,
                    ),  # saturated at the carve-out
                    ("luid_0x00000000_0x00017034_phys_0", 0.0),
                    ("luid_0x00000000_0x00017099_phys_0", 0.0),
                ]
            )
        ),
    )

    devices, _ = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 89.46
    assert devices[0]["used_gb"] == 65.52  # not the 31.64 Dedicated alone reports
    assert devices[0]["total_gb"] - devices[0]["used_gb"] < 24.0


def test_a_confirmed_apu_takes_the_unified_sum_even_unwidened(win_rocm, monkeypatch):
    """The measured gfx1151 has props.total_memory already spanning the pool, so
    nothing widens and total_is_pool stays false. The total is pool-scoped all
    the same, and Dedicated alone plateaus at the carve-out under it, which is
    the reading that reports a loaded APU as mostly free."""
    _apu_host(monkeypatch, unified_used = 4.0 * GB)
    torch = sys.modules["torch"]
    # Both totals agree: the driver has nothing wider to offer.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (0, APU[1]))

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == round(APU[1] / GB, 2)
    # 4.0, not the 2.0 the Dedicated counter alone reports for this adapter.
    assert devices[0]["used_gb"] == 4.0
    assert aggregate == 4.0


def test_the_unified_sum_is_clamped_to_the_widened_total(win_rocm, monkeypatch):
    """Dedicated plus Shared counts driver and desktop allocations too, so it can
    exceed torch's pool. Every other reading is clamped on its way through the
    matcher; this one bypasses it and the payload derives free as total minus
    used, so an unclamped sum publishes negative free."""
    _apu_host(monkeypatch, unified_used = 140.0 * GB)

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["used_gb"] == 128.0
    assert aggregate == 128.0


def test_the_dedicated_snapshot_is_not_sampled_twice(win_rocm, monkeypatch):
    """Each counter query is an out-of-process PowerShell call, and re-sampling
    would also answer from a different instant than the attribution above it."""
    _apu_host(monkeypatch)
    seen = []
    monkeypatch.setattr(
        hw,
        "_rocm_windows_unified_used_bytes",
        lambda dedicated = None: (seen.append(dedicated), 12.0 * GB)[1],
    )

    hw._rocm_windows_per_device_vram([0])
    assert seen and seen[0], "the caller's snapshot was not handed over"


def test_an_unavailable_dedicated_snapshot_is_not_requeried(win_rocm, monkeypatch):
    """When the caller's own Dedicated query came back unavailable, handing None
    to the helper reads as "not supplied" and runs the query a second time, on the
    polled telemetry path, to arrive at the same None."""
    _apu_host(monkeypatch)
    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(
        hw,
        "_rocm_windows_unified_used_bytes",
        lambda dedicated = None: (seen.append(dedicated), None)[1],
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert seen == [], "an unavailable snapshot was handed over and re-queried"
    assert devices[0]["used_gb"] is None
    assert aggregate is None


def test_a_negative_counter_reading_is_declined(win_rocm, monkeypatch):
    """A negative cooked counter is a broken reading, not low usage. The caller
    clamps the upper bound only and the payload derives free as total minus used,
    so a negative sum would publish a negative used and a free above the total."""

    def counters(counter = "Dedicated Usage"):
        if counter == "Dedicated Usage":
            return [("luid_0x00000000_0x0000abcd_phys_0", 2.0 * GB)]
        return [("luid_0x00000000_0x0000abcd_phys_0", -1.0 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", counters)
    assert hw._rocm_windows_unified_used_bytes() is None


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
def test_the_windows_path_is_inert_off_windows(win_rocm, monkeypatch, system):
    """Linux, macOS and WSL (which reports Linux) never reach this path, so no
    change to it can reach them."""
    monkeypatch.setattr(hw.platform, "system", lambda: system)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(DEVICES))
    monkeypatch.setattr(
        hw, "_rocm_props_total_is_carve_out", lambda props: pytest.fail("not on this platform")
    )

    assert hw._rocm_windows_per_device_vram([0, 1]) == ([], None)
