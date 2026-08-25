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

The last section covers the DirectX LUID join layered on top of that counter,
which resolves the single-GPU host capacity ranking alone can never attribute.
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
    """Build a fake `torch` module. devices: list of (name, total_bytes[, gfx])."""
    dev = list(devices)

    class _Props:
        def __init__(self, name, total, gfx):
            self.name = name
            self.total_memory = total
            self.gcnArchName = gfx

    def get_device_properties(i):
        name, total, *arch = dev[i]
        return _Props(name, total, arch[0] if arch else "")

    def mem_get_info(i):
        total = dev[i][1]
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
    # Capacity-ranking path by default: without this a Windows dev box reads its
    # own registry here and the LUID join answers instead. The LUID tests below
    # opt in with their own map.
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: {})
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
# DirectX LUID join
#
# Capacity ranking only emits a value a pairing FORCES, and one visible GPU
# forces nothing -- there is no smaller card for its usage to exceed -- so every
# single-GPU AMD Windows host read Unknown, the common case. The counters name
# each instance after the adapter LUID and DirectX records the same LUID with
# torch's props.name and gcnArchName, so they join on identity instead.
# ----------------------------------------------------------------------------- #
SOLO_DEVICE = [("AMD Radeon RX 9060 XT", 16 * GB, "gfx1200")]
# One dGPU at 3 GiB beside the Basic Render Driver and a second idle placeholder:
# three counters for one visible card, straight off the reporting host.
SOLO_ADAPTERS = [
    ("luid_0x00000000_0x00015369_phys_0", 3 * GB),
    ("luid_0x00000000_0x000183fe_phys_0", 0.0),
    ("luid_0x00000000_0x0001842f_phys_0", 0.0),
]
SOLO_REGISTRY = {0x15369: {"name": "AMD Radeon RX 9060 XT", "gfx": "gfx1200"}}


@pytest.fixture
def win_rocm_solo(win_rocm, monkeypatch):
    """Windows ROCm host with a single visible GPU and a readable DirectX map."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(SOLO_DEVICE, free_equals_total = True))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(SOLO_ADAPTERS))
    )
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: dict(SOLO_REGISTRY))
    return monkeypatch


def test_parse_adapter_luid():
    # DirectX stores one 64-bit AdapterLuid; the counter names it as two halves.
    assert hw._parse_adapter_luid("luid_0x00000000_0x00015369_phys_0") == 0x15369
    assert hw._parse_adapter_luid("luid_0x0000000A_0x00015369_phys_0") == (0xA << 32) | 0x15369
    assert hw._parse_adapter_luid("LUID_0X00000000_0X00015369_PHYS_0") == 0x15369
    assert hw._parse_adapter_luid("Total") is None
    assert hw._parse_adapter_luid("luid_zz_0x1_phys_0") is None


def test_normalize_adapter_name():
    # The two sides spell one card differently; only the marks and spacing move.
    assert hw._normalize_adapter_name("AMD Radeon(TM) 780M Graphics") == hw._normalize_adapter_name(
        "AMD Radeon 780M Graphics"
    )
    assert hw._normalize_adapter_name("AMD Radeon™ RX 9070") == hw._normalize_adapter_name(
        "amd radeon rx 9070"
    )
    # Two models stay two models.
    assert hw._normalize_adapter_name("AMD Radeon RX 9070") != hw._normalize_adapter_name(
        "AMD Radeon RX 9070 XT"
    )


def test_parse_adapter_family_gfx():
    # The driver writes the family, torch the target plus its feature suffixes.
    assert hw._parse_adapter_family_gfx("AMD_NAVI44:gfx1200") == "gfx1200"
    assert hw._parse_adapter_family_gfx("gfx1201:sramecc-:xnack-") == "gfx1201"
    assert hw._parse_adapter_family_gfx("GFX1103") == "gfx1103"
    assert hw._parse_adapter_family_gfx("AMD_NAVI44") == ""
    assert hw._parse_adapter_family_gfx("") == ""


def test_single_gpu_reports_used_instead_of_unknown(win_rocm_solo):
    """The regression: one visible GPU plus placeholder counters read Unknown."""
    result = hw.get_visible_gpu_utilization()
    (device,) = result["devices"]
    assert device["vram_used_gb"] == pytest.approx(3.0, abs = 0.01)
    assert device["vram_total_gb"] == 16.0
    assert device["vram_utilization_pct"] == pytest.approx(18.8, abs = 0.1)
    # The System tab tile reads the aggregate, which needed len(counters) == 1.
    assert result["vram_used_gb_aggregate"] == pytest.approx(3.0, abs = 0.01)


def test_single_gpu_train_page_reports_used(win_rocm_solo):
    """Same figure on get_gpu_utilization, the Train page's GPU Monitor."""
    (device,) = hw.get_gpu_utilization()["devices"]
    assert device["vram_used_gb"] == pytest.approx(3.0, abs = 0.01)
    assert device["gpu_utilization_pct"] == 12.0  # 3D-engine counter, unchanged


def test_luid_join_beats_a_busy_foreign_adapter(win_rocm, monkeypatch):
    """An idle AMD card beside a busy foreign GPU reports the AMD card's own
    usage. Capacity ranking has no vendor key here and declines outright."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(SOLO_DEVICE, free_equals_total = True))
    adapters = [
        ("luid_0x00000000_0x00015369_phys_0", 0.4 * GB),  # visible AMD card, idle
        ("luid_0x00000000_0x00099999_phys_0", 9 * GB),  # NVIDIA/iGPU, not ours
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: dict(SOLO_REGISTRY))

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["used_gb"] == pytest.approx(0.4, abs = 0.01)
    assert aggregate == pytest.approx(0.4, abs = 0.01)


def test_identical_cards_keep_aggregate_but_not_per_device(win_rocm, monkeypatch):
    """Two of the same card share one Description AND one arch, so nothing says
    which counter is which ordinal. The sum does not depend on the pairing."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch([("AMD Radeon RX 7900 XTX", 24 * GB, "gfx1100")] * 2, free_equals_total = True),
    )
    adapters = [
        ("luid_0x00000000_0x0000aaaa_phys_0", 10 * GB),
        ("luid_0x00000000_0x0000bbbb_phys_0", 4 * GB),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
    record = {"name": "AMD Radeon RX 7900 XTX", "gfx": "gfx1100"}
    monkeypatch.setattr(
        hw, "_windows_amd_adapter_records_by_luid", lambda: {0xAAAA: record, 0xBBBB: record}
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [None, None]
    assert aggregate == pytest.approx(14.0, abs = 0.01)


# ── The mixed hosts nobody here owns hardware for ────────────────────────────
IGPU_DGPU_DEVICES = [
    ("AMD Radeon(TM) 780M Graphics", 2 * GB, "gfx1103"),
    ("AMD Radeon RX 9070 XT", 16 * GB, "gfx1201"),
]
IGPU_DGPU_ADAPTERS = [
    ("luid_0x00000000_0x0000c001_phys_0", 0.5 * GB),  # iGPU, driving the display
    ("luid_0x00000000_0x0000d002_phys_0", 1.2 * GB),  # dGPU, small model loaded
]


def test_igpu_and_dgpu_each_report_their_own(win_rocm, monkeypatch):
    """Both visible: two different cards, two counters, and identity says which
    is which. Capacity ranking cannot: the dGPU's 1.2 GiB would also fit the
    2 GiB iGPU, so the two are swappable and it reports both unknown."""
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(IGPU_DGPU_DEVICES, free_equals_total = True)
    )
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IGPU_DGPU_ADAPTERS))
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xC001: {"name": "AMD Radeon(TM) 780M Graphics", "gfx": "gfx1103"},
            0xD002: {"name": "AMD Radeon RX 9070 XT", "gfx": "gfx1201"},
        },
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [
        pytest.approx(0.5, abs = 0.01),
        pytest.approx(1.2, abs = 0.01),
    ]
    assert aggregate == pytest.approx(1.7, abs = 0.01)


def test_a_name_the_two_sides_spell_differently_still_joins(win_rocm, monkeypatch):
    """DirectX takes the Description from the driver INF and HIP from the ASIC
    record, so an iGPU reaches the join under two spellings of one card."""
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(IGPU_DGPU_DEVICES, free_equals_total = True)
    )
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IGPU_DGPU_ADAPTERS))
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xC001: {"name": "AMD Radeon 780M Graphics", "gfx": "gfx1103"},  # no (TM)
            0xD002: {"name": "AMD Radeon(R) RX 9070 XT", "gfx": "gfx1201"},  # (R), not (TM)
        },
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [
        pytest.approx(0.5, abs = 0.01),
        pytest.approx(1.2, abs = 0.01),
    ]


def test_the_arch_answers_when_the_names_do_not(win_rocm, monkeypatch):
    """A generic iGPU Description that normalizing cannot reconcile. The gfx
    target both sides carry separates an iGPU from the dGPU beside it."""
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(IGPU_DGPU_DEVICES, free_equals_total = True)
    )
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IGPU_DGPU_ADAPTERS))
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xC001: {"name": "AMD Radeon(TM) Graphics", "gfx": "gfx1103"},
            0xD002: {"name": "Radeon RX 9070 XT Series", "gfx": "gfx1201"},
        },
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [
        pytest.approx(0.5, abs = 0.01),
        pytest.approx(1.2, abs = 0.01),
    ]
    assert aggregate == pytest.approx(1.7, abs = 0.01)


def test_a_partial_name_pass_still_lets_the_arch_finish_the_job(win_rocm, monkeypatch):
    """Three cards, one uniquely named and two sharing a name but not an arch.

    The name pass places the unique card and leaves the pair unknown, which is a
    real result and used to end the search. It should not: the arch separates
    all three, and the pass that places more devices is the one to keep.
    """
    devices_spec = [
        ("AMD Radeon PRO W7900", 48 * GB, "gfx1100"),
        ("AMD Radeon RX 9070", 16 * GB, "gfx1201"),
        ("AMD Radeon RX 9070", 16 * GB, "gfx1200"),  # same name, different arch
    ]
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(devices_spec, free_equals_total = True))
    adapters = [
        ("luid_0x00000000_0x0000a001_phys_0", 30.0 * GB),
        ("luid_0x00000000_0x0000b002_phys_0", 10.0 * GB),
        ("luid_0x00000000_0x0000c003_phys_0", 5.0 * GB),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xA001: {"name": "AMD Radeon PRO W7900", "gfx": "gfx1100"},
            0xB002: {"name": "AMD Radeon RX 9070", "gfx": "gfx1201"},
            0xC003: {"name": "AMD Radeon RX 9070", "gfx": "gfx1200"},
        },
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1, 2])
    assert [d["used_gb"] for d in devices] == [
        pytest.approx(30.0, abs = 0.01),
        pytest.approx(10.0, abs = 0.01),
        pytest.approx(5.0, abs = 0.01),
    ]
    assert aggregate == pytest.approx(45.0, abs = 0.01)


def test_the_arch_pass_needs_every_record_to_have_one(win_rocm, monkeypatch):
    """A driver too old to write AdapterFamily leaves a record the arch cannot
    key. Running the pass on the rest would let the adapters that do carry one
    stand in for the adapter that does not, so it does not run at all."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch([("AMD Radeon RX 9070", 16 * GB, "gfx1201")], free_equals_total = True),
    )
    adapters = [
        ("luid_0x00000000_0x0000c001_phys_0", 6 * GB),  # the visible card
        ("luid_0x00000000_0x0000d002_phys_0", 9 * GB),  # its hidden same-arch sibling
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
    # Neither Description matches what torch calls the card, so the name pass
    # declines and only the arch could answer -- and one record has no arch.
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xC001: {"name": "AMD Radeon RX 9070 Series"},
            0xD002: {"name": "AMD Radeon RX 9070 Series", "gfx": "gfx1201"},
        },
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["used_gb"] is None
    assert aggregate is None


def test_luid_join_declines_and_falls_back(win_rocm, monkeypatch):
    """Every decline hands the counters back to capacity ranking rather than
    inventing a figure."""
    dev_meta = [{"name": "AMD Radeon RX 9060 XT", "gfx": "gfx1200", "total_bytes": 16 * GB}]
    join = hw._match_adapter_used_by_luid

    # No registry at all (non-Windows, missing key, denied read, partial map).
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: {})
    assert join(SOLO_ADAPTERS, dev_meta) is None

    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: dict(SOLO_REGISTRY))
    # A visible card whose adapter has no record: its counter is unidentified.
    assert (
        join(
            SOLO_ADAPTERS,
            [{"name": "AMD Radeon RX 7800 XT", "gfx": "gfx1101", "total_bytes": 16 * GB}],
        )
        is None
    )
    # A same-named, same-arch adapter HIP does not enumerate: two counters, one ordinal.
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {**SOLO_REGISTRY, 0x77777: dict(SOLO_REGISTRY[0x15369])},
    )
    assert (
        join(
            [*SOLO_ADAPTERS, ("luid_0x00000000_0x00077777_phys_0", 2 * GB)],
            dev_meta,
        )
        is None
    )
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: dict(SOLO_REGISTRY))
    # A record outliving its hardware: usage above the card's own capacity.
    assert join([("luid_0x00000000_0x00015369_phys_0", 20 * GB)], dev_meta) is None


# ── Reading the registry itself ──────────────────────────────────────────────
def _fake_winreg(subkeys):
    """A `winreg` over ``{subkey: {value: data}}``. An Exception in place of a
    subkey's dict raises on open; one in place of a value raises on read."""

    class _Key:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    names = list(subkeys)
    mod = types.ModuleType("winreg")
    mod.HKEY_LOCAL_MACHINE = object()

    def open_key(root, path):
        if root is mod.HKEY_LOCAL_MACHINE:
            return _Key("")  # the DirectX key itself
        entry = subkeys[path]
        if isinstance(entry, Exception):
            raise entry
        return _Key(path)

    def query_value_ex(key, value):
        values = subkeys[key.name]
        if value not in values:
            raise FileNotFoundError(2, "value does not exist")
        data = values[value]
        if isinstance(data, Exception):
            raise data
        return (data, 0)

    mod.OpenKey = open_key
    mod.QueryInfoKey = lambda key: (len(names), 0, 0)
    mod.EnumKey = lambda key, index: names[index]
    mod.QueryValueEx = query_value_ex
    return mod


AMD_RECORD = {
    "VendorId": 0x1002,
    "AdapterLuid": 0x14CF5,
    "Description": "AMD Radeon RX 9060 XT",
    "AdapterFamily": "AMD_NAVI44:gfx1200",
}
BASIC_RENDER_RECORD = {
    "VendorId": 0x1414,
    "AdapterLuid": 0x17DEA,
    "Description": "Microsoft Basic Render Driver",
}


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    return monkeypatch


def test_registry_reads_the_amd_adapters_and_skips_the_rest(on_windows, monkeypatch):
    # ShaderCache is a real sibling of the adapter records and holds an
    # AdapterLuid of 0 -- it is not GUID-named, which is what excludes it.
    monkeypatch.setitem(
        sys.modules,
        "winreg",
        _fake_winreg(
            {
                "ShaderCache": {"AdapterLuid": 0},
                "{aaaaaaaa-0000-0000-0000-000000000000}": BASIC_RENDER_RECORD,
                "{bbbbbbbb-0000-0000-0000-000000000000}": AMD_RECORD,
            }
        ),
    )
    assert hw._windows_amd_adapter_records_by_luid() == {
        0x14CF5: {"name": "AMD Radeon RX 9060 XT", "gfx": "gfx1200"}
    }


def test_a_record_that_will_not_read_declines_the_whole_map(on_windows, monkeypatch):
    """A partial map is indistinguishable from a complete one at the join, which
    would then hand a visible card its hidden twin's counter."""
    denied = PermissionError(5, "access is denied")
    for broken in (
        denied,  # the subkey will not open
        {**AMD_RECORD, "Description": denied},  # an AMD adapter this cannot name
        {**AMD_RECORD, "Description": ""},  # ...or that names itself nothing
        {k: v for k, v in AMD_RECORD.items() if k != "AdapterLuid"},
        {k: v for k, v in AMD_RECORD.items() if k != "VendorId"},  # vendor unknown
    ):
        monkeypatch.setitem(
            sys.modules,
            "winreg",
            _fake_winreg(
                {
                    "{aaaaaaaa-0000-0000-0000-000000000000}": broken,
                    "{bbbbbbbb-0000-0000-0000-000000000000}": AMD_RECORD,
                }
            ),
        )
        assert hw._windows_amd_adapter_records_by_luid() == {}, broken


def test_a_driver_without_adapter_family_still_gives_the_name(on_windows, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "winreg",
        _fake_winreg(
            {
                "{bbbbbbbb-0000-0000-0000-000000000000}": {
                    k: v for k, v in AMD_RECORD.items() if k != "AdapterFamily"
                }
            }
        ),
    )
    assert hw._windows_amd_adapter_records_by_luid() == {0x14CF5: {"name": "AMD Radeon RX 9060 XT"}}


def test_registry_read_is_windows_only(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    assert hw._windows_amd_adapter_records_by_luid() == {}


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
# The capacity a counter is bounded by
#
# Dedicated Usage measures the dedicated segment, so the carve-out is its
# ceiling. #9314 widens total_bytes to the whole driver pool on a unified APU
# for DISPLAY, and every capacity comparison here has to keep using the
# carve-out: against a pool-sized ceiling a counter no visible card could hold
# still fits, and the join stops declining where it should.
# ----------------------------------------------------------------------------- #
def test_counter_capacity_prefers_the_dedicated_total():
    """dedicated_bytes wins when present, total_bytes when it is not, so this is
    right both before and after the widening lands."""
    assert hw._adapter_counter_capacity({"total_bytes": 8 * GB}) == float(8 * GB)
    assert hw._adapter_counter_capacity(
        {"total_bytes": 128 * GB, "dedicated_bytes": 8 * GB}
    ) == float(8 * GB)


def test_the_luid_join_bounds_counters_by_the_carve_out_not_the_pool():
    """An impossible counter must still be impossible after the total widens.
    9 GiB fits no 8 GiB carve-out, so the join has to decline; ranked against a
    128 GiB pool it fits and would be published on a card that never held it."""
    dev_meta = [
        {
            "name": "AMD Radeon(TM) 8060S Graphics",
            "total_bytes": 128 * GB,
            "dedicated_bytes": 8 * GB,
        }
    ]
    matched = hw._attribute_adapter_useds_by_key({"k": [9.0 * GB]}, {"k": [0]}, dev_meta)
    assert matched is None

    # Same device, a usage that does fit the carve-out: attributed normally.
    matched = hw._attribute_adapter_useds_by_key({"k": [2.0 * GB]}, {"k": [0]}, dev_meta)
    assert matched is not None
    assigned, aggregate = matched
    assert assigned[0] == pytest.approx(2.0 * GB)
    assert aggregate == pytest.approx(2.0 * GB)


# ----------------------------------------------------------------------------- #
# The join must not stop at a pass that answered nothing, or place bytes it
# cannot account for
# ----------------------------------------------------------------------------- #
def test_a_name_collision_falls_through_to_the_gfx_pass(win_rocm, monkeypatch):
    """Two cards sharing one Description but differing in arch. The name pass
    succeeds by cardinality yet names neither device, and returning there would
    keep both unknown when gfx separates them."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(
            [
                ("AMD Radeon Graphics", 24 * GB, "gfx1100"),
                ("AMD Radeon Graphics", 16 * GB, "gfx1200"),
            ],
            free_equals_total = True,
        ),
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xAAAA: {"name": "AMD Radeon Graphics", "gfx": "gfx1100"},
            0xBBBB: {"name": "AMD Radeon Graphics", "gfx": "gfx1200"},
        },
    )
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        _subprocess_run(
            adapter_output = _adapter_output(
                [
                    ("luid_0x00000000_0x0000aaaa_phys_0", 9.0 * GB),
                    ("luid_0x00000000_0x0000bbbb_phys_0", 3.0 * GB),
                ]
            )
        ),
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [9.0, 3.0]
    assert aggregate == 12.0


def test_the_measured_strix_halo_registry(win_rocm, monkeypatch):
    """The one real Windows AMD reading taken so far, pinned.

    Windows 11 Pro 26200, AMD Radeon(TM) 8060S Graphics, driver 32.0.21041.1000.
    Two things it establishes that no CI machine could: AdapterLuid comes back as
    an Int64 rather than the bytes a REG_BINARY would give, so int() is a no-op;
    and that driver writes no AdapterFamily at all, so the gfx pass is skipped
    and the NAME pass is what carries the join. DirectX and props.name agreed
    exactly there, which is why it lands.

    The counter set is also not a subset of the DirectX key: instance 94361 has
    1.273 GB shared and no adapter record at all, so anything assuming every
    instance resolves to a record would be wrong.
    """
    records = {86826: {"name": "AMD Radeon(TM) 8060S Graphics"}}  # no "gfx" key
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: records)
    adapters = [
        ("luid_0x00000000_0x00017099_phys_0", 0.0),  # no registry record
        ("luid_0x00000000_0x0001532a_phys_0", 0.947 * GB),  # the AMD adapter
        ("luid_0x00000000_0x00017034_phys_0", 0.0),  # Basic Render Driver
    ]
    dev_meta = [
        {
            "name": "AMD Radeon(TM) 8060S Graphics",
            "gfx": "gfx1151",
            "total_bytes": int(89.465 * GB),
            "dedicated_bytes": int(89.465 * GB),
        }
    ]

    result = hw._match_adapter_used_by_luid(adapters, dev_meta)
    assert result is not None, "the join declined on the one real reading we have"
    per_device, aggregate = result
    assert round(per_device[0] / GB, 3) == 0.947
    assert round(aggregate / GB, 3) == 0.947


def test_the_measured_strix_halo_needs_the_join_for_any_aggregate(win_rocm, monkeypatch):
    """Measured on the Windows gfx1151 while it held a model: THREE counter
    instances for ONE visible GPU.

    The AMD adapter plus two placeholders that never go away. So on this host the
    counter list is never as long as the visible set, the cardinality gate in
    ``_rocm_windows_aggregate_used_bytes`` fails closed permanently, and capacity
    ranking can never supply an aggregate no matter what is loaded. The LUID join
    is not a second opinion here, it is the only path to a figure at all.

    This is the shape a single-GPU Windows AMD host actually has, which is why
    #7072's reporter saw Unknown where the hardware was plainly busy.
    """
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(
            [("AMD Radeon(TM) 8060S Graphics", int(89.465 * GB), "gfx1151")],
            free_equals_total = True,
        ),
    )
    adapters = [
        ("luid_0x00000000_0x0001532a_phys_0", 31.681 * GB),  # the AMD adapter
        ("luid_0x00000000_0x00017034_phys_0", 0.0),  # placeholder, permanent
        ("luid_0x00000000_0x00017099_phys_0", 0.0),  # placeholder, permanent
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {86826: {"name": "AMD Radeon(TM) 8060S Graphics"}},  # no AdapterFamily
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["used_gb"] == pytest.approx(31.68, abs = 0.01)
    assert aggregate == pytest.approx(31.68, abs = 0.01)

    # The same host with the registry unreadable: everything below the join.
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: {})
    _, fallback_aggregate = hw._rocm_windows_per_device_vram([0])
    assert (
        fallback_aggregate is None
    ), "capacity ranking produced an aggregate the measured host cannot give it"


def test_the_join_declines_usage_it_cannot_place(win_rocm, monkeypatch):
    """A hidden AMD card carrying the visible device's torch name, while the
    visible card's own record is spelled differently. The visible device's
    counter sits under a key no device claims, so pairing the remaining one would
    report the hidden card's bytes as the visible card's."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch([("AMD Radeon RX 7900 XTX", 24 * GB)], free_equals_total = True),
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xAAAA: {
                "name": "AMD Radeon RX 7900 XTX 24GB"
            },  # the visible card, spelled differently
            0xBBBB: {"name": "AMD Radeon RX 7900 XTX"},  # a hidden card wearing the torch name
        },
    )
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        _subprocess_run(
            adapter_output = _adapter_output(
                [
                    ("luid_0x00000000_0x0000aaaa_phys_0", 2.0 * GB),
                    ("luid_0x00000000_0x0000bbbb_phys_0", 17.0 * GB),
                ]
            )
        ),
    )

    devices, _ = hw._rocm_windows_per_device_vram([0])
    # 17 GiB is the hidden card's. Unknown is the only honest answer here.
    assert devices[0]["used_gb"] is None
