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
        entry = dev[i]
        if isinstance(entry, Exception):
            raise entry
        name, total, *arch = entry
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
    # Capacity-ranking path by default: without these a Windows dev box asks its
    # own HIP runtime and reads its own registry, and an identity join answers
    # instead. The identity tests below opt in with their own map.
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: {})
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", lambda ordinals, names: None)
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


def test_match_adapter_reports_unknown_when_more_active_than_visible():
    # More adapters actively using VRAM than are visible (a GPU outside the mask):
    # attribution would fabricate a value, so report unknown for every device.
    assert hw._match_adapter_used_to_devices([40 * GB, 0.5 * GB], [8 * GB]) == [None]


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
# HIP's own LUID (idea and the R0600 route from @pablo86gr in #8793)
#
# The exact key: hipDeviceProp_tR0600 carries the same DXGI LUID the counters
# are named after, so the join needs nothing about the card, only the card.
# Tried before the DirectX record, which falls back to matching on what the two
# sides say about it and so cannot separate two of one model.
# ----------------------------------------------------------------------------- #
TWIN_CARDS = [("AMD Radeon RX 7900 XTX", 24 * GB, "gfx1100")] * 2
TWIN_ADAPTERS = [
    ("luid_0x00000000_0x0000aaaa_phys_0", 10 * GB),
    ("luid_0x00000000_0x0000bbbb_phys_0", 4 * GB),
]


def _hip_ids(*identities):
    """Stand in for the ctypes probe: (luid, node_mask) per queried ordinal."""

    def probe(ordinals, names):
        probe.asked = list(ordinals)
        return [identities[o] for o in ordinals]

    probe.asked = []
    return probe


def test_hip_luid_separates_two_of_one_model(win_rocm, monkeypatch):
    """What the DirectX join cannot do: one Description and one arch for both
    cards leaves it the aggregate only, while the LUIDs are still distinct."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(TWIN_CARDS, free_equals_total = True))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(TWIN_ADAPTERS))
    )
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0xAAAA, 0), (0xBBBB, 0)))

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [
        pytest.approx(10.0, abs = 0.01),
        pytest.approx(4.0, abs = 0.01),
    ]
    assert aggregate == pytest.approx(14.0, abs = 0.01)


def test_hip_is_asked_about_the_ordinal_that_answered(win_rocm, monkeypatch):
    """dev_meta is compacted when a device fails to probe, so its positions are
    not HIP ordinals. Asking by position would hand ordinal 0's LUID to the
    card that survived at ordinal 1."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(
            [RuntimeError("ordinal 0 will not probe"), TWIN_CARDS[1]], free_equals_total = True
        ),
    )
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(TWIN_ADAPTERS))
    )
    probe = _hip_ids((0xAAAA, 0), (0xBBBB, 0))
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", probe)

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert probe.asked == [1]
    # Ordinal 1 is the 0xBBBB counter, not the 0xAAAA one its position would name.
    assert devices[0]["used_gb"] == pytest.approx(4.0, abs = 0.01)
    assert aggregate == pytest.approx(4.0, abs = 0.01)


def test_linked_nodes_keep_the_aggregate_but_not_per_device(win_rocm, monkeypatch):
    """Two ordinals behind one LUID. The counters index the adapter's nodes as
    phys_N and nothing says which ordinal owns which, so the sum survives and
    the pairing does not."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(TWIN_CARDS, free_equals_total = True))
    adapters = [
        ("luid_0x00000000_0x0000aaaa_phys_0", 10 * GB),
        ("luid_0x00000000_0x0000aaaa_phys_1", 4 * GB),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
    monkeypatch.setattr(
        hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0xAAAA, 0b01), (0xAAAA, 0b10))
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [None, None]
    assert aggregate == pytest.approx(14.0, abs = 0.01)


def test_a_node_the_visible_ordinals_do_not_own_declines(win_rocm, monkeypatch):
    """One ordinal holding one node of a two-node adapter. The second node's
    usage is on hardware HIP is not showing, so it is not this card's to claim,
    and the node mask is what says so."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([TWIN_CARDS[0]], free_equals_total = True))
    adapters = [
        ("luid_0x00000000_0x0000aaaa_phys_0", 3 * GB),
        ("luid_0x00000000_0x0000aaaa_phys_1", 9 * GB),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0xAAAA, 0b01)))

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["used_gb"] is None
    assert aggregate is None


def test_hip_luid_join_declines_and_falls_back(win_rocm, monkeypatch):
    dev_meta = [
        {
            "visible_ordinal": 0,
            "name": "AMD Radeon RX 9060 XT",
            "gfx": "gfx1200",
            "total_bytes": 16 * GB,
        }
    ]
    join = hw._match_adapter_used_by_hip_luid

    # The runtime cannot be asked (not Windows, DLL not loaded, symbol absent).
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", lambda ordinals, names: None)
    assert join(SOLO_ADAPTERS, dev_meta) is None

    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0x15369, 0)))
    # One physical node cannot report twice; summing would double-count it.
    assert join([SOLO_ADAPTERS[0], SOLO_ADAPTERS[0]], dev_meta) is None
    # A visible card with no counter of its own.
    assert join([("luid_0x00000000_0x00099999_phys_0", 2 * GB)], dev_meta) is None
    # A usage above the card's own capacity.
    assert join([("luid_0x00000000_0x00015369_phys_0", 20 * GB)], dev_meta) is None
    # And the resolvable case still resolves, so the declines above are the reason.
    assigned, aggregate, whole_adapter = join(SOLO_ADAPTERS, dev_meta)
    assert assigned[0] == pytest.approx(3 * GB)
    assert aggregate == pytest.approx(3 * GB)
    # One ordinal owning every node its LUID names, so the engine counters for
    # that LUID are this device's and nothing else's.
    assert whole_adapter == [0x15369]


class _Callable:
    """A ctypes function pointer: callable, and argtypes/restype are settable."""

    def __init__(self, fn):
        self._fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._fn(*args)


def _r0600_blob(
    name,
    luid,
    node_mask = 0,
):
    """The documented prefix of hipDeviceProp_tR0600: name[256], uuid[16],
    luid[8], luidDeviceNodeMask[4]."""
    return (
        name.encode("utf-8").ljust(256, b"\x00")
        + bytes(16)
        + luid.to_bytes(8, "little")
        + node_mask.to_bytes(4, "little")
    )


def _fake_ctypes(
    blobs,
    *,
    loaded = ("amdhip64_7.dll",),
    has_symbol = True,
    rc = 0,
):
    """A `ctypes` whose loaded amdhip64 answers with ``blobs[ordinal]``."""

    class _Buffer:
        def __init__(self, size):
            self.data = bytes(size)

        def __bytes__(self):
            return self.data

    class _Hip:
        def __getattr__(self, symbol):
            if not has_symbol:
                raise AttributeError(symbol)

            def get_properties(buffer, ordinal):
                if rc:
                    return rc
                buffer.data = blobs[ordinal].ljust(len(buffer.data), b"\x00")
                return 0

            return _Callable(get_properties)

    kernel32 = types.SimpleNamespace(
        GetModuleHandleW = _Callable(lambda name: 0x1234 if name in loaded else None)
    )
    mod = types.ModuleType("ctypes")
    mod.WinDLL = lambda name, handle = None, use_last_error = False: (
        kernel32 if name == "kernel32" else _Hip()
    )
    mod.create_string_buffer = _Buffer
    mod.byref = lambda buffer: buffer
    mod.c_wchar_p = mod.c_void_p = mod.c_int = object()
    return mod


@pytest.fixture
def hip_probe_host(monkeypatch):
    """Windows with torch's HIP 7 runtime loaded, so the probe runs for real."""
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(SOLO_DEVICE))
    return monkeypatch


NINE_SIXTY = "AMD Radeon RX 9060 XT"


def test_hip_probe_reads_the_luid_and_node_mask(hip_probe_host, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "ctypes", _fake_ctypes([_r0600_blob(NINE_SIXTY, 0x14AD4, node_mask = 0b11)])
    )
    assert hw._rocm_windows_hip_adapter_ids([0], [NINE_SIXTY]) == [(0x14AD4, 0b11)]


def test_hip_probe_needs_the_name_to_read_back(hip_probe_host, monkeypatch):
    """The R0600 suffix is the ABI version, so the prefix is fixed by the same
    contract that named it, but the offsets here are still hardcoded. Reading
    the name back is what catches a layout that ever moved."""
    monkeypatch.setitem(
        sys.modules, "ctypes", _fake_ctypes([_r0600_blob("Some Other Struct", 0x14AD4)])
    )
    assert hw._rocm_windows_hip_adapter_ids([0], [NINE_SIXTY]) is None


def test_hip_probe_declines_what_it_cannot_answer(hip_probe_host, monkeypatch):
    blob = [_r0600_blob(NINE_SIXTY, 0x14AD4)]
    for label, fake in (
        ("no LUID for this device", _fake_ctypes([_r0600_blob(NINE_SIXTY, 0)])),
        ("the ordinal will not answer", _fake_ctypes(blob, rc = 1)),
        ("the runtime is not in this process", _fake_ctypes(blob, loaded = ())),
        ("the versioned symbol is absent", _fake_ctypes(blob, has_symbol = False)),
    ):
        monkeypatch.setitem(sys.modules, "ctypes", fake)
        assert hw._rocm_windows_hip_adapter_ids([0], [NINE_SIXTY]) is None, label


def test_hip_probe_is_windows_only(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    assert hw._rocm_windows_hip_adapter_ids([0], [NINE_SIXTY]) is None


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


# ----------------------------------------------------------------------------- #
# GPU Engine utilization
#
# The engine counters are instanced per adapter, exactly like the memory ones,
# so the unfiltered 3D sum is every adapter's work: the display iGPU's and the
# Basic Render Driver's alongside the card being monitored.
# ----------------------------------------------------------------------------- #
def _engine_query(monkeypatch, adapters):
    """Run the Train page's poll and hand back the GPU Engine counter path."""
    seen = []
    inner = _subprocess_run(adapter_output = _adapter_output(adapters))

    def fake_run(cmd, *a, **k):
        seen.append(" ".join(cmd) if isinstance(cmd, list) else str(cmd))
        return inner(cmd, *a, **k)

    monkeypatch.setattr(hw.subprocess, "run", fake_run)
    devices = hw.get_gpu_utilization()["devices"]
    (query,) = [c for c in seen if "GPU Engine" in c]
    return devices, query


def test_gpu_utilization_counts_only_this_adapters_engines(win_rocm, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(SOLO_DEVICE, free_equals_total = True))
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0x15369, 0)))

    (device,), engine_query = _engine_query(monkeypatch, SOLO_ADAPTERS)
    assert device["gpu_utilization_pct"] == 12.0
    assert "luid_0x00000000_0x00015369_" in engine_query


def test_gpu_utilization_falls_back_to_every_engine(win_rocm, monkeypatch):
    """Without HIP identity there is no LUID to narrow to, and the whole-host
    sum is still better than no reading."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(SOLO_DEVICE, free_equals_total = True))

    (device,), engine_query = _engine_query(monkeypatch, SOLO_ADAPTERS)
    assert device["gpu_utilization_pct"] == 12.0
    assert "luid_" not in engine_query


def test_a_linked_adapters_hidden_nodes_are_not_this_devices_engines(win_rocm, monkeypatch):
    """A LUID covering a node this ordinal does not own would sum that node's
    work in. The VRAM join is what establishes the device IS the whole adapter,
    and here it does not: one ordinal holding one node of two."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(SOLO_DEVICE, free_equals_total = True))
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0x15369, 0b01)))
    linked = [
        ("luid_0x00000000_0x00015369_phys_0", 3 * GB),
        ("luid_0x00000000_0x00015369_phys_1", 5 * GB),
    ]

    (device,), engine_query = _engine_query(monkeypatch, linked)
    assert device["vram_used_gb"] is None  # the VRAM join declined too
    assert "luid_" not in engine_query
