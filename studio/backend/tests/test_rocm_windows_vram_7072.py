# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows hipMemGetInfo reports free equal to total, so per-GPU VRAM comes from the adapter counter."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from utils.hardware import hardware as hw


def _shared_setup_1(monkeypatch):
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    return devices


def _shared_setup_2(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(IGPU_DGPU_DEVICES, free_equals_total = True)
    )
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IGPU_DGPU_ADAPTERS))
    )


GB = 1024**3
MiB = 1024**2


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


def _engine_output(samples):
    """``GPU Engine`` rows in the shape Get-Counter is asked to emit."""
    if not samples:
        return "__NONE__\n"
    return "".join(f"{name}|{value}\n" for name, value in samples)


# One engine on the solo host's own adapter, so 12.0 narrowed or not.
_DEFAULT_ENGINES = [("pid_1_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3D", 12.0)]


def _subprocess_run(
    *,
    adapter_output = "__NONE__\n",
    util_output = None,
    engine_samples = None,
):
    if util_output is None:
        util_output = _engine_output(_DEFAULT_ENGINES if engine_samples is None else engine_samples)

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
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)
    # Force the capacity-ranking path: a Windows dev box would otherwise query its own HIP/registry.
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: {})
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", lambda ordinals, names: None)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    return monkeypatch


REPORTER_ADAPTERS = [
    ("luid_0x00000000_0x0000d1e2_phys_0", 40.0 * GB),
    ("luid_0x00000000_0x0000e34a_phys_0", 0.5 * GB),
    ("luid_0x00000000_0x0000f001_phys_0", 3 * MiB),
]
DEVICES = [("AMD Radeon PRO W7900", 48 * GB), ("AMD Radeon PRO W7500", 8 * GB)]


def test_system_tab_shows_per_gpu_used(win_rocm, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(DEVICES, free_equals_total = True))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(REPORTER_ADAPTERS))
    )

    devices = hw.get_visible_gpu_utilization()["devices"]
    by_idx = {d["index"]: d for d in devices}
    assert len(devices) == 2
    assert by_idx[0]["vram_total_gb"] == 48.0
    assert by_idx[0]["vram_used_gb"] == pytest.approx(40.0, abs = 0.01)
    assert by_idx[1]["vram_total_gb"] == 8.0
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
    assert sorted(d["index"] for d in devices) == [0, 1]
    assert {d["vram_total_gb"] for d in devices} == {48.0, 8.0}
    assert result["vram_total_gb"] == 48.0


def test_localized_counter_reports_unknown_not_zero(win_rocm, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(DEVICES, free_equals_total = True))
    monkeypatch.setattr(hw.subprocess, "run", _subprocess_run(adapter_output = "__NONE__\n"))

    devices = hw.get_visible_gpu_utilization()["devices"]
    assert len(devices) == 2
    assert {d["vram_total_gb"] for d in devices} == {48.0, 8.0}
    assert all(d["vram_used_gb"] is None for d in devices)
    assert all(d["vram_utilization_pct"] is None for d in devices)


def test_mem_get_info_guard_scopes_to_windows_rocm(monkeypatch):
    torch_mod = _fake_torch(DEVICES, free_equals_total = True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.sys, "platform", "win32")
    win = hw._torch_get_per_device_info([0, 1])
    assert [d["used_gb"] for d in win] == [None, None]
    assert [d["total_gb"] for d in win] == [48.0, 8.0]

    monkeypatch.setattr(hw.sys, "platform", "linux")
    assert [d["used_gb"] for d in hw._torch_get_per_device_info([0, 1])] == [0.0, 0.0]

    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.sys, "platform", "win32")
    assert [d["used_gb"] for d in hw._torch_get_per_device_info([0, 1])] == [0.0, 0.0]


def test_match_adapter_pairs_and_clamps():
    assert hw._match_adapter_used_to_devices([40 * GB, 0.5 * GB], [48 * GB, 8 * GB]) == [
        40 * GB,
        0.5 * GB,
    ]
    assert hw._match_adapter_used_to_devices([100 * GB], [48 * GB]) == [48 * GB]
    assert hw._match_adapter_used_to_devices([40 * GB], [48 * GB, 8 * GB]) == [40 * GB, None]


def test_match_adapter_pairs_single_device_past_placeholder_adapters():
    # gfx1151 Strix Halo shape: placeholders at exactly 0 cannot hold the survivor's bytes.
    assert hw._match_adapter_used_to_devices([1.2 * GB, 0.0, 0.0], [89 * GB]) == [1.2 * GB]
    assert hw._match_adapter_used_to_devices([0.0, 1.2 * GB, 0.0], [89 * GB]) == [1.2 * GB]
    assert hw._match_adapter_used_to_devices([100 * GB, 0.0, 0.0], [48 * GB]) == [None]
    assert hw._match_adapter_used_to_devices([40 * GB, 0.0], [48 * GB]) == [40 * GB]
    # Zero, not cardinality, makes it attributable: a sub-floor counter may be the idle visible card.
    assert hw._match_adapter_used_to_devices([1.2 * GB, 10 * MiB, 0.0], [89 * GB]) == [None]


def test_match_adapter_reports_unknown_when_more_active_than_visible():
    assert hw._match_adapter_used_to_devices([40 * GB, 0.5 * GB], [8 * GB]) == [None]


def test_unified_used_sums_dedicated_and_shared_for_the_compute_adapter(monkeypatch):
    # APU: Dedicated saturates at the carve-out and overflow lands in Shared, so only the sum tracks.
    def fake(counter = "Dedicated Usage"):
        if counter == "Dedicated Usage":
            return [("luid_compute", 30.5 * GB), ("luid_placeholder", 0.0)]
        return [("luid_compute", 19.0 * GB), ("luid_placeholder", 1.3 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() == 30.5 * GB + 19.0 * GB


def test_unified_used_selects_on_dedicated_not_the_sum(monkeypatch):
    def fake(counter = "Dedicated Usage"):
        if counter == "Dedicated Usage":
            return [("luid_compute", 1.2 * GB), ("luid_display", 0.0)]
        return [("luid_compute", 0.15 * GB), ("luid_display", 1.3 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() == 1.2 * GB + 0.15 * GB


def test_unified_used_declines_when_the_compute_adapter_is_ambiguous(monkeypatch):
    def two(counter = "Dedicated Usage"):
        return [("luid_a", 1.2 * GB), ("luid_b", 2.4 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", two)
    assert hw._rocm_windows_unified_used_bytes() is None

    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": [("luid_a", 10 * MiB)],
    )
    assert hw._rocm_windows_unified_used_bytes() is None

    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": None,
    )
    assert hw._rocm_windows_unified_used_bytes() is None


def test_unified_used_declines_rather_than_falling_back_to_dedicated_only(monkeypatch):
    # A failed Shared query declines: dedicated-only is wrong above the carve-out and overstates free.
    def fake(counter = "Dedicated Usage"):
        return [("luid_compute", 1.2 * GB)] if counter == "Dedicated Usage" else None

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() is None


def test_match_adapter_reports_unknown_when_hidden_high_use_adapter_survives_filter():
    assert hw._match_adapter_used_to_devices([40 * GB, 10 * MiB], [8 * GB]) == [None]
    assert hw._match_adapter_used_to_devices([10 * MiB, 40 * GB], [8 * GB]) == [None]


def test_match_adapter_reports_unknown_for_placeholder_fallback():
    assert hw._match_adapter_used_to_devices([50 * MiB, 10 * MiB], [8 * GB]) == [None]
    assert hw._match_adapter_used_to_devices([10 * MiB, 50 * MiB], [8 * GB]) == [None]
    assert hw._match_adapter_used_to_devices([50 * MiB, 10 * MiB, 5 * MiB], [48 * GB, 8 * GB]) == [
        None,
        None,
    ]


def test_match_adapter_reports_unknown_when_usage_not_capacity_ordered():
    assert hw._match_adapter_used_to_devices([7 * GB, 5 * GB], [8 * GB, 48 * GB]) == [None, None]
    assert hw._match_adapter_used_to_devices([7 * GB, 5 * GB], [48 * GB, 8 * GB]) == [None, None]
    assert hw._match_adapter_used_to_devices([12 * GB, 8 * GB], [24 * GB, 24 * GB]) == [None, None]
    assert hw._match_adapter_used_to_devices([5 * GB], [48 * GB, 8 * GB]) == [None, None]
    assert hw._match_adapter_used_to_devices([40 * GB], [48 * GB, 8 * GB]) == [40 * GB, None]


def test_match_adapter_reports_unknown_when_hidden_usage_fits_visible_card():
    assert hw._match_adapter_used_to_devices([40 * GB, 10 * MiB, 6 * GB], [48 * GB, 8 * GB]) == [
        40 * GB,
        None,
    ]
    assert hw._match_adapter_used_to_devices([6 * GB, 40 * GB, 10 * MiB], [48 * GB, 8 * GB]) == [
        40 * GB,
        None,
    ]
    assert hw._match_adapter_used_to_devices([6 * GB, 10 * MiB], [8 * GB]) == [None]


def test_match_adapter_capacity_forced_matrix():
    """Usage is attributed only if counters match the visible devices and exceed every smaller card."""
    m = hw._match_adapter_used_to_devices
    assert m([40 * GB, 0.5 * GB, 3 * MiB], [48 * GB, 8 * GB]) == [40 * GB, None]
    assert m([40 * GB, 20 * GB, 5 * GB, 3 * MiB], [48 * GB, 24 * GB, 8 * GB]) == [
        40 * GB,
        20 * GB,
        None,
    ]
    assert m([40 * GB, 3 * MiB, 3 * MiB], [48 * GB, 8 * GB]) == [None, None]
    assert m([40 * GB, 10 * MiB, 10 * MiB], [48 * GB, 8 * GB]) == [None, None]
    assert m([40 * GB, 20 * GB, 3 * MiB, 3 * MiB], [48 * GB, 24 * GB, 8 * GB]) == [
        None,
        None,
        None,
    ]
    assert m([40 * GB, 6 * GB, 3 * MiB, 3 * MiB], [48 * GB, 24 * GB, 8 * GB]) == [
        None,
        None,
        None,
    ]
    assert m([40 * GB, 10 * MiB], [8 * GB]) == [None]
    assert m([48 * GB, 3 * MiB, 3 * MiB], [24 * GB, 8 * GB]) == [None, None]
    assert m([40 * GB, 7 * GB, 6 * GB, 3 * MiB], [48 * GB, 8 * GB]) == [None, None]
    assert m([40 * GB, 7 * GB, 6 * GB, 3 * MiB, 3 * MiB], [48 * GB, 8 * GB]) == [None, None]
    assert m([50 * MiB, 10 * MiB], [8 * GB]) == [None]
    assert m([50 * MiB, 10 * MiB, 5 * MiB], [48 * GB, 8 * GB]) == [None, None]
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


SOLO_DEVICE = [("AMD Radeon RX 9060 XT", 16 * GB, "gfx1200")]
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
    assert hw._parse_adapter_luid("luid_0x00000000_0x00015369_phys_0") == 0x15369
    assert hw._parse_adapter_luid("luid_0x0000000A_0x00015369_phys_0") == (0xA << 32) | 0x15369
    assert hw._parse_adapter_luid("LUID_0X00000000_0X00015369_PHYS_0") == 0x15369
    assert hw._parse_adapter_luid("Total") is None
    assert hw._parse_adapter_luid("luid_zz_0x1_phys_0") is None


def test_normalize_adapter_name():
    assert hw._normalize_adapter_name("AMD Radeon(TM) 780M Graphics") == hw._normalize_adapter_name(
        "AMD Radeon 780M Graphics"
    )
    assert hw._normalize_adapter_name("AMD Radeon™ RX 9070") == hw._normalize_adapter_name(
        "amd radeon rx 9070"
    )
    assert hw._normalize_adapter_name("AMD Radeon RX 9070") != hw._normalize_adapter_name(
        "AMD Radeon RX 9070 XT"
    )


def test_parse_adapter_family_gfx():
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
    assert result["vram_used_gb_aggregate"] == pytest.approx(3.0, abs = 0.01)


def test_single_gpu_train_page_reports_used(win_rocm_solo):
    """Same figure on get_gpu_utilization, the Train page's GPU Monitor."""
    (device,) = hw.get_gpu_utilization()["devices"]
    assert device["vram_used_gb"] == pytest.approx(3.0, abs = 0.01)
    assert device["gpu_utilization_pct"] == 12.0


def test_luid_join_beats_a_busy_foreign_adapter(win_rocm, monkeypatch):
    """An idle AMD card beside a busy foreign GPU reports the AMD card's own
    usage. Capacity ranking has no vendor key here and declines outright."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(SOLO_DEVICE, free_equals_total = True))
    adapters = [
        ("luid_0x00000000_0x00015369_phys_0", 0.4 * GB),
        ("luid_0x00000000_0x00099999_phys_0", 9 * GB),
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


IGPU_DGPU_DEVICES = [
    ("AMD Radeon(TM) 780M Graphics", 2 * GB, "gfx1103"),
    ("AMD Radeon RX 9070 XT", 16 * GB, "gfx1201"),
]
IGPU_DGPU_ADAPTERS = [
    ("luid_0x00000000_0x0000c001_phys_0", 0.5 * GB),
    ("luid_0x00000000_0x0000d002_phys_0", 1.2 * GB),
]


def test_igpu_and_dgpu_each_report_their_own(win_rocm, monkeypatch):
    """The dGPU keeps its LUID usage while the shared iGPU declines."""
    _shared_setup_2(monkeypatch)
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xC001: {"name": "AMD Radeon(TM) 780M Graphics", "gfx": "gfx1103"},
            0xD002: {"name": "AMD Radeon RX 9070 XT", "gfx": "gfx1201"},
        },
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [None, pytest.approx(1.2, abs = 0.01)]
    assert aggregate is None


def test_a_name_the_two_sides_spell_differently_still_joins(win_rocm, monkeypatch):
    """DirectX takes the Description from the driver INF and HIP from the ASIC
    record, so an iGPU reaches the join under two spellings of one card."""
    _shared_setup_2(monkeypatch)
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xC001: {"name": "AMD Radeon 780M Graphics", "gfx": "gfx1103"},
            0xD002: {"name": "AMD Radeon(R) RX 9070 XT", "gfx": "gfx1201"},
        },
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [None, pytest.approx(1.2, abs = 0.01)]


def test_the_arch_answers_when_the_names_do_not(win_rocm, monkeypatch):
    """A generic iGPU Description that normalizing cannot reconcile. The gfx
    target both sides carry separates an iGPU from the dGPU beside it."""
    _shared_setup_2(monkeypatch)
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0xC001: {"name": "AMD Radeon(TM) Graphics", "gfx": "gfx1103"},
            0xD002: {"name": "Radeon RX 9070 XT Series", "gfx": "gfx1201"},
        },
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["used_gb"] for d in devices] == [None, pytest.approx(1.2, abs = 0.01)]
    assert aggregate is None


def test_a_partial_name_pass_still_lets_the_arch_finish_the_job(win_rocm, monkeypatch):
    """After a partial name pass, the arch pass still runs and places the remaining cards."""
    devices_spec = [
        ("AMD Radeon PRO W7900", 48 * GB, "gfx1100"),
        ("AMD Radeon RX 9070", 16 * GB, "gfx1201"),
        ("AMD Radeon RX 9070", 16 * GB, "gfx1200"),
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
        ("luid_0x00000000_0x0000c001_phys_0", 6 * GB),
        ("luid_0x00000000_0x0000d002_phys_0", 9 * GB),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
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

    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: {})
    assert join(SOLO_ADAPTERS, dev_meta) is None

    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: dict(SOLO_REGISTRY))
    assert (
        join(
            SOLO_ADAPTERS,
            [{"name": "AMD Radeon RX 7800 XT", "gfx": "gfx1101", "total_bytes": 16 * GB}],
        )
        is None
    )
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
    assert join([("luid_0x00000000_0x00015369_phys_0", 20 * GB)], dev_meta) is None


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
            return _Key("")
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


def test_registry_reads_optional_dedicated_memory(on_windows, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "winreg",
        _fake_winreg(
            {
                "{bbbbbbbb-0000-0000-0000-000000000000}": {
                    **AMD_RECORD,
                    "DedicatedVideoMemory": 32 * GB,
                    "DedicatedSystemMemory": 2 * GB,
                }
            }
        ),
    )
    assert hw._windows_amd_adapter_records_by_luid() == {
        0x14CF5: {
            "name": "AMD Radeon RX 9060 XT",
            "gfx": "gfx1200",
            "dedicated_memory_bytes": 34 * GB,
        }
    }


def test_a_record_that_will_not_read_declines_the_whole_map(on_windows, monkeypatch):
    """A partial map is indistinguishable from a complete one at the join, which
    would then hand a visible card its hidden twin's counter."""
    denied = PermissionError(5, "access is denied")
    for broken in (
        denied,
        {**AMD_RECORD, "Description": denied},
        {**AMD_RECORD, "Description": ""},
        {k: v for k, v in AMD_RECORD.items() if k != "AdapterLuid"},
        {k: v for k, v in AMD_RECORD.items() if k != "VendorId"},
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


def test_a_node_count_that_matches_the_wrong_nodes_is_not_ownership(win_rocm, monkeypatch):
    """One ordinal owning node 0 of a linked adapter, whose own sample is missing while
    the node it does NOT own reports: the counts agree, only the identities disagree.
    Taking that as ownership would sum the other node's work as this card's."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([TWIN_CARDS[0]], free_equals_total = True))
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0xAAAA, 0b01)))
    dev_meta = [
        {
            "visible_ordinal": 0,
            "name": TWIN_CARDS[0][0],
            "gfx": "gfx1100",
            "total_bytes": TWIN_CARDS[0][1],
            "dedicated_bytes": TWIN_CARDS[0][1],
        }
    ]

    hidden_only = [("luid_0x00000000_0x0000aaaa_phys_1", 5 * GB)]
    _, _, whole_adapter = hw._match_adapter_used_by_hip_luid(hidden_only, dev_meta)
    assert whole_adapter == [None]

    its_own = [("luid_0x00000000_0x0000aaaa_phys_0", 5 * GB)]
    _, _, whole_adapter = hw._match_adapter_used_by_hip_luid(its_own, dev_meta)
    assert whole_adapter == [0xAAAA]


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

    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", lambda ordinals, names: None)
    assert join(SOLO_ADAPTERS, dev_meta) is None

    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0x15369, 0)))
    assert join([SOLO_ADAPTERS[0], SOLO_ADAPTERS[0]], dev_meta) is None
    assert join([("luid_0x00000000_0x00099999_phys_0", 2 * GB)], dev_meta) is None
    assert join([("luid_0x00000000_0x00015369_phys_0", 20 * GB)], dev_meta) is None
    assigned, aggregate, whole_adapter = join(SOLO_ADAPTERS, dev_meta)
    assert assigned[0] == pytest.approx(3 * GB)
    assert aggregate == pytest.approx(3 * GB)
    assert whole_adapter == [0x15369]


def test_hip_luid_join_bounds_the_counter_by_the_carve_out(win_rocm, monkeypatch):
    """A unified APU is shown the whole driver pool as its total, so bounding
    with that admits any reading at all. The carve-out is what the counter
    fills, and the DirectX join already declines this."""
    apu = [
        {
            "visible_ordinal": 0,
            "name": "AMD Radeon(TM) 8060S Graphics",
            "gfx": "gfx1151",
            "total_bytes": 96 * GB,
            "dedicated_bytes": 0.5 * GB,
        }
    ]
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids((0x15369, 0)))
    over = [("luid_0x00000000_0x00015369_phys_0", 2 * GB)]
    assert hw._match_adapter_used_by_hip_luid(over, apu) is None
    assert hw._attribute_adapter_useds_by_key({"k": [2 * GB]}, {"k": [0]}, apu) is None

    under = [("luid_0x00000000_0x00015369_phys_0", 0.25 * GB)]
    assigned, aggregate, _whole = hw._match_adapter_used_by_hip_luid(under, apu)
    assert assigned[0] == pytest.approx(0.25 * GB)
    assert aggregate == pytest.approx(0.25 * GB)


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


def test_unified_memory_adopts_torch_total_even_when_used_unknown():
    """Windows ROCm unified-memory APU: torch's used is None but its total (the full
    GTT pool) is authoritative. The correction must still adopt the larger total;
    used stays at amd-smi's figure when torch's is unknown."""
    metrics = {"vram_total_gb": 8.0, "vram_used_gb": 2.0, "vram_utilization_pct": 25.0}
    hw._apply_unified_memory_correction(metrics, {"total_gb": 124.0, "used_gb": None, "index": 0})
    assert metrics["vram_total_gb"] == 124.0
    assert metrics["vram_used_gb"] == 2.0
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
    def fake(counter = "Dedicated Usage"):
        return [("luid_compute", 30.5 * GB)] if counter == "Dedicated Usage" else None

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() is None


def test_unified_used_keeps_a_successful_query_that_omits_the_luid(monkeypatch):
    def fake(counter = "Dedicated Usage"):
        if counter == "Dedicated Usage":
            return [("luid_compute", 1.2 * GB)]
        return [("luid_other", 3.0 * GB)]

    monkeypatch.setattr(hw, "_rocm_windows_perf_counter_vram_by_adapter", fake)
    assert hw._rocm_windows_unified_used_bytes() == 1.2 * GB


def test_per_device_vram_uses_the_snapshot_it_is_given(monkeypatch):
    # No re-sample: the cardinality check must see the same sample attribution runs on.
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


# Capacity comparisons must use the dedicated carve-out, not the widened pool total.
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

    matched = hw._attribute_adapter_useds_by_key({"k": [2.0 * GB]}, {"k": [0]}, dev_meta)
    assert matched is not None
    assigned, aggregate = matched
    assert assigned[0] == pytest.approx(2.0 * GB)
    assert aggregate == pytest.approx(2.0 * GB)


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
    """Windows 8060S registry: LUID is an Int64, no AdapterFamily is written, one instance has no record."""
    records = {86826: {"name": "AMD Radeon(TM) 8060S Graphics"}}
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: records)
    adapters = [
        ("luid_0x00000000_0x00017099_phys_0", 0.0),
        ("luid_0x00000000_0x0001532a_phys_0", 0.947 * GB),
        ("luid_0x00000000_0x00017034_phys_0", 0.0),
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
    """gfx1151 shows three counter instances for one GPU, so only the LUID join yields an aggregate."""
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
        ("luid_0x00000000_0x0001532a_phys_0", 31.681 * GB),
        ("luid_0x00000000_0x00017034_phys_0", 0.0),
        ("luid_0x00000000_0x00017099_phys_0", 0.0),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {86826: {"name": "AMD Radeon(TM) 8060S Graphics"}},
    )
    # The counter fake returns the same rows for Shared, so patch the sum directly.
    monkeypatch.setattr(
        hw, "_rocm_windows_unified_used_bytes", lambda dedicated = None: (31.681 + 17.820) * GB
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["used_gb"] == pytest.approx(49.5, abs = 0.01)
    assert aggregate == pytest.approx(49.5, abs = 0.01)

    assert (
        hw._rocm_windows_aggregate_used_bytes(
            [used for _, used in adapters], [float(int(89.465 * GB))]
        )
        is None
    ), "capacity ranking produced an aggregate the measured host cannot give it"

    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", lambda: {})
    _, fallback_aggregate = hw._rocm_windows_per_device_vram([0])
    assert fallback_aggregate == pytest.approx(49.5, abs = 0.01)


def test_the_join_declines_usage_it_cannot_place(win_rocm, monkeypatch):
    """Unclaimed counter must not pair with the remaining card, or a hidden card's bytes read as visible."""
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
            0xAAAA: {"name": "AMD Radeon RX 7900 XTX 24GB"},
            0xBBBB: {"name": "AMD Radeon RX 7900 XTX"},
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
    assert devices[0]["used_gb"] is None


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
    assert devices[0]["used_gb"] == 12.0
    assert aggregate == 12.0


def test_an_unsettled_classifier_does_not_cost_a_card_its_occupancy(win_rocm, monkeypatch):
    """The classifier answers "carve-out" for a discrete card too whenever the
    runtime leaves it unsettled (no integrated flag, HIP below 6.2). The driver
    total comes back equal there, so nothing widened and nothing is unknown."""
    torch = _fake_torch(DEVICES, free_equals_total = True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    devices = _shared_setup_1(monkeypatch)
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


MIXED = [("AMD Radeon(TM) 8060S Graphics", 8 * GB), ("AMD Radeon RX 7900 XTX", 24 * GB)]

MIXED_ADAPTERS = [
    ("luid_0x00000000_0x0000c001_phys_0", 2.0 * GB),
    ("luid_0x00000000_0x0000d1e2_phys_0", 10.0 * GB),
    ("luid_0x00000000_0x0000f001_phys_0", 3 * MiB),
]


def _mixed_host(monkeypatch):
    """APU (8 GiB carve-out, 128 GiB pool) at ordinal 0, discrete 24 GiB at 1."""
    torch = _fake_torch(MIXED, free_equals_total = True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    # _Props has no is_integrated, so the classifier stays patched, keyed on the total.
    monkeypatch.setattr(
        hw, "_rocm_props_total_is_carve_out", lambda props: props.total_memory == 8 * GB
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda i: (
            (96 * GB, 128 * GB) if i == 0 else pytest.fail("a discrete card must not be asked")
        ),
    )
    return torch


def test_widened_apu_total_does_not_cost_the_discrete_card_its_usage(win_rocm, monkeypatch):
    """A widened APU pool must not cost the discrete card its usage; the 10 GiB counter stays on it."""
    _mixed_host(monkeypatch)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(MIXED_ADAPTERS))
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert [d["total_gb"] for d in devices] == [128.0, 24.0]
    assert devices[0]["used_gb"] is None
    assert devices[1]["used_gb"] == pytest.approx(10.0, abs = 0.01)
    assert aggregate is None


def test_widened_apu_total_does_not_make_every_pairing_ambiguous(win_rocm, monkeypatch):
    """Widening the APU to the pool must not make every mixed-host ranking look swappable."""
    _mixed_host(monkeypatch)
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(MIXED_ADAPTERS[:2]))
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert devices[1]["used_gb"] == pytest.approx(10.0, abs = 0.01)
    assert devices[0]["used_gb"] is None


def test_widened_apu_total_does_not_admit_an_impossible_counter(win_rocm, monkeypatch):
    """A 50 GiB counter that fits no visible card must make every device Unknown, not shift the readings."""
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
        ("luid_0x00000000_0x0000a001_phys_0", 50.0 * GB),
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
    """A driver total below props.total_memory is refused, or free clamps to zero on a mostly empty card."""
    torch = _fake_torch(DEVICES, free_equals_total = True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i: (0, 36 * GB) if i == 0 else (0, 8 * GB)
    )
    devices = _shared_setup_1(monkeypatch)
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
    devices = _shared_setup_1(monkeypatch)
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]
    assert devices[0]["used_gb"] == pytest.approx(40.0, abs = 0.01)


def test_the_inventory_path_also_refuses_to_shrink_a_total(monkeypatch):
    """Inventory path must not shrink a total either, since an understated total hides models that fit."""
    # Pin get_device: on a GPU-less CI runner cached detection answers CPU and empties the inventory.
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    torch = _fake_torch(DEVICES)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i: (0, 36 * GB) if i == 0 else (0, 8 * GB)
    )

    devices = hw._torch_get_device_inventory([0, 1])
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]
    assert all(d["used_gb"] is None for d in devices)


def _apu_host(monkeypatch, unified_used = 12.0 * GB):
    """Lone 8 GiB carve-out APU with a 128 GiB pool; the Dedicated plus Shared figure is patched in."""
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
    assert result["vram_total_gb"] == 128.0
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
    """Shared usage is host memory outside props.total_memory, so the sum needs a positively unified
    part."""
    _apu_host(monkeypatch, unified_used = 90.0 * GB)
    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda props: False)

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 8.0
    assert devices[0]["used_gb"] == 2.0
    assert aggregate == 2.0


def test_a_nonzero_sub_threshold_row_declines_the_unified_sum(win_rocm, monkeypatch):
    """A nonzero sub-threshold row could be a hidden GPU's counter, so the unified sum needs exact zeros."""
    _apu_host(monkeypatch, unified_used = 30.0 * GB)
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        _subprocess_run(
            adapter_output = _adapter_output(
                [
                    ("luid_0x00000000_0x0000c001_phys_0", 10 * MiB),
                    ("luid_0x00000000_0x0000d1e2_phys_0", 30.0 * GB),
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
    """APU beside a discrete card: the APU widens, the discrete card keeps its usage and is never probed."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    torch = _fake_torch(
        [
            ("AMD Radeon(TM) 8060S Graphics", 32 * GB),
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
                    ("luid_0x00000000_0x0001532a_phys_0", 31.58 * GB),
                    ("luid_0x00000000_0x0000d1e2_phys_0", 40.0 * GB),
                    ("luid_0x00000000_0x00017034_phys_0", 0.0),
                ]
            )
        ),
    )

    devices, _ = hw._rocm_windows_per_device_vram([0, 1])
    assert devices[0]["total_gb"] == 89.46
    assert devices[1]["used_gb"] == 40.0
    assert probed == [0], f"expected only the APU to be probed, got {probed}"


def test_the_poll_does_not_probe_an_unclassified_discrete_card(win_rocm, monkeypatch):
    """mem_get_info leaves a ~612 MiB HIP context behind, so the poll must not probe unclassified cards."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    torch = _fake_torch(DEVICES)
    monkeypatch.setitem(sys.modules, "torch", torch)
    probed = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (probed.append(i), (0, 99 * GB))[1])
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda props: True)
    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda props: False)
    devices = _shared_setup_1(monkeypatch)
    assert probed == [], "a discrete card was asked for a context it does not need"
    assert [d["total_gb"] for d in devices] == [48.0, 8.0]


def test_a_failed_driver_probe_keeps_the_apu_on_the_dedicated_counter(win_rocm, monkeypatch):
    """A confirmed APU whose probe fails keeps Dedicated usage; summing Shared would fabricate 100%."""
    _apu_host(monkeypatch, unified_used = 40.0 * GB)
    torch = sys.modules["torch"]
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda i: (_ for _ in ()).throw(RuntimeError("hipMemGetInfo failed")),
    )

    devices, _ = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 8.0
    assert devices[0]["used_gb"] == 2.0
    assert devices[0]["used_gb"] <= devices[0]["total_gb"]


def test_the_measured_strix_halo_at_64_gib_held(win_rocm, monkeypatch):
    """Dedicated plateaus at the carve-out at 64 GiB held; alone it overstates free, the OOM direction."""
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
                    ),
                    ("luid_0x00000000_0x00017034_phys_0", 0.0),
                    ("luid_0x00000000_0x00017099_phys_0", 0.0),
                ]
            )
        ),
    )

    devices, _ = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 89.46
    assert devices[0]["used_gb"] == 65.52
    assert devices[0]["total_gb"] - devices[0]["used_gb"] < 24.0


def test_a_confirmed_apu_takes_the_unified_sum_even_unwidened(win_rocm, monkeypatch):
    """Confirmed APU uses the unified sum even unwidened, because Dedicated alone plateaus at carve-out."""
    _apu_host(monkeypatch, unified_used = 4.0 * GB)
    torch = sys.modules["torch"]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (0, APU[1]))

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == round(APU[1] / GB, 2)
    assert devices[0]["used_gb"] == 4.0
    assert aggregate == 4.0


def test_the_unified_sum_is_clamped_to_the_widened_total(win_rocm, monkeypatch):
    """The unified sum is clamped to the widened total, since driver plus desktop use can exceed the
    pool."""
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


# Radeon 780M (gfx1103): mem_get_info total 17303 MiB = 8 GiB carve-out + 75% of the WDDM shared heap.
PHOENIX_CARVE_OUT = 8.0 * GB
PHOENIX_POOL = 17303 * MiB
PHOENIX_RESIDENT = 12.0 * GB
PHOENIX_DEDICATED = 7.90 * GB
PHOENIX_SHARED = 4.10 * GB


def _phoenix_host(
    monkeypatch,
    gfx = "gfx1103",
    name = "AMD Radeon(TM) 780M Graphics",
):
    """The real _rocm_props_are_positively_unified runs unpatched on a pre-6.2 HIP runtime APU."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    torch = _fake_torch([(name, int(PHOENIX_POOL), gfx)])
    torch.__version__ = "2.9.0+rocm6.1"
    torch.version = types.SimpleNamespace(hip = "6.1", cuda = None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (0, int(PHOENIX_POOL)))
    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": [
            (
                "luid_0x00000000_0x0000c001_phys_0",
                PHOENIX_DEDICATED if counter == "Dedicated Usage" else PHOENIX_SHARED,
            )
        ],
    )
    return torch


def test_a_phoenix_apu_is_positively_unified_by_exact_arch(win_rocm):
    """gfx1103 is an integrated part and never shipped as a discrete board, so
    naming it cannot upgrade anything whose total is really carve-out scoped."""

    class _P:
        name = "AMD Radeon(TM) 780M Graphics"
        gcnArchName = "gfx1103"
        total_memory = int(PHOENIX_POOL)
        is_integrated = 0

    assert hw._rocm_props_are_positively_unified(_P()) is True
    _P.gcnArchName = "gfx1103:xnack-"
    assert hw._rocm_props_are_positively_unified(_P()) is True
    _P.gcnArchName = "GFX1103"
    assert hw._rocm_props_are_positively_unified(_P()) is True


def test_a_discrete_card_is_not_positively_unified(win_rocm):
    """The guarantee the numerator rests on: shared bytes are host memory a
    discrete card's props.total_memory never counted."""

    class _P:
        name = "AMD Radeon RX 7900 XTX"
        gcnArchName = "gfx1100"
        total_memory = 24 * GB
        is_integrated = 0

    assert hw._rocm_props_are_positively_unified(_P()) is False
    _P.gcnArchName, _P.name = "gfx1201", "AMD Radeon RX 9070 XT"
    assert hw._rocm_props_are_positively_unified(_P()) is False


def test_a_phoenix_pool_total_takes_the_shared_sum_not_the_plateau(win_rocm, monkeypatch):
    """A phoenix pool total takes the shared sum, not the plateau, or free is overstated by 4.10 GB."""
    _phoenix_host(monkeypatch)

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 16.9
    assert devices[0]["used_gb"] == pytest.approx(12.0, abs = 0.01)
    assert aggregate == pytest.approx(12.0, abs = 0.01)
    assert devices[0]["total_gb"] - devices[0]["used_gb"] == pytest.approx(4.9, abs = 0.01)


def test_a_discrete_card_on_the_same_runtime_keeps_the_dedicated_counter(win_rocm, monkeypatch):
    """The other half of the guarantee, through the same fixture: an unsettled
    runtime does not let Shared Usage into a discrete card's numerator, and its
    total is never probed for a pool it does not have."""
    torch = _phoenix_host(monkeypatch, gfx = "gfx1100", name = "AMD Radeon RX 7900 XTX")
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i: pytest.fail("a discrete card must not be asked")
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0])
    assert devices[0]["total_gb"] == 16.9
    assert devices[0]["used_gb"] == pytest.approx(7.90, abs = 0.01)
    assert aggregate == pytest.approx(7.90, abs = 0.01)


# This card at 17%, other adapters at 61%, so a wrong selection still reads 78.
FOREIGN_ENGINES = [
    ("pid_100_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3D", 12.0),
    ("pid_101_luid_0x00000000_0x00015369_phys_0_eng_1_engtype_3D", 5.0),
    ("pid_200_luid_0x00000000_0x000183fe_phys_0_eng_0_engtype_3D", 44.0),
    ("pid_300_luid_0x00000000_0x0001842f_phys_0_eng_0_engtype_3D", 17.0),
]


def _engine_query(
    monkeypatch,
    adapters,
    engine_samples = None,
):
    """Run the poll; hand back the devices and the counter path."""
    seen = []
    inner = _subprocess_run(adapter_output = _adapter_output(adapters), engine_samples = engine_samples)

    def fake_run(cmd, *a, **k):
        seen.append(" ".join(cmd) if isinstance(cmd, list) else str(cmd))
        return inner(cmd, *a, **k)

    monkeypatch.setattr(hw.subprocess, "run", fake_run)
    devices = hw.get_gpu_utilization()["devices"]
    (query,) = [c for c in seen if "GPU Engine" in c]
    return devices, query


def _solo_host(monkeypatch, identity = None):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(SOLO_DEVICE, free_equals_total = True))
    if identity is not None:
        monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", _hip_ids(identity))


def test_gpu_utilization_counts_only_this_adapters_engines(win_rocm, monkeypatch):
    _solo_host(monkeypatch, (0x15369, 0))

    (device,), _ = _engine_query(monkeypatch, SOLO_ADAPTERS, FOREIGN_ENGINES)
    assert device["gpu_utilization_pct"] == 17.0


def test_gpu_utilization_falls_back_to_every_engine(win_rocm, monkeypatch):
    """No HIP identity means no LUID to narrow to, and the host sum beats no reading."""
    _solo_host(monkeypatch)

    (device,), _ = _engine_query(monkeypatch, SOLO_ADAPTERS, FOREIGN_ENGINES)
    assert device["gpu_utilization_pct"] == 78.0


def test_a_linked_adapters_hidden_nodes_are_not_this_devices_engines(win_rocm, monkeypatch):
    """Narrowing needs the VRAM join to prove the device is the whole adapter."""
    _solo_host(monkeypatch, (0x15369, 0b01))
    linked = [
        ("luid_0x00000000_0x00015369_phys_0", 3 * GB),
        ("luid_0x00000000_0x00015369_phys_1", 5 * GB),
    ]

    (device,), _ = _engine_query(monkeypatch, linked, FOREIGN_ENGINES)
    assert device["vram_used_gb"] is None
    assert device["gpu_utilization_pct"] == 78.0


def test_the_adapter_is_matched_here_and_the_engine_type_in_the_path(win_rocm, monkeypatch):
    """A LUID in the path hands adapter selection to PDH, which no test here can run.
    The engine type stays in the path, where it keeps the sample set to one type per
    process."""
    _solo_host(monkeypatch, (0x15369, 0))

    _, query = _engine_query(monkeypatch, SOLO_ADAPTERS, FOREIGN_ENGINES)
    assert "\\GPU Engine(*engtype_3D*)\\Utilization Percentage" in query
    assert "luid_" not in query
    assert "InstanceName" in query


def test_a_longer_luid_that_starts_the_same_is_not_this_adapter(win_rocm, monkeypatch):
    """0x153690 begins with 0x15369. Matching on a prefix would count it."""
    _solo_host(monkeypatch, (0x15369, 0))

    (device,), _ = _engine_query(
        monkeypatch,
        SOLO_ADAPTERS,
        [
            ("pid_100_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3D", 12.0),
            ("pid_200_luid_0x00000000_0x00153690_phys_0_eng_0_engtype_3D", 44.0),
        ],
    )
    assert device["gpu_utilization_pct"] == 12.0


def test_engtype_spelled_in_lower_case_is_still_a_3d_engine(win_rocm, monkeypatch):
    """Hosts emit engtype_3D and engtype_3d both; PDH folded the case for us before."""
    _solo_host(monkeypatch, (0x15369, 0))

    (device,), _ = _engine_query(
        monkeypatch,
        SOLO_ADAPTERS,
        [
            ("pid_100_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3d", 12.0),
            ("pid_101_LUID_0X00000000_0X00015369_phys_0_eng_1_ENGTYPE_3D", 5.0),
        ],
    )
    assert device["gpu_utilization_pct"] == 17.0


def test_only_3d_engines_count(win_rocm, monkeypatch):
    """A compute engine is not the 3D reading, and no 3D instance is unknown, not zero."""
    _solo_host(monkeypatch, (0x15369, 0))

    (device,), _ = _engine_query(
        monkeypatch,
        SOLO_ADAPTERS,
        [
            ("pid_100_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_Compute", 88.0),
            ("pid_101_luid_0x00000000_0x00015369_phys_0_eng_1_engtype_Copy", 3.0),
        ],
    )
    assert device["gpu_utilization_pct"] is None


def test_an_idle_engine_reads_zero_rather_than_unknown(win_rocm, monkeypatch):
    _solo_host(monkeypatch, (0x15369, 0))

    (device,), _ = _engine_query(
        monkeypatch,
        SOLO_ADAPTERS,
        [
            ("pid_100_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3D", 0.0),
        ],
    )
    assert device["gpu_utilization_pct"] == 0.0


def test_unusable_samples_are_dropped_and_the_sum_is_bounded(win_rocm, monkeypatch):
    """The clamp used to live in the PowerShell expression, so it has to live here now."""
    _solo_host(monkeypatch, (0x15369, 0))

    (device,), _ = _engine_query(
        monkeypatch,
        SOLO_ADAPTERS,
        [
            ("pid_1_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3D", "NaN"),
            ("pid_2_luid_0x00000000_0x00015369_phys_0_eng_1_engtype_3D", "Infinity"),
            ("pid_3_luid_0x00000000_0x00015369_phys_0_eng_2_engtype_3D", -5.0),
            ("pid_4_luid_0x00000000_0x00015369_phys_0_eng_3_engtype_3D", "not-a-number"),
            ("pid_5_luid_0x00000000_0x00015369_phys_0_eng_4_engtype_3D", 70.0),
            ("pid_6_luid_0x00000000_0x00015369_phys_0_eng_5_engtype_3D", 70.0),
        ],
    )
    assert device["gpu_utilization_pct"] == 100.0


def test_an_instance_name_carrying_no_luid_is_not_this_adapters(win_rocm, monkeypatch):
    _solo_host(monkeypatch, (0x15369, 0))

    (device,), _ = _engine_query(
        monkeypatch,
        SOLO_ADAPTERS,
        [
            ("pid_100_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3D", 12.0),
            ("engtype_3D", 44.0),
            ("pid_200_luid_zz_0x1_phys_0_eng_0_engtype_3D", 44.0),
        ],
    )
    assert device["gpu_utilization_pct"] == 12.0


def test_no_engine_instances_at_all_reads_unknown(win_rocm, monkeypatch):
    _solo_host(monkeypatch, (0x15369, 0))

    (device,), _ = _engine_query(monkeypatch, SOLO_ADAPTERS, [])
    assert device["gpu_utilization_pct"] is None


def test_a_high_luid_half_selects_its_own_adapter(win_rocm, monkeypatch):
    """The high half is only ever zero on ordinary hardware, so nothing else covers it."""
    high = (0xA << 32) | 0x15369
    _solo_host(monkeypatch, (high, 0))

    (device,), _ = _engine_query(
        monkeypatch,
        [("luid_0x0000000a_0x00015369_phys_0", 3 * GB)],
        [
            ("pid_1_luid_0x0000000a_0x00015369_phys_0_eng_0_engtype_3D", 12.0),
            ("pid_2_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3D", 44.0),
        ],
    )
    assert device["gpu_utilization_pct"] == 12.0


def test_the_luid_is_internal_and_never_reaches_a_payload(win_rocm, monkeypatch):
    """Keeping the internal LUID out of both payloads is a convention, not a mechanism."""
    _solo_host(monkeypatch, (0x15369, 0))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(SOLO_ADAPTERS))
    )

    def carries_luid(value):
        if isinstance(value, dict):
            return "luid" in value or any(carries_luid(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(carries_luid(v) for v in value)
        return False

    internal, _ = hw._rocm_windows_per_device_vram([0])
    assert internal[0]["luid"] == 0x15369
    assert carries_luid(internal[0])

    for payload in (hw.get_gpu_utilization(), hw.get_visible_gpu_utilization()):
        assert not carries_luid(payload)
        assert 0x15369 not in [v for v in payload.values() if isinstance(v, int)]


RX6800_IGPU = [
    ("AMD Radeon RX 6800", 16 * GB, "gfx1030"),
    ("AMD Radeon(TM) Graphics", int(76.8 * GB), "gfx1036"),
]
DGPU_LUID, IGPU_LUID = 0xD1E2, 0x1532A


def _igpu_beside_dgpu(monkeypatch, *, shared = "default"):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(RX6800_IGPU, free_equals_total = True))
    monkeypatch.setattr(
        hw, "_rocm_props_are_positively_unified", lambda props: props.gcnArchName == "gfx1036"
    )
    dedicated = [
        (f"luid_0x00000000_0x{DGPU_LUID:08x}_phys_0", 4.28 * GB),
        (f"luid_0x00000000_0x{IGPU_LUID:08x}_phys_0", 0.3 * GB),
        ("luid_0x00000000_0x00017034_phys_0", 0.0),
    ]
    if shared == "default":
        shared = [
            (f"luid_0x00000000_0x{DGPU_LUID:08x}_phys_0", 0.1 * GB),
            (f"luid_0x00000000_0x{IGPU_LUID:08x}_phys_0", 1.2 * GB),
        ]
    counters = {"Dedicated Usage": dedicated, "Shared Usage": shared}
    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": counters[counter],
    )
    monkeypatch.setattr(
        hw, "_rocm_windows_hip_adapter_ids", _hip_ids((DGPU_LUID, 0), (IGPU_LUID, 0))
    )


def test_an_igpu_beside_a_discrete_card_reports_its_own_used(win_rocm, monkeypatch):
    _igpu_beside_dgpu(monkeypatch)

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert devices[0]["used_gb"] == pytest.approx(4.28, abs = 0.01)
    assert devices[1]["used_gb"] == pytest.approx(1.5, abs = 0.01)
    assert devices[1]["total_gb"] == 76.8
    assert aggregate == pytest.approx(5.78, abs = 0.01)


def test_a_failed_shared_query_leaves_the_igpu_unknown(win_rocm, monkeypatch):
    """A failed Shared query is not zero: past the carve-out the overflow lives there."""
    _igpu_beside_dgpu(monkeypatch, shared = None)

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert devices[0]["used_gb"] == pytest.approx(4.28, abs = 0.01)
    assert devices[1]["used_gb"] is None
    assert aggregate is None


def test_without_hip_luids_the_igpu_stays_unknown(win_rocm, monkeypatch):
    """Without a LUID the lone busy adapter is the discrete card, so no sum."""
    _igpu_beside_dgpu(monkeypatch)
    monkeypatch.setattr(hw, "_rocm_windows_hip_adapter_ids", lambda ordinals, names: None)

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert devices[1]["used_gb"] is None
    assert aggregate is None


def test_the_system_tab_gets_both_rows_and_the_total(win_rocm, monkeypatch):
    _igpu_beside_dgpu(monkeypatch)

    devices = hw.get_visible_gpu_utilization()["devices"]
    assert [(d["index"], d["vram_used_gb"], d["vram_total_gb"]) for d in devices] == [
        (0, pytest.approx(4.28, abs = 0.01), 16.0),
        (1, pytest.approx(1.5, abs = 0.01), 76.8),
    ]


def test_an_igpu_enumerated_first_is_still_paired_by_luid(win_rocm, monkeypatch):
    _igpu_beside_dgpu(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(list(reversed(RX6800_IGPU)), free_equals_total = True)
    )
    monkeypatch.setattr(
        hw, "_rocm_windows_hip_adapter_ids", _hip_ids((IGPU_LUID, 0), (DGPU_LUID, 0))
    )

    devices, aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert devices[0]["used_gb"] == pytest.approx(1.5, abs = 0.01)
    assert devices[1]["used_gb"] == pytest.approx(4.28, abs = 0.01)
    assert aggregate == pytest.approx(5.78, abs = 0.01)
