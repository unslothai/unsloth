# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The startup notice for AMD Windows drivers with the ROCm/TheRock#7221 idle-eviction bug.

Adrenalin 26.5.1 through PRO 26.9.1 page out live allocations on an idle RDNA4 card, which
freezes multi-GPU hosts; 26.5.1 (32.0.31007.1017) is the first bad build and 26.9.2
(32.0.32015.2008) the fix. Every adapter and driver version here is pinned: nothing reads the
host's registry or WMI.
"""

from __future__ import annotations

import sys
import types

import pytest

import utils.hardware.hardware as hw
from utils.hardware import nvidia

LAST_GOOD = "32.0.22042.14002"  # Adrenalin 26.3.1
FIRST_BAD = "32.0.31007.1017"  # Adrenalin 26.5.1
BROKEN = "32.0.31041.1004"  # Adrenalin 26.8.1
FIXED = "32.0.32015.2008"  # Adrenalin 26.9.2


def _amd(
    name = "AMD Radeon AI PRO R9700",
    gfx = "gfx1201",
    driver = BROKEN,
    **extra,
):
    device = {"vendor": "amd", "name": name, "driver_version": driver, **extra}
    if gfx:
        device["gfx"] = gfx
    return device


@pytest.fixture(autouse = True)
def _fresh_check(monkeypatch):
    monkeypatch.setattr(hw, "_amd_driver_check_started", False)
    monkeypatch.setattr(hw, "_amd_driver_notice", None)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)


def _registry(monkeypatch, *records):
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda *a, **k: {luid: dict(r) for luid, r in enumerate(records)},
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        # DirectX's REG_QWORD as read from an R9700 on Adrenalin 26.8.1.
        (9007201289044972, (32, 0, 31041, 1004)),
        ((32 << 48) | (22042 << 16) | 14002, (32, 0, 22042, 14002)),
        ("32.0.32015.2008", (32, 0, 32015, 2008)),
        (" 32.0.31041.1004 ", (32, 0, 31041, 1004)),
        ("32.0.31041", None),
        ("32.0.abc.1004", None),
        ("32.0.31041.1004 beta", None),
        ("", None),
        (None, None),
        (True, None),
        (0, None),
        (-1, None),
        (1 << 64, None),
    ],
)
def test_driver_version_parses_both_spellings(value, expected):
    assert hw._parse_windows_driver_version(value) == expected


@pytest.mark.parametrize(
    "driver,flagged",
    [
        ("32.0.21001.9005", False),  # older than 26.3.1
        (LAST_GOOD, False),
        ("32.0.22042.14003", False),
        ("32.0.23017.1001", False),  # 26.1.1
        ("32.0.23033.1002", False),  # 26.3.1 R9700 package, measured clean in TheRock#7221
        ("32.0.31007.1016", False),
        (FIRST_BAD, True),
        ("32.0.31036.15", True),  # PRO 26.Q3
        ("32.0.31041.3013", True),  # 26.9.1
        (BROKEN, True),
        ("32.0.32015.2007", True),
        (FIXED, False),
        ("32.0.33010.1003", False),
        ("33.0.10000.1", False),
        ("not.a.driver.version", False),
        (None, False),
    ],
)
def test_only_drivers_inside_the_affected_range_are_flagged(driver, flagged):
    notice = hw.amd_driver_idle_evict_notice([_amd(driver = driver)])
    assert (notice is not None) is flagged
    if flagged:
        assert notice["driver_version"] == driver
        assert notice["severity"] == "warning"
        assert notice["link"] == "https://github.com/ROCm/TheRock/issues/7221"
        assert notice["devices"] == ["AMD Radeon AI PRO R9700"]
        assert notice["message"] == (
            f"AMD driver {driver} has a known bug that can freeze Windows when an AMD GPU "
            "sits idle, most often with more than one GPU. Update to Adrenalin 26.9.2 or later."
        )


def test_a_qwordshaped_driver_version_is_accepted_too():
    notice = hw.amd_driver_idle_evict_notice([_amd(driver = 9007201289044972)])
    assert notice is not None and notice["driver_version"] == BROKEN


def test_a_non_amd_card_is_never_flagged():
    nvidia_card = {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 4090", "driver_version": BROKEN}
    assert hw.amd_driver_idle_evict_notice([nvidia_card]) is None


@pytest.mark.parametrize(
    "name,gfx,flagged",
    [
        ("AMD Radeon RX 7900 XTX", "gfx1100", False),
        ("AMD Radeon(TM) 8060S Graphics", "gfx1151", False),
        ("AMD Radeon RX 9060 XT", "gfx1200", True),
        ("AMD Radeon RX 9070 XT", "gfx1201", True),
        # gfx1250 is Instinct, not RDNA4.
        ("AMD Instinct MI350X", "gfx1250", False),
        # A driver that wrote no AdapterFamily: the name decides.
        ("AMD Radeon RX 9070", None, True),
        ("AMD Radeon AI PRO R9700", None, True),
        ("AMD Radeon RX 7800 XT", None, False),
        ("AMD Radeon 780M Graphics", None, False),
    ],
)
def test_only_rdna4_amd_cards_are_flagged(name, gfx, flagged):
    notice = hw.amd_driver_idle_evict_notice([_amd(name = name, gfx = gfx)])
    assert (notice is not None) is flagged


def test_an_arch_from_the_registry_beats_the_name():
    # A gfx1100 record that happens to carry an RDNA4-looking name is still RDNA3.
    assert hw.amd_driver_idle_evict_notice([_amd(name = "RX 9070 (rebadged)", gfx = "gfx1100")]) is None


def test_one_gpu_is_a_warning():
    assert hw.amd_driver_idle_evict_notice([_amd()])["severity"] == "warning"


@pytest.mark.parametrize(
    "others",
    [
        [_amd()],  # 3x R9700 shape, cut to two
        [_amd(name = "AMD Radeon RX 9070", gfx = "gfx1201")],  # RX 9070 + R9700
        [_amd(name = "AMD Radeon(TM) Graphics", gfx = "gfx1036")],  # Ryzen iGPU driving the display
        [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 3060"}],
        [{"vendor": "intel", "name": "Intel(R) UHD Graphics 770"}],
    ],
)
def test_more_than_one_gpu_is_critical(others):
    notice = hw.amd_driver_idle_evict_notice([_amd(), *others])
    assert notice["severity"] == "critical"
    assert notice["gpu_count"] == 2


def test_no_devices_is_no_notice():
    assert hw.amd_driver_idle_evict_notice([]) is None


class _InlineThread:
    def __init__(
        self,
        target,
        name = None,
        daemon = None,
    ):
        self._target = target

    def start(self):
        self._target()


def _live(monkeypatch, versions):
    monkeypatch.setattr(hw, "_windows_live_amd_driver_versions", lambda: versions)


def test_non_windows_is_a_no_op(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")

    def _no_threads(*a, **k):
        raise AssertionError("no thread off Windows")

    monkeypatch.setattr(hw.threading, "Thread", _no_threads)
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **k: pytest.fail("no inventory off Windows")
    )
    hw.start_amd_driver_check()
    assert hw.amd_driver_warning_report() == {}


def test_windows_check_logs_once_and_publishes(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw.threading, "Thread", _InlineThread)
    calls = []

    def _inventory(**k):
        calls.append(k)
        return {"devices": [_amd(), _amd()]}

    monkeypatch.setattr(hw, "get_physical_gpu_inventory", _inventory)
    _registry(monkeypatch, _amd())
    _live(monkeypatch, {"AMD Radeon AI PRO R9700": BROKEN})
    warnings = []
    monkeypatch.setattr(hw.logger, "warning", lambda msg, *args: warnings.append(msg % args))

    hw.start_amd_driver_check()
    hw.start_amd_driver_check()  # once per process

    assert len(calls) == 1
    assert len(warnings) == 1 and BROKEN in warnings[0] and "TheRock/issues/7221" in warnings[0]
    report = hw.amd_driver_warning_report()
    assert report["driver_warning"]["severity"] == "critical"
    assert report["driver_warning"]["driver_version"] == BROKEN


def test_a_fixed_driver_publishes_nothing(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw.threading, "Thread", _InlineThread)
    _registry(monkeypatch, _amd())
    _live(monkeypatch, {"AMD Radeon AI PRO R9700": FIXED})
    monkeypatch.setattr(hw, "get_physical_gpu_inventory", lambda **k: {"devices": [_amd()]})
    hw.start_amd_driver_check()
    assert hw.amd_driver_warning_report() == {}


@pytest.mark.parametrize(
    "live,flagged",
    [({"AMD Radeon AI PRO R9700": FIXED}, False), ({"AMD Radeon AI PRO R9700": BROKEN}, True)],
)
def test_the_live_adapter_decides_over_a_stale_same_named_record(monkeypatch, live, flagged):
    # One of two identical R9700s was replaced: the registry holds both an old and a current
    # record under one name, and only WMI knows which version is installed.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw.threading, "Thread", _InlineThread)
    _registry(monkeypatch, _amd(driver = BROKEN), _amd(driver = FIXED))
    _live(monkeypatch, live)
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **k: {"devices": [_amd(driver = BROKEN)]}
    )
    hw.start_amd_driver_check()
    assert bool(hw.amd_driver_warning_report()) is flagged


def test_an_unanswered_live_scan_publishes_nothing(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw.threading, "Thread", _InlineThread)
    _registry(monkeypatch, _amd())
    _live(monkeypatch, None)
    monkeypatch.setattr(hw, "get_physical_gpu_inventory", lambda **k: {"devices": [_amd()]})
    hw.start_amd_driver_check()
    assert hw.amd_driver_warning_report() == {}


@pytest.mark.parametrize(
    "records",
    [
        (),
        (_amd(driver = FIXED),),
        (_amd(name = "AMD Radeon RX 7900 XTX", gfx = "gfx1100"),),
    ],
)
def test_no_affected_registry_record_skips_the_live_scan(monkeypatch, records):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw.threading, "Thread", _InlineThread)
    _registry(monkeypatch, *[{k: v for k, v in r.items() if k != "vendor"} for r in records])
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **k: pytest.fail("no WMI scan without a lead")
    )
    monkeypatch.setattr(
        hw, "_windows_live_amd_driver_versions", lambda: pytest.fail("no WMI scan without a lead")
    )
    hw.start_amd_driver_check()
    assert hw.amd_driver_warning_report() == {}


def test_a_stale_registry_record_the_live_scan_drops_publishes_nothing(monkeypatch):
    # A removed R9700 left its old record behind; the card in the box is an RX 7900 XTX.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw.threading, "Thread", _InlineThread)
    _registry(monkeypatch, _amd())
    _live(monkeypatch, {"AMD Radeon RX 7900 XTX": BROKEN})
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **k: {"devices": [_amd(name = "AMD Radeon RX 7900 XTX", gfx = "gfx1100")]},
    )
    hw.start_amd_driver_check()
    assert hw.amd_driver_warning_report() == {}


def test_a_failing_inventory_never_raises(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw.threading, "Thread", _InlineThread)

    def _boom(**k):
        raise RuntimeError("WMI is gone")

    monkeypatch.setattr(hw, "get_physical_gpu_inventory", _boom)
    _registry(monkeypatch, _amd())
    _live(monkeypatch, {"AMD Radeon AI PRO R9700": BROKEN})
    hw.start_amd_driver_check()
    assert hw.amd_driver_warning_report() == {}


def test_a_thread_that_will_not_start_never_raises(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")

    def _cannot_start(*a, **k):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(hw.threading, "Thread", _cannot_start)
    hw.start_amd_driver_check()
    assert hw.amd_driver_warning_report() == {}


def _fake_winreg(subkeys):
    mod = types.ModuleType("winreg")
    mod.HKEY_LOCAL_MACHINE = object()
    names = list(subkeys)

    class _Key:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_key(parent, sub):
        return _Key(None if parent is mod.HKEY_LOCAL_MACHINE else sub)

    def query_value_ex(key, value):
        values = subkeys[key.name]
        if value not in values:
            raise FileNotFoundError(2, "value does not exist")
        return (values[value], 0)

    mod.OpenKey = open_key
    mod.QueryInfoKey = lambda key: (len(names), 0, 0)
    mod.EnumKey = lambda key, index: names[index]
    mod.QueryValueEx = query_value_ex
    return mod


R9700_RECORD = {
    "VendorId": 0x1002,
    "AdapterLuid": 82776,
    "Description": "AMD Radeon AI PRO R9700",
    "AdapterFamily": "AMD_NAVI48:gfx1201",
    "DriverVersion": 9007201289044972,
}


def test_registry_record_carries_the_driver_version(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setitem(
        sys.modules,
        "winreg",
        _fake_winreg({"{cd12ca95-0000-0000-0000-000000000000}": R9700_RECORD}),
    )
    assert hw._windows_amd_adapter_records_by_luid() == {
        82776: {"name": "AMD Radeon AI PRO R9700", "gfx": "gfx1201", "driver_version": BROKEN}
    }


@pytest.mark.parametrize("driver", ["garbage", 0])
def test_an_unreadable_driver_version_does_not_decline_the_map(monkeypatch, driver):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setitem(
        sys.modules,
        "winreg",
        _fake_winreg(
            {"{cd12ca95-0000-0000-0000-000000000000}": {**R9700_RECORD, "DriverVersion": driver}}
        ),
    )
    assert hw._windows_amd_adapter_records_by_luid() == {
        82776: {"name": "AMD Radeon AI PRO R9700", "gfx": "gfx1201"}
    }


def test_live_scan_reads_amd_adapters_by_name(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    seen = {}

    def _run(argv, **kw):
        seen["ps"] = argv[-1]
        return types.SimpleNamespace(
            returncode = 0,
            stdout = f"AMD Radeon AI PRO R9700\t{BROKEN}\r\nAMD Radeon RX 9070 XT\t{BROKEN}\n\n",
        )

    monkeypatch.setattr(hw.subprocess, "run", _run)
    assert hw._windows_live_amd_driver_versions() == {
        "AMD Radeon AI PRO R9700": BROKEN,
        "AMD Radeon RX 9070 XT": BROKEN,
    }
    assert "PCI\\VEN_1002*" in seen["ps"]


def test_a_failed_live_scan_is_none(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode = 1, stdout = "")
    )
    assert hw._windows_live_amd_driver_versions() is None
