# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#18005: one adapter's engine utilization from an in-process PDH query, with a stand-in pdh.dll."""

import ctypes
from ctypes import wintypes

import pytest

from utils.hardware import hardware as hw

_LUID = 0x15369
_MORE_DATA = 0x800007D2


class _Value(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class _Item(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _Value)]


class _FakePdh:
    def __init__(self):
        self.samples = [
            ("pid_1_luid_0x00000000_0x00015369_phys_0_eng_2_engtype_compute 0", 0, 96.5)
        ]
        self.paths, self.collects, self.closed = [], 0, []
        self.fail_collect = False
        self._names = []

    def PdhOpenQueryW(self, source, user, ref):
        ref._obj.value = 11
        return 0

    def PdhAddEnglishCounterW(self, query, path, user, ref):
        self.paths.append(path)
        ref._obj.value = 22
        return 0

    def PdhCollectQueryData(self, query):
        self.collects += 1
        return 0xC0000BBA if self.fail_collect else 0

    def PdhCloseQuery(self, query):
        self.closed.append(query.value if hasattr(query, "value") else query)
        return 0

    def PdhGetFormattedCounterArrayW(self, counter, fmt, size, count, buf):
        need = ctypes.sizeof(_Item) * len(self.samples)
        if buf is None:
            size._obj.value, count._obj.value = need, len(self.samples)
            return _MORE_DATA
        self._names = [ctypes.create_unicode_buffer(n) for n, _, _ in self.samples]
        items = (_Item * len(self.samples))()
        for i, ((_, status, value), name) in enumerate(zip(self.samples, self._names)):
            items[i].szName = ctypes.cast(name, wintypes.LPWSTR)
            items[i].FmtValue.CStatus = status
            items[i].FmtValue.doubleValue = value
        ctypes.memmove(buf, items, need)
        count._obj.value = len(self.samples)
        return 0


class _Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s


@pytest.fixture
def pdh(monkeypatch):
    fake = _FakePdh()
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_rocm_windows_pdh", lambda: fake)
    monkeypatch.setattr(hw, "_rocm_win_pdh_queries", {})
    monkeypatch.setattr(hw, "time", _Clock())
    monkeypatch.setattr(
        hw.subprocess, "run", lambda *a, **k: pytest.fail("the per-adapter read must spawn nothing")
    )
    return fake


def test_reads_the_adapters_busiest_engine_in_process(pdh):
    assert hw._rocm_windows_perf_counter_gpu_util_pct(_LUID) == 96.5
    assert pdh.paths == ["\\GPU Engine(*luid_0x00000000_0x00015369*)\\Utilization Percentage"]


def test_one_query_per_adapter_and_a_rate_needs_an_interval(pdh):
    hw._rocm_windows_perf_counter_gpu_util_pct(_LUID)
    first = pdh.collects
    hw._rocm_windows_perf_counter_gpu_util_pct(_LUID)
    assert pdh.collects == first
    hw.time.now += hw._ROCM_WIN_PDH_MIN_INTERVAL_S
    pdh.samples = [("pid_1_luid_0x00000000_0x00015369_phys_0_eng_2_engtype_compute 0", 0, 40.0)]
    assert hw._rocm_windows_perf_counter_gpu_util_pct(_LUID) == 40.0
    assert pdh.collects == first + 1
    assert len(pdh.paths) == 1


def test_invalid_instances_and_other_adapters_are_not_readings(pdh):
    pdh.samples = [
        ("pid_1_luid_0x00000000_0x00015369_phys_0_eng_2_engtype_compute 0", 0xC0000BC6, 99.0),
        ("pid_2_luid_0x00000000_0x00153690_phys_0_eng_2_engtype_compute 0", 0, 88.0),
        ("pid_3_luid_0x00000000_0x00015369_phys_0_eng_0_engtype_3d", 1, 3.0),
    ]
    assert hw._rocm_windows_perf_counter_gpu_util_pct(_LUID) == 3.0


def test_a_failed_collect_drops_the_query_and_reopens_next_poll(pdh):
    hw._rocm_windows_perf_counter_gpu_util_pct(_LUID)
    pdh.fail_collect = True
    hw.time.now += 5
    assert hw._rocm_windows_perf_counter_gpu_util_pct(_LUID) is None
    assert hw._rocm_win_pdh_queries == {}
    assert pdh.closed
    pdh.fail_collect = False
    hw.time.now += 5
    assert hw._rocm_windows_perf_counter_gpu_util_pct(_LUID) == 96.5
    assert len(pdh.paths) == 2


def test_no_pdh_is_unknown_not_a_crash(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_rocm_win_pdh_queries", {})

    def missing():
        raise OSError("pdh.dll")

    monkeypatch.setattr(hw, "_rocm_windows_pdh", missing)
    assert hw._rocm_windows_perf_counter_gpu_util_pct(_LUID) is None
