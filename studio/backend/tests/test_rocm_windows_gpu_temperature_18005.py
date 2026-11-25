# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#18005: Windows ROCm temperature from D3DKMT_ADAPTER_PERFDATA, with a stand-in gdi32."""

import ctypes

import pytest

from utils.hardware import hardware as hw


class _FakeGdi:
    def __init__(
        self,
        deci_c = 820,
        open_status = 0,
        query_status = 0,
    ):
        self.deci_c, self.open_status, self.query_status = deci_c, open_status, query_status
        self.opened, self.queried, self.closed = [], [], []

    def D3DKMTOpenAdapterFromLuid(self, ref):
        arg = ref._obj
        self.opened.append((arg.AdapterLuid.LowPart, arg.AdapterLuid.HighPart))
        arg.hAdapter = 7
        return self.open_status

    def D3DKMTQueryAdapterInfo(self, ref):
        q = ref._obj
        self.queried.append((q.hAdapter, q.Type, q.PrivateDriverDataSize))
        if self.query_status:
            return self.query_status
        buf = (ctypes.c_char * q.PrivateDriverDataSize).from_address(q.pPrivateDriverData)
        buf[56:60] = int(self.deci_c).to_bytes(4, "little")
        return 0

    def D3DKMTCloseAdapter(self, ref):
        self.closed.append(ref._obj.value)
        return 0


@pytest.fixture
def gdi(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    fake = _FakeGdi()
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: fake, raising = False)
    return fake


def test_reads_tenths_of_a_degree_at_offset_56(gdi):
    assert hw._rocm_windows_d3dkmt_temperature_c(0x15369) == 82.0
    assert gdi.opened == [(0x15369, 0)]
    assert gdi.queried == [(7, 62, 64)]
    assert gdi.closed == [7]


def test_high_luid_half_is_passed_signed(gdi):
    hw._rocm_windows_d3dkmt_temperature_c((0xFFFFFFFE << 32) | 0x10)
    assert gdi.opened == [(0x10, -2)]


@pytest.mark.parametrize("deci_c", [0, 1501])
def test_no_sensor_or_garbage_is_unknown(gdi, deci_c):
    gdi.deci_c = deci_c
    assert hw._rocm_windows_d3dkmt_temperature_c(0x15369) is None


def test_failed_open_or_query_is_unknown_and_closes(gdi):
    gdi.query_status = -1
    assert hw._rocm_windows_d3dkmt_temperature_c(0x15369) is None
    assert gdi.closed == [7]
    gdi.open_status = -1
    assert hw._rocm_windows_d3dkmt_temperature_c(0x15369) is None
    assert gdi.closed == [7]


def test_no_luid_or_not_windows_never_loads_gdi(monkeypatch):
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: pytest.fail("gdi32 loaded"), raising = False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    assert hw._rocm_windows_d3dkmt_temperature_c(None) is None
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    assert hw._rocm_windows_d3dkmt_temperature_c(0x15369) is None


def test_device_entry_carries_its_own_adapters_temperature(monkeypatch):
    monkeypatch.setattr(
        hw, "_rocm_windows_d3dkmt_temperature_c", lambda luid: {0xA: 41.5}.get(luid)
    )
    dev = {"index": 0, "visible_ordinal": 0, "total_gb": 16.0, "used_gb": 4.0, "luid": 0xA}
    entry = hw._rocm_windows_device_payload_entry(hw.DeviceType.CUDA, dev, 12.0)
    assert entry["temperature_c"] == 41.5
    assert entry["power_draw_w"] is None
    assert "luid" not in entry
    assert (
        hw._rocm_windows_device_payload_entry(hw.DeviceType.CUDA, {**dev, "luid": None}, None)[
            "temperature_c"
        ]
        is None
    )
