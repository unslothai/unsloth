# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NPU load from the Windows GPU Engine counters: every host reading is pinned, none is real."""

import time

import pytest

from utils.hardware import hardware as hw

# 0x1a5c9, the shape Get-PnpDeviceProperty returns DEVPKEY_GPU_LUID in.
_NPU_LUID = 107977
_COUNTERS = (
    # Two processes on engine 0 add up; engine 1 is a separate engine, so the busiest one wins.
    "pid_10_luid_0x00000000_0x0001a5c9_phys_0_eng_0_engtype_Compute|30\n"
    "pid_11_luid_0x00000000_0x0001a5c9_phys_0_eng_0_engtype_Compute|12.5\n"
    "pid_11_luid_0x00000000_0x0001a5c9_phys_0_eng_1_engtype_Compute|20\n"
    # A GPU's engine, wildcard-matched by a looser filter, is not the NPU.
    "pid_12_luid_0x00000000_0x00017778_phys_0_eng_0_engtype_3D|99\n"
)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_npu_load_cache", None, raising = False)
    monkeypatch.setattr(hw, "_npu_luid", None, raising = False)
    monkeypatch.setattr(hw, "_npu_luid_at", 0.0, raising = False)
    monkeypatch.setattr(hw, "_npu_hardware", {"present": True, "supported": True}, raising = False)
    clock = _Clock()
    monkeypatch.setattr(hw, "time", clock)
    state = {"luid": f"{_NPU_LUID}\n", "counters": _COUNTERS, "calls": []}

    def run(ps):
        state["calls"].append(ps)
        clock.now += 1.0
        if "Get-PnpDeviceProperty" in ps:
            return state["luid"]
        if "GPU Engine" in ps:
            return state["counters"]
        raise AssertionError(ps)

    monkeypatch.setattr(hw, "_rocm_windows_counter_stdout_uncached", run)
    state["clock"] = clock
    return state


def test_npu_load_is_its_busiest_engine_summed_over_processes(host):
    assert hw.npu_load_percent() == 42.5
    probe, sample = host["calls"]
    assert "DEV_17F0" in probe
    # The counter path is narrowed to the NPU's own adapter.
    assert "luid_0x00000000_0x0001a5c9" in sample.lower()


def test_no_supported_npu_never_spawns_a_probe(host, monkeypatch):
    monkeypatch.setattr(hw, "_npu_hardware", {"present": True, "supported": False})
    assert hw.npu_load_percent() is None
    assert host["calls"] == []


def test_off_windows_there_is_no_load_source(host, monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    assert hw.npu_load_percent() is None
    assert host["calls"] == []


def test_an_adapter_missing_from_the_counters_is_unknown_not_idle(host):
    host["counters"] = "pid_12_luid_0x00000000_0x00017778_phys_0_eng_0_engtype_3D|99\n"
    assert hw.npu_load_percent() is None


def test_no_luid_for_the_npu_is_unknown(host):
    host["luid"] = None
    assert hw.npu_load_percent() is None
    assert all("GPU Engine" not in ps for ps in host["calls"])


def test_a_missing_luid_is_looked_up_once_a_minute(host):
    host["luid"] = None
    for _ in range(5):
        assert hw.npu_load_percent() is None
        host["clock"].now += 10.0
    assert sum("Get-PnpDeviceProperty" in ps for ps in host["calls"]) == 1
    host["clock"].now += 60.0
    hw.npu_load_percent()
    assert sum("Get-PnpDeviceProperty" in ps for ps in host["calls"]) == 2


def test_an_idle_npu_reads_zero(host):
    host["counters"] = "pid_10_luid_0x00000000_0x0001a5c9_phys_0_eng_0_engtype_Compute|0\n"
    assert hw.npu_load_percent() == 0.0


def test_an_infinite_sample_is_not_a_full_load(host):
    host["counters"] = (
        "pid_10_luid_0x00000000_0x0001a5c9_phys_0_eng_0_engtype_Compute|Infinity\n"
        "pid_10_luid_0x00000000_0x0001a5c9_phys_0_eng_1_engtype_Compute|7\n"
    )
    assert hw.npu_load_percent() == 7.0


def test_polls_inside_the_window_share_one_sample(host):
    assert hw.npu_load_percent() == 42.5
    spawned = len(host["calls"])
    host["clock"].now += 1.0
    assert hw.npu_load_percent() == 42.5
    assert len(host["calls"]) == spawned
    host["clock"].now += 30.0
    hw.npu_load_percent()
    # The LUID is resolved once; later windows only re-sample the counters.
    assert len(host["calls"]) == spawned + 1


def test_an_idle_npu_missing_from_the_counters_keeps_its_luid(host):
    # No model loaded: Windows lists no engine instances for the NPU, which is not a failure.
    host["counters"] = "__NONE__\n"
    assert hw.npu_load_percent() is None
    host["clock"].now += 30.0
    assert hw.npu_load_percent() is None
    assert sum("Get-PnpDeviceProperty" in ps for ps in host["calls"]) == 1
    # A driver restart can hand the NPU a new LUID, so a long silence rechecks it.
    host["clock"].now += 60.0
    hw.npu_load_percent()
    host["clock"].now += 30.0
    hw.npu_load_percent()
    assert sum("Get-PnpDeviceProperty" in ps for ps in host["calls"]) == 2


def test_a_luid_that_left_the_counters_is_looked_up_again(host):
    assert hw.npu_load_percent() == 42.5
    host["counters"] = None
    host["clock"].now += 30.0
    assert hw.npu_load_percent() is None
    host["counters"] = _COUNTERS
    host["clock"].now += 30.0
    assert hw.npu_load_percent() == 42.5
    assert sum("Get-PnpDeviceProperty" in ps for ps in host["calls"]) == 2


def test_system_route_reports_the_npu_load(monkeypatch):
    import main
    import utils.hardware

    monkeypatch.setattr(
        main,
        "_get_cached_system_gpu_info",
        lambda logger, **kwargs: ({"available": False}, {"available": False}),
    )
    monkeypatch.setattr(utils.hardware, "npu_load_percent", lambda: 17.5)
    assert main.get_system_info(current_subject = "test")["npu_load_percent"] == 17.5
