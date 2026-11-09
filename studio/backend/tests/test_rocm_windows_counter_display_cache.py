# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm Get-Counter reads: display polls share one PowerShell call per TTL; fit reads stay fresh."""

import types

import pytest

from utils import gpu_memory_events
from utils.hardware import gpu_query
from utils.hardware import hardware as hw

_VRAM = "luid_0x00000000_0x0000d1a2_phys_0|1073741824\nluid_0x00000000_0x0000d1a3_phys_0|0\n"
_UTIL = "pid_42_luid_0x00000000_0x0000d1a2_phys_0_eng_0_engtype_3D|37.5\n"


@pytest.fixture
def counters(monkeypatch):
    monkeypatch.setattr(hw, "_rocm_win_counter_cache", {}, raising = False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu_query, "enabled", lambda: True)
    calls = []

    def run(argv, **kwargs):
        script = argv[-1]
        calls.append(script)
        if "GPU Engine" in script:
            out = _UTIL
        elif "Shared Usage" in script:
            out = "luid_0x00000000_0x0000d1a2_phys_0|2048\n"
        else:
            out = _VRAM
        return types.SimpleNamespace(returncode = 0, stdout = out, stderr = "")

    monkeypatch.setattr(hw.subprocess, "run", run)
    return calls


def _read_all():
    return (
        hw._rocm_windows_perf_counter_vram_by_adapter(),
        hw._rocm_windows_perf_counter_vram_by_adapter("Shared Usage"),
        hw._rocm_windows_perf_counter_gpu_util_pct(),
        hw._rocm_windows_perf_counter_gpu_util_pct(0xD1A2),
    )


def test_display_polls_spawn_each_counter_once_per_ttl(counters):
    fresh = _read_all()
    assert len(counters) == 4
    with gpu_query.display_reads(max_stale = 0):
        for _ in range(5):
            assert _read_all() == fresh
    # One spawn each for dedicated, shared and util; util serves both LUID filters.
    assert len(counters) == 4 + 3
    assert fresh[0] == [
        ("luid_0x00000000_0x0000d1a2_phys_0", 1073741824.0),
        ("luid_0x00000000_0x0000d1a3_phys_0", 0.0),
    ]
    assert fresh[1] == [("luid_0x00000000_0x0000d1a2_phys_0", 2048.0)]
    assert fresh[2] == fresh[3] == 37.5


def test_fit_and_fresh_reads_always_spawn(counters):
    with gpu_query.display_reads(max_stale = 0):
        hw._rocm_windows_perf_counter_vram_by_adapter()
    assert len(counters) == 1
    for _ in range(3):
        hw._rocm_windows_perf_counter_vram_by_adapter()
    assert len(counters) == 4
    with gpu_query.display_reads(max_stale = 0), gpu_query.fresh_reads():
        hw._rocm_windows_perf_counter_vram_by_adapter()
        hw._rocm_windows_perf_counter_gpu_util_pct()
    assert len(counters) == 6


def test_display_cache_expires_and_drops_on_memory_event(counters, monkeypatch):
    with gpu_query.display_reads(max_stale = 0):
        hw._rocm_windows_perf_counter_vram_by_adapter()
        gpu_memory_events.invalidate_gpu_memory("model loaded")
        hw._rocm_windows_perf_counter_vram_by_adapter()
        assert len(counters) == 2
        monkeypatch.setattr(gpu_query, "ttl_for", lambda kind: 0.0)
        hw._rocm_windows_perf_counter_vram_by_adapter()
        assert len(counters) == 3


def test_failed_counter_reads_are_never_cached(counters, monkeypatch):
    def failing(argv, **kwargs):
        counters.append(argv[-1])
        return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "boom")

    monkeypatch.setattr(hw.subprocess, "run", failing)
    with gpu_query.display_reads(max_stale = 0):
        assert hw._rocm_windows_perf_counter_vram_by_adapter() is None
        assert hw._rocm_windows_perf_counter_vram_by_adapter() is None
        assert hw._rocm_windows_perf_counter_gpu_util_pct() is None
    assert len(counters) == 3
