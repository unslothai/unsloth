# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm Get-Counter reads: display polls share one combined PowerShell call per window; fit reads stay fresh."""

import threading
import time
import types

import pytest

from utils import gpu_memory_events
from utils.hardware import gpu_query
from utils.hardware import hardware as hw

_DED = "luid_0x00000000_0x0000d1a2_phys_0|1073741824\nluid_0x00000000_0x0000d1a3_phys_0|0\n"
_SHARED = "luid_0x00000000_0x0000d1a2_phys_0|2048\n"
_UTIL = "pid_42_luid_0x00000000_0x0000d1a2_phys_0_eng_0_engtype_3D|37.5\n"
# One Get-Counter spawn on the Radeon 8060S CI box took about 2 s.
_SPAWN_S = 2.0


class _Clock:
    """Stands in for ``hw.time``: monotonic() is a fake clock, everything else is the real module."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


def _tagged(tag, body):
    return "".join(f"{tag}|{line}\n" for line in body.splitlines())


@pytest.fixture
def counters(monkeypatch):
    monkeypatch.setattr(hw, "_rocm_win_counter_cache", {}, raising = False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu_query, "enabled", lambda: True)
    monkeypatch.setattr(gpu_query, "ttl_for", lambda kind: 3.0)
    clock = _Clock()
    monkeypatch.setattr(hw, "time", clock)
    calls = []
    state = types.SimpleNamespace(calls = calls, clock = clock, spawn_s = _SPAWN_S, drop = ())

    def run(argv, **kwargs):
        script = argv[-1]
        calls.append(script)
        clock.now += state.spawn_s
        if script.count("Get-Counter") != 1:
            raise AssertionError(script)
        if "Get-Counter -Counter" in script:
            parts = {"D": _DED, "S": _SHARED, "U": _UTIL}
            out = "".join(_tagged(t, body) for t, body in parts.items() if t not in state.drop)
        elif "GPU Engine" in script:
            out = _UTIL
        elif "Shared Usage" in script:
            out = _SHARED
        else:
            out = _DED
        return types.SimpleNamespace(returncode = 0, stdout = out or "__NONE__\n", stderr = "")

    monkeypatch.setattr(hw.subprocess, "run", run)
    return state


def _combined(calls):
    return [c for c in calls if "Get-Counter -Counter" in c]


def _read_all():
    return (
        hw._rocm_windows_perf_counter_vram_by_adapter(),
        hw._rocm_windows_perf_counter_vram_by_adapter("Shared Usage"),
        hw._rocm_windows_perf_counter_gpu_util_pct(),
        hw._rocm_windows_perf_counter_gpu_util_pct(0xD1A2),
    )


def test_back_to_back_display_polls_spawn_once(counters):
    fresh = _read_all()
    assert len(counters.calls) == 4 and not _combined(counters.calls)
    with gpu_query.display_reads(max_stale = 0):
        for _ in range(10):
            assert _read_all() == fresh
    # Ten polls of all three counters: one combined spawn, against 30 per-counter spawns before.
    assert len(counters.calls) == 4 + 1
    assert len(_combined(counters.calls)) == 1
    assert fresh[0] == [
        ("luid_0x00000000_0x0000d1a2_phys_0", 1073741824.0),
        ("luid_0x00000000_0x0000d1a3_phys_0", 0.0),
    ]
    assert fresh[1] == [("luid_0x00000000_0x0000d1a2_phys_0", 2048.0)]
    assert fresh[2] == fresh[3] == 37.5


def test_reading_outlives_its_own_probe(counters):
    # 2 s spawns, 1 s between polls: the reading lives 4 x 2 s = 8 s from completion, not the 3 s TTL.
    with gpu_query.display_reads(max_stale = 0):
        spawned_at = []
        for poll in range(20):
            before = len(counters.calls)
            _read_all()
            if len(counters.calls) > before:
                spawned_at.append(poll)
            counters.clock.now += 1.0
    assert spawned_at == [0, 8, 16]
    assert len(counters.calls) == 3


def test_window_is_the_ttl_for_fast_spawns_and_capped_for_slow_ones(counters):
    counters.spawn_s = 0.1
    with gpu_query.display_reads(max_stale = 0):
        _read_all()
        counters.clock.now += 2.9
        _read_all()
        assert len(counters.calls) == 1
        counters.clock.now += 0.1
        _read_all()
        assert len(counters.calls) == 2
        counters.spawn_s = 4.5
        counters.clock.now += 3.0
        _read_all()
        assert len(counters.calls) == 3
        counters.clock.now += 9.9
        _read_all()
        assert len(counters.calls) == 3
        counters.clock.now += 0.1
        _read_all()
        assert len(counters.calls) == 4


def test_fit_and_fresh_reads_always_spawn_their_own_counter(counters):
    with gpu_query.display_reads(max_stale = 0):
        hw._rocm_windows_perf_counter_vram_by_adapter()
    assert len(counters.calls) == 1
    for _ in range(3):
        hw._rocm_windows_perf_counter_vram_by_adapter()
    assert len(counters.calls) == 4
    with gpu_query.display_reads(max_stale = 0), gpu_query.fresh_reads():
        hw._rocm_windows_perf_counter_vram_by_adapter()
        hw._rocm_windows_perf_counter_gpu_util_pct()
    assert len(counters.calls) == 6
    assert len(_combined(counters.calls)) == 1
    assert all(c.startswith("$s=(Get-Counter '") for c in counters.calls[1:])


def test_display_cache_drops_on_memory_event_and_ttl_zero(counters, monkeypatch):
    with gpu_query.display_reads(max_stale = 0):
        hw._rocm_windows_perf_counter_vram_by_adapter()
        hw._rocm_windows_perf_counter_gpu_util_pct()
        assert len(counters.calls) == 1
        gpu_memory_events.invalidate_gpu_memory("model loaded")
        hw._rocm_windows_perf_counter_vram_by_adapter()
        assert len(counters.calls) == 2
        monkeypatch.setattr(gpu_query, "ttl_for", lambda kind: 0.0)
        hw._rocm_windows_perf_counter_vram_by_adapter()
        hw._rocm_windows_perf_counter_vram_by_adapter()
        assert len(counters.calls) == 4
        assert len(_combined(counters.calls)) == 2


def test_failed_combined_reads_fall_back_and_are_never_cached(counters, monkeypatch):
    def failing(argv, **kwargs):
        counters.calls.append(argv[-1])
        return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "boom")

    monkeypatch.setattr(hw.subprocess, "run", failing)
    with gpu_query.display_reads(max_stale = 0):
        assert hw._rocm_windows_perf_counter_vram_by_adapter() is None
        assert hw._rocm_windows_perf_counter_vram_by_adapter() is None
        assert hw._rocm_windows_perf_counter_gpu_util_pct() is None
    # One combined attempt, then own calls: nothing failed is served.
    assert len(_combined(counters.calls)) == 1
    assert len(counters.calls) == 4
    assert hw._rocm_win_counter_cache["display"][3] == {}
    counters.clock.now += hw._ROCM_WIN_COUNTER_EMPTY_RETRY_S - 1
    with gpu_query.display_reads(max_stale = 0):
        assert hw._rocm_windows_perf_counter_vram_by_adapter() is None
    assert len(_combined(counters.calls)) == 1
    # The empty reading still expires, and a GPU memory event drops it too.
    counters.clock.now += 1
    with gpu_query.display_reads(max_stale = 0):
        assert hw._rocm_windows_perf_counter_vram_by_adapter() is None
        gpu_memory_events.invalidate_gpu_memory("model loaded")
        assert hw._rocm_windows_perf_counter_vram_by_adapter() is None
    assert len(_combined(counters.calls)) == 3


def test_a_hung_combined_call_is_not_retried_by_every_read(counters, monkeypatch):
    def hung(argv, **kwargs):
        counters.calls.append(argv[-1])
        counters.clock.now += 5.0
        raise hw.subprocess.TimeoutExpired(argv, 5)

    monkeypatch.setattr(hw.subprocess, "run", hung)
    with gpu_query.display_reads(max_stale = 0):
        assert _read_all() == (None, None, None, None)
    # Base ran 4 own calls (20 s); one combined attempt rides on top (25 s), not one per read (40 s).
    assert len(_combined(counters.calls)) == 1
    assert len(counters.calls) == 5


def test_a_counter_missing_from_the_combined_reading_reads_its_own(counters):
    counters.drop = ("U",)
    fresh = _read_all()
    with gpu_query.display_reads(max_stale = 0):
        for _ in range(2):
            assert _read_all() == fresh
    combined = _combined(counters.calls)
    own_util = [c for c in counters.calls[4:] if c not in combined and "GPU Engine" in c]
    assert len(combined) == 1
    assert len(own_util) == 4
    assert len(counters.calls) == 4 + 1 + 4


def test_concurrent_display_polls_share_one_spawn(counters, monkeypatch):
    real_run = hw.subprocess.run
    gate = threading.Event()

    def slow(argv, **kwargs):
        gate.wait(5)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(hw.subprocess, "run", slow)
    results = []

    def poll():
        with gpu_query.display_reads(max_stale = 0):
            results.append(_read_all())

    threads = [threading.Thread(target = poll) for _ in range(4)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(10)
    assert len(results) == 4 and len(set(map(repr, results))) == 1
    assert len(counters.calls) == 1


def test_combined_script_keeps_each_counter_path_and_format():
    ps = hw._rocm_windows_combined_counter_ps()
    assert ps.count("Get-Counter") == 1
    for _, (tag, path, value) in hw._ROCM_WIN_DISPLAY_COUNTERS.items():
        assert f"'{path}'" in ps
        assert f"'{tag}|{{0}}|{{1}}' -f $_.InstanceName,{value}" in ps
    split = hw._rocm_windows_split_combined(
        _tagged("D", _DED) + _tagged("S", _SHARED) + _tagged("U", _UTIL) + "noise\n"
    )
    assert split == {
        ("vram", "Dedicated Usage"): _DED,
        ("vram", "Shared Usage"): _SHARED,
        ("util", "3D"): _UTIL,
    }
