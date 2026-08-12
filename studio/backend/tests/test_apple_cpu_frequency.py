# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Apple Silicon CPU frequency correction (issue #8519).

psutil <= 7.2.2 divides the pmgr voltage-state tables by 1e6 unconditionally, so
on M4 (where Apple switched the tables from Hz to kHz) it reports "4 MHz" for a
4.5 GHz part. cpu_frequency_mhz() re-reads the tables via ioreg and falls back to
rescaling psutil's value.
"""

import platform
import plistlib
import types

import pytest

from utils.hardware import hardware


def _table(freqs_raw):
    """Build a voltage-statesN-sram blob: uint32 frequency + uint32 voltage pairs."""
    blob = b""
    for raw in freqs_raw:
        blob += raw.to_bytes(4, "little") + (900).to_bytes(4, "little")
    return blob


# Raw values as the tables actually appear: M1-M3 in Hz, M4+ in kHz.
_M3_PERF_TABLE = _table([702_000_000, 2_016_000_000, 4_056_000_000])
_M4_PERF_TABLE = _table([1_050_000, 3_000_000, 4_512_000])
_M4_EFF_TABLE = _table([1_020_000, 2_592_000])
_GPU_TABLE = _table([444_000, 1_398_000])


@pytest.fixture(autouse = True)
def _reset_cache():
    hardware._apple_cpu_peak_mhz = "unprobed"
    yield
    hardware._apple_cpu_peak_mhz = "unprobed"


def _fake_psutil(
    monkeypatch,
    current,
    raises = False,
):
    def cpu_freq():
        if raises:
            raise RuntimeError("no frequency support")
        if current is None:
            return None
        return types.SimpleNamespace(current = current, min = 0.0, max = current)

    monkeypatch.setitem(
        __import__("sys").modules, "psutil", types.SimpleNamespace(cpu_freq = cpu_freq)
    )


def _fake_ioreg(monkeypatch, entries):
    def run(cmd, **kwargs):
        assert cmd[0] == "ioreg"
        return types.SimpleNamespace(stdout = plistlib.dumps(entries), returncode = 0)

    monkeypatch.setattr(hardware.subprocess, "run", run)


class TestVoltageStateFreqs:
    def test_hz_table_m3(self):
        assert hardware._voltage_state_freqs_mhz(_M3_PERF_TABLE) == [702.0, 2016.0, 4056.0]

    def test_khz_table_m4(self):
        assert hardware._voltage_state_freqs_mhz(_M4_PERF_TABLE) == [1050.0, 3000.0, 4512.0]

    def test_skips_zero_and_implausible_entries(self):
        blob = _table([0, 1, 4_512_000, 90_000_000])
        assert hardware._voltage_state_freqs_mhz(blob) == [4512.0]

    def test_truncated_trailing_bytes_ignored(self):
        assert hardware._voltage_state_freqs_mhz(_M4_PERF_TABLE + b"\x01\x02\x03") == [
            1050.0,
            3000.0,
            4512.0,
        ]

    def test_empty(self):
        assert hardware._voltage_state_freqs_mhz(b"") == []


class TestPeakFromIoregEntries:
    def test_picks_highest_cpu_cluster(self):
        entries = [
            {
                "IOObjectClass": "AppleARMIODevice",
                "voltage-states1-sram": _M4_EFF_TABLE,
                "voltage-states5-sram": _M4_PERF_TABLE,
            }
        ]
        assert hardware._peak_cpu_mhz_from_ioreg_entries(entries) == 4512.0

    def test_ignores_gpu_rails(self):
        entries = [{"voltage-states2-sram": _GPU_TABLE}]
        assert hardware._peak_cpu_mhz_from_ioreg_entries(entries) is None

    def test_ignores_unrelated_keys_and_types(self):
        entries = [
            {"voltage-states5": _M4_PERF_TABLE, "IORegistryEntryName": "pmgr"},
            "not-a-dict",
        ]
        assert hardware._peak_cpu_mhz_from_ioreg_entries(entries) is None

    def test_renumbered_tables_still_found(self):
        # M5 renumbered the table indexes; classification is by peak, not index.
        entries = [{"voltage-states13-sram": _M4_PERF_TABLE}]
        assert hardware._peak_cpu_mhz_from_ioreg_entries(entries) == 4512.0


class TestCpuFrequencyMhz:
    def test_non_apple_value_passes_through(self, monkeypatch):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
        _fake_psutil(monkeypatch, 3600.0)
        assert hardware.cpu_frequency_mhz() == 3600.0

    def test_non_apple_low_value_is_not_rescaled(self, monkeypatch):
        # A container reporting a genuinely low clock must not be multiplied.
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
        _fake_psutil(monkeypatch, 400.0)
        assert hardware.cpu_frequency_mhz() == 400.0

    def test_m4_bug_corrected_from_ioreg(self, monkeypatch):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, 4.0)
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        assert hardware.cpu_frequency_mhz() == 4512.0

    def test_m4_bug_falls_back_to_scaling_when_ioreg_fails(self, monkeypatch):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, 4.0)

        def boom(cmd, **kwargs):
            raise OSError("ioreg not found")

        monkeypatch.setattr(hardware.subprocess, "run", boom)
        assert hardware.cpu_frequency_mhz() == 4000.0

    def test_fixed_psutil_value_is_left_alone(self, monkeypatch):
        # Once psutil ships giampaolo/psutil#2824 the value is already plausible,
        # so no ioreg call and no rescale.
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, 4512.0)

        def fail(cmd, **kwargs):
            raise AssertionError("ioreg must not run for a plausible value")

        monkeypatch.setattr(hardware.subprocess, "run", fail)
        assert hardware.cpu_frequency_mhz() == 4512.0

    def test_ioreg_probe_is_cached(self, monkeypatch):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, 4.0)
        calls = []

        def run(cmd, **kwargs):
            calls.append(cmd)
            return types.SimpleNamespace(
                stdout = plistlib.dumps([{"voltage-states5-sram": _M4_PERF_TABLE}]),
                returncode = 0,
            )

        monkeypatch.setattr(hardware.subprocess, "run", run)
        assert hardware.cpu_frequency_mhz() == 4512.0
        assert hardware.cpu_frequency_mhz() == 4512.0
        assert len(calls) == 1

    def test_failed_probe_is_cached_too(self, monkeypatch):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, 4.0)
        calls = []

        def run(cmd, **kwargs):
            calls.append(cmd)
            return types.SimpleNamespace(stdout = b"", returncode = 1)

        monkeypatch.setattr(hardware.subprocess, "run", run)
        assert hardware.cpu_frequency_mhz() == 4000.0
        assert hardware.cpu_frequency_mhz() == 4000.0
        assert len(calls) == 1

    @pytest.mark.parametrize("current", [None, 0.0, -1.0, float("nan"), "fast"])
    def test_missing_or_bogus_value_returns_none(self, monkeypatch, current):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, current)
        assert hardware.cpu_frequency_mhz() is None

    def test_psutil_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
        _fake_psutil(monkeypatch, 3600.0, raises = True)
        assert hardware.cpu_frequency_mhz() is None

    def test_psutil_failure_on_apple_falls_back_to_ioreg(self, monkeypatch):
        # psutil raises on M5, where the table indexes it hardcodes are absent.
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, None, raises = True)
        _fake_ioreg(monkeypatch, [{"voltage-states13-sram": _M4_PERF_TABLE}])
        assert hardware.cpu_frequency_mhz() == 4512.0

    def test_psutil_failure_with_no_ioreg_returns_none(self, monkeypatch):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, None, raises = True)
        _fake_ioreg(monkeypatch, [])
        assert hardware.cpu_frequency_mhz() is None

    def test_psutil_without_cpu_freq_at_all_falls_back_to_ioreg(self, monkeypatch):
        # Observed on GitHub's Apple Silicon runners: psutil ships without the
        # attribute, so the call raises AttributeError rather than returning.
        import sys

        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace())
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        assert hardware.cpu_frequency_mhz() == 4512.0

    def test_zero_reading_on_apple_still_reaches_ioreg(self, monkeypatch):
        # Some M5 builds return 0.0 rather than raising.
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, 0.0)
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        assert hardware.cpu_frequency_mhz() == 4512.0

    def test_concurrent_first_calls_probe_once(self, monkeypatch):
        import concurrent.futures
        import threading
        import time

        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
        _fake_psutil(monkeypatch, 4.0)
        calls = []
        calls_lock = threading.Lock()

        def run(cmd, **kwargs):
            with calls_lock:
                calls.append(cmd)
            time.sleep(0.05)
            return types.SimpleNamespace(
                stdout = plistlib.dumps([{"voltage-states5-sram": _M4_PERF_TABLE}]),
                returncode = 0,
            )

        monkeypatch.setattr(hardware.subprocess, "run", run)
        with concurrent.futures.ThreadPoolExecutor(max_workers = 12) as pool:
            results = list(pool.map(lambda _: hardware.cpu_frequency_mhz(), range(12)))
        assert results == [4512.0] * 12
        assert len(calls) == 1


@pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64"),
    reason = "Apple Silicon only: reads this host's real IORegistry tables",
)
class TestOnRealAppleSilicon:
    """Runs unmocked on macOS CI, where the tables are the real thing."""

    def test_reported_frequency_is_plausible(self):
        mhz = hardware.cpu_frequency_mhz()
        if mhz is None:
            # Virtualised Apple Silicon (GitHub's macos-14/15 runners) ships a
            # psutil with no cpu_freq at all AND no pmgr voltage-state tables,
            # so None is the correct answer and the UI just omits the row.
            pytest.skip("neither psutil nor ioreg exposes a CPU clock on this host")
        assert hardware._MIN_PLAUSIBLE_CPU_MHZ <= mhz <= hardware._MAX_PLAUSIBLE_CPU_MHZ

    def test_ioreg_reader_agrees_with_psutil(self):
        # On M1-M3 psutil is already correct, so the ioreg reader must match it;
        # on M4+ psutil is the broken side, so compare against its x1000 rescale.
        import psutil

        peak = hardware._read_apple_cpu_peak_mhz()
        if peak is None:
            pytest.skip("no voltage-state tables exposed on this host")
        raw = psutil.cpu_freq().current
        expected = raw if raw >= hardware._MIN_PLAUSIBLE_CPU_MHZ else raw * 1000
        assert peak == pytest.approx(expected, rel = 0.15)
