# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""``patch_psutil_cpu_freq`` -- Apple Silicon M4+ CPU frequency units (#8519).

psutil <= 7.2.2 divides the pmgr voltage-state tables by 1e6 unconditionally,
but Apple switched them from Hz to kHz on M4, so ``psutil.cpu_freq()`` comes
back ~1000x too small. Runs on every OS: the platform is faked, so Linux and
Windows CI exercise the same paths a Mac would.
"""

from __future__ import annotations

import platform
import plistlib
import sys
import types

import pytest

from unsloth import import_fixes as IF


def _table(raw_freqs):
    """A voltage-statesN-sram blob: uint32 frequency + uint32 voltage pairs."""
    blob = b""
    for raw in raw_freqs:
        blob += raw.to_bytes(4, "little") + (900).to_bytes(4, "little")
    return blob


# Raw values as the tables actually appear: M1-M3 in Hz, M4+ in kHz.
_M3_PERF_TABLE = _table([702_000_000, 2_016_000_000, 4_056_000_000])
_M4_PERF_TABLE = _table([1_050_000, 3_000_000, 4_512_000])
_M4_EFF_TABLE = _table([1_020_000, 2_592_000])
_GPU_TABLE = _table([444_000, 1_398_000])


def _scpufreq(current, minimum = 0.0, maximum = 0.0):
    import psutil
    return psutil._ntuples.scpufreq(current, minimum, maximum)


@pytest.fixture(autouse = True)
def _reset_probe_cache():
    IF._apple_cpu_freq_range = "unprobed"
    yield
    IF._apple_cpu_freq_range = "unprobed"


@pytest.fixture
def fake_m4(monkeypatch):
    """Pretend this interpreter is running on Apple Silicon."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")


def _fake_ioreg(monkeypatch, entries):
    import subprocess
    def run(cmd, **kwargs):
        assert cmd[0] == "ioreg"
        return types.SimpleNamespace(stdout = plistlib.dumps(entries), returncode = 0)

    monkeypatch.setattr(subprocess, "run", run)


def _install_fake_psutil(monkeypatch, sample):
    """Swap psutil.cpu_freq for one returning `sample` (value or callable)."""
    import psutil

    def cpu_freq(percpu = False):
        return sample(percpu) if callable(sample) else sample

    monkeypatch.setattr(psutil, "cpu_freq", cpu_freq)
    return psutil


class TestVoltageStateDecoding:
    def test_hz_table_m3(self):
        assert IF._apple_voltage_state_freqs_mhz(_M3_PERF_TABLE) == [702.0, 2016.0, 4056.0]

    def test_khz_table_m4(self):
        assert IF._apple_voltage_state_freqs_mhz(_M4_PERF_TABLE) == [1050.0, 3000.0, 4512.0]

    def test_zero_and_implausible_entries_dropped(self):
        assert IF._apple_voltage_state_freqs_mhz(_table([0, 1, 4_512_000, 90_000_000])) == [4512.0]

    def test_trailing_partial_entry_ignored(self):
        assert IF._apple_voltage_state_freqs_mhz(_M4_PERF_TABLE + b"\x01\x02\x03") == [
            1050.0,
            3000.0,
            4512.0,
        ]

    def test_empty_blob(self):
        assert IF._apple_voltage_state_freqs_mhz(b"") == []


class TestFreqRangeFromEntries:
    def test_spans_all_cpu_clusters(self):
        entries = [
            {
                "voltage-states1-sram": _M4_EFF_TABLE,
                "voltage-states5-sram": _M4_PERF_TABLE,
            }
        ]
        assert IF._apple_cpu_freq_range_from_ioreg_entries(entries) == (1020.0, 4512.0)

    def test_gpu_rails_excluded(self):
        assert IF._apple_cpu_freq_range_from_ioreg_entries([{"voltage-states2-sram": _GPU_TABLE}]) is None

    def test_renumbered_tables_still_found(self):
        # M5 renumbered the indexes; classification is by peak, not by index.
        entries = [{"voltage-states13-sram": _M4_PERF_TABLE}]
        assert IF._apple_cpu_freq_range_from_ioreg_entries(entries) == (1050.0, 4512.0)

    def test_non_dict_and_non_bytes_values_ignored(self):
        entries = ["not-a-dict", {"voltage-states5-sram": "not-bytes", "IORegistryEntryName": "pmgr"}]
        assert IF._apple_cpu_freq_range_from_ioreg_entries(entries) is None


class TestPatchApplication:
    def test_no_op_off_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        import psutil

        before = psutil.cpu_freq
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq is before

    def test_no_op_on_intel_mac(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        import psutil

        before = psutil.cpu_freq
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq is before

    def test_patch_is_idempotent(self, monkeypatch, fake_m4):
        psutil = _install_fake_psutil(monkeypatch, _scpufreq(4.0, 1.0, 4.0))
        IF.patch_psutil_cpu_freq()
        first = psutil.cpu_freq
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq is first

    def test_m4_reading_corrected_from_ioreg(self, monkeypatch, fake_m4):
        psutil = _install_fake_psutil(monkeypatch, _scpufreq(4.0, 1.0, 4.0))
        _fake_ioreg(
            monkeypatch,
            [{"voltage-states1-sram": _M4_EFF_TABLE, "voltage-states5-sram": _M4_PERF_TABLE}],
        )
        IF.patch_psutil_cpu_freq()
        result = psutil.cpu_freq()
        assert (result.current, result.min, result.max) == (4512.0, 1020.0, 4512.0)

    def test_falls_back_to_rescaling_without_ioreg(self, monkeypatch, fake_m4):
        import subprocess

        psutil = _install_fake_psutil(monkeypatch, _scpufreq(4.0, 1.0, 4.0))

        def boom(cmd, **kwargs):
            raise OSError("ioreg not found")

        monkeypatch.setattr(subprocess, "run", boom)
        IF.patch_psutil_cpu_freq()
        result = psutil.cpu_freq()
        assert (result.current, result.min, result.max) == (4000.0, 1000.0, 4000.0)

    def test_plausible_reading_untouched(self, monkeypatch, fake_m4):
        # M1-M3 today, and every chip once psutil ships giampaolo/psutil#2824.
        import subprocess

        psutil = _install_fake_psutil(monkeypatch, _scpufreq(4056.0, 702.0, 4056.0))
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: pytest.fail("ioreg must not run for a plausible value")
        )
        IF.patch_psutil_cpu_freq()
        result = psutil.cpu_freq()
        assert (result.current, result.min, result.max) == (4056.0, 702.0, 4056.0)

    def test_percpu_list_is_corrected_elementwise(self, monkeypatch, fake_m4):
        samples = [_scpufreq(4.0, 1.0, 4.0), _scpufreq(2.5, 1.0, 2.5)]
        psutil = _install_fake_psutil(
            monkeypatch, lambda percpu: list(samples) if percpu else samples[0]
        )
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        IF.patch_psutil_cpu_freq()
        percpu = psutil.cpu_freq(percpu = True)
        assert isinstance(percpu, list) and len(percpu) == 2
        assert all(sample.current == 4512.0 for sample in percpu)
        assert psutil.cpu_freq().current == 4512.0

    def test_return_type_and_signature_preserved(self, monkeypatch, fake_m4):
        import psutil as real_psutil

        psutil = _install_fake_psutil(monkeypatch, _scpufreq(4.0, 1.0, 4.0))
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        IF.patch_psutil_cpu_freq()
        assert isinstance(psutil.cpu_freq(), real_psutil._ntuples.scpufreq)
        assert psutil.cpu_freq.__name__ == "cpu_freq"

    @pytest.mark.parametrize("sample", [None, "junk", 0.0])
    def test_unusable_readings_pass_through(self, monkeypatch, fake_m4, sample):
        value = sample if sample in (None, "junk") else _scpufreq(0.0, 0.0, 0.0)
        psutil = _install_fake_psutil(monkeypatch, value)
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq() == value

    def test_psutil_exception_is_covered_by_ioreg(self, monkeypatch, fake_m4):
        # psutil raises on M5, where the table indexes it hardcodes are absent.
        def boom(percpu = False):
            raise RuntimeError("no voltage-states table at the expected index")

        import psutil

        monkeypatch.setattr(psutil, "cpu_freq", boom)
        _fake_ioreg(monkeypatch, [{"voltage-states13-sram": _M4_PERF_TABLE}])
        IF.patch_psutil_cpu_freq()
        result = psutil.cpu_freq()
        assert (result.current, result.min, result.max) == (4512.0, 1050.0, 4512.0)

    def test_psutil_exception_still_raises_without_ioreg(self, monkeypatch, fake_m4):
        import psutil
        import subprocess

        def boom(percpu = False):
            raise RuntimeError("no voltage-states table at the expected index")

        monkeypatch.setattr(psutil, "cpu_freq", boom)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
        IF.patch_psutil_cpu_freq()
        with pytest.raises(RuntimeError):
            psutil.cpu_freq()

    def test_percpu_exception_is_not_synthesised(self, monkeypatch, fake_m4):
        # A per-core breakdown is not something the shared tables can supply.
        import psutil

        def boom(percpu = False):
            raise RuntimeError("no voltage-states table at the expected index")

        monkeypatch.setattr(psutil, "cpu_freq", boom)
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        IF.patch_psutil_cpu_freq()
        with pytest.raises(RuntimeError):
            psutil.cpu_freq(percpu = True)
        with pytest.raises(RuntimeError):
            psutil.cpu_freq(True)

    def test_concurrent_first_calls_probe_once(self, monkeypatch, fake_m4):
        import concurrent.futures
        import subprocess
        import threading
        import time

        calls = []
        calls_lock = threading.Lock()

        def run(cmd, **kwargs):
            with calls_lock:
                calls.append(cmd)
            time.sleep(0.05)
            return types.SimpleNamespace(
                stdout = plistlib.dumps([{"voltage-states5-sram": _M4_PERF_TABLE}]), returncode = 0
            )

        psutil = _install_fake_psutil(monkeypatch, _scpufreq(4.0, 1.0, 4.0))
        monkeypatch.setattr(subprocess, "run", run)
        IF.patch_psutil_cpu_freq()
        with concurrent.futures.ThreadPoolExecutor(max_workers = 12) as pool:
            results = list(pool.map(lambda _: psutil.cpu_freq().current, range(12)))
        assert results == [4512.0] * 12
        assert len(calls) == 1

    def test_probe_runs_once_across_calls(self, monkeypatch, fake_m4):
        import subprocess

        calls = []

        def run(cmd, **kwargs):
            calls.append(cmd)
            return types.SimpleNamespace(
                stdout = plistlib.dumps([{"voltage-states5-sram": _M4_PERF_TABLE}]), returncode = 0
            )

        psutil = _install_fake_psutil(monkeypatch, _scpufreq(4.0, 1.0, 4.0))
        monkeypatch.setattr(subprocess, "run", run)
        IF.patch_psutil_cpu_freq()
        for _ in range(5):
            assert psutil.cpu_freq().current == 4512.0
        assert len(calls) == 1


@pytest.mark.skipif(
    not (sys.platform == "darwin" and platform.machine() == "arm64"),
    reason = "Apple Silicon only: reads this host's real IORegistry tables",
)
class TestOnRealAppleSilicon:
    def test_reported_frequency_is_plausible(self):
        import psutil

        IF.patch_psutil_cpu_freq()
        current = psutil.cpu_freq().current
        assert IF._APPLE_MIN_PLAUSIBLE_CPU_MHZ <= current <= IF._APPLE_MAX_PLAUSIBLE_CPU_MHZ

    def test_ioreg_reader_agrees_with_unaffected_psutil(self):
        # On M1-M3 psutil is already right, so our reader must match it. On M4+
        # psutil is the broken one, so compare against its x1000 rescale instead.
        import psutil

        freq_range = IF._apple_cpu_freq_range_mhz()
        if freq_range is None:
            pytest.skip("ioreg voltage-state tables unavailable on this host")
        raw = psutil.cpu_freq().current
        expected = raw if raw >= IF._APPLE_MIN_PLAUSIBLE_CPU_MHZ else raw * 1000
        assert freq_range[1] == pytest.approx(expected, rel = 0.15)
