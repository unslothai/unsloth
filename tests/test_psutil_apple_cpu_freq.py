# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

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


def _scpufreq(
    current,
    minimum = 0.0,
    maximum = 0.0,
):
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
        assert (
            IF._apple_cpu_freq_range_from_ioreg_entries([{"voltage-states2-sram": _GPU_TABLE}])
            is None
        )

    def test_renumbered_tables_still_found(self):
        # M5 renumbered the indexes; classification is by peak, not by index.
        entries = [{"voltage-states13-sram": _M4_PERF_TABLE}]
        assert IF._apple_cpu_freq_range_from_ioreg_entries(entries) == (1050.0, 4512.0)

    def test_non_dict_and_non_bytes_values_ignored(self):
        entries = [
            "not-a-dict",
            {"voltage-states5-sram": "not-bytes", "IORegistryEntryName": "pmgr"},
        ]
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

    def test_no_op_when_psutil_has_no_cpu_freq(self, monkeypatch, fake_m4):
        # GitHub's Apple Silicon runners ship exactly this psutil: no cpu_freq attribute at all.
        import psutil

        monkeypatch.delattr(psutil, "cpu_freq", raising = False)
        IF.patch_psutil_cpu_freq()
        assert not hasattr(psutil, "cpu_freq")

    def test_caller_argument_mistakes_keep_their_error(self, monkeypatch, fake_m4):
        # The wrapper stands in for psutil globally, so a TypeError from a bad
        # call must not be mistaken for psutil declining to read the clock.
        import psutil

        def cpu_freq(percpu = False):
            raise RuntimeError("psutil declines")

        monkeypatch.setattr(psutil, "cpu_freq", cpu_freq)
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq().current == 4512.0
        with pytest.raises(TypeError):
            psutil.cpu_freq(unknown = True)
        with pytest.raises(TypeError):
            psutil.cpu_freq(False, False)

    def test_probe_lock_exists_before_any_call(self):
        # A lazily built lock is two locks when two threads reach it at once, and then neither excludes the other.
        import threading
        assert isinstance(IF._apple_cpu_freq_lock, type(threading.Lock()))

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
            subprocess,
            "run",
            lambda *a, **k: pytest.fail("ioreg must not run for a plausible value"),
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
    def test_unusable_readings_without_tables_pass_through(self, monkeypatch, fake_m4, sample):
        value = sample if sample in (None, "junk") else _scpufreq(0.0, 0.0, 0.0)
        psutil = _install_fake_psutil(monkeypatch, value)
        _fake_ioreg(monkeypatch, [])
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq() == value

    def test_psutil_exception_is_covered_by_ioreg(self, monkeypatch, fake_m4):
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

        # psutil raises on M5, where the table indexes it hardcodes are absent.
        def boom(percpu = False):
            raise RuntimeError("no voltage-states table at the expected index")

        monkeypatch.setattr(psutil, "cpu_freq", boom)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
        IF.patch_psutil_cpu_freq()
        with pytest.raises(RuntimeError):
            psutil.cpu_freq()

    def test_percpu_exception_is_covered_too(self, monkeypatch, fake_m4):
        # macOS has no per-core clock, so psutil's own percpu answer there is a one-element list; the stand-in keeps
        # that shape for both call forms.
        import psutil

        def boom(percpu = False):
            raise RuntimeError("no voltage-states table at the expected index")

        monkeypatch.setattr(psutil, "cpu_freq", boom)
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        IF.patch_psutil_cpu_freq()
        for call in (lambda: psutil.cpu_freq(percpu = True), lambda: psutil.cpu_freq(True)):
            result = call()
            assert isinstance(result, list) and len(result) == 1
            assert result[0].current == 4512.0

    def test_empty_percpu_list_is_covered(self, monkeypatch, fake_m4):
        # giampaolo/psutil#2382 returns [] when the clock is undeterminable.
        psutil = _install_fake_psutil(monkeypatch, lambda percpu: [] if percpu else None)
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq().current == 4512.0
        percpu = psutil.cpu_freq(percpu = True)
        assert isinstance(percpu, list) and percpu[0].current == 4512.0

    def test_none_stays_none_without_tables(self, monkeypatch, fake_m4):
        import subprocess

        psutil = _install_fake_psutil(monkeypatch, lambda percpu: [] if percpu else None)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq() is None
        assert psutil.cpu_freq(percpu = True) == []

    @pytest.mark.parametrize("bogus", [0.0, -1.0, float("nan")])
    def test_unusable_apple_reading_recovers_from_tables(self, monkeypatch, fake_m4, bogus):
        psutil = _install_fake_psutil(monkeypatch, _scpufreq(bogus, 0.0, 0.0))
        _fake_ioreg(monkeypatch, [{"voltage-states5-sram": _M4_PERF_TABLE}])
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq().current == 4512.0

    @pytest.mark.parametrize("bogus", [0.0, -1.0])
    def test_unusable_apple_reading_is_left_alone_without_tables(self, monkeypatch, fake_m4, bogus):
        import subprocess

        psutil = _install_fake_psutil(monkeypatch, _scpufreq(bogus, 0.0, 0.0))
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
        IF.patch_psutil_cpu_freq()
        assert psutil.cpu_freq().current == bogus

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
def _raw_apple_reading():
    """This host's own psutil reading, or None when it has none to give.

    psutil gates cpu_freq on a runtime probe on macOS, so on Apple Silicon the
    attribute can be missing outright (virtualised runners) or present and
    raising (M5, whose tables are not at the indexes psutil hardcodes). Both are
    supported hosts, so a test that needs a raw reading skips rather than fails.
    """
    import psutil

    reader = getattr(psutil, "cpu_freq", None)
    if reader is None:
        return None
    try:
        sample = reader()
    except Exception:
        return None
    current = getattr(sample, "current", None)
    return current if isinstance(current, (int, float)) and current > 0 else None


class TestOnRealAppleSilicon:
    def test_reported_frequency_is_plausible(self):
        import psutil

        IF.patch_psutil_cpu_freq()
        reader = getattr(psutil, "cpu_freq", None)
        if reader is None:
            # Nothing was wrapped, so there is nothing to assert about.
            pytest.skip("psutil exposes no cpu_freq on this host")
        try:
            sample = reader()
        except Exception as exception:
            # psutil declined and the tables were unreadable too, so the wrapper correctly re-raised rather than
            # inventing a number.
            pytest.skip(f"no CPU clock available on this host ({exception})")
        if sample is None:
            pytest.skip("psutil reports the clock as undeterminable on this host")
        assert IF._APPLE_MIN_PLAUSIBLE_CPU_MHZ <= sample.current <= IF._APPLE_MAX_PLAUSIBLE_CPU_MHZ

    def test_ioreg_reader_agrees_with_unaffected_psutil(self):
        # On M1-M3 psutil is already right, so our reader must match it. On M4+
        # psutil is the broken one, so compare against its x1000 rescale instead.
        freq_range = IF._apple_cpu_freq_range_mhz()
        if freq_range is None:
            pytest.skip("ioreg voltage-state tables unavailable on this host")
        raw = _raw_apple_reading()
        if raw is None:
            pytest.skip("psutil has no reading of its own on this host to compare against")
        expected = raw if raw >= IF._APPLE_MIN_PLAUSIBLE_CPU_MHZ else raw * 1000
        assert freq_range[1] == pytest.approx(expected, rel = 0.15)
