# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD Windows GPU utilization reads Get-Counter values the same on comma-decimal hosts (de-DE)."""

import types

import pytest

from utils.hardware import gpu_query
from utils.hardware import hardware as hw

_INVARIANT = "CookedValue.ToString([cultureinfo]::InvariantCulture)"
_ENGINES = (
    ("pid_42_luid_0x00000000_0x0000d1a2_phys_0_eng_0_engtype_3D", "12.5"),
    ("pid_43_luid_0x00000000_0x0000d1a2_phys_0_eng_0_engtype_3D", "25.25"),
)


def _de_de_powershell(samples):
    """PowerShell on a de-DE host: -f prints 12,5 unless the script asks for the invariant culture."""
    calls = []

    def run(argv, **kwargs):
        script = argv[-1]
        calls.append(script)
        out = ""
        for name, value in samples:
            if _INVARIANT not in script:
                value = value.replace(".", ",")
            line = f"{name}|{value}\n"
            out += f"U|{line}" if "Get-Counter -Counter" in script else line
        return types.SimpleNamespace(returncode = 0, stdout = out or "__NONE__\n", stderr = "")

    return run, calls


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_rocm_win_counter_cache", {}, raising = False)

    def use(samples, combined):
        monkeypatch.setattr(gpu_query, "enabled", lambda: combined)
        monkeypatch.setattr(gpu_query, "display_cache_allowed", lambda: combined)
        monkeypatch.setattr(gpu_query, "ttl_for", lambda kind: 3.0)
        run, calls = _de_de_powershell(samples)
        monkeypatch.setattr(hw.subprocess, "run", run)
        return calls

    return use


@pytest.mark.parametrize("combined", [True, False], ids = ["combined", "own-call"])
def test_de_de_host_reports_fractional_utilization(windows, combined):
    calls = windows(_ENGINES, combined)
    assert hw._rocm_windows_perf_counter_gpu_util_pct() == 37.8
    assert ("Get-Counter -Counter" in calls[0]) is combined


def test_de_de_host_sums_only_the_selected_adapter(windows):
    windows(
        _ENGINES + (("pid_44_luid_0x00000000_0x0000beef_phys_0_eng_0_engtype_3D", "50.5"),), False
    )
    assert hw._rocm_windows_perf_counter_gpu_util_pct(0xD1A2) == 37.8
    assert hw._rocm_windows_perf_counter_gpu_util_pct(0xBEEF) == 50.5


def test_comma_decimal_output_is_skipped_not_misread():
    # What the old script printed on de-DE: no sample parses, so no reading rather than a wrong one.
    out = "".join(f"{name}|{value.replace('.', ',')}\n" for name, value in _ENGINES)
    assert _parse(out) is None


def test_invariant_infinity_and_nan_samples_are_dropped():
    name = _ENGINES[0][0]
    out = f"{name}|Infinity\n{name}|-Infinity\n{name}|NaN\n{name}|-1\n{name}|12.5\n"
    assert _parse(out) == 12.5
    assert _parse(f"{name}|Infinity\n{name}|NaN\n") is None


def test_both_scripts_format_utilization_invariant():
    assert _INVARIANT in hw._rocm_windows_combined_counter_ps()
    assert hw._ROCM_WIN_DISPLAY_COUNTERS[("util", "3D")][2].endswith(_INVARIANT)


def _parse(stdout):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hw.platform, "system", lambda: "Windows")
        mp.setattr(hw, "_rocm_windows_counter_stdout", lambda key, ps: stdout)
        return hw._rocm_windows_perf_counter_gpu_util_pct()
