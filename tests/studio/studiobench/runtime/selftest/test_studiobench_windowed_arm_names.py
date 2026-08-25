# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A MISTYPED FLAG MUST NOT COST A BROWSER AND TWO STUDIO SERVERS.

`--windowed-arm` names the arm that is allowed the windowed readiness gate, and the name is checked
against the arms the run actually has. That check used to happen after both Studio installs had
been launched, the pacer bound and the browser opened, and the `SystemExit` it raises for
`--windowed-arm treatments` was raised from a place with no cleanup around it: the `finally` that
calls `bundle.close()`, `pacer.stop()`, `stop_studio()` and cancels the watchdog does not begin
until the cell loop far below. So the typo exited the process and left the heavy children running,
holding their ports.

It is a pure argument check -- it needs nothing that has been launched -- so it now runs before any
of it. These tests pin both halves: the check itself, and the fact that it happens FIRST.

    python -m pytest tests/studio/studiobench/runtime/selftest/test_studiobench_windowed_arm_names.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench import __main__ as M  # noqa: E402


# ── the check itself ────────────────────────────────────────────────


def test_the_arms_of_the_run_are_accepted():
    assert M._windowed_arms("treatment", ["base", "treatment"]) == {"treatment"}
    assert M._windowed_arms(" base , treatment ", ["base", "treatment"]) == {"base", "treatment"}


def test_nothing_named_is_nothing_gated():
    assert M._windowed_arms("", ["base"]) == set()
    assert M._windowed_arms(None, ["base"]) == set()


def test_a_typo_is_refused_and_says_which_arms_exist():
    with pytest.raises(SystemExit) as raised:
        M._windowed_arms("treatments", ["base", "treatment"])
    assert "['treatments']" in str(raised.value)
    assert "['base', 'treatment']" in str(raised.value)


def test_naming_the_treatment_arm_of_a_run_that_has_no_treatment_is_refused():
    """Without `--ab` there is one arm. Naming the other one is not a harmless no-op: the caller
    believes a gate is in force that nothing in the run will ever apply."""
    with pytest.raises(SystemExit):
        M._windowed_arms("treatment", ["base"])


# ── and it happens before anything is started ───────────────────────


def test_a_bad_arm_name_is_refused_before_any_process_is_started(monkeypatch):
    """THE DEFECT. Every entry point that starts something is trapped here, so if the refusal moves
    back below any of them this fails with the trap's own error instead of the SystemExit.

    The watchdog is the first of them in `run`, then the Studio install or the health check on an
    attached one, then the pacer, then the browser. None of them is reached.
    """
    from studiobench import pacer as pacer_mod
    from studiobench.runtime import browser as browser_mod
    from studiobench.runtime import lifecycle

    started: list = []

    def _trap(name):
        def _boom(*_a, **_kw):
            started.append(name)
            raise AssertionError(f"{name} ran before the arm names were checked")

        return _boom

    monkeypatch.setattr(browser_mod, "install_wall_clock_watchdog", _trap("the watchdog"))
    monkeypatch.setattr(browser_mod, "launch", _trap("the browser"))
    monkeypatch.setattr(lifecycle, "install_studio", _trap("the Studio install"))
    monkeypatch.setattr(lifecycle, "launch_studio", _trap("the Studio launch"))
    monkeypatch.setattr(lifecycle, "wait_for_healthz", _trap("the health check"))
    monkeypatch.setattr(pacer_mod, "Pacer", _trap("the pacer"))

    with pytest.raises(SystemExit) as raised:
        M.main(["--windowed-arm", "treatments", "--attach", "http://127.0.0.1:1"])

    assert "--windowed-arm names ['treatments']" in str(raised.value), str(raised.value)
    assert started == [], started
