# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A bad --windowed-arm name is refused before any process is started, so nothing is left running."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench import __main__ as M  # noqa: E402


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


def test_a_bad_arm_name_is_refused_before_any_process_is_started(monkeypatch):
    """The refusal must precede the watchdog, install, pacer and browser, which each start something."""
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
    monkeypatch.setattr(lifecycle, "install_studio", _trap("the Unsloth install"))
    monkeypatch.setattr(lifecycle, "launch_studio", _trap("the Unsloth launch"))
    monkeypatch.setattr(lifecycle, "wait_for_healthz", _trap("the health check"))
    monkeypatch.setattr(pacer_mod, "Pacer", _trap("the pacer"))

    with pytest.raises(SystemExit) as raised:
        M.main(["--windowed-arm", "treatments", "--attach", "http://127.0.0.1:1"])

    assert "--windowed-arm names ['treatments']" in str(raised.value), str(raised.value)
    assert started == [], started
