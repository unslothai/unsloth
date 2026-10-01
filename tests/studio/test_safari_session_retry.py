# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The native Safari leg retries opening its session, and nothing else.

`Composer (macos-latest, safari)` failed on unrelated pull requests with
`SessionNotCreatedException: The session timed out while connecting to a Safari instance`,
raised by `webdriver.Safari()` before a page was loaded. That is the hosted runner. The
retry is bounded, covers only session creation, and re-raises the last failure, so a Safari
that never starts still fails the leg.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _playwright_robust import open_session_with_retry  # noqa: E402


class _NotCreated(Exception):
    pass


def _factory(failures: int):
    calls = {"n": 0}

    def make():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise _NotCreated(f"timed out {calls['n']}")
        return "session"

    return make, calls


def test_a_slow_start_is_retried_until_a_session_opens():
    make, calls = _factory(failures = 2)
    slept = []
    assert (
        open_session_with_retry(make, retry_on = _NotCreated, sleep = slept.append, log = lambda _m: None)
        == "session"
    )
    assert calls["n"] == 3
    assert slept == [5.0, 5.0]


def test_a_safari_that_never_starts_still_fails():
    make, calls = _factory(failures = 10)
    with pytest.raises(_NotCreated, match = "timed out 3"):
        open_session_with_retry(
            make, retry_on = _NotCreated, sleep = lambda _s: None, log = lambda _m: None
        )
    assert calls["n"] == 3, "the retry must stay bounded"


def test_other_errors_are_not_retried():
    calls = {"n": 0}

    def make():
        calls["n"] += 1
        raise ValueError("a real bug")

    with pytest.raises(ValueError):
        open_session_with_retry(
            make, retry_on = _NotCreated, sleep = lambda _s: None, log = lambda _m: None
        )
    assert calls["n"] == 1


def test_the_safari_driver_opens_its_session_through_the_retry_and_only_there():
    tree = ast.parse((HERE / "selenium_composer_safari.py").read_text(encoding = "utf-8"))
    direct = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Safari"
    ]
    assert not direct, "webdriver.Safari() is called directly again, outside the retry"
    wrapped = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "open_session_with_retry"
    ]
    assert len(wrapped) == 1
    retry_on = {k.arg: k.value for k in wrapped[0].keywords}.get("retry_on")
    assert (
        getattr(retry_on, "id", None) == "SessionNotCreatedException"
    ), "only a failure to create the session may be retried"
