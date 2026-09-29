# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The watchdog may not hard-exit a wait still inside its own timeout, or the run reports
only "wedged somewhere". So the budget must restart on `wall_kick()`, and no two
turn-scaled waits may run without a kick between them. `playwright_chat_ui.py` drives
Playwright at import, so its half is read from source; the watchdog is run for real.
"""

from __future__ import annotations

import ast
import contextlib
import io
import sys
import threading
import time
from pathlib import Path
from unittest import mock

STUDIO_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDIO_TESTS))

import _playwright_robust as robust  # noqa: E402
from _playwright_robust import _WallClockWatchdog  # noqa: E402

CHAT_UI = STUDIO_TESTS / "playwright_chat_ui.py"
CHAT_UI_SRC = CHAT_UI.read_text(encoding = "utf-8")
CHAT_UI_TREE = ast.parse(CHAT_UI_SRC)


def _fired(
    budget_s,
    *,
    kicks = (),
    cancel_after = None,
    run_for = None,
):
    """Run a watchdog for `run_for` seconds, kicking at each offset in `kicks`."""
    fired = threading.Event()
    watchdog = _WallClockWatchdog(budget_s, fired.set).start()
    started = time.monotonic()
    try:
        for at in kicks:
            time.sleep(max(0.0, started + at - time.monotonic()))
            watchdog.kick()
        if cancel_after is not None:
            time.sleep(max(0.0, started + cancel_after - time.monotonic()))
            watchdog.cancel()
        time.sleep(max(0.0, started + run_for - time.monotonic()))
    finally:
        watchdog.cancel()
    return fired.is_set()


# Every deadline in these three has 0.6s of slack, the last one included: a kick at 0.8s
# moves expiry to 1.8s and observation stops at 1.2s, so only a 600ms overshoot could
# decide the result rather than the watchdog.
def test_an_unkicked_watchdog_still_fires_at_its_budget():
    assert _fired(0.4, run_for = 1.6)


def test_a_kick_restarts_the_budget():
    """The defect: setup ran before the wait, so the wait inherited what setup left."""
    assert not _fired(1.0, kicks = (0.4, 0.8), run_for = 1.2)


def test_a_cancelled_watchdog_does_not_fire():
    assert not _fired(0.8, cancel_after = 0.2, run_for = 1.4)


def test_a_watchdog_that_expires_during_start_still_exits():
    # `install_wall_clock_watchdog` builds the handle its own expiry callback reads, so at
    # a deadline of 0 the thread can reach that callback before the name is bound. It then
    # dies of NameError in the daemon thread and the run silently loses its watchdog.
    codes = []
    with mock.patch.object(robust.os, "_exit", codes.append):
        robust.install_wall_clock_watchdog(0.0, label = "ui")
        time.sleep(0.5)
    assert codes == [2], codes


def test_a_total_cap_is_a_ceiling_no_kick_can_move():
    """What makes an outer bound around this process a sum instead of a guess.

    Without it a caller sizing a backstop has nothing to size against: every kick moves
    the deadline, so the exit lands at a wall-clock time the caller cannot predict."""
    fired = threading.Event()
    watchdog = _WallClockWatchdog(10.0, fired.set, total_deadline_s = 0.7).start()
    started = time.monotonic()
    try:
        while time.monotonic() - started < 1.4:
            time.sleep(0.05)
            watchdog.kick()  # kicking throughout must not push past the ceiling
    finally:
        watchdog.cancel()
    assert fired.is_set()
    assert watchdog.at_ceiling()


def test_without_a_total_cap_nothing_is_at_the_ceiling():
    watchdog = _WallClockWatchdog(10.0, lambda: None)
    watchdog.kick()
    assert not watchdog.at_ceiling()


def _watchdog_message(kick):
    """The line a real `install_wall_clock_watchdog` prints on expiry, minus the exit."""
    watchdog = robust.install_wall_clock_watchdog(30.0, label = "ui")
    watchdog.cancel()
    if kick:
        watchdog.kick()
    buf = io.StringIO()
    with mock.patch.object(robust.os, "_exit"), contextlib.redirect_stderr(buf):
        watchdog._on_expiry()
    return buf.getvalue().splitlines()[0]


def test_the_message_names_what_actually_ran_out():
    # The scripts that never kick are measuring the whole run, not inactivity; telling
    # their reader to look for a step sends them after one that never existed.
    assert "hit 30s wall-clock deadline" in _watchdog_message(kick = False)
    assert "30s with no step reported" in _watchdog_message(kick = True)


def _chat_ui_wall_timeout_s(
    turn_timeout_ms,
    load_timeout_ms = 180_000,
    fetch_timeout_ms = 30_000,
):
    """Evaluate the script's own WALL_TIMEOUT_S expression at a given set of budgets."""
    wanted = {
        "TURN_TIMEOUT_MS",
        "LOAD_FETCH_TIMEOUT_MS",
        "FETCH_TIMEOUT_MS",
        "_WALL_FLOOR_S",
        "_LONGEST_WAIT_S",
        "WALL_TIMEOUT_S",
    }
    body = [
        node
        for node in CHAT_UI_TREE.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in wanted
    ]
    assert {n.targets[0].id for n in body} == wanted, "the wall-timeout constants moved"
    env = {
        "STUDIO_UI_TURN_TIMEOUT_MS": str(turn_timeout_ms),
        "STUDIO_UI_LOAD_TIMEOUT_MS": str(load_timeout_ms),
        "STUDIO_UI_FETCH_TIMEOUT_MS": str(fetch_timeout_ms),
    }
    ns = {"os": type("_os", (), {"environ": env})}
    exec(compile(ast.Module(body = body, type_ignores = []), str(CHAT_UI), "exec"), ns)
    return ns["WALL_TIMEOUT_S"], ns["_LONGEST_WAIT_S"]


def test_the_wall_budget_outlasts_the_longest_single_wait():
    # 540000 is studio-mac-ui-smoke.yml's; 180000 is every other runner's default.
    for turn_timeout_ms in (180_000, 540_000):
        wall, longest_wait = _chat_ui_wall_timeout_s(turn_timeout_ms)
        assert wall >= longest_wait + 120, (turn_timeout_ms, wall, longest_wait)


def test_a_raised_fetch_budget_also_raises_the_wall():
    # Every budget in the max is an env var, so none of them may be left out on the
    # grounds that no lane raises it today.
    wall, longest_wait = _chat_ui_wall_timeout_s(180_000, fetch_timeout_ms = 900_000)
    assert longest_wait == 900.0
    assert wall >= 900.0 + 120


def test_a_raised_load_budget_also_raises_the_wall():
    # The Kaggle lane sets STUDIO_UI_LOAD_TIMEOUT_MS to 600000 and leaves the turn timeout
    # at its default, so the load fetch, not the turn, is the longest wait there.
    wall, longest_wait = _chat_ui_wall_timeout_s(180_000, load_timeout_ms = 600_000)
    assert longest_wait == 600.0
    assert wall >= 600.0 + 120


def test_the_linux_default_keeps_the_budget_it_had():
    assert _chat_ui_wall_timeout_s(180_000)[0] == 720.0


def _turn_scaled_wait_lines():
    """Lines of the waits whose timeout is TURN_TIMEOUT_MS or a multiple of it."""
    lines = []
    for node in ast.walk(CHAT_UI_TREE):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "timeout":
                continue
            names = {n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)}
            if "TURN_TIMEOUT_MS" in names:
                lines.append(kw.value.lineno)
    return sorted(lines)


def test_every_pair_of_turn_scaled_waits_is_separated_by_a_kick():
    kicks = sorted(
        node.lineno
        for node in ast.walk(CHAT_UI_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "wall_kick"
    )
    waits = _turn_scaled_wait_lines()
    assert len(waits) >= 5, waits
    for first, second in zip(waits, waits[1:]):
        assert any(
            first < k < second for k in kicks
        ), f"the waits at lines {first} and {second} share one watchdog budget"


# Named steps and per-step budgets.


def test_a_step_budget_is_a_ceiling_kicks_inside_the_step_cannot_move():
    """A step that keeps reporting progress but never finishes still ends, and ends as that step."""
    fired = threading.Event()
    watchdog = _WallClockWatchdog(10.0, fired.set).start()
    started = time.monotonic()
    try:
        watchdog.begin_step("slow step", budget_s = 0.6)
        while time.monotonic() - started < 1.4:
            time.sleep(0.05)
            watchdog.kick()
    finally:
        watchdog.cancel()
    assert fired.is_set()
    assert watchdog.at_step_ceiling()
    assert watchdog.step_name == "slow step"


def test_the_next_step_gets_its_own_budget():
    # Each of the three steps runs 0.5s against a 0.8s budget: none may inherit the last.
    fired = threading.Event()
    watchdog = _WallClockWatchdog(10.0, fired.set).start()
    try:
        for name in ("one", "two", "three"):
            watchdog.begin_step(name, budget_s = 0.8)
            time.sleep(0.5)
    finally:
        watchdog.cancel()
    assert not fired.is_set()


def test_a_step_without_a_budget_keeps_the_inactivity_budget():
    watchdog = _WallClockWatchdog(10.0, lambda: None)
    watchdog.begin_step("unbudgeted")
    assert not watchdog.at_step_ceiling()
    assert watchdog.step_budget_s is None


def _named_step_message(budget_s, *, expire_step):
    watchdog = robust.install_wall_clock_watchdog(30.0, label = "ui")
    watchdog.cancel()
    watchdog.begin_step("theme toggle x3", budget_s = budget_s)
    if expire_step:
        with watchdog._lock:
            watchdog._deadline = watchdog._step_ceiling
    buf = io.StringIO()
    with mock.patch.object(robust.os, "_exit"), contextlib.redirect_stderr(buf):
        watchdog._on_expiry()
    return buf.getvalue().splitlines()[0]


def test_the_message_names_the_step_that_ran_out():
    assert "step 'theme toggle x3' used its whole 90s budget" in _named_step_message(
        90.0, expire_step = True
    )
    assert "30s with no progress in step 'theme toggle x3'" in _named_step_message(
        None, expire_step = False
    )


def test_an_uncaught_exception_is_reported_with_its_step():
    watchdog = _WallClockWatchdog(10.0, lambda: None)
    original = sys.excepthook
    seen = []
    try:
        sys.excepthook = lambda *a: seen.append(a[0])
        robust.report_failing_step(watchdog, label = "ui")
        watchdog.begin_step("model picker: open + drive search bar")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            try:
                raise TimeoutError("Timeout 30000ms exceeded.\nCall log: ...")
            except TimeoutError:
                sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook = original
    assert seen == [TimeoutError], "the previous hook must still print the traceback"
    line = buf.getvalue().strip()
    assert line.startswith("[ui] FAIL in step 'model picker: open + drive search bar' after ")
    assert line.endswith("TimeoutError: Timeout 30000ms exceeded."), line


def test_no_step_begun_adds_nothing():
    watchdog = _WallClockWatchdog(10.0, lambda: None)
    original = sys.excepthook
    try:
        sys.excepthook = lambda *a: None
        robust.report_failing_step(watchdog, label = "ui")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            sys.excepthook(ValueError, ValueError("x"), None)
    finally:
        sys.excepthook = original
    assert buf.getvalue() == ""


def test_wait_until_returns_the_first_truthy_value():
    values = iter([None, 0, [], "ready"])
    assert (
        robust.wait_until(lambda: next(values), timeout_s = 5, what = "x", interval_s = 0.01) == "ready"
    )


def test_wait_until_names_what_it_waited_for():
    started = time.monotonic()
    try:
        robust.wait_until(lambda: 0, timeout_s = 0.3, what = "monitor resumes polling", interval_s = 0.05)
    except TimeoutError as exc:
        assert "monitor resumes polling" in str(exc)
        assert "last value 0" in str(exc)
    else:
        raise AssertionError("wait_until returned for a predicate that never came true")
    assert time.monotonic() - started < 2.0


def test_wait_until_pauses_through_the_page_when_given_one():
    # Event handlers only run inside a Playwright call, so the pause must be one.
    page = mock.Mock()
    values = iter([False, True])
    robust.wait_until(lambda: next(values), timeout_s = 5, what = "x", interval_s = 0.25, page = page)
    page.wait_for_timeout.assert_called_once_with(250.0)


def test_step_budgets_stretch_but_never_shrink(monkeypatch):
    monkeypatch.delenv(robust.STEP_BUDGET_SCALE_ENV, raising = False)
    assert robust.step_budget_s(60) == 60.0
    monkeypatch.setenv(robust.STEP_BUDGET_SCALE_ENV, "3")
    assert robust.step_budget_s(60) == 180.0
    monkeypatch.setenv(robust.STEP_BUDGET_SCALE_ENV, "0.1")
    assert robust.step_budget_s(60) == 60.0
    monkeypatch.setenv(robust.STEP_BUDGET_SCALE_ENV, "fast")
    assert robust.step_budget_s(60) == 60.0
