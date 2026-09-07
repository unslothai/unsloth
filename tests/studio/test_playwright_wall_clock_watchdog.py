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
