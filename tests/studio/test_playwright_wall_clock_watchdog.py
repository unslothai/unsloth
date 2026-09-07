# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The watchdog may not hard-exit a wait still inside its own timeout, or the run reports
only "wedged somewhere". So the budget must restart on `wall_kick()`, and no two
turn-scaled waits may run without a kick between them. `playwright_chat_ui.py` drives
Playwright at import, so its half is read from source; the watchdog is run for real.
"""

from __future__ import annotations

import ast
import sys
import threading
import time
from pathlib import Path

STUDIO_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDIO_TESTS))

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


# Every offset leaves 0.6s of slack, so an oversleeping runner cannot decide the result.
def test_an_unkicked_watchdog_still_fires_at_its_budget():
    assert _fired(0.4, run_for = 1.6)


def test_a_kick_restarts_the_budget():
    """The defect: setup ran before the wait, so the wait inherited what setup left."""
    assert not _fired(1.0, kicks = (0.4, 0.8), run_for = 1.6)


def test_a_cancelled_watchdog_does_not_fire():
    assert not _fired(0.8, cancel_after = 0.2, run_for = 1.4)


def _chat_ui_wall_timeout_s(turn_timeout_ms):
    """Evaluate the script's own WALL_TIMEOUT_S expression at a given turn timeout."""
    wanted = {"TURN_TIMEOUT_MS", "_WALL_FLOOR_S", "_LONGEST_WAIT_S", "WALL_TIMEOUT_S"}
    body = [
        node
        for node in CHAT_UI_TREE.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in wanted
    ]
    assert {n.targets[0].id for n in body} == wanted, "the wall-timeout constants moved"
    env = {"STUDIO_UI_TURN_TIMEOUT_MS": str(turn_timeout_ms)}
    ns = {"os": type("_os", (), {"environ": env})}
    exec(compile(ast.Module(body = body, type_ignores = []), str(CHAT_UI), "exec"), ns)
    return ns["WALL_TIMEOUT_S"], ns["_LONGEST_WAIT_S"]


def test_the_wall_budget_outlasts_the_longest_single_wait():
    # 540000 is studio-mac-ui-smoke.yml's; 180000 is every other runner's default.
    for turn_timeout_ms in (180_000, 540_000):
        wall, longest_wait = _chat_ui_wall_timeout_s(turn_timeout_ms)
        assert wall >= longest_wait + 120, (turn_timeout_ms, wall, longest_wait)


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
