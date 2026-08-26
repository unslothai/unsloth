# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two harness defects recorded as workspace task #102, held closed.

DEFECT ONE: the census and the parity digest ran INSIDE the measured window. Nineteen
querySelectorAll passes over ~195,000 elements plus a 5.6 MB structural serialisation, charged to
whichever action they happened to follow. It reported `delete_message` at 14.3 fps when the action
costs 49.0, and `message_menu` at 17.1 when it costs 73.8 -- an instrument whose own cost grows
with exactly the quantity under investigation, biasing every reading in the direction that makes
the standing DOM look smaller than it is.

DEFECT TWO: the "New chat" click fell back to `page.goto` without telling anyone. A goto is a full
document navigation -- bundle re-execution, runtime rehydration -- and it was being timed as
though it were the client-side subtree rebuild `thread_reopen` exists to measure. That produced
thread_reopen at 6.0 fps, a number about a page load quoted as a number about a thread.

Both are ordering and reporting properties, so both are testable without a browser, which is why
they are asserted here rather than left to be noticed again in a payload.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.runtime.types import ActionContext, Cell, Slot  # noqa: E402
from studiobench.scene import actions as A  # noqa: E402
from studiobench.scene.schedule import Scene, SceneRunner  # noqa: E402


class _Recorder:
    def __init__(self):
        self.rows = []

    def emit(self, row):
        self.rows.append(row)


class _Window:
    def __init__(self, log, name):
        self._log = log
        self.name = name
        self.notes = {}
        self.duration_ms = 1.0

    def __enter__(self):
        self._log.append(("open", self.name))
        return self

    def __exit__(self, *exc):
        self._log.append(("close", self.name))
        return False

    def note(self, key, value):
        self.notes[key] = value


def _runner(order: list):
    cell = Cell(cell_id = "r10K.base.rep0", rung = "10K", rung_tokens = 10_000)

    def _open(name, kind):
        return _Window(order, name)

    runner = SceneRunner(
        cell = cell,
        page = None,
        cdp = None,
        dom = None,
        recorder = _Recorder(),
        open_window = _open,
        log = lambda _m: None,
    )
    runner._census = lambda: (order.append(("census", "")), {"elements": 195_631})[1]
    runner._parity = lambda: (order.append(("parity", "")), {"parity_attempted": True})[1]
    return runner


# ── defect one: the observation is outside the window ───────────────


def test_the_census_and_the_digest_are_taken_after_the_action_window_closes():
    """THE REGRESSION TEST FOR THE 14.3-vs-49.0 fps READING.

    Ordering, asserted directly: both observations must appear in the log AFTER the close of the
    window they describe. If either moves back inside, its cost lands on the action again.
    """
    order: list = []
    runner = _runner(order)
    scene = Scene(name = "one", slots = [Slot(action = "keystroke", t_start_ms = 0, budget_ms = 50)])
    runner.run(scene, time.monotonic())

    names = [kind for kind, _ in order]
    close_at = names.index("close")
    assert "census" in names and "parity" in names, "the observations were dropped entirely"
    assert names.index("census") > close_at, "the census is still inside the measured window"
    assert names.index("parity") > close_at, "the parity digest is still inside the window"


def test_the_action_row_still_carries_both_observations():
    """Moving them out must not lose them. The row is where every consumer reads them from."""
    order: list = []
    runner = _runner(order)
    scene = Scene(name = "one", slots = [Slot(action = "keystroke", t_start_ms = 0, budget_ms = 50)])
    rows = runner.run(scene, time.monotonic())
    assert rows[0]["census"] == {"elements": 195_631}
    assert rows[0]["parity"] == {"parity_attempted": True}
    assert rows[0]["observation_outside_window"] is True


def test_the_gap_windows_census_is_taken_before_the_gap_opens():
    """A gap window measures the page doing nothing but stream. A 195,000-element walk at the end
    of it is not nothing, and it cannot be moved to AFTER the close either: the gap ends exactly
    when the next slot is due, so a walk there would push the action past its own slot and turn an
    instrument cost into a missed slot."""
    order: list = []
    runner = _runner(order)
    runner._gap_window("stream:gap12", until_ms = 300, t0 = time.monotonic())
    names = [kind for kind, _ in order]
    assert names.index("census") < names.index("open")


# ── defect two: the silent substitution ─────────────────────────────


class _Page:
    """A page whose New chat button is present but refuses to be clicked, which is the real
    condition: the sidebar's sticky group label covers it, so Playwright's actionability retries
    burn their whole timeout on an element that is visible, enabled and stable."""

    def __init__(self, clickable: bool):
        self.clickable = clickable
        self.goto_calls: list[str] = []

    def query_selector(self, _selector):
        page = self

        class _Handle:
            def click(self, timeout = None):
                if not page.clickable:
                    raise TimeoutError("element is covered by another element")

        return _Handle()

    def goto(self, url, **_kwargs):
        self.goto_calls.append(url)

    def evaluate(
        self,
        script = "",
        *_a,
        **_k,
    ):
        # The end of the thread, which `thread_reopen` reads out of the DOM before it touches
        # anything so that it has a string to recognise the rebuilt thread by. It has to be a
        # string here or the action refuses on that precondition and never reaches the transition
        # this file is about.
        if "data-role=" in script and "user" in script:
            return "studiobench turn 8: continue with unit 3"
        # Otherwise a thread with messages in it, so `thread_reopen` gets past its own precondition
        # and reaches the transition under test.
        return 18

    def wait_for_timeout(self, _ms):
        return None


def _ctx(page) -> ActionContext:
    return ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = "r10K.base.rep0", rung = "10K", rung_tokens = 10_000),
        window = None,
        args = {"thread_id": "t1", "base_url": "http://localhost:1"},
        budget_ms = 30_000,
        dom = None,
        log = lambda _m: None,
    )


def test_a_clickable_control_reports_the_click_path():
    got = A._click_or_navigate(_ctx(_Page(True)), "button", "http://x/")
    assert got.ok is True
    assert got.path == "click"
    assert got.navigated is False


def test_a_substituted_navigation_is_reported_as_a_navigation_not_as_a_click():
    """The whole defect in one assertion: the old function returned `True` here and the caller had
    no way to learn that a completely different operation had been performed."""
    page = _Page(False)
    got = A._click_or_navigate(_ctx(page), "button", "http://x/")
    assert got.ok is True
    assert got.path == "navigate"
    assert got.navigated is True
    assert got.reason, "the navigation carries no explanation of why the click failed"
    assert page.goto_calls == ["http://x/"]


def test_thread_reopen_refuses_to_report_a_timing_for_a_page_navigation():
    """LOUD, AND NOT A MEASUREMENT. `ran = False` rather than a plausible-looking number.

    An `expect_ok = False` would have left the timing in the row, and a timing in a row is a
    number somebody eventually averages. The action did not happen, so it reports that it did not
    happen, and `ActionResult.__post_init__` empties `timings` for it.
    """
    page = _Page(False)
    result = A.thread_reopen(_ctx(page))
    assert result.ran is False
    assert result.timings == {}
    assert "not a thread rebuild" in (result.reason or "")
    assert "navigation" in (result.reason or "")


def test_thread_reopen_says_so_in_the_log_as_well_as_in_the_row():
    """A payload nobody opens is not where this should first become visible."""
    lines: list[str] = []
    ctx = _ctx(_Page(False))
    ctx.log = lines.append
    A.thread_reopen(ctx)
    assert any("NOT MEASURED" in line for line in lines), lines


# ── the action bar does not exist while a reply is being written ────


class _RunningPage:
    """A page whose reply finishes after `runs_for` polls."""

    def __init__(self, runs_for: int):
        self.runs_for = runs_for
        self.polls = 0
        self.waits = 0

    def evaluate(self, script, *_a, **_k):
        if "isRunning" in script:
            self.polls += 1
            return self.polls <= self.runs_for
        return 18

    def wait_for_timeout(self, ms):
        # Actually sleeps, like Playwright's. A no-op here lets the bounded wait spin thousands of
        # times inside its deadline, which exhausted the scripted reply and made a test about a
        # reply that never lands pass for the opposite reason.
        self.waits += 1
        time.sleep(ms / 1000.0)


def _running_ctx(page, budget_ms = 12000):
    return ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = "r100K.A0.rep0", rung = "100K", rung_tokens = 100_000),
        window = None,
        args = {"thread_id": "t1", "base_url": "http://localhost:1"},
        budget_ms = budget_ms,
        dom = None,
        log = lambda _m: None,
    )


def test_it_waits_for_a_generating_reply_instead_of_reporting_no_menu():
    """THE DEFECT, and it cost this action on every CI run since the branch opened.

    Unsloth hides a message's action bar while the message is generating, and the film schedules
    `message_menu` about four seconds after a `send_turn` whose reply runs for roughly fourteen.
    So the harness asked for a menu on a message that was still being written, got no More button,
    and reported NOT RUN -- an accurate observation of a question nobody should have asked yet.
    Waiting is what a user does, and it is the only thing that makes the action measurable.
    """
    page = _RunningPage(runs_for = 3)
    assert A._wait_for_the_reply_to_land(_running_ctx(page)) is True
    assert page.waits >= 1, "it returned without ever waiting"


def test_a_reply_that_never_lands_is_reported_honestly_rather_than_waited_on_forever():
    """The wait is bounded by the slot's own budget: a reply that never finishes must not swallow
    the rest of the film."""
    page = _RunningPage(runs_for = 10_000)
    assert A._wait_for_the_reply_to_land(_running_ctx(page, budget_ms = 200)) is False


def test_an_idle_thread_is_not_waited_on_at_all():
    """No stream, no wait: the action must not add latency to the common case, and a wait that
    always fires would change what every other reading is taken against."""
    page = _RunningPage(runs_for = 0)
    assert A._wait_for_the_reply_to_land(_running_ctx(page)) is True
    assert page.waits == 0
