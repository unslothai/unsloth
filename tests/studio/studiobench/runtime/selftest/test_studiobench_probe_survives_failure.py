# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cell that dies after `--click-probe` ran still reports what the probe measured.

THE PROBE HAS NOWHERE ELSE TO LIVE. `composer_click_ms` is taken inside a `setup:composer_click`
window, and that window's row is emitted by the session whatever happens next, so the click
timing survives a cell that fails later. The attribution block has no window of its own: it runs
before the window opens, it is not noted on one, and the only copy of it is the field the cell row
is built from.

AND THE CELL CAN STILL DIE AFTER IT. `_press_send` sets the 90 s bound on the composer click
alone. Everything after it -- the `page.fill`, the send-button lookup and its click -- still runs
under the 8 s default action timeout installed in `runtime/browser.py`, which is the timeout a
large rung was already observed to blow through: that is why the click needed a bound of its own.
So the exact run that pays for the probe, at the rung the probe exists for, is the one that loses
it. Assigned on the way out of `_run_inner`, the whole block went missing from the failure cell.

`CellRunner.run` is driven, with the browser stubbed at the boundary it crosses. The row that gets
asserted is read back out of the payload the REAL recorder wrote.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.fixture.corpus import RungPlan, Unit  # noqa: E402
from studiobench.runtime.session import CellRunner, Session  # noqa: E402
from studiobench.runtime.types import BenchContext, Cell, Paths, Recorder  # noqa: E402

CENSUS = {"messages": 4, "elements": 31_637, "highlight_spans": 1_485}


class _Page:
    """Everything `_run_inner` touches before the send, and a `fill` that fails like the real one.

    Playwright raises `TimeoutError` out of `page.fill` when the composer cannot be filled inside
    the default action timeout. Raised from `fill` rather than from the send click because it is
    the first unbounded step after the probe.
    """

    def __init__(self, fail_on: str = "fill") -> None:
        self.fail_on = fail_on

    def _maybe_fail(self, where: str) -> None:
        if where == self.fail_on:
            raise TimeoutError(f"page.{where}: Timeout 8000ms exceeded")

    def goto(
        self,
        url,
        wait_until = None,
        timeout = None,
    ):
        self._maybe_fail("goto")

    def wait_for_selector(
        self,
        selector,
        timeout = None,
    ):
        return None

    def evaluate(
        self,
        expr,
        arg = None,
    ):
        if "counts()" in expr:
            return dict(CENSUS)
        return 0

    def click(
        self,
        selector,
        timeout = None,
    ):
        return None

    def fill(self, selector, value):
        self._maybe_fail("fill")

    def wait_for_timeout(self, ms):
        return None

    def query_selector(self, selector):
        return types.SimpleNamespace(
            click = lambda: None,
            bounding_box = lambda: {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
        )

    def dispatch_event(self, selector, event):
        return None

    def eval_on_selector(self, selector, expr):
        return None

    @property
    def mouse(self):
        return types.SimpleNamespace(click = lambda *a: None, move = lambda *a: None)


class _Pacer:
    def reset(self) -> None:
        pass

    def load(self, reasoning, content, **kwargs) -> None:
        pass

    def expected_duration_ms(self, reasoning, content, cadence) -> float:
        return 1_000.0


class _Seeder:
    #: `None`, so `measure_chars_per_token` cannot reach for the network from a unit test.
    auth = None

    def seed(self, plan):
        # No messages, so the mount wait is the selector and not a count that never arrives, and
        # both markers `SeededThread` declares are present and `None` for the same reason:
        # `_wait_for_thread` reads `last_marker` unconditionally, and a stub that omits it fails on
        # the attribute rather than on the failure these tests are about.
        return types.SimpleNamespace(
            thread_id = "t-1",
            seconds = 0.0,
            messages = 0,
            first_marker = None,
            last_marker = None,
        )


def _unit() -> Unit:
    return Unit(
        index = 0,
        kind = "reasoning",
        reasoning = "thinking about it ",
        content = "the answer ",
        chars = 29,
        sha256 = "0" * 64,
    )


def _plan() -> RungPlan:
    return RungPlan(
        rung = "500K",
        target_tokens = 500_000,
        target_chars = 2_000_000,
        streamed_unit = _unit(),
    )


def _runner(
    tmp_path,
    *,
    click_probe: bool,
    fail_on: str = "fill",
):
    paths = Paths.under(tmp_path / "out")
    recorder = Recorder(paths.payload_jsonl, "sess-1")
    ctx = BenchContext(
        page = _Page(fail_on = fail_on),
        base_url = "http://127.0.0.1:65535",
        session_id = "sess-1",
        paths = paths,
        recorder = recorder,
        log = lambda *_a: None,
    )
    runner = CellRunner(
        session = Session(ctx = ctx),
        pacer = _Pacer(),
        seeder = _Seeder(),
        corpus = None,
        base_url = ctx.base_url,
        model_id = "m",
        tier = "quick",
        paths = paths,
        log = lambda *_a: None,
        click_probe = click_probe,
    )
    return runner, paths, recorder


def _cell(cell_id: str = "r500K.A0.rep0") -> Cell:
    return Cell(cell_id = cell_id, rung = "500K", rung_tokens = 500_000, session_id = "sess-1")


def _cell_rows(paths) -> list[dict]:
    rows = [
        json.loads(line)
        for line in paths.payload_jsonl.read_text(encoding = "utf-8").splitlines()
        if line
    ]
    return [r for r in rows if r.get("row_type") == "cell"]


def test_a_cell_that_dies_after_the_probe_still_reports_the_attribution(tmp_path):
    """REGRESSION. The probe ran, the send did not, and the numbers must not go with it."""

    runner, paths, recorder = _runner(tmp_path, click_probe = True)
    row = runner.run(_cell(), _plan())
    recorder.close()

    assert row["completed"] is False
    assert row["failure"]["kind"] == "TimeoutError"

    # Out of the payload, not out of the returned dict: the file is what a failed run leaves
    # behind, and it is where the diagnostic has to be readable from.
    cells = _cell_rows(paths)
    assert len(cells) == 1
    attribution = cells[0]["click_attribution"]
    assert attribution["click_attribution_attempted"] is True
    # The two readings the flag exists to produce: what the driver pays and what a user pays.
    assert "click_ms" in attribution and "mouse_ms" in attribution


def test_the_attribution_of_one_cell_never_lands_on_the_next(tmp_path):
    """CONTROL. A cell that dies BEFORE its own probe has no attribution to report.

    The reading is filed from the cell's `finally`, so the field has to be cleared per cell rather
    than beside the click that sets it. A stale block re-filed under the next cell id would be
    worse than the loss this test's neighbour is about: a measurement of one rung reported as
    another's.
    """

    runner, paths, recorder = _runner(tmp_path, click_probe = True)
    runner.run(_cell(), _plan())

    # The same runner, and this time the page is gone before the probe can run.
    runner.session.ctx.page = _Page(fail_on = "goto")
    runner.run(_cell("r500K.A0.rep1"), _plan())
    recorder.close()

    first, second = _cell_rows(paths)
    assert "click_attribution" in first
    assert second["completed"] is False
    assert "click_attribution" not in second


def test_a_cell_that_never_asked_for_the_probe_reports_none(tmp_path):
    """CONTROL. Without `--click-probe` there is nothing to preserve, and the failed cell says so
    by carrying no attribution at all rather than an empty one."""

    runner, paths, recorder = _runner(tmp_path, click_probe = False)
    runner.run(_cell(), _plan())
    recorder.close()

    cells = _cell_rows(paths)
    assert cells[0]["completed"] is False
    assert "click_attribution" not in cells[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
