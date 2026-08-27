# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A reply that never finished is a FAILED cell, not a completed one carrying a note.

`_drain_stream` reports rather than raises -- "wait for the run to end, or say plainly that it did
not" -- and the value it reported was written to `row["stream"]` and read by nothing. No gate, no
report column, no `--assert-liveness` check, no exit code. So the state this tool exists to catch,
the app still generating three times past its own cadence AND 120 s beyond that with the whole
film already run, came back as `completed: true`, `ok` in the summary and exit 0, with the cell's
actions and its frame windows scored and paired into the A/B ratio against an arm that DID finish.

That is the crash-beats-limp rule inverted: a build that cannot finish reads as a build that had
nothing to say. Every other "it would not do what it claimed" in this harness already fails --
`stop_generation` that never stops sets `expect_ok = False`, an action that did not run is a
liveness problem -- and this was the one that did not.

Driven through the shipped `CellRunner.run` and `_run_inner`, with the seams that leave this
process stubbed where they cross it. `_drain_stream` itself is the REAL one, over a real loop on a
fake clock, so what it returns is observed rather than asserted from a table.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.report.build import score_payload  # noqa: E402
from studiobench.runtime import session as session_mod  # noqa: E402
from studiobench.runtime.session import CellRunner, Session  # noqa: E402
from studiobench.runtime.types import BenchContext, Cell, Paths, Recorder  # noqa: E402

CENSUS = {
    "messages": 4,
    "elements": 1200,
    "highlight_spans": 200,
    "assistant_chars": 1120,
}


class _Clock:
    """A monotonic clock the page's own waits advance, so the real drain loop runs to its real
    deadline without this test sleeping through it."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += ms / 1000.0


class _Page:
    def __init__(self, clock: _Clock, *, stops_running_after_ms: float | None) -> None:
        self.clock = clock
        self.stops_running_after_ms = stops_running_after_ms
        self._send = types.SimpleNamespace(click = lambda: None)

    # -- playwright surface -------------------------------------------------
    def goto(self, *a, **k) -> None:
        pass

    def wait_for_selector(self, *a, **k) -> None:
        pass

    def click(self, *a, **k) -> None:
        pass

    def fill(self, *a, **k) -> None:
        pass

    def wait_for_timeout(self, ms) -> None:
        self.clock.advance(float(ms))

    def query_selector(self, selector):
        return self._send if "Send message" in selector else None

    def evaluate(self, expr, *args):
        if "isRunning" in expr:
            if self.stops_running_after_ms is None:
                return True
            return self.clock.t * 1000.0 < self.stops_running_after_ms
        if "assistantChars" in expr:
            return CENSUS["assistant_chars"]
        return 0


class _Pacer:
    """A pacer that serves, in full, whatever it was asked to load.

    It synthesises one completed `StreamStats` per `load`, which is what a healthy cell produces:
    the planned-stream check in `_run_inner` reads those, so a stub that recorded nothing would
    fail every cell here for a reason this file is not about.
    """

    def __init__(self, expected_ms: float) -> None:
        self.expected_ms = expected_ms
        self.streams: list[dict] = []

    def reset(self) -> None:
        self.streams = []

    def load(
        self,
        reasoning,
        content,
        *,
        cadence = "field",
        tag = "",
        **k,
    ) -> None:
        self.streams.append(
            {
                "tag": tag,
                "chars_sent": len(reasoning) + len(content),
                "completed": True,
                "disconnected": False,
                "chunks": 150,
            }
        )

    def expected_duration_ms(self, reasoning, content, cadence) -> float:
        return self.expected_ms

    def last_stats(self) -> dict:
        return self.streams[-1] if self.streams else {}

    def all_stats(self) -> list[dict]:
        return list(self.streams)


#: (action name, timing key, value). The three actions `scoring.from_payload.ACTION_SOURCES`
#: reads, so a cell that completes carries enough weight to be scored rather than failing the
#: coverage floor for an unrelated reason.
SCENE_ACTIONS = (
    ("keystroke", "p95_ms", 40.0),
    ("message_menu", "open_ms", 90.0),
    ("scroll_after", "gesture_ms", 120.0),
)


class _SceneRunner:
    """The film, reduced to the rows a film writes: three timed actions and one frame window.

    It writes them through the recorder the shipped `SceneRunner` is handed, so the scoring path
    below is the real one reading real rows rather than a table this test asserts against itself.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, scene, t0) -> list:
        rec = self.kwargs["recorder"]
        cell_id = self.kwargs["cell"].cell_id
        out = []
        for name, timing_key, value in SCENE_ACTIONS:
            row = {
                "row_type": "action",
                "cell_id": cell_id,
                "action": name,
                "ran": True,
                "expect_ok": True,
                "expect": {},
                "timings": {timing_key: value},
                "slot_missed": False,
                "census": dict(CENSUS),
            }
            rec.emit(dict(row))
            out.append(row)
        rec.emit(
            {
                "row_type": "window",
                "cell_id": cell_id,
                "name": "stream:film",
                "kind": "stream",
                "t_open_ms": 0.0,
                "duration_ms": 1000.0,
                "instruments": {
                    "frames": {
                        "frames_attempted": True,
                        "max_frame_ms": 30.0,
                        "frame_gaps_ms": [16.0, 17.0, 16.0],
                    }
                },
            }
        )
        return out


def _seed(plan):
    """What `Seeder.seed` hands back. Zero messages, so the thread wait is the empty-page one.

    The two markers are `None` for the same reason `messages` is 0: the readiness gate takes the
    empty-page path here, and a marker it never asserts on must not be invented. They are PRESENT
    rather than absent because `SeededThread` declares them and `_wait_for_thread` reads
    `last_marker` unconditionally; a stub without them fails on the attribute instead of on the
    behaviour these tests are about.
    """

    return types.SimpleNamespace(
        thread_id = "t1",
        seconds = 0.5,
        messages = 0,
        first_marker = None,
        last_marker = None,
    )


@pytest.fixture
def cell_runner(monkeypatch, tmp_path):
    """A `CellRunner` wired to the real session, the real recorder and a fake browser."""

    clock = _Clock()
    state = {"clock": clock, "expected_ms": 1000.0, "stops_running_after_ms": 0.0}

    monkeypatch.setattr(
        session_mod,
        "time",
        types.SimpleNamespace(
            monotonic = clock.monotonic,
            sleep = lambda s: None,
            time = time.time,
            perf_counter = time.perf_counter,
        ),
    )
    monkeypatch.setattr(session_mod, "paint_floor_ms", lambda page: 8.0)
    monkeypatch.setattr(session_mod, "dom_signature", lambda page: dict(CENSUS))
    monkeypatch.setattr(
        session_mod,
        "measure_chars_per_token",
        lambda *a, **k: {"chars_per_token": 3.7, "source": "stubbed"},
    )
    monkeypatch.setattr(session_mod, "cdp_metrics", lambda cdp: {})
    monkeypatch.setattr(session_mod, "cdp_counters", lambda before, after: {})
    monkeypatch.setattr(session_mod, "SceneRunner", _SceneRunner)
    monkeypatch.setattr(session_mod, "dump_diagnostics", lambda *a, **k: None)

    paths = Paths.under(tmp_path / "out")
    recorder = Recorder(paths.payload_jsonl, "sess-1")
    ctx = BenchContext(
        page = None,
        cdp = None,
        base_url = "http://127.0.0.1:5399",
        session_id = "sess-1",
        tier = "quick",
        paths = paths,
        recorder = recorder,
        log = lambda msg: None,
    )
    session = Session(ctx = ctx)

    def build():
        ctx.page = _Page(clock, stops_running_after_ms = state["stops_running_after_ms"])
        return CellRunner(
            session = session,
            pacer = _Pacer(state["expected_ms"]),
            seeder = types.SimpleNamespace(seed = _seed, auth = None),
            corpus = None,
            base_url = "http://127.0.0.1:5399",
            model_id = "studiobench-pacer",
            tier = "quick",
            paths = paths,
            log = lambda msg: None,
        )

    state["build"] = build
    state["paths"] = paths
    state["recorder"] = recorder
    return state


def _cell():
    # NOT the 10K rung: that one additionally runs the seeded-vs-streamed equivalence check, which
    # is a different subject.
    return Cell(cell_id = "r1K.A0.rep0", rung = "1K", rung_tokens = 1_000, tier = "quick")


def _plan():
    unit = types.SimpleNamespace(reasoning = "r" * 100, content = "c" * 900, kind = "tail")
    return types.SimpleNamespace(
        rung = "1K",
        streamed_unit = unit,
        seeded_units = [],
        follow_up_units = [],
        seeded_chars = 0,
        streamed_chars = 1000,
        target_chars = 1000,
        target_tokens = 1_000,
    )


def _rows(state):
    state["recorder"].close()
    text = state["paths"].payload_jsonl.read_text(encoding = "utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


# ── what the real drain returns ──────────────────────────────────────────────────────────────


def test_the_drain_reports_rather_than_raises_when_the_reply_never_ends(cell_runner):
    """The premise, taken from the shipped `_drain_stream` rather than asserted about it."""

    cell_runner["stops_running_after_ms"] = None  # never stops
    runner = cell_runner["build"]()

    drained = runner._drain_stream(runner.session.ctx.page, 1000.0)

    assert drained["finished"] is False
    assert "three times past its own cadence" in drained["reason"]


# ── what the cell does with it ───────────────────────────────────────────────────────────────


def test_a_cell_whose_reply_never_finished_does_not_complete(cell_runner):
    cell_runner["stops_running_after_ms"] = None
    runner = cell_runner["build"]()

    row = runner.run(_cell(), _plan())

    assert row["completed"] is False
    assert row["failure"]["kind"] == "RuntimeError"
    assert "never finished" in row["failure"]["message"]
    # The evidence ships in the same row as the failure: how long was waited, and how much the
    # pacer actually delivered.
    assert row["stream"]["finished"] is False
    assert row["pacer"]["last"]["chunks"] == 150
    assert row["pacer"]["streams"][0]["tag"] == "r1K.A0.rep0"


def test_the_rung_scores_incomplete_and_the_run_cannot_exit_zero(cell_runner):
    """The consequence a reader sees: the ladder says INCOMPLETE and `completion_exit_code`
    counts a cell that did not complete."""

    from studiobench.__main__ import completion_exit_code

    cell_runner["stops_running_after_ms"] = None
    runner = cell_runner["build"]()

    row = runner.run(_cell(), _plan())
    ladder = score_payload(cell_runner["paths"].payload_jsonl, [1_000])

    assert ladder.rungs[0].complete is False
    assert "RuntimeError" in (ladder.rungs[0].incomplete_reason or "")
    assert completion_exit_code([row]) == 1


# ── the controls ─────────────────────────────────────────────────────────────────────────────


def test_a_cell_whose_reply_finished_still_completes(cell_runner):
    """The control that matters: the ordinary cell is untouched, and its readings are scored."""

    cell_runner["stops_running_after_ms"] = 0.0  # already idle when the drain opens
    runner = cell_runner["build"]()

    row = runner.run(_cell(), _plan())

    assert row["completed"] is True
    assert row["stream"]["finished"] is True
    assert row["assistant_chars_in_dom"] == CENSUS["assistant_chars"]
    assert score_payload(cell_runner["paths"].payload_jsonl, [1_000]).rungs[0].complete is True


def test_a_reply_that_ends_late_but_ends_still_completes(cell_runner):
    """The other control, and the reason the deadline is what it is: a slow finish is a finding
    the cell must KEEP, not a failure. Only a reply that never ends at all is one."""

    cell_runner["expected_ms"] = 1000.0
    cell_runner["stops_running_after_ms"] = 100_000.0  # 100 s, inside 3 x 1 s + 120 s
    runner = cell_runner["build"]()

    row = runner.run(_cell(), _plan())

    assert row["completed"] is True
    assert row["stream"]["finished"] is True
    assert row["stream"]["drain_ms"] > 90_000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
