# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cell that under-streamed must not report COMPLETE.

`send_turn` used to call `pacer.reset()` before loading the next turn, and `CellRunner` records
only `pacer.last_stats()`. So on any rung that streams more than once -- 10K upwards -- the opening
reply's `StreamStats` were discarded by the first follow-up and the first follow-up's by the
second. The only other liveness signal is the UI no longer running, and a later turn that finishes
satisfies it on behalf of an earlier one that did not. Measured against the real pacer: an opening
reply whose client went away after 4,624 of 10,000 characters was erased by a follow-up that
delivered its 1,500 in full, and the cell was marked complete and scored against a thread half the
size of the rung it is named for.

That is the defect class this whole benchmark has been burned by repeatedly: a measurement that
under-measures and still reports success. It is worse than no measurement, because people act on
it. So the stats for every planned turn are kept and checked, and a cell that did not stream what
it planned fails by name.

Three levels. The first drives the REAL pacer over a REAL socket through the REAL `send_turn`. The
second drives the shipped `CellRunner` over dictated streams and asserts the consequence a reader
sees. The third runs the whole cell against a real pacer over real wire bytes, so the control --
an ordinary multi-turn cell still completes -- is not taken on trust either.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.pacer import Pacer, check_planned_streams  # noqa: E402
from studiobench.runtime import session as session_mod  # noqa: E402
from studiobench.runtime.types import ActionContext, Cell, Window  # noqa: E402
from studiobench.scene.actions import send_turn  # noqa: E402

CELL_ID = "r10K.A0.rep0"
OPENING = ("R" * 2_000, "C" * 8_000)
FOLLOW = ("r" * 300, "c" * 1_200)


# ── level 1: the pacer and the action, over a real socket ────────────────────────────────────


def _consume(pacer: Pacer, *, stop_after_bytes: int | None = None) -> None:
    """Read one stream off the wire. `stop_after_bytes` closes the socket mid-reply, which is what
    an interrupted opening reply looks like from the pacer's side."""
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.settimeout(60)
    conn.connect((pacer.host, pacer.port))
    body = json.dumps(
        {
            "model": "studiobench-pacer",
            "stream": True,
            "messages": [{"role": "user", "content": "go"}],
        }
    ).encode()
    conn.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body
    )
    seen = b""
    while b"[DONE]" not in seen:
        if stop_after_bytes is not None and len(seen) >= stop_after_bytes:
            break
        try:
            got = conn.recv(65536)
        except socket.timeout:
            break
        if not got:
            break
        seen += got
    conn.close()


class _Page:
    """Just enough page for `send_turn`: a composer, a keyboard, and an `isRunning` that is true
    for as long as the follow-up's client is draining the stream."""

    def __init__(self, pacer: Pacer) -> None:
        self.pacer = pacer
        self.running = False
        self.messages = 4
        self._thread: threading.Thread | None = None

    def query_selector(self, selector):
        return object() if "Message input" in selector else None

    def fill(self, *a, **k) -> None:
        pass

    def wait_for_timeout(self, ms) -> None:
        time.sleep(ms / 1000.0)

    @property
    def keyboard(self):
        return types.SimpleNamespace(press = self._press)

    def _press(self, key) -> None:
        self.running = True
        self.messages += 2

        def run() -> None:
            _consume(self.pacer)
            self.running = False

        self._thread = threading.Thread(target = run, daemon = True)
        self._thread.start()

    def drain(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout = 60)

    def evaluate(self, expr, *args):
        if "isRunning" in expr:
            return self.running
        if "messageCount" in expr:
            return self.messages
        return 0


def _ctx(page, pacer, queue, cursor) -> ActionContext:
    return ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = CELL_ID, rung = "10K", rung_tokens = 10_000, tier = "quick"),
        window = Window(name = "action:send_turn", kind = "action", cell = None, t_open_ms = 0.0),
        args = {
            "cell_id": CELL_ID,
            "cadence": "fast",
            "_pacer": pacer,
            "_stream_queue": queue,
            "_stream_cursor": cursor,
        },
        budget_ms = 20_000,
        dom = None,
        log = lambda msg: None,
    )


def test_send_turn_keeps_the_stats_of_every_turn_before_it():
    """THE BUG, at the level it lives at. An opening reply that under-delivered, then two
    follow-ups that did not. Before the fix the pacer held one stream at the end of this."""

    pacer = Pacer().start()
    try:
        pacer.reset()
        pacer.load(OPENING[0], OPENING[1], cadence = "fast", tag = CELL_ID)
        _consume(pacer, stop_after_bytes = 20_000)
        time.sleep(0.5)

        opening = pacer.last_stats()
        planned_chars = len(OPENING[0]) + len(OPENING[1])
        assert opening["chars_sent"] < planned_chars, "the opening reply did not under-deliver"
        assert opening["completed"] is False

        queue = [
            {"reasoning": FOLLOW[0], "content": FOLLOW[1], "kind": "prose"},
            {"reasoning": FOLLOW[0], "content": FOLLOW[1], "kind": "code"},
        ]
        cursor = {"i": 0}
        page = _Page(pacer)
        tags = []
        for _ in (1, 2):
            result = send_turn(_ctx(page, pacer, queue, cursor))
            page.drain()
            time.sleep(0.4)
            assert result.ran is True
            tags.append(result.expect["pacer_tag"])

        streams = pacer.all_stats()
        # The opening reply is STILL THERE, and still says it did not finish.
        assert [s["tag"] for s in streams] == [CELL_ID] + tags
        assert streams[0]["completed"] is False
        assert streams[0]["chars_sent"] == opening["chars_sent"]
        assert all(s["completed"] for s in streams[1:])

        check = check_planned_streams(
            streams,
            [{"tag": CELL_ID, "turn": "opening", "chars": planned_chars}]
            + [
                {"tag": t, "turn": f"follow_up{i}", "chars": len(FOLLOW[0]) + len(FOLLOW[1])}
                for i, t in enumerate(tags, start = 1)
            ],
        )
        assert check["ok"] is False
        assert CELL_ID in check["reason"] and "did not complete" in check["reason"]
    finally:
        pacer.stop()


# ── the check on its own ─────────────────────────────────────────────────────────────────────


def test_the_check_passes_when_every_planned_turn_streamed_in_full():
    streams = [
        {"tag": "c1", "chars_sent": 100, "completed": True, "disconnected": False},
        {"tag": "c1#turn1", "chars_sent": 50, "completed": True, "disconnected": False},
    ]
    got = check_planned_streams(
        streams,
        [
            {"tag": "c1", "turn": "opening", "chars": 100},
            {"tag": "c1#turn1", "turn": "follow_up1", "chars": 50},
        ],
    )
    assert got["ok"] is True
    assert got["reason"] is None
    assert got["extra"] == []


def test_a_short_turn_fails_the_check_with_both_counts_named():
    got = check_planned_streams(
        [{"tag": "c1", "chars_sent": 40, "completed": True, "disconnected": False}],
        [{"tag": "c1", "turn": "opening", "chars": 100}],
    )
    assert got["ok"] is False
    assert "delivered 40 of the 100 characters planned" in got["reason"]


def test_a_turn_that_never_reached_the_pacer_fails_the_check():
    got = check_planned_streams([], [{"tag": "c1", "turn": "opening", "chars": 100}])
    assert got["ok"] is False
    assert "never reached the pacer" in got["reason"]
    assert got["turns"][0]["found"] is False


def test_the_stop_actions_throwaway_turn_is_extra_and_not_a_failure():
    """`stop_generation` sends its own turn against whatever script is loaded and cancels it, so a
    second stream carries the tag of the turn before it. Matching takes the FIRST stream per tag,
    and the aborted throwaway is reported as `extra` rather than failing the cell."""

    streams = [
        {"tag": "c1", "chars_sent": 100, "completed": True, "disconnected": False},
        {"tag": "c1", "chars_sent": 12, "completed": False, "disconnected": True},
    ]
    got = check_planned_streams(streams, [{"tag": "c1", "turn": "opening", "chars": 100}])
    assert got["ok"] is True
    assert len(got["extra"]) == 1 and got["extra"][0]["disconnected"] is True


def test_a_cell_with_nothing_planned_is_not_checked():
    got = check_planned_streams([], [])
    assert got["checked"] is False and got["ok"] is True


# ── level 2: what the cell does with it ──────────────────────────────────────────────────────


CENSUS = {"messages": 6, "elements": 1200, "highlight_spans": 200, "assistant_chars": 1120}


class _CellPage:
    def goto(self, *a, **k) -> None:
        pass

    def wait_for_selector(self, *a, **k) -> None:
        pass

    def click(self, *a, **k) -> None:
        pass

    def fill(self, *a, **k) -> None:
        pass

    def wait_for_timeout(self, ms) -> None:
        pass

    def query_selector(self, selector):
        return types.SimpleNamespace(click = lambda: None) if "Send message" in selector else None

    def evaluate(self, expr, *args):
        if "isRunning" in expr:
            return False
        if "assistantChars" in expr:
            return CENSUS["assistant_chars"]
        return 0


class _RecordedPacer:
    """A pacer whose streams are dictated by the test, so the cell path can be driven over both
    outcomes without a socket."""

    def __init__(self, streams: list[dict]) -> None:
        self.streams = streams

    def reset(self) -> None:
        pass

    def load(self, *a, **k) -> None:
        pass

    def expected_duration_ms(self, reasoning, content, cadence) -> float:
        return 1000.0

    def last_stats(self) -> dict:
        return self.streams[-1] if self.streams else {}

    def all_stats(self) -> list[dict]:
        return list(self.streams)


class _SceneRunner:
    """One `send_turn` that ran, so the cell has a follow-up to demand of the pacer."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, scene, t0) -> list:
        row = {
            "row_type": "action",
            "cell_id": self.kwargs["cell"].cell_id,
            "action": "send_turn",
            "ran": True,
            "expect_ok": True,
            "expect": {
                "turn_index": 1,
                "streamed_chars": 1_500,
                "pacer_tag": f"{self.kwargs['cell'].cell_id}#turn1",
            },
            "timings": {"to_first_token_ms": 40.0},
            "slot_missed": False,
            "census": dict(CENSUS),
        }
        self.kwargs["recorder"].emit(dict(row))
        return [row]


def _plan():
    unit = types.SimpleNamespace(reasoning = "r" * 2_000, content = "c" * 8_000, kind = "tail")
    follow = types.SimpleNamespace(reasoning = "r" * 300, content = "c" * 1_200, kind = "prose")
    return types.SimpleNamespace(
        rung = "10K",
        streamed_unit = unit,
        seeded_units = [],
        follow_up_units = [follow],
        seeded_chars = 0,
        streamed_chars = 10_000,
        target_chars = 11_500,
        target_tokens = 10_000,
    )


@pytest.fixture
def cell_runner(monkeypatch, tmp_path):
    from studiobench.runtime.session import CellRunner, Session
    from studiobench.runtime.types import BenchContext, Paths, Recorder

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
    # NOT the 10K equivalence path, which reseeds a mirror thread and is a different subject.
    monkeypatch.setattr(session_mod, "EQUIVALENCE_RUNG", "1K")

    paths = Paths.under(tmp_path / "out")
    recorder = Recorder(paths.payload_jsonl, "sess-1")
    ctx = BenchContext(
        page = _CellPage(),
        cdp = None,
        base_url = "http://127.0.0.1:5399",
        session_id = "sess-1",
        tier = "quick",
        paths = paths,
        recorder = recorder,
        log = lambda msg: None,
    )
    session = Session(ctx = ctx)

    def build(streams):
        return CellRunner(
            session = session,
            pacer = _RecordedPacer(streams),
            seeder = types.SimpleNamespace(
                seed = lambda plan: types.SimpleNamespace(
                    thread_id = "t1",
                    seconds = 0.5,
                    messages = 0,
                    # Both markers `SeededThread` declares, present and None: the readiness gate
                    # reads `last_marker` unconditionally, so a stub that omits it fails on the
                    # attribute rather than on the stream accounting these tests are about.
                    first_marker = None,
                    last_marker = None,
                ),
                auth = None,
            ),
            corpus = None,
            base_url = "http://127.0.0.1:5399",
            model_id = "studiobench-pacer",
            tier = "quick",
            paths = paths,
            log = lambda msg: None,
        )

    return build


def _cell():
    return Cell(cell_id = CELL_ID, rung = "10K", rung_tokens = 10_000, tier = "quick")


def test_a_cell_whose_opening_reply_under_delivered_does_not_complete(cell_runner):
    """THE CONSEQUENCE. The follow-up finished, so the UI is idle and the drain check passes; the
    opening reply delivered 4,624 of 10,000 characters. Before the fix this row read
    `completed: true`."""

    runner = cell_runner(
        [
            {"tag": CELL_ID, "chars_sent": 4_624, "completed": False, "disconnected": True},
            {
                "tag": f"{CELL_ID}#turn1",
                "chars_sent": 1_500,
                "completed": True,
                "disconnected": False,
            },
        ]
    )

    row = runner.run(_cell(), _plan())

    assert row["stream"]["finished"] is True
    assert row["completed"] is False
    assert "did not stream what it planned" in row["failure"]["message"]
    # The named reason and the per-turn evidence reach the payload with the failure.
    check = row["pacer"]["check"]
    assert check["ok"] is False
    assert check["turns"][0]["chars_sent"] == 4_624
    assert check["turns"][0]["planned_chars"] == 10_000
    assert check["turns"][1]["ok"] is True


def test_a_cell_that_streamed_every_planned_turn_still_completes(cell_runner):
    """The control. An unremarkable multi-turn cell is untouched, and every turn is on the row."""

    runner = cell_runner(
        [
            {"tag": CELL_ID, "chars_sent": 10_000, "completed": True, "disconnected": False},
            {
                "tag": f"{CELL_ID}#turn1",
                "chars_sent": 1_500,
                "completed": True,
                "disconnected": False,
            },
        ]
    )

    row = runner.run(_cell(), _plan())

    assert row["completed"] is True
    assert row["pacer"]["check"]["ok"] is True
    assert [s["tag"] for s in row["pacer"]["streams"]] == [CELL_ID, f"{CELL_ID}#turn1"]


def _scene_with(rows: list[dict]):
    """A scene runner that emits exactly the action rows given."""

    class _Fixed:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self, scene, t0) -> list:
            out = []
            for spec in rows:
                row = {
                    "row_type": "action",
                    "cell_id": self.kwargs["cell"].cell_id,
                    "slot_missed": False,
                    "timings": {},
                    "expect": {},
                    "census": dict(CENSUS),
                    **spec,
                }
                self.kwargs["recorder"].emit(dict(row))
                out.append(row)
            return out

    return _Fixed


#: A `send_turn` that RAN, loaded the pacer and pressed Enter, and whose own assertion failed.
SEND_TURN_THAT_FAILED = {
    "action": "send_turn",
    "ran": True,
    "expect_ok": False,
    "expect": {"turn_index": 1, "streamed_chars": 1_500, "pacer_tag": f"{CELL_ID}#turn1"},
    "reason": "the send did not start a new streaming reply",
}


def test_a_follow_up_that_was_sent_but_never_streamed_fails_the_cell(cell_runner, monkeypatch):
    """`ran = True, expect_ok = False` is an ATTEMPTED turn that did not stream, not an unattempted
    one. Skipping it let the cell pass the stream check with `planned_turns: 1`, complete, and
    score against a thread one turn short of its rung."""

    monkeypatch.setattr(session_mod, "SceneRunner", _scene_with([SEND_TURN_THAT_FAILED]))
    runner = cell_runner(
        [{"tag": CELL_ID, "chars_sent": 10_000, "completed": True, "disconnected": False}]
    )

    row = runner.run(_cell(), _plan())

    assert row["completed"] is False
    assert row["pacer"]["check"]["planned_turns"] == 2
    assert f"{CELL_ID}#turn1" in row["failure"]["message"]
    assert "never reached the pacer" in row["failure"]["message"]


def test_a_follow_up_that_streamed_but_never_joined_the_thread_fails_the_cell(
    cell_runner, monkeypatch
):
    """The other half. The pacer served every byte, so the stream check alone passes; the thread
    did not grow, so every later action, the peak census and the equivalence mirror still read a
    thread one turn short."""

    monkeypatch.setattr(session_mod, "SceneRunner", _scene_with([SEND_TURN_THAT_FAILED]))
    runner = cell_runner(
        [
            {"tag": CELL_ID, "chars_sent": 10_000, "completed": True, "disconnected": False},
            {
                "tag": f"{CELL_ID}#turn1",
                "chars_sent": 1_500,
                "completed": True,
                "disconnected": False,
            },
        ]
    )

    row = runner.run(_cell(), _plan())

    assert row["pacer"]["check"]["ok"] is True, "the bytes did go out; that is not the complaint"
    assert row["completed"] is False
    assert "follow-up turn 1 was sent but" in row["failure"]["message"]
    assert "did not start a new streaming reply" in row["failure"]["message"]


def test_an_ordinary_action_whose_assertion_failed_does_not_fail_the_cell(cell_runner, monkeypatch):
    """THE SCOPE OF THE RULE, deliberately. `select_text` selecting nothing voids its own timing --
    `scoring.from_payload._action_measure` already returns `Measure.failed` for it -- but it does
    not change the workload the rest of the cell measured, and failing the cell would throw away a
    whole cell's frame readings for a gesture that missed."""

    monkeypatch.setattr(
        session_mod,
        "SceneRunner",
        _scene_with(
            [
                {
                    "action": "select_text",
                    "ran": True,
                    "expect_ok": False,
                    "reason": "the selection did not cover the message",
                },
                {
                    "action": "send_turn",
                    "ran": True,
                    "expect_ok": True,
                    "expect": {
                        "turn_index": 1,
                        "streamed_chars": 1_500,
                        "pacer_tag": f"{CELL_ID}#turn1",
                    },
                },
            ]
        ),
    )
    runner = cell_runner(
        [
            {"tag": CELL_ID, "chars_sent": 10_000, "completed": True, "disconnected": False},
            {
                "tag": f"{CELL_ID}#turn1",
                "chars_sent": 1_500,
                "completed": True,
                "disconnected": False,
            },
        ]
    )

    row = runner.run(_cell(), _plan())

    assert row["completed"] is True
    assert row["expect_failures"] == 1


def test_a_send_turn_that_did_not_run_is_not_demanded_of_the_pacer(cell_runner, monkeypatch):
    """The other control, and the reason the planned list is built from the ACTION rows. At the
    small rungs the queue is empty and `send_turn` reports NOT RUN; nothing was streamed for it and
    nothing must be required of it."""

    class _NoTurn(_SceneRunner):
        def run(self, scene, t0) -> list:
            row = {
                "row_type": "action",
                "cell_id": self.kwargs["cell"].cell_id,
                "action": "send_turn",
                "ran": False,
                "expect_ok": None,
                "slot_missed": False,
                "expect": {},
                "timings": {},
                "reason": "the stream queue is exhausted (0 turns planned)",
                "census": dict(CENSUS),
            }
            self.kwargs["recorder"].emit(dict(row))
            return [row]

    monkeypatch.setattr(session_mod, "SceneRunner", _NoTurn)
    runner = cell_runner(
        [{"tag": CELL_ID, "chars_sent": 10_000, "completed": True, "disconnected": False}]
    )

    row = runner.run(_cell(), _plan())

    assert row["completed"] is True
    assert row["pacer"]["check"]["planned_turns"] == 1


# ── level 3: the whole cell, against a real pacer over real wire bytes ───────────────────────


class _WirePage:
    """A page whose sends really do fetch a stream off the pacer, so `isRunning` is true for
    exactly as long as bytes are arriving. Everything the browser does with them is out of scope
    here; what is in scope is that a healthy multi-turn cell passes the new check when the streams
    are real rather than dictated by the test."""

    def __init__(self, pacer: Pacer) -> None:
        self.pacer = pacer
        self.running = False
        self.messages = 4
        self._thread: threading.Thread | None = None

    def goto(self, *a, **k) -> None:
        pass

    def wait_for_selector(self, *a, **k) -> None:
        pass

    def click(self, *a, **k) -> None:
        pass

    def fill(self, *a, **k) -> None:
        pass

    def wait_for_timeout(self, ms) -> None:
        time.sleep(min(ms, 200) / 1000.0)

    def query_selector(self, selector):
        if "Send message" in selector:
            return types.SimpleNamespace(click = lambda: self.send())
        return object() if "Message input" in selector else None

    @property
    def keyboard(self):
        return types.SimpleNamespace(press = lambda key: self.send())

    def send(self) -> None:
        self.running = True
        self.messages += 2

        def run() -> None:
            _consume(self.pacer)
            self.running = False

        self._thread = threading.Thread(target = run, daemon = True)
        self._thread.start()

    def evaluate(self, expr, *args):
        if "isRunning" in expr:
            return self.running
        if "messageCount" in expr:
            return self.messages
        # THE THREAD'S LENGTH AS WELL AS THE MOUNTED COUNT. `send_turn` proves a send worked by
        # `threadTotal()` growing, not `messageCount()`, so that a windowed arm whose window slides
        # is not read as a send that did nothing. This page models a fully mounted arm, where the
        # two are the same number; without the second name the shipped action sees 0 both sides and
        # every follow-up reports that it never started a reply.
        if "threadTotal" in expr:
            return self.messages
        if "assistantChars" in expr:
            return CENSUS["assistant_chars"]
        return 0


class _RealSendTurnScene:
    """The film, reduced to the two `send_turn` slots that matter here, running the SHIPPED action
    against the shipped base_args the session builds."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, scene, t0) -> list:
        page = self.kwargs["page"]
        rows = []
        for _ in range(2):
            while page.running:
                time.sleep(0.05)
            ctx = ActionContext(
                page = page,
                cdp = None,
                cell = self.kwargs["cell"],
                window = Window(name = "action:send_turn", kind = "action", cell = None, t_open_ms = 0.0),
                args = dict(self.kwargs["base_args"]),
                budget_ms = 20_000,
                dom = None,
                log = lambda msg: None,
            )
            result = send_turn(ctx)
            row = result.row("send_turn", "action:send_turn", self.kwargs["cell"].cell_id)
            row["census"] = dict(CENSUS)
            self.kwargs["recorder"].emit(dict(row))
            rows.append(row)
        while page.running:
            time.sleep(0.05)
        return rows


def test_a_healthy_multi_turn_cell_passes_the_check_over_real_wire_bytes(monkeypatch, tmp_path):
    """The control the fix most needs: a cell that streamed everything it planned, with the pacer,
    the action and the check all real, must still complete. The dictated-stream tests above prove
    the failure path; this proves the ordinary path was not broken to get it."""

    from studiobench.runtime.session import CellRunner, Session
    from studiobench.runtime.types import BenchContext, Paths, Recorder

    monkeypatch.setattr(session_mod, "paint_floor_ms", lambda page: 8.0)
    monkeypatch.setattr(session_mod, "dom_signature", lambda page: dict(CENSUS))
    monkeypatch.setattr(
        session_mod,
        "measure_chars_per_token",
        lambda *a, **k: {"chars_per_token": 3.7, "source": "stubbed"},
    )
    monkeypatch.setattr(session_mod, "cdp_metrics", lambda cdp: {})
    monkeypatch.setattr(session_mod, "cdp_counters", lambda before, after: {})
    monkeypatch.setattr(session_mod, "SceneRunner", _RealSendTurnScene)
    monkeypatch.setattr(session_mod, "dump_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(session_mod, "EQUIVALENCE_RUNG", "1K")

    pacer = Pacer().start()
    try:
        paths = Paths.under(tmp_path / "out")
        recorder = Recorder(paths.payload_jsonl, "sess-1")
        page = _WirePage(pacer)
        ctx = BenchContext(
            page = page,
            cdp = None,
            base_url = "http://127.0.0.1:5399",
            session_id = "sess-1",
            tier = "quick",
            paths = paths,
            recorder = recorder,
            log = lambda msg: None,
        )
        runner = CellRunner(
            session = Session(ctx = ctx),
            pacer = pacer,
            seeder = types.SimpleNamespace(
                seed = lambda plan: types.SimpleNamespace(
                    thread_id = "t1",
                    seconds = 0.5,
                    messages = 0,
                    # Both markers `SeededThread` declares, present and None: the readiness gate
                    # reads `last_marker` unconditionally, so a stub that omits it fails on the
                    # attribute rather than on the stream accounting these tests are about.
                    first_marker = None,
                    last_marker = None,
                ),
                auth = None,
            ),
            corpus = None,
            base_url = "http://127.0.0.1:5399",
            model_id = "studiobench-pacer",
            tier = "quick",
            paths = paths,
            log = lambda msg: None,
            cadence = "fast",
        )
        unit = types.SimpleNamespace(reasoning = "R" * 400, content = "C" * 1_200, kind = "tail")
        follow = types.SimpleNamespace(reasoning = "r" * 100, content = "c" * 300, kind = "prose")
        plan = types.SimpleNamespace(
            rung = "10K",
            streamed_unit = unit,
            seeded_units = [],
            follow_up_units = [follow, follow],
            seeded_chars = 0,
            streamed_chars = 1_600,
            target_chars = 2_400,
            target_tokens = 10_000,
        )

        row = runner.run(_cell(), plan)

        assert row["completed"] is True, row.get("failure")
        check = row["pacer"]["check"]
        assert check["ok"] is True and check["planned_turns"] == 3
        # Every planned turn delivered EXACTLY what it was loaded with, over the wire.
        assert [t["chars_sent"] for t in check["turns"]] == [1_600, 400, 400]
        assert [t["tag"] for t in check["turns"]] == [
            CELL_ID,
            f"{CELL_ID}#turn1",
            f"{CELL_ID}#turn2",
        ]
    finally:
        pacer.stop()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
