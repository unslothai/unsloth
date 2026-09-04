# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Replies joining and leaving a batch the MLX worker keeps open."""

import multiprocessing as _mp
import queue as _queue
import threading
from types import SimpleNamespace

import pytest

from core.inference import worker
from core.inference.worker import RowRefused
from core.inference.worker import StopLedger


class _RespQueue:
    def __init__(self):
        self.sent = []

    def put(self, item, *a, **k):
        self.sent.append(item)


class _Session:
    """A batch that hands back whatever the test scripted for each reply."""

    def __init__(self, *, width):
        # Not named "width": no production session has that attribute.
        self.opened_at = width
        self.rows, self.ending, self.refusals = [], [], []
        self.settled, self.script, self.stats_at_retire = {}, {}, {}
        self.will_not_take, self.closed, self.admitted = set(), False, []

    @property
    def rows_in_flight(self):
        return len(self.rows)

    @property
    def handles(self):
        return list(self.rows)

    def take_stats(self, handle):
        return self.settled.pop(handle, None)

    def admit(self, request, handle):
        if handle[0] in self.will_not_take or handle in self.will_not_take:
            self.refusals.append(handle)
            raise RowRefused("it does not prepare like the replies in the batch")
        self.admitted.append((handle, request))
        self.rows.append(handle)
        return ""

    def _retire(self, handle):
        self.rows.remove(handle)
        self.settled[handle] = self.stats_at_retire.get(handle, {})
    def step(self):
        for handle in list(self.rows):
            events = self.script.get(handle) or []
            if not events:
                continue
            event = events.pop(0)
            if event is None:
                self._retire(handle)
            yield handle, event

    def withdraw(self, handles):
        for handle in [h for h in handles if h in self.rows]:
            self._retire(handle)
            yield handle, None

    def close(self):
        self.closed = True


class _Backend:
    def __init__(self, reason = None, script = None):
        self.reason = reason
        self.script = script or {}
        self.sessions = []
        self.on_open = None
        self.on_reason = None
        self.apart = []

    def generate_chat_response(self, **kwargs):
        self.apart.append(kwargs)
        yield "apart"

    def resident_unavailable_reason(self, request):
        if self.on_reason is not None:
            self.on_reason()
        return self.reason

    def open_resident_batch(self, *, width, adapter_state = None):
        session = _Session(width = width)
        session.script.update(self.script)
        if self.on_open is not None:
            self.on_open(session)
        self.sessions.append(session)
        return session


def _cmd(request_id = "r1", **extra):
    return {"type": "generate", "request_id": request_id, "messages": [], **extra}


def _batch(backend = None):
    resp = _RespQueue()
    return worker._ResidentBatch(backend or _Backend(), resp), resp


def test_replies_a_command_asked_for_together_are_addressed_and_counted_by_row():
    batch, resp = _batch()
    batch.admit(_cmd(rows = [{"seed": 1}, {"seed": 2}]), None)
    for row, stats in ((0, {"n": 1}), (1, {"n": 2})):
        batch.session.script[("r1", row)] = ["ab"[row], None]
        batch.session.stats_at_retire[("r1", row)] = stats
    for _ in range(2):
        batch.step()

    rows = [(m["type"], m.get("row")) for m in resp.sent]
    assert ("token", 0) in rows and ("token", 1) in rows
    assert rows.count(("row_done", 0)) == 1 and rows.count(("row_done", 1)) == 1
    done = [m for m in resp.sent if m["type"] == "row_done"]
    assert sorted(m["stats"]["n"] for m in done) == [1, 2]
    assert [m["type"] for m in resp.sent][-1] == "gen_done"
    assert resp.sent[-1]["stats"] is None


IDLE = object()


def _run_loop(monkeypatch, backend, cmds):
    """Drive the real worker loop over a scripted queue."""
    import queue as _stdqueue

    from utils.hardware import hardware as _hw
    from loggers.config import LogConfig
    import core.inference.mlx_inference as mlx_mod

    monkeypatch.setenv("ENVIRONMENT_TYPE", "development")
    inert = lambda *a, **k: None                                        # noqa: E731
    for name, value in (("is_apple_silicon", lambda: True), ("apply_gpu_ids", inert),
                        ("_recorded_local_base", lambda m: (None, False)),
                        ("_hub_targets_are_local", lambda *a, **k: True),
                        ("_activate_transformers_version", inert), ("_handle_load", inert)):
        monkeypatch.setattr(worker, name, value)
    monkeypatch.setattr(_hw, "detect_hardware", inert)
    monkeypatch.setattr(_hw, "DEVICE", _hw.DeviceType.MLX)
    monkeypatch.setattr(LogConfig, "setup_logging", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(mlx_mod, "MLXInferenceBackend", lambda *a, **k: backend)

    class _Script:
        def __init__(self, items):
            self._items = list(items)

        def get(self, timeout = None):
            if self._items and (item := self._items.pop(0)) is not IDLE:
                return item
            raise _stdqueue.Empty

    resp = _RespQueue()
    never = SimpleNamespace(is_set = lambda: False, clear = lambda: None, set = lambda: None)
    worker.run_inference_process(
        cmd_queue = _Script([*cmds, {"type": "shutdown"}]), resp_queue = resp,
        cancel_event = never, config = {"model_name": "unsloth/orpheus-3b-0.1-ft"})
    return resp


def test_a_one_slot_load_runs_its_reply_without_ever_opening_a_batch(monkeypatch):
    """A batch one reply wide is one nothing can join, so it is never opened."""
    backend = _Backend()

    resp = _run_loop(monkeypatch, backend, [_cmd("r1", parallel_slots = 1), IDLE])

    assert backend.sessions == [], "a batch was opened for a single reply"
    assert [m["type"] for m in resp.sent if m.get("request_id") == "r1"] == ["token", "gen_done"]


_HOLD_CASES = [
    pytest.param(
        dict(
            cmds = [("r1", 4), ("r2", 2), ("r3", 4), ("r4", 4)],
            script = {"r1": ["a", None], "r2": ["b", None]},
            widths = [4, 2, 4],
            order = ["r1", "r2", "r3", "r4"],
        ),
        id = "the head runs on a queue that never falls quiet",
    ),
    pytest.param(
        dict(
            cmds = [("r1", 4), ("r2", 2), ("r3", 2), ("r4", 2)],
            script = {"r1": ["a"] * 4 + [None], "r2": ["b", "b", "b", None],
                      "r3": ["c", "c", None], "r4": ["d", None]},
            idle = 12,
            widths = [4, 2],
            order = ["r1", "r2", "r3", "r4"],
        ),
        id = "everything held behind one incompatible command still batches together",
    ),
    pytest.param(
        dict(
            cmds = [("r1", 2), ("r2", 2), ("r3", 2)],
            script = {"r1": ["a"] * 4 + [None], "r2": ["b", "b", None], "r3": ["c", None]},
            refuse = ("r2", 0),
            idle = 12,
            refusals = (0, "r2"),
            order = ["r1", "r2", "r3"],
        ),
        id = "a command the batch turned away is not offered to it again",
    ),
    pytest.param(
        dict(
            cmds = [("r1", 4), ("r0", 2), ("r2", 2), ("r3", 2)],
            script = {"r1": ["a"] * 5 + [None], "r0": ["x", "x", "x", None],
                      "r2": ["b", None], "r3": ["c", None]},
            refuse = ("r2", 1),
            idle = 16,
            refusals = (1, "r2"),
            order = ["r1", "r0", "r2", "r3"],
        ),
        id = "a head the batch turns away keeps its place over what waited behind it",
    ),
]


@pytest.mark.parametrize("case", _HOLD_CASES)
def test_the_hold_releases_its_head_and_keeps_its_order(monkeypatch, case):
    """The hold's whole contract, over the traffic that breaks each part of it.

    Its head runs once the batch drains rather than once the queue falls quiet, and into
    a decoding batch that can still take it. A row the session itself refuses is not
    offered again, and goes back to the head rather than behind later arrivals.
    """
    backend = _Backend()
    backend.script = {(name, None): events for name, events in case["script"].items()}
    if "refuse" in case:
        who, when = case["refuse"]
        backend.on_open = lambda session: (
            session.will_not_take.add(who) if len(backend.sessions) == when else None
        )

    resp = _run_loop(
        monkeypatch,
        backend,
        [_cmd(name, parallel_slots = width) for name, width in case["cmds"]]
        + [IDLE] * case.get("idle", 0),
    )

    admitted = [handle[0] for session in backend.sessions for handle, _r in session.admitted]
    assert admitted == case["order"], f"the hold lost or reordered a command: {admitted}"
    if "widths" in case:
        assert [session.opened_at for session in backend.sessions] == case["widths"]
    if "refusals" in case:
        index, name = case["refusals"]
        assert backend.sessions[index].refusals == [(name, None)]
    if case.get("idle"):
        terminal = [m["request_id"] for m in resp.sent if m.get("type") in ("gen_done", "gen_error")]
        assert sorted(terminal) == sorted(case["order"]), f"a held reply was lost: {resp.sent}"


class _DecliningBackend:
    def __init__(self, reason):
        self._reason, self.batched = reason, 0
        self.last_generation_stats = {"usage": {"completion_tokens": 2}}

    def batch_unavailable_reason(self, requests):
        return None if len(requests) < 2 else self._reason

    def generate_chat_response(self, **kwargs):
        # Echoes this row: a fallback decoding row zero twice would look the same.
        yield f"seed {kwargs['seed']}"
        yield f"seed {kwargs['seed']} done"

    def generate_chat_batch(self, requests, **kwargs):
        self.batched += 1
        raise AssertionError("a declined request set must not reach the batch")


def _run(reason, cancelled = False):
    from core.inference import worker

    sent, backend, cancel = [], _DecliningBackend(reason), threading.Event()
    if cancelled:
        cancel.set()
    worker._handle_generate_rows(
        backend,
        {"request_id": "r1", "messages": [{"role": "user", "content": "hi"}],
         "rows": [{"seed": 1}, {"seed": 2}], "parallel_slots": 2},
        SimpleNamespace(put = sent.append), cancel,
    )
    return backend, sent


def test_a_declined_request_set_is_decoded_one_reply_at_a_time():
    """Reporting the command done without decoding would answer a caller's two."""
    backend, sent = _run("a reply asks for stop sequences")
    kinds = [event["type"] for event in sent if event["type"] != "batch_state"]
    tokens = [event for event in sent if event["type"] == "token"]

    assert backend.batched == 0
    assert kinds == ["token", "token", "row_done", "token", "token", "row_done", "gen_done"]
    assert [(event["row"], event["text"]) for event in tokens] == [
        (0, "seed 1"), (0, "seed 1 done"), (1, "seed 2"), (1, "seed 2 done"),
    ], "each row decodes with the request it was given, not with row zero's"
    assert [event["row"] for event in sent if event["type"] == "row_done"] == [0, 1]
    assert all(event["request_id"] == "r1" for event in sent)
    assert all(event["stats"] == backend.last_generation_stats
               for event in sent if event["type"] == "row_done")


def test_a_cancelled_fallback_still_reports_every_row_done():
    """A row nobody decoded is still a row somebody is waiting on."""
    _backend, sent = _run("a reply asks for stop sequences", cancelled = True)

    assert [event["type"] for event in sent if event["type"] != "batch_state"] == ["row_done", "row_done", "gen_done"]
    assert [event["row"] for event in sent if event["type"] == "row_done"] == [0, 1]
    assert all(event["stats"] is None for event in sent if event["type"] == "row_done")


def test_a_batch_that_will_not_open_still_answers_the_reply(monkeypatch):
    """Building the batch is what fails, and the reply predates any batch existing."""
    def refuse(**_kwargs):
        raise RuntimeError("BatchGenerator.insert missing")

    backend = _Backend()
    backend.open_resident_batch = refuse
    resp = _run_loop(monkeypatch, backend, [_cmd(parallel_slots = 2), IDLE])

    assert [m["type"] for m in resp.sent if m["type"] != "status"] == ["token", "gen_done"]
    assert len(backend.apart) == 1, "it was decoded one reply at a time"
