# SPDX-License-Identifier: AGPL-3.0-only
"""Stopping the dispatcher wakes it at once instead of waiting out its response-queue poll."""

import multiprocessing as mp
import queue
import threading
import time

import core.inference.orchestrator as orch_mod
from core.inference.orchestrator import InferenceOrchestrator

_WAKE = {"type": orch_mod._DISPATCHER_WAKE}


def _with_queue(resp_queue):
    o = InferenceOrchestrator()
    o._resp_queue = resp_queue
    return o


def test_stop_does_not_wait_for_the_poll_timeout(monkeypatch):
    # A poll this long would make the old stop hit its join timeout instead of returning.
    monkeypatch.setattr(orch_mod, "_DISPATCH_POLL_INTERVAL", 60.0)
    monkeypatch.setattr(orch_mod, "_DISPATCH_STOP_TIMEOUT", 10.0)
    o = _with_queue(mp.get_context("spawn").Queue())
    thread = o._start_dispatcher()
    time.sleep(0.2)  # parked in its poll
    started = time.monotonic()
    o._stop_dispatcher()
    assert not thread.is_alive()
    assert time.monotonic() - started < 5.0


def test_failed_wake_still_stops_on_the_poll(monkeypatch):
    monkeypatch.setattr(orch_mod, "_DISPATCH_POLL_INTERVAL", 0.1)

    class _NoPut(queue.Queue):
        def put_nowait(self, item):
            raise OSError("closed")

    o = _with_queue(_NoPut())
    thread = o._start_dispatcher()
    o._stop_dispatcher()
    assert not thread.is_alive()


def test_stray_wake_is_invisible_to_every_reader(monkeypatch):
    q = queue.Queue()
    o = _with_queue(q)
    observed = []
    monkeypatch.setattr(o, "_observe_response", lambda resp, worker: observed.append(resp) or resp)
    q.put(_WAKE)
    q.put({"type": "token", "request_id": "r1", "text": "a"})
    q.put(_WAKE)
    assert o._read_resp(timeout = 1.0) == {"type": "token", "request_id": "r1", "text": "a"}
    assert o._read_resp(timeout = 0.05) is None  # only a wake left: reads as empty
    assert observed == [{"type": "token", "request_id": "r1", "text": "a"}]
    q.put(_WAKE)
    q.put({"type": "gen_done", "request_id": "r1"})
    assert o._drain_queue() == [{"type": "gen_done", "request_id": "r1"}]


def test_restarted_dispatcher_skips_a_stray_wake_and_keeps_order(monkeypatch):
    # A wake left by a stop that raced a real reply must neither stop the next dispatcher on the
    # same worker nor reach or reorder a mailbox.
    monkeypatch.setattr(orch_mod, "_DISPATCH_POLL_INTERVAL", 0.05)
    q = queue.Queue()
    o = _with_queue(q)
    mailbox = queue.Queue()
    with o._mailbox_lock:
        o._mailboxes["r1"] = mailbox
    q.put(_WAKE)
    q.put({"type": "token", "request_id": "r1", "text": "a"})
    q.put(_WAKE)
    q.put({"type": "gen_done", "request_id": "r1"})
    thread = o._start_dispatcher()
    got = [mailbox.get(timeout = 5) for _ in range(2)]
    assert [r["type"] for r in got] == ["token", "gen_done"]
    assert thread.is_alive()
    o._stop_dispatcher()
    assert not thread.is_alive() and mailbox.empty()


def test_wake_is_sent_only_while_stopping_a_live_dispatcher():
    q = queue.Queue()
    o = _with_queue(q)
    o._stop_dispatcher()  # none running
    assert q.empty()
    stopped = threading.Thread(target = lambda: None)
    stopped.start()
    stopped.join()
    o._dispatcher_thread = stopped
    o._stop_dispatcher()
    assert q.empty()
