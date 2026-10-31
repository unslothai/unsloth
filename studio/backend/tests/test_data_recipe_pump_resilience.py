# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A handler error must not kill the pump, or the job stays active and its workflow key never retires."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.data_recipe.jobs.manager import JobManager  # noqa: E402
from core.data_recipe.jobs.types import Job  # noqa: E402


class _FakeProc:
    def __init__(self, alive: bool = True):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _ScriptedQueue:
    def __init__(self, events):
        self._events = list(events)

    def get(self, timeout = None):
        if self._events:
            return self._events.pop(0)
        raise queue.Empty

    def get_nowait(self):
        if self._events:
            return self._events.pop(0)
        raise queue.Empty


def _wait_until(predicate, timeout = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _manager_with_active_job():
    m = JobManager.__new__(JobManager)
    m._lock = threading.Lock()
    job = Job(job_id = "job-test")
    job.status = "active"
    m._job = job
    m._proc = _FakeProc(alive = True)
    m._mp_q = _ScriptedQueue([])
    return m


def test_pump_survives_handler_exception_and_still_finalizes(monkeypatch):
    m = _manager_with_active_job()
    handled: list = []

    def fake_handle(job, event):
        if event.get("type") == "boom":
            raise RuntimeError("malformed log line")
        handled.append(event.get("type"))

    emitted: list = []
    retired: list = []
    monkeypatch.setattr(m, "_handle_event", fake_handle)
    monkeypatch.setattr(m, "_emit", lambda e: emitted.append(e))
    monkeypatch.setattr(m, "_retire_workflow_key", lambda j: retired.append(j))

    m._mp_q = _ScriptedQueue(
        [{"type": "boom"}, {"type": "log"}, {"type": "boom"}, {"type": "progress"}]
    )

    pump = threading.Thread(target = m._pump_loop, daemon = True)
    pump.start()
    try:
        assert _wait_until(
            lambda: handled == ["log", "progress"]
        ), "pump must keep processing events after a handler raises"
        assert pump.is_alive()
    finally:
        m._proc._alive = False
        pump.join(timeout = 5)

    assert not pump.is_alive()
    assert m._job.status == "error"
    assert retired and retired[0] is m._job


def test_pump_finalizes_when_drain_raises(monkeypatch):
    m = _manager_with_active_job()
    monkeypatch.setattr(m, "_emit", lambda e: None)
    retired: list = []
    monkeypatch.setattr(m, "_retire_workflow_key", lambda j: retired.append(j))

    class _BadDrainQueue:
        def get(self, timeout = None):
            raise queue.Empty

        def get_nowait(self):
            raise RuntimeError("corrupt drain payload")

    m._proc = _FakeProc(alive = False)
    m._mp_q = _BadDrainQueue()

    m._pump_loop()

    assert m._job.status == "error"
    assert retired and retired[0] is m._job


def test_pump_finalizes_when_read_keeps_raising_on_dead_worker(monkeypatch):
    m = _manager_with_active_job()
    monkeypatch.setattr(m, "_emit", lambda e: None)
    retired: list = []
    monkeypatch.setattr(m, "_retire_workflow_key", lambda j: retired.append(j))

    class _BrokenReadQueue:
        def get(self, timeout = None):
            raise RuntimeError("broken queue pipe")

        def get_nowait(self):
            raise queue.Empty

    m._proc = _FakeProc(alive = False)
    m._mp_q = _BrokenReadQueue()

    pump = threading.Thread(target = m._pump_loop, daemon = True)
    pump.start()
    pump.join(timeout = 5)
    assert not pump.is_alive(), "pump must finalize a dead worker even when reads keep raising"
    assert m._job.status == "error"
    assert retired and retired[0] is m._job


def test_finished_pump_leaves_the_next_run_alone(monkeypatch):
    # Run 1's worker exits and run 2 starts before run 1's pump notices: run 1's pump must
    # finalize run 1 only, not fail run 2 and revoke the key run 2's worker is about to use.
    m = _manager_with_active_job()
    first = m._job
    first.status = "completed"
    first.internal_api_key_id = 1
    m._proc._alive = False
    emitted: list = []
    monkeypatch.setattr(m, "_emit", lambda e: emitted.append(e))
    revoked: list = []

    class _Storage:
        @staticmethod
        def revoke_internal_api_key(key_id):
            revoked.append(key_id)

    import auth

    monkeypatch.setattr(auth, "storage", _Storage, raising = False)
    second = Job(job_id = "job-next")
    second.status = "pending"
    second.internal_api_key_id = 2
    second_q = _ScriptedQueue([])

    class _NextRunStartsDuringRead:
        def get(self, timeout = None):
            with m._lock:  # what start() does for run 2
                m._job, m._proc, m._mp_q = second, _FakeProc(alive = True), second_q
            raise queue.Empty

        def get_nowait(self):
            raise queue.Empty

    m._mp_q = _NextRunStartsDuringRead()
    pump = threading.Thread(target = m._pump_loop, daemon = True)
    pump.start()
    pump.join(timeout = 5)

    assert not pump.is_alive(), "run 1's pump must stop instead of serving run 2's queue"
    assert second.status == "pending" and second.error is None
    assert second.internal_api_key_id == 2 and 2 not in revoked
    assert emitted == []
