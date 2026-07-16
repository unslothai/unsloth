# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data-recipe job pump resilience.

The pump is the sole consumer of worker events and sole writer of the job
snapshot the status/SSE endpoints read; a handler error must not kill it, or the
job stays wedged "active" and the workflow key is never retired. Fakes only.
"""

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
    m._subs = []
    m._events = []
    m._seq = 0
    job = Job(job_id = "job-test", owner_subject = "owner-a")
    job.status = "active"
    m._job = job
    m._proc = _FakeProc(alive = True)
    m._mp_q = _ScriptedQueue([])
    return m


def test_replaced_dead_generation_still_retires_its_workflow_key(monkeypatch):
    manager = _manager_with_active_job()
    old_job = manager._job
    manager._proc._alive = False
    retired = []

    def replace_during_drain(_queue):
        manager._job = Job(job_id = "job-new", owner_subject = "owner-b")
        manager._proc = _FakeProc(alive = True)
        manager._mp_q = _ScriptedQueue([])
        return []

    monkeypatch.setattr(manager, "_drain_queue", replace_during_drain)
    monkeypatch.setattr(manager, "_retire_workflow_key", retired.append)
    manager._pump_loop()
    assert retired == [old_job]


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
        m._proc._alive = False  # worker exits -> pump should finalize and stop
        pump.join(timeout = 5)

    assert not pump.is_alive()
    # The exited worker is finalized as error (not left wedged "active") and the
    # workflow key is retired despite the earlier handler exceptions.
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

    m._pump_loop()  # returns once it sees the dead worker

    assert m._job.status == "error"
    assert retired and retired[0] is m._job


def test_pump_finalizes_when_read_keeps_raising_on_dead_worker(monkeypatch):
    # A read that keeps raising after the child died must not spin the pump
    # forever: once the worker is gone it falls through to finalize.
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


def test_current_status_exposes_only_global_busy_state_to_non_owner():
    manager = _manager_with_active_job()

    owner_status = manager.get_current_status("owner-a")
    assert owner_status is not None
    assert owner_status["job_id"] == "job-test"

    # Other accounts see only the global busy bit, never job details or controls.
    assert manager.get_current_status("owner-b") == {"busy": True}
    assert manager.get_status("job-test", "owner-b") is None
    assert manager.cancel("job-test", "owner-b") is False
    assert manager._proc.is_alive()


def test_current_status_does_not_report_busy_for_non_blocking_old_job():
    manager = _manager_with_active_job()
    manager._proc._alive = False
    manager._job.status = "completed"

    assert manager.get_current_status("owner-b") is None
    owner_status = manager.get_current_status("owner-a")
    assert owner_status is not None
    assert owner_status["status"] == "completed"
