# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""job_events keeps the per-job queue registered only while the worker runs.

``_emit()`` writes to ``_jobs[job_id]`` while the worker runs; if an early SSE
disconnect removed that queue, later events would be dropped and a reconnect
would see only ``[DONE]`` and mark a running job complete. So keep it on an early
disconnect of a running job, but drop it on a terminal exit or a disconnect after
the job already finished; ``_reap_finished_jobs`` sweeps any leftovers.
"""

import queue
import sqlite3
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.rag.ingestion as ing


def test_early_disconnect_keeps_queue_registered(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    # Job is still running; nothing terminal has happened.
    monkeypatch.setattr(ing, "get_job_status", lambda _jid: {"status": "running"})
    jid = "job-early-disconnect"
    ing._jobs[jid] = queue.Queue()
    try:
        gen = ing.job_events(jid)
        next(gen)  # enter loop: Empty -> non-terminal -> heartbeat
        gen.close()  # client disconnects before the job finishes
        assert (
            jid in ing._jobs
        ), "queue must survive an early disconnect so the worker can still emit"
    finally:
        ing._jobs.pop(jid, None)


def test_terminal_sentinel_removes_queue(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    jid = "job-terminal-sentinel"
    q = queue.Queue()
    q.put({"type": "progress", "stage": "embedding", "progress": 0.5})
    q.put(None)  # worker finished -> sentinel
    ing._jobs[jid] = q
    try:
        events = list(ing.job_events(jid))  # drains progress, then None -> terminal
        assert any(e.get("type") == "progress" for e in events)
        assert jid not in ing._jobs, "queue must be removed once the job is terminal"
    finally:
        ing._jobs.pop(jid, None)


def test_disconnect_after_terminal_event_removes_queue(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    # Worker finished: the DB row is terminal and a complete event is queued. The
    # UI reads that event and disconnects (reader.cancel) before the None sentinel,
    # so the queue must still drop rather than linger until the next reap.
    monkeypatch.setattr(ing, "get_job_status", lambda _jid: {"status": "completed"})
    jid = "job-disconnect-after-complete"
    q = queue.Queue()
    q.put({"type": "complete", "num_chunks": 3})
    q.put(None)
    ing._jobs[jid] = q
    try:
        gen = ing.job_events(jid)
        assert next(gen)["type"] == "complete"  # client receives the terminal event
        gen.close()  # disconnects before draining the sentinel
        assert jid not in ing._jobs, "a finished job's queue must drop on disconnect"
    finally:
        ing._jobs.pop(jid, None)


def test_transient_status_read_failure_does_not_end_stream(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    # The heartbeat poll hits a momentarily-locked DB. That must not propagate: the
    # SSE route would turn the raised error into a terminal {type: error} frame and
    # the UI would drop a document whose worker is still running. The stream should
    # heartbeat and keep the queue so the worker can finish / a reconnect can resume.
    calls = {"n": 0}

    def flaky_status(_jid):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return {"status": "running"}

    monkeypatch.setattr(ing, "get_job_status", flaky_status)
    jid = "job-transient-read-failure"
    ing._jobs[jid] = queue.Queue()
    try:
        gen = ing.job_events(jid)
        assert next(gen) == {"type": "heartbeat"}  # transient error -> heartbeat, no raise
        gen.close()
        assert jid in ing._jobs, "an unconfirmed (transient-error) status must keep the queue"
    finally:
        ing._jobs.pop(jid, None)


def test_terminal_db_status_removes_queue(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    # No events arrive, but the DB row reports the job finished (hard worker death
    # that skipped the sentinel): the stream ends and the queue is reaped.
    monkeypatch.setattr(ing, "get_job_status", lambda _jid: {"status": "completed"})
    jid = "job-terminal-db"
    ing._jobs[jid] = queue.Queue()
    try:
        list(ing.job_events(jid))
        assert jid not in ing._jobs, "a terminal DB status must remove the queue"
    finally:
        ing._jobs.pop(jid, None)
