# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""job_events keeps the per-job queue registered until the job is terminal.

``_emit()`` writes to ``_jobs[job_id]`` while the worker runs; if an early SSE
disconnect removed that queue, later events would be dropped and a reconnect
would see only ``[DONE]`` and mark a running job complete. Remove only on a
terminal exit; ``_reap_finished_jobs`` sweeps leftovers.
"""

import queue
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
