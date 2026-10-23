# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep a running job's event queue on early SSE disconnect; dropping it marks a live job complete."""

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
    monkeypatch.setattr(ing, "get_job_status", lambda _jid: {"status": "running"})
    jid = "job-early-disconnect"
    ing._jobs[jid] = queue.Queue()
    try:
        gen = ing.job_events(jid)
        next(gen)
        gen.close()
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
    q.put(None)
    ing._jobs[jid] = q
    try:
        events = list(ing.job_events(jid))
        assert any(e.get("type") == "progress" for e in events)
        assert jid not in ing._jobs, "queue must be removed once the job is terminal"
    finally:
        ing._jobs.pop(jid, None)


def test_disconnect_after_terminal_event_removes_queue(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    # The UI disconnects on the complete event before the None sentinel; the queue must still drop.
    monkeypatch.setattr(ing, "get_job_status", lambda _jid: {"status": "completed"})
    jid = "job-disconnect-after-complete"
    q = queue.Queue()
    q.put({"type": "complete", "num_chunks": 3})
    q.put(None)
    ing._jobs[jid] = q
    try:
        gen = ing.job_events(jid)
        assert next(gen)["type"] == "complete"
        gen.close()
        assert jid not in ing._jobs, "a finished job's queue must drop on disconnect"
    finally:
        ing._jobs.pop(jid, None)


def test_disconnect_after_cancelled_event_removes_queue(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(ing, "get_job_status", lambda _jid: {"status": "cancelled"})
    jid = "job-disconnect-after-cancelled"
    q = queue.Queue()
    q.put({"type": "error", "stage": "cancelled", "error": "Document was deleted"})
    q.put(None)
    ing._jobs[jid] = q
    try:
        gen = ing.job_events(jid)
        assert next(gen)["stage"] == "cancelled"
        gen.close()
        assert jid not in ing._jobs, "a cancelled job's queue must drop on disconnect"
    finally:
        ing._jobs.pop(jid, None)


def test_reaper_removes_cancelled_queue(monkeypatch):
    monkeypatch.setattr(ing, "get_job_status", lambda _jid: {"status": "cancelled"})
    jid = "job-reap-cancelled"
    ing._jobs[jid] = queue.Queue()
    try:
        ing._reap_finished_jobs()
        assert jid not in ing._jobs, "a cancelled job must not remain in the queue registry"
    finally:
        ing._jobs.pop(jid, None)


def test_transient_status_read_failure_does_not_end_stream(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    # A locked DB on heartbeat must not raise: the SSE route would emit a terminal error frame.
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
        assert next(gen) == {"type": "heartbeat"}
        gen.close()
        assert jid in ing._jobs, "an unconfirmed (transient-error) status must keep the queue"
    finally:
        ing._jobs.pop(jid, None)


def test_terminal_db_status_removes_queue(monkeypatch):
    monkeypatch.setattr(ing, "_SSE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(ing, "get_job_status", lambda _jid: {"status": "completed"})
    jid = "job-terminal-db"
    ing._jobs[jid] = queue.Queue()
    try:
        list(ing.job_events(jid))
        assert jid not in ing._jobs, "a terminal DB status must remove the queue"
    finally:
        ing._jobs.pop(jid, None)


def test_consumed_internal_job_removes_terminal_row_and_queue(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE ingestion_jobs(id TEXT PRIMARY KEY, status TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO ingestion_jobs(id, status) VALUES(?, ?)",
            [("terminal", "completed"), ("active", "running")],
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(ing.rag_db, "get_connection", lambda: sqlite3.connect(db_path))
    ing._jobs["terminal"] = queue.Queue()
    ing._jobs["active"] = queue.Queue()
    try:
        assert ing.delete_terminal_job("terminal") is True
        assert ing.delete_terminal_job("active") is False
        assert "terminal" not in ing._jobs
        assert "active" in ing._jobs
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("SELECT id FROM ingestion_jobs ORDER BY id").fetchall() == [
                ("active",)
            ]
        finally:
            conn.close()
    finally:
        ing._jobs.pop("terminal", None)
        ing._jobs.pop("active", None)
