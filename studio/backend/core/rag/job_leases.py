# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Renewable SQLite leases for RAG work shared by multiple backend processes."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone

from storage import rag_db
from utils.workspace_context import (
    current_workspace_subject,
    run_in_workspace,
    workspace_thread,
)

logger = logging.getLogger(__name__)

INGESTION = "ingestion"
FOLDER_SYNC = "folder_sync"

_OWNER_ID = str(uuid.uuid4())
_LEASE_SECONDS = 30
_HEARTBEAT_SECONDS = 5
_active: set[tuple[str, str, str]] = set()
_lock = threading.Lock()
_wake = threading.Event()
_thread: threading.Thread | None = None


class JobLeaseLost(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deadline() -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds = _LEASE_SECONDS)).isoformat()


def claim(conn, kind: str, job_id: str) -> bool:
    """Claim an unleased or expired job inside the caller's transaction."""
    cursor = conn.execute(
        "INSERT INTO rag_job_leases(kind, job_id, owner_id, expires_at) VALUES(?,?,?,?) "
        "ON CONFLICT(kind, job_id) DO UPDATE SET owner_id=excluded.owner_id, "
        "expires_at=excluded.expires_at WHERE rag_job_leases.owner_id=excluded.owner_id "
        "OR rag_job_leases.expires_at<=?",
        (kind, job_id, _OWNER_ID, _deadline(), _now()),
    )
    return cursor.rowcount == 1


def owned_by_this_process(conn, kind: str, job_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM rag_job_leases WHERE kind=? AND job_id=? AND owner_id=? "
            "AND expires_at>?",
            (kind, job_id, _OWNER_ID, _now()),
        ).fetchone()
        is not None
    )


def renew_owned(conn, kind: str, job_id: str) -> bool:
    """Renew only if no other process reclaimed the job."""
    cursor = conn.execute(
        "UPDATE rag_job_leases SET expires_at=? WHERE kind=? AND job_id=? AND owner_id=?",
        (_deadline(), kind, job_id, _OWNER_ID),
    )
    return cursor.rowcount == 1


def activate(kind: str, job_id: str) -> None:
    """Renew a committed claim until the local worker releases it."""
    global _thread
    with _lock:
        subject = current_workspace_subject()
        _active.add((subject, kind, job_id))
        if _thread is None or not _thread.is_alive():
            try:
                _thread = workspace_thread(target = _heartbeat, daemon = True)
                _thread.start()
            except Exception:
                _thread = None
                _active.discard((subject, kind, job_id))
                raise
    _wake.set()


def release(kind: str, job_id: str) -> None:
    """Stop renewal and remove this process's persisted claim."""
    subject = current_workspace_subject()
    with _lock:
        _active.discard((subject, kind, job_id))
    try:
        conn = rag_db.get_connection()
        try:
            conn.execute(
                "DELETE FROM rag_job_leases WHERE kind=? AND job_id=? AND owner_id=?",
                (kind, job_id, _OWNER_ID),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning("failed to release RAG job lease", exc_info = True)


def _renew_workspace(active: tuple[tuple[str, str], ...]) -> None:
    conn = rag_db.get_connection()
    try:
        deadline = _deadline()
        for kind, job_id in active:
            conn.execute(
                "UPDATE rag_job_leases SET expires_at=? "
                "WHERE kind=? AND job_id=? AND owner_id=?",
                (deadline, kind, job_id, _OWNER_ID),
            )
        conn.commit()
    finally:
        conn.close()


def _heartbeat() -> None:
    while True:
        with _lock:
            active = tuple(_active)
        if not active:
            _wake.wait()
            _wake.clear()
            continue
        try:
            by_subject: dict[str, list[tuple[str, str]]] = {}
            for subject, kind, job_id in active:
                by_subject.setdefault(subject, []).append((kind, job_id))
            for subject, workspace_jobs in by_subject.items():
                try:
                    run_in_workspace(subject, _renew_workspace, tuple(workspace_jobs))
                except Exception:
                    logger.warning(
                        "failed to renew RAG job leases for workspace %s",
                        subject,
                        exc_info = True,
                    )
        except Exception:
            logger.warning("failed to renew RAG job leases", exc_info = True)
        _wake.wait(_HEARTBEAT_SECONDS)
        _wake.clear()
