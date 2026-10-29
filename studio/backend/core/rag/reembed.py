# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Re-embeds docs indexed under a different embedder (dense search skips them) from stored chunk
text, keeping ids; one job per account; follows current model, so a mid-job switch is absorbed."""

from __future__ import annotations

from core.training.account_jobs import account_is_retired, account_key
from utils.account_context import account_thread
import logging
import threading
import uuid

from core.rag import account_db as rag_db

from . import config, embeddings, ingestion, store

logger = logging.getLogger(__name__)

# An embedder that cannot load fails every document; stop instead of failing them one by one.
_MAX_CONSECUTIVE_FAILURES = 3
_RETIRE_JOIN_SECONDS = 10.0

_jobs: dict = {}
_checked: set = set()
_lock = threading.Lock()


class _Stopped(Exception):
    pass


class _Job:
    def __init__(self, identity: str | None = None) -> None:
        self.id = str(uuid.uuid4())
        # Set on automatic jobs: they write this vector space only and stop on a model switch or fallback.
        self.identity = identity
        self.status = "running"
        self.done = 0
        self.failed = 0
        self.remaining = 0
        self.clipped_chunks = 0
        self.current = 0.0
        self.error: str | None = None
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        finished = self.done + self.failed
        total = finished + self.remaining
        return {
            "id": self.id,
            "status": self.status,
            "done": self.done,
            "failed": self.failed,
            "remaining": self.remaining,
            "clipped_chunks": self.clipped_chunks,
            "progress": 1.0 if not total else min(1.0, (finished + self.current) / total),
            "error": self.error,
        }


def _stale_documents(conn, identity: str) -> list[dict]:
    """Completed, searchable documents whose vectors another embedder wrote; files before
    archived conversation turns, newest first."""
    # Chunks, not num_chunks, which archived conversation turns leave at 0.
    rows = conn.execute(
        "SELECT id, scope, embedding_model FROM documents d WHERE status='completed' "
        "AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id=d.id) AND NOT EXISTS "
        "(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        "AND (d.linked_folder_id IS NULL OR EXISTS "
        "(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id)) "
        "ORDER BY d.scope LIKE 'convarchive#_%' ESCAPE '#', d.created_at DESC, d.rowid DESC"
    ).fetchall()
    return [
        dict(r)
        for r in rows
        if not config.embedding_identity_matches(r["embedding_model"], identity)
    ]


def stale_counts() -> dict:
    """How many uploaded documents and archived conversation turns need re-embedding."""
    identity = embeddings.embedding_identity(config.effective_embedding_model())
    conn = rag_db.get_connection()
    try:
        rows = _stale_documents(conn, identity)
    finally:
        conn.close()
    archived = sum(r["scope"].startswith(store.CONVERSATION_ARCHIVE_PREFIX) for r in rows)
    return {"documents": len(rows) - archived, "archived": archived}


# Chunks were sized with the previous model's tokenizer.
_clip = embeddings.clip_to_limit


def _reembed_document(job: _Job, doc: dict, model: str) -> bool:
    conn = rag_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT id, text FROM chunks WHERE document_id=? ORDER BY chunk_index", (doc["id"],)
        ).fetchall()
    finally:
        conn.close()
    # Per document: an ST failure can hand the model to llama-server, whose tokenizer and limit differ.
    count, limit = embeddings.token_counter(model), embeddings.max_tokens(model)
    texts = []
    clipped = 0
    for row in rows:
        text, was_clipped = _clip(row["text"], count, limit)
        texts.append(text)
        clipped += was_clipped

    def on_progress(done: int, total: int) -> None:
        if job.stop.is_set() or account_is_retired():
            raise _Stopped
        job.current = done / total

    vectors, identity = ingestion._embed_all(texts, model, on_progress)
    if job.identity is not None and identity != job.identity:
        raise _Stopped
    conn = rag_db.get_connection()
    try:
        swapped = store.replace_document_vectors(
            conn,
            doc["id"],
            doc["embedding_model"],
            [r["id"] for r in rows],
            vectors,
            identity,
            model,
        )
    finally:
        conn.close()
    if swapped:
        job.done += 1
        job.remaining -= 1
        job.clipped_chunks += clipped
        if clipped:
            logger.info(
                "re-embedded document %s with %d of %d chunks clipped to %s tokens",
                doc["id"],
                clipped,
                len(rows),
                limit,
            )
    return swapped


def _run(job: _Job) -> None:
    # Per model: a failed document is not retried; a refused one only once the row it read changes.
    failed: set[tuple[str, str]] = set()
    refused: set[tuple[str, str | None, str]] = set()
    failures = 0
    try:
        while True:
            if job.stop.is_set() or account_is_retired():
                raise _Stopped
            model = config.effective_embedding_model()
            identity = embeddings.embedding_identity(model)
            if job.identity is not None and identity != job.identity:
                raise _Stopped
            conn = rag_db.get_connection()
            try:
                pending = [
                    d
                    for d in _stale_documents(conn, identity)
                    if (d["id"], identity) not in failed
                    and (d["id"], d["embedding_model"], identity) not in refused
                ]
            finally:
                conn.close()
            if job.identity is not None and not _upgraded_from_llama(pending, job.identity):
                raise _Stopped
            job.remaining = len(pending)
            if not pending:
                job.status = "completed"
                return
            doc = pending[0]
            job.current = 0.0
            try:
                if not _reembed_document(job, doc, model):
                    refused.add((doc["id"], doc["embedding_model"], identity))
                failures = 0
            except _Stopped:
                raise
            except embeddings.EmbeddingModelDownloadRequiredError as exc:
                # The model, not the document: every other document would fail the same way.
                job.error = str(exc)
                job.status = "failed"
                return
            except Exception as exc:  # noqa: BLE001 - one bad document must not end the job
                logger.warning("re-embedding document %s failed", doc["id"], exc_info = True)
                failed.add((doc["id"], identity))
                job.failed += 1
                job.remaining -= 1
                job.error = str(exc)
                failures += 1
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    job.status = "failed"
                    return
    except _Stopped:
        job.status = "cancelled"
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        logger.exception("re-embed job %s failed", job.id)
        job.error = str(exc)
        job.status = "failed"
    finally:
        job.current = 0.0


def start(identity: str | None = None) -> dict:
    """Start re-embedding this account's stale documents, or return the job already running."""
    if account_is_retired():
        raise RuntimeError("Account is retired")
    key = account_key("reembed")
    with _lock:
        job = _jobs.get(key)
        if job is not None and job.status == "running":
            if identity is None:
                # A Re-embed click adopts an automatic job, which then follows the current model.
                job.identity = None
            return job.snapshot()
        job = _Job(identity)
        job.remaining = sum(stale_counts().values())
        job.thread = account_thread(target = _run, args = (job,), daemon = True)
        _jobs[key] = job
        # Before the worker runs: a job that ends at once must still read as started to the client.
        started = job.snapshot()
        # Under the lock, so retirement never finds a registered job it cannot join yet.
        job.thread.start()
    return started


def _upgraded_from_llama(stale: list[dict], identity: str) -> bool:
    """Every stale document is this identity's model written by llama-server."""
    model = config.embedding_identity_model(identity)
    return all(
        d["embedding_model"].startswith("llama-server:")
        and config.embedding_identity_model(d["embedding_model"]) == model
        for d in stale
    )


def start_after_backend_change(identity: str) -> None:
    """Re-embed on its own only after an upgrade moved the same model from llama-server to
    sentence-transformers; a runtime fallback the other way must not overwrite the ST vectors."""
    if not identity.startswith("sentence-transformers:"):
        return
    key = account_key("reembed")
    with _lock:
        if (key, identity) in _checked or key in _jobs:
            return
        _checked.add((key, identity))
    model = config.embedding_identity_model(identity)
    if model != config.effective_embedding_model():
        return
    conn = rag_db.get_connection()
    try:
        stale = _stale_documents(conn, identity)
    finally:
        conn.close()
    if stale and _upgraded_from_llama(stale, identity):
        start(identity)


def status() -> dict | None:
    with _lock:
        job = _jobs.get(account_key("reembed"))
    return job.snapshot() if job is not None else None


def cancel() -> dict | None:
    with _lock:
        job = _jobs.get(account_key("reembed"))
    if job is None:
        return None
    job.stop.set()
    return job.snapshot()


def retire_account_reembed() -> None:
    """Stop this account's job before its storage moves."""
    key = account_key("reembed")
    with _lock:
        job = _jobs.get(key)
    if job is None or job.thread is None:
        return
    job.stop.set()
    if job.thread is not threading.current_thread():
        job.thread.join(timeout = _RETIRE_JOIN_SECONDS)
        if job.thread.is_alive():
            # Left registered, so a retried retirement waits for it again.
            raise RuntimeError("Retired account re-embed job has not stopped")
    with _lock:
        _jobs.pop(key, None)
