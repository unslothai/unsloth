# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Document ingestion pipeline.

Follows the studio's existing job pattern (`core/data_recipe/jobs/manager.py`):
spawn a fresh subprocess per job with ``mp.get_context("spawn")`` and stream
progress events back over a queue. The subprocess does the heavy work
(parse → chunk → load embedder → embed in batches) and ships
``(chunks, vectors)`` batches back. The parent persists everything:
sqlite rows, Qdrant points, and (at job completion) a rebuilt BM25 index.

Only the parent process holds the Qdrant local-mode file lock — the
subprocess never opens it directly. This keeps search available
throughout the lifetime of an ingestion job.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as queue_module
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from storage.studio_db import get_connection
from utils.rag.config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_EMBED_BATCH_SIZE,
    RAG_EMBEDDING_MODEL,
)

from . import bm25, embeddings, vector_store
from .vector_store import kb_scope, thread_scope

logger = logging.getLogger(__name__)

_CTX = mp.get_context("spawn")
_QUEUE_TIMEOUT_SECONDS = 300


# ------------------------------------------------------------------
# Subprocess worker
# ------------------------------------------------------------------

def _subprocess_worker(
    stored_path: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    out_queue: Any,
    chunking_strategy: str = "standard",
    mode: str = "text",
) -> None:
    try:
        from core.rag.chunking import chunk_pages, chunk_pages_with_spans
        from core.rag.parsers import parse

        out_queue.put({"type": "progress", "stage": "parse", "progress": 0.05})
        # want_images is True only for multimodal KBs. The image side of
        # the pipeline lands in Phase 3B-multimodal; for now the parser
        # collects the bytes anyway in case we want them later, but only
        # the text pages are consumed.
        parsed = parse(Path(stored_path), want_images = (mode == "multimodal"))
        pages = parsed.pages
        if not pages:
            out_queue.put({"type": "error", "error": "no extractable text in document"})
            return

        out_queue.put({"type": "progress", "stage": "load_model", "progress": 0.1})
        from core.rag.embeddings import (
            get_embedder,
            late_chunk_encode,
            token_counter,
        )

        model = get_embedder(model_name)
        counter = token_counter(model_name)
        dim = int(model.get_sentence_embedding_dimension())
        out_queue.put({"type": "dim", "dim": dim})

        if chunking_strategy == "late":
            _run_late_chunking(
                pages = pages,
                chunk_size = chunk_size,
                overlap = overlap,
                counter = counter,
                model_name = model_name,
                late_chunk_encode = late_chunk_encode,
                out_queue = out_queue,
            )
        else:
            _run_standard_chunking(
                pages = pages,
                chunk_size = chunk_size,
                overlap = overlap,
                counter = counter,
                batch_size = batch_size,
                model = model,
                chunk_pages = chunk_pages,
                out_queue = out_queue,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingestion subprocess failed")
        out_queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})


def _run_standard_chunking(
    *,
    pages,
    chunk_size,
    overlap,
    counter,
    batch_size,
    model,
    chunk_pages,
    out_queue,
) -> None:
    out_queue.put({"type": "progress", "stage": "chunk", "progress": 0.2})
    chunks = chunk_pages(
        pages,
        max_tokens = chunk_size,
        overlap_tokens = overlap,
        token_counter = counter,
    )
    if not chunks:
        out_queue.put({"type": "error", "error": "chunker produced no chunks"})
        return

    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vectors = model.encode(
            [c.text for c in batch],
            batch_size = batch_size,
            normalize_embeddings = True,
            convert_to_numpy = True,
            show_progress_bar = False,
        )
        out_queue.put(
            {
                "type": "chunks_batch",
                "first_index": i,
                "chunks": [
                    {
                        "text": c.text,
                        "token_count": c.token_count,
                        "page_number": c.page_number,
                    }
                    for c in batch
                ],
                "vectors": vectors.tolist(),
            }
        )
        progress = 0.3 + 0.65 * min(1.0, (i + len(batch)) / total)
        out_queue.put({"type": "progress", "stage": "embed", "progress": progress})

    out_queue.put({"type": "complete", "num_chunks": total})


def _run_late_chunking(
    *,
    pages,
    chunk_size,
    overlap,
    counter,
    model_name,
    late_chunk_encode,
    out_queue,
) -> None:
    """Late chunking: chunk once over the whole doc, embed in a single pass.

    There's no per-batch streaming here — the whole doc is encoded in
    one forward pass (or one per window for long docs). We ship all
    chunks back to the parent in one message; the parent's pump still
    handles them via the same chunks_batch handler.
    """
    from core.rag.chunking import chunk_pages_with_spans

    out_queue.put({"type": "progress", "stage": "chunk", "progress": 0.2})
    full_doc, chunks, char_spans = chunk_pages_with_spans(
        pages,
        max_tokens = chunk_size,
        overlap_tokens = overlap,
        token_counter = counter,
    )
    if not chunks:
        out_queue.put({"type": "error", "error": "chunker produced no chunks"})
        return

    out_queue.put({"type": "progress", "stage": "embed", "progress": 0.4})
    vectors = late_chunk_encode(
        full_doc,
        char_spans,
        model_name = model_name,
        normalize = True,
    )

    out_queue.put({"type": "progress", "stage": "embed", "progress": 0.9})
    out_queue.put(
        {
            "type": "chunks_batch",
            "first_index": 0,
            "chunks": [
                {
                    "text": c.text,
                    "token_count": c.token_count,
                    "page_number": c.page_number,
                }
                for c in chunks
            ],
            "vectors": [v.tolist() for v in vectors],
        }
    )
    out_queue.put({"type": "complete", "num_chunks": len(chunks)})


# ------------------------------------------------------------------
# Job manager (parent side)
# ------------------------------------------------------------------

class _JobState:
    def __init__(self, job_id: str, document_id: str, scope: str) -> None:
        self.job_id = job_id
        self.document_id = document_id
        self.scope = scope
        self.status = "pending"
        self.stage: str | None = None
        self.progress: float = 0.0
        self.error: str | None = None
        self.subscribers: list[queue_module.Queue[dict]] = []
        self.lock = threading.Lock()

    def push_event(self, event: dict) -> None:
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue_module.Full:
                pass

    def subscribe(self) -> queue_module.Queue[dict]:
        q: queue_module.Queue[dict] = queue_module.Queue(maxsize = 256)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue_module.Queue[dict]) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


_jobs: dict[str, _JobState] = {}
_jobs_lock = threading.Lock()


def get_job_state(job_id: str) -> _JobState | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _scope_for(kb_id: str | None, thread_id: str | None) -> str:
    if kb_id:
        return kb_scope(kb_id)
    if thread_id:
        return thread_scope(thread_id)
    raise ValueError("must supply kb_id or thread_id")


def _update_job_row(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    keys = list(fields.keys())
    set_clause = ", ".join(f"{k} = ?" for k in keys)
    values = list(fields.values()) + [job_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE rag_ingestion_jobs SET {set_clause} WHERE id = ?", values)
        conn.commit()


def _update_document_row(document_id: str, **fields: Any) -> None:
    if not fields:
        return
    keys = list(fields.keys())
    set_clause = ", ".join(f"{k} = ?" for k in keys)
    values = list(fields.values()) + [document_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE rag_documents SET {set_clause} WHERE id = ?", values)
        conn.commit()


def _insert_chunks_and_collect_for_bm25(
    document_id: str,
    scope: str,
    first_index: int,
    chunks_meta: list[dict],
    vectors: list[list[float]],
) -> list[dict]:
    """Insert chunks into sqlite + Qdrant; return [{id, text}] for BM25."""
    rows: list[tuple] = []
    points: list[dict] = []
    bm25_rows: list[dict] = []
    for offset, (meta, vec) in enumerate(zip(chunks_meta, vectors)):
        chunk_index = first_index + offset
        chunk_id = str(uuid4())
        rows.append(
            (
                chunk_id,
                document_id,
                chunk_index,
                meta["text"],
                meta["token_count"],
                meta["page_number"],
            )
        )
        points.append(
            {
                "id": chunk_id,
                "vector": vec,
                "payload": {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": meta["text"],
                    "page_number": meta["page_number"],
                },
            }
        )
        bm25_rows.append({"id": chunk_id, "text": meta["text"]})
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO rag_chunks
            (id, document_id, chunk_index, text, token_count, page_number)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    vector_store.upsert_chunks(scope, points)
    return bm25_rows


def _all_scope_chunks(scope: str) -> list[dict]:
    if scope.startswith("kb_"):
        kb_id = scope[len("kb_"):]
        sql = (
            "SELECT c.id, c.text FROM rag_chunks c "
            "JOIN rag_documents d ON d.id = c.document_id "
            "WHERE d.kb_id = ?"
        )
        bind = (kb_id,)
    elif scope.startswith("thread_"):
        thread_id = scope[len("thread_"):]
        sql = (
            "SELECT c.id, c.text FROM rag_chunks c "
            "JOIN rag_documents d ON d.id = c.document_id "
            "WHERE d.thread_id = ?"
        )
        bind = (thread_id,)
    else:
        return []
    with get_connection() as conn:
        rows = conn.execute(sql, bind).fetchall()
    return [{"id": r["id"], "text": r["text"]} for r in rows]


def _pump(
    state: _JobState,
    proc: Any,
    out_queue: Any,
) -> None:
    """Drain queue messages until the subprocess signals complete/error or dies."""
    bm25_buffer: list[dict] = []
    embedding_dim: int | None = None
    final_status = "failed"
    final_error: str | None = None
    final_num_chunks = 0

    started_at = int(time.time())
    state.status = "running"
    _update_job_row(state.job_id, status = "running", started_at = started_at)
    _update_document_row(state.document_id, status = "running")
    state.push_event({"type": "status", "status": "running"})

    try:
        while True:
            try:
                msg = out_queue.get(timeout = _QUEUE_TIMEOUT_SECONDS)
            except queue_module.Empty:
                if not proc.is_alive():
                    final_error = "subprocess exited without completion message"
                    break
                continue
            mtype = msg.get("type")
            if mtype == "progress":
                state.stage = msg.get("stage")
                state.progress = float(msg.get("progress", 0.0))
                _update_job_row(
                    state.job_id,
                    stage = state.stage,
                    progress = state.progress,
                )
                state.push_event(msg)
            elif mtype == "dim":
                embedding_dim = int(msg["dim"])
                vector_store.ensure_collection(state.scope, embedding_dim)
            elif mtype == "chunks_batch":
                if embedding_dim is None:
                    # defensive: subprocess should always emit "dim" first
                    embedding_dim = len(msg["vectors"][0]) if msg["vectors"] else None
                    if embedding_dim is not None:
                        vector_store.ensure_collection(state.scope, embedding_dim)
                bm25_rows = _insert_chunks_and_collect_for_bm25(
                    state.document_id,
                    state.scope,
                    int(msg["first_index"]),
                    msg["chunks"],
                    msg["vectors"],
                )
                bm25_buffer.extend(bm25_rows)
            elif mtype == "complete":
                final_status = "completed"
                final_num_chunks = int(msg.get("num_chunks", len(bm25_buffer)))
                break
            elif mtype == "error":
                final_error = str(msg.get("error", "unknown error"))
                break
            else:
                logger.warning("ingestion: unknown message type %r", mtype)
    finally:
        proc.join(timeout = 30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout = 5)

    finished_at = int(time.time())
    if final_status == "completed":
        full_scope_chunks = _all_scope_chunks(state.scope)
        bm25.rebuild_index(state.scope, full_scope_chunks)
        _update_document_row(
            state.document_id,
            status = "completed",
            num_chunks = final_num_chunks,
        )
        _update_job_row(
            state.job_id,
            status = "completed",
            progress = 1.0,
            stage = "done",
            finished_at = finished_at,
        )
        state.status = "completed"
        state.progress = 1.0
        state.push_event(
            {
                "type": "complete",
                "num_chunks": final_num_chunks,
            }
        )
    else:
        _update_document_row(
            state.document_id,
            status = "failed",
            error = final_error,
        )
        _update_job_row(
            state.job_id,
            status = "failed",
            error = final_error,
            finished_at = finished_at,
        )
        state.status = "failed"
        state.error = final_error
        state.push_event({"type": "error", "error": final_error})


def enqueue_ingestion(
    document_id: str,
    stored_path: Path,
    *,
    kb_id: str | None = None,
    thread_id: str | None = None,
    embedding_model: str | None = None,
    chunking_strategy: str = "standard",
    mode: str = "text",
) -> str:
    """Create the job row, spawn the subprocess, and start the pump thread.

    Returns the job_id. The caller can poll via ``GET /api/rag/jobs/{job_id}/events``
    or read the ``rag_ingestion_jobs`` table directly.

    chunking_strategy / mode default to today's behaviour. KB-scoped
    uploads should pass the KB's stored values; per-thread uploads
    default unless an override is set in chat_settings.
    """
    from utils.rag.config import resolve_embedder

    scope = _scope_for(kb_id, thread_id)
    model_name = (
        embedding_model
        or resolve_embedder(mode, chunking_strategy)
        or RAG_EMBEDDING_MODEL
    )
    job_id = str(uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO rag_ingestion_jobs
            (id, document_id, status, progress, stage)
            VALUES (?, ?, 'pending', 0.0, 'queued')
            """,
            (job_id, document_id),
        )
        conn.commit()

    state = _JobState(job_id = job_id, document_id = document_id, scope = scope)
    with _jobs_lock:
        _jobs[job_id] = state

    out_queue = _CTX.Queue()
    proc = _CTX.Process(
        target = _subprocess_worker,
        args = (
            str(stored_path),
            model_name,
            RAG_CHUNK_SIZE,
            RAG_CHUNK_OVERLAP,
            RAG_EMBED_BATCH_SIZE,
            out_queue,
            chunking_strategy,
            mode,
        ),
        daemon = True,
    )
    proc.start()
    pump_thread = threading.Thread(
        target = _pump,
        args = (state, proc, out_queue),
        name = f"rag-ingest-pump-{job_id[:8]}",
        daemon = True,
    )
    pump_thread.start()
    return job_id


def delete_document_artifacts(document_id: str, scope: str) -> None:
    """Remove a document's vectors, then rebuild BM25 for the scope.

    The caller is responsible for the sqlite cascade (deleting the
    rag_documents row triggers ON DELETE CASCADE on rag_chunks).
    """
    vector_store.delete_document(scope, document_id)
    remaining = _all_scope_chunks(scope)
    if remaining:
        bm25.rebuild_index(scope, remaining)
    else:
        bm25.delete_scope(scope)


def delete_scope_artifacts(scope: str) -> None:
    vector_store.delete_scope(scope)
    bm25.delete_scope(scope)


def purge_thread_documents(thread_ids: list[str]) -> None:
    """Remove all RAG artifacts owned by the given chat thread ids.

    Used by the chat-thread DELETE handlers because rag_documents has
    no FK cascade to chat_threads (see schema comment).
    """
    if not thread_ids:
        return
    import os
    from pathlib import Path

    from utils.paths.storage_roots import rag_uploads_root

    placeholders = ",".join("?" for _ in thread_ids)
    uploads_root = Path(os.path.realpath(rag_uploads_root()))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT stored_path FROM rag_documents WHERE thread_id IN ({placeholders})",
            thread_ids,
        ).fetchall()
        conn.execute(
            f"DELETE FROM rag_documents WHERE thread_id IN ({placeholders})",
            thread_ids,
        )
        conn.commit()
    for row in rows:
        try:
            real = Path(os.path.realpath(row["stored_path"]))
            real.relative_to(uploads_root)
        except (OSError, ValueError):
            continue
        real.unlink(missing_ok = True)
    for thread_id in thread_ids:
        delete_scope_artifacts(thread_scope(thread_id))


def purge_all_thread_documents() -> None:
    """Drop every per-thread RAG artifact. Used by clear-all-history."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM rag_documents WHERE thread_id IS NOT NULL"
        ).fetchall()
    purge_thread_documents([r["thread_id"] for r in rows])
