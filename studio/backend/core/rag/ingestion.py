# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-process threaded ingestion: parse -> chunk -> embed -> store.

``start_ingestion`` returns ``(document_id, job_id)`` immediately and runs on a
daemon thread, pushing progress onto a per-job queue (``job_events`` streams it
as SSE; ``get_job_status`` reads the persisted row). Documents are deduped by
content hash per scope; ``store.add_chunks`` is incremental, the embedder a
shared warm singleton.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import threading

from storage import rag_db

from . import captioner, chunking, config, embeddings, parsers, store

logger = logging.getLogger(__name__)

# Per-job event queues, drained by job_events; ``None`` ends the stream.
_jobs: dict[str, "queue.Queue"] = {}
_jobs_lock = threading.Lock()

# Embedding batch size; bounds peak memory.
_EMBED_BATCH = 64


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _emit(job_id: str, event: dict) -> None:
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is not None:
        q.put(event)


def _set_job(
    conn,
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE ingestion_jobs SET "
        "status=COALESCE(?, status), "
        "stage=COALESCE(?, stage), "
        "progress=COALESCE(?, progress), "
        "error=COALESCE(?, error) "
        "WHERE id=?",
        (status, stage, progress, error, job_id),
    )
    conn.commit()


def _progress(conn, job_id: str, stage: str, progress: float) -> None:
    _set_job(conn, job_id, status = "running", stage = stage, progress = progress)
    _emit(job_id, {"type": "progress", "stage": stage, "progress": progress})


def _embed_all(texts: list[str], model_name: str | None):
    """Embed texts in batches into a flat list of vectors."""
    vectors: list = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        out = embeddings.encode(batch, model_name = model_name, normalize = True)
        vectors.extend(out)
    return vectors


def _run(
    job_id: str,
    document_id: str,
    scope: str,
    stored_path: str,
    model_name: str | None,
) -> None:
    conn = rag_db.get_connection()
    try:
        _progress(conn, job_id, "parsing", 0.1)
        pages = parsers.parse(stored_path)
        if config.CAPTION_IMAGES and stored_path.lower().endswith(".pdf"):
            # Caption figures, splice into page text (no-op without vision model).
            try:
                figures = parsers.render_pdf_figures(
                    stored_path, max_figures = config.CAPTION_MAX_IMAGES
                )
            except Exception:
                logger.warning(
                    "figure rendering failed for job %s", job_id, exc_info = True
                )
                figures = []
            if figures:
                _progress(conn, job_id, "captioning", 0.2)
                captions = captioner.caption_images(figures)
                pages = captioner.splice_captions(pages, captions)

        _progress(conn, job_id, "chunking", 0.3)
        count = embeddings.token_counter(model_name)
        chunks = chunking.chunk_pages(
            pages,
            max_tokens = config.CHUNK_TOKENS,
            overlap = config.CHUNK_OVERLAP,
            count = count,
        )
        if not chunks:
            store.set_document_status(conn, document_id, "completed", num_chunks = 0)
            _set_job(conn, job_id, status = "completed", stage = "done", progress = 1.0)
            _emit(job_id, {"type": "complete", "num_chunks": 0})
            return

        _progress(conn, job_id, "embedding", 0.5)
        vectors = _embed_all([c.text for c in chunks], model_name)

        # Locate each chunk's highlight regions (non-PDFs/failures yield none).
        regions = None
        if stored_path.lower().endswith(".pdf"):
            try:
                from . import locators

                regions = locators.pdf_regions_for_chunks(stored_path, pages, chunks)
            except Exception:
                logger.warning(
                    "pdf region location failed for job %s", job_id, exc_info = True
                )
                regions = None

        _progress(conn, job_id, "storing", 0.9)
        store.add_chunks(conn, scope, document_id, chunks, vectors, regions)
        store.set_document_status(
            conn, document_id, "completed", num_chunks = len(chunks)
        )

        _set_job(conn, job_id, status = "completed", stage = "done", progress = 1.0)
        _emit(job_id, {"type": "complete", "num_chunks": len(chunks)})
    except Exception as exc:  # noqa: BLE001 - report any failure to the client
        logger.exception("ingestion job %s failed", job_id)
        try:
            store.set_document_status(conn, document_id, "failed", error = str(exc))
            _set_job(conn, job_id, status = "failed", stage = "error", error = str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("failed to record ingestion failure for job %s", job_id)
        _emit(job_id, {"type": "error", "stage": "error", "error": str(exc)})
    finally:
        conn.close()
        _emit(job_id, None)


def start_ingestion(
    scope: str,
    kb_id: str | None,
    thread_id: str | None,
    filename: str,
    stored_path: str,
    *,
    model_name: str | None = None,
) -> tuple[str, str]:
    """Create the document + job rows and spawn the worker, returning
    ``(document_id, job_id)``. A duplicate content hash in this scope returns the
    existing id with an already-completed job (no re-ingest)."""
    ext = os.path.splitext(stored_path)[1].lower()
    if ext not in config.UPLOAD_EXTS:
        raise ValueError(f"unsupported file type: {ext}")

    sha = _sha256_file(stored_path)
    conn = rag_db.get_connection()
    try:
        existing = store.document_by_hash(conn, scope, sha)
        if existing is not None:
            job_id = _new_job(conn, existing, scope, status = "completed", progress = 1.0)
            with _jobs_lock:
                _jobs[job_id] = queue.Queue()
            _emit(job_id, {"type": "complete", "num_chunks": 0, "deduped": True})
            _emit(job_id, None)
            return existing, job_id

        document_id = store.create_document(
            conn,
            scope = scope,
            filename = filename,
            sha256 = sha,
            kb_id = kb_id,
            thread_id = thread_id,
            status = "pending",
            stored_path = stored_path,
        )
        job_id = _new_job(conn, document_id, scope)
    finally:
        conn.close()

    with _jobs_lock:
        _jobs[job_id] = queue.Queue()
    threading.Thread(
        target = _run,
        args = (job_id, document_id, scope, stored_path, model_name),
        daemon = True,
    ).start()
    return document_id, job_id


def _new_job(
    conn,
    document_id: str,
    scope: str,
    *,
    status: str = "pending",
    progress: float = 0.0,
) -> str:
    import uuid
    from datetime import datetime, timezone

    job_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO ingestion_jobs(id, document_id, scope, status, stage, progress, created_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (
            job_id,
            document_id,
            scope,
            status,
            None,
            progress,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return job_id


def job_events(job_id: str):
    """Yield job events for SSE; ends when the worker signals completion."""
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is None:
        return
    while True:
        event = q.get()
        if event is None:
            break
        yield event
    with _jobs_lock:
        _jobs.pop(job_id, None)


def get_job_status(job_id: str) -> dict | None:
    """Read the persisted ingestion job row (status / stage / progress / error)."""
    conn = rag_db.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
