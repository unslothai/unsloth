# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-process threaded ingestion: parse -> chunk -> embed -> store.
``start_ingestion`` returns ``(document_id, job_id)`` immediately and runs on a
daemon thread, pushing progress onto a per-job queue (streamed as SSE by
``job_events``). Documents are deduped by content hash per scope."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import re
import threading

from storage import rag_db

from . import captioner, chunking, config, embeddings, job_leases, parsers, store

logger = logging.getLogger(__name__)

# Per-job event queues, drained by job_events; None ends the stream.
_jobs: dict[str, "queue.Queue"] = {}
_workers: dict[str, threading.Thread] = {}
_jobs_lock = threading.Lock()

_EMBED_BATCH = 64
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

# Poll with a timeout so the generator notices a gone client or a worker that died without the None sentinel.
_SSE_POLL_SECONDS = 1.0
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _remove_upload(stored_path: str | None, *, keep_path: str | None = None) -> None:
    if not stored_path:
        return
    try:
        target = os.path.realpath(stored_path)
        if keep_path is not None and target == os.path.realpath(keep_path):
            return
        from utils.paths import rag_uploads_root

        uploads = os.path.realpath(str(rag_uploads_root()))
        if os.path.isfile(target) and os.path.commonpath([uploads, target]) == uploads:
            os.remove(target)
    except Exception:  # noqa: BLE001 - upload cleanup must not block ingestion.
        logger.warning("failed to remove RAG upload %s", stored_path, exc_info = True)


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
    if not job_leases.renew_owned(conn, job_leases.INGESTION, job_id):
        raise job_leases.JobLeaseLost("Ingestion job lease was reclaimed")
    _set_job(conn, job_id, status = "running", stage = stage, progress = progress)
    _emit(job_id, {"type": "progress", "stage": stage, "progress": progress})


def _abort_if_document_deleted(conn, job_id: str, document_id: str) -> bool:
    """Retire the job when a project delete or a discarded upload removed its document.

    Opens the write transaction the caller then commits into, so a delete cannot land between
    the check and the write. Chunks carry no foreign key to the document, so writing after one
    would strand rows under a dead scope, and completing would report a deleted document as
    indexed and retire the document it was replacing.
    """
    conn.execute("BEGIN IMMEDIATE")
    if store.get_document(conn, document_id) is not None:
        return False
    conn.rollback()
    _set_job(conn, job_id, status = "cancelled", stage = "done", progress = 1.0)
    _emit(job_id, {"type": "error", "stage": "cancelled", "error": "Document was deleted"})
    return True


def _embed_pass(texts: list[str], model_name: str | None):
    """One batched pass. Returns ``(vectors, identity, changed)``, ``changed`` when the
    embedder swapped part way and the vectors therefore span two spaces."""
    vectors: list = []
    identity: str | None = None
    changed = False
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        out, batch_identity = embeddings.encode_with_identity(
            batch, model_name = model_name, normalize = True
        )
        changed = changed or (identity is not None and batch_identity != identity)
        identity = batch_identity
        vectors.extend(out)
    return vectors, identity or embeddings.embedding_identity(model_name), changed


def _embed_all(texts: list[str], model_name: str | None):
    """Embed texts in batches. Returns ``(vectors, identity)`` of the embedder that
    produced them. An ST encode failure swaps the process to llama-server, and a swap
    between batches would leave one document holding vectors from two spaces, so the
    document restarts under the backend that took over. That swap is one-way, so the
    second pass is uniform."""
    for _ in range(2):
        vectors, identity, changed = _embed_pass(texts, model_name)
        if not changed:
            return vectors, identity
        logger.warning("embedder changed mid-document; re-embedding under the new one")
    return vectors, identity


def _ocr_scanned_pages(
    pages: list,
    stored_path: str,
    conn,
    job_id: str,
    ocr: bool | None = None,
) -> tuple[list, set[int]]:
    """Replace text on near-empty (scanned/image-only) PDF pages with vision-model OCR
    so image PDFs become searchable. ``ocr`` overrides ``config.OCR_SCANNED`` per upload
    (``None`` = config default); no-op without scanned pages or a vision model. OCR'd
    pages have no text layer, so no preview highlight regions, but stay searchable.
    Returns ``(pages, ocred)``: new ``Page`` objects for OCR'd pages (originals
    otherwise) and the set of page numbers actually transcribed."""
    if not (config.OCR_SCANNED if ocr is None else ocr):
        return pages, set()
    scanned = [
        p.page_number
        for p in pages
        if p.page_number is not None and len((p.text or "").strip()) < config.OCR_MIN_CHARS
    ]
    if not scanned or captioner.vision_endpoint() is None:
        return pages, set()
    if len(scanned) > config.OCR_MAX_PAGES:
        logger.warning(
            "OCR: %d scanned pages exceed OCR_MAX_PAGES=%d; pages past the cap stay "
            "untranscribed (raise RAG_OCR_MAX_PAGES to cover them)",
            len(scanned),
            config.OCR_MAX_PAGES,
        )
    scanned = scanned[: config.OCR_MAX_PAGES]
    _progress(conn, job_id, "ocr", 0.25)
    page_pngs = parsers.render_pdf_pages(stored_path, scanned, dpi = config.OCR_DPI)
    texts = captioner.ocr_pages(page_pngs)
    if not texts:
        return pages, set()

    from .parsers import Page

    out: list = []
    ocred: set[int] = set()
    for page in pages:
        text = texts.get(page.page_number)
        if text:
            original = (page.text or "").strip()
            merged = text if not original or original in text else f"{original}\n\n{text}"
            out.append(Page(text = merged, page_number = page.page_number, char_count = len(merged)))
            ocred.add(page.page_number)
        else:
            out.append(page)
    return out, ocred


def _replace_old_document(
    conn, replaces: tuple[str, str | None] | None, keep_path: str, document_id: str
) -> None:
    """Drop the document this ingestion replaced (stale embedder / empty prior
    ingest), called only after the replacement completed successfully.

    Checked against the replacement inside the transaction that retires the old row: every
    store helper commits, so a delete that removed the replacement between the completion and
    this call would otherwise take the still-searchable document it was replacing with it.
    """
    if replaces is None:
        return
    old_id, old_path = replaces
    try:
        conn.execute("BEGIN IMMEDIATE")
        if store.get_document(conn, document_id) is None:
            conn.rollback()
            return
        store.delete_document(conn, old_id)
        _remove_upload(old_path, keep_path = keep_path)
    except Exception:  # noqa: BLE001 - the new document is already live
        logger.warning("failed to remove replaced document %s", old_id, exc_info = True)


def _run(
    job_id: str,
    document_id: str,
    scope: str,
    stored_path: str,
    model_name: str | None,
    ocr: bool | None = None,
    caption: bool | None = None,
    replaces: tuple[str, str | None] | None = None,
) -> None:
    conn = None
    try:
        conn = rag_db.get_connection()
        _progress(conn, job_id, "parsing", 0.1)
        pages = parsers.parse(stored_path)
        is_pdf = stored_path.lower().endswith(".pdf")
        ocred: set[int] = set()
        if is_pdf:
            pages, ocred = _ocr_scanned_pages(pages, stored_path, conn, job_id, ocr = ocr)
        caption_on = config.CAPTION_IMAGES if caption is None else caption
        # Skip all figure work (PDF rasterization included) without a vision model.
        if caption_on and is_pdf and captioner.vision_endpoint() is not None:
            # Tile figure pages, transcribe+describe each tile, then merge/dedup/splice into the page text so
            # small labels and every sub-figure are captured.
            try:
                fig_pages = parsers.pages_with_figures(
                    stored_path,
                    max_pages = config.CAPTION_MAX_PAGES,
                    # Skip only pages OCR actually transcribed; a scanned figure page past the OCR cap still tiles.
                    exclude_pages = ocred,
                )
                tiles = (
                    parsers.render_pdf_figure_tiles(
                        stored_path,
                        fig_pages,
                        dpi = config.FIGURE_DPI,
                        rows = config.FIGURE_TILE_ROWS,
                        cols = config.FIGURE_TILE_COLS,
                        overlap = config.FIGURE_TILE_OVERLAP,
                        fullpage = config.FIGURE_FULLPAGE,
                        max_tiles = config.CAPTION_MAX_IMAGES,
                    )
                    if fig_pages
                    else []
                )
            except Exception:
                logger.warning("figure tiling failed for job %s", job_id, exc_info = True)
                tiles = []
            if tiles:
                _progress(conn, job_id, "captioning", 0.28)
                captions = captioner.merge_page_captions(captioner.caption_images(tiles))
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
            # An empty parse still completes the document and retires the one it replaces, so it needs the same
            # guard as the chunk write.
            if _abort_if_document_deleted(conn, job_id, document_id):
                return
            # inside the write transaction the guard opened, as _progress does
            if not job_leases.renew_owned(conn, job_leases.INGESTION, job_id):
                raise job_leases.JobLeaseLost("Ingestion job lease was reclaimed")
            store.set_document_status(conn, document_id, "completed", num_chunks = 0)
            _replace_old_document(conn, replaces, stored_path, document_id)
            _set_job(conn, job_id, status = "completed", stage = "done", progress = 1.0)
            _emit(job_id, {"type": "complete", "num_chunks": 0})
            return

        _progress(conn, job_id, "embedding", 0.5)
        # An ST encode failure swaps the process to llama-server, so the embedder that produced these
        # vectors is only known once they exist.
        vectors, identity = _embed_all([c.text for c in chunks], model_name)
        store.set_document_embedding_model(conn, document_id, identity)

        # Locate each chunk's highlight regions (non-PDFs/failures yield none).
        regions = None
        if stored_path.lower().endswith(".pdf"):
            try:
                from . import locators
                regions = locators.pdf_regions_for_chunks(stored_path, pages, chunks)
            except Exception:
                logger.warning("pdf region location failed for job %s", job_id, exc_info = True)
                regions = None

        _progress(conn, job_id, "storing", 0.9)
        if _abort_if_document_deleted(conn, job_id, document_id):
            return
        store.add_chunks(conn, scope, document_id, chunks, vectors, regions)
        # add_chunks commits and releases the lock, so retake it: a delete landing in that gap must not be
        # recorded as a completed ingestion.
        if _abort_if_document_deleted(conn, job_id, document_id):
            return
        store.set_document_status(conn, document_id, "completed", num_chunks = len(chunks))
        _replace_old_document(conn, replaces, stored_path, document_id)

        _set_job(conn, job_id, status = "completed", stage = "done", progress = 1.0)
        _emit(job_id, {"type": "complete", "num_chunks": len(chunks)})
    except job_leases.JobLeaseLost:
        logger.info("ingestion job %s stopped after its lease was reclaimed", job_id)
    except Exception as exc:  # noqa: BLE001 - report any failure to the client
        logger.exception("ingestion job %s failed", job_id)
        try:
            if conn is None:
                conn = rag_db.get_connection()
            store.set_document_status(conn, document_id, "failed", error = str(exc))
            _set_job(conn, job_id, status = "failed", stage = "error", error = str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("failed to record ingestion failure for job %s", job_id)
        _emit(job_id, {"type": "error", "stage": "error", "error": str(exc)})
    finally:
        if conn is not None:
            conn.close()
        job_leases.release(job_leases.INGESTION, job_id)
        with _jobs_lock:
            _workers.pop(job_id, None)
        _emit(job_id, None)


def start_ingestion(
    scope: str,
    kb_id: str | None,
    thread_id: str | None,
    filename: str,
    stored_path: str,
    *,
    project_id: str | None = None,
    model_name: str | None = None,
    ocr: bool | None = None,
    caption: bool | None = None,
    dedupe: bool = True,
    linked_folder_id: str | None = None,
    linked_relative_path: str | None = None,
    background: bool = True,
    content_hash: str | None = None,
) -> tuple[str, str]:
    """Create the document + job rows and spawn the worker, returning
    ``(document_id, job_id)``. A duplicate content hash in this scope returns the
    existing id with an already-completed job (no re-ingest).

    ``content_hash`` lets a caller that already hashed ``stored_path`` (linked-folder
    reconciliation hashes it to detect content-identical renames) pass that digest
    through instead of paying for a second full read of the file. Must be the lowercase
    hex sha256 of ``stored_path``; a mismatched value would misfile the document under
    the wrong hash, so it is trusted as given and never reverified here."""
    ext = os.path.splitext(stored_path)[1].lower()
    if ext not in config.UPLOAD_EXTS:
        raise ValueError(f"unsupported file type: {ext}")

    # Reclaim queues for finished jobs so the registry stays bounded.
    _reap_finished_jobs()

    if content_hash is not None and not _SHA256_HEX_RE.match(content_hash):
        raise ValueError("content_hash must be a lowercase hex sha256 digest")
    sha = content_hash or _sha256_file(stored_path)
    conn = rag_db.get_connection()
    try:
        # Name the embedder before BEGIN IMMEDIATE: it runs nvidia-smi and may import torch, and holding a
        # RESERVED lock that long fails concurrent writers with "database is locked".
        effective_model = model_name or config.effective_embedding_model()
        effective_identity = embeddings.embedding_identity(effective_model)
        # The job lease is committed in the same transaction as the document, so cleanup never observes an
        # unowned in-flight document.
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute(
            "SELECT 1 FROM linked_folder_retired_scopes WHERE scope=?", (scope,)
        ).fetchone():
            conn.rollback()
            raise RuntimeError("Owning scope is being deleted")
        # (old_document_id, old_stored_path) replaced by this upload; deleted by the worker only after the
        # replacement completes, so a failed re-index never destroys the still-searchable original.
        replaces: tuple[str, str | None] | None = None
        existing = store.document_by_hash(conn, scope, sha) if dedupe else None
        if existing is not None:
            doc = store.get_document(conn, existing)
            empty_completed = (
                doc is not None and doc.get("status") == "completed" and not doc.get("num_chunks")
            )
            # Vectors from a different embedder are stale, so re-uploading must re-index; NULL (legacy rows)
            # is assumed current. Only completed rows are replaceable, since a running duplicate's writes must
            # not land on a deleted document.
            stale_model = (
                doc is not None
                and doc.get("status") == "completed"
                and not config.embedding_identity_matches(
                    doc.get("embedding_model"), effective_identity
                )
            )
            if empty_completed or stale_model:
                # Zero chunks previously, or a different embedder: re-ingest rather than dedupe.
                replaces = (existing, doc.get("stored_path"))
            else:
                job_id = _new_job(conn, existing, scope, status = "completed", progress = 1.0)
                _remove_upload(stored_path)
                with _jobs_lock:
                    _jobs[job_id] = queue.Queue()
                _emit(
                    job_id,
                    {"type": "complete", "num_chunks": doc.get("num_chunks") or 0, "deduped": True},
                )
                _emit(job_id, None)
                return existing, job_id
        if dedupe:
            for failed in store.failed_documents_by_hash(conn, scope, sha):
                store.delete_document(conn, failed["id"], commit = False)
                _remove_upload(failed.get("stored_path"), keep_path = stored_path)

        document_id = store.create_document(
            conn,
            scope = scope,
            filename = filename,
            sha256 = sha,
            kb_id = kb_id,
            thread_id = thread_id,
            project_id = project_id,
            status = "pending",
            stored_path = stored_path,
            embedding_model = effective_identity,
            linked_folder_id = linked_folder_id,
            linked_relative_path = linked_relative_path,
            commit = False,
        )
        job_id = _new_job(conn, document_id, scope)
    finally:
        conn.close()

    try:
        job_leases.activate(job_leases.INGESTION, job_id)
        with _jobs_lock:
            _jobs[job_id] = queue.Queue()
        args = (
            job_id,
            document_id,
            scope,
            stored_path,
            effective_model,
            ocr,
            caption,
            replaces,
        )
        if not background:
            _run(*args)
            return document_id, job_id
        worker = threading.Thread(
            target = _run,
            # effective_model, not the raw model_name, pins the embedder for the whole job: a Settings change
            # mid-ingestion must not switch tokenizer or embedder between batches.
            args = args,
            daemon = True,
        )
        with _jobs_lock:
            _workers[job_id] = worker
        worker.start()
    except Exception:
        with _jobs_lock:
            _workers.pop(job_id, None)
        job_leases.release(job_leases.INGESTION, job_id)
        fail_stalled_job(job_id, "Ingestion worker could not start")
        raise
    return document_id, job_id


def job_worker_alive(job_id: str) -> bool:
    """Return whether this process still has a live worker for a persisted job."""
    with _jobs_lock:
        worker = _workers.get(job_id)
    return worker is not None and worker.is_alive()


def fail_stalled_job(job_id: str, error: str) -> bool:
    """Fail a nonterminal job only after its in-process worker has exited."""
    if job_worker_alive(job_id):
        return False
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT document_id, status FROM ingestion_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None or row["status"] in _TERMINAL_JOB_STATUSES:
            conn.rollback()
            return False
        conn.execute(
            "UPDATE documents SET status='failed', error=? "
            "WHERE id=? AND status IN ('pending','running')",
            (error, row["document_id"]),
        )
        conn.execute(
            "UPDATE ingestion_jobs SET status='failed', stage='error', error=? WHERE id=?",
            (error, job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    _emit(job_id, {"type": "error", "stage": "error", "error": error})
    _emit(job_id, None)
    return True


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
    if status not in _TERMINAL_JOB_STATUSES:
        if not job_leases.claim(conn, job_leases.INGESTION, job_id):
            raise RuntimeError("Could not claim ingestion job")
    conn.commit()
    return job_id


def _reap_finished_jobs() -> None:
    """Drop per-job queues whose DB row already reached a terminal status.

    Otherwise removed only by ``job_events`` after the ``None`` sentinel, so a
    caller that polls ``/jobs/{id}`` instead of streaming would grow ``_jobs``
    forever. Safe while streaming: ``job_events`` holds its queue reference.
    """
    with _jobs_lock:
        job_ids = list(_jobs.keys())
    for jid in job_ids:
        row = get_job_status(jid)
        if row is not None and row.get("status") in _TERMINAL_JOB_STATUSES:
            with _jobs_lock:
                _jobs.pop(jid, None)


def delete_terminal_job(job_id: str) -> bool:
    """Remove a consumed internal job without racing an active ingestion worker."""
    conn = rag_db.get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM ingestion_jobs WHERE id=? AND status IN ('completed','failed')",
            (job_id,),
        )
        conn.commit()
    finally:
        conn.close()
    if cursor.rowcount:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        return True
    return False


def job_events(job_id: str):
    """Yield job events for SSE; ends when the worker signals completion.

    Timed ``get`` so the generator can't block forever: it wakes to heartbeat,
    to notice a disconnected client, and to stop on a terminal DB status (a hard
    worker death that skipped the ``None`` sentinel). Drops the queue only on a
    terminal exit, never on an early client disconnect.

    It deliberately does *not* end on idle alone: a long silent stage (e.g.
    embedding a large doc) is not a failure, and ending there would send
    ``[DONE]`` with the row still pending, which the client treats as completion.
    The stream ends only on a terminal status, the ``None`` sentinel, or disconnect.
    """
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is None:
        return
    terminal = False
    try:
        while True:
            try:
                event = q.get(timeout = _SSE_POLL_SECONDS)
            except queue.Empty:
                try:
                    row = get_job_status(job_id)
                except Exception:  # noqa: BLE001
                    # A transient status read must not abort the stream: routes/rag.py would turn it into a terminal
                    # error frame and the UI would drop a document whose worker is still running.
                    logger.warning(
                        "job_events status read failed for %s; continuing", job_id, exc_info = True
                    )
                    yield {"type": "heartbeat"}
                    continue
                if row is None or row.get("status") in _TERMINAL_JOB_STATUSES:
                    # Worker finished (or row gone); stop and let the client reconcile via getJob.
                    terminal = True
                    break
                yield {"type": "heartbeat"}
                continue
            if event is None:
                terminal = True
                break
            yield event
    finally:
        # Keep the queue while the worker runs so an early disconnect can reconnect and resume; terminal is
        # still False at disconnect, since _run writes the DB status before emitting it.
        if not terminal:
            try:
                row = get_job_status(job_id)
                terminal = row is None or row.get("status") in _TERMINAL_JOB_STATUSES
            except Exception:  # noqa: BLE001
                # Cannot confirm terminality, so keep the queue rather than orphan a live worker's events.
                terminal = False
        if terminal:
            with _jobs_lock:
                _jobs.pop(job_id, None)


def get_job_status(job_id: str) -> dict | None:
    """Read the persisted ingestion job row (status / stage / progress / error), plus
    the document's ``num_chunks`` so a client polling to completion learns the chunk
    count (the SSE ``complete`` frame carries it, but the poll/reconcile path does not)."""
    conn = rag_db.get_connection()
    try:
        row = conn.execute(
            "SELECT j.*, d.num_chunks AS num_chunks FROM ingestion_jobs j "
            "LEFT JOIN documents d ON d.id = j.document_id WHERE j.id=?",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
