# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG API routes.

Surface:
  - Knowledge-base CRUD
  - Document upload (KB-scoped and per-thread)
  - Document list/delete
  - Ingestion-job SSE stream
  - Search (BM25 / dense / hybrid)

Per-thread document uploads are scoped to a single chat thread and
share the same chunk/embed/index pipeline as KB documents — they only
differ in the scope key (``thread_<id>`` vs ``kb_<id>``) and lifecycle
(per-thread docs are dropped when the thread is deleted, via the
ON DELETE CASCADE on rag_documents.thread_id).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue as queue_module
import time
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject, get_current_subject_sse


async def _sse_auth(
    token: str | None = Query(None),
    authorization: str | None = Header(None),
) -> str:
    return await get_current_subject_sse(token, authorization)
from core.rag import embeddings, ingestion, reranker, retrieval, vector_store
from core.rag.vector_store import kb_scope, thread_scope
from loggers import get_logger
from storage.studio_db import get_connection
from utils.paths.storage_roots import ensure_dir, rag_uploads_root
from utils.rag.config import (
    RAG_MAX_UPLOAD_MB,
    RAG_RERANK_CANDIDATE_K,
    RAG_UPLOAD_EXTS,
)

router = APIRouter()
logger = get_logger(__name__)


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------

ChunkingStrategy = Literal["standard", "late"]
KBMode = Literal["text", "multimodal"]


class CreateKBRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 200)
    description: str | None = None
    embedding_model: str | None = None
    # Phase 3 introduces two per-KB knobs. Both default to today's
    # behaviour so existing API clients are unaffected. The (multimodal,
    # late) combination is rejected at create time — see _validate_mode_combo.
    chunking_strategy: ChunkingStrategy = "standard"
    mode: KBMode = "text"


class KBResponse(BaseModel):
    id: str
    name: str
    description: str | None
    embedding_model: str
    chunking_strategy: ChunkingStrategy
    mode: KBMode
    created_at: int


class KBListResponse(BaseModel):
    knowledge_bases: list[KBResponse]


class DocumentResponse(BaseModel):
    id: str
    kb_id: str | None
    thread_id: str | None
    filename: str
    content_type: str | None
    status: str
    num_chunks: int
    byte_size: int
    error: str | None
    created_at: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class ThreadIndexSummary(BaseModel):
    thread_id: str
    title: str | None
    num_documents: int
    num_chunks: int


class ThreadIndexListResponse(BaseModel):
    threads: list[ThreadIndexSummary]


class UploadResponse(BaseModel):
    document_id: str
    job_id: str
    filename: str


class SearchRequest(BaseModel):
    query: str = Field(min_length = 1, max_length = 4000)
    kb_id: str | None = None
    thread_id: str | None = None
    top_k: int = Field(default = 10, ge = 1, le = 100)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    document_ids: list[str] | None = None
    enable_rerank: bool = False
    reranker_model: str | None = None


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
    page_number: int | None = None
    filename: str | None = None
    kind: str = "text"
    image_url: str | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    return name or "document"


def _now_ms() -> int:
    return int(time.time())


def _row_to_kb(row: Any) -> KBResponse:
    # chunking_strategy / mode may be absent on rows fetched through a
    # pre-Phase-3 connection in tests; fall back to the schema defaults.
    keys = row.keys() if hasattr(row, "keys") else ()
    chunking_strategy = (
        row["chunking_strategy"]
        if "chunking_strategy" in keys
        else "standard"
    )
    mode = row["mode"] if "mode" in keys else "text"
    return KBResponse(
        id = row["id"],
        name = row["name"],
        description = row["description"],
        embedding_model = row["embedding_model"],
        chunking_strategy = chunking_strategy,
        mode = mode,
        created_at = row["created_at"],
    )


def _validate_mode_combo(mode: KBMode, chunking_strategy: ChunkingStrategy) -> None:
    """Reject the one illegal (mode, strategy) combination.

    No public open-weight embedder supports both late-chunking pooling
    and shared text/image embedding. Surface the constraint as a 400
    rather than failing silently during ingestion.
    """
    if mode == "multimodal" and chunking_strategy == "late":
        raise HTTPException(
            status_code = 400,
            detail = (
                "Late chunking is not supported in multimodal mode — "
                "the multimodal embedder does not expose per-token "
                "embeddings. Pick 'standard' chunking or 'text' mode."
            ),
        )


def _row_to_document(row: Any) -> DocumentResponse:
    return DocumentResponse(
        id = row["id"],
        kb_id = row["kb_id"],
        thread_id = row["thread_id"],
        filename = row["filename"],
        content_type = row["content_type"],
        status = row["status"],
        num_chunks = row["num_chunks"],
        byte_size = row["byte_size"],
        error = row["error"],
        created_at = row["created_at"],
    )


def _kb_or_404(kb_id: str) -> Any:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM rag_knowledge_bases WHERE id = ?",
            (kb_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code = 404, detail = "Knowledge base not found")
    return row


def _thread_or_404(thread_id: str) -> None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM chat_threads WHERE id = ?",
            (thread_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code = 404, detail = "Thread not found")


def _document_or_404(document_id: str) -> Any:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM rag_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code = 404, detail = "Document not found")
    return row


async def _save_upload(file: UploadFile) -> tuple[Path, str, int]:
    filename = _sanitize_filename(file.filename or "document")
    ext = Path(filename).suffix.lower()
    if ext not in RAG_UPLOAD_EXTS:
        allowed = ", ".join(sorted(RAG_UPLOAD_EXTS))
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported file type: {ext}. Allowed: {allowed}",
        )
    upload_dir = ensure_dir(rag_uploads_root())
    stored_name = f"{uuid4().hex}_{Path(filename).stem}{ext}"
    stored_path = upload_dir / stored_name
    max_bytes = RAG_MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with open(stored_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                f.close()
                stored_path.unlink(missing_ok = True)
                raise HTTPException(
                    status_code = 413,
                    detail = f"File exceeds {RAG_MAX_UPLOAD_MB} MB limit",
                )
            f.write(chunk)
    if written == 0:
        stored_path.unlink(missing_ok = True)
        raise HTTPException(status_code = 400, detail = "Empty upload payload")
    return stored_path, filename, written


def _start_ingestion(
    *,
    filename: str,
    stored_path: Path,
    byte_size: int,
    content_type: str | None,
    kb_id: str | None,
    thread_id: str | None,
    embedding_model: str,
    chunking_strategy: str = "standard",
    mode: str = "text",
) -> UploadResponse:
    document_id = str(uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO rag_documents
            (id, kb_id, thread_id, filename, content_type, stored_path,
             status, num_chunks, byte_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
            """,
            (
                document_id,
                kb_id,
                thread_id,
                filename,
                content_type,
                str(stored_path),
                byte_size,
                _now_ms(),
            ),
        )
        conn.commit()
    job_id = ingestion.enqueue_ingestion(
        document_id = document_id,
        stored_path = stored_path,
        kb_id = kb_id,
        thread_id = thread_id,
        embedding_model = embedding_model,
        chunking_strategy = chunking_strategy,
        mode = mode,
    )
    return UploadResponse(document_id = document_id, job_id = job_id, filename = filename)


def _unlink_if_under_uploads(path: Path) -> None:
    try:
        real = Path(os.path.realpath(path))
        root = Path(os.path.realpath(rag_uploads_root()))
        real.relative_to(root)
    except (OSError, ValueError):
        return
    real.unlink(missing_ok = True)


# ------------------------------------------------------------------
# Knowledge bases
# ------------------------------------------------------------------

@router.post("/knowledge-bases", response_model = KBResponse)
def create_knowledge_base(
    payload: CreateKBRequest,
    current_subject: str = Depends(get_current_subject),
) -> KBResponse:
    from utils.rag.config import resolve_embedder

    _validate_mode_combo(payload.mode, payload.chunking_strategy)

    kb_id = str(uuid4())
    # If the caller didn't override embedding_model, resolve from the
    # Phase-3 matrix using their (mode, strategy) selection. Unknown
    # combos fall back to the legacy default — see resolve_embedder.
    embedding_model = (
        payload.embedding_model
        or resolve_embedder(payload.mode, payload.chunking_strategy)
    )
    created_at = _now_ms()
    with get_connection() as conn:
        try:
            conn.execute(
                """
                INSERT INTO rag_knowledge_bases
                (id, name, description, owner_user_id, embedding_model,
                 chunking_strategy, mode, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kb_id,
                    payload.name,
                    payload.description,
                    current_subject,
                    embedding_model,
                    payload.chunking_strategy,
                    payload.mode,
                    created_at,
                ),
            )
            conn.commit()
        except Exception as exc:
            raise HTTPException(
                status_code = 409,
                detail = f"Could not create KB: {exc}",
            ) from exc
    return KBResponse(
        id = kb_id,
        name = payload.name,
        description = payload.description,
        embedding_model = embedding_model,
        chunking_strategy = payload.chunking_strategy,
        mode = payload.mode,
        created_at = created_at,
    )


@router.get("/knowledge-bases", response_model = KBListResponse)
def list_knowledge_bases(
    current_subject: str = Depends(get_current_subject),
) -> KBListResponse:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM rag_knowledge_bases ORDER BY created_at DESC"
        ).fetchall()
    return KBListResponse(knowledge_bases = [_row_to_kb(r) for r in rows])


class ReingestKBRequest(BaseModel):
    """All fields optional — omitting one keeps the KB's current value."""
    chunking_strategy: ChunkingStrategy | None = None
    mode: KBMode | None = None
    embedding_model: str | None = None


class ReingestResponse(BaseModel):
    job_ids: list[str]
    document_ids: list[str]


def _reingest_scope(
    *,
    kb_id: str | None,
    thread_id: str | None,
    chunking_strategy: str,
    mode: str,
    embedding_model: str,
) -> ReingestResponse:
    """Wipe scope artifacts and re-enqueue every stored document.

    The chat_settings / per-thread defaults aren't touched — caller is
    responsible for updating any associated metadata before calling.
    """
    scope = kb_scope(kb_id) if kb_id else thread_scope(thread_id)  # type: ignore[arg-type]
    with get_connection() as conn:
        if kb_id:
            rows = conn.execute(
                "SELECT id, stored_path FROM rag_documents WHERE kb_id = ?",
                (kb_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, stored_path FROM rag_documents WHERE thread_id = ?",
                (thread_id,),
            ).fetchall()
        # Delete the rag_documents rows (cascade drops chunks); the
        # uploaded file on disk is preserved so we can re-ingest from
        # it. We re-INSERT a fresh row per stored_path below.
        doc_ids = [r["id"] for r in rows]
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            conn.execute(
                f"DELETE FROM rag_documents WHERE id IN ({placeholders})",
                doc_ids,
            )
            conn.commit()
    ingestion.delete_scope_artifacts(scope)

    job_ids: list[str] = []
    new_doc_ids: list[str] = []
    for row in rows:
        stored_path = Path(row["stored_path"])
        if not stored_path.is_file():
            continue
        filename = stored_path.name
        # Strip the UUID prefix we attached at upload time so the
        # re-inserted document carries the original name.
        if "_" in filename:
            _uuid_prefix, _, original = filename.partition("_")
            if original:
                filename = original
        upload = _start_ingestion(
            filename = filename,
            stored_path = stored_path,
            byte_size = stored_path.stat().st_size,
            content_type = None,
            kb_id = kb_id,
            thread_id = thread_id,
            embedding_model = embedding_model,
            chunking_strategy = chunking_strategy,
            mode = mode,
        )
        job_ids.append(upload.job_id)
        new_doc_ids.append(upload.document_id)
    return ReingestResponse(job_ids = job_ids, document_ids = new_doc_ids)


@router.post(
    "/knowledge-bases/{kb_id}/reingest",
    response_model = ReingestResponse,
)
def reingest_knowledge_base(
    kb_id: str,
    payload: ReingestKBRequest,
    current_subject: str = Depends(get_current_subject),
) -> ReingestResponse:
    from utils.rag.config import resolve_embedder

    kb_row = _kb_or_404(kb_id)
    keys = kb_row.keys() if hasattr(kb_row, "keys") else ()
    current_strategy = (
        kb_row["chunking_strategy"]
        if "chunking_strategy" in keys
        else "standard"
    )
    current_mode = kb_row["mode"] if "mode" in keys else "text"
    current_embedder = kb_row["embedding_model"]

    new_strategy = payload.chunking_strategy or current_strategy
    new_mode = payload.mode or current_mode
    _validate_mode_combo(new_mode, new_strategy)

    new_embedder = (
        payload.embedding_model
        or (
            current_embedder
            if (new_strategy == current_strategy and new_mode == current_mode)
            else resolve_embedder(new_mode, new_strategy)
        )
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE rag_knowledge_bases
            SET chunking_strategy = ?, mode = ?, embedding_model = ?
            WHERE id = ?
            """,
            (new_strategy, new_mode, new_embedder, kb_id),
        )
        conn.commit()

    return _reingest_scope(
        kb_id = kb_id,
        thread_id = None,
        chunking_strategy = new_strategy,
        mode = new_mode,
        embedding_model = new_embedder,
    )


@router.post(
    "/threads/{thread_id}/reingest",
    response_model = ReingestResponse,
)
def reingest_thread_documents(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
) -> ReingestResponse:
    """Rebuild a thread's RAG index using the current defaults.

    No body — per-thread strategy/mode overrides aren't exposed in v1.
    """
    from utils.rag.config import RAG_EMBEDDING_MODEL

    return _reingest_scope(
        kb_id = None,
        thread_id = thread_id,
        chunking_strategy = "standard",
        mode = "text",
        embedding_model = RAG_EMBEDDING_MODEL,
    )


@router.delete("/knowledge-bases/{kb_id}")
def delete_knowledge_base(
    kb_id: str,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    _kb_or_404(kb_id)
    with get_connection() as conn:
        doc_rows = conn.execute(
            "SELECT stored_path FROM rag_documents WHERE kb_id = ?",
            (kb_id,),
        ).fetchall()
        conn.execute("DELETE FROM rag_knowledge_bases WHERE id = ?", (kb_id,))
        conn.commit()
    for row in doc_rows:
        _unlink_if_under_uploads(Path(row["stored_path"]))
    ingestion.delete_scope_artifacts(kb_scope(kb_id))
    return {"ok": True}


# ------------------------------------------------------------------
# Document upload (KB and per-thread)
# ------------------------------------------------------------------

@router.post("/knowledge-bases/{kb_id}/documents", response_model = UploadResponse)
async def upload_kb_document(
    kb_id: str,
    file: UploadFile,
    current_subject: str = Depends(get_current_subject),
) -> UploadResponse:
    kb_row = _kb_or_404(kb_id)
    stored_path, filename, byte_size = await _save_upload(file)
    # Defensive .get() — rows fetched through a connection that pre-dates
    # the Phase 3 schema (e.g. in tests) lack chunking_strategy/mode;
    # fall back to the same defaults as the column.
    kb_keys = kb_row.keys() if hasattr(kb_row, "keys") else ()
    chunking_strategy = (
        kb_row["chunking_strategy"]
        if "chunking_strategy" in kb_keys
        else "standard"
    )
    mode = kb_row["mode"] if "mode" in kb_keys else "text"
    return _start_ingestion(
        filename = filename,
        stored_path = stored_path,
        byte_size = byte_size,
        content_type = file.content_type,
        kb_id = kb_id,
        thread_id = None,
        embedding_model = kb_row["embedding_model"],
        chunking_strategy = chunking_strategy,
        mode = mode,
    )


@router.post("/threads/{thread_id}/documents", response_model = UploadResponse)
async def upload_thread_document(
    thread_id: str,
    file: UploadFile,
    current_subject: str = Depends(get_current_subject),
) -> UploadResponse:
    from utils.rag.config import RAG_EMBEDDING_MODEL

    # Don't validate against chat_threads — a brand-new chat won't be
    # persisted there until after the first runStart/runEnd. Users who
    # attach a document on a fresh thread would otherwise hit a 404.
    stored_path, filename, byte_size = await _save_upload(file)
    return _start_ingestion(
        filename = filename,
        stored_path = stored_path,
        byte_size = byte_size,
        content_type = file.content_type,
        kb_id = None,
        thread_id = thread_id,
        embedding_model = RAG_EMBEDDING_MODEL,
    )


# ------------------------------------------------------------------
# Document list / delete
# ------------------------------------------------------------------

@router.get("/knowledge-bases/{kb_id}/documents", response_model = DocumentListResponse)
def list_kb_documents(
    kb_id: str,
    current_subject: str = Depends(get_current_subject),
) -> DocumentListResponse:
    _kb_or_404(kb_id)
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM rag_documents WHERE kb_id = ? ORDER BY created_at DESC",
            (kb_id,),
        ).fetchall()
    return DocumentListResponse(documents = [_row_to_document(r) for r in rows])


@router.get("/threads/{thread_id}/documents", response_model = DocumentListResponse)
def list_thread_documents(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
) -> DocumentListResponse:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM rag_documents WHERE thread_id = ? ORDER BY created_at DESC",
            (thread_id,),
        ).fetchall()
    return DocumentListResponse(documents = [_row_to_document(r) for r in rows])


@router.get("/images/{document_id}/{filename}")
def get_rag_image(
    document_id: str,
    filename: str,
    current_subject: str = Depends(get_current_subject),
) -> FileResponse:
    """Serve an image extracted during multimodal ingestion.

    Files live under ``rag_uploads_root() / 'images' / <document_id>``.
    We realpath-check the resolved file against that root to refuse
    path-traversal attempts (``..`` segments, symlinks pointing
    elsewhere). Filenames are constrained to a single path component.
    """
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code = 400, detail = "Invalid filename")
    root = Path(os.path.realpath(rag_uploads_root() / "images"))
    candidate = (rag_uploads_root() / "images" / document_id / filename)
    try:
        real = Path(os.path.realpath(candidate))
        real.relative_to(root)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code = 404, detail = "Image not found") from exc
    if not real.is_file():
        raise HTTPException(status_code = 404, detail = "Image not found")
    return FileResponse(str(real))


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    row = _document_or_404(document_id)
    scope = (
        kb_scope(row["kb_id"]) if row["kb_id"] else thread_scope(row["thread_id"])
    )
    with get_connection() as conn:
        conn.execute("DELETE FROM rag_documents WHERE id = ?", (document_id,))
        conn.commit()
    _unlink_if_under_uploads(Path(row["stored_path"]))
    ingestion.delete_document_artifacts(document_id, scope)
    return {"ok": True}


@router.get("/thread-indexes", response_model = ThreadIndexListResponse)
def list_thread_indexes(
    current_subject: str = Depends(get_current_subject),
) -> ThreadIndexListResponse:
    """List every chat thread that has at least one RAG document.

    LEFT JOIN to chat_threads so threads that were never persisted
    (user attached a file but never sent a message) still show up —
    just with a null title.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                d.thread_id AS thread_id,
                t.title AS title,
                COUNT(DISTINCT d.id) AS num_documents,
                COALESCE(SUM(d.num_chunks), 0) AS num_chunks
            FROM rag_documents d
            LEFT JOIN chat_threads t ON t.id = d.thread_id
            WHERE d.thread_id IS NOT NULL
            GROUP BY d.thread_id, t.title
            ORDER BY MAX(d.created_at) DESC
            """
        ).fetchall()
    return ThreadIndexListResponse(
        threads = [
            ThreadIndexSummary(
                thread_id = r["thread_id"],
                title = r["title"],
                num_documents = int(r["num_documents"]),
                num_chunks = int(r["num_chunks"]),
            )
            for r in rows
        ]
    )


@router.delete("/threads/{thread_id}/documents")
def clear_thread_documents(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Purge every RAG document attached to ``thread_id``.

    Removes the per-thread Qdrant collection, the bm25 index, the
    rag_documents/rag_chunks rows, and the uploaded files. The chat
    thread itself is untouched.
    """
    ingestion.purge_thread_documents([thread_id])
    return {"ok": True}


# ------------------------------------------------------------------
# Ingestion job SSE
# ------------------------------------------------------------------

@router.get("/jobs/{job_id}/events")
async def job_events(
    job_id: str,
    request: Request,
    current_subject: str = Depends(_sse_auth),
) -> StreamingResponse:
    state = ingestion.get_job_state(job_id)
    if state is None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM rag_ingestion_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            raise HTTPException(status_code = 404, detail = "Job not found")
        return StreamingResponse(
            _replay_terminal_state(row),
            media_type = "text/event-stream",
        )

    consumer_queue = state.subscribe()

    async def stream():
        try:
            initial = {
                "type": "status",
                "status": state.status,
                "stage": state.stage,
                "progress": state.progress,
            }
            yield f"data: {json.dumps(initial)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.get_event_loop().run_in_executor(
                        None,
                        consumer_queue.get,
                        True,
                        15.0,
                    )
                except queue_module.Empty:
                    yield ": keep-alive\n\n"
                    if state.status in ("completed", "failed"):
                        break
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("complete", "error"):
                    break
        finally:
            state.unsubscribe(consumer_queue)

    return StreamingResponse(stream(), media_type = "text/event-stream")


async def _replay_terminal_state(row: Any):
    payload = {
        "type": "status",
        "status": row["status"],
        "stage": row["stage"],
        "progress": row["progress"],
        "error": row["error"],
    }
    yield f"data: {json.dumps(payload)}\n\n"


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

@router.post("/search", response_model = SearchResponse)
def search(
    payload: SearchRequest,
    current_subject: str = Depends(get_current_subject),
) -> SearchResponse:
    if bool(payload.kb_id) == bool(payload.thread_id):
        raise HTTPException(
            status_code = 400,
            detail = "exactly one of kb_id or thread_id must be supplied",
        )
    if payload.kb_id:
        _kb_or_404(payload.kb_id)
        scope = kb_scope(payload.kb_id)
    else:
        scope = thread_scope(payload.thread_id)

    # When reranking is opt-in, pull a wider candidate pool so the
    # CrossEncoder has more to choose from before truncating to top_k.
    candidate_k = (
        max(payload.top_k, RAG_RERANK_CANDIDATE_K)
        if payload.enable_rerank
        else payload.top_k
    )

    if payload.mode == "bm25":
        hits = retrieval.retrieve_bm25(scope, payload.query, candidate_k)
    elif payload.mode == "dense":
        hits = retrieval.retrieve_dense(
            scope,
            payload.query,
            candidate_k,
            document_ids = payload.document_ids,
        )
    else:
        hits = retrieval.retrieve_hybrid(
            scope,
            payload.query,
            k = candidate_k,
            document_ids = payload.document_ids,
        )

    chunk_ids = [h.chunk_id for h in hits]
    chunk_lookup: dict[str, dict] = {}
    if chunk_ids:
        placeholders = ",".join("?" for _ in chunk_ids)
        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.text,
                       c.page_number, c.kind, c.image_path, c.linked_chunk_id,
                       d.filename
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        for r in rows:
            chunk_lookup[r["chunk_id"]] = dict(r)

    if payload.enable_rerank:
        pairs = [
            (hit, chunk_lookup[hit.chunk_id]["text"])
            for hit in hits
            if hit.chunk_id in chunk_lookup
        ]
        hits = reranker.rerank(
            payload.query,
            pairs,
            model_name = payload.reranker_model,
            top_k = payload.top_k,
        )
    else:
        hits = hits[: payload.top_k]

    out: list[SearchHit] = []
    for hit in hits:
        meta = chunk_lookup.get(hit.chunk_id)
        if not meta:
            continue
        kind = meta.get("kind", "text") or "text"
        image_url: str | None = None
        if kind == "image" and meta.get("image_path"):
            image_url = (
                f"/api/rag/images/{meta['document_id']}/"
                f"{Path(meta['image_path']).name}"
            )
        out.append(
            SearchHit(
                chunk_id = hit.chunk_id,
                document_id = meta["document_id"],
                chunk_index = meta["chunk_index"],
                text = meta["text"] or "",
                score = hit.score,
                page_number = meta.get("page_number"),
                filename = meta.get("filename"),
                kind = kind,
                image_url = image_url,
            )
        )
    return SearchResponse(hits = out)
