# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG API: KB CRUD, document upload (KB + per-thread), ingestion SSE, search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue as queue_module
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote
from uuid import uuid4

import jwt
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject, get_current_subject_sse
from auth.storage import get_jwt_secret


async def _sse_auth(
    token: str | None = Query(None),
    authorization: str | None = Header(None),
) -> str:
    return await get_current_subject_sse(token, authorization)


from core.rag import embeddings, ingestion, reranker, retrieval, vector_store
from core.rag.authorization import document_for_subject_or_404
from core.rag.locators import backfill_document_locators
from core.rag.vector_store import kb_scope, thread_scope
from loggers import get_logger
from storage.studio_db import (
    get_connection,
    list_chat_settings,
    upsert_chat_settings_merge,
)
from utils.paths.storage_roots import ensure_dir, rag_uploads_root, resolve_under_root
from utils.rag.config import (
    RAG_MAX_UPLOAD_MB,
    RAG_RERANK_CANDIDATE_K,
    RAG_UPLOAD_EXTS,
)

router = APIRouter()
logger = get_logger(__name__)


# --- Pydantic schemas ---

ChunkingStrategy = Literal["standard", "late"]
KBMode = Literal["text", "multimodal"]


class CreateKBRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 200)
    description: str | None = None
    embedding_model: str | None = None
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
    # True when an identical file (same content hash) was already indexed
    # in this scope, so no new ingestion job was started. job_id is "".
    already_indexed: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length = 1, max_length = 4000)
    kb_id: str | None = None
    thread_id: str | None = None
    top_k: int = Field(default = 10, ge = 1, le = 100)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    document_ids: list[str] | None = None
    enable_rerank: bool = False
    reranker_model: str | None = None
    min_score: float = Field(default = 0.0, ge = 0.0, le = 1.0)


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
    source_page_index: int | None = None
    page_char_start: int | None = None
    page_char_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class PrefetchRequest(BaseModel):
    """Prefetch RAG for an external-provider turn: studio decomposes the
    question (via the pre-cached helper) and retrieves, so the frontend can
    inject the chunks before calling the provider."""

    query: str = Field(min_length = 1, max_length = 4000)
    kb_id: str | None = None
    thread_id: str | None = None
    top_k: int = Field(default = 10, ge = 1, le = 100)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    enable_rerank: bool = False
    reranker_model: str | None = None
    min_score: float = Field(default = 0.0, ge = 0.0, le = 1.0)


class PrefetchResponse(BaseModel):
    queries: list[str]
    hits: list[SearchHit]


# --- Helpers ---


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    return name or "document"


def _now_ms() -> int:
    return int(time.time())


from core.rag.scope import resolve_scope_embedder as _resolve_scope_embedder  # noqa: E402


def _row_to_kb(row: Any) -> KBResponse:
    keys = row.keys() if hasattr(row, "keys") else ()
    chunking_strategy = (
        row["chunking_strategy"] if "chunking_strategy" in keys else "standard"
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
    """Reject (multimodal, late) — no embedder supports both at once."""
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


async def _save_upload(file: UploadFile) -> tuple[Path, str, int, str]:
    import anyio

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
    # Hash the bytes as they stream so we can dedup identical re-uploads
    # within a scope without re-reading the file.
    hasher = hashlib.sha256()
    # Route writes through anyio worker thread so the event loop stays free.
    # Outer try/except cleans up partial files after async-with closes the fd
    # (Windows refuses unlink on an open fd).
    try:
        async with await anyio.open_file(stored_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code = 413,
                        detail = f"File exceeds {RAG_MAX_UPLOAD_MB} MB limit",
                    )
                hasher.update(chunk)
                await f.write(chunk)
    except HTTPException:
        stored_path.unlink(missing_ok = True)
        raise
    if written == 0:
        stored_path.unlink(missing_ok = True)
        raise HTTPException(status_code = 400, detail = "Empty upload payload")
    return stored_path, filename, written, hasher.hexdigest()


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
    content_hash: str | None = None,
) -> UploadResponse:
    document_id = str(uuid4())
    with get_connection() as conn:
        # Dedup: if an identical file (same content hash) is already
        # indexed in this scope, skip re-ingestion. Only a 'completed'
        # row counts — a failed/in-flight prior attempt should be allowed
        # to retry. Scope is the same kb_id or thread_id the upload
        # targets (a file shared across two KBs is indexed in each).
        if content_hash:
            if kb_id is not None:
                existing = conn.execute(
                    "SELECT id, filename FROM rag_documents "
                    "WHERE kb_id = ? AND content_hash = ? AND status = 'completed' "
                    "LIMIT 1",
                    (kb_id, content_hash),
                ).fetchone()
            else:
                existing = conn.execute(
                    "SELECT id, filename FROM rag_documents "
                    "WHERE thread_id = ? AND content_hash = ? AND status = 'completed' "
                    "LIMIT 1",
                    (thread_id, content_hash),
                ).fetchone()
            if existing is not None:
                # Drop the redundant upload we just wrote to disk; the
                # already-indexed copy stays the source of truth.
                _unlink_if_under_uploads(stored_path)
                return UploadResponse(
                    document_id = existing["id"],
                    job_id = "",
                    filename = existing["filename"],
                    already_indexed = True,
                )
        conn.execute(
            """
            INSERT INTO rag_documents
            (id, kb_id, thread_id, filename, content_type, stored_path,
             status, num_chunks, byte_size, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
            """,
            (
                document_id,
                kb_id,
                thread_id,
                filename,
                content_type,
                str(stored_path),
                byte_size,
                content_hash,
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


# --- Knowledge bases ---


@router.post("/knowledge-bases", response_model = KBResponse)
def create_knowledge_base(
    payload: CreateKBRequest,
    current_subject: str = Depends(get_current_subject),
) -> KBResponse:
    from utils.rag.config import resolve_embedder

    _validate_mode_combo(payload.mode, payload.chunking_strategy)

    kb_id = str(uuid4())
    # No override: resolve from (mode, strategy) matrix.
    embedding_model = payload.embedding_model or resolve_embedder(
        payload.mode, payload.chunking_strategy
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


class RagDefaults(BaseModel):
    chunking_strategy: ChunkingStrategy = "standard"
    mode: KBMode = "text"
    embedding_model: str | None = None


class UpdateRagDefaultsRequest(BaseModel):
    """Patch shape — only fields present overwrite stored values."""

    chunking_strategy: ChunkingStrategy | None = None
    mode: KBMode | None = None
    embedding_model: str | None = None


_DEFAULTS_KEY = "rag.defaults"


def _load_rag_defaults() -> RagDefaults:
    settings = list_chat_settings()
    raw = settings.get(_DEFAULTS_KEY) or {}
    if not isinstance(raw, dict):
        raw = {}
    return RagDefaults(
        chunking_strategy = raw.get("chunking_strategy") or "standard",
        mode = raw.get("mode") or "text",
        embedding_model = raw.get("embedding_model"),
    )


@router.get("/defaults", response_model = RagDefaults)
def get_rag_defaults(
    current_subject: str = Depends(get_current_subject),
) -> RagDefaults:
    return _load_rag_defaults()


@router.post("/warmup")
def warmup_rag_embedder(
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Preload the configured default embedder so the first retrieval is warm.

    Called from the frontend when the user enables the RAG pill — moves
    the cold-load latency out of the first chat-completion path, where a
    multi-second load can race the llama-server prefill timeout.
    """
    from utils.rag.config import resolve_embedder

    defaults = _load_rag_defaults()
    model_name = defaults.embedding_model or resolve_embedder(
        defaults.mode,
        defaults.chunking_strategy,
    )
    try:
        embeddings.get_embedder(model_name)
    except Exception as exc:  # noqa: BLE001
        # Log the detailed exception server-side; return a generic message
        # so internal paths / stack info aren't exposed to the client.
        logger.warning("RAG warmup failed for %s: %s", model_name, exc)
        return {"ok": False, "model": model_name, "error": "Failed to load embedder"}
    return {"ok": True, "model": model_name}


@router.post("/reranker/precache")
def precache_rag_reranker(
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Download the reranker weights (~1.1 GB) into the HF cache.

    Called from the frontend the moment the user flips the "Use
    reranker" switch ON so the cost lands on the explicit toggle
    instead of the first chat turn — where a multi-minute download
    looks like a hung tool call.
    """
    from core.rag.reranker import precache_reranker
    from utils.rag.config import RAG_RERANKER_MODEL

    try:
        precache_reranker()
    except Exception as exc:  # noqa: BLE001
        # Detailed exception logged server-side; client gets a generic
        # message so internal paths / stack info aren't exposed.
        logger.warning(
            "RAG reranker precache failed",
            model = RAG_RERANKER_MODEL,
            error = str(exc),
        )
        return {
            "ok": False,
            "model": RAG_RERANKER_MODEL,
            "error": "Failed to download reranker",
        }
    return {"ok": True, "model": RAG_RERANKER_MODEL}


@router.put("/defaults", response_model = RagDefaults)
def set_rag_defaults(
    payload: UpdateRagDefaultsRequest,
    current_subject: str = Depends(get_current_subject),
) -> RagDefaults:
    current = _load_rag_defaults()
    new_strategy = payload.chunking_strategy or current.chunking_strategy
    new_mode = payload.mode or current.mode
    # PATCH-style: empty string clears, null/missing keeps current.
    if payload.embedding_model is None:
        new_embedder = current.embedding_model
    elif payload.embedding_model.strip() == "":
        new_embedder = None
    else:
        new_embedder = payload.embedding_model.strip()
    _validate_mode_combo(new_mode, new_strategy)

    upsert_chat_settings_merge(
        {
            _DEFAULTS_KEY: {
                "chunking_strategy": new_strategy,
                "mode": new_mode,
                "embedding_model": new_embedder,
            }
        }
    )
    return RagDefaults(
        chunking_strategy = new_strategy,
        mode = new_mode,
        embedding_model = new_embedder,
    )


class ThreadRagSettings(BaseModel):
    chunking_strategy: ChunkingStrategy = "standard"
    mode: KBMode = "text"
    embedding_model: str | None = None


class UpdateThreadRagSettingsRequest(BaseModel):
    chunking_strategy: ChunkingStrategy | None = None
    mode: KBMode | None = None
    embedding_model: str | None = None


def _thread_settings_key(thread_id: str) -> str:
    return f"thread:{thread_id}:rag"


def _load_thread_settings(thread_id: str) -> ThreadRagSettings:
    """Per-thread RAG settings (chat_settings['thread:<id>:rag']) with defaults fallback."""
    settings = list_chat_settings()
    raw = settings.get(_thread_settings_key(thread_id)) or {}
    if not isinstance(raw, dict):
        raw = {}
    fallback = _load_rag_defaults()
    return ThreadRagSettings(
        chunking_strategy = (raw.get("chunking_strategy") or fallback.chunking_strategy),
        mode = raw.get("mode") or fallback.mode,
        embedding_model = raw.get("embedding_model") or fallback.embedding_model,
    )


@router.get(
    "/threads/{thread_id}/settings",
    response_model = ThreadRagSettings,
)
def get_thread_rag_settings(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
) -> ThreadRagSettings:
    return _load_thread_settings(thread_id)


@router.put(
    "/threads/{thread_id}/settings",
    response_model = ThreadRagSettings,
)
def set_thread_rag_settings(
    thread_id: str,
    payload: UpdateThreadRagSettingsRequest,
    current_subject: str = Depends(get_current_subject),
) -> ThreadRagSettings:
    current = _load_thread_settings(thread_id)
    new_strategy = payload.chunking_strategy or current.chunking_strategy
    new_mode = payload.mode or current.mode
    if payload.embedding_model is None:
        new_embedder = current.embedding_model
    elif payload.embedding_model.strip() == "":
        new_embedder = None
    else:
        new_embedder = payload.embedding_model.strip()
    _validate_mode_combo(new_mode, new_strategy)

    upsert_chat_settings_merge(
        {
            _thread_settings_key(thread_id): {
                "chunking_strategy": new_strategy,
                "mode": new_mode,
                "embedding_model": new_embedder,
            }
        }
    )
    return ThreadRagSettings(
        chunking_strategy = new_strategy,
        mode = new_mode,
        embedding_model = new_embedder,
    )


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
    """Wipe scope artifacts and re-enqueue every document; metadata untouched."""
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
        # Drop rag_documents (chunks cascade); files on disk are reused below.
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
        # Strip the upload-time UUID prefix; keep the original filename.
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
        kb_row["chunking_strategy"] if "chunking_strategy" in keys else "standard"
    )
    current_mode = kb_row["mode"] if "mode" in keys else "text"
    current_embedder = kb_row["embedding_model"]

    new_strategy = payload.chunking_strategy or current_strategy
    new_mode = payload.mode or current_mode
    _validate_mode_combo(new_mode, new_strategy)

    new_embedder = payload.embedding_model or (
        current_embedder
        if (new_strategy == current_strategy and new_mode == current_mode)
        else resolve_embedder(new_mode, new_strategy)
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
    payload: UpdateThreadRagSettingsRequest | None = None,
    current_subject: str = Depends(get_current_subject),
) -> ReingestResponse:
    """Rebuild a thread's RAG index; optional body updates settings before reingest."""
    from utils.rag.config import resolve_embedder

    if payload is None:
        payload = UpdateThreadRagSettingsRequest()
    if (
        payload.chunking_strategy is not None
        or payload.mode is not None
        or payload.embedding_model is not None
    ):
        settings = set_thread_rag_settings(
            thread_id,
            payload,
            current_subject = current_subject,
        )
    else:
        settings = _load_thread_settings(thread_id)

    embedder = settings.embedding_model or resolve_embedder(
        settings.mode,
        settings.chunking_strategy,
    )
    return _reingest_scope(
        kb_id = None,
        thread_id = thread_id,
        chunking_strategy = settings.chunking_strategy,
        mode = settings.mode,
        embedding_model = embedder,
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


# --- Document upload (KB and per-thread) ---


@router.post("/knowledge-bases/{kb_id}/documents", response_model = UploadResponse)
async def upload_kb_document(
    kb_id: str,
    file: UploadFile,
    current_subject: str = Depends(get_current_subject),
) -> UploadResponse:
    kb_row = _kb_or_404(kb_id)
    stored_path, filename, byte_size, content_hash = await _save_upload(file)
    # Tolerate pre-Phase-3 rows missing chunking_strategy/mode.
    kb_keys = kb_row.keys() if hasattr(kb_row, "keys") else ()
    chunking_strategy = (
        kb_row["chunking_strategy"] if "chunking_strategy" in kb_keys else "standard"
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
        content_hash = content_hash,
    )


@router.post("/threads/{thread_id}/documents", response_model = UploadResponse)
async def upload_thread_document(
    thread_id: str,
    file: UploadFile,
    current_subject: str = Depends(get_current_subject),
) -> UploadResponse:
    from utils.rag.config import resolve_embedder

    # No chat_threads check — fresh threads aren't persisted until first run.
    stored_path, filename, byte_size, content_hash = await _save_upload(file)
    settings = _load_thread_settings(thread_id)
    embedder = settings.embedding_model or resolve_embedder(
        settings.mode,
        settings.chunking_strategy,
    )
    return _start_ingestion(
        filename = filename,
        stored_path = stored_path,
        byte_size = byte_size,
        content_type = file.content_type,
        kb_id = None,
        thread_id = thread_id,
        embedding_model = embedder,
        chunking_strategy = settings.chunking_strategy,
        mode = settings.mode,
        content_hash = content_hash,
    )


# --- Document list / delete ---


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
    """Serve an extracted image; realpath-check against the uploads root."""
    document_for_subject_or_404(document_id, current_subject)
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code = 400, detail = "Invalid filename")
    root = Path(os.path.realpath(rag_uploads_root() / "images"))
    candidate = rag_uploads_root() / "images" / document_id / filename
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
    scope = kb_scope(row["kb_id"]) if row["kb_id"] else thread_scope(row["thread_id"])
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
    """List threads with >=1 RAG doc. LEFT JOIN keeps unpersisted threads (null title)."""
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
    """Drop all RAG artifacts for thread_id; chat thread itself untouched."""
    ingestion.purge_thread_documents([thread_id])
    return {"ok": True}


# --- Ingestion job SSE ---


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


# --- Search ---


# --- Document preview (file + preview-target) ---


PreviewMediaKind = Literal["pdf", "text", "docx", "html", "image", "unknown"]
PreviewChunkKind = Literal["text", "image", "caption"]


class PreviewPdfRegion(BaseModel):
    pageIndex: int
    pageNumber: int | None = None
    x: float
    y: float
    width: float
    height: float
    confidence: Literal["exact"]
    source: str


class PreviewTargetResponse(BaseModel):
    """Per contracts.md §1.2 / §1.3 — single shape covering both
    cited-chunk and document-row preview modes. The §1.3 metadata-only
    mode returns ``None`` for ``chunkId``/``chunkIndex``/``targetPage``/
    ``snippet``/``kind``/``imageUrl`` (Q2: no first-chunk guessing).
    """

    documentId: str
    filename: str
    contentType: str | None
    mediaKind: PreviewMediaKind
    byteSize: int
    status: str
    kbId: str | None
    threadId: str | None
    chunkId: str | None
    chunkIndex: int | None
    targetPage: int | None
    snippet: str | None
    kind: PreviewChunkKind | None
    imageUrl: str | None
    sourcePageIndex: int | None
    pageCharStart: int | None
    pageCharEnd: int | None
    lineStart: int | None
    lineEnd: int | None
    pdfRegions: list[PreviewPdfRegion] = Field(default_factory = list)


class PreviewFileUrlResponse(BaseModel):
    url: str
    expiresAt: int


class LocatorBackfillResponse(BaseModel):
    documentId: str
    totalChunks: int
    matched: int
    alreadyLocated: int
    ambiguous: int
    missing: int
    skipped: int
    regionsMatched: int
    pagesRefreshed: int


# Extension allowlist for inline rendering / disposition. Anything not in
# this map collapses to ("application/octet-stream", attachment, "unknown").
# .html / .htm intentionally serve as text/plain attachment (decisions Q7 +
# Risk #3) so an uploaded HTML cannot execute in the app origin.
_PREVIEW_EXT_MAP: dict[str, tuple[str, str, PreviewMediaKind]] = {
    ".pdf": ("application/pdf", "inline", "pdf"),
    ".txt": ("text/plain; charset=utf-8", "inline", "text"),
    ".md": ("text/markdown; charset=utf-8", "inline", "text"),
    ".markdown": ("text/markdown; charset=utf-8", "inline", "text"),
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "attachment",
        "docx",
    ),
    ".html": ("text/plain; charset=utf-8", "attachment", "html"),
    ".htm": ("text/plain; charset=utf-8", "attachment", "html"),
    ".png": ("image/png", "inline", "image"),
    ".jpg": ("image/jpeg", "inline", "image"),
    ".jpeg": ("image/jpeg", "inline", "image"),
    ".gif": ("image/gif", "inline", "image"),
    ".webp": ("image/webp", "inline", "image"),
}


def _ascii_only(value: str) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _content_disposition_header(filename: str, disposition: str) -> str:
    """Build a Content-Disposition header. Non-ASCII filenames use RFC 5987
    ``filename*=UTF-8''…`` alongside an ASCII-only ``filename=`` fallback so
    older clients still get something readable.
    """
    from urllib.parse import quote as _urlquote

    safe = filename.replace('"', "").replace("\r", "").replace("\n", "")
    if _ascii_only(safe):
        return f'{disposition}; filename="{safe}"'
    ascii_fallback = safe.encode("ascii", "replace").decode("ascii")
    encoded = _urlquote(safe, safe = "")
    return f'{disposition}; filename="{ascii_fallback}"; ' f"filename*=UTF-8''{encoded}"


def _preview_file_metadata(filename: str) -> tuple[str, str, PreviewMediaKind]:
    """Map a stored filename to (content_type, disposition, mediaKind).

    Unknown extensions always force ``application/octet-stream`` +
    ``attachment`` + ``unknown`` so the browser cannot sniff a sensitive
    type and inline it (Risk #3).
    """
    ext = Path(filename).suffix.lower()
    return _PREVIEW_EXT_MAP.get(
        ext, ("application/octet-stream", "attachment", "unknown")
    )


_PREVIEW_FILE_AUDIENCE = "rag-preview-file"
_PREVIEW_FILE_TTL_SECONDS = 5 * 60
_JWT_ALGORITHM = "HS256"


def _parse_pdf_regions(value: str | None) -> list[PreviewPdfRegion]:
    if not value:
        return []
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[PreviewPdfRegion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            region = PreviewPdfRegion(**item)
        except Exception:
            continue
        if (
            0 <= region.x <= 1
            and 0 <= region.y <= 1
            and region.width > 0
            and region.height > 0
        ):
            out.append(region)
    return out


def _preview_file_token(
    *,
    subject: str,
    document_id: str,
) -> tuple[str, int]:
    secret = get_jwt_secret(subject)
    if secret is None:
        raise HTTPException(status_code = 401, detail = "Invalid or expired token")
    expires = datetime.now(timezone.utc) + timedelta(seconds = _PREVIEW_FILE_TTL_SECONDS)
    payload = {
        "sub": subject,
        "aud": _PREVIEW_FILE_AUDIENCE,
        "document_id": document_id,
        "exp": expires,
    }
    token = jwt.encode(payload, secret, algorithm = _JWT_ALGORITHM)
    return token, int(expires.timestamp())


def _subject_from_preview_file_token(document_id: str, token: str) -> str:
    try:
        unverified = jwt.decode(
            token,
            options = {
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
            },
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code = 401, detail = "Invalid preview token") from exc
    subject = unverified.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(status_code = 401, detail = "Invalid preview token")
    secret = get_jwt_secret(subject)
    if secret is None:
        raise HTTPException(status_code = 401, detail = "Invalid preview token")
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms = [_JWT_ALGORITHM],
            audience = _PREVIEW_FILE_AUDIENCE,
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code = 401, detail = "Invalid preview token") from exc
    if payload.get("document_id") != document_id:
        raise HTTPException(status_code = 401, detail = "Invalid preview token")
    return subject


def _resolve_document_file_or_404(doc_row: Any, document_id: str) -> Path:
    try:
        resolved = resolve_under_root(
            doc_row["stored_path"],
            root = rag_uploads_root(),
        )
    except ValueError as exc:
        # Symlink escape / ``..`` / absolute outside root — collapse to
        # "file not found" (the auth row exists, the bytes do not).
        logger.warning(
            "RAG preview: stored_path escaped uploads root for doc %s: %s",
            document_id,
            exc,
        )
        raise HTTPException(
            status_code = 404,
            detail = "Document file not found",
        ) from exc

    if not resolved.is_file():
        raise HTTPException(
            status_code = 404,
            detail = "Document file not found",
        )
    return resolved


def _parse_range_header(range_header: str | None, size: int) -> tuple[int, int] | None:
    if not range_header:
        return None
    if not range_header.startswith("bytes="):
        raise ValueError("unsupported range unit")
    spec = range_header[len("bytes=") :].strip()
    if "," in spec or "-" not in spec:
        raise ValueError("multiple or malformed ranges are not supported")
    start_s, end_s = spec.split("-", 1)
    if not start_s and not end_s:
        raise ValueError("empty range")
    if not start_s:
        suffix = int(end_s)
        if suffix <= 0:
            raise ValueError("invalid suffix range")
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    if start < 0 or end < start or start >= size:
        raise ValueError("range outside file")
    return start, min(end, size - 1)


def _iter_file_range(path: Path, start: int, end: int):
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = fh.read(min(64 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _serve_document_file_row(
    doc_row: Any,
    document_id: str,
    range_header: str | None,
) -> FileResponse | Response | StreamingResponse:
    resolved = _resolve_document_file_or_404(doc_row, document_id)
    content_type, disposition, _media_kind = _preview_file_metadata(doc_row["filename"])
    safe_name = _sanitize_filename(doc_row["filename"])
    headers = {
        "Content-Disposition": _content_disposition_header(safe_name, disposition),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=0, must-revalidate",
        "Accept-Ranges": "bytes",
    }
    size = resolved.stat().st_size

    try:
        byte_range = _parse_range_header(range_header, size)
    except (TypeError, ValueError):
        range_headers = dict(headers)
        range_headers["Content-Range"] = f"bytes */{size}"
        return Response(status_code = 416, headers = range_headers)

    if byte_range is not None:
        start, end = byte_range
        range_headers = dict(headers)
        range_headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        range_headers["Content-Length"] = str(end - start + 1)
        return StreamingResponse(
            _iter_file_range(resolved, start, end),
            status_code = 206,
            media_type = content_type,
            headers = range_headers,
        )

    # Starlette's FileResponse handles ordinary downloads efficiently. We
    # still advertise Accept-Ranges so PDF.js can switch to explicit range
    # requests via the signed URL path.
    return FileResponse(
        path = str(resolved),
        media_type = content_type,
        headers = headers,
    )


@router.get(
    "/documents/{document_id}/preview-target",
    response_model = PreviewTargetResponse,
)
def get_document_preview_target(
    document_id: str,
    chunk_id: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
) -> PreviewTargetResponse:
    """Resolve preview metadata for a document, optionally focused on a chunk.

    See contracts.md §1 for the response shape. ``chunk_id`` is the durable
    ``rag_chunks.id`` carried as ``backendChunkId`` on the frontend; when
    supplied it MUST belong to ``document_id`` or we collapse to 404 (so a
    probe cannot enumerate cross-document chunk ids).
    """
    doc_row = document_for_subject_or_404(document_id, current_subject)
    _ct, _disposition, media_kind = _preview_file_metadata(doc_row["filename"])

    base = {
        "documentId": doc_row["id"],
        "filename": doc_row["filename"],
        "contentType": doc_row["content_type"],
        "mediaKind": media_kind,
        "byteSize": int(doc_row["byte_size"]),
        "status": doc_row["status"],
        "kbId": doc_row["kb_id"],
        "threadId": doc_row["thread_id"],
    }

    if not chunk_id:
        # Q2: document-row preview returns metadata only — frontend MUST
        # NOT fall back to "first chunk".
        return PreviewTargetResponse(
            **base,
            chunkId = None,
            chunkIndex = None,
            targetPage = None,
            snippet = None,
            kind = None,
            imageUrl = None,
            sourcePageIndex = None,
            pageCharStart = None,
            pageCharEnd = None,
            lineStart = None,
            lineEnd = None,
            pdfRegions = [],
        )

    # Single connection enforces membership AND fetches the row in one
    # query. Splitting this into a separate `chunk_belongs_to_document`
    # call would open a second SQLite connection and create a TOCTOU
    # window — if the chunk is deleted between the two calls, the data
    # fetch returns None and the route 500s on the next attribute access
    # (devils-advocate D1.1). The cross-document case still collapses to
    # the same 404 the auth helper emits — never 400 (would leak doc
    # existence).
    with get_connection() as conn:
        chunk_row = conn.execute(
            """
            SELECT id, chunk_index, page_number, text, kind, image_path,
                   source_page_index, page_char_start, page_char_end,
                   line_start, line_end, pdf_regions_json
            FROM rag_chunks WHERE id = ? AND document_id = ?
            """,
            (chunk_id, document_id),
        ).fetchone()

    if chunk_row is None:
        raise HTTPException(
            status_code = 404,
            detail = "Document not found",
        )

    chunk_kind: PreviewChunkKind = chunk_row["kind"] or "text"  # type: ignore[assignment]
    image_url: str | None = None
    if chunk_kind == "image" and chunk_row["image_path"]:
        image_url = (
            f"/api/rag/images/{doc_row['id']}/" f"{Path(chunk_row['image_path']).name}"
        )

    return PreviewTargetResponse(
        **base,
        chunkId = chunk_row["id"],
        chunkIndex = int(chunk_row["chunk_index"]),
        targetPage = (
            int(chunk_row["page_number"])
            if chunk_row["page_number"] is not None
            else None
        ),
        snippet = chunk_row["text"] or "",
        kind = chunk_kind,
        imageUrl = image_url,
        sourcePageIndex = chunk_row["source_page_index"],
        pageCharStart = chunk_row["page_char_start"],
        pageCharEnd = chunk_row["page_char_end"],
        lineStart = chunk_row["line_start"],
        lineEnd = chunk_row["line_end"],
        pdfRegions = _parse_pdf_regions(chunk_row["pdf_regions_json"]),
    )


@router.post(
    "/documents/{document_id}/locators/backfill",
    response_model = LocatorBackfillResponse,
)
def backfill_document_locators_route(
    document_id: str,
    current_subject: str = Depends(get_current_subject),
) -> LocatorBackfillResponse:
    """In-place locator backfill for existing citations.

    This preserves ``document_id`` and ``chunk_id``. Chunks are updated only
    when their text has one unambiguous match in the parsed document text;
    duplicate or missing matches remain null.
    """
    doc_row = document_for_subject_or_404(document_id, current_subject)
    resolved = _resolve_document_file_or_404(doc_row, document_id)
    try:
        result = backfill_document_locators(document_id, resolved)
    except Exception as exc:
        logger.warning(
            "RAG locator backfill failed",
            document_id = document_id,
            error = str(exc),
        )
        raise HTTPException(
            status_code = 400,
            detail = "Document locators could not be backfilled",
        ) from exc
    return LocatorBackfillResponse(
        documentId = result.document_id,
        totalChunks = result.total_chunks,
        matched = result.matched,
        alreadyLocated = result.already_located,
        ambiguous = result.ambiguous,
        missing = result.missing,
        skipped = result.skipped,
        regionsMatched = result.regions_matched,
        pagesRefreshed = result.pages_refreshed,
    )


@router.get(
    "/documents/{document_id}/file-url",
    response_model = PreviewFileUrlResponse,
)
def get_document_file_url(
    document_id: str,
    current_subject: str = Depends(get_current_subject),
) -> PreviewFileUrlResponse:
    """Mint a short-lived signed URL for PDF.js range requests.

    The normal bearer-protected `/file` route stays available for blob
    fallback; this route keeps bearer access tokens out of query strings.
    """
    document_for_subject_or_404(document_id, current_subject)
    token, expires_at = _preview_file_token(
        subject = current_subject,
        document_id = document_id,
    )
    url = (
        f"/api/rag/documents/{quote(document_id, safe = '')}/file-signed"
        f"?token={quote(token, safe = '')}"
    )
    return PreviewFileUrlResponse(url = url, expiresAt = expires_at)


@router.get("/documents/{document_id}/file-signed", response_model = None)
def get_signed_document_file(
    document_id: str,
    request: Request,
    token: str = Query(...),
) -> FileResponse | Response | StreamingResponse:
    subject = _subject_from_preview_file_token(document_id, token)
    doc_row = document_for_subject_or_404(document_id, subject)
    return _serve_document_file_row(
        doc_row,
        document_id,
        request.headers.get("range"),
    )


@router.get("/documents/{document_id}/file", response_model = None)
def get_document_file(
    document_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
) -> FileResponse | Response | StreamingResponse:
    """Serve the original uploaded bytes for ``document_id``.

    Path resolution per contracts.md §2.1: the route NEVER accepts a
    client-supplied filename. ``stored_path`` is DB-issued and we run it
    through ``resolve_under_root`` which delegates to ``_assert_contained``
    (realpath + symlink/junction-safe). Any escape (symlink, junction,
    ``..``, absolute outside root) collapses to the second 404.

    Content-Type and disposition come from the extension allowlist
    (``_preview_file_metadata``). HTML/DOCX/unknown serve as
    ``attachment`` with safe content-type so an uploaded ``.html`` can
    never execute in the app origin (Risk #3 / decisions Q7).
    """
    doc_row = document_for_subject_or_404(document_id, current_subject)
    return _serve_document_file_row(
        doc_row,
        document_id,
        request.headers.get("range"),
    )


def _execute_search(
    scope: str,
    *,
    scope_embedder: str | None,
    query: str,
    mode: str,
    top_k: int,
    document_ids: list[str] | None,
    enable_rerank: bool,
    reranker_model: str | None,
    min_score: float,
) -> list[SearchHit]:
    """Run one retrieval against an already-resolved scope. Shared by the
    /search and /prefetch endpoints."""
    candidate_k = (
        max(top_k, RAG_RERANK_CANDIDATE_K) if enable_rerank else top_k
    )

    if mode == "bm25":
        hits = retrieval.retrieve_bm25(scope, query, candidate_k)
    elif mode == "dense":
        hits = retrieval.retrieve_dense(
            scope,
            query,
            candidate_k,
            document_ids = document_ids,
            embedder_model = scope_embedder,
        )
    else:
        hits = retrieval.retrieve_hybrid(
            scope,
            query,
            k = candidate_k,
            document_ids = document_ids,
            embedder_model = scope_embedder,
        )

    if min_score > 0.0:
        hits = retrieval.filter_by_min_score(hits, min_score)

    chunk_ids = [h.chunk_id for h in hits]
    chunk_lookup: dict[str, dict] = {}
    if chunk_ids:
        placeholders = ",".join("?" for _ in chunk_ids)
        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.text,
                       c.page_number, c.kind, c.image_path, c.linked_chunk_id,
                       c.source_page_index, c.page_char_start, c.page_char_end,
                       c.line_start, c.line_end,
                       d.filename
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        for r in rows:
            chunk_lookup[r["chunk_id"]] = dict(r)

    if enable_rerank:
        pairs = [
            (hit, chunk_lookup[hit.chunk_id]["text"])
            for hit in hits
            if hit.chunk_id in chunk_lookup
        ]
        hits = reranker.rerank(
            query,
            pairs,
            model_name = reranker_model,
            top_k = top_k,
        )
    else:
        hits = hits[:top_k]

    out: list[SearchHit] = []
    for hit in hits:
        meta = chunk_lookup.get(hit.chunk_id)
        if not meta:
            continue
        kind = meta.get("kind", "text") or "text"
        image_url: str | None = None
        if kind == "image" and meta.get("image_path"):
            image_url = (
                f"/api/rag/images/{meta['document_id']}/{Path(meta['image_path']).name}"
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
                source_page_index = meta.get("source_page_index"),
                page_char_start = meta.get("page_char_start"),
                page_char_end = meta.get("page_char_end"),
                line_start = meta.get("line_start"),
                line_end = meta.get("line_end"),
            )
        )
    return out


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

    # Query must use the same embedder as the scope (dim must match).
    scope_embedder = _resolve_scope_embedder(scope)
    logger.info(
        "RAG search: scope=%s embedder=%s mode=%s top_k=%d min_score=%.3f rerank=%s query=%r",
        scope,
        scope_embedder or "<default>",
        payload.mode,
        payload.top_k,
        payload.min_score,
        payload.enable_rerank,
        payload.query[:120],
    )

    out = _execute_search(
        scope,
        scope_embedder = scope_embedder,
        query = payload.query,
        mode = payload.mode,
        top_k = payload.top_k,
        document_ids = payload.document_ids,
        enable_rerank = payload.enable_rerank,
        reranker_model = payload.reranker_model,
        min_score = payload.min_score,
    )
    logger.info("RAG search: returning %d hits", len(out))
    return SearchResponse(hits = out)


@router.post("/prefetch", response_model = PrefetchResponse)
def prefetch(
    payload: PrefetchRequest,
    current_subject: str = Depends(get_current_subject),
) -> PrefetchResponse:
    """External-provider RAG prefetch: decompose the question via the helper,
    retrieve per sub-query, merge+dedup, return chunks for the frontend to
    inject before calling the provider. The local tool path is unaffected."""
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

    scope_embedder = _resolve_scope_embedder(scope)

    # Momentarily load the pre-cached helper to split the question into up to
    # 3 focused queries; falls back to [query] on any failure.
    from core.rag.query_decompose import decompose_query

    queries = decompose_query(payload.query)
    logger.info(
        "RAG prefetch: scope=%s embedder=%s mode=%s n_queries=%d rerank=%s",
        scope,
        scope_embedder or "<default>",
        payload.mode,
        len(queries),
        payload.enable_rerank,
    )

    # Retrieve per query, merge, dedup by chunk_id (keep first/highest-ranked
    # occurrence), then cap at top_k.
    merged: list[SearchHit] = []
    seen: set[str] = set()
    for q in queries:
        hits = _execute_search(
            scope,
            scope_embedder = scope_embedder,
            query = q,
            mode = payload.mode,
            top_k = payload.top_k,
            document_ids = None,
            enable_rerank = payload.enable_rerank,
            reranker_model = payload.reranker_model,
            min_score = payload.min_score,
        )
        for h in hits:
            if h.chunk_id in seen:
                continue
            seen.add(h.chunk_id)
            merged.append(h)

    merged = merged[: payload.top_k]
    logger.info("RAG prefetch: returning %d merged hits", len(merged))
    return PrefetchResponse(queries = queries, hits = merged)
