# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP API for the RAG engine.

Knowledge-base CRUD, document upload (per-KB and per-thread), ingestion job
progress over Server-Sent Events, document listing/deletion, and a direct
hybrid/lexical/dense search endpoint for the UI. All endpoints are authenticated
with ``get_current_subject``. Studio stores RAG data globally per install (it is
single-tenant), so the subject gates access rather than partitioning data.

If the sqlite-vec extension is unavailable the router still mounts but every
endpoint returns 503 with a clear message, so ordinary chat is never affected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.rag import config, ingestion, retrieval, store
from storage import rag_db
from utils.paths import ensure_dir, rag_uploads_root

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_rag() -> None:
    if not rag_db.RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAG is unavailable: the sqlite-vec extension could not be loaded.",
        )


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "").strip() or "document"
    base = _SAFE.sub("_", base)
    return base[:200]


def _save_upload(file: UploadFile) -> tuple[str, str]:
    """Persist an upload under the rag uploads root. Returns (stored_path, filename)."""
    filename = _sanitize_filename(file.filename or "document")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(config.UPLOAD_EXTS)}",
        )
    uploads = ensure_dir(rag_uploads_root())
    stored_path = str(uploads / f"{uuid.uuid4().hex}{ext}")
    size = 0
    with open(stored_path, "wb") as out:
        while True:
            block = file.file.read(1 << 20)
            if not block:
                break
            size += len(block)
            out.write(block)
    if size == 0:
        os.remove(stored_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return stored_path, filename


def _doc_view(row: dict) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "status": row["status"],
        "error": row.get("error"),
        "numChunks": row.get("num_chunks") or 0,
        "kbId": row.get("kb_id"),
        "threadId": row.get("thread_id"),
        "createdAt": row.get("created_at"),
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CreateKbRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None


class UpdateKbRequest(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    description: str | None = None


class SearchRequest(BaseModel):
    query: str
    kb_id: str | None = None
    thread_id: str | None = None
    top_k: int = Field(default=config.TOP_K_HYBRID, ge=1, le=50)
    min_score: float = 0.0
    mode: str = "hybrid"  # hybrid | lexical | dense


# ---------------------------------------------------------------------------
# Knowledge bases
# ---------------------------------------------------------------------------
@router.get("/knowledge-bases")
def list_knowledge_bases(subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        kbs = store.list_kbs(conn)
        out = []
        for kb in kbs:
            docs = store.list_documents(conn, store.kb_scope(kb["id"]))
            out.append(
                {
                    "id": kb["id"],
                    "name": kb["name"],
                    "description": kb.get("description"),
                    "createdAt": kb.get("created_at"),
                    "documentCount": len(docs),
                }
            )
        return {"knowledgeBases": out}
    finally:
        conn.close()


@router.post("/knowledge-bases")
def create_knowledge_base(
    payload: CreateKbRequest, subject: str = Depends(get_current_subject)
) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        kb_id = store.create_kb(
            conn,
            name=payload.name.strip(),
            description=(payload.description or None),
            embedding_model=config.EMBEDDING_MODEL,
        )
        return {"id": kb_id, "name": payload.name.strip()}
    finally:
        conn.close()


@router.patch("/knowledge-bases/{kb_id}")
def update_knowledge_base(
    kb_id: str, payload: UpdateKbRequest, subject: str = Depends(get_current_subject)
) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        sets, params = [], []
        if payload.name is not None:
            sets.append("name=?")
            params.append(payload.name.strip())
        if payload.description is not None:
            sets.append("description=?")
            params.append(payload.description or None)
        if sets:
            params.append(kb_id)
            conn.execute(f"UPDATE knowledge_bases SET {', '.join(sets)} WHERE id=?", params)
            conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@router.delete("/knowledge-bases/{kb_id}")
def delete_knowledge_base(
    kb_id: str, subject: str = Depends(get_current_subject)
) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        store.delete_kb(conn, kb_id)
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Documents (KB + per-thread)
# ---------------------------------------------------------------------------
@router.post("/knowledge-bases/{kb_id}/documents")
async def upload_kb_document(
    kb_id: str,
    file: UploadFile = File(...),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
    finally:
        conn.close()
    stored_path, filename = _save_upload(file)
    document_id, job_id = ingestion.start_ingestion(
        store.kb_scope(kb_id), kb_id, None, filename, stored_path
    )
    return {"documentId": document_id, "jobId": job_id, "filename": filename}


@router.get("/knowledge-bases/{kb_id}/documents")
def list_kb_documents(kb_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        docs = store.list_documents(conn, store.kb_scope(kb_id))
        return {"documents": [_doc_view(d) for d in docs]}
    finally:
        conn.close()


@router.post("/threads/{thread_id}/documents")
async def upload_thread_document(
    thread_id: str,
    file: UploadFile = File(...),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    stored_path, filename = _save_upload(file)
    document_id, job_id = ingestion.start_ingestion(
        store.thread_scope(thread_id), None, thread_id, filename, stored_path
    )
    return {"documentId": document_id, "jobId": job_id, "filename": filename}


@router.get("/threads/{thread_id}/documents")
def list_thread_documents(
    thread_id: str, subject: str = Depends(get_current_subject)
) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        docs = store.list_documents(conn, store.thread_scope(thread_id))
        return {"documents": [_doc_view(d) for d in docs]}
    finally:
        conn.close()


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        if store.get_document(conn, document_id) is None:
            raise HTTPException(status_code=404, detail="Document not found")
        store.delete_document(conn, document_id)
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Ingestion jobs
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}")
def job_status(job_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    row = ingestion.get_job_status(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": row["id"],
        "documentId": row["document_id"],
        "status": row["status"],
        "stage": row.get("stage"),
        "progress": row.get("progress") or 0.0,
        "error": row.get("error"),
    }


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, subject: str = Depends(get_current_subject)) -> StreamingResponse:
    _require_rag()

    def gen():
        try:
            for event in ingestion.job_events(job_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Direct search (UI / debug)
# ---------------------------------------------------------------------------
@router.post("/search")
def search(payload: SearchRequest, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    if payload.kb_id:
        scope = store.kb_scope(payload.kb_id)
    elif payload.thread_id:
        scope = store.thread_scope(payload.thread_id)
    else:
        raise HTTPException(status_code=400, detail="Provide kb_id or thread_id")

    conn = rag_db.get_connection()
    try:
        if payload.mode == "lexical":
            hits = retrieval.retrieve_lexical(conn, scope, payload.query, payload.top_k)
        elif payload.mode == "dense":
            hits = retrieval.retrieve_dense(conn, scope, payload.query, payload.top_k)
        else:
            hits = retrieval.retrieve_hybrid(conn, scope, payload.query, k=payload.top_k)
        hits = retrieval.filter_min_score(hits, payload.min_score)
        rows = store.chunks_by_id(conn, [h.chunk_id for h in hits])
        results = []
        for h in hits:
            r = rows.get(h.chunk_id)
            if r is None:
                continue
            results.append(
                {
                    "chunkId": h.chunk_id,
                    "documentId": r["document_id"],
                    "filename": r["filename"],
                    "page": r["page_number"],
                    "score": h.score,
                    "text": r["text"],
                }
            )
        return {"results": results}
    finally:
        conn.close()
