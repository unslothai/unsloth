# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP API for the RAG engine: KB CRUD, uploads, SSE ingestion, search.

Single-tenant: the subject gates access, not data. Without sqlite-vec the router
mounts but every endpoint returns 503.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.rag import config, ingestion, retrieval, store
from storage import rag_db
from utils.paths import ensure_dir, rag_uploads_root

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_rag() -> None:
    if not rag_db.RAG_AVAILABLE:
        raise HTTPException(
            status_code = 503,
            detail = "RAG is unavailable: the sqlite-vec extension could not be loaded.",
        )


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "").strip() or "document"
    base = _SAFE.sub("_", base)
    return base[:200]


def _save_upload(file: UploadFile) -> tuple[str, str]:
    """Persist an upload; returns (stored_path, filename)."""
    filename = _sanitize_filename(file.filename or "document")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.UPLOAD_EXTS:
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported file type '{ext}'. Allowed: {sorted(config.UPLOAD_EXTS)}",
        )
    uploads = ensure_dir(rag_uploads_root())
    stored_path = str(uploads / f"{uuid.uuid4().hex}{ext}")
    size = 0
    cap = config.MAX_UPLOAD_BYTES
    too_big = False
    with open(stored_path, "wb") as out:
        while True:
            block = file.file.read(1 << 20)
            if not block:
                break
            size += len(block)
            if cap and size > cap:
                too_big = True
                break
            out.write(block)
    if too_big:
        os.remove(stored_path)
        raise HTTPException(
            status_code = 413,
            detail = f"File exceeds the {cap // (1024 * 1024)} MB upload limit.",
        )
    if size == 0:
        os.remove(stored_path)
        raise HTTPException(status_code = 400, detail = "Uploaded file is empty.")
    return stored_path, filename


def _remove_stored_upload(stored_path: str | None) -> None:
    """Best-effort cleanup for files saved by _save_upload."""
    if not stored_path:
        return
    try:
        uploads = os.path.realpath(str(rag_uploads_root()))
        target = os.path.realpath(stored_path)
        if os.path.isfile(target) and os.path.commonpath([uploads, target]) == uploads:
            os.remove(target)
    except Exception:  # noqa: BLE001 - DB/index deletion has already succeeded.
        logger.warning("failed to remove RAG upload %s", stored_path, exc_info = True)


def _doc_view(row: dict) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "status": row["status"],
        "error": row.get("error"),
        "numChunks": row.get("num_chunks") or 0,
        "kbId": row.get("kb_id"),
        "threadId": row.get("thread_id"),
        "projectId": row.get("project_id"),
        "createdAt": row.get("created_at"),
    }


class CreateKbRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 200)
    description: str | None = None


class UpdateKbRequest(BaseModel):
    name: str | None = Field(default = None, max_length = 200)
    description: str | None = None


class SearchRequest(BaseModel):
    query: str
    kb_id: str | None = None
    thread_id: str | None = None
    project_id: str | None = None
    top_k: int = Field(default = config.TOP_K_HYBRID, ge = 1, le = 50)
    min_score: float = 0.0
    mode: str = "hybrid"  # hybrid | lexical | dense


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
            name = payload.name.strip(),
            description = (payload.description or None),
            embedding_model = config.effective_embedding_model(),
        )
        return {"id": kb_id, "name": payload.name.strip()}
    finally:
        conn.close()


@router.patch("/knowledge-bases/{kb_id}")
def update_knowledge_base(
    kb_id: str,
    payload: UpdateKbRequest,
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code = 404, detail = "Knowledge base not found")
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
def delete_knowledge_base(kb_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code = 404, detail = "Knowledge base not found")
        store.delete_kb(conn, kb_id)
        return {"ok": True}
    finally:
        conn.close()


@router.post("/knowledge-bases/{kb_id}/documents")
async def upload_kb_document(
    kb_id: str,
    file: UploadFile = File(...),
    ocr: bool | None = Form(None),
    caption: bool | None = Form(None),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code = 404, detail = "Knowledge base not found")
    finally:
        conn.close()
    stored_path, filename = _save_upload(file)
    document_id, job_id = ingestion.start_ingestion(
        store.kb_scope(kb_id), kb_id, None, filename, stored_path, ocr = ocr, caption = caption
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
    ocr: bool | None = Form(None),
    caption: bool | None = Form(None),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    stored_path, filename = _save_upload(file)
    document_id, job_id = ingestion.start_ingestion(
        store.thread_scope(thread_id),
        None,
        thread_id,
        filename,
        stored_path,
        ocr = ocr,
        caption = caption,
    )
    return {"documentId": document_id, "jobId": job_id, "filename": filename}


@router.get("/threads/{thread_id}/documents")
def list_thread_documents(thread_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        docs = store.list_documents(conn, store.thread_scope(thread_id))
        return {"documents": [_doc_view(d) for d in docs]}
    finally:
        conn.close()


@router.post("/projects/{project_id}/documents")
async def upload_project_document(
    project_id: str,
    file: UploadFile = File(...),
    ocr: bool | None = Form(None),
    caption: bool | None = Form(None),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    from storage.studio_db import get_chat_project

    if get_chat_project(project_id) is None:
        raise HTTPException(status_code = 404, detail = "Project not found")
    stored_path, filename = _save_upload(file)
    document_id, job_id = ingestion.start_ingestion(
        store.project_scope(project_id),
        None,
        None,
        filename,
        stored_path,
        project_id = project_id,
        ocr = ocr,
        caption = caption,
    )
    return {"documentId": document_id, "jobId": job_id, "filename": filename}


@router.get("/projects/{project_id}/documents")
def list_project_documents(project_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        docs = store.list_documents(conn, store.project_scope(project_id))
        return {"documents": [_doc_view(d) for d in docs]}
    finally:
        conn.close()


@router.get("/documents")
def list_all_uploaded_documents(subject: str = Depends(get_current_subject)) -> dict:
    """Every uploaded file across chats, projects, and knowledge bases (settings
    Data tab)."""
    _require_rag()
    conn = rag_db.get_connection()
    try:
        docs = store.list_all_documents(conn)
        kb_names = {kb["id"]: kb["name"] for kb in store.list_kbs(conn)}
    finally:
        conn.close()

    from storage.studio_db import list_chat_projects

    project_names = {p["id"]: p["name"] for p in list_chat_projects(include_archived = True)}

    out = []
    for doc in docs:
        view = _doc_view(doc)
        stored_path = doc.get("stored_path")
        size = None
        if stored_path:
            try:
                size = os.path.getsize(stored_path)
            except OSError:
                size = None
        view["sizeBytes"] = size
        view["kbName"] = kb_names.get(doc.get("kb_id"))
        view["projectName"] = project_names.get(doc.get("project_id"))
        out.append(view)
    return {"documents": out}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        doc = store.get_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code = 404, detail = "Document not found")
        store.delete_document(conn, document_id)
        _remove_stored_upload(doc.get("stored_path"))
        return {"ok": True}
    finally:
        conn.close()


@router.get("/jobs/{job_id}")
def job_status(job_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    row = ingestion.get_job_status(job_id)
    if row is None:
        raise HTTPException(status_code = 404, detail = "Job not found")
    return {
        "id": row["id"],
        "documentId": row["document_id"],
        "status": row["status"],
        "stage": row.get("stage"),
        "progress": row.get("progress") or 0.0,
        "error": row.get("error"),
        "numChunks": row.get("num_chunks") or 0,
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
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/search")
def search(payload: SearchRequest, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    if payload.kb_id:
        scope = store.kb_scope(payload.kb_id)
    else:
        scopes = []
        if payload.project_id:
            scopes.append(store.project_scope(payload.project_id))
        if payload.thread_id:
            scopes.append(store.thread_scope(payload.thread_id))
        if not scopes:
            raise HTTPException(status_code = 400, detail = "Provide kb_id, project_id, or thread_id")
        scope = scopes[0] if len(scopes) == 1 else scopes

    conn = rag_db.get_connection()
    try:
        if payload.mode == "lexical":
            hits = retrieval.retrieve_lexical(conn, scope, payload.query, payload.top_k)
        elif payload.mode == "dense":
            hits = retrieval.retrieve_dense(conn, scope, payload.query, payload.top_k)
        else:
            hits = retrieval.retrieve_hybrid(conn, scope, payload.query, k = payload.top_k)
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


# Per-process secret so pdf.js range requests fetch the file without a bearer
# header; tokens only work on this server instance.
_PREVIEW_SECRET = secrets.token_bytes(32)
_PREVIEW_TTL = 600  # seconds

_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".markdown": "text/markdown; charset=utf-8",
    # Served as plain text, never text/html: an uploaded HTML document rendered
    # same-origin would execute its scripts with access to the app's storage.
    ".html": "text/plain; charset=utf-8",
    ".htm": "text/plain; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _sign_document(document_id: str) -> str:
    exp = int(time.time()) + _PREVIEW_TTL
    payload = f"{document_id}.{exp}"
    sig = hmac.new(_PREVIEW_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_document_token(token: str) -> str | None:
    try:
        document_id, exp_s, sig = token.rsplit(".", 2)
    except ValueError:
        return None
    expected = hmac.new(
        _PREVIEW_SECRET, f"{document_id}.{exp_s}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        if int(exp_s) < int(time.time()):
            return None
    except ValueError:
        return None
    return document_id


@router.get("/documents/{document_id}/preview-target")
def preview_target(
    document_id: str,
    chunk_id: str | None = Query(default = None),
    subject: str = Depends(get_current_subject),
) -> dict:
    """Resolve a citation to filename, page, and highlight regions."""
    _require_rag()
    conn = rag_db.get_connection()
    try:
        doc = store.get_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code = 404, detail = "Document not found")
        ext = os.path.splitext(doc["filename"])[1].lower()
        out = {
            "documentId": document_id,
            "filename": doc["filename"],
            "mediaKind": "pdf" if ext == ".pdf" else "text",
            "targetPage": None,
            "pdfRegions": [],
            "text": None,
        }
        if chunk_id:
            row = conn.execute(
                "SELECT text, page_number, pdf_regions_json FROM chunks WHERE id=?",
                (chunk_id,),
            ).fetchone()
            if row is not None:
                out["text"] = row["text"]
                out["targetPage"] = row["page_number"]
                if row["pdf_regions_json"]:
                    try:
                        out["pdfRegions"] = json.loads(row["pdf_regions_json"])
                    except Exception:
                        out["pdfRegions"] = []
        return out
    finally:
        conn.close()


@router.get("/documents/{document_id}/file-url")
def document_file_url(document_id: str, subject: str = Depends(get_current_subject)) -> dict:
    """Mint a short-lived signed URL for the source file."""
    _require_rag()
    conn = rag_db.get_connection()
    try:
        doc = store.get_document(conn, document_id)
        if doc is None or not doc.get("stored_path"):
            raise HTTPException(status_code = 404, detail = "Document file not available")
    finally:
        conn.close()
    token = _sign_document(document_id)
    return {"url": f"/api/rag/documents/{document_id}/file-signed?token={token}"}


@router.get("/documents/{document_id}/file-signed", response_model = None)
def document_file_signed(document_id: str, token: str = Query(...)) -> FileResponse:
    """Serve the source file gated by the HMAC token (no bearer) so pdf.js range
    requests work."""
    _require_rag()
    signed_id = _verify_document_token(token)
    if signed_id != document_id:
        raise HTTPException(status_code = 401, detail = "Invalid or expired token")
    conn = rag_db.get_connection()
    try:
        doc = store.get_document(conn, document_id)
    finally:
        conn.close()
    stored_path = (doc or {}).get("stored_path")
    if not doc or not stored_path or not os.path.isfile(stored_path):
        raise HTTPException(status_code = 404, detail = "Document file not found")
    # Confine to the uploads root (defense in depth).
    uploads = os.path.realpath(str(rag_uploads_root()))
    if not os.path.realpath(stored_path).startswith(uploads):
        raise HTTPException(status_code = 403, detail = "Forbidden")
    ext = os.path.splitext(doc["filename"])[1].lower()
    return FileResponse(
        stored_path,
        media_type = _CONTENT_TYPES.get(ext, "application/octet-stream"),
        filename = doc["filename"],
    )
