# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP API for the RAG engine: KB CRUD, uploads, SSE ingestion, search.

Single-tenant: the subject gates access, not data.

Without a working sqlite-vec the router still mounts, and one contract covers every
endpoint. The KB list is polled, so it answers 200 with an empty list carrying an
availability marker, which is what lets a client tell an empty store from a machine
where RAG cannot run. Every other endpoint answers 503 stating the same reason.
Nothing logs a traceback for it: the condition is fixed for the session and rag_db
warns about it exactly once.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.rag import config, folder_sync, ingestion, retrieval, store
from storage import rag_db
from utils.paths import ensure_dir, rag_uploads_root

logger = logging.getLogger(__name__)

router = APIRouter()


_UNAVAILABLE_DETAIL = "RAG is unavailable: the sqlite-vec extension could not be loaded."


def _require_rag() -> None:
    """Gate an endpoint on RAG being runnable here.

    Covers both halves of unavailable: sqlite-vec never imported, and it imported but
    its native library will not load. 503 with a stated reason rather than the 500 plus
    traceback a raising connection would produce, and rag_db's warn-once keeps the log
    quiet however often this fires.
    """
    if not rag_db.rag_available():
        raise HTTPException(status_code = 503, detail = _UNAVAILABLE_DETAIL)


@contextmanager
def _rag_unavailable_as_503(cleanup_path: str | None = None) -> Iterator[None]:
    """Report RagExtensionUnavailable as the same 503, wherever it is raised.

    _require_rag() has normally answered for the session already; this closes the window
    where the very first request is the one that discovers the missing library, and it
    reaches the connections ingestion opens for itself. ``cleanup_path`` removes an
    upload that was saved before the failure, so nothing is orphaned in the uploads
    root. Real database errors are left alone.
    """
    try:
        yield
    except rag_db.RagExtensionUnavailable as exc:
        _remove_stored_upload(cleanup_path)
        raise HTTPException(status_code = 503, detail = _UNAVAILABLE_DETAIL) from exc


def _rag_connection() -> sqlite3.Connection:
    """rag_db.get_connection() with the unavailable case reported as 503."""
    with _rag_unavailable_as_503():
        return rag_db.get_connection()


def _availability(available: bool) -> dict:
    """Availability marker carried by the KB list, the one response that degrades
    rather than erroring.

    Additive: a client that only reads the list is unaffected, one that reads this can
    say "RAG cannot run here" instead of showing an empty page that looks ready to use
    and offering a Create that can only 503.
    """
    return {
        "ragAvailable": available,
        "ragUnavailableReason": None if available else _UNAVAILABLE_DETAIL,
    }


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "").strip() or "document"
    base = _SAFE.sub("_", base)
    if len(base) <= 200:
        return base
    # Trim the stem, not the extension: _save_upload gates on the extension, so
    # a plain truncation would reject a long-named .txt as "unsupported".
    stem, ext = os.path.splitext(base)
    if not ext or len(ext) > 32:
        return base[:200]
    return stem[: 200 - len(ext)] + ext


def _persist_upload_stream(source, filename: str, *, empty_detail: str) -> tuple[str, str]:
    """Copy a validated document stream into the managed uploads root."""
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
    try:
        with open(stored_path, "wb") as out:
            while True:
                block = source.read(1 << 20)
                if not block:
                    break
                size += len(block)
                if cap and size > cap:
                    break
                out.write(block)
    except OSError:
        _remove_stored_upload(stored_path)
        raise
    if cap and size > cap:
        _remove_stored_upload(stored_path)
        raise HTTPException(
            status_code = 413,
            detail = f"File exceeds the {cap // (1024 * 1024)} MB upload limit.",
        )
    if size == 0:
        _remove_stored_upload(stored_path)
        raise HTTPException(status_code = 400, detail = empty_detail)
    return stored_path, filename


def _save_upload(file: UploadFile) -> tuple[str, str]:
    """Persist a browser upload; returns (stored_path, filename)."""
    filename = _sanitize_filename(file.filename or "document")
    return _persist_upload_stream(
        file.file,
        filename,
        empty_detail = "Uploaded file is empty.",
    )


def _save_native_path_upload(lease: str) -> tuple[str, str]:
    """Persist a desktop drop; returns (stored_path, filename).

    The webview never gets to name a path directly: Rust signs the path it saw and we
    re-verify + re-stat that grant here before reading a byte.
    """
    from utils.native_path_leases import NativePathLeaseError, verify_native_path_lease

    try:
        grant = verify_native_path_lease(
            lease,
            operation = "attach",
            expected_kind = "attachment",
            expected_path_type = "file",
            allowed_suffixes = sorted(config.UPLOAD_EXTS),
        )
    except NativePathLeaseError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc

    filename = _sanitize_filename(grant.canonical_path.name)
    try:
        with open(grant.canonical_path, "rb") as source:
            return _persist_upload_stream(
                source,
                filename,
                empty_detail = "Dropped file is empty.",
            )
    except OSError as exc:
        raise HTTPException(status_code = 400, detail = "Dropped file could not be read.") from exc


def _resolve_document_upload(
    file: UploadFile | None, native_path_lease: str | None
) -> tuple[str, str]:
    if native_path_lease:
        return _save_native_path_upload(native_path_lease)
    if file is None:
        raise HTTPException(status_code = 400, detail = "No file was provided.")
    return _save_upload(file)


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


def _is_managed_preview_path(stored_path: str) -> bool:
    uploads = os.path.realpath(str(rag_uploads_root()))
    try:
        common = os.path.commonpath([uploads, os.path.realpath(stored_path)])
        return os.path.normcase(common) == os.path.normcase(uploads)
    except ValueError:
        return False


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
        "linkedFolderId": row.get("linked_folder_id"),
        "managed": bool(row.get("linked_folder_id")),
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


class LinkFolderRequest(BaseModel):
    name: str | None = Field(default = None, alias = "displayName", max_length = 200)
    auto_sync: bool = Field(default = True, alias = "autoSync")
    native_path_lease: str = Field(alias = "nativePathLease", min_length = 1)


class UpdateFolderRequest(BaseModel):
    name: str | None = Field(default = None, max_length = 200)
    auto_sync: bool | None = Field(default = None, alias = "autoSync")


def _resolve_linked_folder_path(native_path_lease: str, *, verifier = None) -> str:
    """Resolve a desktop grant; the injectable verifier keeps resolution unit-testable."""
    from utils.native_path_leases import NativePathLeaseError, verify_native_path_lease

    verify = verifier or verify_native_path_lease
    try:
        grant = verify(
            native_path_lease,
            operation = "link-documents",
            expected_kind = "document-folder",
            expected_path_type = "directory",
        )
        return folder_sync.validate_folder_path(str(grant.canonical_path))
    except NativePathLeaseError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc


def _folder_view(row: dict) -> dict:
    status = (
        "syncing"
        if row["status"] == "syncing"
        else "error"
        if row["status"] in {"error", "retired"}
        else "idle"
    )
    return {
        "id": row["id"],
        "displayName": row["name"],
        "scopeType": row["scope_type"],
        "scopeId": row["scope_id"],
        "status": status,
        "error": row.get("last_error"),
        "lastSyncedAt": row.get("last_scan_at"),
        "documentCount": row.get("file_count", 0),
        "activeJobId": row.get("active_job_id"),
        "scopeName": row.get("scope_name"),
        "createdAt": row["created_at"],
    }


def _create_linked_folder(scope_type: str, scope_id: str, payload: LinkFolderRequest) -> dict:
    path = _resolve_linked_folder_path(payload.native_path_lease)
    try:
        folder, job_id = folder_sync.create_folder_with_sync(
            scope_type = scope_type,
            scope_id = scope_id,
            path = path,
            name = payload.name,
            auto_sync = payload.auto_sync,
        )
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    job = folder_sync.get_job(job_id)
    return {"linkedFolder": _folder_view(folder), "job": _folder_job_view(job)}


@router.get("/knowledge-bases")
def list_knowledge_bases(subject: str = Depends(get_current_subject)) -> dict:
    try:
        conn = rag_db.get_connection()
    except rag_db.RagExtensionUnavailable:
        # RAG_AVAILABLE only covers the import; the native library can still fail to
        # load per connection (a missing vec0 binary in the venv). The UI polls this
        # list, so 500ing here costs a traceback every few seconds for a condition that
        # never changes within a session. rag_db has warned once; an empty list is what
        # a machine without RAG has anyway. The marker is what keeps that honest: it is
        # the difference between "no knowledge bases yet" and "RAG cannot run here", and
        # without it the empty page looks ready to use. Only the unavailable case
        # degrades: a locked or corrupt database still raises.
        return {"knowledgeBases": [], **_availability(False)}
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
        return {"knowledgeBases": out, **_availability(True)}
    finally:
        conn.close()


@router.post("/knowledge-bases")
def create_knowledge_base(
    payload: CreateKbRequest, subject: str = Depends(get_current_subject)
) -> dict:
    _require_rag()
    conn = _rag_connection()
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
    conn = _rag_connection()
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
    conn = _rag_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code = 404, detail = "Knowledge base not found")
        scope = store.kb_scope(kb_id)
        folders = folder_sync.retire_scope(scope)
        stored_paths = [
            row["stored_path"]
            for row in conn.execute("SELECT stored_path FROM documents WHERE scope=?", (scope,))
        ]
        for folder in folders:
            try:
                folder_sync.delete_folder(folder["id"])
            except Exception:
                logger.warning(
                    "failed to delete retired linked folder %s", folder["id"], exc_info = True
                )
        store.delete_kb(conn, kb_id)
        for stored_path in stored_paths:
            _remove_stored_upload(stored_path)
        return {"ok": True}
    finally:
        conn.close()


def _raise_if_scope_retired(scope: str) -> None:
    if folder_sync.scope_retired(scope):
        raise HTTPException(status_code = 409, detail = "Knowledge base is being deleted")


@router.post("/knowledge-bases/{kb_id}/documents")
async def upload_kb_document(
    kb_id: str,
    file: UploadFile | None = File(None),
    native_path_lease: str | None = Form(None, alias = "nativePathLease"),
    ocr: bool | None = Form(None),
    caption: bool | None = Form(None),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    conn = _rag_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code = 404, detail = "Knowledge base not found")
    finally:
        conn.close()
    scope = store.kb_scope(kb_id)
    _raise_if_scope_retired(scope)
    stored_path, filename = _resolve_document_upload(file, native_path_lease)
    try:
        with folder_sync.scope_lock(scope):
            _raise_if_scope_retired(scope)
            with _rag_unavailable_as_503(stored_path):
                document_id, job_id = ingestion.start_ingestion(
                    scope, kb_id, None, filename, stored_path, ocr = ocr, caption = caption
                )
    except HTTPException:
        _remove_stored_upload(stored_path)
        raise
    return {"documentId": document_id, "jobId": job_id, "filename": filename}


@router.get("/knowledge-bases/{kb_id}/documents")
def list_kb_documents(kb_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = _rag_connection()
    try:
        docs = store.list_documents(conn, store.kb_scope(kb_id))
        return {"documents": [_doc_view(d) for d in docs]}
    finally:
        conn.close()


@router.post("/knowledge-bases/{kb_id}/linked-folders")
def link_kb_folder(
    kb_id: str,
    payload: LinkFolderRequest,
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    conn = rag_db.get_connection()
    try:
        if store.get_kb(conn, kb_id) is None:
            raise HTTPException(status_code = 404, detail = "Knowledge base not found")
    finally:
        conn.close()
    return _create_linked_folder("knowledge_base", kb_id, payload)


@router.post("/threads/{thread_id}/documents")
async def upload_thread_document(
    thread_id: str,
    file: UploadFile | None = File(None),
    native_path_lease: str | None = Form(None, alias = "nativePathLease"),
    ocr: bool | None = Form(None),
    caption: bool | None = Form(None),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    stored_path, filename = _resolve_document_upload(file, native_path_lease)
    with _rag_unavailable_as_503(stored_path):
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
    conn = _rag_connection()
    try:
        docs = store.list_documents(conn, store.thread_scope(thread_id))
        return {"documents": [_doc_view(d) for d in docs]}
    finally:
        conn.close()


@router.post("/projects/{project_id}/documents")
async def upload_project_document(
    project_id: str,
    file: UploadFile | None = File(None),
    native_path_lease: str | None = Form(None, alias = "nativePathLease"),
    ocr: bool | None = Form(None),
    caption: bool | None = Form(None),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    from storage.studio_db import get_chat_project

    if get_chat_project(project_id) is None:
        raise HTTPException(status_code = 404, detail = "Project not found")
    stored_path, filename = _resolve_document_upload(file, native_path_lease)
    with _rag_unavailable_as_503(stored_path):
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
    conn = _rag_connection()
    try:
        docs = store.list_documents(conn, store.project_scope(project_id))
        return {"documents": [_doc_view(d) for d in docs]}
    finally:
        conn.close()


@router.post("/projects/{project_id}/linked-folders")
def link_project_folder(
    project_id: str,
    payload: LinkFolderRequest,
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    from storage.studio_db import get_chat_project

    if get_chat_project(project_id) is None:
        raise HTTPException(status_code = 404, detail = "Project not found")
    return _create_linked_folder("project", project_id, payload)


@router.get("/linked-folders")
def list_linked_folders(
    scope_type: str | None = Query(default = None),
    scope_id: str | None = Query(default = None),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    if bool(scope_type) != bool(scope_id):
        raise HTTPException(
            status_code = 400, detail = "scope_type and scope_id must be provided together"
        )
    if scope_type:
        if scope_type not in {"knowledge_base", "project"}:
            raise HTTPException(status_code = 400, detail = "Unsupported linked-folder scope")
        scope = (
            store.kb_scope(scope_id)
            if scope_type == "knowledge_base"
            else store.project_scope(scope_id)
        )
        rows = folder_sync.list_folders(scope)
    else:
        conn = rag_db.get_connection()
        try:
            scopes = [
                row["scope"] for row in conn.execute("SELECT DISTINCT scope FROM linked_folders")
            ]
            kb_names = {row["id"]: row["name"] for row in store.list_kbs(conn)}
        finally:
            conn.close()
        from storage.studio_db import list_chat_projects

        project_names = {
            row["id"]: row["name"] for row in list_chat_projects(include_archived = True)
        }
        rows = [row for scope in scopes for row in folder_sync.list_folders(scope)]
        for row in rows:
            names = kb_names if row["scope_type"] == "knowledge_base" else project_names
            row["scope_name"] = names.get(row["scope_id"])
    return {"linkedFolders": [_folder_view(row) for row in rows]}


@router.patch("/linked-folders/{folder_id}")
def update_linked_folder(
    folder_id: str,
    payload: UpdateFolderRequest,
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    try:
        row = folder_sync.update_folder(folder_id, name = payload.name, auto_sync = payload.auto_sync)
    except KeyError as exc:
        raise HTTPException(status_code = 404, detail = "Linked folder not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    return {"linkedFolder": _folder_view(row)}


@router.delete("/linked-folders/{folder_id}")
def unlink_folder(
    folder_id: str,
    remove_index: bool = Query(default = True),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    if not folder_sync.delete_folder(folder_id, remove_index = remove_index):
        raise HTTPException(status_code = 404, detail = "Linked folder not found")
    return {"ok": True}


@router.post("/linked-folders/{folder_id}/sync")
def sync_folder(folder_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    try:
        return {"job": _folder_job_view(folder_sync.get_job(folder_sync.request_sync(folder_id)))}
    except KeyError as exc:
        raise HTTPException(status_code = 404, detail = "Linked folder not found") from exc


@router.post("/linked-folders/{folder_id}/rebuild")
def rebuild_folder(folder_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    try:
        return {
            "job": _folder_job_view(
                folder_sync.get_job(folder_sync.request_sync(folder_id, rebuild = True))
            )
        }
    except KeyError as exc:
        raise HTTPException(status_code = 404, detail = "Linked folder not found") from exc


@router.get("/documents")
def list_all_uploaded_documents(subject: str = Depends(get_current_subject)) -> dict:
    """Every uploaded file across chats, projects, and knowledge bases (settings
    Data tab)."""
    _require_rag()
    conn = _rag_connection()
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
    conn = _rag_connection()
    try:
        doc = store.get_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code = 404, detail = "Document not found")
        if doc.get("linked_folder_id"):
            raise HTTPException(
                status_code = 409,
                detail = "Linked-folder documents are managed by folder synchronization",
            )
        store.delete_document(conn, document_id)
        _remove_stored_upload(doc.get("stored_path"))
        return {"ok": True}
    finally:
        conn.close()


@router.get("/jobs/{job_id}")
def job_status(job_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    with _rag_unavailable_as_503():
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


def _folder_job_view(row: dict) -> dict:
    processed = (
        (row.get("added") or 0)
        + (row.get("changed") or 0)
        + (row.get("deleted") or 0)
        + (row.get("failed") or 0)
    )
    return {
        "id": row["id"],
        "linkedFolderId": row["folder_id"],
        "mode": row["kind"],
        "status": row["status"],
        "stage": row.get("stage"),
        "progress": row.get("progress") or 0.0,
        "discoveredFiles": row.get("discovered") or 0,
        "processedFiles": processed,
        "indexedFiles": (row.get("added") or 0) + (row.get("changed") or 0),
        "removedFiles": row.get("deleted") or 0,
        "failedFiles": row.get("failed") or 0,
        "error": row.get("error"),
        "createdAt": row.get("created_at"),
        "completedAt": row.get("completed_at"),
    }


@router.get("/linked-folder-jobs/{job_id}")
def folder_job_status(job_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    row = folder_sync.get_job(job_id)
    if row is None:
        raise HTTPException(status_code = 404, detail = "Folder sync job not found")
    return _folder_job_view(row)


@router.get("/linked-folder-jobs/{job_id}/events")
def folder_job_events(
    job_id: str, subject: str = Depends(get_current_subject)
) -> StreamingResponse:
    _require_rag()
    if folder_sync.get_job(job_id) is None:
        raise HTTPException(status_code = 404, detail = "Folder sync job not found")

    def gen():
        for event in folder_sync.job_events(job_id):
            view = _folder_job_view(event)
            view["type"] = (
                "complete"
                if event["status"] == "completed"
                else "error"
                if event["status"] == "failed"
                else "progress"
            )
            yield f"data: {json.dumps(view)}\n\n"
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

    conn = _rag_connection()
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
    conn = _rag_connection()
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
    conn = _rag_connection()
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
    # Token first: this is the one endpoint with no bearer, and _require_rag() now opens
    # a connection on its first call, which is not work an unverified token should buy.
    signed_id = _verify_document_token(token)
    if signed_id != document_id:
        raise HTTPException(status_code = 401, detail = "Invalid or expired token")
    _require_rag()
    conn = _rag_connection()
    try:
        doc = store.get_document(conn, document_id)
    finally:
        conn.close()
    stored_path = (doc or {}).get("stored_path")
    if not doc or not stored_path or not os.path.isfile(stored_path):
        raise HTTPException(status_code = 404, detail = "Document file not found")
    # Confine to the uploads root (defense in depth).
    if not _is_managed_preview_path(stored_path):
        raise HTTPException(status_code = 403, detail = "Forbidden")
    ext = os.path.splitext(doc["filename"])[1].lower()
    return FileResponse(
        stored_path,
        media_type = _CONTENT_TYPES.get(ext, "application/octet-stream"),
        filename = doc["filename"],
    )
