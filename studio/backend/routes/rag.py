# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP API for the RAG engine: KB CRUD, uploads, SSE ingestion, search. Single-tenant: the subject gates
access, not data.

Without a working sqlite-vec the router still mounts, and one contract covers every endpoint. The KB list is
polled, so it answers 200 with an empty list carrying an availability marker, which is what lets a client tell
an empty store from a machine where RAG cannot run. Every other endpoint answers 503 stating the same reason.
Nothing logs a traceback for it: the condition is fixed for the session and rag_db warns about it exactly once.
"""

from __future__ import annotations

from core.training.account_jobs import account_path
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import time
import unicodedata
import uuid
from contextlib import contextmanager
from typing import Annotated, Iterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject, request_admitted_without_credential
from core.rag import config, folder_sync, ingestion, retrieval, store
from storage import rag_db
from utils.account_context import (
    OWNER,
    AccountContext,
    bind_account,
    current_account,
    reset_account,
)
from utils.paths import ensure_dir, rag_uploads_root

logger = logging.getLogger(__name__)

router = APIRouter()


_UNAVAILABLE_DETAIL = "RAG is unavailable: the sqlite-vec extension could not be loaded."


def _require_rag() -> None:
    """Gate an endpoint on RAG being runnable here. Covers both halves of unavailable: sqlite-vec never
    imported, and it imported but its native library will not load. 503 with a stated reason rather
    than the 500 plus traceback a raising connection would produce, and rag_db's warn-once keeps the
    log quiet however often this fires."""
    if not rag_db.rag_available():
        raise HTTPException(status_code = 503, detail = _UNAVAILABLE_DETAIL)


@contextmanager
def _rag_unavailable_as_503(cleanup_path: str | None = None) -> Iterator[None]:
    """Report RagExtensionUnavailable as the same 503, wherever it is raised. _require_rag() has
    normally answered for the session already; this closes the window where the very first request
    is the one that discovers the missing library, and it reaches the connections ingestion opens
    for itself. ``cleanup_path`` removes an upload that was saved before the failure, so nothing is
    orphaned in the uploads root. Real database errors are left alone."""
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
    """Availability marker carried by the KB list, the one response that degrades rather than erroring.
    Additive: a client that only reads the list is unaffected, one that reads this can say "RAG cannot run here"
    instead of showing an empty page that looks ready to use and offering a Create that can only 503.
    """
    return {
        "ragAvailable": available,
        "ragUnavailableReason": None if available else _UNAVAILABLE_DETAIL,
    }


def _document_label(name: str) -> str:
    # A display label, never a path: the bytes are stored at uploads/<uuid><ext>.
    base = "".join(
        (" " if ch.isspace() else "")
        if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp")
        else ch
        for ch in name or ""
    )
    # Not \s+: U+3000 and the other Zs spaces belong to the name.
    base = re.sub(r" +", " ", base).strip() or "document"
    if len(base) <= 200:
        return base
    # Trim the stem, not the extension: _save_upload gates on the extension, so
    # a plain truncation would reject a long-named .txt as "unsupported".
    stem, ext = os.path.splitext(base)
    if not ext or len(ext) > 32:
        return base[:200]
    return stem[: 200 - len(ext)] + ext


# Names treated as Windows paths. Each is also a legal POSIX filename; the list stays
# narrow because splitting a real name loses part of it. One leading backslash counts as
# UNC because multipart parsing unescapes "\\" to "\".
_LOOKS_LIKE_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\?|\.{1,2}\\)")


def _sanitize_filename(name: str) -> str:
    """Label a browser upload, whose client-supplied name may carry a path."""
    # Not ntpath.basename: it reads "P:L statement.pdf", how macOS stores a Finder "/",
    # as drive "P:". Classify first: splitting on "/" would strip the drive from "C:/a\b".
    raw = name or ""
    parts = re.split(r"[\\/]", raw) if _LOOKS_LIKE_WINDOWS_PATH.match(raw) else raw.split("/")
    return _document_label(parts[-1])


def _persist_upload_stream(source, filename: str, *, empty_detail: str) -> tuple[str, str, str]:
    """Copy a validated document stream into the managed uploads root.

    Returns ``(stored_path, filename, content_hash)``; the digest spares ingestion a
    second full read of the file.
    """
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
    digest = hashlib.sha256()
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
                digest.update(block)
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
    return stored_path, filename, digest.hexdigest()


def _save_upload(file: UploadFile) -> tuple[str, str, str]:
    """Persist a browser upload; returns (stored_path, filename, content_hash)."""
    filename = _sanitize_filename(file.filename or "document")
    return _persist_upload_stream(
        file.file,
        filename,
        empty_detail = "Uploaded file is empty.",
    )


def _save_native_path_upload(lease: str) -> tuple[str, str, str]:
    """Persist a desktop drop; returns (stored_path, filename, content_hash).

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

    account_path(grant.canonical_path)
    # Path.name is one component, so a "\" in it is part of the name, as in "AC\DC.pdf".
    filename = _document_label(grant.canonical_path.name)
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
) -> tuple[str, str, str]:
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


def _stored_size(stored_path: str | None) -> int | None:
    """Size of a document's stored bytes, or None when it has no readable file.

    A document keeps its row after the upload (or linked-folder snapshot) behind
    it has gone, so a missing path is expected rather than an error.
    """
    if not stored_path:
        return None
    try:
        return os.path.getsize(stored_path)
    except OSError:
        return None


def _doc_view(row: dict, *, with_size: bool = False) -> dict:
    """Wire form of a document row.

    `with_size` is opt-in because it stats each row: the sources panel sorts by
    size and the settings Data tab shows it, but the KB and thread lists render
    neither and are polled every four seconds while something indexes.
    """
    view = {
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
    if with_size:
        view["sizeBytes"] = _stored_size(row.get("stored_path"))
    return view


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
    mode: str = "hybrid"


class LinkFolderRequest(BaseModel):
    name: str | None = Field(default = None, alias = "displayName", max_length = 200)
    auto_sync: bool = Field(default = True, alias = "autoSync")
    native_path_lease: str = Field(alias = "nativePathLease", min_length = 1)


class UpdateFolderRequest(BaseModel):
    name: str | None = Field(default = None, max_length = 200)
    auto_sync: bool | None = Field(default = None, alias = "autoSync")


def _resolve_linked_folder_path(
    native_path_lease: str, *, verifier = None
) -> tuple[str, tuple[int, int]]:
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
        device_id = getattr(grant, "device_id", None)
        file_id = getattr(grant, "file_id", None)
        if device_id is None or file_id is None:
            raise NativePathLeaseError("Native folder grant has no stable identity.")
        return (
            folder_sync.validate_folder_path(str(grant.canonical_path)),
            (device_id, file_id),
        )
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


def _scope_for_owner(scope_type: str, scope_id: str) -> str:
    return (
        store.kb_scope(scope_id)
        if scope_type == "knowledge_base"
        else store.project_scope(scope_id)
    )


def _require_scope_owner(
    scope_type: str,
    scope_id: str,
    conn: sqlite3.Connection | None = None,
) -> None:
    """404 unless the scope's owner still exists. ``conn`` reuses a connection the caller already holds:
    sqlite-vec loads per connection, so opening a second one to read a single row pays that twice.
    """
    if scope_type == "knowledge_base":
        if conn is not None:
            exists = store.get_kb(conn, scope_id) is not None
        else:
            owner_conn = _rag_connection()
            try:
                exists = store.get_kb(owner_conn, scope_id) is not None
            finally:
                owner_conn.close()
        detail = "Knowledge base not found"
    else:
        from storage.studio_db import get_chat_project
        exists = get_chat_project(scope_id) is not None
        detail = "Project not found"
    if not exists:
        raise HTTPException(status_code = 404, detail = detail)


def _require_document_owner(conn: sqlite3.Connection, document: dict) -> None:
    if document.get("kb_id") and store.get_kb(conn, document["kb_id"]) is None:
        raise HTTPException(status_code = 404, detail = "Document not found")
    if document.get("project_id"):
        from storage.studio_db import get_chat_project
        if get_chat_project(document["project_id"]) is None:
            raise HTTPException(status_code = 404, detail = "Document not found")


def _create_linked_folder(scope_type: str, scope_id: str, payload: LinkFolderRequest) -> dict:
    path, signed_identity = _resolve_linked_folder_path(payload.native_path_lease)
    try:
        with folder_sync.scope_lock(_scope_for_owner(scope_type, scope_id)):
            _require_scope_owner(scope_type, scope_id)
            folder, job_id = folder_sync.create_folder_with_sync(
                scope_type = scope_type,
                scope_id = scope_id,
                path = path,
                expected_identity = signed_identity,
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
        # RAG_AVAILABLE only covers the import; the native library can still fail to load per connection (a missing
        # vec0 binary in the venv). The UI polls this list, so 500ing costs a traceback every few seconds for a
        # condition that never changes in a session, and rag_db has warned once. The marker is the difference between
        # "no knowledge bases yet" and "RAG cannot run here". Only the unavailable case degrades: a locked or corrupt
        # database still raises.
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
    with _rag_unavailable_as_503():
        deleted = folder_sync.retire_and_delete_kb(kb_id)
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Knowledge base not found")
    try:
        folder_sync.delete_retired_scope(store.kb_scope(kb_id))
    except Exception:
        logger.warning("failed to delete retired knowledge-base scope %s", kb_id, exc_info = True)
    return {"ok": True}


def _raise_if_scope_retired(scope: str, detail: str = "Knowledge base is being deleted") -> None:
    if folder_sync.scope_retired(scope):
        raise HTTPException(status_code = 409, detail = detail)


# The three upload routes stay sync so FastAPI runs them in the threadpool; their
# copy + start_ingestion work would stall every other request on the event loop.
@router.post("/knowledge-bases/{kb_id}/documents")
def upload_kb_document(
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
    stored_path, filename, content_hash = _resolve_document_upload(file, native_path_lease)
    try:
        with folder_sync.scope_lock(scope):
            _require_scope_owner("knowledge_base", kb_id)
            _raise_if_scope_retired(scope)
            with _rag_unavailable_as_503(stored_path):
                document_id, job_id = ingestion.start_ingestion(
                    scope,
                    kb_id,
                    None,
                    filename,
                    stored_path,
                    ocr = ocr,
                    caption = caption,
                    content_hash = content_hash,
                )
    except Exception:
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
    _require_scope_owner("knowledge_base", kb_id)
    return _create_linked_folder("knowledge_base", kb_id, payload)


# Stays sync for the reason above upload_kb_document.
@router.post("/threads/{thread_id}/documents")
def upload_thread_document(
    thread_id: str,
    file: UploadFile | None = File(None),
    native_path_lease: str | None = Form(None, alias = "nativePathLease"),
    ocr: bool | None = Form(None),
    caption: bool | None = Form(None),
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    stored_path, filename, content_hash = _resolve_document_upload(file, native_path_lease)
    with _rag_unavailable_as_503(stored_path):
        document_id, job_id = ingestion.start_ingestion(
            store.thread_scope(thread_id),
            None,
            thread_id,
            filename,
            stored_path,
            ocr = ocr,
            caption = caption,
            content_hash = content_hash,
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


def _discard_document(document_id: str) -> None:
    """Drop a document and its upload after the scope it was ingested for disappeared."""
    conn = _rag_connection()
    try:
        document = store.get_document(conn, document_id) or {}
        store.delete_document(conn, document_id)
    finally:
        conn.close()
    # Same uploads-root confinement as every other cleanup path, and best-effort for the same reason: on Windows
    # commonpath raises across drives and os.remove raises while the ingestion worker still holds the file, neither
    # of which should turn this into a 500.
    _remove_stored_upload(document.get("stored_path"))


# Stays sync for the reason above upload_kb_document.
@router.post("/projects/{project_id}/documents")
def upload_project_document(
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
    scope = store.project_scope(project_id)
    _raise_if_scope_retired(scope, "Project is being deleted")
    stored_path, filename, content_hash = _resolve_document_upload(file, native_path_lease)
    try:
        with folder_sync.scope_lock(scope):
            _require_scope_owner("project", project_id)
            _raise_if_scope_retired(scope, "Project is being deleted")
            with _rag_unavailable_as_503(stored_path):
                document_id, job_id = ingestion.start_ingestion(
                    scope,
                    None,
                    None,
                    filename,
                    stored_path,
                    project_id = project_id,
                    ocr = ocr,
                    caption = caption,
                    content_hash = content_hash,
                )
    except Exception:
        _remove_stored_upload(stored_path)
        raise
    # the project delete runs in the threadpool and can commit after the check above, once its own
    # RAG cleanup has already listed the project's documents
    if get_chat_project(project_id) is None:
        _discard_document(document_id)
        raise HTTPException(status_code = 404, detail = "Project not found")
    return {"documentId": document_id, "jobId": job_id, "filename": filename}


@router.get("/projects/{project_id}/documents")
def list_project_documents(project_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    _require_scope_owner("project", project_id)
    conn = _rag_connection()
    try:
        docs = store.list_documents(conn, store.project_scope(project_id))
        # Only here and the settings list carry sizes: this one is the sources
        # panel, which sorts by them.
        return {"documents": [_doc_view(d, with_size = True) for d in docs]}
    finally:
        conn.close()


@router.post("/projects/{project_id}/linked-folders")
def link_project_folder(
    project_id: str,
    payload: LinkFolderRequest,
    subject: str = Depends(get_current_subject),
) -> dict:
    _require_rag()
    _require_scope_owner("project", project_id)
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
        rows = [row for row in rows if row["scope_name"] is not None]
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
        view = _doc_view(doc, with_size = True)
        view["kbName"] = kb_names.get(doc.get("kb_id"))
        view["projectName"] = project_names.get(doc.get("project_id"))
        out.append(view)
    return {"documents": out}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, subject: str = Depends(get_current_subject)) -> dict:
    _require_rag()
    conn = _rag_connection()
    try:
        # Read and delete in one transaction so this serializes against an ingestion worker
        # publishing a replacement of this document (ingestion._replace_old_document takes the
        # same lock). Otherwise the worker could retire this row between the read and the
        # delete, leaving the delete a silent no-op and the source back under the new id.
        conn.execute("BEGIN IMMEDIATE")
        doc = store.get_visible_document(conn, document_id)
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


# POST too: quick tunnels hold a streamed GET until it closes. The hidden GET keeps old clients.
@router.post("/jobs/{job_id}/events")
@router.get("/jobs/{job_id}/events", include_in_schema = False)
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


# POST too, for the same reason as /jobs/{job_id}/events above.
@router.post("/linked-folder-jobs/{job_id}/events")
@router.get("/linked-folder-jobs/{job_id}/events", include_in_schema = False)
def folder_job_events(
    job_id: str, subject: str = Depends(get_current_subject)
) -> StreamingResponse:
    _require_rag()
    if folder_sync.get_job(job_id) is None:
        raise HTTPException(status_code = 404, detail = "Folder sync job not found")

    def gen():
        for event in folder_sync.job_events(job_id):
            if event is None:
                yield ": keepalive\n\n"
                continue
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
    # One connection for the whole request; the ownership check reads a single row.
    conn = _rag_connection()
    try:
        if payload.kb_id:
            _require_scope_owner("knowledge_base", payload.kb_id, conn)
            scope = store.kb_scope(payload.kb_id)
        else:
            scopes = []
            if payload.project_id:
                _require_scope_owner("project", payload.project_id, conn)
                scopes.append(store.project_scope(payload.project_id))
            if payload.thread_id:
                scopes.append(store.thread_scope(payload.thread_id))
            if not scopes:
                raise HTTPException(
                    status_code = 400, detail = "Provide kb_id, project_id, or thread_id"
                )
            scope = scopes[0] if len(scopes) == 1 else scopes

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
_PREVIEW_TTL = 600

_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".markdown": "text/markdown; charset=utf-8",
    # Served as plain text, never text/html: an uploaded HTML document rendered same-origin would
    # execute its scripts with access to the app's storage.
    ".html": "text/plain; charset=utf-8",
    ".htm": "text/plain; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# The signed token carries the account it was minted for; empty means the owner.
_DOCUMENT_TOKEN_VERSION = "v2"


def _document_mac(document_id: str, exp: str, account_id: str) -> str:
    payload = f"rag:{_DOCUMENT_TOKEN_VERSION}:{account_id}:{document_id}.{exp}"
    return hmac.new(_PREVIEW_SECRET, payload.encode(), hashlib.sha256).hexdigest()


def _sign_document(document_id: str) -> str:
    account = current_account()
    account_id = "" if account.is_owner else account.account_id
    exp = str(int(time.time()) + _PREVIEW_TTL)
    return f"{exp}.{account_id}.{_document_mac(document_id, exp, account_id)}"


def _verify_document_token(document_id: str, token: str) -> AccountContext | None:
    """The account whose uploads ``token`` opens for ``document_id``, or None once deactivated."""
    try:
        exp_s, account_id, sig = token.split(".", 2)
    except ValueError:
        return None
    if "." in sig:
        return None
    if not hmac.compare_digest(sig, _document_mac(document_id, exp_s, account_id)):
        return None
    try:
        if int(exp_s) < int(time.time()):
            return None
    except ValueError:
        return None
    if not account_id:
        return OWNER
    from auth.storage import get_account_by_id

    account = get_account_by_id(account_id)
    return None if account is None or account.is_owner else account


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
        doc = store.get_visible_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code = 404, detail = "Document not found")
        _require_document_owner(conn, doc)
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
                "SELECT text, page_number, pdf_regions_json FROM chunks "
                "WHERE id=? AND document_id=?",
                (chunk_id, document_id),
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
def document_file_url(
    document_id: str,
    subject: str = Depends(get_current_subject),
    no_credential: Annotated[bool, Depends(request_admitted_without_credential)] = False,
) -> dict:
    """Mint a short-lived signed URL for the source file."""
    if no_credential:
        raise HTTPException(
            status_code = 403,
            detail = "Document links can only be created from the Unsloth UI or with an API key.",
        )
    _require_rag()
    conn = _rag_connection()
    try:
        doc = store.get_visible_document(conn, document_id)
        if doc is None or not doc.get("stored_path"):
            raise HTTPException(status_code = 404, detail = "Document file not available")
        _require_document_owner(conn, doc)
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
    account = _verify_document_token(document_id, token)
    if account is None:
        raise HTTPException(status_code = 401, detail = "Invalid or expired token")
    # No auth dependency on this route, so without this bind every read hits the owner's store.
    marker = bind_account(account)
    try:
        _require_rag()
        conn = _rag_connection()
        try:
            doc = store.get_visible_document(conn, document_id)
            if doc is not None:
                _require_document_owner(conn, doc)
        finally:
            conn.close()
        stored_path = (doc or {}).get("stored_path")
        if not doc or not stored_path or not os.path.isfile(stored_path):
            raise HTTPException(status_code = 404, detail = "Document file not found")
        if not _is_managed_preview_path(stored_path):
            raise HTTPException(status_code = 403, detail = "Forbidden")
    finally:
        reset_account(marker)
    ext = os.path.splitext(doc["filename"])[1].lower()
    return FileResponse(
        stored_path,
        media_type = _CONTENT_TYPES.get(ext, "application/octet-stream"),
        # linked documents are named by a posix relative path, invalid in this header
        filename = doc["filename"].rsplit("/", 1)[-1],
    )


# Formats a <textarea> represents faithfully, so an edit round-trips byte for byte.
# .pdf and .docx are deliberately absent: neither survives being retyped as plain
# text, so both are display-only.
_EDITABLE_EXTS = {".txt", ".md", ".markdown", ".html", ".htm"}

# What the modal's "View" tab shows, per extension. "source" means the raw text is
# the only honest view, so the modal offers no toggle and goes straight to the
# editor. Anything richer gets a View/Edit pair: "markdown" renders it, "extracted"
# shows the text the indexer derived from a file whose source is not that text.
#
# Keyed by extension so a newly supported upload type (see config.UPLOAD_EXTS)
# picks a view here and the client needs no change.
_PREVIEW_MODES = {
    ".md": "markdown",
    ".markdown": "markdown",
    # Rendered in the chat artifact canvas: a sandboxed, opaque-origin iframe with
    # network access denied by default. That sandbox is what makes rendering an
    # uploaded page safe -- it must never be dropped into the app's own origin.
    ".html": "html",
    ".htm": "html",
    # No source to edit, so the extracted text is both the view and the whole of it.
    ".docx": "extracted",
}
# Read (and accept) at most this much text. Beyond it the preview is truncated for
# display and editing is refused, because saving a truncated body would silently
# delete the tail of the file.
_MAX_TEXT_EDIT_BYTES = 1024 * 1024
# The same bound for text with no byte form of its own (a .docx extraction). One
# character is at most four UTF-8 bytes, so this can never exceed the byte cap.
_MAX_TEXT_EDIT_CHARS = _MAX_TEXT_EDIT_BYTES // 4


# A .docx is a zip, and a small one can hold text that expands to many times its size on
# disk. Slicing the extraction only bounds the reply -- the parser has already built the
# whole document by then, so the allocation happened regardless. The archive declares how
# much it unpacks to, which costs no decompression to read, so one that cannot possibly fit
# the cap is never handed to the parser at all. Generous against the character cap because
# a .docx is mostly markup: the text inside is a fraction of the XML around it.
_MAX_DOCX_UNPACKED_BYTES = 16 * 1024 * 1024


def _docx_text(stored_path: str) -> tuple[str, bool]:
    """A Word document's extracted text, bounded before it is extracted.

    Returns ``("", False)`` for an archive too large to preview, which reads as truncated
    with nothing to show -- distinct from an empty document, which is faithfully empty.
    """
    import zipfile

    from core.rag import parsers

    try:
        with zipfile.ZipFile(stored_path) as archive:
            unpacked = sum(item.file_size for item in archive.infolist())
    except Exception:  # noqa: BLE001 - not a readable zip; let the parser report it
        unpacked = 0
    if unpacked > _MAX_DOCX_UNPACKED_BYTES:
        return "", False
    # Accumulated, not joined in one pass: joining allocates a second full copy of text the
    # parser has already built, and stopping at the cap keeps the excess out of the reply.
    pages: list[str] = []
    size = 0
    for page in parsers.parse(stored_path):
        pages.append(page.text)
        size += len(page.text) + 1
        if size > _MAX_TEXT_EDIT_CHARS:
            return "\n".join(pages)[:_MAX_TEXT_EDIT_CHARS], False
    return "\n".join(pages), True


def _document_text(stored_path: str, ext: str) -> tuple[str, bool]:
    """The document's own text, and whether it is the whole of it verbatim.

    ``False`` means the text on screen is not a faithful copy of the file, so
    saving it back would destroy the part that did not survive the trip. Two ways
    that happens: the body was cut off at the size cap, or a byte was not valid
    UTF-8 and decoded to U+FFFD (saving would rewrite that byte as the encoding of
    the replacement character, corrupting content the user never touched).

    A .docx has no text form on disk, so it goes back through the ingestion parser
    and what the modal shows is exactly what was chunked and embedded. That is
    never editable, and is capped here because a small compressed file can extract
    to many megabytes.
    """
    if ext == ".docx":
        return _docx_text(stored_path)
    with open(stored_path, "rb") as handle:
        # One byte past the cap distinguishes "exactly at the cap" from "longer".
        raw = handle.read(_MAX_TEXT_EDIT_BYTES + 1)
    complete = len(raw) <= _MAX_TEXT_EDIT_BYTES
    raw = raw[:_MAX_TEXT_EDIT_BYTES]
    # replace, not strict: a preview must never 500 on a stray byte. Whether the
    # decode was lossy decides editability, not whether it succeeded.
    text = raw.decode("utf-8", errors = "replace")
    return text, complete and text.encode("utf-8") == raw


def _newline_style(text: str) -> str | None:
    """The one line ending this text uses, or ``None`` when it mixes conventions.

    A <textarea> hands back "\\n" for every line whatever the file used -- the HTML API value is
    newline-normalized -- so an edit can only be written back verbatim when a single convention
    covers the whole file and the client can restore it. A file that mixes them (or uses a lone
    CR) has no such convention, so it is shown and not edited rather than silently rewritten.
    """
    crlf, lf, cr = text.count("\r\n"), text.count("\n"), text.count("\r")
    if crlf and crlf == lf == cr:
        return "\r\n"
    return "\n" if not cr else None


@router.get("/documents/{document_id}/content")
def document_content(document_id: str, subject: str = Depends(get_current_subject)) -> dict:
    """Text of a source for the preview modal, plus whether it may be edited.

    The editability decision lives here rather than in the client so there is one
    copy of the rule; the client renders whatever this returns.
    """
    _require_rag()
    conn = _rag_connection()
    try:
        doc = store.get_visible_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code = 404, detail = "Document not found")
        _require_document_owner(conn, doc)
    finally:
        conn.close()

    ext = os.path.splitext(doc["filename"])[1].lower()
    out = {
        "documentId": document_id,
        "filename": doc["filename"],
        "mediaKind": "pdf" if ext == ".pdf" else "text",
        "preview": _PREVIEW_MODES.get(ext, "source"),
        "text": None,
        "editable": False,
        "truncated": False,
        "readOnlyReason": None,
        # The line ending the editor must restore on save; see _newline_style.
        "newline": "\n",
    }
    if ext == ".pdf":
        # Rendered from the signed file URL by pdf.js, so no text is sent here.
        out["readOnlyReason"] = "PDFs are shown as the original document and cannot be edited here."
        return out

    stored_path = doc.get("stored_path")
    if not stored_path or not os.path.isfile(stored_path):
        raise HTTPException(status_code = 404, detail = "Document file not available")
    # Same uploads-root confinement as the signed file route: a stored_path that
    # escaped the managed root is never read, let alone written.
    if not _is_managed_preview_path(stored_path):
        raise HTTPException(status_code = 403, detail = "Forbidden")

    try:
        out["text"], faithful = _document_text(stored_path, ext)
    except Exception as exc:  # noqa: BLE001 - a broken file is a preview failure, not a 500
        logger.warning("failed to read document %s for preview", document_id, exc_info = True)
        raise HTTPException(
            status_code = 422, detail = f"Could not read this document ({exc})"
        ) from exc

    if ext not in _EDITABLE_EXTS:
        out["truncated"] = not faithful
        # An empty document reads back faithfully empty, so nothing-and-unfaithful is the
        # archive that was too large to unpack rather than a document with no words in it.
        out["readOnlyReason"] = (
            "This Word document is too large to preview here."
            if not faithful and not out["text"]
            else "Word documents are shown as the text indexed and cannot be edited here."
        )
    elif doc.get("linked_folder_id"):
        # Editing the snapshot would be undone by the next folder sync, and the
        # original in the user's folder is never written to. So: display only.
        out["readOnlyReason"] = (
            "This source is synced from a linked folder, so it is read-only here. "
            "Edit the file in the folder instead."
        )
    elif doc.get("status") in ("pending", "running"):
        out["readOnlyReason"] = "This source is still indexing."
    elif not faithful:
        # Either cut off at the cap or holding a byte that is not valid UTF-8.
        # Saving back what is on screen would destroy whatever did not survive
        # the trip, so it is shown and not editable.
        out["truncated"] = True
        out["readOnlyReason"] = (
            "This file cannot be edited here: it is too large or is not valid UTF-8 text."
        )
    elif (newline := _newline_style(out["text"])) is None:
        # The editor works in LF and restores one convention on save, which a file using
        # several cannot survive: saving it back would rewrite line endings the user never
        # touched, the same reason a lossy decode above is read-only.
        out["readOnlyReason"] = (
            "This file mixes line endings, so editing it here would rewrite them."
        )
    else:
        out["editable"] = True
        out["newline"] = newline
    return out


def _release_replacement_claim(document_id: str) -> None:
    """Undo the claim when the replacement never started.

    Best-effort: the alternative to a failed release is a source stuck reading as
    indexing, which is worse than a logged warning.
    """
    try:
        conn = _rag_connection()
        try:
            conn.execute(
                "UPDATE documents SET status='completed' WHERE id=? AND status='running'",
                (document_id,),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - the caller is already raising
        logger.warning("failed to release edit claim on %s", document_id, exc_info = True)


class UpdateDocumentContentRequest(BaseModel):
    # Characters, which bound the request cheaply; the byte length that actually
    # governs what may be written is checked against the cap below, since one
    # character can encode to four bytes.
    text: str = Field(max_length = _MAX_TEXT_EDIT_BYTES)


@router.put("/documents/{document_id}/content")
def update_document_content(
    document_id: str,
    payload: UpdateDocumentContentRequest,
    subject: str = Depends(get_current_subject),
) -> dict:
    """Save an edited source and re-index it, returning the replacement document.

    The edit is written to a *new* file in the managed uploads root and ingested as
    a replacement, never over the existing one. The document it replaces is retired
    by the ingestion worker only once the re-index completes, so a parse or embed
    failure leaves the original searchable rather than destroying it. Files outside
    the uploads root -- every original the user linked or dragged in -- are never
    opened for writing.
    """
    _require_rag()
    conn = _rag_connection()
    try:
        doc = store.get_visible_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code = 404, detail = "Document not found")
        _require_document_owner(conn, doc)
    finally:
        conn.close()

    if doc.get("linked_folder_id"):
        raise HTTPException(
            status_code = 409,
            detail = "Linked-folder documents are managed by folder synchronization",
        )
    ext = os.path.splitext(doc["filename"])[1].lower()
    if ext not in _EDITABLE_EXTS:
        raise HTTPException(status_code = 400, detail = f"'{ext}' documents cannot be edited")
    if doc.get("status") in ("pending", "running"):
        # An ingestion worker is reading the current file and will write this
        # document's rows; replacing it underneath would race that job.
        raise HTTPException(status_code = 409, detail = "This source is still indexing")
    old_path = doc.get("stored_path")
    if not old_path or not os.path.isfile(old_path):
        raise HTTPException(status_code = 404, detail = "Document file not available")
    if not _is_managed_preview_path(old_path):
        raise HTTPException(status_code = 403, detail = "Forbidden")

    body = payload.text.encode("utf-8")
    # max_length counts characters; this counts what actually lands on disk. A
    # payload that passed the field check can still exceed the cap (one character
    # encodes to up to four bytes), and saving it would produce a file the next
    # GET has to truncate -- turning the source the user just saved read-only.
    if len(body) > _MAX_TEXT_EDIT_BYTES:
        raise HTTPException(
            status_code = 413,
            detail = f"Text exceeds the {_MAX_TEXT_EDIT_BYTES // 1024} KB edit limit.",
        )

    scope = doc["scope"]
    _raise_if_scope_retired(scope, "The owner of this source is being deleted")

    uploads = ensure_dir(rag_uploads_root())
    stored_path = str(uploads / f"{uuid.uuid4().hex}{ext}")
    with open(stored_path, "wb") as handle:
        handle.write(body)
    try:
        with folder_sync.scope_lock(scope):
            _raise_if_scope_retired(scope, "The owner of this source is being deleted")
            # An ordinary upload dedupes by content hash, so a scope never holds the same
            # bytes twice. A replacement cannot go through that path -- start_ingestion's
            # dedupe branch owns `replaces` and its early return would drop it -- so saving
            # this source into another one's exact bytes would index the same content under
            # two documents, and retrieval (which dedupes only by chunk id) would return
            # both copies. Refused rather than silently merged: the two sources keep their
            # own names, and quietly retiring the one being edited would make it vanish
            # into a file the user did not open.
            #
            # Inside the scope lock, which is what makes it hold: outside it, two clients
            # editing two different sources to the same new bytes both see no twin, then
            # admit one after the other and index the content twice. The lock is the same
            # one start_ingestion's admission runs under, so a losing writer sees the
            # winner's row.
            conn = _rag_connection()
            try:
                twin = store.document_by_hash(conn, scope, hashlib.sha256(body).hexdigest())
            finally:
                conn.close()
            if twin is not None and twin != document_id:
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        "This edit would make the source identical to another one in this "
                        "project."
                    ),
                )
            with _rag_unavailable_as_503(stored_path):
                # The claim on the source lives inside start_ingestion's admission
                # transaction, alongside the replacement row that records what it is
                # waiting for -- the status read above came from a closed connection, so
                # it cannot serialize two saves on its own, and claiming here instead
                # would leave the claim committed and unexplained if this process died
                # before the row existed. A second concurrent save matches no row there
                # and arrives as ReplacementClaimUnavailable.
                try:
                    new_id, job_id = ingestion.start_ingestion(
                        scope,
                        doc.get("kb_id"),
                        doc.get("thread_id"),
                        # Same scope columns and same filename: the replacement is
                        # the same source, so retrieval keeps finding it where it
                        # was.
                        doc["filename"],
                        stored_path,
                        project_id = doc.get("project_id"),
                        dedupe = False,
                        replaces = (document_id, old_path),
                    )
                except ingestion.ReplacementClaimUnavailable as exc:
                    # Lost the race: the claim was never taken, so there is nothing to
                    # release and the winner's save is untouched.
                    raise HTTPException(status_code = 409, detail = str(exc)) from exc
                except Exception:
                    # Past the admission transaction (a worker that could not start), so
                    # the claim is committed and only this can hand it back.
                    _release_replacement_claim(document_id)
                    raise
    except Exception:
        _remove_stored_upload(stored_path)
        raise
    return {"documentId": new_id, "jobId": job_id, "filename": doc["filename"]}
