# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from .factory_base import Factory, seeder

KB_ID = "rag-matrix-kb"
DOCUMENT_ID = "rag-matrix-document"
JOB_ID = "rag-matrix-job"
FOLDER_JOB_ID = "rag-matrix-folder-job"
PROJECT_ID = "rag-matrix-project"
THREAD_ID = "rag-matrix-thread"
FILENAME = "rag-matrix-sentinel.txt"
KB_NAME = "rag-matrix-kb-name"
EDITED = "rag-matrix-edited"
CONTENT = b"rag matrix sentinel document body\n"
# Unsigned, so verify_native_path_lease rejects it before any folder is created.
BAD_LEASE = {"nativePathLease": "rag-matrix-invalid-lease"}


def _connection(account):
    from storage import rag_db
    from utils.account_context import run_as
    return run_as(account, rag_db.get_connection)


def _uploads_dir(account):
    from utils.account_context import run_as
    from utils.paths import ensure_dir, rag_uploads_root
    return run_as(account, lambda: ensure_dir(rag_uploads_root()))


def _store_file(account) -> str:
    path = _uploads_dir(account) / FILENAME
    path.write_bytes(CONTENT)
    return str(path)


def _create_kb(conn) -> None:
    from core.rag import store
    store.create_kb(conn, name = KB_NAME, description = None, kb_id = KB_ID)


def _create_document(account, conn, *, scope: str, **columns) -> None:
    import hashlib

    from core.rag import store
    store.create_document(
        conn,
        scope = scope,
        filename = FILENAME,
        sha256 = hashlib.sha256(CONTENT).hexdigest(),
        status = "completed",
        stored_path = _store_file(account),
        document_id = DOCUMENT_ID,
        **columns,
    )


@seeder("rag-kb")
def seed_rag_kb(account) -> dict[str, str]:
    conn = _connection(account)
    try:
        _create_kb(conn)
    finally:
        conn.close()
    return {"kb_id": KB_ID}


@seeder("rag-document")
def seed_rag_document(account) -> dict[str, str]:
    from core.rag import store

    conn = _connection(account)
    try:
        _create_kb(conn)
        _create_document(account, conn, scope = store.kb_scope(KB_ID), kb_id = KB_ID)
    finally:
        conn.close()
    return {"document_id": DOCUMENT_ID, "kb_id": KB_ID}


@seeder("rag-job")
def seed_rag_job(account) -> dict[str, str]:
    from core.rag import store

    conn = _connection(account)
    try:
        _create_kb(conn)
        _create_document(account, conn, scope = store.kb_scope(KB_ID), kb_id = KB_ID)
        conn.execute(
            "INSERT INTO ingestion_jobs(id, document_id, scope, status, stage, progress, "
            "created_at) VALUES(?,?,?,'completed','done',1.0,'2026-01-01T00:00:00+00:00')",
            (JOB_ID, DOCUMENT_ID, store.kb_scope(KB_ID)),
        )
        conn.commit()
    finally:
        conn.close()
    return {"job_id": JOB_ID, "document_id": DOCUMENT_ID}


@seeder("rag-folder")
def seed_rag_folder(account) -> dict[str, str]:
    from core.rag import folder_sync
    from utils.account_context import run_as
    from utils.paths import workspace_root

    conn = _connection(account)
    try:
        _create_kb(conn)
    finally:
        conn.close()
    linked = run_as(account, workspace_root) / "rag-matrix-linked"
    linked.mkdir(parents = True, exist_ok = True)
    (linked / FILENAME).write_bytes(CONTENT)
    folder = run_as(
        account,
        lambda: folder_sync.create_folder(
            scope_type = "knowledge_base",
            scope_id = KB_ID,
            path = str(linked),
            name = KB_NAME,
            auto_sync = False,
        ),
    )
    conn = _connection(account)
    try:
        conn.execute(
            "INSERT INTO linked_folder_sync_jobs(id, folder_id, kind, status, stage, progress, "
            "created_at, completed_at) VALUES(?,?,'sync','completed','done',1.0,"
            "'2026-01-01T00:00:00+00:00','2026-01-01T00:00:01+00:00')",
            (FOLDER_JOB_ID, folder["id"]),
        )
        conn.commit()
    finally:
        conn.close()
    return {"folder_id": folder["id"], "job_id": FOLDER_JOB_ID}


@seeder("rag-project")
def seed_rag_project(account) -> dict[str, str]:
    import sqlite3
    from contextlib import closing

    from utils.account_context import run_as
    from utils.paths import studio_db_path

    with closing(sqlite3.connect(run_as(account, studio_db_path))) as conn:
        conn.execute(
            "INSERT INTO chat_projects (id,name,root_path,created_at,updated_at) "
            "VALUES (?,?,NULL,1000,1000)",
            (PROJECT_ID, KB_NAME),
        )
        conn.commit()
    return {"project_id": PROJECT_ID}


@seeder("rag-project-document")
def seed_rag_project_document(account) -> dict[str, str]:
    from core.rag import store

    seed_rag_project(account)
    conn = _connection(account)
    try:
        _create_document(
            account, conn, scope = store.project_scope(PROJECT_ID), project_id = PROJECT_ID
        )
    finally:
        conn.close()
    return {"project_id": PROJECT_ID, "document_id": DOCUMENT_ID}


@seeder("rag-thread-document")
def seed_rag_thread_document(account) -> dict[str, str]:
    from core.rag import store

    conn = _connection(account)
    try:
        _create_document(account, conn, scope = store.thread_scope(THREAD_ID), thread_id = THREAD_ID)
    finally:
        conn.close()
    return {"thread_id": THREAD_ID, "document_id": DOCUMENT_ID}


_LIST_SCOPE = (
    "the scope is not resolved to an owner, so an unknown kb/thread lists that account's own "
    "empty scope with 200 instead of 404"
)
_INGESTION_SSE = (
    "the ingestion event stream is keyed by an in-process per-account queue, so a foreign "
    "account finds none and gets an immediate 200 [DONE] without reading this account's rows"
)
_MULTIPART = (
    "the JSON-only matrix client cannot send the multipart upload, so the owning account stops "
    "at 400 after the ownership check that returns 404 to everyone else"
)
_MULTIPART_UNSCOPED = (
    "thread uploads have no thread-existence check and need a multipart body the JSON matrix "
    "client cannot send, so every actor stops at the same 400 and no case can discriminate"
)
_LEASE = (
    "linking needs a signed desktop path grant, so the owning account stops at 400 after the "
    "ownership check that returns 404 to everyone else"
)

FACTORIES = {
    "routes.rag:PATCH:/knowledge-bases/{kb_id}": Factory("rag-kb", {"name": EDITED}),
    "routes.rag:DELETE:/knowledge-bases/{kb_id}": Factory("rag-kb"),
    "routes.rag:GET:/knowledge-bases/{kb_id}/documents": Factory(
        "rag-document",
        fragment = FILENAME,
        absent = FILENAME,
        owner = (200,),
        wrong = (200,),
        reason = _LIST_SCOPE,
    ),
    "routes.rag:POST:/knowledge-bases/{kb_id}/documents": Factory(
        "rag-kb",
        success = 400,
        fragment = "No file was provided",
        self_expected = (400,),
        reason = _MULTIPART,
    ),
    "routes.rag:POST:/knowledge-bases/{kb_id}/linked-folders": Factory(
        "rag-kb",
        BAD_LEASE,
        success = 400,
        fragment = "Native path grant",
        self_expected = (400,),
        reason = _LEASE,
    ),
    "routes.rag:DELETE:/documents/{document_id}": Factory("rag-document"),
    "routes.rag:GET:/documents/{document_id}/file-url": Factory(
        "rag-document", fragment = "file-signed"
    ),
    "routes.rag:GET:/documents/{document_id}/preview-target": Factory(
        "rag-document", fragment = FILENAME
    ),
    "routes.rag:GET:/jobs/{job_id}": Factory("rag-job", fragment = DOCUMENT_ID),
    "routes.rag:GET:/jobs/{job_id}/events": Factory(
        "rag-job",
        fragment = "[DONE]",
        absent = DOCUMENT_ID,
        owner = (200,),
        wrong = (200,),
        reason = _INGESTION_SSE,
    ),
    "routes.rag:POST:/jobs/{job_id}/events": Factory(
        "rag-job",
        fragment = "[DONE]",
        absent = DOCUMENT_ID,
        owner = (200,),
        wrong = (200,),
        reason = _INGESTION_SSE,
    ),
    "routes.rag:GET:/linked-folder-jobs/{job_id}": Factory("rag-folder", fragment = FOLDER_JOB_ID),
    "routes.rag:GET:/linked-folder-jobs/{job_id}/events": Factory("rag-folder", fragment = "[DONE]"),
    "routes.rag:POST:/linked-folder-jobs/{job_id}/events": Factory("rag-folder", fragment = "[DONE]"),
    "routes.rag:PATCH:/linked-folders/{folder_id}": Factory(
        "rag-folder", {"name": EDITED}, fragment = EDITED
    ),
    "routes.rag:DELETE:/linked-folders/{folder_id}": Factory("rag-folder"),
    "routes.rag:POST:/linked-folders/{folder_id}/sync": Factory("rag-folder"),
    "routes.rag:POST:/linked-folders/{folder_id}/rebuild": Factory("rag-folder"),
    "routes.rag:GET:/projects/{project_id}/documents": Factory(
        "rag-project-document", fragment = FILENAME
    ),
    "routes.rag:POST:/projects/{project_id}/documents": Factory(
        "rag-project",
        success = 400,
        fragment = "No file was provided",
        self_expected = (400,),
        reason = _MULTIPART,
    ),
    "routes.rag:POST:/projects/{project_id}/linked-folders": Factory(
        "rag-project",
        BAD_LEASE,
        success = 400,
        fragment = "Native path grant",
        self_expected = (400,),
        reason = _LEASE,
    ),
    "routes.rag:GET:/threads/{thread_id}/documents": Factory(
        "rag-thread-document",
        fragment = FILENAME,
        absent = FILENAME,
        owner = (200,),
        wrong = (200,),
        reason = _LIST_SCOPE,
    ),
}

SKIPPED = {
    "routes.rag:POST:/threads/{thread_id}/documents": _MULTIPART_UNSCOPED,
    "routes.rag:GET:/documents/{document_id}/file-signed": (
        "signed document downloads resolve in the owner's store for every account, so the route "
        "404s for the account that minted the token; product finding, see report"
    ),
}
