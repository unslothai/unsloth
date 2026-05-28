# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import storage.studio_db as studio_db
from auth.authentication import get_current_subject


@pytest.fixture(scope = "module")
def app():
    import sys

    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from main import app as _app

    return _app


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return tmp_path


def _uid() -> str:
    return str(uuid.uuid4())


def _make_client(app, subject: str = "alice"):
    app.dependency_overrides[get_current_subject] = lambda: subject
    return TestClient(app, raise_server_exceptions = True)


def _clear_overrides(app):
    app.dependency_overrides.clear()


def _seed_doc(conn, doc_id: str, kb_id: str, stored_path: str) -> None:
    conn.execute(
        """
        INSERT INTO rag_knowledge_bases
        (id, name, embedding_model, owner_user_id, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (kb_id, "KB", "embedder", "alice", 1_700_000_000),
    )
    conn.execute(
        """
        INSERT INTO rag_documents
        (id, kb_id, thread_id, filename, content_type, stored_path, status,
         num_chunks, byte_size, created_at)
        VALUES (?, ?, NULL, ?, ?, ?, 'completed', 1, 10, ?)
        """,
        (doc_id, kb_id, "report.pdf", "application/pdf", stored_path, 1_700_000_001),
    )


def test_preview_target_returns_nullable_locator_fields(app, db_env):
    doc_id, kb_id, chunk_id = _uid(), _uid(), _uid()
    stored = db_env / "rag" / "uploads" / "report.pdf"
    stored.parent.mkdir(parents = True, exist_ok = True)
    stored.write_bytes(b"%PDF-1.4")

    with studio_db.get_connection() as conn:
        _seed_doc(conn, doc_id, kb_id, str(stored))
        conn.execute(
            """
            INSERT INTO rag_chunks
            (id, document_id, chunk_index, text, token_count, page_number,
             source_page_index, page_char_start, page_char_end, line_start,
             line_end)
            VALUES (?, ?, 2, ?, 8, 4, 3, 20, 52, 6, 7)
            """,
            (chunk_id, doc_id, "highlight me"),
        )

    client = _make_client(app)
    try:
        resp = client.get(
            f"/api/rag/documents/{doc_id}/preview-target?chunk_id={chunk_id}"
        )
    finally:
        _clear_overrides(app)

    assert resp.status_code == 200
    body = resp.json()
    assert body["sourcePageIndex"] == 3
    assert body["pageCharStart"] == 20
    assert body["pageCharEnd"] == 52
    assert body["lineStart"] == 6
    assert body["lineEnd"] == 7


def test_preview_target_old_null_locator_rows_still_work(app, db_env):
    doc_id, kb_id, chunk_id = _uid(), _uid(), _uid()
    stored = db_env / "rag" / "uploads" / "legacy.pdf"
    stored.parent.mkdir(parents = True, exist_ok = True)
    stored.write_bytes(b"%PDF-1.4")

    with studio_db.get_connection() as conn:
        _seed_doc(conn, doc_id, kb_id, str(stored))
        conn.execute(
            """
            INSERT INTO rag_chunks
            (id, document_id, chunk_index, text, token_count, page_number)
            VALUES (?, ?, 0, ?, 4, 1)
            """,
            (chunk_id, doc_id, "legacy"),
        )

    client = _make_client(app)
    try:
        resp = client.get(
            f"/api/rag/documents/{doc_id}/preview-target?chunk_id={chunk_id}"
        )
    finally:
        _clear_overrides(app)

    assert resp.status_code == 200
    body = resp.json()
    assert body["snippet"] == "legacy"
    assert body["sourcePageIndex"] is None
    assert body["pageCharStart"] is None
    assert body["pageCharEnd"] is None
    assert body["lineStart"] is None
    assert body["lineEnd"] is None
