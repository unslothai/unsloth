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


def _insert_kb(conn, kb_id: str, owner: str = "alice") -> None:
    conn.execute(
        "INSERT INTO rag_knowledge_bases "
        "(id, name, embedding_model, owner_user_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (kb_id, f"KB-{kb_id[:6]}", "bge-small", owner, 1_700_000_000),
    )


def _insert_doc(conn, doc_id: str, kb_id: str, stored_path: str, filename: str) -> None:
    conn.execute(
        "INSERT INTO rag_documents "
        "(id, kb_id, thread_id, filename, content_type, stored_path, status, "
        "num_chunks, byte_size, created_at) "
        "VALUES (?, ?, NULL, ?, 'text/plain', ?, 'completed', 1, 64, ?)",
        (doc_id, kb_id, filename, stored_path, 1_700_000_000),
    )


def _insert_chunk(conn, chunk_id: str, doc_id: str, text: str) -> None:
    conn.execute(
        "INSERT INTO rag_chunks "
        "(id, document_id, chunk_index, text, token_count, page_number) "
        "VALUES (?, ?, 0, ?, 5, NULL)",
        (chunk_id, doc_id, text),
    )


def test_backfill_preserves_ids_and_updates_unique_locator(app, db_env, monkeypatch):
    doc_id, kb_id, chunk_id = _uid(), _uid(), _uid()
    stored = db_env / "rag" / "uploads" / "paper.txt"
    stored.parent.mkdir(parents = True, exist_ok = True)
    stored.write_text("Intro line\nUnique quote here.\nEnd.", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

    with studio_db.get_connection() as conn:
        _insert_kb(conn, kb_id)
        _insert_doc(conn, doc_id, kb_id, str(stored), "paper.txt")
        _insert_chunk(conn, chunk_id, doc_id, "Unique quote here.")

    client = _make_client(app, "alice")
    try:
        resp = client.post(f"/api/rag/documents/{doc_id}/locators/backfill")
        target_resp = client.get(
            f"/api/rag/documents/{doc_id}/preview-target?chunk_id={chunk_id}"
        )
    finally:
        _clear_overrides(app)

    assert resp.status_code == 200
    body = resp.json()
    assert body["documentId"] == doc_id
    assert body["matched"] == 1
    assert body["ambiguous"] == 0

    target = target_resp.json()
    assert target["documentId"] == doc_id
    assert target["chunkId"] == chunk_id
    assert target["sourcePageIndex"] == 0
    assert target["lineStart"] == 2
    assert target["pageCharStart"] in (len("Intro line\n"), len("Intro line\r\n"))


def test_backfill_leaves_ambiguous_matches_null(app, db_env, monkeypatch):
    doc_id, kb_id, chunk_id = _uid(), _uid(), _uid()
    stored = db_env / "rag" / "uploads" / "paper.txt"
    stored.parent.mkdir(parents = True, exist_ok = True)
    stored.write_text("Repeat me.\nOther text.\nRepeat me.", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

    with studio_db.get_connection() as conn:
        _insert_kb(conn, kb_id)
        _insert_doc(conn, doc_id, kb_id, str(stored), "paper.txt")
        _insert_chunk(conn, chunk_id, doc_id, "Repeat me.")

    client = _make_client(app, "alice")
    try:
        resp = client.post(f"/api/rag/documents/{doc_id}/locators/backfill")
        target_resp = client.get(
            f"/api/rag/documents/{doc_id}/preview-target?chunk_id={chunk_id}"
        )
    finally:
        _clear_overrides(app)

    assert resp.status_code == 200
    body = resp.json()
    assert body["matched"] == 0
    assert body["ambiguous"] == 1

    target = target_resp.json()
    assert target["sourcePageIndex"] is None
    assert target["pageCharStart"] is None
    assert target["lineStart"] is None
