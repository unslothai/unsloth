# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for document_for_subject_or_404.

Authorization rules under test (contracts.md §1 / §2, Risk #1):

- KB documents: KB must exist and KB.owner_user_id must equal current_subject.
- Thread documents: thread must exist in chat_threads; current-Studio single-user
  invariant means any authenticated subject can access, BUT the thread row must
  exist (a missing thread is 404, not silent grant).
- Missing document or missing KB both collapse to 404.
- KB with NULL owner_user_id is NOT accessible (legacy row guard).
- Both not-found and not-authorized return HTTP 404 with identical detail to
  prevent document-existence leaking.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

import storage.studio_db as studio_db
from core.rag.authorization import document_for_subject_or_404


# ── Fixtures ──────────────────────────────────────────────────────────


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _uid() -> str:
    return str(uuid.uuid4())


def _insert_kb(conn, kb_id: str, owner: str | None = "user-alice") -> None:
    conn.execute(
        """
        INSERT INTO rag_knowledge_bases (id, name, embedding_model, owner_user_id, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (kb_id, f"KB-{kb_id[:8]}", "bge-small", owner, 1_700_000_000),
    )


def _insert_thread(conn, thread_id: str) -> None:
    conn.execute(
        """
        INSERT INTO chat_threads (id, title, model_type, model_id, archived, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (thread_id, "Test Thread", "base", "llama3", 0, 1_700_000_000),
    )


def _insert_kb_doc(conn, doc_id: str, kb_id: str, stored_path: str = "doc.pdf") -> None:
    conn.execute(
        """
        INSERT INTO rag_documents
            (id, kb_id, thread_id, filename, content_type, stored_path, status,
             num_chunks, byte_size, created_at)
        VALUES (?, ?, NULL, ?, ?, ?, 'completed', 0, 1024, ?)
        """,
        (doc_id, kb_id, "report.pdf", "application/pdf", stored_path, 1_700_000_000),
    )


def _insert_thread_doc(
    conn, doc_id: str, thread_id: str, stored_path: str = "doc.txt"
) -> None:
    conn.execute(
        """
        INSERT INTO rag_documents
            (id, kb_id, thread_id, filename, content_type, stored_path, status,
             num_chunks, byte_size, created_at)
        VALUES (?, NULL, ?, ?, ?, ?, 'completed', 0, 512, ?)
        """,
        (doc_id, thread_id, "note.txt", "text/plain", stored_path, 1_700_000_000),
    )


# ── KB-document authorization ─────────────────────────────────────────


def test_kb_doc_correct_owner_returns_row(tmp_path, monkeypatch):
    """KB doc authorized when KB.owner_user_id == current_subject."""
    _reset_db(tmp_path, monkeypatch)
    doc_id, kb_id = _uid(), _uid()
    with studio_db.get_connection() as conn:
        _insert_kb(conn, kb_id, owner = "alice")
        _insert_kb_doc(conn, doc_id, kb_id)
    row = document_for_subject_or_404(doc_id, "alice")
    assert row["id"] == doc_id


def test_kb_doc_wrong_owner_raises_404(tmp_path, monkeypatch):
    """KB doc returns 404 when current_subject != KB.owner_user_id."""
    _reset_db(tmp_path, monkeypatch)
    doc_id, kb_id = _uid(), _uid()
    with studio_db.get_connection() as conn:
        _insert_kb(conn, kb_id, owner = "alice")
        _insert_kb_doc(conn, doc_id, kb_id)
    with pytest.raises(HTTPException) as exc_info:
        document_for_subject_or_404(doc_id, "mallory")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Document not found"


def test_kb_doc_null_owner_raises_404(tmp_path, monkeypatch):
    """KB with NULL owner_user_id is not accessible through the helper (legacy guard)."""
    _reset_db(tmp_path, monkeypatch)
    doc_id, kb_id = _uid(), _uid()
    with studio_db.get_connection() as conn:
        _insert_kb(conn, kb_id, owner = None)
        _insert_kb_doc(conn, doc_id, kb_id)
    with pytest.raises(HTTPException) as exc_info:
        document_for_subject_or_404(doc_id, "alice")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Document not found"


def test_kb_doc_missing_kb_raises_404(tmp_path, monkeypatch):
    """Document rows whose KB was deleted collapse to 404.

    Insert both KB and doc, then delete the KB (ON DELETE CASCADE removes the doc
    too). A subsequent lookup for the doc id must return 404, not 500.
    If for some reason the doc row survives (e.g. FK off), the helper must
    still 404 because the KB is gone.
    """
    _reset_db(tmp_path, monkeypatch)
    doc_id, kb_id = _uid(), _uid()
    with studio_db.get_connection() as conn:
        _insert_kb(conn, kb_id, owner = "alice")
        _insert_kb_doc(conn, doc_id, kb_id)
        # Delete the KB; cascade drops the doc too.
        conn.execute("DELETE FROM rag_knowledge_bases WHERE id = ?", (kb_id,))
    # doc_id is gone post-cascade → 404.
    with pytest.raises(HTTPException) as exc_info:
        document_for_subject_or_404(doc_id, "alice")
    assert exc_info.value.status_code == 404


# ── Thread-document authorization (single-user invariant) ─────────────


def test_thread_doc_existing_thread_grants_access(tmp_path, monkeypatch):
    """Thread doc is accessible when thread exists (single-user Studio invariant)."""
    _reset_db(tmp_path, monkeypatch)
    doc_id, thread_id = _uid(), _uid()
    with studio_db.get_connection() as conn:
        _insert_thread(conn, thread_id)
        _insert_thread_doc(conn, doc_id, thread_id)
    row = document_for_subject_or_404(doc_id, "any-authenticated-user")
    assert row["id"] == doc_id


def test_thread_doc_nonexistent_thread_raises_404(tmp_path, monkeypatch):
    """A missing thread_id does NOT silently grant access — it must be 404."""
    _reset_db(tmp_path, monkeypatch)
    doc_id, thread_id = _uid(), _uid()
    # Doc's thread_id has no matching chat_threads row.
    with studio_db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO rag_documents
                (id, kb_id, thread_id, filename, content_type, stored_path, status,
                 num_chunks, byte_size, created_at)
            VALUES (?, NULL, ?, 'x.txt', 'text/plain', 'x.txt', 'completed', 0, 1, ?)
            """,
            (doc_id, thread_id, 1_700_000_000),
        )
    with pytest.raises(HTTPException) as exc_info:
        document_for_subject_or_404(doc_id, "alice")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Document not found"


# ── Missing document ──────────────────────────────────────────────────


def test_missing_document_raises_404(tmp_path, monkeypatch):
    """Completely absent document_id returns 404 with canonical detail."""
    _reset_db(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as exc_info:
        document_for_subject_or_404("nonexistent-id", "alice")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Document not found"


def test_empty_document_id_raises_404(tmp_path, monkeypatch):
    """Empty string document_id raises 404 rather than hitting the DB."""
    _reset_db(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as exc_info:
        document_for_subject_or_404("", "alice")
    assert exc_info.value.status_code == 404


def test_empty_subject_raises_404(tmp_path, monkeypatch):
    """Empty subject raises 404 — cannot authorize without a subject."""
    _reset_db(tmp_path, monkeypatch)
    doc_id, kb_id = _uid(), _uid()
    with studio_db.get_connection() as conn:
        _insert_kb(conn, kb_id, owner = "alice")
        _insert_kb_doc(conn, doc_id, kb_id)
    with pytest.raises(HTTPException) as exc_info:
        document_for_subject_or_404(doc_id, "")
    assert exc_info.value.status_code == 404
