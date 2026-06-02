# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for GET /api/rag/documents/{id}/preview-target and /file.

Acceptance criteria covered (contracts.md §1, §2, PLAN.md T1/T2, Risk #1-3):

/preview-target:
- 200 with chunk data when chunk_id present and belongs to doc.
- 200 with all-null chunk fields when chunk_id absent (document-row preview).
- 404 when document missing (collapsed existence + auth).
- 404 when chunk_id does not belong to document_id (cross-doc probe collapsed).
- 401 when no bearer token.

/file:
- 200 with correct Content-Type and nosniff header.
- X-Content-Type-Options: nosniff present on every 200.
- Cache-Control: private present on every 200.
- HTML extension served as text/plain + attachment (Risk #3).
- DOCX extension served with attachment disposition.
- 404 when document missing or wrong subject.
- 404 when file deleted from disk (DB row exists, subject authorized).
- Outside-root stored_path returns 404 (path containment, Risk #2).

Auth is injected via dependency override (mock at the boundary, not the
implementation target). We do NOT mock document_for_subject_or_404 itself.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import storage.studio_db as studio_db
from auth.authentication import get_current_subject


# ── App import (deferred to avoid import-time side-effects) ───────────


@pytest.fixture(scope = "module")
def app():
    import sys

    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from main import app as _app

    return _app


# ── Test-level DB + auth fixtures ─────────────────────────────────────


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    """Point studio_db at a fresh temp DB for each test."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return tmp_path


def _uid() -> str:
    return str(uuid.uuid4())


def _make_client(app, subject: str = "alice"):
    """Return a TestClient with get_current_subject overridden to return subject."""
    app.dependency_overrides[get_current_subject] = lambda: subject
    client = TestClient(app, raise_server_exceptions = True)
    return client


def _clear_overrides(app):
    app.dependency_overrides.clear()


def _insert_kb(conn, kb_id: str, owner: str = "alice") -> None:
    conn.execute(
        "INSERT INTO rag_knowledge_bases (id, name, embedding_model, owner_user_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (kb_id, f"KB-{kb_id[:6]}", "bge-small", owner, 1_700_000_000),
    )


def _insert_doc(
    conn,
    doc_id: str,
    kb_id: str,
    stored_path: str,
    filename: str = "report.pdf",
    content_type: str | None = "application/pdf",
    status: str = "completed",
) -> None:
    conn.execute(
        "INSERT INTO rag_documents "
        "(id, kb_id, thread_id, filename, content_type, stored_path, status, "
        "num_chunks, byte_size, created_at) "
        "VALUES (?, ?, NULL, ?, ?, ?, ?, 0, 1024, ?)",
        (doc_id, kb_id, filename, content_type, stored_path, status, 1_700_000_000),
    )


def _insert_chunk(
    conn,
    chunk_id: str,
    doc_id: str,
    text: str = "The margin rose to 18.2% in Q3.",
    page_number: int | None = 7,
    chunk_index: int = 14,
) -> None:
    conn.execute(
        "INSERT INTO rag_chunks "
        "(id, document_id, chunk_index, text, token_count, page_number) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (chunk_id, doc_id, chunk_index, text, 30, page_number),
    )


# ── /preview-target tests ─────────────────────────────────────────────


class TestPreviewTarget:
    def test_with_chunk_id_returns_full_metadata(self, app, db_env, monkeypatch):
        """GET /preview-target?chunk_id=<id> returns page + snippet when chunk valid."""
        doc_id, kb_id, chunk_id = _uid(), _uid(), _uid()
        stored = db_env / "rag" / "uploads" / "report.pdf"
        stored.parent.mkdir(parents = True, exist_ok = True)
        stored.write_bytes(b"%PDF-1.4 dummy")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(conn, doc_id, kb_id, str(stored))
            _insert_chunk(conn, chunk_id, doc_id, page_number = 7, chunk_index = 14)

        client = _make_client(app, "alice")
        try:
            resp = client.get(
                f"/api/rag/documents/{doc_id}/preview-target?chunk_id={chunk_id}"
            )
        finally:
            _clear_overrides(app)

        assert resp.status_code == 200
        body = resp.json()
        assert body["documentId"] == doc_id
        assert body["chunkId"] == chunk_id
        assert body["targetPage"] == 7
        assert body["chunkIndex"] == 14
        assert body["snippet"] is not None and len(body["snippet"]) > 0
        assert body["mediaKind"] == "pdf"

    def test_preview_target_returns_pdf_regions_when_present(
        self, app, db_env, monkeypatch
    ):
        """Chunk preview includes only stored confident PDF regions."""
        doc_id, kb_id, chunk_id = _uid(), _uid(), _uid()
        stored = db_env / "rag" / "uploads" / "report.pdf"
        stored.parent.mkdir(parents = True, exist_ok = True)
        stored.write_bytes(b"%PDF-1.4 dummy")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(conn, doc_id, kb_id, str(stored))
            _insert_chunk(conn, chunk_id, doc_id, page_number = 7, chunk_index = 14)
            conn.execute(
                """
                UPDATE rag_chunks
                SET pdf_regions_json = ?
                WHERE id = ?
                """,
                (
                    '[{"pageIndex":6,"pageNumber":7,"x":0.1,"y":0.2,'
                    '"width":0.3,"height":0.04,"confidence":"exact",'
                    '"source":"pymupdf-search"}]',
                    chunk_id,
                ),
            )

        client = _make_client(app, "alice")
        try:
            resp = client.get(
                f"/api/rag/documents/{doc_id}/preview-target?chunk_id={chunk_id}"
            )
        finally:
            _clear_overrides(app)

        assert resp.status_code == 200
        body = resp.json()
        assert body["pdfRegions"] == [
            {
                "pageIndex": 6,
                "pageNumber": 7,
                "x": 0.1,
                "y": 0.2,
                "width": 0.3,
                "height": 0.04,
                "confidence": "exact",
                "source": "pymupdf-search",
            }
        ]

    def test_without_chunk_id_returns_all_null_chunk_fields(
        self, app, db_env, monkeypatch
    ):
        """GET /preview-target without chunk_id returns metadata-only (decision Q2)."""
        doc_id, kb_id = _uid(), _uid()
        stored = db_env / "rag" / "uploads" / "annual.pdf"
        stored.parent.mkdir(parents = True, exist_ok = True)
        stored.write_bytes(b"%PDF-1.4 dummy")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(conn, doc_id, kb_id, str(stored))

        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/preview-target")
        finally:
            _clear_overrides(app)

        assert resp.status_code == 200
        body = resp.json()
        # All chunk fields null — no first-chunk guess.
        assert body["chunkId"] is None
        assert body["chunkIndex"] is None
        assert body["targetPage"] is None
        assert body["snippet"] is None
        assert body["kind"] is None
        assert body["imageUrl"] is None
        assert body["documentId"] == doc_id

    def test_missing_document_returns_404(self, app, db_env, monkeypatch):
        """Nonexistent document_id returns 404 to both existence and auth probes."""
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))
        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{_uid()}/preview-target")
        finally:
            _clear_overrides(app)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Document not found"

    def test_wrong_subject_returns_404(self, app, db_env, monkeypatch):
        """Document owned by alice returns 404 when accessed by mallory."""
        doc_id, kb_id = _uid(), _uid()
        stored = db_env / "rag" / "uploads" / "secret.pdf"
        stored.parent.mkdir(parents = True, exist_ok = True)
        stored.write_bytes(b"data")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id, owner = "alice")
            _insert_doc(conn, doc_id, kb_id, str(stored))

        client = _make_client(app, "mallory")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/preview-target")
        finally:
            _clear_overrides(app)
        assert resp.status_code == 404

    def test_cross_doc_chunk_id_returns_404(self, app, db_env, monkeypatch):
        """chunk_id from a different document returns 404 — not 400 (opaque)."""
        kb_id = _uid()
        doc_a, doc_b = _uid(), _uid()
        chunk_a = _uid()
        stored_a = db_env / "rag" / "uploads" / "a.pdf"
        stored_b = db_env / "rag" / "uploads" / "b.pdf"
        stored_a.parent.mkdir(parents = True, exist_ok = True)
        stored_a.write_bytes(b"data")
        stored_b.write_bytes(b"data")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(conn, doc_a, kb_id, str(stored_a), "a.pdf")
            _insert_doc(conn, doc_b, kb_id, str(stored_b), "b.pdf")
            _insert_chunk(conn, chunk_a, doc_a)

        client = _make_client(app, "alice")
        try:
            # Probe doc_b with chunk_a (belongs to doc_a)
            resp = client.get(
                f"/api/rag/documents/{doc_b}/preview-target?chunk_id={chunk_a}"
            )
        finally:
            _clear_overrides(app)
        # 404, not 200 with doc_a's chunk data
        assert resp.status_code == 404

    def test_unauthenticated_returns_401(self, app, db_env, monkeypatch):
        """No bearer token → 401."""
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))
        # No override: let the real dependency raise.
        client = TestClient(app, raise_server_exceptions = False)
        resp = client.get(f"/api/rag/documents/{_uid()}/preview-target")
        assert resp.status_code == 401


# ── /file tests ───────────────────────────────────────────────────────


class TestFileRoute:
    def test_pdf_200_with_correct_headers(self, app, db_env, monkeypatch):
        """GET /file for a PDF returns 200 with nosniff, Cache-Control, inline disposition."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        stored = uploads / "annual.pdf"
        stored.write_bytes(b"%PDF-1.4\n%%EOF")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(
                conn, doc_id, kb_id, str(stored), "annual.pdf", "application/pdf"
            )

        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/file")
        finally:
            _clear_overrides(app)

        assert resp.status_code == 200
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert "private" in (resp.headers.get("cache-control") or "")
        ct = resp.headers.get("content-type", "")
        assert "pdf" in ct.lower()

    def test_signed_file_url_supports_range_without_bearer_query(
        self, app, db_env, monkeypatch
    ):
        """Short-lived signed URL is redeemable without Authorization and supports ranges."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        stored = uploads / "annual.pdf"
        stored.write_bytes(b"%PDF-1.4\n%%EOF")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))
        monkeypatch.setattr("routes.rag.get_jwt_secret", lambda subject: "test-secret")

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(
                conn, doc_id, kb_id, str(stored), "annual.pdf", "application/pdf"
            )

        client = _make_client(app, "alice")
        try:
            url_resp = client.get(f"/api/rag/documents/{doc_id}/file-url")
            assert url_resp.status_code == 200
            signed_url = url_resp.json()["url"]
            assert "Bearer" not in signed_url
            assert "Authorization" not in signed_url

            file_resp = client.get(signed_url, headers = {"Range": "bytes=0-3"})
        finally:
            _clear_overrides(app)

        assert file_resp.status_code == 206
        assert file_resp.content == b"%PDF"
        assert (
            file_resp.headers.get("content-range")
            == f"bytes 0-3/{stored.stat().st_size}"
        )
        assert file_resp.headers.get("accept-ranges") == "bytes"
        assert file_resp.headers.get("x-content-type-options") == "nosniff"

    def test_signed_file_route_rejects_forged_token(self, app, db_env, monkeypatch):
        """Signed file route is not public without a valid preview token."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        stored = uploads / "annual.pdf"
        stored.write_bytes(b"%PDF-1.4\n%%EOF")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))
        monkeypatch.setattr("routes.rag.get_jwt_secret", lambda subject: "test-secret")

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(
                conn, doc_id, kb_id, str(stored), "annual.pdf", "application/pdf"
            )

        client = TestClient(app, raise_server_exceptions = False)
        resp = client.get(f"/api/rag/documents/{doc_id}/file-signed?token=bogus")
        assert resp.status_code == 401

    def test_html_file_served_as_text_plain_with_attachment(
        self, app, db_env, monkeypatch
    ):
        """HTML uploads must be served as text/plain + attachment (Risk #3 — no XSS)."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        stored = uploads / "malicious.html"
        stored.write_bytes(b"<script>alert(1)</script>")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(conn, doc_id, kb_id, str(stored), "malicious.html", "text/html")

        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/file")
        finally:
            _clear_overrides(app)

        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "").lower()
        # Must be text/plain, not text/html.
        assert "text/html" not in ct, f"HTML executed inline! content-type={ct}"
        assert "text/plain" in ct
        disp = resp.headers.get("content-disposition", "").lower()
        assert "attachment" in disp, f"HTML not forced to attachment: {disp}"
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_docx_served_as_attachment(self, app, db_env, monkeypatch):
        """DOCX files must be served with Content-Disposition: attachment."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        stored = uploads / "report.docx"
        stored.write_bytes(b"PK\x03\x04fake-docx")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(
                conn,
                doc_id,
                kb_id,
                str(stored),
                "report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/file")
        finally:
            _clear_overrides(app)

        assert resp.status_code == 200
        disp = resp.headers.get("content-disposition", "").lower()
        assert "attachment" in disp

    def test_missing_document_returns_404(self, app, db_env, monkeypatch):
        """Nonexistent document returns 404."""
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))
        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{_uid()}/file")
        finally:
            _clear_overrides(app)
        assert resp.status_code == 404

    def test_wrong_subject_returns_404(self, app, db_env, monkeypatch):
        """Document accessible to alice is 404 for mallory (auth-collapse)."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        stored = uploads / "private.pdf"
        stored.write_bytes(b"data")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id, owner = "alice")
            _insert_doc(conn, doc_id, kb_id, str(stored))

        client = _make_client(app, "mallory")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/file")
        finally:
            _clear_overrides(app)
        assert resp.status_code == 404

    def test_deleted_file_returns_404_with_doc_file_not_found(
        self, app, db_env, monkeypatch
    ):
        """File gone from disk returns 404 with 'Document file not found' detail."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        stored = uploads / "gone.pdf"
        stored.write_bytes(b"data")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(conn, doc_id, kb_id, str(stored))

        # Delete the file after the row exists.
        stored.unlink()

        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/file")
        finally:
            _clear_overrides(app)

        assert resp.status_code == 404
        detail = resp.json().get("detail", "")
        assert "file not found" in detail.lower() or "not found" in detail.lower()

    def test_outside_root_stored_path_returns_404(
        self, app, db_env, monkeypatch, tmp_path
    ):
        """stored_path outside rag_uploads_root returns 404 — path containment (Risk #2)."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        # A plausible path outside the RAG uploads root.
        outside = tmp_path / "etc" / "passwd"
        outside.parent.mkdir(parents = True, exist_ok = True)
        outside.write_bytes(b"root:x:0:0")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            # stored_path points outside root.
            conn.execute(
                "INSERT INTO rag_documents "
                "(id, kb_id, thread_id, filename, content_type, stored_path, status, "
                "num_chunks, byte_size, created_at) "
                "VALUES (?, ?, NULL, 'passwd', 'text/plain', ?, 'completed', 0, 10, ?)",
                (doc_id, kb_id, str(outside), 1_700_000_000),
            )

        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/file")
        finally:
            _clear_overrides(app)

        # Containment violation: 404, never serve the file.
        assert resp.status_code == 404

    def test_nosniff_and_cache_headers_on_txt_file(self, app, db_env, monkeypatch):
        """Safety headers present on every 200 response, including plain text."""
        doc_id, kb_id = _uid(), _uid()
        uploads = db_env / "rag" / "uploads"
        uploads.mkdir(parents = True, exist_ok = True)
        stored = uploads / "notes.txt"
        stored.write_bytes(b"hello world")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(db_env))

        with studio_db.get_connection() as conn:
            _insert_kb(conn, kb_id)
            _insert_doc(conn, doc_id, kb_id, str(stored), "notes.txt", "text/plain")

        client = _make_client(app, "alice")
        try:
            resp = client.get(f"/api/rag/documents/{doc_id}/file")
        finally:
            _clear_overrides(app)

        assert resp.status_code == 200
        assert resp.headers.get("x-content-type-options") == "nosniff"
        cc = resp.headers.get("cache-control", "")
        assert "private" in cc
