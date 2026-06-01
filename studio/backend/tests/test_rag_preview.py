# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDF region locators + citation preview routes.

A small PDF is generated in-memory with PyMuPDF so the locator's
``page.search_for`` runs against a real rendered page, then ingested through the
real threaded pipeline (with stubbed embeddings for speed). Verifies that chunks
carry highlight regions and that the preview-target / signed-file routes work.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("sqlite_vec")


def _make_pdf(path) -> None:
    import pymupdf

    doc = pymupdf.open()
    body = (
        "BERT is designed to pre-train deep bidirectional representations.\n"
        "The two pre-training objectives are masked language modeling and next "
        "sentence prediction.\n"
        "The Transformer base model uses eight attention heads in each layer.\n"
    )
    for _ in range(3):  # a few pages of the same distinctive text
        page = doc.new_page()
        page.insert_text((72, 72), body, fontsize = 11)
    doc.save(str(path))
    doc.close()


def _ingest(home, pdf_path):
    from core.rag import ingestion, store
    from storage import rag_db

    conn = rag_db.get_connection()
    kb_id = store.create_kb(conn, name = "kb")
    conn.close()
    doc_id, job_id = ingestion.start_ingestion(
        store.kb_scope(kb_id), kb_id, None, "doc.pdf", str(pdf_path)
    )
    t0 = time.time()
    while time.time() - t0 < 30:
        s = ingestion.get_job_status(job_id)
        if s and s["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert s and s["status"] == "completed", s
    return kb_id, doc_id


def test_chunks_carry_pdf_regions(rag_home, stub_embeddings):
    from utils.paths import ensure_dir, rag_uploads_root

    pdf = ensure_dir(rag_uploads_root()) / "doc.pdf"
    _make_pdf(pdf)
    kb_id, doc_id = _ingest(rag_home, pdf)

    from storage import rag_db

    conn = rag_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT pdf_regions_json FROM chunks WHERE document_id=?", (doc_id,)
        ).fetchall()
        stored_path = conn.execute(
            "SELECT stored_path FROM documents WHERE id=?", (doc_id,)
        ).fetchone()["stored_path"]
    finally:
        conn.close()

    assert rows, "no chunks were stored"
    assert stored_path and stored_path.endswith(".pdf")
    with_regions = [r for r in rows if r["pdf_regions_json"]]
    assert with_regions, "expected at least one chunk with PDF highlight regions"
    import json

    region = json.loads(with_regions[0]["pdf_regions_json"])[0]
    for key in ("pageIndex", "x", "y", "width", "height"):
        assert key in region
        if key in ("x", "y", "width", "height"):
            assert 0.0 <= region[key] <= 1.0


def test_preview_routes_and_signed_file(rag_home, stub_embeddings):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes.rag import router

    from utils.paths import ensure_dir, rag_uploads_root

    pdf = ensure_dir(rag_uploads_root()) / "doc.pdf"
    _make_pdf(pdf)
    kb_id, doc_id = _ingest(rag_home, pdf)

    app = FastAPI()
    app.include_router(router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    c = TestClient(app)

    res = c.post(
        "/api/rag/search",
        json = {
            "query": "masked language modeling next sentence",
            "kb_id": kb_id,
            "mode": "lexical",
        },
    ).json()["results"]
    assert res
    chunk_id = res[0]["chunkId"]

    pt = c.get(
        f"/api/rag/documents/{doc_id}/preview-target", params = {"chunk_id": chunk_id}
    ).json()
    assert pt["mediaKind"] == "pdf"
    assert pt["text"]

    url = c.get(f"/api/rag/documents/{doc_id}/file-url").json()["url"]
    full = c.get(url)
    assert full.status_code == 200 and full.content[:4] == b"%PDF"
    rng = c.get(url, headers = {"Range": "bytes=0-99"})
    assert rng.status_code in (200, 206)
    assert (
        c.get(
            f"/api/rag/documents/{doc_id}/file-signed",
            params = {"token": "bad.token.sig"},
        ).status_code
        == 401
    )


def test_sign_verify_roundtrip(rag_home):
    from routes import rag as rag_routes

    tok = rag_routes._sign_document("doc-123")
    assert rag_routes._verify_document_token(tok) == "doc-123"
    assert (
        rag_routes._verify_document_token("doc-123.0.deadbeef") is None
    )  # expired/bad
    assert rag_routes._verify_document_token("garbage") is None
