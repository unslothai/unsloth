# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ingestion lifecycle tests: pending -> completed, SSE events, dedupe, delete."""

import os
import time

import pytest

from core.rag import ingestion, store
from storage import rag_db


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding = "utf-8")
    return str(path)


def _drain(job_id):
    return list(ingestion.job_events(job_id))


def _wait_completed(job_id, timeout = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = ingestion.get_job_status(job_id)
        if status and status["status"] in ("completed", "failed"):
            return status
        time.sleep(0.05)
    raise AssertionError("ingestion did not finish in time")


def test_ingestion_lifecycle_pending_to_completed(rag_home, stub_embeddings, tmp_path):
    path = _write(tmp_path, "doc.txt", "alpha bravo charlie " * 50)
    scope = store.kb_scope("K1")
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "doc.txt", path)

    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, doc_id)["status"] in {
            "pending",
            "running",
            "completed",
        }
    finally:
        conn.close()

    events = _drain(job_id)
    assert any(e["type"] == "progress" for e in events)
    assert events[-1]["type"] == "complete"
    assert events[-1]["num_chunks"] > 0

    status = _wait_completed(job_id)
    assert status["status"] == "completed"
    assert status["progress"] == 1.0

    conn = rag_db.get_connection()
    try:
        doc = store.get_document(conn, doc_id)
        assert doc["status"] == "completed"
        assert doc["num_chunks"] > 0
        assert store.search_lexical(conn, scope, "alpha", 10)
    finally:
        conn.close()


def test_ingestion_dedupe_by_hash(rag_home, stub_embeddings, tmp_path):
    path = _write(tmp_path, "doc.txt", "alpha bravo charlie")
    scope = store.kb_scope("K1")
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "doc.txt", path)
    _drain(job_id)
    _wait_completed(job_id)

    # Identical content -> same doc id, no re-ingest.
    path2 = _write(tmp_path, "copy.txt", "alpha bravo charlie")
    doc_id2, job_id2 = ingestion.start_ingestion(scope, "K1", None, "copy.txt", path2)
    events = _drain(job_id2)
    assert doc_id2 == doc_id
    assert any(e.get("deduped") for e in events)

    conn = rag_db.get_connection()
    try:
        assert len(store.list_documents(conn, scope)) == 1
    finally:
        conn.close()


def test_ingestion_reingests_when_existing_has_zero_chunks(
    rag_home, stub_embeddings, tmp_path
):
    # A prior ingest of identical bytes that yielded no chunks (e.g. a scanned PDF
    # before a vision model loaded) must re-ingest, not dedupe to the empty record.
    path = _write(tmp_path, "doc.txt", "alpha bravo charlie " * 50)
    sha = ingestion._sha256_file(path)
    scope = store.kb_scope("K1")
    conn = rag_db.get_connection()
    try:
        empty_id = store.create_document(
            conn, scope = scope, filename = "old.txt", sha256 = sha
        )
        store.set_document_status(conn, empty_id, "completed", num_chunks = 0)
    finally:
        conn.close()

    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "doc.txt", path)
    events = _drain(job_id)
    _wait_completed(job_id)

    assert not any(e.get("deduped") for e in events)  # not a dedupe -> real ingest
    assert doc_id != empty_id
    conn = rag_db.get_connection()
    try:
        docs = store.list_documents(conn, scope)
        assert len(docs) == 1  # the empty record was removed, replaced by the new one
        assert docs[0]["num_chunks"] > 0
    finally:
        conn.close()


def test_ingestion_dedupe_removes_duplicate_upload(rag_home, stub_embeddings):
    from utils.paths import ensure_dir, rag_uploads_root

    uploads = ensure_dir(rag_uploads_root())
    first_path = uploads / "doc.txt"
    duplicate_path = uploads / "copy.txt"
    first_path.write_text("alpha bravo charlie", encoding = "utf-8")
    duplicate_path.write_text("alpha bravo charlie", encoding = "utf-8")
    scope = store.project_scope("P1")

    doc_id, job_id = ingestion.start_ingestion(
        scope,
        None,
        None,
        "doc.txt",
        str(first_path),
        project_id = "P1",
    )
    _drain(job_id)
    _wait_completed(job_id)

    doc_id2, job_id2 = ingestion.start_ingestion(
        scope,
        None,
        None,
        "copy.txt",
        str(duplicate_path),
        project_id = "P1",
    )
    events = _drain(job_id2)
    assert doc_id2 == doc_id
    assert any(e.get("deduped") for e in events)
    assert first_path.exists()
    assert not duplicate_path.exists()


def test_ingestion_retry_replaces_failed_hash(rag_home, stub_embeddings):
    from utils.paths import ensure_dir, rag_uploads_root

    uploads = ensure_dir(rag_uploads_root())
    old_path = uploads / "failed.txt"
    retry_path = uploads / "retry.txt"
    old_path.write_text("alpha bravo charlie", encoding = "utf-8")
    retry_path.write_text("alpha bravo charlie", encoding = "utf-8")
    scope = store.project_scope("P1")
    sha = ingestion._sha256_file(str(old_path))

    conn = rag_db.get_connection()
    try:
        failed_id = store.create_document(
            conn,
            scope = scope,
            filename = "failed.txt",
            sha256 = sha,
            project_id = "P1",
            status = "failed",
            stored_path = str(old_path),
        )
    finally:
        conn.close()

    doc_id, job_id = ingestion.start_ingestion(
        scope,
        None,
        None,
        "retry.txt",
        str(retry_path),
        project_id = "P1",
    )
    events = _drain(job_id)
    assert doc_id != failed_id
    assert not any(e.get("deduped") for e in events)
    assert not old_path.exists()
    assert retry_path.exists()

    status = _wait_completed(job_id)
    assert status["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, failed_id) is None
        assert store.get_document(conn, doc_id)["status"] == "completed"
    finally:
        conn.close()


def test_delete_document_route_removes_stored_upload(rag_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes.rag import router
    from utils.paths import ensure_dir, rag_uploads_root

    upload = ensure_dir(rag_uploads_root()) / "delete-me.txt"
    upload.write_text("alpha bravo", encoding = "utf-8")
    scope = store.project_scope("P1")

    conn = rag_db.get_connection()
    try:
        doc_id = store.create_document(
            conn,
            scope = scope,
            filename = "delete-me.txt",
            sha256 = "delete-route-sha",
            project_id = "P1",
            status = "completed",
            stored_path = str(upload),
        )
    finally:
        conn.close()

    app = FastAPI()
    app.include_router(router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    client = TestClient(app)

    res = client.delete(f"/api/rag/documents/{doc_id}")
    assert res.status_code == 200
    assert not upload.exists()

    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, doc_id) is None
    finally:
        conn.close()


def test_get_job_status_includes_num_chunks(rag_home, stub_embeddings, tmp_path):
    # The poll/reconcile path reads num_chunks from get_job_status (the SSE complete
    # frame carries it, but a client that falls back to polling needs it here too).
    path = _write(tmp_path, "doc.txt", "alpha bravo charlie " * 50)
    scope = store.kb_scope("K1")
    _doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "doc.txt", path)
    _drain(job_id)
    _wait_completed(job_id)
    status = ingestion.get_job_status(job_id)
    assert status["status"] == "completed"
    assert status["num_chunks"] and status["num_chunks"] > 0


def test_save_upload_rejects_oversize_file(rag_home, monkeypatch):
    # A file over the cap is rejected (413) and its partial bytes are cleaned up.
    import io

    from fastapi import HTTPException

    from core.rag import config
    from routes import rag as rag_routes
    from utils.paths import rag_uploads_root

    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024)

    class _Up:
        filename = "big.txt"
        file = io.BytesIO(b"x" * 4096)

    with pytest.raises(HTTPException) as ei:
        rag_routes._save_upload(_Up())
    assert ei.value.status_code == 413
    assert list(rag_uploads_root().glob("*.txt")) == []  # partial upload removed


def test_ingestion_delete_removes_all_rows(rag_home, stub_embeddings, tmp_path):
    path = _write(tmp_path, "doc.txt", "alpha bravo charlie delta")
    scope = store.kb_scope("K1")
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "doc.txt", path)
    _drain(job_id)
    _wait_completed(job_id)

    conn = rag_db.get_connection()
    try:
        store.delete_document(conn, doc_id)
        assert store.get_document(conn, doc_id) is None
        assert store.search_lexical(conn, scope, "alpha", 10) == []
        assert store.list_documents(conn, scope) == []
    finally:
        conn.close()


def test_ingestion_rejects_unsupported_ext(rag_home, stub_embeddings, tmp_path):
    path = _write(tmp_path, "doc.xyz", "alpha")
    with pytest.raises(ValueError):
        ingestion.start_ingestion(store.kb_scope("K1"), "K1", None, "doc.xyz", path)


def test_ingestion_empty_doc_completes_with_zero_chunks(
    rag_home, stub_embeddings, tmp_path
):
    path = _write(tmp_path, "empty.txt", "   \n  ")
    scope = store.kb_scope("K1")
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "empty.txt", path)
    events = _drain(job_id)
    assert events[-1]["type"] == "complete"
    assert events[-1]["num_chunks"] == 0
    status = _wait_completed(job_id)
    assert status["status"] == "completed"


@pytest.mark.skipif(
    os.environ.get("RAG_REAL_EMBEDDER") != "1",
    reason = "set RAG_REAL_EMBEDDER=1 to run the real sentence-transformers test",
)
def test_ingestion_with_real_embedder(rag_home, tmp_path):
    path = _write(
        tmp_path, "doc.txt", "The Kestrel-9 turbine is rated at 9.5 megawatts."
    )
    scope = store.kb_scope("K1")
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "doc.txt", path)
    _drain(job_id)
    status = _wait_completed(job_id, timeout = 120.0)
    assert status["status"] == "completed"

    from core.rag import retrieval

    conn = rag_db.get_connection()
    try:
        hits = retrieval.retrieve_hybrid(
            conn, scope, "how much power does the turbine make?", k = 5
        )
        assert hits and hits[0].chunk_id == f"{doc_id}:0"
    finally:
        conn.close()
