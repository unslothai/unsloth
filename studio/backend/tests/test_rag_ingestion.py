# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ingestion lifecycle tests: pending -> completed, SSE events, dedupe, delete.

Uses the ``stub_embeddings`` fixture so no sentence-transformers model is
downloaded. There is also one optional test that exercises the real embedder,
guarded by RAG_REAL_EMBEDDER=1.
"""

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
    """Collect all SSE events for a job until the stream ends."""
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

    # Document starts pending.
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, doc_id)["status"] == "pending"
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
        # Chunks are searchable after ingestion.
        assert store.search_lexical(conn, scope, "alpha", 10)
    finally:
        conn.close()


def test_ingestion_dedupe_by_hash(rag_home, stub_embeddings, tmp_path):
    path = _write(tmp_path, "doc.txt", "alpha bravo charlie")
    scope = store.kb_scope("K1")
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, "doc.txt", path)
    _drain(job_id)
    _wait_completed(job_id)

    # Re-uploading identical content returns the same doc id, no re-ingest.
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
