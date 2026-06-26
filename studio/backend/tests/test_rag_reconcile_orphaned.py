# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Startup reconciliation must not strip chunks from already-completed docs.

A crash can leave an ingestion_jobs row non-terminal after the worker already
committed the document as ``completed`` with all its chunks. Reconciliation flips
the orphaned job to ``failed`` but must touch the document (and its chunks) only
when it actually transitions the document to ``failed`` -- otherwise a completed
source loses every chunk yet still reports ``completed``, so retrieval finds
nothing and dedup (``status != 'failed'``) blocks re-ingest.
"""

import math

from core.rag import store
from core.rag.chunking import Chunk
from storage import rag_db

VOCAB = ["alpha", "bravo", "charlie", "delta"]


def _embed(text):
    v = [float(text.lower().count(w)) for w in VOCAB]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _chunk(text, index = 0):
    return Chunk(
        text = text,
        token_count = len(text.split()),
        page_number = None,
        source_page_index = 0,
        chunk_index = index,
        page_char_start = 0,
        page_char_end = len(text),
    )


def _add_doc(conn, scope, doc_id, status, texts):
    store.create_document(
        conn, scope = scope, filename = f"{doc_id}.txt", sha256 = doc_id, document_id = doc_id
    )
    store.add_chunks(
        conn, scope, doc_id, [_chunk(t, i) for i, t in enumerate(texts)], [_embed(t) for t in texts]
    )
    store.set_document_status(conn, doc_id, status, num_chunks = len(texts))


def _orphan_job(
    conn,
    doc_id,
    scope,
    status = "running",
):
    conn.execute(
        "INSERT INTO ingestion_jobs(id, document_id, scope, status, stage, progress, created_at) "
        "VALUES(?,?,?,?,?,?,datetime('now'))",
        (f"job-{doc_id}", doc_id, scope, status, "embedding", 0.5),
    )
    conn.commit()


def _chunk_count(conn, doc_id):
    return conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (doc_id,)).fetchone()[0]


def _job_status(conn, doc_id):
    return conn.execute(
        "SELECT status FROM ingestion_jobs WHERE id=?", (f"job-{doc_id}",)
    ).fetchone()["status"]


def test_completed_doc_keeps_chunks_when_its_job_is_orphaned(rag_conn):
    # Worker finished the document but crashed before retiring the job row.
    _add_doc(rag_conn, "kb_a", "done", "completed", ["alpha bravo", "charlie delta"])
    _orphan_job(rag_conn, "done", "kb_a")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1

    # Document stays completed with all chunks; dedup still finds it.
    assert store.get_document(rag_conn, "done")["status"] == "completed"
    assert _chunk_count(rag_conn, "done") == 2
    assert store.document_by_hash(rag_conn, "kb_a", "done") == "done"
    # The orphaned job is reconciled to completed (not failed), so the UI's getJob
    # fallback doesn't flag a searchable document as a failed ingestion.
    assert _job_status(rag_conn, "done") == "completed"


def test_in_flight_doc_is_failed_and_its_chunks_dropped(rag_conn):
    # Partial chunks committed, document never marked terminal -> genuine orphan.
    _add_doc(rag_conn, "kb_a", "partial", "processing", ["alpha bravo"])
    _orphan_job(rag_conn, "partial", "kb_a")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1

    assert store.get_document(rag_conn, "partial")["status"] == "failed"
    assert _chunk_count(rag_conn, "partial") == 0
    # Failed doc is re-ingestible (not deduped).
    assert store.document_by_hash(rag_conn, "kb_a", "partial") is None


def test_already_failed_doc_has_its_chunks_dropped(rag_conn):
    # Worker committed chunks then marked the doc 'failed', but crashed before
    # retiring the job row. Reconcile won't re-flip the doc (already failed), but
    # its chunks must still be purged so they aren't retrievable/citable.
    _add_doc(rag_conn, "kb_a", "failed_doc", "failed", ["alpha bravo"])
    _orphan_job(rag_conn, "failed_doc", "kb_a")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1

    assert store.get_document(rag_conn, "failed_doc")["status"] == "failed"
    assert _chunk_count(rag_conn, "failed_doc") == 0
