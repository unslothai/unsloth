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


def test_a_crashed_edit_releases_the_claim_on_the_source_it_replaced(rag_conn):
    """An edit claims its original with status='running' and the worker hands that claim back
    on every non-successful exit -- but a crash never reaches that finally, and the
    relationship lived only in the dead process. The replacement row records it, so startup
    is the last thing able to release the original; left claimed it polls as indexing forever
    and refuses both a retry and a removal."""
    _add_doc(rag_conn, "kb_a", "original", "running", ["alpha bravo"])
    store.create_document(
        rag_conn,
        scope = "kb_a",
        filename = "edited.txt",
        sha256 = "edited",
        document_id = "edited",
        replaces_document_id = "original",
    )
    _orphan_job(rag_conn, "edited", "kb_a")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1

    # The replacement never landed, so it is failed and carries no citable chunks...
    assert store.get_document(rag_conn, "edited")["status"] == "failed"
    assert _chunk_count(rag_conn, "edited") == 0
    # ...and the source it was replacing is editable again, with its own chunks intact.
    assert store.get_document(rag_conn, "original")["status"] == "completed"
    assert _chunk_count(rag_conn, "original") == 1


def test_a_completed_replacement_retires_the_source_it_replaced(rag_conn):
    """A crash between marking the replacement completed and retiring the original leaves
    two finished documents. Releasing the claim and keeping both would publish the source
    twice, so recovery finishes the job the worker had all but done: the replacement is a
    chunked, completed document, so it wins."""
    _add_doc(rag_conn, "kb_a", "original", "running", ["alpha bravo"])
    _add_doc(rag_conn, "kb_a", "edited", "completed", ["charlie delta"])
    rag_conn.execute("UPDATE documents SET replaces_document_id='original' WHERE id='edited'")
    rag_conn.commit()
    _orphan_job(rag_conn, "edited", "kb_a")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1

    # Exactly one survives, and it is the edited one with its chunks.
    assert store.get_document(rag_conn, "original") is None
    assert _chunk_count(rag_conn, "original") == 0
    assert store.get_document(rag_conn, "edited")["status"] == "completed"
    assert _chunk_count(rag_conn, "edited") == 1
    assert _job_status(rag_conn, "edited") == "completed"


def test_recovery_lets_a_delete_beat_a_completed_replacement(rag_conn):
    """The original can be deleted through Settings or another backend while the crashed
    job sits unreconciled. Publishing the replacement then brings back a source the user
    removed, under a new id -- so the delete wins here exactly as it does in the worker."""
    _add_doc(rag_conn, "kb_a", "edited", "completed", ["charlie delta"])
    # No 'original' row: it was deleted after the crash.
    rag_conn.execute("UPDATE documents SET replaces_document_id='original' WHERE id='edited'")
    rag_conn.commit()
    _orphan_job(rag_conn, "edited", "kb_a")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1

    assert store.get_document(rag_conn, "edited") is None, "a deleted source came back"
    assert _chunk_count(rag_conn, "edited") == 0
    assert _job_status(rag_conn, "edited") == "cancelled"


def test_recovery_removes_the_retired_source_file(rag_conn, tmp_path, monkeypatch):
    """The row holding the path is deleted here and nothing sweeps the uploads root, so a
    file left behind is leaked for good."""
    import utils.paths

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(utils.paths, "rag_uploads_root", lambda: uploads)
    retired = uploads / "original.txt"
    retired.write_text("body", encoding = "utf-8")

    _add_doc(rag_conn, "kb_a", "original", "running", ["alpha bravo"])
    rag_conn.execute("UPDATE documents SET stored_path=? WHERE id='original'", (str(retired),))
    _add_doc(rag_conn, "kb_a", "edited", "completed", ["charlie delta"])
    rag_conn.execute("UPDATE documents SET replaces_document_id='original' WHERE id='edited'")
    rag_conn.commit()
    _orphan_job(rag_conn, "edited", "kb_a")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1

    assert store.get_document(rag_conn, "original") is None
    assert not retired.exists(), "the retired source's file was left behind"


def test_reconcile_leaves_an_unclaimed_replaced_source_alone(rag_conn):
    # The release is guarded on 'running': a replaced document that was never claimed (the
    # stale-embedder dedupe path re-ingests a 'completed' row) must not be rewritten.
    _add_doc(rag_conn, "kb_a", "prior", "failed", [])
    store.create_document(
        rag_conn,
        scope = "kb_a",
        filename = "redo.txt",
        sha256 = "redo",
        document_id = "redo",
        replaces_document_id = "prior",
    )
    _orphan_job(rag_conn, "redo", "kb_a")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1
    assert store.get_document(rag_conn, "prior")["status"] == "failed"


def test_live_foreign_lease_is_preserved_then_reconciled_after_expiry(rag_conn):
    _add_doc(rag_conn, "kb_a", "foreign", "processing", ["alpha bravo"])
    _orphan_job(rag_conn, "foreign", "kb_a")
    rag_conn.execute(
        "INSERT INTO rag_job_leases(kind, job_id, owner_id, expires_at) "
        "VALUES('ingestion', 'job-foreign', 'other-backend', '9999-12-31T00:00:00+00:00')"
    )
    rag_conn.commit()

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 0
    assert store.get_document(rag_conn, "foreign")["status"] == "processing"
    assert _job_status(rag_conn, "foreign") == "running"
    assert _chunk_count(rag_conn, "foreign") == 1

    rag_conn.execute(
        "UPDATE rag_job_leases SET expires_at='2000-01-01T00:00:00+00:00' "
        "WHERE kind='ingestion' AND job_id='job-foreign'"
    )
    rag_conn.commit()

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 1
    assert store.get_document(rag_conn, "foreign")["status"] == "failed"
    assert _job_status(rag_conn, "foreign") == "failed"
    assert _chunk_count(rag_conn, "foreign") == 0
    assert (
        rag_conn.execute(
            "SELECT 1 FROM rag_job_leases WHERE kind='ingestion' AND job_id='job-foreign'"
        ).fetchone()
        is None
    )


def test_cancelled_job_is_terminal_and_survives_a_restart(rag_conn):
    # The worker cancelled itself because the document was deleted mid-ingestion.
    # A restart must leave that verdict alone: rewriting it to 'failed' reports a
    # deliberate cancellation to the UI's getJob fallback as an indexing failure.
    _add_doc(rag_conn, "kb_a", "cancelled_doc", "processing", ["alpha bravo"])
    _orphan_job(rag_conn, "cancelled_doc", "kb_a", status = "cancelled")

    assert rag_db.reconcile_orphaned_ingestion_jobs() == 0

    assert _job_status(rag_conn, "cancelled_doc") == "cancelled"
