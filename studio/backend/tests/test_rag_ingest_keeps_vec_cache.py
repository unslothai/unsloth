# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A normal upload extends the warm dense cache in place; only a real identity change moves the epoch."""

import time

import pytest

np = pytest.importorskip("numpy")

from core.rag import ingestion, store, vec_cache
from storage import rag_db


def _ingest(tmp_path, scope, name, text):
    path = tmp_path / name
    path.write_text(text, encoding = "utf-8")
    doc_id, job_id = ingestion.start_ingestion(scope, "K1", None, name, str(path))
    deadline = time.time() + 30
    while time.time() < deadline:
        status = ingestion.get_job_status(job_id)
        if status and status["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.02)
    assert status["status"] == "completed", status
    return doc_id


def _vector(conn, chunk_id):
    row = conn.execute("SELECT embedding FROM chunks_vec WHERE chunk_id=?", (chunk_id,)).fetchone()
    conn.commit()
    return np.frombuffer(row[0], dtype = np.float32).copy()


def _search(conn, scope, q, monkeypatch, cache):
    monkeypatch.setenv("RAG_VEC_CACHE", "1" if cache else "0")
    return store.search_dense(conn, scope, q, 5)


def test_ingest_appends_to_the_warm_cache_without_a_rebuild(
    rag_conn, stub_embeddings, tmp_path, monkeypatch
):
    scope = store.kb_scope("K1")
    first = _ingest(tmp_path, scope, "a.txt", "alpha bravo charlie " * 80)
    q = _vector(rag_conn, f"{first}:0")
    _search(rag_conn, scope, q, monkeypatch, True)
    assert vec_cache.drain()
    warm = vec_cache.read_epoch(rag_conn, scope)
    rag_conn.commit()
    key = vec_cache._key(store._db_file(rag_conn), scope, stub_embeddings)
    assert vec_cache._entries[key].epoch == warm
    rows_before = vec_cache._entries[key].n

    builds = []
    real_build = vec_cache._build
    monkeypatch.setattr(vec_cache, "_build", lambda k: (builds.append(k), real_build(k)))

    second = _ingest(tmp_path, scope, "b.txt", "xray yankee zulu " * 80)
    after = vec_cache.read_epoch(rag_conn, scope)
    rag_conn.commit()
    assert after[:2] == warm[:2] and after[2] > warm[2]
    entry = vec_cache._entries.get(key)
    assert entry is not None and entry.epoch == after
    added = rag_conn.execute(
        "SELECT count(*) FROM chunks WHERE document_id=?", (second,)
    ).fetchone()[0]
    rag_conn.commit()
    assert added > 0 and entry.n == rows_before + added

    q2 = _vector(rag_conn, f"{second}:0")
    seen = []
    rag_conn.set_trace_callback(seen.append)
    try:
        got = _search(rag_conn, scope, q2, monkeypatch, True)
    finally:
        rag_conn.set_trace_callback(None)
    assert not [s for s in seen if "MATCH" in s.upper()], "expected a cache hit"
    vec_cache.drain()
    assert builds == []
    want = _search(rag_conn, scope, q2, monkeypatch, False)
    assert got[0][0] == f"{second}:0"
    assert [c for c, _ in got] == [c for c, _ in want]
    assert all(abs(a[1] - b[1]) < 1e-5 for a, b in zip(got, want))


def test_only_a_changed_embedding_model_moves_the_epoch(rag_conn):
    scope = store.kb_scope("K1")
    doc = store.create_document(
        rag_conn, scope = scope, filename = "f", sha256 = "h", embedding_model = "st:m"
    )

    def epoch():
        value = vec_cache.read_epoch(rag_conn, scope)
        rag_conn.commit()
        return value[0]

    e0 = epoch()
    store.set_document_embedding_model(rag_conn, doc, "st:m")
    assert epoch() == e0
    store.set_document_embedding_model(rag_conn, doc, "llama:m")
    assert epoch() == e0 + 1
    assert store.get_document(rag_conn, doc)["embedding_model"] == "llama:m"
    legacy = store.create_document(rag_conn, scope = scope, filename = "g", sha256 = "i")
    store.set_document_embedding_model(rag_conn, legacy, "st:m")
    assert epoch() == e0 + 2
    rag_conn.execute(
        "UPDATE documents SET embedding_model=? WHERE id=?",
        (rag_db.VECTORS_DROPPED + "st:m", doc),
    )
    rag_conn.commit()
    e1 = epoch()
    store.set_document_embedding_model(rag_conn, doc, "st:m")
    assert epoch() == e1 + 1
    assert store.get_document(rag_conn, doc)["embedding_model"] == "st:m"
