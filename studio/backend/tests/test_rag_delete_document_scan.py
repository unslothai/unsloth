# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Document delete scans chunks_fts/chunks_vec once and removes exactly what per-chunk deletes did."""

import random
import sqlite3

import pytest

from core.rag import store
from core.rag.chunking import Chunk
from storage import rag_db

# python.org macOS builds lack loadable SQLite extensions, so vec0 cannot load even with sqlite-vec.
pytestmark = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE or not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason = "sqlite-vec is not installed or this sqlite3 cannot load extensions",
)

DIM = 4
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel".split()


def _chunk(text, index):
    return Chunk(
        text = text,
        token_count = len(text.split()),
        page_number = None,
        source_page_index = 0,
        chunk_index = index,
        page_char_start = 0,
        page_char_end = len(text),
    )


def _vec(rng):
    return [rng.random() + 0.01 for _ in range(DIM)]


def _add(conn, rng, scope, doc_id, texts):
    store.create_document(
        conn,
        scope = scope,
        filename = doc_id,
        sha256 = doc_id,
        document_id = doc_id,
        status = "ready",
        created_at = "2026-01-01T00:00:00",
    )
    if texts:
        store.add_chunks(
            conn,
            scope,
            doc_id,
            [_chunk(t, i) for i, t in enumerate(texts)],
            [_vec(rng) for _ in texts],
        )


def _populate(
    conn,
    docs = 30,
    per_doc = 6,
    seed = 0,
):
    rng = random.Random(seed)
    scopes = ["kb:a", "kb:b", "thread:t"]
    for d in range(docs):
        texts = [
            "shared alpha text" if rng.random() < 0.2 else " ".join(rng.choices(WORDS, k = 8))
            for _ in range(rng.randint(0, per_doc))
        ]
        _add(conn, rng, scopes[d % 3], f"doc{d}", texts)
    # Edge rows: re-added chunk, NULL-scope fts row, int chunk_id beside its text twin, cross-scope vec row, orphans.
    store.add_chunks(conn, "kb:b", "doc1", [_chunk("readded alpha", 0)], [_vec(rng)])
    conn.execute(
        "INSERT INTO chunks_fts(text, chunk_id, scope) VALUES('null scope alpha', 'doc4:0', NULL)"
    )
    _add(conn, rng, "kb:a", "77", [])
    conn.execute(
        "INSERT INTO chunks(id, document_id, scope, chunk_index, text) "
        "VALUES('5', '77', 'kb:a', 0, 'numeric alpha')"
    )
    conn.execute("INSERT INTO chunks_fts(text, chunk_id, scope) VALUES('numeric alpha', 5, 'kb:a')")
    conn.execute(
        "INSERT INTO chunks_fts(text, chunk_id, scope) VALUES('numeric alpha', '5', 'kb:a')"
    )
    conn.execute(
        "INSERT INTO chunks_vec(scope, chunk_id, embedding) VALUES('kb:b', 'doc6:0', ?)",
        (store._f32(_vec(rng)),),
    )
    conn.execute(
        "INSERT INTO chunks_fts(text, chunk_id, scope) VALUES('ghost alpha', 'ghost:0', 'kb:a')"
    )
    conn.execute(
        "INSERT INTO chunks_vec(scope, chunk_id, embedding) VALUES('kb:a', 'ghost:0', ?)",
        (store._f32(_vec(rng)),),
    )
    conn.commit()


def _dump(conn):
    return [
        [tuple(r) for r in conn.execute(sql)]
        for sql in (
            "SELECT * FROM documents ORDER BY id",
            "SELECT * FROM chunks ORDER BY rowid",
            "SELECT rowid, text, chunk_id, typeof(chunk_id), scope FROM chunks_fts ORDER BY rowid",
            "SELECT rowid, scope, chunk_id, embedding FROM chunks_vec ORDER BY rowid",
        )
    ]


def _per_chunk_delete(conn, document_id, *, keep_document):
    """The previous implementation: one chunk_id=? delete per chunk on each index."""
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,))]
    for chunk_id in ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
        conn.execute("DELETE FROM chunks_vec WHERE chunk_id=?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
    if not keep_document:
        conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    conn.commit()


def _copy(conn):
    import sqlite_vec

    other = sqlite3.connect(":memory:")
    other.row_factory = sqlite3.Row
    other.enable_load_extension(True)
    sqlite_vec.load(other)
    other.enable_load_extension(False)
    conn.backup(other)
    return other


def test_delete_removes_the_same_rows_as_per_chunk_deletes(rag_conn):
    conn = rag_conn
    _populate(conn)
    reference = _copy(conn)
    try:
        order = ["77", "doc1", "doc4", "doc6", "missing", "doc0"] + [
            f"doc{d}" for d in range(7, 30)
        ]
        for i, doc_id in enumerate(order):
            keep = i % 3 == 0
            if keep:
                rag_db._delete_document_chunks(conn, doc_id)
                conn.commit()
            else:
                store.delete_document(conn, doc_id)
            _per_chunk_delete(reference, doc_id, keep_document = keep)
            assert _dump(conn) == _dump(reference), doc_id
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
        for c in (conn, reference):
            _add(c, random.Random(1), "kb:a", "tail", ["tail alpha", "tail bravo"])
        assert _dump(conn) == _dump(reference)
        assert store.search_lexical(conn, "kb:a", "alpha", 20) == store.search_lexical(
            reference, "kb:a", "alpha", 20
        )
    finally:
        reference.close()


def _vm_steps(conn, fn):
    steps = [0]

    def tick():
        steps[0] += 1

    conn.set_progress_handler(tick, 1)
    try:
        fn()
    finally:
        conn.set_progress_handler(None, 1)
    return steps[0]


def test_delete_cost_does_not_scale_with_chunks_times_corpus(rag_conn):
    """Per-chunk deletes full-scanned both indexes, so a 40-chunk doc cost ~40x a 1-chunk one."""
    conn = rag_conn
    rng = random.Random(0)
    for d in range(40):
        _add(
            conn, rng, "kb:a", f"filler{d}", [" ".join(rng.choices(WORDS, k = 8)) for _ in range(20)]
        )
    _add(conn, rng, "kb:a", "small", ["alpha one"])
    _add(conn, rng, "kb:a", "large", [f"alpha {i}" for i in range(40)])

    small = _vm_steps(conn, lambda: store.delete_document(conn, "small"))
    large = _vm_steps(conn, lambda: store.delete_document(conn, "large"))
    assert large < 3 * small, (small, large)
    assert (
        conn.execute("SELECT count(*) FROM chunks_fts WHERE chunk_id LIKE 'large:%'").fetchone()[0]
        == 0
    )
    assert (
        conn.execute("SELECT count(*) FROM chunks_vec WHERE chunk_id LIKE 'large:%'").fetchone()[0]
        == 0
    )


def _scope_fixture(conn, rng):
    conn.execute(
        "INSERT INTO linked_folders(id, scope_type, scope_id, scope, path, name, delete_remove_index, "
        "status, created_at, updated_at) VALUES('F1','kb','keep','kb:keep','/tmp/x','x',1,'idle',?,?)",
        ("2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    for d in range(12):
        _add(conn, rng, "kb:gone", f"gone{d}", [f"alpha gone {d} {i}" for i in range(d % 4 * 3)])
        _add(conn, rng, "kb:keep", f"keep{d}", [f"alpha keep {d} {i}" for i in range(3)])
        _add(
            conn, rng, "kb:keep", f"linked{d}", [f"alpha linked {d} {i}" for i in range(d % 3 + 1)]
        )
        conn.execute(
            "INSERT INTO linked_folder_files(folder_id, relative_path, size_bytes, mtime_ns, "
            "document_id, synced_at) VALUES('F1', ?, 1, 1, ?, '2026-01-01T00:00:00')",
            (f"r{d}.txt", f"linked{d}"),
        )
    conn.execute(
        "INSERT INTO linked_folder_retired_scopes(scope, retired_at) "
        "VALUES('kb:gone', '2026-01-01T00:00:00')"
    )
    conn.commit()


def _full_dump(conn):
    return _dump(conn) + [
        [tuple(r) for r in conn.execute(sql)]
        for sql in (
            "SELECT * FROM rag_vec_epoch ORDER BY scope",
            "SELECT * FROM linked_folder_files ORDER BY folder_id, relative_path",
            "SELECT id FROM linked_folders ORDER BY id",
        )
    ]


@pytest.mark.parametrize("op", ["scope", "folder"])
def test_retired_scope_and_folder_deletes_scan_indexes_once(rag_conn, monkeypatch, op):
    from core.rag import folder_sync

    conn = rag_conn
    _scope_fixture(conn, random.Random(3))
    reference = _copy(conn)
    statements = []
    real_connection = rag_db.get_connection

    def traced_connection():
        traced = real_connection()
        traced.set_trace_callback(statements.append)
        return traced

    monkeypatch.setattr(folder_sync.rag_db, "get_connection", traced_connection)
    try:
        if op == "scope":
            assert folder_sync.delete_retired_scope("kb:gone") is True
            gone = [
                r[0] for r in reference.execute("SELECT id FROM documents WHERE scope='kb:gone'")
            ]
        else:
            assert folder_sync._delete_retired_folder("F1") is True
            gone = [r[0] for r in reference.execute("SELECT document_id FROM linked_folder_files")]
        assert len(gone) == 12
        if op == "folder":
            reference.execute("DELETE FROM linked_folder_files WHERE folder_id='F1'")
            reference.execute("DELETE FROM linked_folders WHERE id='F1'")
        for doc_id in gone:
            store.delete_document(reference, doc_id, commit = False)
        reference.commit()

        conn.rollback()
        assert _full_dump(conn) == _full_dump(reference)
        for table in ("chunks_fts", "chunks_vec"):
            scans = [s for s in statements if s.startswith(f"DELETE FROM {table} ")]
            assert len(scans) == 1, (table, scans)
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
        assert store.search_lexical(conn, "kb:keep", "alpha", 50) == store.search_lexical(
            reference, "kb:keep", "alpha", 50
        )
        assert store.search_lexical(conn, "kb:gone", "alpha", 50) == []
    finally:
        reference.close()


def test_delete_documents_batches_past_the_variable_limit(rag_conn, monkeypatch):
    conn = rag_conn
    rng = random.Random(4)
    for d in range(7):
        _add(conn, rng, "kb:a", f"b{d}", [f"alpha {d} {i}" for i in range(d % 3)])
    _add(conn, rng, "kb:a", "stay", ["alpha stay"])
    reference = _copy(conn)
    try:
        monkeypatch.setattr(rag_db, "_DOC_ID_BATCH", 2)
        ids = [f"b{d}" for d in range(7)] + ["b0", "missing"]
        store.delete_documents(conn, ids)
        for doc_id in ids:
            store.delete_document(reference, doc_id)
        assert _full_dump(conn) == _full_dump(reference)
    finally:
        reference.close()
