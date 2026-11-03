# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dense cache answers exactly what vec0 answers, and never from a snapshot it was not built at."""

import math

import pytest

np = pytest.importorskip("numpy")

from core.rag import config, store, vec_cache
from core.rag.chunking import Chunk
from storage import rag_db

DIM = 16


def _chunk(i):
    return Chunk(
        text = f"t{i}",
        token_count = 1,
        page_number = None,
        source_page_index = 0,
        chunk_index = i,
        page_char_start = 0,
        page_char_end = 2,
    )


def _add(
    conn,
    scope,
    doc,
    vectors,
    dim = DIM,
):
    store.create_document(conn, scope = scope, filename = "f", sha256 = doc, document_id = doc)
    store.add_chunks(conn, scope, doc, [_chunk(i) for i in range(len(vectors))], vectors)


def _vectors(
    n,
    seed = 0,
    dim = DIM,
):
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((n, dim)).astype(np.float32)
    out[::7] *= 9.0  # unnormalised rows
    return out


def _vec0_matches(conn):
    seen = []
    conn.set_trace_callback(seen.append)
    return seen


def _search(conn, scope, q, k, monkeypatch, cache):
    monkeypatch.setenv("RAG_VEC_CACHE", "1" if cache else "0")
    return store.search_dense(conn, scope, q, k)


def _warm(conn, scope, q, monkeypatch):
    _search(conn, scope, q, 5, monkeypatch, True)
    assert vec_cache.drain()


def _same(a, b):
    assert len(a) == len(b)
    for (ca, sa), (cb, sb) in zip(a, b):
        assert abs(sa - sb) < 1e-5, (ca, sa, cb, sb)
    if not a:
        return
    # Rows tied with the last kept score may be swapped for one another; everything above it must agree.
    edge = a[-1][1] + 1e-5
    assert {c for c, s in a if s > edge} == {c for c, s in b if s > edge}


def _hit(conn, scope, q, k, monkeypatch):
    seen = _vec0_matches(conn)
    try:
        got = _search(conn, scope, q, k, monkeypatch, True)
    finally:
        conn.set_trace_callback(None)
    assert not [s for s in seen if "MATCH" in s.upper()], "expected a cache hit"
    return got


@pytest.mark.parametrize("block, fetch", [(None, None), (7, 11)])
def test_cache_matches_vec0_on_unnormalised_nan_zero_and_tied_rows(
    rag_conn, monkeypatch, block, fetch
):
    # Small sizes push the build across fetch blocks and the scan across matvec blocks.
    if block:
        monkeypatch.setattr(vec_cache, "_BLOCK", block)
        monkeypatch.setattr(vec_cache, "_FETCH", fetch)
    rows = _vectors(300)
    rows[5] = np.nan
    rows[9] = 0.0
    rows[11] = rows[12] = rows[13]  # ties
    _add(rag_conn, "kb_a", "d", rows)
    q = rows[13] * 3 + 0.01
    _warm(rag_conn, "kb_a", q, monkeypatch)
    for k in (1, 5, 30, 200, 400):
        want = _search(rag_conn, "kb_a", q, k, monkeypatch, False)
        got = _hit(rag_conn, "kb_a", q, k, monkeypatch)
        _same(got, want)
    assert "d:5" not in {c for c, _ in got} and "d:9" not in {c for c, _ in got}


def test_zero_query_returns_nothing_on_both_paths(rag_conn, monkeypatch):
    _add(rag_conn, "kb_a", "d", _vectors(20))
    _warm(rag_conn, "kb_a", _vectors(1)[0], monkeypatch)
    zero = [0.0] * DIM
    assert _hit(rag_conn, "kb_a", zero, 5, monkeypatch) == []
    assert _search(rag_conn, "kb_a", zero, 5, monkeypatch, False) == []


def test_every_writer_moves_the_epoch(rag_conn, monkeypatch):
    def epoch(scope = "kb_a"):
        return vec_cache.read_epoch(rag_conn, scope)

    e0 = epoch()
    _add(rag_conn, "kb_a", "d1", _vectors(3))
    e1 = epoch()
    assert e1[:2] == e0[:2] and e1[2] > e0[2]
    monkeypatch.setattr(config, "effective_embedding_model", lambda fresh = False: "m")
    assert store.replace_document_vectors(
        rag_conn, "d1", None, ["d1:0", "d1:1", "d1:2"], _vectors(3, seed = 1), "m2", "m"
    )
    e2 = epoch()
    assert e2[0] > e1[0]
    source = dict(rag_conn.execute("SELECT * FROM documents WHERE id='d1'").fetchone())
    store.copy_documents(rag_conn, [(source, None)], "kb_b", thread_id = "t")
    rag_conn.commit()
    assert epoch("kb_b")[2] > 0
    store.delete_document(rag_conn, "d1")
    assert epoch()[0] > e2[0]
    # Width change: no chunks row moves, the global epoch does.
    g = epoch()[1]
    _add(rag_conn, "kb_c", "w", _vectors(2, dim = DIM * 2), dim = DIM * 2)
    assert epoch()[1] == g + 1


def test_append_extends_the_entry_without_a_rebuild(rag_conn, monkeypatch):
    _add(rag_conn, "kb_a", "d1", _vectors(50))
    q = _vectors(1, seed = 3)[0]
    _warm(rag_conn, "kb_a", q, monkeypatch)
    _add(rag_conn, "kb_a", "d2", [q * 2])
    got = _hit(rag_conn, "kb_a", q, 3, monkeypatch)
    assert got[0][0] == "d2:0"
    _same(got, _search(rag_conn, "kb_a", q, 3, monkeypatch, False))


@pytest.mark.parametrize(
    "writer", ["reembed", "copy_into", "delete_doc", "delete_scope", "raw_sql", "raw_sql_insert"]
)
def test_no_stale_answer_after_any_write(rag_conn, monkeypatch, writer):
    _add(rag_conn, "kb_a", "d1", _vectors(40))
    _add(rag_conn, "kb_a", "d2", _vectors(40, seed = 2))
    q = _vectors(1, seed = 4)[0]
    _warm(rag_conn, "kb_a", q, monkeypatch)
    _hit(rag_conn, "kb_a", q, 10, monkeypatch)
    if writer == "reembed":
        monkeypatch.setattr(config, "effective_embedding_model", lambda fresh = False: "m")
        flipped = _vectors(40, seed = 9)
        flipped[17] = q
        ids = [f"d1:{i}" for i in range(40)]
        assert store.replace_document_vectors(rag_conn, "d1", None, ids, flipped, "m", "m")
    elif writer == "copy_into":
        _add(rag_conn, "kb_src", "s", [q])
        source = dict(rag_conn.execute("SELECT * FROM documents WHERE id='s'").fetchone())
        store.copy_documents(rag_conn, [(source, None)], "kb_a", thread_id = "t")
        rag_conn.commit()
    elif writer == "delete_doc":
        store.delete_document(rag_conn, "d2")
    elif writer == "delete_scope":
        for doc in ("d1", "d2"):
            store.delete_document(rag_conn, doc)
    elif writer == "raw_sql_insert":
        # An older build appending rows never calls note_append; the scope's newest rowid still moves.
        other = rag_db.get_connection()
        try:
            other.execute(
                "INSERT INTO chunks(id, document_id, scope, chunk_index, text) "
                "VALUES('d2:99', 'd2', 'kb_a', 99, 'x')"
            )
            other.execute(
                "INSERT INTO chunks_vec(scope, chunk_id, embedding) VALUES('kb_a', 'd2:99', ?)",
                (np.asarray(q, dtype = np.float32).tobytes(),),
            )
            other.commit()
        finally:
            other.close()
    else:
        # An older build deleting rows knows nothing of the cache; the triggers still fire.
        other = rag_db.get_connection()
        try:
            other.execute(
                "DELETE FROM chunks_vec WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id='d1')"
            )
            other.execute("DELETE FROM chunks WHERE document_id='d1'")
            other.commit()
        finally:
            other.close()
    for _ in range(2):
        want = _search(rag_conn, "kb_a", q, 10, monkeypatch, False)
        got = _search(rag_conn, "kb_a", q, 10, monkeypatch, True)
        _same(got, want)
        vec_cache.drain()
    _same(_hit(rag_conn, "kb_a", q, 10, monkeypatch), want)


def test_a_reader_snapshot_never_sees_a_later_commit(rag_conn, monkeypatch):
    _add(rag_conn, "kb_a", "d1", _vectors(30))
    q = _vectors(1, seed = 5)[0]
    _warm(rag_conn, "kb_a", q, monkeypatch)
    rag_conn.execute("BEGIN")
    try:
        before = _search(rag_conn, "kb_a", q, 5, monkeypatch, True)
        writer = rag_db.get_connection()
        try:
            _add(writer, "kb_a", "late", [q])
            # Same chunk ids, new vectors: only the epoch tells the old snapshot these are not its rows.
            monkeypatch.setattr(config, "effective_embedding_model", lambda fresh = False: "m")
            moved = _vectors(30, seed = 8)
            moved[3] = q
            ids = [f"d1:{i}" for i in range(30)]
            assert store.replace_document_vectors(writer, "d1", None, ids, moved, "m", "m")
        finally:
            writer.close()
        vec_cache.drain()
        _search(rag_conn, "kb_a", q, 5, monkeypatch, True)
        vec_cache.drain()
        during = _search(rag_conn, "kb_a", q, 5, monkeypatch, True)
    finally:
        rag_conn.commit()
    _same(during, before)
    assert "late:0" not in {c for c, _ in during} and during[0][0] != "d1:3"
    after = _search(rag_conn, "kb_a", q, 5, monkeypatch, True)
    assert {after[0][0], after[1][0]} == {"late:0", "d1:3"}


@pytest.mark.parametrize("mode", ["off", "no_numpy", "over_budget", "raises"])
def test_fallbacks_answer_like_vec0(rag_conn, monkeypatch, mode):
    _add(rag_conn, "kb_a", "d1", _vectors(60))
    q = _vectors(1, seed = 6)[0]
    want = _search(rag_conn, "kb_a", q, 10, monkeypatch, False)
    if mode == "off":
        monkeypatch.setenv("RAG_VEC_CACHE", "0")
    elif mode == "no_numpy":
        monkeypatch.setattr(vec_cache, "_np", lambda: None)
    elif mode == "over_budget":
        monkeypatch.setenv("RAG_VEC_CACHE_MB", "0.001")
    else:
        _warm(rag_conn, "kb_a", q, monkeypatch)

        def boom(*_a, **_k):
            raise RuntimeError("injected")

        monkeypatch.setattr(vec_cache, "_search_entry", boom)
    if mode != "off":
        monkeypatch.setenv("RAG_VEC_CACHE", "1")
    for _ in range(2):
        assert store.search_dense(rag_conn, "kb_a", q, 10) == want
        vec_cache.drain()
    if mode == "over_budget":
        assert not vec_cache._entries


def test_a_database_without_the_epoch_table_still_serves(rag_conn, monkeypatch):
    for name in (
        "rag_vec_epoch_chunk_delete",
        "rag_vec_epoch_document_model",
        "rag_vec_epoch_document_delete",
    ):
        rag_conn.execute(f"DROP TRIGGER {name}")
    rag_conn.execute("DROP TABLE rag_vec_epoch")
    rag_conn.commit()
    _add(rag_conn, "kb_a", "d1", _vectors(20))
    q = _vectors(1, seed = 7)[0]
    for _ in range(2):
        got = _search(rag_conn, "kb_a", q, 5, monkeypatch, True)
        vec_cache.drain()
    assert got == _search(rag_conn, "kb_a", q, 5, monkeypatch, False)
    assert not vec_cache._entries
    assert math.isfinite(got[0][1])


def test_the_builder_opens_and_closes_through_the_connection_gate(monkeypatch, tmp_path):
    from core.rag import vec_cache
    from storage import sqlite_gate

    opened = []
    real = sqlite_gate.connect
    monkeypatch.setattr(sqlite_gate, "connect", lambda *a, **kw: opened.append(a) or real(*a, **kw))
    conn = vec_cache._connect(str(tmp_path / "x.db"))
    assert isinstance(conn, sqlite_gate.GatedConnection) and len(opened) == 1
    conn.close()


def test_appends_write_in_place_and_an_older_entry_keeps_its_rows(rag_conn, monkeypatch):
    rows = _vectors(40)
    _add(rag_conn, "kb_a", "d", rows)
    q = rows[3]
    _warm(rag_conn, "kb_a", q, monkeypatch)
    (old,) = vec_cache._entries.values()
    before = _hit(rag_conn, "kb_a", q, 10, monkeypatch)
    stores = {id(old.store)}
    for i in range(6):
        _add(rag_conn, "kb_a", f"e{i}", _vectors(5, seed = 100 + i))
        (cur,) = vec_cache._entries.values()
        stores.add(id(cur.store))
    assert cur.n == 70
    # Geometric growth: six appends reallocate at most a couple of times, not once each.
    assert len(stores) <= 3
    # _search_entry gives distances; search_dense gives 1 - distance.
    assert vec_cache._search_entry(np, old, q, 10)[0] == [
        (c, pytest.approx(1 - s, abs = 1e-6)) for c, s in before
    ]
    _same(
        _hit(rag_conn, "kb_a", q, 70, monkeypatch),
        _search(rag_conn, "kb_a", q, 70, monkeypatch, False),
    )


def test_block_scan_scratch_stays_bounded_by_block_plus_fetch(monkeypatch):
    import tracemalloc

    monkeypatch.setattr(vec_cache, "_BLOCK", 4096)
    n, f = 200_000, 4096
    store = vec_cache._Store(np, n, 1)
    store.mat[:] = np.random.default_rng(0).standard_normal((n, 1), dtype = np.float32)
    store.rowids[:] = np.arange(n)
    store.ids[:] = "c"
    entry = vec_cache._Entry((0, 0, 0), n, store, 0)
    tracemalloc.start()
    vec_cache._search_entry(np, entry, [1.0], f)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # Result list aside, a scan that kept every block's candidates would hold ~n * 12 bytes (2.4 MB).
    assert peak < 1.5 * 2**20, peak
