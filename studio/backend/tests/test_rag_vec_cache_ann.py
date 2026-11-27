# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The HNSW tier answers only at the reader's epoch, keeps appends in its exact delta, and falls back to vec0 on
deletes, re-embeds and a missing wheel."""

import os

import pytest

np = pytest.importorskip("numpy")
if os.environ.get("CI"):
    import usearch  # noqa: F401
else:
    pytest.importorskip("usearch")

from core.rag import store, vec_ann, vec_cache

DIM = 16


@pytest.fixture(autouse = True)
def _small_ann(monkeypatch):
    monkeypatch.setenv("RAG_VEC_ANN_MIN_ROWS", "50")
    monkeypatch.setenv("RAG_VEC_CACHE", "1")
    vec_cache.clear()
    yield
    vec_cache.drain()
    vec_cache.clear()


def _chunk(i):
    from core.rag.chunking import Chunk
    return Chunk(
        text = f"t{i}",
        token_count = 1,
        page_number = None,
        source_page_index = 0,
        chunk_index = i,
        page_char_start = 0,
        page_char_end = 2,
    )


def _add(conn, scope, doc, vectors):
    store.create_document(conn, scope = scope, filename = "f", sha256 = doc, document_id = doc)
    store.add_chunks(conn, scope, doc, [_chunk(i) for i in range(len(vectors))], vectors)


def _vectors(n, seed = 0):
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


def _dense(
    conn,
    q,
    k = 10,
):
    seen = []
    conn.set_trace_callback(seen.append)
    try:
        hits = store.search_dense(conn, "kb", q, k)
    finally:
        conn.set_trace_callback(None)
    return hits, any("MATCH" in s.upper() for s in seen)


def _warm(conn, q):
    _dense(conn, q)
    assert vec_cache.drain(60)
    (entry,) = vec_cache._ann.values()
    return entry


def test_a_gigantic_scope_is_answered_by_hnsw_not_vec0(rag_conn, monkeypatch):
    rows = _vectors(400)
    _add(rag_conn, "kb", "d", rows)
    _warm(rag_conn, rows[7])
    assert not vec_cache._entries
    hits, scanned = _dense(rag_conn, rows[7], 10)
    assert not scanned and hits[0][0] == "d:7" and hits[0][1] > 0.999
    monkeypatch.setenv("RAG_VEC_CACHE", "0")
    want = store.search_dense(rag_conn, "kb", rows[7], 10)
    assert len({c for c, _ in hits} & {c for c, _ in want}) >= 9
    assert all(abs(a - b) < 2e-3 for (_, a), (_, b) in zip(hits, want))


def test_appends_join_the_delta_without_a_rebuild(rag_conn):
    _add(rag_conn, "kb", "d", _vectors(300))
    before = _warm(rag_conn, _vectors(1)[0])
    new = _vectors(5, seed = 7)
    _add(rag_conn, "kb", "e", new)
    (after,) = vec_cache._ann.values()
    assert after.base == before.base and len(after.ann.delta_keys) == 5
    hits, scanned = _dense(rag_conn, new[2], 3)
    assert not scanned and hits[0][0] == "e:2"


def test_a_delete_falls_back_to_vec0_until_the_rebuild(rag_conn):
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    _add(rag_conn, "kb", "x", _vectors(60, seed = 3))
    _warm(rag_conn, rows[1])
    store.delete_document(rag_conn, "x")
    hits, scanned = _dense(rag_conn, rows[1], 20)
    assert scanned and not any(c.startswith("x:") for c, _ in hits)
    assert vec_cache.drain(60)
    hits, scanned = _dense(rag_conn, rows[1], 20)
    assert not scanned and not any(c.startswith("x:") for c, _ in hits)


def test_a_re_embed_is_never_answered_from_the_old_vectors(rag_conn):
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    _warm(rag_conn, rows[0])
    flipped = -rows
    ids = [f"d:{i}" for i in range(300)]
    doc = rag_conn.execute("SELECT embedding_model FROM documents WHERE id='d'").fetchone()[0]
    from core.rag import config

    model = config.effective_embedding_model(fresh = True)
    assert store.replace_document_vectors(rag_conn, "d", doc, ids, flipped, model, model)
    hits, scanned = _dense(rag_conn, rows[0], 5)
    assert scanned
    assert vec_cache.drain(60)
    hits, scanned = _dense(rag_conn, -rows[0], 5)
    assert not scanned and hits[0][0] == "d:0"


def test_a_restart_maps_the_saved_generation_instead_of_rebuilding(rag_conn, monkeypatch):
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    first = _warm(rag_conn, rows[4])
    vec_cache.clear()
    monkeypatch.setattr(vec_ann, "new_index", lambda dim: pytest.fail("rebuilt instead of loading"))
    again = _warm(rag_conn, rows[4])
    assert again.base == first.base
    hits, scanned = _dense(rag_conn, rows[4], 3)
    assert not scanned and hits[0][0] == "d:4"


def test_a_restart_catches_up_appends_made_while_it_was_down(rag_conn, monkeypatch):
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    _warm(rag_conn, rows[4])
    vec_cache.clear()
    monkeypatch.setenv("RAG_VEC_CACHE", "0")
    new = _vectors(4, seed = 11)
    _add(rag_conn, "kb", "e", new)
    monkeypatch.setenv("RAG_VEC_CACHE", "1")
    monkeypatch.setattr(
        vec_ann, "new_index", lambda dim: pytest.fail("rebuilt instead of catching up")
    )
    entry = _warm(rag_conn, new[1])
    assert len(entry.ann.delta_keys) == 4
    hits, scanned = _dense(rag_conn, new[1], 3)
    assert not scanned and hits[0][0] == "e:1"


def test_superseded_generations_are_deleted(rag_conn):
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    _add(rag_conn, "kb", "x", _vectors(60, seed = 3))
    # Only the path: a held entry keeps its index mapped, and Windows cannot delete a mapped file.
    first = _warm(rag_conn, rows[0]).base
    store.delete_document(rag_conn, "x")
    _dense(rag_conn, rows[0])
    assert vec_cache.drain(60)
    (second,) = vec_cache._ann.values()
    assert second.base != first
    assert not os.path.exists(first + ".usearch")


@pytest.mark.parametrize("mode", ["off", "no_wheel"])
def test_without_the_tier_the_exact_and_vec0_paths_answer(rag_conn, monkeypatch, mode):
    if mode == "off":
        monkeypatch.setenv("RAG_VEC_ANN", "0")
    else:
        monkeypatch.setattr(vec_ann, "usearch", lambda: None)
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    _dense(rag_conn, rows[3])
    assert vec_cache.drain(60)
    assert not vec_cache._ann and vec_cache._entries
    hits, scanned = _dense(rag_conn, rows[3], 3)
    assert not scanned and hits[0][0] == "d:3"


def test_rows_another_writer_added_are_never_missed(rag_conn, monkeypatch):
    # With the cache off for the write, note_append sees nothing: as with a second process or an older build.
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    _warm(rag_conn, rows[0])
    monkeypatch.setenv("RAG_VEC_CACHE", "0")
    new = _vectors(3, seed = 21)
    _add(rag_conn, "kb", "e", new)
    monkeypatch.setenv("RAG_VEC_CACHE", "1")
    hits, scanned = _dense(rag_conn, new[0], 3)
    assert scanned and hits[0][0] == "e:0"


def test_the_disk_cap_evicts_the_least_recently_used_scope(rag_conn, monkeypatch):
    monkeypatch.setenv("RAG_VEC_ANN_DISK_MB", "0.001")
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    first = _warm(rag_conn, rows[0]).base
    store.create_document(rag_conn, scope = "kb2", filename = "f", sha256 = "e", document_id = "e")
    store.add_chunks(rag_conn, "kb2", "e", [_chunk(i) for i in range(300)], _vectors(300, seed = 4))
    store.search_dense(rag_conn, "kb2", rows[0], 5)
    assert vec_cache.drain(60)
    (only,) = vec_cache._ann.values()
    assert only.base != first and os.path.exists(only.base + ".usearch")
    assert not os.path.exists(first + ".usearch")
    hits, scanned = _dense(rag_conn, rows[0], 3)
    assert scanned and hits[0][0] == "d:0"


def test_an_exact_entry_that_outgrows_the_budget_moves_to_hnsw(rag_conn, monkeypatch):
    _add(rag_conn, "kb", "d", _vectors(30))
    _dense(rag_conn, _vectors(1)[0])
    assert vec_cache.drain(60)
    (small,) = vec_cache._entries.values()
    monkeypatch.setenv("RAG_VEC_CACHE_MB", str(3 * small.nbytes / 2**20))
    new = _vectors(300, seed = 5)
    _add(rag_conn, "kb", "e", new)
    assert vec_cache.drain(60)
    assert not vec_cache._entries and vec_cache._ann
    hits, scanned = _dense(rag_conn, new[9], 3)
    assert not scanned and hits[0][0] == "e:9"


def test_a_large_catch_up_rebuilds_without_loading_the_delta(rag_conn, monkeypatch):
    rows = _vectors(300)
    _add(rag_conn, "kb", "d", rows)
    first = _warm(rag_conn, rows[0]).base
    vec_cache.clear()
    monkeypatch.setenv("RAG_VEC_CACHE", "0")
    _add(rag_conn, "kb", "e", _vectors(4, seed = 13))
    monkeypatch.setenv("RAG_VEC_CACHE", "1")
    monkeypatch.setattr(vec_cache, "_delta_limit", lambda n: 2)
    monkeypatch.setattr(
        vec_cache, "_delta_since", lambda *a: pytest.fail("loaded the delta before rebuilding")
    )
    entry = _warm(rag_conn, rows[0])
    assert entry.base != first and entry.ann.n == 304 and not len(entry.ann.delta_keys)
