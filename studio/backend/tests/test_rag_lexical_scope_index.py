# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The scope index on chunks_fts_content must leave lexical results byte-identical to the unindexed filter."""

import random

from core.rag import store
from core.rag.chunking import Chunk
from storage import rag_db

WORDS = (
    "the of and to in is was what how who when where history city army album storm river "
    "bridge castle railway station election season film song ship island church school"
).split()


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


def _add(
    conn,
    rng,
    scope,
    doc_id,
    n,
    ordinal = None,
):
    texts = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 25))) for _ in range(n)]
    store.create_document(
        conn,
        scope = scope,
        filename = f"{doc_id}.txt",
        sha256 = doc_id,
        document_id = doc_id,
        status = "completed",
        archive_ordinal = ordinal,
        commit = False,
    )
    for i, text in enumerate(texts):
        store._insert_chunk(conn, scope, doc_id, _chunk(text, i), None)


def _corpus(conn, rng):
    scopes = ["kb_big", "kb_small"] + [f"convarchive_t{i}" for i in range(6)]
    docs = []
    for d in range(60):
        scope = rng.choice(scopes)
        doc_id = f"d{d}"
        _add(conn, rng, scope, doc_id, rng.randint(1, 8), d if scope.startswith("conv") else None)
        docs.append(doc_id)
    conn.commit()
    # Deletes then inserts leave gaps and reuse the top rowids, which the rowid bounds must cover.
    store.delete_documents(conn, rng.sample(docs, 15))
    for d in range(60, 75):
        _add(conn, rng, rng.choice(scopes), f"d{d}", rng.randint(1, 5), d)
    conn.commit()
    return scopes


def _cases(rng, scopes, n):
    cases = []
    for _ in range(n):
        query = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 6)))
        scope = rng.choice(scopes + ["kb_missing", scopes[:3] + ["kb_missing"], scopes[2:5]])
        mode = rng.choice(["plain", "oldest_first", "newest_first"])
        cases.append((query, scope, rng.choice([1, 5, 30, 256]), mode))
    return cases


def _run(conn, cases):
    out = []
    for query, scope, k, mode in cases:
        kwargs = {} if mode == "plain" else {mode: True}
        out.append(store.search_lexical(conn, scope, query, k, **kwargs))
    return out


def test_scope_index_is_created_on_the_content_table(rag_conn):
    assert rag_db.fts_scope_index_exists(rag_conn)


def test_indexed_scope_filter_matches_unindexed_filter_exactly(rag_conn):
    rng = random.Random(0)
    scopes = _corpus(rag_conn, rng)
    cases = _cases(rng, scopes, 400)
    statements = []
    rag_conn.set_trace_callback(statements.append)
    indexed = _run(rag_conn, cases)
    rag_conn.set_trace_callback(None)
    assert any(rag_db.FTS_SCOPE_INDEX in sql for sql in statements)
    rag_conn.execute(f"DROP INDEX {rag_db.FTS_SCOPE_INDEX}")
    rag_conn.commit()
    assert not rag_db.fts_scope_index_exists(rag_conn)
    unindexed = _run(rag_conn, cases)
    assert sum(map(len, indexed)) > 1000
    # Same ids, same order (ties included) and bit-identical bm25 scores.
    assert indexed == unindexed


def test_scope_without_rows_returns_nothing(rag_conn):
    rng = random.Random(1)
    _add(rag_conn, rng, "kb_a", "a", 3)
    _add(rag_conn, rng, "kb_b", "b", 3)
    rag_conn.commit()
    store.delete_documents(rag_conn, ["b"])
    assert store.search_lexical(rag_conn, "kb_b", "the", 10) == []
    assert store.search_lexical(rag_conn, "kb_none", "the", 10) == []
    assert store.search_lexical(rag_conn, ["kb_b", "kb_a"], "the of and to in is was", 10)


def test_scope_probe_never_runs_below_the_joins(rag_conn):
    rng = random.Random(2)
    _add(rag_conn, rng, "convarchive_a", "a", 5, 0)
    _add(rag_conn, rng, "kb_b", "b", 5)
    rag_conn.commit()
    for scope in ("convarchive_a", ["convarchive_a", "kb_b"]):
        statements = []
        rag_conn.set_trace_callback(statements.append)
        store.search_lexical(rag_conn, scope, "the of", 5, oldest_first = True)
        rag_conn.set_trace_callback(None)
        sql = next(s for s in statements if "bm25" in s)
        plan = [r[3] for r in rag_conn.execute("EXPLAIN QUERY PLAN " + sql)]
        join = next(i for i, d in enumerate(plan) if d.startswith("SEARCH c "))
        probes = [i for i, d in enumerate(plan) if rag_db.FTS_SCOPE_INDEX in d]
        # SQLite places a multi-value IN probe below the joins, so every out-of-scope match pays them.
        assert all(i < join for i in probes), plan
        assert probes or not isinstance(scope, str), plan
