# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Store tests: incremental writes, dedupe, delete, scope, dense + lexical."""

import math
import re
import sqlite3

import pytest

from core.rag import store
from core.rag.chunking import Chunk
from growth import assert_linear  # tests/_shared, on sys.path via tests/conftest.py

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def embed(text):
    v = [float(text.lower().count(w)) for w in VOCAB]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _chunk(
    text,
    index = 0,
    page = None,
):
    return Chunk(
        text = text,
        token_count = len(text.split()),
        page_number = page,
        source_page_index = 0,
        chunk_index = index,
        page_char_start = 0,
        page_char_end = len(text),
    )


def _add_doc(conn, scope, doc_id, filename, sha, texts):
    chunks = [_chunk(t, i) for i, t in enumerate(texts)]
    vectors = [embed(t) for t in texts]
    store.create_document(conn, scope = scope, filename = filename, sha256 = sha, document_id = doc_id)
    store.add_chunks(conn, scope, doc_id, chunks, vectors)


def test_lexical_returns_only_matching_docs(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "d1.txt", "h1", ["alpha bravo charlie"])
    _add_doc(rag_conn, "kb_a", "d2", "d2.txt", "h2", ["golf hotel india"])
    hits = store.search_lexical(rag_conn, "kb_a", "alpha", 10)
    assert [cid for cid, _ in hits] == ["d1:0"]


def test_scope_isolation(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo"])
    _add_doc(rag_conn, "kb_b", "d2", "f", "h2", ["alpha bravo"])
    assert [cid for cid, _ in store.search_lexical(rag_conn, "kb_b", "alpha", 10)] == ["d2:0"]


def test_match_query_sanitizes_special_chars():
    assert store._match_query('AND OR "quote" (paren) -dash') != ""


def test_lexical_does_not_crash_on_punctuation(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo"])
    store.search_lexical(rag_conn, "kb_a", 'NEAR("x" AND', 5)


def test_dense_ranks_by_cosine(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha alpha"])
    _add_doc(rag_conn, "kb_a", "d2", "f", "h2", ["hotel golf"])
    ranked = store.search_dense(rag_conn, "kb_a", embed("alpha"), 10)
    assert ranked[0][0] == "d1:0" and ranked[0][1] > 0.99


def test_dense_skips_nan_vectors_instead_of_raising(rag_conn):
    # A float16 embedder wrote NaN vectors, which vec0 answers with a NULL distance.
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha alpha"])
    store.create_document(rag_conn, scope = "kb_a", filename = "g", sha256 = "h2", document_id = "d2")
    store.add_chunks(rag_conn, "kb_a", "d2", [_chunk("alpha")], [[math.nan] * len(VOCAB)])
    ranked = store.search_dense(rag_conn, "kb_a", embed("alpha"), 10)
    assert [cid for cid, _ in ranked] == ["d1:0"]


def test_dense_knn_binds_k_rather_than_limit(rag_conn):
    """vec0's KNN bound must arrive as ``k = ?``, not a bare ``LIMIT ?``, which SQLite forwards
    to a virtual table's planner only from 3.41 on. CI's SQLite accepts both and returns the
    same rows, so nothing else here catches a revert and only the executed statement can say."""
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha alpha"])
    seen = []
    rag_conn.set_trace_callback(seen.append)
    try:
        store.search_dense(rag_conn, "kb_a", embed("alpha"), 5)
    finally:
        rag_conn.set_trace_callback(None)

    knn = [sql for sql in seen if "chunks_vec" in sql and "MATCH" in sql.upper()]
    assert knn, f"search_dense issued no vec0 MATCH query; statements were {seen}"
    for sql in knn:
        flat = " ".join(sql.split())
        assert re.search(r"\bk\s*=", flat), f"vec0 KNN query has no k constraint: {flat}"
        assert not re.search(
            r"\bLIMIT\b", flat, re.IGNORECASE
        ), f"vec0 KNN query still leans on LIMIT, which pre-3.41 SQLite never forwards: {flat}"


def test_dense_empty_before_any_ingest(rag_conn):
    assert store.search_dense(rag_conn, "kb_a", embed("alpha"), 10) == []


def test_dedupe_by_hash(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "SHA", ["alpha"])
    assert store.document_by_hash(rag_conn, "kb_a", "SHA") == "d1"
    assert store.document_by_hash(rag_conn, "kb_a", "OTHER") is None


def test_delete_document_purges_all_tables(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo"])
    store.delete_document(rag_conn, "d1")
    assert store.search_lexical(rag_conn, "kb_a", "alpha", 10) == []
    assert store.search_dense(rag_conn, "kb_a", embed("alpha"), 10) == []
    assert store.chunks_by_id(rag_conn, ["d1:0"]) == {}
    assert store.get_document(rag_conn, "d1") is None


def test_incremental_add_is_flat(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo charlie"])
    before = rag_conn.execute(
        "SELECT rowid, chunk_id FROM chunks_fts WHERE scope='kb_a'"
    ).fetchall()
    _add_doc(rag_conn, "kb_a", "d2", "f", "h2", ["delta echo foxtrot"])
    after = rag_conn.execute(
        "SELECT rowid, chunk_id FROM chunks_fts WHERE scope='kb_a' AND chunk_id LIKE 'd1:%'"
    ).fetchall()
    before_d1 = [(r["rowid"], r["chunk_id"]) for r in before if r["chunk_id"].startswith("d1:")]
    after_d1 = [(r["rowid"], r["chunk_id"]) for r in after]
    assert before_d1 == after_d1


def test_chunks_by_id_joins_filename(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "paper.pdf", "h1", ["body text here"])
    rows = store.chunks_by_id(rag_conn, ["d1:0"])
    assert rows["d1:0"]["filename"] == "paper.pdf"
    assert rows["d1:0"]["text"] == "body text here"


def test_kb_crud_and_delete_cascades(rag_conn):
    kb_id = store.create_kb(rag_conn, name = "My KB", description = "d", kb_id = "K1")
    assert store.get_kb(rag_conn, kb_id)["name"] == "My KB"
    assert [k["id"] for k in store.list_kbs(rag_conn)] == ["K1"]

    scope = store.kb_scope("K1")
    _add_doc(rag_conn, scope, "doc1", "f", "h1", ["alpha bravo"])
    store.delete_kb(rag_conn, "K1")
    assert store.get_kb(rag_conn, "K1") is None
    assert store.list_documents(rag_conn, scope) == []
    assert store.search_lexical(rag_conn, scope, "alpha", 10) == []


def test_kb_delete_rolls_back_when_document_cleanup_fails(rag_conn, monkeypatch):
    store.create_kb(rag_conn, name = "My KB", kb_id = "K1")
    scope = store.kb_scope("K1")
    _add_doc(rag_conn, scope, "doc1", "one.txt", "h1", ["alpha bravo"])
    _add_doc(rag_conn, scope, "doc2", "two.txt", "h2", ["charlie delta"])
    original_delete = store.delete_document
    calls = []

    def fail_after_delete(
        conn,
        document_id,
        *,
        commit = True,
    ):
        calls.append((document_id, commit))
        original_delete(conn, document_id, commit = commit)
        raise sqlite3.OperationalError("database is busy")

    monkeypatch.setattr(store, "delete_document", fail_after_delete)
    with pytest.raises(sqlite3.OperationalError, match = "database is busy"):
        store.delete_kb(rag_conn, "K1")

    assert calls == [("doc1", False)]
    assert store.get_kb(rag_conn, "K1") is not None
    assert sorted(document["id"] for document in store.list_documents(rag_conn, scope)) == [
        "doc1",
        "doc2",
    ]


def _link_folder(
    conn,
    folder_id,
    scope,
    path = "/tmp/linked",
):
    conn.execute(
        "INSERT INTO linked_folders(id, scope_type, scope_id, scope, path, name, "
        "auto_sync, status, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,1,'idle','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')",
        (folder_id, "knowledge_base", scope.removeprefix("kb_"), scope, path, "linked"),
    )
    conn.commit()


def test_lexical_fast_path_only_while_no_folder_rows_exist(rag_conn):
    """The plain FTS query is used exactly when the filters could not exclude anything."""
    _add_doc(rag_conn, "kb_a", "d1", "d1.txt", "h1", ["alpha bravo charlie"])
    assert store.linked_folder_rows_exist(rag_conn) is False

    _link_folder(rag_conn, "f1", "kb_a")
    assert store.linked_folder_rows_exist(rag_conn) is True

    rag_conn.execute("DELETE FROM linked_folders")
    rag_conn.execute(
        "INSERT INTO linked_folder_retired_scopes(scope, retired_at) "
        "VALUES('kb_a', '2026-01-01T00:00:00+00:00')"
    )
    rag_conn.commit()
    assert store.linked_folder_rows_exist(rag_conn) is True


def test_lexical_hides_retired_scope_and_unmapped_linked_document(rag_conn):
    """The filters still apply once a folder row exists, fast path or not."""
    _add_doc(rag_conn, "kb_a", "d1", "d1.txt", "h1", ["alpha bravo charlie"])
    _add_doc(rag_conn, "kb_a", "d2", "d2.txt", "h2", ["alpha delta echo"])
    _link_folder(rag_conn, "f1", "kb_a")
    rag_conn.execute("UPDATE documents SET linked_folder_id='f1' WHERE id='d2'")
    rag_conn.commit()
    assert [cid for cid, _ in store.search_lexical(rag_conn, "kb_a", "alpha", 10)] == ["d1:0"]

    rag_conn.execute(
        "INSERT INTO linked_folder_files(folder_id, relative_path, size_bytes, mtime_ns, "
        "document_id, synced_at) VALUES('f1', 'd2.txt', 1, 1, 'd2', "
        "'2026-01-01T00:00:00+00:00')"
    )
    rag_conn.commit()
    assert sorted(cid for cid, _ in store.search_lexical(rag_conn, "kb_a", "alpha", 10)) == [
        "d1:0",
        "d2:0",
    ]

    rag_conn.execute(
        "INSERT INTO linked_folder_retired_scopes(scope, retired_at) "
        "VALUES('kb_a', '2026-01-01T00:00:00+00:00')"
    )
    rag_conn.commit()
    assert store.search_lexical(rag_conn, "kb_a", "alpha", 10) == []


def test_lexical_results_match_across_both_query_forms(rag_conn):
    """The fast path is a shortcut in work, not in behaviour."""
    for i in range(12):
        _add_doc(
            rag_conn, "kb_a", f"d{i}", f"d{i}.txt", f"h{i}", [f"alpha bravo {'charlie ' * (i % 4)}"]
        )
    fast = store.search_lexical(rag_conn, "kb_a", "alpha bravo", 5)
    _link_folder(rag_conn, "f1", "kb_b")
    filtered = store.search_lexical(rag_conn, "kb_a", "alpha bravo", 5)
    assert fast == filtered


def test_lexical_gate_and_read_share_one_snapshot(rag_conn, monkeypatch):
    """Gate and FTS read must share one snapshot, so a scope retired in between cannot leak rows."""
    from storage import rag_db

    _add_doc(rag_conn, "kb_a", "d1", "d1.txt", "h1", ["alpha bravo charlie"])
    observed = {}
    real = store.linked_folder_rows_exist

    def retire_midway(conn):
        observed["gate"] = real(conn)
        writer = rag_db.get_connection()
        try:
            writer.execute(
                "INSERT INTO linked_folder_retired_scopes(scope, retired_at) "
                "VALUES('kb_a', '2026-01-01T00:00:00+00:00')"
            )
            writer.commit()
        finally:
            writer.close()
        observed["after_commit"] = real(conn)
        return observed["gate"]

    monkeypatch.setattr(store, "linked_folder_rows_exist", retire_midway)
    hits = store.search_lexical(rag_conn, "kb_a", "alpha", 10)

    assert observed["gate"] is False
    assert observed["after_commit"] is False
    assert [cid for cid, _ in hits] == ["d1:0"]
    monkeypatch.setattr(store, "linked_folder_rows_exist", real)
    assert store.search_lexical(rag_conn, "kb_a", "alpha", 10) == []


def test_lexical_reuses_a_transaction_the_caller_already_opened(rag_conn):
    """A caller holding a transaction keeps its own snapshot, and keeps it open."""
    _add_doc(rag_conn, "kb_a", "d1", "d1.txt", "h1", ["alpha bravo charlie"])
    rag_conn.execute("BEGIN")
    assert [cid for cid, _ in store.search_lexical(rag_conn, "kb_a", "alpha", 10)] == ["d1:0"]
    assert rag_conn.in_transaction
    rag_conn.commit()


def test_gate_ignores_a_purged_tombstone(rag_conn):
    """The gate must ignore purged tombstones, or every delete would disable the fast path for good."""
    _add_doc(rag_conn, "kb_a", "d1", "d1.txt", "h1", ["alpha bravo charlie"])
    rag_conn.execute(
        "INSERT INTO linked_folder_retired_scopes(scope, retired_at) "
        "VALUES('kb_gone', '2026-01-01T00:00:00+00:00')"
    )
    rag_conn.commit()
    assert store.linked_folder_rows_exist(rag_conn) is True

    rag_conn.execute(
        "UPDATE linked_folder_retired_scopes SET purged_at='2026-01-01T00:00:01+00:00'"
    )
    rag_conn.commit()
    assert store.linked_folder_rows_exist(rag_conn) is False
    assert [cid for cid, _ in store.search_lexical(rag_conn, "kb_a", "alpha", 10)] == ["d1:0"]


def test_gate_counts_a_folder_document_that_outlived_its_folder(rag_conn):
    """The gate must count an orphan document that outlived its folder, so unlinked content stays hidden."""
    _link_folder(rag_conn, "f1", "kb_a")
    store.create_document(
        rag_conn,
        scope = "kb_a",
        filename = "secret.md",
        sha256 = "h1",
        document_id = "orphan",
        linked_folder_id = "f1",
    )
    store.add_chunks(
        rag_conn, "kb_a", "orphan", [_chunk("alpha bravo secret")], [embed("alpha bravo")]
    )
    assert store.search_lexical(rag_conn, "kb_a", "alpha", 10) == []

    rag_conn.execute("DELETE FROM linked_folders WHERE id='f1'")
    rag_conn.commit()
    assert store.linked_folder_rows_exist(rag_conn) is True
    assert store.search_lexical(rag_conn, "kb_a", "alpha", 10) == []


def _pasted_prose(words: int) -> str:
    """Pasted prose must be purely alphabetic; mixed letters and digits short-circuit the identifier
    test."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    return " ".join(
        letters[index % 26]
        + letters[(index // 26) % 26]
        + letters[(index // 676) % 26]
        + letters[(index // 17576) % 26]
        + "qz"
        for index in range(words)
    )


def test_a_pasted_log_does_not_make_the_archive_query_quadratic(monkeypatch):
    """Query shaping must scan the question once, not per token, or the cost grows quadratically."""
    scans = {"n": 0}
    real = store._TOKEN

    class CountingToken:
        def findall(self, text):
            scans["n"] += 1
            return real.findall(text)

    monkeypatch.setattr(store, "_TOKEN", CountingToken())

    question = f"what is the current value of ZQXVARA123 {_pasted_prose(2000)}"
    expressions = store.conversation_match_queries(question)

    assert expressions and expressions[0].startswith('"zqxvara123"')
    # Anything growing with token count means the quadratic is back.
    assert scans["n"] <= 2, f"tokenized the question {scans['n']} times"


def test_query_shaping_stays_cheap_on_a_pasted_log():
    """Wall-clock companion to the scan count: pasted-log query shaping must stay under 1.0s."""
    expressions = assert_linear(
        store.conversation_match_queries,
        lambda n: f"what is the current value of ZQXVARA123\n{_pasted_prose(n)}",
        "paste shaping",
        1_500,
    )
    assert expressions and expressions[0] == '"zqxvara123"'


def test_a_quoted_function_word_survives_the_stopword_filter():
    """Quoted words are not stopwords: a quoted word must survive the filter that drops it unquoted."""
    quoted = store.conversation_match_queries('What did I say about "this"?')
    plain = store.conversation_match_queries("What did I say about this?")

    assert quoted == ['"say" OR "this"']
    assert plain == ['"say"']
    assert len(quoted) == 1


def test_the_candidate_window_is_cut_in_conversation_order(rag_home, rag_conn):
    """Candidate windows are cut in conversation order; cutting by chunk id could drop both true ends."""
    import types

    from core.rag import store

    conn = rag_conn
    scope = "convarchive_legacy"
    documents, per_document = 100, 3
    position_of = {}
    for position in range(documents):
        document_id = f"{(position + documents // 2) % documents:04d}-turn"
        position_of[document_id] = position
        store.create_document(
            conn,
            scope = scope,
            thread_id = "t",
            filename = "earlier turn",
            sha256 = f"h{position}",
            status = "completed",
            embedding_model = "m",
            archive_messages = 2,
            archive_ordinal = None,
            document_id = document_id,
            created_at = "2026-01-01T00:00:00+00:00",
            commit = False,
        )
        chunks = [
            types.SimpleNamespace(
                chunk_index = index,
                text = "ZQXTIEBREAK legacy turn statement",
                page_number = None,
                source_page_index = None,
                token_count = 5,
                char_count = 20,
            )
            for index in range(per_document)
        ]
        store.add_chunks(conn, scope, document_id, chunks, [[0.0] * 4] * per_document)
    conn.commit()

    everything = store.search_lexical(conn, scope, "ZQXTIEBREAK", documents * per_document + 10)
    assert len(everything) == documents * per_document
    assert len({score for _, score in everything}) == 1

    half = 128
    oldest = [
        c for c, _ in store.search_lexical(conn, scope, "ZQXTIEBREAK", half, oldest_first = True)
    ]
    newest = [
        c for c, _ in store.search_lexical(conn, scope, "ZQXTIEBREAK", half, newest_first = True)
    ]
    in_oldest = sorted({position_of[chunk.rsplit(":", 1)[0]] for chunk in oldest})
    in_newest = sorted({position_of[chunk.rsplit(":", 1)[0]] for chunk in newest})

    assert 0 in in_oldest, in_oldest
    assert documents - 1 in in_newest, in_newest
    assert in_oldest[0] == 0 and in_oldest == list(range(len(in_oldest))), in_oldest
    assert in_newest[-1] == documents - 1, in_newest
    assert in_newest == list(range(documents - len(in_newest), documents)), in_newest
    assert not set(in_oldest) & set(in_newest)


def test_the_candidate_order_survives_a_re_embed(rag_home, rag_conn):
    """A re-embed must keep the rowid that the candidate ORDER BY sorts on, else order silently changes."""
    import types

    from core.rag import store

    conn = rag_conn
    scope = "convarchive_reembed"
    turns = 5

    def _document_id(position):
        return f"{turns - position:04d}-turn"

    def _write(
        position,
        model,
        *,
        rowid = None,
        created = None,
        ordinal = None,
    ):
        store.create_document(
            conn,
            scope = scope,
            thread_id = "t",
            filename = "earlier turn",
            sha256 = f"h{position}",
            status = "completed",
            embedding_model = model,
            archive_messages = 2,
            archive_ordinal = ordinal,
            document_id = _document_id(position),
            # One tick for every turn, so only rowid records order.
            created_at = created or "2026-01-01T00:00:00+00:00",
            rowid = rowid,
            commit = False,
        )
        store.add_chunks(
            conn,
            scope,
            _document_id(position),
            [
                types.SimpleNamespace(
                    chunk_index = 0,
                    text = "ZQXREEMBED legacy turn statement",
                    page_number = None,
                    source_page_index = None,
                    token_count = 5,
                    char_count = 20,
                )
            ],
            [[0.0] * 4],
        )

    for position in range(turns):
        _write(position, "old-model")
    conn.commit()

    def _positions(**direction):
        hits = store.search_lexical(conn, scope, "ZQXREEMBED", turns + 10, **direction)
        return [turns - int(chunk.rsplit(":", 1)[0].split("-")[0]) for chunk, _s in hits]

    assert _positions(oldest_first = True) == list(range(turns))
    rowids_before = dict(
        conn.execute("SELECT id, rowid FROM documents WHERE scope=?", (scope,)).fetchall()
    )

    for position in [4, 3, 2]:
        identity = store.document_rewrite_identity(conn, _document_id(position)) or {}
        store.delete_document(conn, _document_id(position), commit = False)
        _write(
            position,
            "new-model",
            rowid = identity.get("rowid"),
            created = identity.get("created_at"),
            ordinal = identity.get("archive_ordinal"),
        )
    conn.commit()

    # The premise, and the test is vacuous without it: the rows really were replaced.
    assert {
        row[0]
        for row in conn.execute("SELECT embedding_model FROM documents WHERE scope=?", (scope,))
    } == {"old-model", "new-model"}
    assert (
        dict(conn.execute("SELECT id, rowid FROM documents WHERE scope=?", (scope,)).fetchall())
        == rowids_before
    )
    assert _positions(oldest_first = True) == list(range(turns))
    assert _positions(newest_first = True) == list(reversed(range(turns)))


def test_a_legacy_archive_still_gets_two_different_ends(rag_home, rag_conn):
    """Legacy NULL-ordinal archives must still give two distinct window ends, not one repeated query."""
    import types

    from core.rag import store

    conn = rag_conn
    scope = "convarchive_legacy"
    for index in range(8):
        document = store.create_document(
            conn,
            scope = scope,
            thread_id = "t",
            filename = f"earlier turn {index}",
            sha256 = f"h{index}",
            status = "completed",
            embedding_model = "m",
            archive_messages = 2,
            archive_ordinal = None,
            commit = False,
        )
        chunk = types.SimpleNamespace(
            chunk_index = 0,
            text = f"ZQXLEGACY token number {index}",
            page_number = None,
            source_page_index = None,
            token_count = 5,
            char_count = 20,
        )
        store.add_chunks(conn, scope, document, [chunk], [[0.0, 0.0, 0.0, 0.0]])
    conn.commit()

    oldest = [
        chunk for chunk, _ in store.search_lexical(conn, scope, "ZQXLEGACY", 3, oldest_first = True)
    ]
    newest = [
        chunk for chunk, _ in store.search_lexical(conn, scope, "ZQXLEGACY", 3, newest_first = True)
    ]

    assert oldest and newest
    assert oldest != newest, "both ends of the fetch returned the same rows"
    assert not set(oldest) & set(newest), (oldest, newest)
