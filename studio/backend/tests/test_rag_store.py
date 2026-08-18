# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Store tests: incremental writes, dedupe, delete, scope, dense + lexical."""

import math
import sqlite3

import pytest

from core.rag import store
from core.rag.chunking import Chunk

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
    assert [cid for cid, _ in hits] == ["d1:0"]  # d2 not returned (score 0)


def test_scope_isolation(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo"])
    _add_doc(rag_conn, "kb_b", "d2", "f", "h2", ["alpha bravo"])
    assert [cid for cid, _ in store.search_lexical(rag_conn, "kb_b", "alpha", 10)] == ["d2:0"]


def test_match_query_sanitizes_special_chars():
    assert store._match_query('AND OR "quote" (paren) -dash') != ""


def test_lexical_does_not_crash_on_punctuation(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha bravo"])
    # Must not raise on FTS operators in the query.
    store.search_lexical(rag_conn, "kb_a", 'NEAR("x" AND', 5)


def test_dense_ranks_by_cosine(rag_conn):
    _add_doc(rag_conn, "kb_a", "d1", "f", "h1", ["alpha alpha"])
    _add_doc(rag_conn, "kb_a", "d2", "f", "h2", ["hotel golf"])
    ranked = store.search_dense(rag_conn, "kb_a", embed("alpha"), 10)
    assert ranked[0][0] == "d1:0" and ranked[0][1] > 0.99


def test_dense_empty_before_any_ingest(rag_conn):
    # No chunks_vec table yet -> [], no crash.
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
    # Adding doc2 must not touch doc1's fts rowids (append, not rebuild).
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
    # d2 belongs to a folder but has no mapping row yet, so it is not searchable.
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
    _link_folder(rag_conn, "f1", "kb_b")  # another scope, so nothing is excluded
    filtered = store.search_lexical(rag_conn, "kb_a", "alpha bravo", 5)
    assert fast == filtered


def test_lexical_gate_and_read_share_one_snapshot(rag_conn, monkeypatch):
    """A scope retired between the gate and the FTS read must not reach the read.

    Otherwise the gate decides against a state the read no longer sees, and rows from the
    retired scope take slots the caller loses at hydration.
    """
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
    # Same connection, same call, after another connection committed the retirement.
    assert observed["after_commit"] is False
    assert [cid for cid, _ in hits] == ["d1:0"]
    # The snapshot is released, so the next call sees the retirement and hides the row.
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
    """Deleting a knowledge base must not disable the fast path for good.

    delete_retired_scope keeps the tombstone and only stamps purged_at, so a gate that
    counted it would take the filtered query forever after the first ordinary delete.
    """
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
    """An orphan left by a crash before _install_mapping stays hidden after unlink.

    Unlink collects only mapped documents, so the folder row goes and this one does not;
    the gate has to see the document itself or unlinked content becomes searchable.
    """
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
    """Distinct ordinary words, as a pasted log or source file supplies them.

    Purely alphabetic on purpose: a token mixing letters and digits short-circuits the
    identifier test on its first clause and never reaches the scan being measured, so a
    synthetic `tok1 tok2 ...` paste hides the cost that real prose pays.
    """
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
    """Shaping the archive query must not re-tokenize the question once per token.

    `conversation_match_queries` runs on the LATEST USER MESSAGE, and the message that
    forces a compaction is very often a pasted log or source file. Re-scanning the whole
    question inside the per-token identifier test made the shaping cost grow with the
    square of the question's length: 48 KB of pasted prose measured at 4.6s and 96 KB at
    17.7s of pure CPU, against 2.3ms for the same text through `_match_query`. The recall
    path can run the shaping several times per request -- once per widening iteration in
    `conversation_archive.recall`, and again for each rung of the over-budget top_k
    backoff -- so the multiplier lands on the one turn that compacts the thread.

    Counted rather than timed, so the guard is deterministic: the number of full scans of
    the question is what has to stay bounded, not the wall clock on one machine.
    """
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
    # Once for the lower-cased tokens, once for the raw ones. Anything that grows with
    # the token count is the quadratic coming back.
    assert scans["n"] <= 2, f"tokenized the question {scans['n']} times"


def test_query_shaping_stays_cheap_on_a_pasted_log():
    """The wall-clock companion to the scan count, with a wide margin.

    6000 pasted words is roughly a 48 KB paste, which is one source file. Unfixed this
    takes about 4.6s of CPU; linear it takes about 6ms. A 1.0s ceiling is unreachable by
    a linear implementation on any machine that can run this suite at all.
    """
    import time

    question = f"what is the current value of ZQXVARA123\n{_pasted_prose(6000)}"
    started = time.perf_counter()
    expressions = store.conversation_match_queries(question)
    elapsed = time.perf_counter() - started
    assert expressions and expressions[0] == '"zqxvara123"'
    assert elapsed < 1.0, f"shaping a 6000-word paste took {elapsed:.2f}s"
