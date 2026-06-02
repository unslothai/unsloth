"""FTS5 lexical index lifecycle tests (sqlite-vec backed rag.db)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

pytest.importorskip("sqlite_vec")


@pytest.fixture
def isolated_rag_db(tmp_path, monkeypatch):
    """Point rag.db at tmp_path and reset the cached connection so each
    test gets a fresh database."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    from core.rag import db as rag_db

    rag_db._reset_for_tests()
    yield tmp_path
    rag_db._reset_for_tests()


def _chunk(cid: str, text: str, document_id: str = "doc1", index: int = 0) -> dict:
    return {
        "id": cid,
        "vector": [1.0, 0.0, 0.0, 0.0],
        "payload": {
            "document_id": document_id,
            "chunk_index": index,
            "kind": "text",
            "text": text,
        },
    }


def test_lexical_search_roundtrip(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_test"
    vector_store.upsert_chunks(
        scope,
        [
            _chunk("c1", "the quick brown fox jumps over the lazy dog", index = 0),
            _chunk("c2", "machine learning models predict outputs from inputs", index = 1),
            _chunk("c3", "fox terriers are small dogs", index = 2),
        ],
    )
    results = vector_store.search_lexical(scope, "fox", k = 3)
    ids = [cid for cid, _ in results]
    assert "c1" in ids
    assert "c3" in ids
    assert "c2" not in ids
    # Scores are flipped to higher-is-better.
    assert all(score >= 0 for _, score in results)


def test_lexical_empty_returns_empty(isolated_rag_db):
    from core.rag import vector_store

    assert vector_store.search_lexical("kb_nonexistent", "anything", k = 5) == []


def test_lexical_blank_query_returns_empty(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_blank"
    vector_store.upsert_chunks(scope, [_chunk("a", "alpha beta gamma")])
    # No word tokens → no MATCH expression → empty (not a syntax error).
    assert vector_store.search_lexical(scope, "!!! ???", k = 5) == []


def test_lexical_query_operators_are_defused(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_ops"
    vector_store.upsert_chunks(scope, [_chunk("a", "alpha beta gamma")])
    # Bare FTS5 operators would raise without sanitization.
    results = vector_store.search_lexical(scope, "alpha OR NOT (beta)", k = 5)
    assert [cid for cid, _ in results] == ["a"]


def test_lexical_scope_isolation(isolated_rag_db):
    from core.rag import vector_store

    vector_store.upsert_chunks("kb_one", [_chunk("x", "shared keyword here")])
    vector_store.upsert_chunks("kb_two", [_chunk("y", "shared keyword here")])
    results = vector_store.search_lexical("kb_one", "keyword", k = 5)
    assert [cid for cid, _ in results] == ["x"]


def test_lexical_delete_scope(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_del"
    vector_store.upsert_chunks(scope, [_chunk("a", "alpha beta gamma")])
    assert vector_store.search_lexical(scope, "alpha", k = 1)
    vector_store.delete_scope(scope)
    assert vector_store.search_lexical(scope, "alpha", k = 1) == []


def test_lexical_delete_document(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_doc_del"
    vector_store.upsert_chunks(
        scope,
        [
            _chunk("keep1", "alpha keyword", document_id = "keep"),
            _chunk("drop1", "beta keyword", document_id = "drop"),
        ],
    )
    vector_store.delete_document(scope, "drop")
    ids = [cid for cid, _ in vector_store.search_lexical(scope, "keyword", k = 5)]
    assert ids == ["keep1"]


def test_lexical_reingest_is_idempotent(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_reingest"
    vector_store.upsert_chunks(scope, [_chunk("a", "alpha beta")])
    vector_store.upsert_chunks(scope, [_chunk("a", "gamma delta")])
    # No duplicate FTS row; old text no longer matches.
    assert vector_store.search_lexical(scope, "alpha", k = 5) == []
    ids = [cid for cid, _ in vector_store.search_lexical(scope, "gamma", k = 5)]
    assert ids == ["a"]


def test_lexical_caption_kind_is_indexed(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_caption"
    point = _chunk("cap", "Figure 1: a diagram of the pipeline")
    point["payload"]["kind"] = "caption"
    vector_store.upsert_chunks(scope, [point])
    ids = [cid for cid, _ in vector_store.search_lexical(scope, "diagram", k = 5)]
    assert ids == ["cap"]


def test_fts_backfill_from_existing_vectors(isolated_rag_db):
    """rag.db files created before FTS5 have vectors but an empty FTS table;
    opening the connection backfills it once."""
    import sqlite_vec

    from core.rag import db as rag_db

    # Seed vectors directly, bypassing upsert_chunks' incremental FTS insert.
    conn = rag_db.get_rag_connection()
    conn.execute(
        """
        INSERT INTO rag_vectors
            (chunk_id, scope, document_id, chunk_index, kind, dim, vector, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy",
            "kb_legacy",
            "doc",
            0,
            "text",
            4,
            sqlite_vec.serialize_float32([1.0, 0.0, 0.0, 0.0]),
            '{"text": "backfilled lexical content", "kind": "text"}',
        ),
    )
    conn.execute("DELETE FROM rag_chunks_fts")
    conn.commit()
    # Force a fresh connection so _ensure_schema runs the guarded backfill.
    rag_db._reset_for_tests()

    from core.rag import vector_store

    ids = [cid for cid, _ in vector_store.search_lexical("kb_legacy", "lexical", k = 5)]
    assert ids == ["legacy"]
