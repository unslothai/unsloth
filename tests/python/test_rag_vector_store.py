"""sqlite-vec backed RAG vector store tests."""

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
    """Point rag.db at tmp_path and reset the cached connection so
    each test gets a fresh database.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    from core.rag import db as rag_db

    rag_db._reset_for_tests()
    yield tmp_path
    rag_db._reset_for_tests()


def test_upsert_and_search_returns_nearest_first(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_test"
    vector_store.ensure_collection(scope, dim = 4)  # no-op under sqlite-vec
    vector_store.upsert_chunks(
        scope,
        [
            {
                "id": "p1",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "payload": {"document_id": "doc1", "chunk_index": 0, "text": "first"},
            },
            {
                "id": "p2",
                "vector": [0.0, 1.0, 0.0, 0.0],
                "payload": {"document_id": "doc1", "chunk_index": 1, "text": "second"},
            },
        ],
    )
    results = vector_store.search(scope, [1.0, 0.0, 0.0, 0.0], top_k = 2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "p1"
    # Cosine mapped to [0, 1]; closer = higher.
    assert results[0]["score"] > results[1]["score"]


def test_collection_exists_tracks_populated_scope(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_to_delete"
    assert not vector_store.collection_exists(scope)
    vector_store.upsert_chunks(
        scope,
        [
            {
                "id": "sole",
                "vector": [1.0, 0.0, 0.0],
                "payload": {"document_id": "d", "chunk_index": 0},
            }
        ],
    )
    assert vector_store.collection_exists(scope)
    vector_store.delete_scope(scope)
    assert not vector_store.collection_exists(scope)


def test_delete_document_removes_only_its_points(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_doc_del"
    vector_store.upsert_chunks(
        scope,
        [
            {
                "id": "a",
                "vector": [1.0, 0.0, 0.0],
                "payload": {"document_id": "keep", "chunk_index": 0},
            },
            {
                "id": "b",
                "vector": [0.0, 1.0, 0.0],
                "payload": {"document_id": "drop", "chunk_index": 0},
            },
        ],
    )
    vector_store.delete_document(scope, "drop")
    results = vector_store.search(scope, [0.0, 1.0, 0.0], top_k = 5)
    doc_ids = {r["payload"]["document_id"] for r in results}
    assert "drop" not in doc_ids
    assert "keep" in doc_ids


def test_search_filtered_by_document_ids(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_filter"
    vector_store.upsert_chunks(
        scope,
        [
            {
                "id": "a",
                "vector": [1.0, 0.0, 0.0],
                "payload": {"document_id": "alpha", "chunk_index": 0},
            },
            {
                "id": "b",
                "vector": [1.0, 0.0, 0.0],
                "payload": {"document_id": "beta", "chunk_index": 0},
            },
        ],
    )
    results = vector_store.search(
        scope,
        [1.0, 0.0, 0.0],
        top_k = 5,
        document_ids = ["alpha"],
    )
    doc_ids = {r["payload"]["document_id"] for r in results}
    assert doc_ids == {"alpha"}


def test_upsert_overwrites_on_conflicting_chunk_id(isolated_rag_db):
    from core.rag import vector_store

    scope = "kb_overwrite"
    vector_store.upsert_chunks(
        scope,
        [
            {
                "id": "same",
                "vector": [1.0, 0.0, 0.0],
                "payload": {"document_id": "d", "chunk_index": 0, "v": "v1"},
            }
        ],
    )
    vector_store.upsert_chunks(
        scope,
        [
            {
                "id": "same",
                "vector": [0.0, 1.0, 0.0],
                "payload": {"document_id": "d", "chunk_index": 0, "v": "v2"},
            }
        ],
    )
    results = vector_store.search(scope, [0.0, 1.0, 0.0], top_k = 5)
    assert len(results) == 1
    assert results[0]["payload"]["v"] == "v2"
