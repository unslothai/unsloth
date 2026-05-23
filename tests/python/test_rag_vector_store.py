"""Qdrant local-mode vector store tests (skipped if qdrant-client is unavailable)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

pytest.importorskip("qdrant_client")


@pytest.fixture
def isolated_qdrant(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    from core.rag import vector_store

    # Reset client cache so the fixture's tmp path is used.
    vector_store._client = None
    yield tmp_path
    vector_store._client = None


def test_ensure_and_upsert_and_search(isolated_qdrant):
    from core.rag import vector_store

    scope = "kb_test"
    vector_store.ensure_collection(scope, dim = 4)
    points = [
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
    ]
    vector_store.upsert_chunks(scope, points)
    results = vector_store.search(scope, [1.0, 0.0, 0.0, 0.0], top_k = 2)
    assert results
    assert results[0]["chunk_id"] == "p1"


def test_delete_scope_removes_collection(isolated_qdrant):
    from core.rag import vector_store

    scope = "kb_to_delete"
    vector_store.ensure_collection(scope, dim = 3)
    assert vector_store.collection_exists(scope)
    vector_store.delete_scope(scope)
    assert not vector_store.collection_exists(scope)


def test_delete_document_removes_only_its_points(isolated_qdrant):
    from core.rag import vector_store

    scope = "kb_doc_del"
    vector_store.ensure_collection(scope, dim = 3)
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
