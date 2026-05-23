"""BM25 index lifecycle tests (skipped if bm25s is unavailable)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

pytest.importorskip("bm25s")


@pytest.fixture
def isolated_bm25_root(tmp_path, monkeypatch):
    from utils.paths import storage_roots

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    # Reset module-level cache between tests.
    from core.rag import bm25

    bm25._cache.clear()
    return tmp_path


def test_bm25_index_search_roundtrip(isolated_bm25_root):
    from core.rag import bm25

    scope = "kb_test"
    chunks = [
        {"id": "c1", "text": "the quick brown fox jumps over the lazy dog"},
        {"id": "c2", "text": "machine learning models predict outputs from inputs"},
        {"id": "c3", "text": "fox terriers are small dogs"},
    ]
    bm25.rebuild_index(scope, chunks)
    results = bm25.search(scope, "fox", k = 3)
    ids = [cid for cid, _ in results]
    assert "c1" in ids
    assert "c3" in ids


def test_bm25_empty_returns_empty(isolated_bm25_root):
    from core.rag import bm25

    assert bm25.search("kb_nonexistent", "anything", k = 5) == []


def test_bm25_delete_scope(isolated_bm25_root):
    from core.rag import bm25

    scope = "kb_del"
    chunks = [{"id": "a", "text": "alpha beta gamma"}]
    bm25.rebuild_index(scope, chunks)
    assert bm25.search(scope, "alpha", k = 1)
    bm25.delete_scope(scope)
    assert bm25.search(scope, "alpha", k = 1) == []


def test_bm25_rebuild_replaces_old_corpus(isolated_bm25_root):
    from core.rag import bm25

    scope = "kb_replace"
    bm25.rebuild_index(scope, [{"id": "old", "text": "alpha beta"}])
    bm25.rebuild_index(scope, [{"id": "new", "text": "gamma delta"}])
    results = bm25.search(scope, "alpha", k = 5)
    ids = [cid for cid, _ in results]
    assert "old" not in ids
