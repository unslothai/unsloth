"""Reranker tests — skipped if sentence_transformers is unavailable.

These tests load a real CrossEncoder, so they're slow and gated under
the ``server`` marker so a default ``pytest`` run skips them. Force
with ``pytest -m server``.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

pytest.importorskip("sentence_transformers")


def test_rerank_empty_returns_empty():
    from core.rag.reranker import rerank

    assert rerank("anything", []) == []


@pytest.mark.server
def test_rerank_reorders_by_relevance(monkeypatch):
    """Hide the relevant chunk at the back of the input and check it bubbles up."""
    monkeypatch.setenv("UNSLOTH_RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    from core.rag.reranker import rerank, unload
    from core.rag.retrieval import Hit

    pairs = [
        (Hit("noise1", 0.0), "Cats are small carnivorous mammals."),
        (Hit("noise2", 0.0), "The Eiffel Tower is in Paris, France."),
        (Hit("noise3", 0.0), "Python is a programming language."),
        (Hit("answer", 0.0), "The speed of light in vacuum is approximately 299792458 meters per second."),
    ]
    try:
        ranked = rerank("How fast does light travel?", pairs, top_k = 2)
        assert ranked
        assert ranked[0].chunk_id == "answer"
    finally:
        unload()


@pytest.mark.server
def test_unload_clears_singleton(monkeypatch):
    monkeypatch.setenv("UNSLOTH_RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    from core.rag import reranker
    from core.rag.retrieval import Hit

    reranker.rerank("q", [(Hit("a", 0.0), "some text")])
    assert reranker._model is not None
    reranker.unload()
    assert reranker._model is None
