"""Unit tests for RAG RRF fusion — no external deps."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

from core.rag.retrieval import Hit, _rrf_fuse


def test_rrf_fuses_two_rankings():
    bm25 = [Hit("a", 10.0), Hit("b", 8.0), Hit("c", 5.0)]
    dense = [Hit("c", 0.9), Hit("b", 0.8), Hit("d", 0.5)]
    fused = _rrf_fuse([bm25, dense], rrf_k = 60, top_k = 3)
    ids = [h.chunk_id for h in fused]
    # b appears at rank 2 in both -> highest fused score
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"} or set(ids) == {"b", "c", "a"}


def test_rrf_top_k_limits_output():
    rankings = [
        [Hit(f"r1_{i}", 0.0) for i in range(20)],
        [Hit(f"r2_{i}", 0.0) for i in range(20)],
    ]
    fused = _rrf_fuse(rankings, rrf_k = 60, top_k = 5)
    assert len(fused) == 5


def test_rrf_unique_ranking():
    # Single ranking — fused order matches input order.
    ranking = [Hit("x", 0.0), Hit("y", 0.0), Hit("z", 0.0)]
    fused = _rrf_fuse([ranking], rrf_k = 60, top_k = 3)
    assert [h.chunk_id for h in fused] == ["x", "y", "z"]


def test_rrf_preserves_payload_from_first_ranking():
    a = Hit("a", 1.0, document_id = "doc1", chunk_index = 5)
    b = Hit("a", 2.0, document_id = "doc2", chunk_index = 7)
    fused = _rrf_fuse([[a], [b]], rrf_k = 60, top_k = 1)
    # First sighting wins for payload (deterministic)
    assert fused[0].document_id == "doc1"
    assert fused[0].chunk_index == 5
