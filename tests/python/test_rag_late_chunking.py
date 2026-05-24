"""Late chunking tests (Phase 3B-late).

Pure-python coverage of `chunk_pages_with_spans` runs always. The
encoder test loads a small SentenceTransformer and is gated behind the
existing `server` marker so default `pytest` runs skip it.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

from core.rag.chunking import chunk_pages_with_spans
from core.rag.parsers import ParsedPage


def _wc_counter(text: str) -> int:
    return max(1, len(text.split()))


def test_spans_index_back_to_full_doc_text():
    pages = [
        ParsedPage(text = "# Section A\n\n" + ("alpha " * 20), page_number = 1),
        ParsedPage(text = "# Section B\n\n" + ("beta " * 20), page_number = 2),
    ]
    full_doc, chunks, char_spans = chunk_pages_with_spans(
        pages,
        max_tokens = 12,
        overlap_tokens = 0,
        token_counter = _wc_counter,
    )
    assert chunks
    assert len(chunks) == len(char_spans)
    for chunk, (start, end) in zip(chunks, char_spans):
        # The chunk text must be exactly the slice of full_doc it claims.
        assert full_doc[start:end] == chunk.text


def test_chunks_inherit_page_number_by_overlap():
    pages = [
        ParsedPage(text = "page-one text here", page_number = 1),
        ParsedPage(text = "page-two text here", page_number = 2),
    ]
    _full_doc, chunks, _spans = chunk_pages_with_spans(
        pages,
        max_tokens = 4,
        overlap_tokens = 0,
        token_counter = _wc_counter,
    )
    pages_seen = {c.page_number for c in chunks}
    assert pages_seen <= {1, 2}
    # Both pages should contribute at least one chunk.
    assert 1 in pages_seen
    assert 2 in pages_seen


def test_full_doc_joins_pages_with_blank_line_separator():
    pages = [
        ParsedPage(text = "first", page_number = 1),
        ParsedPage(text = "second", page_number = 2),
    ]
    full_doc, _chunks, _spans = chunk_pages_with_spans(
        pages,
        max_tokens = 5,
        overlap_tokens = 0,
        token_counter = _wc_counter,
    )
    assert "first" in full_doc
    assert "second" in full_doc
    # The two pages must be separated by exactly one blank line.
    assert "first\n\nsecond" in full_doc


@pytest.mark.server
def test_late_chunk_encode_returns_one_vector_per_span():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("torch")
    # all-MiniLM-L6-v2 is ~80MB and embeds at 384 dims.
    import os
    os.environ.setdefault(
        "UNSLOTH_RAG_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    from core.rag import embeddings as embeddings_module

    embeddings_module._model = None  # force re-load
    embeddings_module._model_name = None

    doc_text = (
        "# Intro\n\n"
        "The quick brown fox jumps over the lazy dog.\n\n"
        "# Methods\n\n"
        "We trained the model on a corpus of 100M tokens.\n\n"
        "# Results\n\n"
        "Accuracy improved by 12% over the baseline."
    )
    # char_spans for three chunks — one per section, picked manually.
    char_spans = [
        (doc_text.index("The quick"), doc_text.index("\n\n# Methods")),
        (doc_text.index("We trained"), doc_text.index("\n\n# Results")),
        (doc_text.index("Accuracy"), len(doc_text)),
    ]
    vectors = embeddings_module.late_chunk_encode(doc_text, char_spans)
    assert len(vectors) == len(char_spans)
    dim = vectors[0].shape[0]
    for v in vectors:
        assert v.shape == (dim,)
