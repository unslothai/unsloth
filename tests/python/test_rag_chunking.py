"""Unit tests for RAG chunking — pure-python, no external deps."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))

from core.rag.chunking import Chunk, chunk_pages
from core.rag.parsers import ParsedPage


def _wc_counter(text: str) -> int:
    return max(1, len(text.split()))


def test_chunk_pages_splits_long_text():
    text = "Lorem ipsum dolor sit amet. " * 200
    chunks = chunk_pages(
        [ParsedPage(text = text)],
        max_tokens = 50,
        overlap_tokens = 5,
        token_counter = _wc_counter,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert _wc_counter(chunk.text) <= 55  # max + small slack from atomic split granularity


def test_chunk_pages_short_text_is_one_chunk():
    text = "Just a short sentence."
    chunks = chunk_pages(
        [ParsedPage(text = text)],
        max_tokens = 50,
        overlap_tokens = 5,
        token_counter = _wc_counter,
    )
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_pages_preserves_page_numbers():
    chunks = chunk_pages(
        [
            ParsedPage(text = "Page one content here.", page_number = 1),
            ParsedPage(text = "Page two content here.", page_number = 2),
        ],
        max_tokens = 50,
        overlap_tokens = 0,
        token_counter = _wc_counter,
    )
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}


def test_chunk_pages_no_empty_chunks():
    text = "\n\n\n\n\nReal content\n\n\n\n\n"
    chunks = chunk_pages(
        [ParsedPage(text = text)],
        max_tokens = 50,
        overlap_tokens = 0,
        token_counter = _wc_counter,
    )
    for chunk in chunks:
        assert chunk.text.strip()


def test_chunk_pages_overlap_produces_repeated_tokens():
    # Build a list of unique numbered sentences so we can detect overlap.
    sentences = [f"sentence-{i}" for i in range(40)]
    text = " ".join(sentences)
    chunks = chunk_pages(
        [ParsedPage(text = text)],
        max_tokens = 10,
        overlap_tokens = 4,
        token_counter = _wc_counter,
    )
    if len(chunks) >= 2:
        first_tail_words = set(chunks[0].text.split()[-4:])
        second_head_words = set(chunks[1].text.split()[:4])
        # At least one word should appear in both
        assert first_tail_words & second_head_words


def test_chunk_pages_splits_on_markdown_headings():
    # Phase 3A: heading separators take priority over paragraph breaks
    # so chunks start at section boundaries when the parser emits
    # Markdown.
    md = (
        "# First Section\n\n"
        + "alpha " * 30
        + "\n\n## Subsection A\n\n"
        + "beta " * 30
        + "\n\n# Second Section\n\n"
        + "gamma " * 30
    )
    chunks = chunk_pages(
        [ParsedPage(text = md)],
        max_tokens = 25,
        overlap_tokens = 0,
        token_counter = _wc_counter,
    )
    # We expect multiple chunks and at least one to begin at a heading.
    assert len(chunks) >= 2
    starts_at_heading = sum(
        1 for c in chunks if c.text.lstrip().startswith(("# ", "## "))
    )
    assert starts_at_heading >= 1, (
        f"expected at least one chunk to start at a Markdown heading; "
        f"got starts: {[c.text[:20] for c in chunks]}"
    )
