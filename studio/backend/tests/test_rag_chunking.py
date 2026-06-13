# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunking unit tests (no DB, no model)."""

from core.rag.chunking import chunk_pages
from core.rag.parsers import Page, parse_text

WORDS = lambda t: len(t.split())  # noqa: E731


def _page(text: str, page_number = None) -> Page:
    return Page(text = text, page_number = page_number, char_count = len(text))


def test_chunk_token_bounds_and_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_pages([_page(text)], max_tokens = 128, overlap = 24, count = WORDS)
    assert len(chunks) >= 3
    assert all(c.token_count <= 128 for c in chunks)
    a, b = chunks[0].text.split(), chunks[1].text.split()
    shared = next((n for n in range(60, 0, -1) if a[-n:] == b[:n]), 0)
    assert shared == 24  # exactly overlap tokens carried


def test_chunk_never_exceeds_max_with_overlap_carry():
    """Overlap carry is trimmed so no chunk exceeds max_tokens (else the embedder overflows)."""
    s1 = " ".join("a" for _ in range(10))
    s2 = " ".join("b" for _ in range(95))  # near max
    chunks = chunk_pages(
        [_page(f"{s1}. {s2}")], max_tokens = 100, overlap = 24, count = WORDS
    )
    assert all(c.token_count <= 100 for c in chunks), [c.token_count for c in chunks]


def test_chunk_indices_are_sequential():
    chunks = chunk_pages(
        [_page("alpha. " * 200)], max_tokens = 32, overlap = 0, count = WORDS
    )
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_tracks_source_page_index():
    pages = [_page("alpha bravo " * 80, 1), _page("charlie delta " * 80, 2)]
    chunks = chunk_pages(pages, max_tokens = 32, overlap = 0, count = WORDS)
    page0 = [c for c in chunks if c.source_page_index == 0]
    page1 = [c for c in chunks if c.source_page_index == 1]
    assert page0 and page1
    assert all(c.page_number == 1 for c in page0)
    assert all(c.page_number == 2 for c in page1)


def test_chunk_char_offsets_locate_text_in_page():
    # Each chunk's char span must slice back to text containing it.
    page_text = "alpha bravo charlie delta echo foxtrot golf hotel " * 30
    pages = [_page(page_text, 1)]
    chunks = chunk_pages(pages, max_tokens = 16, overlap = 0, count = WORDS)
    assert len(chunks) > 1
    for c in chunks:
        assert 0 <= c.page_char_start < c.page_char_end <= len(page_text)
        sliced = page_text[c.page_char_start : c.page_char_end]
        assert c.text in sliced or sliced.strip() == c.text


def test_empty_page_yields_no_chunks():
    chunks = chunk_pages([_page("   \n  ")], max_tokens = 32, overlap = 0, count = WORDS)
    assert chunks == []


def test_parse_text_single_page():
    pages = parse_text("hello world")
    assert len(pages) == 1
    assert pages[0].char_count == len("hello world")
