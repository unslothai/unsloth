# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""pdf_regions_for_chunks reads each page once per run and matches per-chunk location."""

import random

import pytest

pymupdf = pytest.importorskip("pymupdf")

from core.rag import locators
from core.rag.chunking import chunk_pages
from core.rag.locators import (
    MAX_ANCHOR_WORDS,
    LocatorMatch,
    _anchor_tokens,
    _norm_token,
    _regions_for_match,
)
from core.rag.parsers import parse_pdf_bytes

_WORDS = [
    "alpha", "beta", "gamma", "delta", "signiﬁcant", "eﬀort", "Revenue", "Q1",
    "growth", "model", "data", "**bold**", "(paren)", "end.", "the", "of", "and", "cell|row",
]  # fmt: skip


def _pdf(tmp_path):
    """Five dense pages: page 1 is empty, page 3 is rotated."""
    rng = random.Random(7)
    doc = pymupdf.open()
    for i in range(5):
        page = doc.new_page()
        if i == 1:
            continue
        for y in range(60, 780, 12):
            page.insert_text((50, y), " ".join(rng.choice(_WORDS) for _ in range(12)), fontsize = 9)
        if i == 3:
            page.set_rotation(90)
    path = tmp_path / "memo.pdf"
    doc.save(str(path))
    doc.close()
    pages, _ = parse_pdf_bytes(path.read_bytes())
    chunks = chunk_pages(pages, max_tokens = 64, overlap = 8, count = lambda t: len(t.split()))
    return path, pages, chunks


def _one_by_one(path, pages, chunks):
    """Each chunk located on a freshly opened page, as before the memo."""
    doc = pymupdf.open(str(path))
    try:
        return [
            _regions_for_match(
                doc,
                pages[c.source_page_index].text,
                LocatorMatch(
                    c.source_page_index, c.page_number, c.page_char_start, c.page_char_end
                ),
            )
            for c in chunks
        ]
    finally:
        doc.close()


@pytest.fixture
def get_text_calls(monkeypatch):
    calls = []
    original = pymupdf.Page.get_text

    def counting(self, *args, **kwargs):
        calls.append(self.number)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", counting)
    return calls


def test_regions_match_per_chunk_location_in_page_order(tmp_path, get_text_calls):
    path, pages, chunks = _pdf(tmp_path)
    expected = _one_by_one(path, pages, chunks)
    assert sum(1 for r in expected if r) > len(chunks) // 2
    del get_text_calls[:]
    assert locators.pdf_regions_for_chunks(path, pages, chunks) == expected
    assert get_text_calls == sorted({c.source_page_index for c in chunks})


def test_regions_match_when_pages_interleave(tmp_path, get_text_calls):
    """A page revisited after another page must not reuse the other page's words."""
    path, pages, chunks = _pdf(tmp_path)
    shuffled = list(chunks)
    random.Random(1).shuffle(shuffled)
    expected = _one_by_one(path, pages, shuffled)
    assert locators.pdf_regions_for_chunks(path, pages, shuffled) == expected


def test_a_failed_page_read_is_retried_for_the_next_chunk(tmp_path, monkeypatch):
    path, pages, chunks = _pdf(tmp_path)
    expected = _one_by_one(path, pages, chunks)
    first_page = [i for i, c in enumerate(chunks) if c.source_page_index == 0]
    assert len(first_page) >= 3 and expected[first_page[0]] and expected[first_page[1]]
    original = pymupdf.Page.get_text
    state = {"failed": False}

    def flaky(self, *args, **kwargs):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("transient")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", flaky)
    got = locators.pdf_regions_for_chunks(path, pages, chunks)
    assert got[first_page[0]] == []
    assert got[first_page[1] :] == expected[first_page[1] :]


def test_anchor_tokens_equal_the_full_normalized_prefix():
    rng = random.Random(3)
    for _ in range(200):
        text = " ".join(
            rng.choice(_WORDS + ["...", "--", "|", "—"]) for _ in range(rng.randint(0, 40))
        )
        match = LocatorMatch(0, 1, 0, len(text))
        raw = text.replace("|", " ").split()
        if len(raw) >= 6:
            raw = raw[1:-1]
        full = [t for t in (_norm_token(w) for w in raw) if t]
        assert _anchor_tokens(text, match) == full[:MAX_ANCHOR_WORDS]
