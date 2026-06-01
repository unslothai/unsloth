# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDF region locators: map a chunk back to highlight rectangles on its page.

Computed once at ingest from the live parse (no backfill, no persisted page
text). For each chunk we take a short anchor phrase from the start of its page
span and locate it in the page's word list (``page.get_text("words")``), which
shares the exact extraction source as the chunk text -- same ligatures, same
hyphenation, same tokenization -- so matching is robust where a glyph-exact
``page.search_for`` is not (it misses ligatures like ``ﬁ`` and dehyphenated
words). Matched words are unioned per line into rectangles, normalized to 0..1
of the page so the frontend can draw highlights at any zoom.

Conservative by design: a missing PyMuPDF, too short an anchor, or an anchor
that cannot be located uniquely yields an empty region list rather than a
guessed highlight.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Anchor sizing. We take up to MAX_ANCHOR_WORDS interior words from the chunk's
# leading span and shrink toward MIN_ANCHOR_WORDS to recover a unique match.
MAX_ANCHOR_WORDS = 12
MIN_ANCHOR_WORDS = 4


@dataclass(frozen = True)
class LocatorMatch:
    page_index: int
    page_number: int | None
    start: int
    end: int


def _norm_token(token: str) -> str:
    """Canonical form for matching: NFKC (decomposes ligatures), casefold,
    strip surrounding punctuation/markdown. Returns "" for punctuation-only."""
    token = unicodedata.normalize("NFKC", token).casefold()
    return token.strip(" \t\r\n*#`[]()_.,;:!?\"'“”‘’-–—…|/\\")


def _anchor_tokens(page_text: str, match: LocatorMatch) -> list[str]:
    """Normalized anchor tokens from the start of the chunk's page span.

    The first and last whitespace-delimited tokens are dropped when the span is
    long enough, since a chunk boundary often slices through a word.
    """
    segment = page_text[match.start : match.end]
    raw = segment.split()
    if len(raw) >= MIN_ANCHOR_WORDS + 2:
        raw = raw[1:-1]
    tokens = [t for t in (_norm_token(w) for w in raw) if t]
    return tokens[:MAX_ANCHOR_WORDS]


def _find_subsequences(haystack: list[str], needle: list[str]) -> list[int]:
    """All start indices where ``needle`` occurs consecutively in ``haystack``."""
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return []
    first = needle[0]
    out: list[int] = []
    for i in range(n - m + 1):
        if haystack[i] == first and haystack[i : i + m] == needle:
            out.append(i)
    return out


def _locate(page_words: list, needle: list[str]) -> list[int] | None:
    """Return the matched word indices for the best anchor, or None.

    Tries the full anchor first, then progressively shorter prefixes, accepting
    the first length that matches exactly once. A still-ambiguous match falls
    back to its first occurrence (the anchor was lifted from this chunk's own
    leading span, so the first hit is almost always the chunk's location).
    """
    # Filtered tokens keep an index map back to the original word rects so
    # punctuation-only words never break an otherwise-contiguous phrase.
    tokens: list[str] = []
    idx_map: list[int] = []
    for j, w in enumerate(page_words):
        t = _norm_token(w[4])
        if t:
            tokens.append(t)
            idx_map.append(j)

    ambiguous_first: list[int] | None = None
    for size in range(len(needle), MIN_ANCHOR_WORDS - 1, -1):
        sub = needle[:size]
        hits = _find_subsequences(tokens, sub)
        if len(hits) == 1:
            p = hits[0]
            return [idx_map[p + k] for k in range(size)]
        if hits and ambiguous_first is None:
            p = hits[0]
            ambiguous_first = [idx_map[p + k] for k in range(size)]
    return ambiguous_first


def _rects_from_words(page_words: list, indices: list[int], pw: float, ph: float):
    """Union matched words per (block, line) into normalized page rectangles."""
    lines: dict[tuple, list[float]] = {}
    for j in indices:
        w = page_words[j]
        x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
        key = (w[5], w[6])  # block, line
        box = lines.get(key)
        if box is None:
            lines[key] = [x0, y0, x1, y1]
        else:
            box[0], box[1] = min(box[0], x0), min(box[1], y0)
            box[2], box[3] = max(box[2], x1), max(box[3], y1)

    out: list[dict[str, Any]] = []
    for x0, y0, x1, y1 in lines.values():
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            continue
        out.append(
            {
                "x": max(0.0, min(1.0, x0 / pw)),
                "y": max(0.0, min(1.0, y0 / ph)),
                "width": max(0.0, min(1.0, w / pw)),
                "height": max(0.0, min(1.0, h / ph)),
            }
        )
    return out


def _regions_for_match(
    doc: Any, page_text: str, match: LocatorMatch
) -> list[dict[str, Any]]:
    try:
        if match.page_index < 0 or match.page_index >= len(doc):
            return []
        needle = _anchor_tokens(page_text, match)
        if len(needle) < MIN_ANCHOR_WORDS:
            return []
        page = doc[match.page_index]
        page_words = page.get_text("words") or []
        if not page_words:
            return []
        indices = _locate(page_words, needle)
        if not indices:
            return []
        pw = float(page.rect.width)
        ph = float(page.rect.height)
        if pw <= 0 or ph <= 0:
            return []
        rects = _rects_from_words(page_words, indices, pw, ph)
        for r in rects:
            r["pageIndex"] = match.page_index
            r["pageNumber"] = match.page_number
        return rects
    except Exception:
        return []


def pdf_regions_for_chunks(
    pdf_path: Path, pages: list, chunks: list
) -> list[list[dict[str, Any]]]:
    """Region rectangles for each chunk (parallel to ``chunks``).

    ``pages`` are the parsed ``Page`` objects; ``chunks`` carry
    ``source_page_index`` / ``page_char_start`` / ``page_char_end``. Non-PDFs and
    any failure return empty lists, never an exception.
    """
    pdf_path = Path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        return [[] for _ in chunks]
    try:
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return [[] for _ in chunks]

    regions: list[list[dict[str, Any]]] = []
    try:
        for chunk in chunks:
            page_index = getattr(chunk, "source_page_index", None)
            start = getattr(chunk, "page_char_start", None)
            end = getattr(chunk, "page_char_end", None)
            if page_index is None or start is None or end is None:
                regions.append([])
                continue
            if page_index < 0 or page_index >= len(pages):
                regions.append([])
                continue
            match = LocatorMatch(
                page_index = int(page_index),
                page_number = getattr(chunk, "page_number", None),
                start = int(start),
                end = int(end),
            )
            regions.append(_regions_for_match(doc, pages[page_index].text, match))
        return regions
    finally:
        doc.close()
