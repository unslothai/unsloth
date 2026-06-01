# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDF region locators: map a chunk back to highlight rectangles on its page.

Computed once at ingest from the live parse (no backfill, no persisted page
text). For each chunk we take a short anchor phrase from its page span, confirm
that phrase occurs exactly once on the rendered PDF page, and ask PyMuPDF for the
rectangle(s) covering it -- normalized to 0..1 of the page so the frontend can
draw highlights at any zoom. Conservative by design: no PyMuPDF, no anchor, or a
non-unique match yields an empty region list rather than a guessed highlight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen = True)
class LocatorMatch:
    page_index: int
    page_number: int | None
    start: int
    end: int


def _normalize_with_map(text: str) -> str:
    """Casefold + collapse runs of whitespace to single spaces (trimmed)."""
    chars: list[str] = []
    last_space = False
    for ch in text:
        if ch.isspace():
            if chars and not last_space:
                chars.append(" ")
            last_space = True
            continue
        chars.append(ch.casefold())
        last_space = False
    return "".join(chars).strip()


def _normalized_occurrences(haystack: str, needle: str) -> int:
    nh = _normalize_with_map(haystack)
    nn = _normalize_with_map(needle)
    if not nh or not nn:
        return 0
    count, cursor = 0, 0
    while True:
        idx = nh.find(nn, cursor)
        if idx < 0:
            return count
        count += 1
        cursor = idx + 1


def _region_anchor(page_text: str, match: LocatorMatch) -> str | None:
    """A 3-16 word phrase from the chunk's page span, used to locate it."""
    segment = page_text[match.start : match.end]
    words = [w.strip(" \t\r\n*#`[]()") for w in segment.split()]
    words = [w for w in words if len(w) >= 2]
    if len(words) < 3:
        return None
    anchor = " ".join(words[: min(16, len(words))])
    return anchor if len(anchor) >= 12 else None


def _regions_for_match(
    doc: Any, match: LocatorMatch, anchor: str
) -> list[dict[str, Any]]:
    try:
        if match.page_index >= len(doc):
            return []
        page = doc[match.page_index]
        raw_text = page.get_text("text") or ""
        # Only highlight when the anchor is unambiguous on the rendered page.
        if _normalized_occurrences(raw_text, anchor) != 1:
            return []
        rects = page.search_for(anchor) or []
        pw = float(page.rect.width)
        ph = float(page.rect.height)
        if pw <= 0 or ph <= 0:
            return []
        out: list[dict[str, Any]] = []
        for rect in rects:
            w = max(0.0, float(rect.x1 - rect.x0))
            h = max(0.0, float(rect.y1 - rect.y0))
            if w <= 0 or h <= 0:
                continue
            out.append(
                {
                    "pageIndex": match.page_index,
                    "pageNumber": match.page_number,
                    "x": max(0.0, min(1.0, float(rect.x0) / pw)),
                    "y": max(0.0, min(1.0, float(rect.y0) / ph)),
                    "width": max(0.0, min(1.0, w / pw)),
                    "height": max(0.0, min(1.0, h / ph)),
                }
            )
        return out
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
            anchor = _region_anchor(pages[page_index].text, match)
            if not anchor:
                regions.append([])
                continue
            regions.append(_regions_for_match(doc, match, anchor))
        return regions
    finally:
        doc.close()
