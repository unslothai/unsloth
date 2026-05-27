# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backfill and PDF-region helpers for durable RAG chunk locators."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loggers import get_logger
from storage.studio_db import get_connection

from . import vector_store
from .parsers import ParsedPage, parse
from .vector_store import kb_scope, thread_scope

logger = get_logger(__name__)


@dataclass(frozen = True)
class LocatorMatch:
    page_index: int
    page_number: int | None
    start: int
    end: int
    line_start: int
    line_end: int


@dataclass(frozen = True)
class BackfillResult:
    document_id: str
    total_chunks: int
    matched: int
    already_located: int
    ambiguous: int
    missing: int
    skipped: int
    regions_matched: int
    pages_refreshed: int


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.count("\n", 0, start) + 1
    line_end = text.count("\n", 0, max(start, end - 1)) + 1
    return line_start, line_end


def _find_exact(page_text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    out: list[tuple[int, int]] = []
    cursor = 0
    while True:
        idx = page_text.find(needle, cursor)
        if idx < 0:
            break
        out.append((idx, idx + len(needle)))
        cursor = idx + 1
    return out


def _normalize_with_map(text: str) -> tuple[str, list[int], list[int]]:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    last_space = False
    for idx, ch in enumerate(text):
        if ch.isspace():
            if chars and not last_space:
                chars.append(" ")
                starts.append(idx)
                ends.append(idx + 1)
            elif chars and last_space:
                ends[-1] = idx + 1
            last_space = True
            continue
        chars.append(ch.casefold())
        starts.append(idx)
        ends.append(idx + 1)
        last_space = False

    first = 0
    while first < len(chars) and chars[first] == " ":
        first += 1
    last = len(chars)
    while last > first and chars[last - 1] == " ":
        last -= 1
    return "".join(chars[first:last]), starts[first:last], ends[first:last]


def _find_normalized(page_text: str, needle: str) -> list[tuple[int, int]]:
    norm_page, starts, ends = _normalize_with_map(page_text)
    norm_needle, _needle_starts, _needle_ends = _normalize_with_map(needle)
    if not norm_page or not norm_needle:
        return []
    out: list[tuple[int, int]] = []
    cursor = 0
    while True:
        idx = norm_page.find(norm_needle, cursor)
        if idx < 0:
            break
        end_idx = idx + len(norm_needle) - 1
        if 0 <= idx < len(starts) and 0 <= end_idx < len(ends):
            out.append((starts[idx], ends[end_idx]))
        cursor = idx + 1
    return out


def _locate_unique(text: str, pages: list[ParsedPage]) -> tuple[LocatorMatch | None, str]:
    text = (text or "").strip()
    if not text:
        return None, "missing"

    matches: list[LocatorMatch] = []
    for page_index, page in enumerate(pages):
        for start, end in _find_exact(page.text, text):
            line_start, line_end = _line_bounds(page.text, start, end)
            matches.append(
                LocatorMatch(
                    page_index = page_index,
                    page_number = page.page_number,
                    start = start,
                    end = end,
                    line_start = line_start,
                    line_end = line_end,
                )
            )
    if len(matches) == 1:
        return matches[0], "matched"
    if len(matches) > 1:
        return None, "ambiguous"

    for page_index, page in enumerate(pages):
        for start, end in _find_normalized(page.text, text):
            line_start, line_end = _line_bounds(page.text, start, end)
            matches.append(
                LocatorMatch(
                    page_index = page_index,
                    page_number = page.page_number,
                    start = start,
                    end = end,
                    line_start = line_start,
                    line_end = line_end,
                )
            )
    if len(matches) == 1:
        return matches[0], "matched"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "missing"


def _replace_document_pages(document_id: str, pages: list[ParsedPage]) -> None:
    now = int(time.time())
    rows = [
        (
            document_id,
            index,
            page.page_number,
            page.text,
            len(page.text),
            len(page.text.splitlines()),
            now,
        )
        for index, page in enumerate(pages)
    ]
    with get_connection() as conn:
        conn.execute("DELETE FROM rag_document_pages WHERE document_id = ?", (document_id,))
        if rows:
            conn.executemany(
                """
                INSERT INTO rag_document_pages
                (document_id, page_index, page_number, text, char_count,
                 line_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()


def _region_anchor(page_text: str, match: LocatorMatch) -> str | None:
    segment = page_text[match.start : match.end]
    words = [w.strip(" \t\r\n*#`[]()") for w in segment.split()]
    words = [w for w in words if len(w) >= 2]
    if len(words) < 3:
        return None
    anchor = " ".join(words[: min(16, len(words))])
    return anchor if len(anchor) >= 12 else None


def _normalized_occurrences(haystack: str, needle: str) -> int:
    norm_haystack, _starts, _ends = _normalize_with_map(haystack)
    norm_needle, _needle_starts, _needle_ends = _normalize_with_map(needle)
    if not norm_haystack or not norm_needle:
        return 0
    count = 0
    cursor = 0
    while True:
        idx = norm_haystack.find(norm_needle, cursor)
        if idx < 0:
            return count
        count += 1
        cursor = idx + 1


def pdf_regions_for_match(
    pdf_path: Path,
    pages: list[ParsedPage],
    match: LocatorMatch,
) -> list[dict[str, Any]]:
    """Return normalized PDF rectangles for a unique chunk match.

    Regions are intentionally conservative: no PyMuPDF, no page, no
    unique anchor, or no positive-area rectangles all produce an empty
    list rather than guessed highlights.
    """
    if pdf_path.suffix.lower() != ".pdf":
        return []
    if match.page_index < 0 or match.page_index >= len(pages):
        return []
    anchor = _region_anchor(pages[match.page_index].text, match)
    if not anchor:
        return []

    try:
        import pymupdf
    except Exception:
        return []

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return []

    try:
        return _pdf_regions_for_match_doc(doc, pages, match, anchor)
    finally:
        doc.close()


def _pdf_regions_for_match_doc(
    doc: Any,
    pages: list[ParsedPage],
    match: LocatorMatch,
    anchor: str,
) -> list[dict[str, Any]]:
    try:
        if match.page_index >= len(doc):
            return []
        page = doc[match.page_index]
        raw_text = page.get_text("text") or ""
        if _normalized_occurrences(raw_text, anchor) != 1:
            return []
        rects = page.search_for(anchor) or []
        page_rect = page.rect
        page_width = float(page_rect.width)
        page_height = float(page_rect.height)
        if page_width <= 0 or page_height <= 0:
            return []

        out: list[dict[str, Any]] = []
        for rect in rects:
            width = max(0.0, float(rect.x1 - rect.x0))
            height = max(0.0, float(rect.y1 - rect.y0))
            if width <= 0 or height <= 0:
                continue
            out.append(
                {
                    "pageIndex": match.page_index,
                    "pageNumber": match.page_number,
                    "x": max(0.0, min(1.0, float(rect.x0) / page_width)),
                    "y": max(0.0, min(1.0, float(rect.y0) / page_height)),
                    "width": max(0.0, min(1.0, width / page_width)),
                    "height": max(0.0, min(1.0, height / page_height)),
                    "confidence": "exact",
                    "source": "pymupdf-search",
                }
            )
        return out
    except Exception:
        return []


def pdf_regions_for_chunks(
    pdf_path: Path,
    pages: list[ParsedPage],
    chunks: list[Any],
) -> list[list[dict[str, Any]]]:
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
            line_start, line_end = _line_bounds(pages[page_index].text, start, end)
            match = LocatorMatch(
                page_index = int(page_index),
                page_number = getattr(chunk, "page_number", None),
                start = int(start),
                end = int(end),
                line_start = line_start,
                line_end = line_end,
            )
            anchor = _region_anchor(pages[match.page_index].text, match)
            if not anchor:
                regions.append([])
                continue
            regions.append(_pdf_regions_for_match_doc(doc, pages, match, anchor))
        return regions
    finally:
        doc.close()


def _scope_for_document(kb_id: str | None, thread_id: str | None) -> str | None:
    if kb_id:
        return kb_scope(kb_id)
    if thread_id:
        return thread_scope(thread_id)
    return None


def _update_vector_payloads(scope: str | None, updates: dict[str, dict[str, Any]]) -> None:
    if not scope or not updates:
        return
    try:
        vector_store.update_chunk_payload_fields(scope, updates)
    except Exception as exc:
        logger.warning(
            "RAG locator backfill: vector payload update failed",
            error = str(exc),
        )


def backfill_document_locators(document_id: str, stored_path: Path) -> BackfillResult:
    parsed = parse(stored_path, want_images = False)
    pages = parsed.pages
    _replace_document_pages(document_id, pages)

    with get_connection() as conn:
        doc_row = conn.execute(
            "SELECT kb_id, thread_id FROM rag_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if doc_row is None:
            return BackfillResult(document_id, 0, 0, 0, 0, 0, 0, 0, len(pages))

        rows = conn.execute(
            """
            SELECT id, text, kind, page_number, source_page_index,
                   page_char_start, page_char_end, line_start, line_end,
                   pdf_regions_json
            FROM rag_chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
            """,
            (document_id,),
        ).fetchall()

    scope = _scope_for_document(doc_row["kb_id"], doc_row["thread_id"])
    total = len(rows)
    matched = 0
    already_located = 0
    ambiguous = 0
    missing = 0
    skipped = 0
    regions_matched = 0
    sql_updates: list[tuple[Any, ...]] = []
    vector_updates: dict[str, dict[str, Any]] = {}

    for row in rows:
        kind = row["kind"] or "text"
        text = row["text"] or ""
        if kind not in ("text", "caption") or not text.strip():
            skipped += 1
            continue

        existing_complete = (
            row["source_page_index"] is not None
            and row["page_char_start"] is not None
            and row["page_char_end"] is not None
            and row["line_start"] is not None
            and row["line_end"] is not None
        )

        match: LocatorMatch | None
        status: str
        if existing_complete:
            already_located += 1
            page_index = int(row["source_page_index"])
            if 0 <= page_index < len(pages):
                match = LocatorMatch(
                    page_index = page_index,
                    page_number = row["page_number"],
                    start = int(row["page_char_start"]),
                    end = int(row["page_char_end"]),
                    line_start = int(row["line_start"]),
                    line_end = int(row["line_end"]),
                )
            else:
                match = None
            status = "already_located"
        else:
            match, status = _locate_unique(text, pages)
            if status == "matched" and match is not None:
                matched += 1
            elif status == "ambiguous":
                ambiguous += 1
                continue
            else:
                missing += 1
                continue

        if match is None:
            continue

        regions = pdf_regions_for_match(stored_path, pages, match)
        regions_json = json.dumps(regions, separators = (",", ":")) if regions else None
        if regions:
            regions_matched += 1

        if status == "matched" or (regions and not row["pdf_regions_json"]):
            sql_updates.append(
                (
                    match.page_number,
                    match.page_index,
                    match.start,
                    match.end,
                    match.line_start,
                    match.line_end,
                    regions_json,
                    row["id"],
                )
            )
            vector_updates[row["id"]] = {
                "page_number": match.page_number,
                "source_page_index": match.page_index,
                "page_char_start": match.start,
                "page_char_end": match.end,
                "line_start": match.line_start,
                "line_end": match.line_end,
                "pdf_regions": regions,
            }

    if sql_updates:
        with get_connection() as conn:
            conn.executemany(
                """
                UPDATE rag_chunks
                SET page_number = COALESCE(page_number, ?),
                    source_page_index = ?,
                    page_char_start = ?,
                    page_char_end = ?,
                    line_start = ?,
                    line_end = ?,
                    pdf_regions_json = COALESCE(?, pdf_regions_json)
                WHERE id = ?
                """,
                sql_updates,
            )
            conn.commit()
        _update_vector_payloads(scope, vector_updates)

    return BackfillResult(
        document_id = document_id,
        total_chunks = total,
        matched = matched,
        already_located = already_located,
        ambiguous = ambiguous,
        missing = missing,
        skipped = skipped,
        regions_matched = regions_matched,
        pages_refreshed = len(pages),
    )
