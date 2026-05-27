# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .parsers import ParsedPage

# Match "Figure 1:", "Figure 1.2:", "Fig. 3.", "Table 4:" etc. at line-start,
# tolerating leading bold markers. Used to break chunks BEFORE such captions
# so the caption ends up at the start of its own chunk — dense embeddings
# pool over the whole chunk, so figure references buried at the end get
# diluted by surrounding body text.
_FIGURE_BOUNDARY_RE = re.compile(
    # Number forms covered: "1", "12", "1.2", "B.1" (appendix-style),
    # tolerating bold wrappers around either the label or the number.
    r"^\**(?:Figure|Fig\.|Table|Tab\.)\s+[A-Z]?\.?\d+(?:\.\d+)?\**[\.:]",
    re.MULTILINE | re.IGNORECASE,
)


def _split_at_figure_boundaries(text: str) -> list[str]:
    """Split markdown at the start of each figure / table caption.

    Each segment starts with either the original text head or a "Figure N:" /
    "Table N:" line, so the caption anchors the embedding of the chunk it
    lands in. Returns the original text as a single-element list when no
    captions are found.
    """
    matches = list(_FIGURE_BOUNDARY_RE.finditer(text))
    if not matches:
        return [text]
    segments: list[str] = []
    last = 0
    for m in matches:
        if m.start() > last:
            segments.append(text[last : m.start()])
        last = m.start()
    segments.append(text[last:])
    return [s for s in segments if s.strip()]


@dataclass(frozen = True)
class Chunk:
    text: str
    token_count: int
    page_number: int | None = None


TokenCounter = Callable[[str], int]


def _char_token_estimate(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _split_on(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    if len(parts) == 1:
        return parts
    glued: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            glued.append(part + separator)
        else:
            if part:
                glued.append(part)
    return [p for p in glued if p]


def _atomic_split(
    text: str,
    separators: tuple[str, ...],
    max_tokens: int,
    count: TokenCounter,
) -> list[str]:
    if count(text) <= max_tokens:
        return [text]
    for sep in separators:
        pieces = _split_on(text, sep)
        if len(pieces) <= 1:
            continue
        out: list[str] = []
        for piece in pieces:
            if count(piece) <= max_tokens:
                out.append(piece)
            else:
                tail = separators[separators.index(sep) + 1 :]
                out.extend(_atomic_split(piece, tail, max_tokens, count))
        return out
    approx_chars = max(1, max_tokens * 4)
    return [text[i : i + approx_chars] for i in range(0, len(text), approx_chars)]


def _merge(
    pieces: list[str],
    max_tokens: int,
    overlap_tokens: int,
    count: TokenCounter,
) -> list[str]:
    """Greedy-merge into <= max_tokens chunks with overlap."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0
    for piece in pieces:
        piece_tokens = count(piece)
        if buffer and buffer_tokens + piece_tokens > max_tokens:
            chunks.append("".join(buffer))
            if overlap_tokens > 0:
                overlap: list[str] = []
                running = 0
                for prev in reversed(buffer):
                    prev_tokens = count(prev)
                    if running + prev_tokens > overlap_tokens:
                        break
                    overlap.insert(0, prev)
                    running += prev_tokens
                buffer = list(overlap)
                buffer_tokens = running
            else:
                buffer = []
                buffer_tokens = 0
        buffer.append(piece)
        buffer_tokens += piece_tokens
    if buffer:
        chunks.append("".join(buffer))
    return [c.strip() for c in chunks if c.strip()]


# Markdown headings first so layout-aware parser output splits at sections.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
)


def chunk_pages(
    pages: list[ParsedPage],
    *,
    max_tokens: int,
    overlap_tokens: int,
    token_counter: TokenCounter | None = None,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Split pages independently so page_number stays attached to chunks."""
    count = token_counter or _char_token_estimate
    out: list[Chunk] = []
    for page in pages:
        for segment in _split_at_figure_boundaries(page.text):
            atomic = _atomic_split(segment, separators, max_tokens, count)
            merged = _merge(atomic, max_tokens, overlap_tokens, count)
            for piece in merged:
                out.append(
                    Chunk(
                        text = piece,
                        token_count = count(piece),
                        page_number = page.page_number,
                    )
                )
    return out


_PAGE_SEPARATOR = "\n\n"


def chunk_pages_with_spans(
    pages: list[ParsedPage],
    *,
    max_tokens: int,
    overlap_tokens: int,
    token_counter: TokenCounter | None = None,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> tuple[str, list[Chunk], list[tuple[int, int]]]:
    """Late-chunking variant: joins pages so the embedder sees the whole doc.

    Returns ``(full_doc, chunks, char_spans)``; ``char_spans[i]`` is the
    (start, end) char offset of ``chunks[i].text`` inside ``full_doc``.
    Page numbers are recovered by overlap with the original page ranges.
    """
    count = token_counter or _char_token_estimate

    parts: list[str] = []
    page_ranges: list[tuple[int, int, int | None]] = []
    cursor = 0
    for index, page in enumerate(pages):
        parts.append(page.text)
        start = cursor
        end = cursor + len(page.text)
        page_ranges.append((start, end, page.page_number))
        cursor = end
        if index < len(pages) - 1:
            cursor += len(_PAGE_SEPARATOR)
    full_doc = _PAGE_SEPARATOR.join(parts)

    atomic: list[str] = []
    for segment in _split_at_figure_boundaries(full_doc):
        atomic.extend(_atomic_split(segment, separators, max_tokens, count))
    merged = _merge(atomic, max_tokens, overlap_tokens, count)

    chunks: list[Chunk] = []
    char_spans: list[tuple[int, int]] = []
    search_cursor = 0
    for piece in merged:
        text = piece.strip()
        if not text:
            continue
        idx = full_doc.find(text, search_cursor)
        if idx < 0:
            # Overlap can push past a chunk's true start; restart from head.
            idx = full_doc.find(text)
        if idx < 0:
            continue
        end_idx = idx + len(text)
        page_number = _page_for_span(idx, end_idx, page_ranges)
        chunks.append(
            Chunk(
                text = text,
                token_count = count(text),
                page_number = page_number,
            )
        )
        char_spans.append((idx, end_idx))
        # Advance past start (not end) so overlapping next chunk is findable.
        search_cursor = idx + 1

    return full_doc, chunks, char_spans


def _page_for_span(
    start: int,
    end: int,
    page_ranges: list[tuple[int, int, int | None]],
) -> int | None:
    for ps, pe, pn in page_ranges:
        if start < pe and end > ps:
            return pn
    return None
