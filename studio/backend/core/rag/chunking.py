# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .parsers import ParsedPage


@dataclass(frozen = True)
class Chunk:
    text: str
    token_count: int
    page_number: int | None = None


TokenCounter = Callable[[str], int]


def _char_token_estimate(text: str) -> int:
    # Rough 4 chars / token heuristic — only used if no real tokenizer is provided.
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
                tail = separators[separators.index(sep) + 1:]
                out.extend(_atomic_split(piece, tail, max_tokens, count))
        return out
    # No separator made progress — hard-slice by characters.
    approx_chars = max(1, max_tokens * 4)
    return [text[i : i + approx_chars] for i in range(0, len(text), approx_chars)]


def _merge(
    pieces: list[str],
    max_tokens: int,
    overlap_tokens: int,
    count: TokenCounter,
) -> list[str]:
    """Greedy-merge atomic pieces into chunks <= max_tokens with overlap between adjacent chunks."""
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


DEFAULT_SEPARATORS: tuple[str, ...] = (
    # Markdown heading boundaries first — when the parser emits
    # layout-aware Markdown (PDF via pymupdf4llm, DOCX via mammoth,
    # HTML via markdownify) chunks split at section breaks rather
    # than mid-paragraph. Falls back to the original separators on
    # plain text input where headings are absent.
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
    """Split parsed pages into overlapping chunks.

    Each page is split independently so page_number stays meaningful for
    PDFs — cross-page chunks would lose source attribution.
    """
    count = token_counter or _char_token_estimate
    out: list[Chunk] = []
    for page in pages:
        atomic = _atomic_split(page.text, separators, max_tokens, count)
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
    """Late-chunking-friendly variant of :func:`chunk_pages`.

    Joins all pages into a single document so the embedder sees the
    whole text in one pass (that's the point of late chunking — chunk
    vectors that carry full-document context via the model's
    bidirectional attention).

    Returns ``(full_doc, chunks, char_spans)`` where
    ``char_spans[i] = (start, end)`` are byte-character offsets of
    ``chunks[i].text`` inside ``full_doc``. The embedder layer maps
    char spans → token spans via the tokenizer's offsets_mapping and
    mean-pools per chunk.

    Page-number metadata on each :class:`Chunk` is recovered from the
    chunk's char span — the first page whose range overlaps the chunk
    wins. PDFs keep useful citations even though chunking ignores page
    boundaries here.
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

    atomic = _atomic_split(full_doc, separators, max_tokens, count)
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
            # Overlap can push the search cursor past a chunk's true
            # start — restart from the document head as a fallback.
            idx = full_doc.find(text)
        if idx < 0:
            # Chunker output diverged from the source (rare — happens
            # if a separator-splice mangled the text). Skip the chunk
            # rather than corrupt the vector store with a wrong span.
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
        # Advance past the *start* of this chunk so an overlapping
        # next chunk can still be found.
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
