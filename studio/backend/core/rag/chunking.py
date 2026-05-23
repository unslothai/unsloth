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


def chunk_pages(
    pages: list[ParsedPage],
    *,
    max_tokens: int,
    overlap_tokens: int,
    token_counter: TokenCounter | None = None,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
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
