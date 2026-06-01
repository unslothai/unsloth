# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recursive-separator chunking with token overlap. Page-aware.

Same shape as a LangChain recursive splitter: split on the coarsest separator
that fits, recurse to finer ones, then greedy-merge pieces into <= max_tokens
chunks carrying ``overlap`` tokens between neighbours.

Beyond the base splitter we also record, for every chunk, the character
``[page_char_start, page_char_end)`` span it occupies WITHIN its source page's
text and the 0-based ``source_page_index`` into the pages list. A later locator
pass uses those offsets to map a chunk back onto the rendered PDF page for
citation highlighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .parsers import Page

TokenCounter = Callable[[str], int]
SEPARATORS = ("\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", "")


@dataclass(frozen = True)
class Chunk:
    text: str
    token_count: int
    page_number: int | None
    source_page_index: int
    chunk_index: int
    page_char_start: int
    page_char_end: int


def _split(
    text: str, seps: tuple[str, ...], max_tokens: int, count: TokenCounter
) -> list[str]:
    """Recursively split text into pieces each <= max_tokens (best effort).

    Concatenating the returned pieces in order reproduces ``text`` exactly, so a
    caller can recover each piece's char offset by running length.
    """
    if count(text) <= max_tokens:
        return [text]
    for i, sep in enumerate(seps):
        parts = list(text) if sep == "" else text.split(sep)
        if len(parts) <= 1:
            continue
        if sep:  # re-attach the separator we split on
            parts = [p + sep for p in parts[:-1]] + parts[-1:]
        out: list[str] = []
        for p in parts:
            out.extend(
                [p]
                if count(p) <= max_tokens
                else _split(p, seps[i + 1 :], max_tokens, count)
            )
        return [p for p in out if p]
    n = max(1, max_tokens * 4)
    return [text[j : j + n] for j in range(0, len(text), n)]


def _merge(
    pieces: list[str],
    starts: list[int],
    max_tokens: int,
    overlap: int,
    count: TokenCounter,
) -> list[tuple[str, int, int]]:
    """Greedy-merge pieces into <= max_tokens chunks with token overlap.

    ``starts[i]`` is the char offset of ``pieces[i]`` in the page text. Returns a
    list of ``(chunk_text, char_start, char_end)`` where the span covers the raw
    (pre-strip) extent of the merged pieces in the source page.
    """
    chunks: list[tuple[str, int, int]] = []
    buf: list[str] = []
    buf_starts: list[int] = []
    buf_tok = 0

    def _flush() -> None:
        raw = "".join(buf)
        stripped = raw.strip()
        if not stripped:
            return
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        start = buf_starts[0] + lead
        end = buf_starts[0] + len(raw) - trail
        chunks.append((stripped, start, end))

    for piece, start in zip(pieces, starts):
        pt = count(piece)
        if buf and buf_tok + pt > max_tokens:
            _flush()
            carry, carry_starts, run = [], [], 0
            for prev, prev_start in zip(reversed(buf), reversed(buf_starts)):
                if run + count(prev) > overlap:
                    break
                carry.insert(0, prev)
                carry_starts.insert(0, prev_start)
                run += count(prev)
            buf, buf_starts, buf_tok = carry, carry_starts, run
        buf.append(piece)
        buf_starts.append(start)
        buf_tok += pt
    if buf:
        _flush()
    return chunks


def chunk_pages(
    pages: list[Page], *, max_tokens: int, overlap: int, count: TokenCounter
) -> list[Chunk]:
    """Split each page into overlapping chunks, tracking per-page char offsets."""
    out: list[Chunk] = []
    for page_index, page in enumerate(pages):
        pieces = _split(page.text, SEPARATORS, max_tokens, count)
        # Char offset of each piece in the page text. _split is offset-preserving
        # (pieces concatenate back to page.text), so a running cursor is exact.
        starts: list[int] = []
        cursor = 0
        for piece in pieces:
            starts.append(cursor)
            cursor += len(piece)
        for text, char_start, char_end in _merge(
            pieces, starts, max_tokens, overlap, count
        ):
            out.append(
                Chunk(
                    text = text,
                    token_count = count(text),
                    page_number = page.page_number,
                    source_page_index = page_index,
                    chunk_index = len(out),
                    page_char_start = char_start,
                    page_char_end = char_end,
                )
            )
    return out
