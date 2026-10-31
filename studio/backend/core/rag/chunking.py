# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Page-aware recursive-separator chunking with token overlap; chunks record
page_char_start/page_char_end and source_page_index for locator highlights."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    whole_document_token_count: int | None = None


@dataclass(frozen = True, slots = True)
class _Piece:
    text: str
    token_count: int


def _split(
    text: str,
    seps: tuple[str, ...],
    max_tokens: int,
    count: TokenCounter,
    token_count: int | None = None,
) -> list[_Piece]:
    """Recursively split into pieces each <= max_tokens (best effort). Pieces
    rejoin to ``text`` exactly, so offsets are a running length."""
    if token_count is None:
        token_count = count(text)
    if token_count <= max_tokens:
        return [_Piece(text, token_count)]
    for i, sep in enumerate(seps):
        parts = list(text) if sep == "" else text.split(sep)
        if len(parts) <= 1:
            continue
        if sep:
            parts = [p + sep for p in parts[:-1]] + parts[-1:]
        out: list[_Piece] = []
        for p in parts:
            if not p:
                continue
            tokens = count(p)
            out.extend(
                [_Piece(p, tokens)]
                if tokens <= max_tokens
                else _split(p, seps[i + 1 :], max_tokens, count, tokens)
            )
        return out
    n = max(1, max_tokens * 4)
    return [_Piece(part, count(part)) for j in range(0, len(text), n) if (part := text[j : j + n])]


def _count_all(count: TokenCounter, texts: list[str]) -> list[int] | None:
    """Counts for ``texts`` from the counter's ``many`` batch form, or None when it has none."""
    many = getattr(count, "many", None)
    if many is None:
        return None
    counts = list(many(texts))
    if len(counts) != len(texts):
        raise ValueError(f"token counter returned {len(counts)} counts for {len(texts)} texts")
    return counts


def _split_all(
    texts: list[str],
    token_counts: list[int],
    seps: tuple[str, ...],
    max_tokens: int,
    count: TokenCounter,
) -> list[list[_Piece]]:
    """``_split`` for many texts at once, level by level: every part cut at the same depth is counted
    in one batch. Counts are pure, so the pieces equal ``_split``'s for each text."""
    out: list[list[_Piece] | None] = [None] * len(texts)
    # Parts cut with seps[i], keyed by i: (text index, parts).
    levels: dict[int, list[tuple[int, list[str]]]] = {}
    sliced: list[tuple[int, list[str]]] = []
    for idx, (text, token_count) in enumerate(zip(texts, token_counts)):
        if token_count <= max_tokens:
            out[idx] = [_Piece(text, token_count)]
            continue
        for i, sep in enumerate(seps):
            parts = list(text) if sep == "" else text.split(sep)
            if len(parts) <= 1:
                continue
            if sep:
                parts = [p + sep for p in parts[:-1]] + parts[-1:]
            levels.setdefault(i, []).append((idx, [p for p in parts if p]))
            break
        else:
            n = max(1, max_tokens * 4)
            sliced.append(
                (idx, [part for j in range(0, len(text), n) if (part := text[j : j + n])])
            )
    if sliced:
        counts = _count_all(count, [part for _, parts in sliced for part in parts])
        k = 0
        for idx, parts in sliced:
            out[idx] = [_Piece(part, counts[k + j]) for j, part in enumerate(parts)]
            k += len(parts)
    for i, members in levels.items():
        flat = [part for _, parts in members for part in parts]
        children = _split_all(flat, _count_all(count, flat), seps[i + 1 :], max_tokens, count)
        k = 0
        for idx, parts in members:
            pieces: list[_Piece] = []
            for child in children[k : k + len(parts)]:
                pieces.extend(child)
            out[idx] = pieces
            k += len(parts)
    return out


def _merge(
    pieces: list[_Piece], starts: list[int], max_tokens: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Greedy-merge pieces into <= max_tokens chunks with token overlap.
    ``starts[i]`` is ``pieces[i]``'s page char offset; returns
    ``(chunk_text, char_start, char_end)`` spans."""
    chunks: list[tuple[str, int, int]] = []
    buf: list[_Piece] = []
    buf_starts: list[int] = []
    buf_tok = 0

    def _flush() -> None:
        raw = "".join(piece.text for piece in buf)
        stripped = raw.strip()
        if not stripped:
            return
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        start = buf_starts[0] + lead
        end = buf_starts[0] + len(raw) - trail
        chunks.append((stripped, start, end))

    for piece, start in zip(pieces, starts):
        # Reuse counts after the GGUF tokenizer's cache evicts earlier pieces.
        pt = piece.token_count
        if buf and buf_tok + pt > max_tokens:
            _flush()
            # Bound carry so carry + piece fits max_tokens, or the embedder overflows.
            carry_budget = min(overlap, max(0, max_tokens - pt))
            carry, carry_starts, run = [], [], 0
            for prev, prev_start in zip(reversed(buf), reversed(buf_starts)):
                if run + prev.token_count > carry_budget:
                    break
                carry.insert(0, prev)
                carry_starts.insert(0, prev_start)
                run += prev.token_count
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
    """Split each page into overlapping chunks, tracking per-page char offsets. A counter with a
    ``many`` batch form is asked once per stage for the whole document; the chunks are the same."""
    page_counts = _count_all(count, [page.text for page in pages])
    batched = page_counts is not None
    split = (
        _split_all([page.text for page in pages], page_counts, SEPARATORS, max_tokens, count)
        if batched
        else None
    )
    out: list[Chunk] = []
    # Batched: (out index, text) whose overlap-free count is still owed.
    owed: list[tuple[int, str]] = []
    for page_index, page in enumerate(pages):
        pieces = split[page_index] if batched else _split(page.text, SEPARATORS, max_tokens, count)
        # _split preserves offsets, so a running cursor gives exact ones.
        starts: list[int] = []
        cursor = 0
        for piece in pieces:
            starts.append(cursor)
            cursor += len(piece.text)
        previous_end = 0
        for text, char_start, char_end in _merge(pieces, starts, max_tokens, overlap):
            token_count = -1 if batched else count(text)
            overlap_chars = min(len(text), max(0, previous_end - char_start))
            unique_text = text[overlap_chars:].lstrip("\r\n")
            if unique_text == text:
                whole = token_count
            elif batched:
                whole = -1
                owed.append((len(out), unique_text))
            else:
                whole = count(unique_text)
            out.append(
                Chunk(
                    text = text,
                    token_count = token_count,
                    page_number = page.page_number,
                    source_page_index = page_index,
                    chunk_index = len(out),
                    page_char_start = char_start,
                    page_char_end = char_end,
                    whole_document_token_count = whole,
                )
            )
            previous_end = max(previous_end, char_end)
    if not batched or not out:
        return out
    token_counts = _count_all(count, [chunk.text for chunk in out])
    whole_counts = dict(zip((k for k, _ in owed), _count_all(count, [t for _, t in owed]) or ()))
    return [
        replace(
            chunk,
            token_count = token_counts[k],
            whole_document_token_count = whole_counts.get(k, token_counts[k]),
        )
        for k, chunk in enumerate(out)
    ]
