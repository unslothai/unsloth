# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PNG text chunks (tEXt, zTXt, iTXt): walk, read, strip and write them without decoding pixels."""

from __future__ import annotations

import struct
import zlib
from typing import Iterator, Optional

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TEXT_CHUNKS = (b"tEXt", b"zTXt", b"iTXt")


def iter_png_chunks(data: bytes) -> Iterator[tuple[bytes, bytes, int, int]]:
    """(kind, body, start, end) per chunk, start/end spanning the whole chunk."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file.")
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(data):
            # Truncated file: stop at what is readable rather than fail the whole card.
            return
        yield kind, data[start:end], offset, end + 4
        offset = end + 4
        if kind == b"IEND":
            return


def _inflate(data: bytes, budget: list[int]) -> bytes:
    # Bounded across the whole file: a few KB of zlib can expand to gigabytes, and a PNG can repeat chunks.
    inflater = zlib.decompressobj()
    out = inflater.decompress(data, max(budget[0], 1))
    budget[0] -= len(out)
    if inflater.unconsumed_tail or not inflater.eof or budget[0] < 0:
        raise ValueError("compressed text chunk too large or truncated")
    return out


def chunk_key(body: bytes) -> str:
    return body.partition(b"\x00")[0].decode("latin-1").lower()


def read_text_chunk(
    kind: bytes, body: bytes, keys: tuple[str, ...], budget: list[int]
) -> Optional[tuple[str, bytes]]:
    """A wanted text chunk's lower-cased key and raw text bytes; other chunks are never inflated."""
    key = chunk_key(body)
    if key not in keys:
        return None
    rest = body.partition(b"\x00")[2]
    try:
        if kind == b"tEXt":
            return key, rest
        if kind == b"zTXt":
            return key, _inflate(rest[1:], budget)
        compressed, rest = rest[0], rest[2:]
        _language, _, rest = rest.partition(b"\x00")
        _translated, _, text = rest.partition(b"\x00")
        return key, _inflate(text, budget) if compressed else text
    except (zlib.error, IndexError, ValueError):
        return None


def strip_png_text(data: bytes) -> bytes:
    """The PNG without its text chunks, so a stored avatar does not carry a stale card."""
    out = bytearray(PNG_SIGNATURE)
    for kind, _body, start, end in iter_png_chunks(data):
        if kind in TEXT_CHUNKS:
            continue
        out.extend(data[start:end])
    return bytes(out)


def png_chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)
