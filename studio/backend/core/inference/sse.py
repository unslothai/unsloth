# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared inbound server-sent-event framing.

SSE transport pieces are arbitrary byte ranges: a multi-byte character or a `data:` line can
straddle two chunks, and lines end in \\n, \\r\\n, or \\r. Reframing used to be re-implemented per
consumer and each copy lost data differently; line reassembly and event boundaries live here once.
"""

from __future__ import annotations

import codecs
from typing import Any, AsyncIterator


class SSELineDecoder:
    """Reassemble complete lines from arbitrarily split text/byte pieces.

    Carries a partial line and any split multi-byte character across feed() calls; a trailing
    "\\r" is held one piece so "\\r\\n" never counts as two line endings. The pending line is
    kept as fragments so a long unterminated stream does not become a quadratic copy.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors = "replace")
        self._parts: list[str] = []
        self._first = True

    def feed(self, piece: Any) -> list[str]:
        if isinstance(piece, (bytes, bytearray, memoryview)):
            text = self._decoder.decode(bytes(piece))
        else:
            # A str piece can never complete bytes the decoder still holds; drain it first.
            text = self._decoder.decode(b"", final = True)
            text += piece if isinstance(piece, str) else str(piece)
        if text and self._first:
            self._first = False
            if text.startswith("\ufeff"):
                text = text[1:]
        lines: list[str] = []
        if text and self._parts and self._parts[-1].endswith("\r"):
            lines.append("".join(self._parts)[:-1])
            self._parts.clear()
            if text[0] == "\n":
                text = text[1:]
        start = 0
        index = 0
        end = len(text)
        while index < end:
            char = text[index]
            if char == "\n":
                lines.append("".join(self._parts) + text[start:index])
                self._parts.clear()
                index += 1
                start = index
            elif char == "\r":
                if index + 1 == end:
                    break
                lines.append("".join(self._parts) + text[start:index])
                self._parts.clear()
                index += 2 if text[index + 1] == "\n" else 1
                start = index
            else:
                index += 1
        if start < end:
            self._parts.append(text[start:])
        return lines

    def flush(self) -> list[str]:
        """The last line at end-of-stream, or [] when the stream ended on a line boundary.

        Declares the stream ended: a "\\r" still held is treated as a line ending. Feeding
        more pieces afterwards works but cannot rejoin that held "\\r" with its "\\n".
        """
        lines = self.feed(self._decoder.decode(b"", final = True))
        if self._parts and (tail := "".join(self._parts)):
            lines.append(tail[:-1] if tail.endswith("\r") else tail)
            self._parts.clear()
        return lines


class SSEDecoder:
    """feed() pieces, get each complete event's data payload.

    Multi-line `data:` fields join with "\\n" per the SSE spec, one optional space after the colon
    is stripped, and comment/`event:`/`id:` lines are ignored: Studio's chat wire is data-only.
    """

    def __init__(self) -> None:
        self._lines = SSELineDecoder()
        self._data: list[str] = []

    def _line(self, line: str) -> str | None:
        if line == "":
            if not self._data:
                return None
            data = "\n".join(self._data)
            self._data = []
            # The spec aborts dispatch on an empty data buffer: "data:" alone emits nothing.
            return data if data else None
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            self._data.append(value)
        elif line == "data":
            self._data.append("")
        return None

    def feed(self, piece: Any) -> list[str]:
        values: list[str] = []
        for line in self._lines.feed(piece):
            data = self._line(line)
            if data is not None:
                values.append(data)
        return values

    def flush(self) -> list[str]:
        values = [data for line in self._lines.flush() if (data := self._line(line)) is not None]
        if self._data:
            # A truncated stream still yields its last event; the caller's parse decides.
            if tail := "\n".join(self._data):
                values.append(tail)
            self._data = []
        return values


async def iter_sse_lines(chunks: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Complete lines from an async str/bytes chunk stream. Does not close ``chunks``."""
    decoder = SSELineDecoder()
    async for chunk in chunks:
        for line in decoder.feed(chunk):
            yield line
    for line in decoder.flush():
        yield line
