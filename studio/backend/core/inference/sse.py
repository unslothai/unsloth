# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inbound SSE framing: transport pieces may split a line or a UTF-8 character anywhere."""

from __future__ import annotations

import codecs
from typing import Any, AsyncIterator

from core.inference.http_stream import SSE_LINE_END


class SSELineDecoder:
    """Complete lines from split text/byte pieces; a trailing "\\r" waits one piece for its "\\n".

    The pending line is kept as fragments so a long unterminated line is not a quadratic copy.
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
        held_cr = bool(self._parts) and self._parts[-1].endswith("\r")
        if not held_cr and "\n" not in text and "\r" not in text:
            if text:
                self._parts.append(text)
            return []
        pending = "".join(self._parts) + text
        self._parts.clear()
        cut = len(pending) - pending.endswith("\r")
        *lines, rest = SSE_LINE_END.split(pending[:cut])
        rest += pending[cut:]
        if rest:
            self._parts.append(rest)
        return lines

    def flush(self) -> list[str]:
        """The unterminated last line at end-of-stream; a held "\\r" counts as a line ending."""
        lines = self.feed(self._decoder.decode(b"", final = True))
        if self._parts and (tail := "".join(self._parts)):
            lines.append(tail[:-1] if tail.endswith("\r") else tail)
            self._parts.clear()
        return lines


class SSEDecoder:
    """Each complete event's joined `data:` payload; other fields are ignored (chat wire is data-only)."""

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
