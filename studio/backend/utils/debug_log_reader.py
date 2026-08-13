# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded reads of a log file for the Settings > Debugging viewer.

The active session log is never rotated and only pruned at startup (run.py
retains the newest 20 files), so it can be many GB by the time someone opens
this. Everything here seeks from the end and reads a bounded window, so cost
does not scale with file size.

read_tail and read_since return REDACTED lines. The raw reader is private so a
later caller cannot forget.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.log_redaction import redact_log_text

BLOCK_BYTES = 65_536
DEFAULT_TAIL_LINES = 1_000
MAX_TAIL_LINES = 2_000  # == MAX_LINES_PER_RESPONSE: a larger ?lines= was silently capped
# /api is not gzipped (GZipMiddleware is scoped to the assets sub-app), so this
# is what actually goes on the wire on the first paint.
MAX_TAIL_BYTES = 1_048_576
MAX_APPEND_BYTES = 524_288
MAX_LINE_BYTES = 32_768
MAX_LINES_PER_RESPONSE = 2_000

_CURSOR_PREFIX = "c1."


@dataclass
class ReadResult:
    lines: list[str] = field(default_factory = list)
    cursor: Optional[str] = None
    reset: bool = False
    reset_reason: Optional[str] = None
    dropped_bytes: int = 0
    truncated_head: bool = False
    more_pending: bool = False
    size_bytes: int = 0


def _file_key(stat: os.stat_result, name: str) -> str:
    # Identity only: nothing in here may change when the file is appended to.
    # st_ctime_ns does (on Linux it is the metadata change time, so it moves on
    # every write), which made every poll look like a rotation and resend the
    # whole tail. st_ino can be 0 on some Windows filesystems, so name and
    # device carry the identity there; a truncation is caught separately by the
    # offset > size check.
    return f"{name}|{stat.st_dev}|{stat.st_ino}"


def encode_cursor(key: str, offset: int) -> str:
    raw = json.dumps({"k": key, "o": int(offset)}, separators = (",", ":")).encode("utf-8")
    return _CURSOR_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: Optional[str]) -> Optional[tuple[str, int]]:
    """None for anything unusable: a foreign cursor is answered with a fresh
    tail, never an error, so a poll loop cannot flash failures."""
    if not cursor or not isinstance(cursor, str) or not cursor.startswith(_CURSOR_PREFIX):
        return None
    body = cursor[len(_CURSOR_PREFIX) :]
    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        key = payload["k"]
        offset = int(payload["o"])
    except Exception:
        return None
    if not isinstance(key, str) or offset < 0:
        return None
    return key, offset


def _split_lines(data: bytes, *, drop_partial_head: bool) -> tuple[list[str], bool]:
    truncated_head = False
    if drop_partial_head:
        first = data.find(b"\n")
        remainder = b"" if first == -1 else data[first + 1 :]
        if not remainder:
            # The whole bounded window sits inside ONE record: either it has no
            # line break at all, or its only one is the terminator at the very
            # end. Dropping the partial head then left nothing, so a single
            # record bigger than the window (a native dump, a \r-only progress
            # run, a giant JSON line) rendered an EMPTY pane on a log that was
            # megabytes long, and the cursor still advanced past it. Keep the
            # record's tail instead; that is the end everyone is reading for.
            body = data if first == -1 else data[:first]
            remainder = body[-MAX_LINE_BYTES:]
        data = remainder
        truncated_head = True
    text = data.decode("utf-8", errors = "replace")
    raw = text.split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    lines: list[str] = []
    for line in raw:
        line = line.rstrip("\r")
        # A single enormous line (a native dump, a giant JSON record) is split
        # rather than dropped, so the viewer never silently loses content.
        while len(line) > MAX_LINE_BYTES:
            lines.append(line[:MAX_LINE_BYTES])
            line = line[MAX_LINE_BYTES:]
        lines.append(line)
    return lines, truncated_head


def _redact(lines: list[str]) -> list[str]:
    return [redact_log_text(line) for line in lines]


def read_tail(path: Path, max_lines: int = DEFAULT_TAIL_LINES) -> ReadResult:
    max_lines = max(1, min(int(max_lines), MAX_TAIL_LINES))
    stat = path.stat()
    size = stat.st_size
    result = ReadResult(size_bytes = size)
    result.cursor = encode_cursor(_file_key(stat, path.name), size)
    result.reset = True
    if size == 0:
        return result

    chunks: list[bytes] = []
    pos = size
    newlines = 0
    scanned = 0
    with open(path, "rb") as handle:
        while pos > 0 and newlines <= max_lines and scanned < MAX_TAIL_BYTES:
            step = min(BLOCK_BYTES, pos, MAX_TAIL_BYTES - scanned)
            pos -= step
            handle.seek(pos)
            block = handle.read(step)
            if not block:
                break
            chunks.insert(0, block)
            newlines += block.count(b"\n")
            scanned += len(block)

    data = b"".join(chunks)
    lines, truncated = _split_lines(data, drop_partial_head = pos > 0)
    result.truncated_head = truncated
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        result.truncated_head = True
    result.lines = _redact(lines[-MAX_LINES_PER_RESPONSE:])
    return result


def read_since(
    path: Path,
    cursor: Optional[str],
    max_lines: int = DEFAULT_TAIL_LINES,
) -> ReadResult:
    """Appended lines only, or a fresh tail when the cursor cannot apply."""
    decoded = decode_cursor(cursor)
    if decoded is None:
        result = read_tail(path, max_lines)
        result.reset_reason = "initial" if not cursor else "cursor_stale"
        return result

    key, offset = decoded
    stat = path.stat()
    current_key = _file_key(stat, path.name)
    size = stat.st_size

    if current_key != key:
        result = read_tail(path, max_lines)
        result.reset_reason = "rotated"
        return result
    if offset > size:
        # Reopened in "w" mode, or truncated underneath us.
        result = read_tail(path, max_lines)
        result.reset_reason = "truncated"
        return result

    result = ReadResult(size_bytes = size)
    if offset == size:
        result.cursor = encode_cursor(current_key, offset)
        return result

    start = offset
    pending = size - offset
    if pending > MAX_APPEND_BYTES:
        start = size - MAX_APPEND_BYTES
        result.dropped_bytes = start - offset
    with open(path, "rb") as handle:
        handle.seek(start)
        data = handle.read(size - start)

    # Stop at the last newline and leave the cursor before the partial line, so
    # a half-written record is never emitted twice. If an unterminated remainder
    # grows past a line's worth, flush it, otherwise a writer that never emits a
    # newline would stall the viewer for good.
    last_newline = data.rfind(b"\n")
    if last_newline == -1:
        if len(data) < MAX_LINE_BYTES:
            result.cursor = encode_cursor(current_key, start)
            return result
        consumed = len(data)
        body = data
    else:
        consumed = last_newline + 1
        body = data[:consumed]

    # Cap by BYTES, before decoding, so the cursor can stop where the response
    # stops. Slicing the decoded lines instead threw away the oldest of a burst
    # while still advancing the cursor past them, so a model load that logged
    # more than MAX_LINES_PER_RESPONSE lines between two polls lost the head of
    # its own failure, and the response said dropped_bytes = 0. The remainder is
    # not lost now, it arrives on the next poll.
    newline_count = body.count(b"\n")
    if newline_count > MAX_LINES_PER_RESPONSE:
        cut = -1
        for _ in range(MAX_LINES_PER_RESPONSE):
            cut = body.find(b"\n", cut + 1)
        consumed = cut + 1
        body = body[:consumed]
        result.more_pending = True

    lines, truncated = _split_lines(body, drop_partial_head = result.dropped_bytes > 0)
    result.truncated_head = truncated
    result.lines = _redact(lines)
    result.cursor = encode_cursor(current_key, start + consumed)
    return result
