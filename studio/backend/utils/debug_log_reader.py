# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded reads of a log file for the Settings > Logs viewer.

The active session log is never rotated and only pruned at startup (run.py
retains the newest 20 files), so it can be many GB by the time someone opens
this. Initial reads seek from the end; polling processes bounded pages without
skipping redaction context.

read_tail and read_since return REDACTED lines. The raw reader is private so a
later caller cannot forget.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from copy import copy
import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.log_redaction import StreamingLogRedactor

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
MAX_CURSOR_STATES = 256
_OMITTED_CONTEXT_BYTES = 4096


@dataclass
class _OmittedRecord:
    start: int
    sensitive: bool = False
    continuation: Optional[str] = None
    private_key: bool = False
    quote: Optional[str] = None
    escaped: bool = False


@dataclass
class _CursorState:
    path: str
    redactor: StreamingLogRedactor
    partial_start: Optional[int]
    omitted: Optional[_OmittedRecord]


_CURSOR_STATES: OrderedDict[str, _CursorState] = OrderedDict()
_CURSOR_LOCK = threading.Lock()


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
    # Identity only: st_ctime_ns changes on append on Linux, which made every poll look like a rotation and resend the
    # whole tail; st_ino can be 0 on Windows, so name and device carry it there.
    return f"{name}|{stat.st_dev}|{stat.st_ino}"


def encode_cursor(
    key: str,
    offset: int,
    state_id: Optional[str] = None,
) -> str:
    payload = {"k": key, "o": int(offset)}
    if state_id is not None:
        payload["s"] = state_id
    raw = json.dumps(payload, separators = (",", ":")).encode("utf-8")
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


def _remember_cursor(
    path: Path,
    key: str,
    offset: int,
    redactor: StreamingLogRedactor,
    partial_start: Optional[int] = None,
    omitted: Optional[_OmittedRecord] = None,
) -> str:
    cursor = encode_cursor(key, offset, secrets.token_urlsafe(16))
    state = _CursorState(os.path.abspath(path), copy(redactor), partial_start, copy(omitted))
    with _CURSOR_LOCK:
        _CURSOR_STATES[cursor] = state
        while len(_CURSOR_STATES) > MAX_CURSOR_STATES:
            _CURSOR_STATES.popitem(last = False)
    return cursor


def _restore_cursor(path: Path, cursor: str) -> Optional[_CursorState]:
    with _CURSOR_LOCK:
        state = _CURSOR_STATES.get(cursor)
        if state is None or state.path != os.path.abspath(path):
            return None
        _CURSOR_STATES.move_to_end(cursor)
        return _CursorState(
            state.path, copy(state.redactor), state.partial_start, copy(state.omitted)
        )


def _scan_omitted_record(
    redactor: StreamingLogRedactor,
    state: _OmittedRecord,
    context: bytes,
    piece: bytes,
    terminated: bool,
) -> None:
    scan = (context + piece).decode("utf-8", errors = "replace")
    state.sensitive |= redactor.omitted_record_chunk_has_sensitive_context(scan)
    state.continuation = redactor.omitted_record_continuation_kind(
        scan, state.continuation, state.sensitive
    )
    state.private_key = redactor.omitted_record_private_key_state(scan, state.private_key)
    quote_scan = piece.decode("utf-8", errors = "replace") if state.quote else scan
    state.quote, state.escaped = redactor.omitted_record_quote_state(
        quote_scan, state.quote, state.escaped
    )
    if terminated:
        if state.quote is not None or state.continuation is not None:
            redactor.mark_omitted_sensitive_record(state.quote, state.continuation)
        if state.private_key:
            redactor.mark_omitted_private_key_block()


def _split_lines(data: bytes, *, drop_partial_head: bool) -> tuple[list[str], bool]:
    truncated_head = False
    if drop_partial_head:
        first = data.find(b"\n")
        remainder = b"" if first == -1 else data[first + 1 :]
        if not remainder:
            # The whole window sits inside ONE record (no line break, or only the terminator at the end), so dropping
            # the partial head left nothing: a record bigger than the window (native dump, \r-only progress run, giant
            # JSON line) rendered an EMPTY pane on a megabyte log while the cursor still advanced past it. Keep the
            # record's tail.
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
        lines.append(line)
    return lines, truncated_head


def _redact(lines: list[str], redactor: StreamingLogRedactor) -> list[str]:
    result: list[str] = []
    for line in lines:
        # Redact physical records before splitting them for display.
        line = redactor.redact_record(line)
        while len(line) > MAX_LINE_BYTES:
            result.append(line[:MAX_LINE_BYTES])
            line = line[MAX_LINE_BYTES:]
        result.append(line)
    return result


def read_tail(path: Path, max_lines: int = DEFAULT_TAIL_LINES) -> ReadResult:
    max_lines = max(1, min(int(max_lines), MAX_TAIL_LINES))
    stat = path.stat()
    size = stat.st_size
    key = _file_key(stat, path.name)
    redactor = StreamingLogRedactor()
    result = ReadResult(size_bytes = size)
    result.reset = True
    if size == 0:
        result.cursor = _remember_cursor(path, key, size, redactor)
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
    partial_start = None
    if data and not data.endswith(b"\n"):
        partial_start = size - len(data.rsplit(b"\n", 1)[-1])
        complete = _redact(lines[:-1], redactor)
        lines = complete + _redact(lines[-1:], copy(redactor))
    else:
        lines = _redact(lines, redactor)
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        result.truncated_head = True
    result.lines = lines[-MAX_LINES_PER_RESPONSE:]
    result.cursor = _remember_cursor(path, key, size, redactor, partial_start)
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

    state = _restore_cursor(path, cursor)
    if state is None:
        result = read_tail(path, max_lines)
        result.reset_reason = "cursor_stale"
        return result

    result = ReadResult(size_bytes = size)
    if offset == size:
        result.cursor = cursor
        return result
    if state.partial_start is not None:
        offset = state.partial_start
        result.reset = True
        result.reset_reason = "partial_record"

    start = offset
    read_start = start
    if state.omitted is not None:
        overlap = min(_OMITTED_CONTEXT_BYTES, MAX_APPEND_BYTES // 2)
        read_start = max(state.omitted.start, start - overlap)
    with open(path, "rb") as handle:
        handle.seek(read_start)
        raw = handle.read(min(MAX_APPEND_BYTES, size - read_start))
    context, data = raw[: start - read_start], raw[start - read_start :]
    result.more_pending = read_start + len(raw) < size

    if state.omitted is not None:
        newline = data.find(b"\n")
        consumed = len(data) if newline == -1 else newline + 1
        _scan_omitted_record(state.redactor, state.omitted, context, data[:consumed], newline != -1)
        start += consumed
        data = data[consumed:]
        if newline == -1:
            result.cursor = _remember_cursor(
                path, current_key, start, state.redactor, omitted = state.omitted
            )
            return result
        state.omitted = None

    if not data:
        result.cursor = _remember_cursor(path, current_key, start, state.redactor)
        return result

    # Stop at the last newline and leave the cursor before the partial line
    last_newline = data.rfind(b"\n")
    if last_newline == -1:
        if len(data) < MAX_APPEND_BYTES:
            result.cursor = _remember_cursor(path, current_key, start, state.redactor)
            return result
        omitted = _OmittedRecord(start)
        _scan_omitted_record(state.redactor, omitted, b"", data, False)
        result.lines = ["[oversized log record omitted]"]
        result.cursor = _remember_cursor(
            path, current_key, start + len(data), state.redactor, omitted = omitted
        )
        return result
    else:
        consumed = last_newline + 1
        body = data[:consumed]

    # Cap by BYTES before decoding so the cursor stops where the response stops: slicing decoded lines threw away the
    # oldest of a burst while advancing past them, reporting dropped_bytes = 0.
    # A model load logging more than MAX_LINES_PER_RESPONSE lines between polls lost the head of its own failure; the
    # remainder now arrives next poll.
    newline_count = body.count(b"\n")
    if newline_count > MAX_LINES_PER_RESPONSE:
        cut = -1
        for _ in range(MAX_LINES_PER_RESPONSE):
            cut = body.find(b"\n", cut + 1)
        consumed = cut + 1
        body = body[:consumed]
        result.more_pending = True

    lines, truncated = _split_lines(body, drop_partial_head = False)
    result.truncated_head = truncated
    result.lines = _redact(lines, state.redactor)
    result.cursor = _remember_cursor(path, current_key, start + consumed, state.redactor)
    return result
