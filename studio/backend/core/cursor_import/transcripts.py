# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a Cursor agent transcript into Studio chat messages.

A transcript is JSONL where each line is either a turn --
``{"role": "user"|"assistant", "message": {"content": [parts]}}`` -- or a session
event such as ``{"type": "turn_ended"}``. Turn parts are ``text`` or ``tool_use``
(name plus input); Cursor keeps no tool results and no per-line timestamps in
this file.

Two consequences shape the mapping. Tool calls are imported without a ``result``,
which the existing conversation import already does for the same reason (a
ShareGPT file has no results either), so the UI has a shape it can render. And
because the file carries no clock, message times are spread from the file's own
creation time, one millisecond per message, which is what the frontend importer
does with ``baseTs + idx``: the ordering is real even though the spacing is not.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Cursor wraps the real prompt in <user_query> and injects context blocks around
# it. Importing the injected blocks would train and display Cursor's harness
# rather than the conversation, so the query is preferred when present.
_USER_QUERY = re.compile(r"<user_query>(.*?)</user_query>", re.DOTALL)
_INJECTED_BLOCKS = (
    "image_files",
    "system_reminder",
    "attached_files",
    "additional_data",
    "environment_details",
    "timestamp",
)
# A thinking block Cursor withholds from the transcript. It marks where reasoning
# happened but carries none of it, so it is dropped rather than shown as content.
_REDACTED = re.compile(r"^[ \t]*\[REDACTED\][ \t]*$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")

TITLE_MAX_CHARS = 120
_DEFAULT_TITLE = "Cursor session"


@dataclass
class CursorTranscript:
    """One imported session: a thread and its messages, ready to store."""

    session_id: str
    path: Path
    title: str
    created_at_ms: int
    updated_at_ms: int
    messages: list[dict] = field(default_factory = list)
    tool_calls: int = 0
    skipped_records: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.messages


def _strip_injected(text: str) -> str:
    for tag in _INJECTED_BLOCKS:
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags = re.DOTALL)
        # Some blocks are emitted self-closing or unterminated at a turn's end.
        text = re.sub(rf"</?{tag}\s*/?>", "", text)
    return text


def clean_user_text(text: str) -> str:
    """The prompt the user actually typed, without Cursor's injected context."""
    queries = _USER_QUERY.findall(text)
    if queries:
        joined = "\n\n".join(query.strip() for query in queries if query.strip())
        if joined:
            return _BLANK_RUN.sub("\n\n", joined).strip()
    return _BLANK_RUN.sub("\n\n", _strip_injected(text)).strip()


def clean_assistant_text(text: str) -> str:
    """Assistant prose with withheld thinking markers removed."""
    return _BLANK_RUN.sub("\n\n", _REDACTED.sub("", text)).strip()


def _text_of(part: dict) -> str:
    value = part.get("text")
    return value if isinstance(value, str) else ""


def _message_id(session_id: str, index: int) -> str:
    """Stable per (session, position) so re-importing updates instead of piling up."""
    digest = hashlib.sha1(f"{session_id}:{index}".encode("utf-8")).hexdigest()
    return f"cursor-{digest[:16]}"


def _assistant_parts(content: list, session_id: str, index: int) -> tuple[list[dict], int]:
    parts: list[dict] = []
    tool_calls = 0
    for position, raw in enumerate(content):
        if not isinstance(raw, dict):
            continue
        kind = raw.get("type")
        if kind == "text":
            text = clean_assistant_text(_text_of(raw))
            if text:
                parts.append({"type": "text", "text": text})
        elif kind == "tool_use":
            name = raw.get("name")
            arguments = raw.get("input")
            parts.append(
                {
                    "type": "tool-call",
                    "toolCallId": f"{_message_id(session_id, index)}-{position}",
                    "toolName": str(name) if name else "unknown",
                    "args": arguments if isinstance(arguments, dict) else {},
                }
            )
            tool_calls += 1
    return parts, tool_calls


def _user_parts(content: Any) -> list[dict]:
    if isinstance(content, str):
        text = clean_user_text(content)
        return [{"type": "text", "text": text}] if text else []
    if not isinstance(content, list):
        return []
    parts: list[dict] = []
    for raw in content:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") != "text":
            continue
        text = clean_user_text(_text_of(raw))
        if text:
            parts.append({"type": "text", "text": text})
    return parts


def _title_from(text: str) -> str:
    """First line of the opening prompt, matching the frontend's fallback title."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return _DEFAULT_TITLE
    if len(first_line) <= TITLE_MAX_CHARS:
        return first_line
    return first_line[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def _file_times_ms(path: Path) -> tuple[int, int]:
    """``(created, modified)`` in epoch milliseconds, from the transcript file."""
    info = path.stat()
    created = getattr(info, "st_birthtime", None) or info.st_ctime
    modified = max(info.st_mtime, created)
    return int(created * 1000), int(modified * 1000)


def read_transcript(
    path: Path,
    thread_id: str,
    *,
    session_id: Optional[str] = None,
) -> CursorTranscript:
    """Parse one transcript file into a thread's worth of messages.

    A line that is not JSON, or not a turn, is counted and skipped: session
    events are expected, and a half-written last line is normal for a session
    that is still open.
    """
    resolved_session = session_id or path.stem
    created_ms, updated_ms = _file_times_ms(path)

    turns: list[tuple[str, list[dict]]] = []
    tool_calls = 0
    skipped = 0
    index = 0
    with path.open(encoding = "utf-8", errors = "replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            role = record.get("role")
            message = record.get("message")
            if role not in ("user", "assistant") or not isinstance(message, dict):
                skipped += 1
                continue

            content = message.get("content")
            if role == "user":
                parts = _user_parts(content)
            else:
                parts, calls = _assistant_parts(
                    content if isinstance(content, list) else [],
                    resolved_session,
                    index,
                )
                tool_calls += calls
            if not parts:
                # An empty turn carries nothing to show and would render as a
                # blank bubble; the frontend importer drops these too.
                skipped += 1
                continue
            turns.append((role, parts))
            index += 1

    messages: list[dict] = []
    parent_id: Optional[str] = None
    for position, (role, parts) in enumerate(turns):
        message_id = _message_id(resolved_session, position)
        messages.append(
            {
                "id": message_id,
                "threadId": thread_id,
                "parentId": parent_id,
                "role": role,
                "content": parts,
                "createdAt": created_ms + position,
                "metadata": {"importedFrom": "cursor", "cursorSessionId": resolved_session},
            }
        )
        parent_id = message_id

    first_user = next(
        (
            part["text"]
            for role, parts in turns
            if role == "user"
            for part in parts
            if part.get("type") == "text" and part.get("text")
        ),
        "",
    )
    return CursorTranscript(
        session_id = resolved_session,
        path = path,
        title = _title_from(first_user),
        created_at_ms = created_ms,
        updated_at_ms = max(updated_ms, created_ms + max(0, len(messages) - 1)),
        messages = messages,
        tool_calls = tool_calls,
        skipped_records = skipped,
    )
