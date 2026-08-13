# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a Claude Code session transcript into Studio chat messages.

A transcript is JSONL where each line is one event. The conversation lives in
``type: "user"`` and ``type: "assistant"`` records, each carrying ``message``,
a stable ``uuid``, and an ISO ``timestamp``; everything else -- ``system``
notices, ``file-history-snapshot``, ``progress``, ``summary`` -- is bookkeeping
and is skipped. Records walk a ``parentUuid`` chain, and sidechain (subagent)
records share the file, so the main conversation is reconstructed by following
the chain rather than by trusting file order.

The mapping differs from Cursor's in two ways that come from Claude Code
keeping more. Every record is timestamped, so messages keep their real times
instead of being spread one millisecond apart from the file's creation. And the
model's tool calls are answered in the file: a ``tool_use`` block in an
assistant message is matched by a ``tool_result`` block in a later user record,
so imported calls carry their ``result`` where Cursor's could not. The
``thinking`` blocks a response may open with are the model's reasoning,
recorded but not prose the user reads, so they are dropped rather than shown as
content.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

TITLE_MAX_CHARS = 120
_DEFAULT_TITLE = "Claude session"

# A slash-command invocation or a captured command's echo is the harness
# talking, not the user, so it does not become message content. Claude Code pads
# the tags with whitespace and newlines, so the match is not anchored to the
# very start.
_COMMAND = re.compile(r"<command-(?:name|message|args)>")
_LOCAL_COMMAND_STDOUT = re.compile(r"<local-command-stdout>")
_LOCAL_COMMAND_STDOUT_CLOSE = re.compile(r"</local-command-stdout>")


@dataclass
class ClaudeTranscript:
    """One imported session: a thread and its messages, ready to store."""

    session_id: str
    thread_id: str
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


def _timestamp_ms(record: dict) -> Optional[int]:
    """A record's time in epoch milliseconds, from its ISO 8601 string."""
    raw = record.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo = timezone.utc)
    return int(parsed.timestamp() * 1000)


def _file_times_ms(path: Path) -> tuple[int, int]:
    """``(created, modified)`` in epoch milliseconds, for records that lack one."""
    info = path.stat()
    created = getattr(info, "st_birthtime", None) or info.st_ctime
    modified = max(info.st_mtime, created)
    return int(created * 1000), int(modified * 1000)


def _message_id(session_id: str, uuid: str, fallback_index: int) -> str:
    """Stable per record when Claude gives it a uuid, else per position."""
    key = uuid if uuid else f"index:{fallback_index}"
    digest = hashlib.sha1(f"{session_id}:{key}".encode("utf-8")).hexdigest()
    return f"claude-{digest[:16]}"


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _user_text(text: str) -> str:
    """The part of a user turn that is a real prompt, or "" when there is none.

    Slash-command invocations and their captured output are the harness talking,
    not the user, so they come out. Command output is cut rather than merely
    detected, since it can be appended to a real prompt that precedes it.
    """
    if _COMMAND.search(text):
        return ""
    text = _LOCAL_COMMAND_STDOUT_CLOSE.split(text)[0]
    text = _LOCAL_COMMAND_STDOUT.split(text)[0]
    return _clean_text(text)


def _result_text(content: Any) -> str:
    """A tool result's body as plain text, from whichever shape it took."""
    if isinstance(content, str):
        return _clean_text(content)
    if isinstance(content, list):
        chunks = [
            _clean_text(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return _clean_text("\n\n".join(chunk for chunk in chunks if chunk))
    return ""


def _user_parts(record: dict) -> tuple[list[dict], dict[str, str]]:
    """The text a user turn contributes, plus the tool results it answers.

    A user record is either a real prompt (string content) or a batch of
    ``tool_result`` blocks feeding the model (list content). The results are
    matched back to their calls by ``tool_use_id`` and returned on the side, so
    the assistant message that made the call can carry its outcome.
    """
    content = record.get("message", {}).get("content")
    parts: list[dict] = []
    tool_results: dict[str, str] = {}
    if isinstance(content, str):
        text = _user_text(content)
        if text:
            parts.append({"type": "text", "text": text})
        return parts, tool_results
    if not isinstance(content, list):
        return parts, tool_results
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = _user_text(block.get("text") or "")
            if text:
                parts.append({"type": "text", "text": text})
        elif kind == "tool_result":
            tool_use_id = block.get("tool_use_id")
            result = _result_text(block.get("content"))
            if tool_use_id and result:
                tool_results[str(tool_use_id)] = result
    return parts, tool_results


def _assistant_parts(record: dict, session_id: str, index: int) -> tuple[list[dict], int]:
    """Text and tool calls from one assistant record.

    ``thinking`` blocks are dropped: they are the model's reasoning, recorded
    but not prose for the conversation. ``tool_use`` blocks become ``tool-call``
    parts keyed by the tool's own id, so a later ``tool_result`` can be attached
    to its call.
    """
    parts: list[dict] = []
    tool_calls = 0
    content = record.get("message", {}).get("content")
    if not isinstance(content, list):
        return parts, tool_calls
    for position, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = _clean_text(block.get("text") or "")
            if text:
                parts.append({"type": "text", "text": text})
        elif kind == "tool_use":
            tool_id = block.get("id") or f"{_message_id(session_id, '', index)}-{position}"
            arguments = block.get("input")
            parts.append(
                {
                    "type": "tool-call",
                    "toolCallId": str(tool_id),
                    "toolName": str(block.get("name") or "unknown"),
                    "args": arguments if isinstance(arguments, dict) else {},
                }
            )
            tool_calls += 1
        # "thinking" and anything else carry no displayable prose.
    return parts, tool_calls


def _title_from(text: str) -> str:
    """First line of the opening prompt, matching the frontend's fallback title."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return _DEFAULT_TITLE
    if len(first_line) <= TITLE_MAX_CHARS:
        return first_line
    return first_line[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def _conversation_records(records: list[dict]) -> list[dict]:
    """The main conversation's user/assistant records, in order.

    Records are stored newest-appended, but sidechains and the occasional
    compaction boundary break a pure read-down-the-file order. When the records
    carry a ``parentUuid`` chain, following it from the main root yields the
    conversation the user actually had and leaves sidechain branches out; when
    they do not, file order is the honest fallback.
    """
    main = [
        record
        for record in records
        # isMeta records are injected context (the caveat preamble of a resumed
        # session, a command's bookkeeping), not turns the user typed.
        if not record.get("isSidechain")
        and not record.get("isMeta")
        and record.get("type") in ("user", "assistant")
    ]
    if not any(record.get("parentUuid") for record in main):
        return main
    children: dict[Optional[str], list[dict]] = {}
    for record in main:
        children.setdefault(record.get("parentUuid"), []).append(record)
    ordered: list[dict] = []

    def walk(uuid: Optional[str]) -> None:
        for child in children.get(uuid, []):
            ordered.append(child)
            walk(child.get("uuid"))

    walk(None)
    # Anything the walk missed -- a break in the chain, an orphan whose parent
    # was compacted away -- is kept in its file place rather than dropped.
    seen = {id(record) for record in ordered}
    ordered.extend(record for record in main if id(record) not in seen)
    return ordered


def read_transcript(
    path: Path,
    thread_id: str,
    *,
    session_id: Optional[str] = None,
) -> ClaudeTranscript:
    """Parse one session file into a thread's worth of messages.

    A line that is not JSON, or not part of the conversation, is counted and
    skipped: bookkeeping records are expected, and a half-written last line is
    normal for a session that is still open.
    """
    resolved_session = session_id or path.stem
    fallback_created_ms, fallback_updated_ms = _file_times_ms(path)

    records: list[dict] = []
    skipped = 0
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
            if isinstance(record, dict):
                records.append(record)
            else:
                skipped += 1

    messages: list[dict] = []
    tool_calls = 0
    # Calls wait for their result, which arrives in a later user record, so the
    # parts that hold them stay reachable by tool id until the result shows.
    open_calls: dict[str, dict] = {}
    previous_id: Optional[str] = None
    for index, record in enumerate(_conversation_records(records)):
        role = record.get("type")
        message_id = _message_id(resolved_session, str(record.get("uuid") or ""), index)
        if role == "user":
            parts, results = _user_parts(record)
            # Results answer calls from earlier assistant records, which are
            # already built; attach them there, then decide if this turn has
            # any prose of its own to add.
            for tool_use_id, result in results.items():
                call = open_calls.pop(tool_use_id, None)
                if call is not None:
                    call["result"] = result
            if not parts:
                continue
        else:
            parts, calls = _assistant_parts(record, resolved_session, index)
            tool_calls += calls
            for part in parts:
                if part.get("type") == "tool-call":
                    open_calls[part["toolCallId"]] = part
            if not parts:
                skipped += 1
                continue
        timestamp_ms = _timestamp_ms(record)
        messages.append(
            {
                "id": message_id,
                "threadId": thread_id,
                "parentId": previous_id,
                "role": role,
                "content": parts,
                "createdAt": timestamp_ms
                if timestamp_ms is not None
                else fallback_created_ms + index,
                "metadata": {
                    "importedFrom": "claude",
                    "claudeSessionId": resolved_session,
                },
            }
        )
        previous_id = message_id

    first_user = next(
        (
            part["text"]
            for message in messages
            if message["role"] == "user"
            for part in message["content"]
            if part.get("type") == "text" and part.get("text")
        ),
        "",
    )
    created = messages[0]["createdAt"] if messages else fallback_created_ms
    updated = messages[-1]["createdAt"] if messages else fallback_updated_ms
    return ClaudeTranscript(
        session_id = resolved_session,
        thread_id = thread_id,
        path = path,
        title = _title_from(first_user),
        created_at_ms = created,
        updated_at_ms = max(updated, fallback_updated_ms),
        messages = messages,
        tool_calls = tool_calls,
        skipped_records = skipped,
    )
