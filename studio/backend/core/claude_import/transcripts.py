# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Convert a Claude Code session transcript (JSONL) into Studio chat messages."""

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

# Not anchored: Claude Code pads these tags with whitespace.
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
    info = path.stat()
    created = getattr(info, "st_birthtime", None) or info.st_ctime
    modified = max(info.st_mtime, created)
    return int(created * 1000), int(modified * 1000)


def _message_id(session_id: str, uuid: str, fallback_index: int) -> str:
    key = uuid if uuid else f"index:{fallback_index}"
    digest = hashlib.sha1(f"{session_id}:{key}".encode("utf-8")).hexdigest()
    return f"claude-{digest[:16]}"


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _user_text(text: str) -> str:
    # Command output is cut, not just detected: it can follow a real prompt.
    if _COMMAND.search(text):
        return ""
    text = _LOCAL_COMMAND_STDOUT_CLOSE.split(text)[0]
    text = _LOCAL_COMMAND_STDOUT.split(text)[0]
    return _clean_text(text)


def _result_text(content: Any) -> str:
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
    """Text parts plus ``{tool_use_id: result}`` for the calls this record answers."""
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
    return parts, tool_calls


def _title_from(text: str) -> str:
    """Matches the frontend's fallback title."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return _DEFAULT_TITLE
    if len(first_line) <= TITLE_MAX_CHARS:
        return first_line
    return first_line[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def _conversation_records(records: list[dict]) -> list[dict]:
    """File order, not a tree walk: the import ledger relies on append-only order; ``parentUuid`` rebuilds branches."""
    return [
        record
        for record in records
        if not record.get("isSidechain")
        and not record.get("isMeta")
        and record.get("type") in ("user", "assistant")
    ]


def _imported_parent_id(
    record: dict, imported_ids: dict[str, str], by_uuid: dict[str, dict]
) -> Optional[str]:
    """Nearest ancestor that became a message (skipped records are still named as parents)."""
    parent = record.get("parentUuid")
    seen: set[str] = set()
    while parent and parent not in seen:
        seen.add(parent)
        message_id = imported_ids.get(parent)
        if message_id is not None:
            return message_id
        ancestor = by_uuid.get(parent)
        parent = ancestor.get("parentUuid") if ancestor else None
    return None


def read_transcript(
    path: Path,
    thread_id: str,
    *,
    session_id: Optional[str] = None,
) -> ClaudeTranscript:
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
    open_calls: dict[str, dict] = {}
    conversation = _conversation_records(records)
    by_uuid = {str(record["uuid"]): record for record in conversation if record.get("uuid")}
    imported_ids: dict[str, str] = {}
    for index, record in enumerate(conversation):
        role = record.get("type")
        message_id = _message_id(resolved_session, str(record.get("uuid") or ""), index)
        if role == "user":
            parts, results = _user_parts(record)
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
                "parentId": _imported_parent_id(record, imported_ids, by_uuid),
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
        if record.get("uuid"):
            imported_ids[str(record["uuid"])] = message_id

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
