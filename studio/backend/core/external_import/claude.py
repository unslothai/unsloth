# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Claude Code: ``~/.claude/projects/<encoded-path>/<session>.jsonl``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.external_import import (
    Source,
    SourceProject,
    Transcript,
    clean_text,
    display_name,
    file_times_ms,
    first_user_text,
    iso_ms,
    read_jsonl,
    stable_id,
    title_from,
    tool_call,
)

# Not anchored: Claude Code pads these tags with whitespace.
_COMMAND = re.compile(r"<command-(?:name|message|args)>")
_LOCAL_STDOUT = re.compile(r"</?local-command-stdout>")


def list_projects(home: Path) -> list[SourceProject]:
    root = home / "projects"
    if not root.is_dir():
        return []
    projects = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        sessions = sorted(p for p in entry.iterdir() if p.is_file() and p.suffix == ".jsonl")
        if sessions:
            projects.append(SourceProject(entry.name, display_name(entry.name), sessions))
    return projects


def _user_text(text: str) -> str:
    # Slash-command echoes are the harness talking; captured command output is cut, it can follow a prompt.
    if _COMMAND.search(text):
        return ""
    return clean_text(_LOCAL_STDOUT.split(text)[0])


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        content = "\n\n".join(
            block.get("text") or ""
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return clean_text(content) if isinstance(content, str) else ""


def _parts(record: dict, message_id: str) -> tuple[list[dict], dict[str, str]]:
    """Text and tool-call parts, plus ``{tool_use_id: result}`` this record answers. Thinking is dropped."""
    content = record.get("message", {}).get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [], {}
    user = record.get("type") == "user"
    parts, results = [], {}
    for position, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = (_user_text if user else clean_text)(block.get("text") or "")
            if text:
                parts.append({"type": "text", "text": text})
        elif kind == "tool_use" and not user:
            parts.append(tool_call(str(block.get("id") or f"{message_id}-{position}"), block))
        elif kind == "tool_result" and user and block.get("tool_use_id"):
            # empty output still marks a finished call; Studio replays "" differently from none
            results[str(block["tool_use_id"])] = _result_text(block.get("content"))
    return parts, results


def read_transcript(path: Path, thread_id: str, session_id: str) -> Transcript:
    file_created, file_updated = file_times_ms(path)
    # file order preserves the append-only import ledger
    # attachments and system lines link ancestry; compact_boundary inherits prior if logicalParentUuid is absent or forward.
    records = []
    links: dict[str, Any] = {}
    previous = None
    for line in read_jsonl(path):
        if line.get("uuid"):
            compacted = line.get("subtype") == "compact_boundary" and not line.get("parentUuid")
            parent = line.get("parentUuid")
            if compacted:
                logical = line.get("logicalParentUuid")
                parent = logical if logical in links else previous
            links[str(line["uuid"])] = parent
            previous = str(line["uuid"])
        if (
            line.get("type") in ("user", "assistant")
            and not line.get("isSidechain")
            and not line.get("isMeta")
        ):
            records.append(line)
    imported: dict[str, str] = {}
    # parallel tool results hang off their own call; continue from the reply's last block, not a fork.
    tail_of: dict[str, str] = {}
    active_reply = None
    active_blocks: list[str] = []
    open_calls: dict[str, dict] = {}
    messages: list[dict] = []
    for index, record in enumerate(records):
        uuid = str(record.get("uuid") or "")
        reply = record.get("message", {}).get("id") if record["type"] == "assistant" else None
        if record["type"] != "assistant" or reply != active_reply:
            if active_blocks:
                tail_of.update((block, active_blocks[-1]) for block in active_blocks)
            active_reply = reply
            active_blocks = []
        message_id = stable_id("claude", session_id, uuid or f"index:{index}", length = 16)
        parts, results = _parts(record, message_id)
        for call_id, result in results.items():
            if call_id in open_calls:
                open_calls.pop(call_id)["result"] = result
        if not parts:
            continue
        open_calls.update((p["toolCallId"], p) for p in parts if p["type"] == "tool-call")
        # use the nearest imported ancestor so rewinds stay on their original branch.
        parent, seen = record.get("parentUuid"), set()
        while parent and parent not in imported and parent not in seen:
            seen.add(parent)
            parent = links.get(parent)
        parent = tail_of.get(parent, parent)
        timestamp = iso_ms(record.get("timestamp"))
        messages.append(
            {
                "id": message_id,
                "threadId": thread_id,
                "parentId": imported.get(parent) if parent else None,
                "role": record["type"],
                "content": parts,
                "createdAt": timestamp if timestamp is not None else file_created + index,
                "metadata": {"importedFrom": "claude", "claudeSessionId": session_id},
            }
        )
        if uuid:
            imported[uuid] = message_id
            if reply:
                active_blocks.append(uuid)
    if active_blocks:
        tail_of.update((block, active_blocks[-1]) for block in active_blocks)
    return Transcript(
        session_id = session_id,
        thread_id = thread_id,
        title = title_from(first_user_text(messages), "Claude session"),
        created_at_ms = messages[0]["createdAt"] if messages else file_created,
        updated_at_ms = max(messages[-1]["createdAt"] if messages else 0, file_updated),
        messages = messages,
    )


SOURCE = Source(
    key = "claude",
    label = "Claude",
    home_env = "UNSLOTH_CLAUDE_HOME",
    default_home = ".claude",
    list_projects = list_projects,
    read_transcript = read_transcript,
)
