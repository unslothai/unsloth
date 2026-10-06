# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Codex: ``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<id>.jsonl[.zst]``, grouped by ``session_meta.cwd``.

Lines are ``{timestamp, type, payload}`` (codex-rs ``RolloutLine``). Text comes from the
``user_message`` / ``agent_message`` events, which is what Codex's own thread history replays:
the ``response_item`` user messages also carry injected AGENTS.md / environment context.
"""

from __future__ import annotations

import json
import re
from itertools import islice
from pathlib import Path

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

_UNKNOWN_FOLDER = "unknown-folder"


def _meta(path: Path) -> dict:
    """The rollout's session_meta payload ({} when the file cannot say)."""
    try:
        first = next(islice(read_jsonl(path), 1), {})
    except (OSError, ImportError):
        return {}
    payload = first.get("payload") if first.get("type") == "session_meta" else None
    return payload if isinstance(payload, dict) else {}


def list_projects(home: Path) -> list[SourceProject]:
    root = home / "sessions"
    if not root.is_dir():
        return []
    by_cwd: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("rollout-*.jsonl*")):
        if not path.name.endswith((".jsonl", ".jsonl.zst")):
            continue
        meta = _meta(path)
        # Subagent threads repeat work their parent already shows.
        source = meta.get("source")
        if isinstance(source, dict) and "subagent" in source:
            continue
        # The raw cwd is the identity (hashed into the project id); "a-b" and "a/b" stay apart.
        by_cwd.setdefault(str(meta.get("cwd") or _UNKNOWN_FOLDER), []).append(path)
    return [
        SourceProject(cwd, display_name(re.sub(r"[^A-Za-z0-9]+", "-", cwd)), paths)
        for cwd, paths in by_cwd.items()
    ]


def _output_text(output) -> str:
    if isinstance(output, list):
        output = "\n\n".join(item.get("text") or "" for item in output if isinstance(item, dict))
    return clean_text(output) if isinstance(output, str) else ""


def _call(payload: dict, message_id: str) -> dict:
    raw = payload.get("arguments") if payload["type"] == "function_call" else payload.get("input")
    try:
        args = json.loads(raw) if payload["type"] == "function_call" else {"input": raw}
    except (TypeError, ValueError):
        args = {"input": raw}
    return tool_call(
        str(payload.get("call_id") or message_id),
        {"name": payload.get("name"), "input": args},
    )


def read_transcript(path: Path, thread_id: str, session_id: str) -> Transcript:
    file_created, file_updated = file_times_ms(path)
    messages: list[dict] = []
    open_calls: dict[str, dict] = {}
    for index, line in enumerate(read_jsonl(path)):
        payload = line.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = (line.get("type"), payload.get("type"))
        message_id = stable_id("codex", session_id, str(index), length = 16)
        if kind == ("event_msg", "user_message"):
            role, parts = (
                "user",
                [{"type": "text", "text": clean_text(str(payload.get("message") or ""))}],
            )
        elif kind == ("event_msg", "agent_message"):
            role, parts = (
                "assistant",
                [{"type": "text", "text": clean_text(str(payload.get("message") or ""))}],
            )
        elif kind in (("response_item", "function_call"), ("response_item", "custom_tool_call")):
            role, parts = "assistant", [_call(payload, message_id)]
            open_calls[parts[0]["toolCallId"]] = parts[0]
        elif kind in (
            ("response_item", "function_call_output"),
            ("response_item", "custom_tool_call_output"),
        ):
            call = open_calls.pop(str(payload.get("call_id")), None)
            result = _output_text(payload.get("output"))
            if call is not None and result:
                call["result"] = result
            continue
        else:
            continue
        if parts[0]["type"] == "text" and not parts[0]["text"]:
            continue
        timestamp = iso_ms(line.get("timestamp"))
        messages.append(
            {
                "id": message_id,
                "threadId": thread_id,
                "parentId": messages[-1]["id"] if messages else None,
                "role": role,
                "content": parts,
                "createdAt": timestamp if timestamp is not None else file_created + index,
                "metadata": {"importedFrom": "codex", "codexSessionId": session_id},
            }
        )
    return Transcript(
        session_id = session_id,
        thread_id = thread_id,
        title = title_from(first_user_text(messages), "Codex session"),
        created_at_ms = messages[0]["createdAt"] if messages else file_created,
        updated_at_ms = max(messages[-1]["createdAt"] if messages else 0, file_updated),
        messages = messages,
    )


SOURCE = Source(
    key = "codex",
    label = "Codex",
    home_env = "CODEX_HOME",
    default_home = ".codex",
    list_projects = list_projects,
    read_transcript = read_transcript,
)
