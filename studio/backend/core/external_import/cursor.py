# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cursor transcripts lack tool results and timestamps; times are file creation plus 1 ms per message."""

from __future__ import annotations

import base64
import re
from pathlib import Path, PureWindowsPath

from core.external_import import (
    Source,
    SourceProject,
    Transcript,
    clean_text,
    display_name,
    file_times_ms,
    first_user_text,
    read_jsonl,
    stable_id,
    title_from,
    tool_call,
)
from utils.device_sync_settings import sniff_image_type

NO_FOLDER_SLUG = "empty-window"

_USER_QUERY = re.compile(r"<user_query>(.*?)</user_query>", re.DOTALL)
_INJECTED_TAGS = (
    "image_files|system_reminder|attached_files|additional_data|environment_details|timestamp"
)
_INJECTED = re.compile(rf"<({_INJECTED_TAGS})>.*?</\1>|</?(?:{_INJECTED_TAGS})\s*/?>", re.DOTALL)
_REDACTED = re.compile(r"^[ \t]*\[REDACTED\][ \t]*$", re.MULTILINE)
_IMAGE_FILES = re.compile(r"<image_files>(.*?)</image_files>", re.DOTALL)
_LISTED_FILE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$", re.MULTILINE)
_PLACEHOLDER = re.compile(r"^[ \t]*\[Image\][ \t]*$|</?user_query>", re.MULTILINE)


def _sessions(state_dir: Path) -> list[Path]:
    root = state_dir / "agent-transcripts"
    if not root.is_dir():
        return []
    sessions = []
    for entry in sorted(root.iterdir()):
        nested = entry / f"{entry.name}.jsonl"
        if entry.is_file() and entry.suffix == ".jsonl":
            sessions.append(entry)  # older flat layout
        elif nested.is_file():
            # Subagent transcripts beside it are skipped: they repeat the parent's history.
            sessions.append(nested)
    return sessions


def list_projects(home: Path) -> list[SourceProject]:
    root = home / "projects"
    if not root.is_dir():
        return []
    entries = [e for e in sorted(root.iterdir()) if e.is_dir() and not e.name.startswith(".")]
    # A session started before a folder was opened is filed under both that folder and the
    # no-folder window; the folder keeps it, so the no-folder window goes last.
    entries.sort(key = lambda e: e.name == NO_FOLDER_SLUG)
    claimed: set[str] = set()
    projects = []
    for entry in entries:
        sessions = [p for p in _sessions(entry) if p.stem not in claimed]
        if sessions:
            claimed.update(p.stem for p in sessions)
            name = "No folder open" if entry.name == NO_FOLDER_SLUG else display_name(entry.name)
            projects.append(SourceProject(entry.name, name, sessions))
    return projects


def _user_text(text: str, pictured: bool) -> str:
    queries = [q.strip() for q in _USER_QUERY.findall(text) if q.strip()]
    if queries:
        return clean_text("\n\n".join(queries))
    text = _INJECTED.sub("", text)
    # Only beside a kept image: an emptied turn would shift every later message id.
    return clean_text(_PLACEHOLDER.sub("", text) if pictured else text)


def _images(text: str, assets: Path) -> list[dict]:
    parts = []
    for block in _IMAGE_FILES.findall(text):
        for listed in _LISTED_FILE.findall(block):
            try:
                data = (assets / PureWindowsPath(listed).name).read_bytes()
            except OSError:
                continue
            if mime := sniff_image_type(data):
                image = f"data:{mime};base64,{base64.b64encode(data).decode()}"
                parts.append({"type": "image", "image": image})
    return parts


def _parts(role: str, content, message_id: str, assets: Path) -> list[dict]:
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    parts = []
    for position, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        text = block.get("text") if isinstance(block.get("text"), str) else ""
        if block.get("type") == "text":
            if role == "user":
                images = _images(text, assets)
                parts.extend(images)
                text = _user_text(text, bool(images))
            else:
                text = clean_text(_REDACTED.sub("", text))
            if text:
                parts.append({"type": "text", "text": text})
        elif block.get("type") == "tool_use" and role == "assistant":
            parts.append(tool_call(f"{message_id}-{position}", block))
    return parts


def read_transcript(path: Path, thread_id: str, session_id: str) -> Transcript:
    created, updated = file_times_ms(path)
    transcripts = path.parent if path.parent.name == "agent-transcripts" else path.parent.parent
    messages: list[dict] = []
    for record in read_jsonl(path):
        role, message = record.get("role"), record.get("message")
        if role not in ("user", "assistant") or not isinstance(message, dict):
            continue
        message_id = stable_id("cursor", session_id, str(len(messages)), length = 16)
        parts = _parts(role, message.get("content"), message_id, transcripts.parent / "assets")
        if not parts:
            continue
        messages.append(
            {
                "id": message_id,
                "threadId": thread_id,
                "parentId": messages[-1]["id"] if messages else None,
                "role": role,
                "content": parts,
                "createdAt": created + len(messages),
                "metadata": {"importedFrom": "cursor", "cursorSessionId": session_id},
            }
        )
    return Transcript(
        session_id = session_id,
        thread_id = thread_id,
        title = title_from(first_user_text(messages), "Cursor session"),
        created_at_ms = created,
        updated_at_ms = max(updated, created + max(0, len(messages) - 1)),
        messages = messages,
    )


SOURCE = Source(
    key = "cursor",
    label = "Cursor",
    home_env = "UNSLOTH_CURSOR_HOME",
    default_home = ".cursor",
    list_projects = list_projects,
    read_transcript = read_transcript,
)
