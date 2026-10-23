# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import another tool's local conversations into Studio, one Studio project per source project.

Re-imports are how new turns arrive, so whatever the user changed in Studio wins: titles,
project placement, archived flags, edited or deleted messages, deleted chats.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from loggers import get_logger
from storage import studio_db

logger = get_logger(__name__)

TITLE_MAX_CHARS = 120


@dataclass
class ImportSummary:
    projects: int = 0
    chats: int = 0
    # Zero on a second import, which is how the UI says "up to date".
    new_chats: int = 0
    messages: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory = list)


@dataclass
class SourceProject:
    slug: str
    name: str
    sessions: list[Path]


@dataclass
class Transcript:
    session_id: str
    thread_id: str
    title: str
    created_at_ms: int
    updated_at_ms: int
    messages: list[dict]
    # Which physical file the messages came from; a change (Codex revert) re-bases the ledger
    # at ``inherited``, the count of leading messages carried over from the previous revision.
    revision: str = ""
    inherited: int = 0


@dataclass(frozen = True)
class Source:
    # Ledger key and id prefix; changing it duplicates every imported chat.
    key: str
    label: str
    home_env: str
    default_home: str
    list_projects: Callable[[Path], list[SourceProject]]
    read_transcript: Callable[[Path, str, str], Transcript]
    # Cheaper than list_projects for the status probe that runs whenever Settings > Data opens.
    count_sessions: Optional[Callable[[Path], int]] = None
    # Thread identity for a session file; Codex keys by session_meta.id (a revert writes a new file).
    session_key: Optional[Callable[[Path], str]] = None

    def session_id(self, path: Path) -> str:
        return self.session_key(path) if self.session_key else session_id_of(path)

    def session_count(self) -> int:
        home = self.home()
        if self.count_sessions is not None:
            return self.count_sessions(home)
        return sum(len(project.sessions) for project in self.list_projects(home))

    def home(self, override: Optional[Path] = None) -> Path:
        if override is not None:
            return Path(override).expanduser()
        from_env = (os.environ.get(self.home_env) or "").strip()
        return Path(from_env).expanduser() if from_env else Path.home() / self.default_home


def stable_id(
    prefix: str,
    *parts: str,
    length: int = 12,
) -> str:
    digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def project_id_for(source: Source, slug: str) -> str:
    return stable_id(source.key, slug)


def session_id_of(path: Path) -> str:
    # Not path.stem: Codex compresses old x.jsonl to x.jsonl.zst, same session.
    return path.name.split(".", 1)[0]


def thread_id_for(source: Source, session_id: str) -> str:
    return stable_id(f"{source.key}-thread", session_id)


def display_name(slug: str) -> str:
    """The slug with the user's home folder dropped; dashes are lossy, so the real path is unknowable."""
    folded = re.sub("-+", "-", slug).strip("-")
    try:
        home = re.sub(r"[^A-Za-z0-9]+", "-", str(Path.home())).strip("-")
    except (OSError, RuntimeError):
        home = ""
    if home and folded.lower().startswith(f"{home.lower()}-"):
        return folded[len(home) + 1 :]
    return folded or slug


def title_from(text: str, default: str) -> str:
    """Matches the frontend's fallback title."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return default
    if len(first_line) <= TITLE_MAX_CHARS:
        return first_line
    return first_line[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def file_times_ms(path: Path) -> tuple[int, int]:
    info = path.stat()
    created = getattr(info, "st_birthtime", None) or info.st_ctime
    return int(created * 1000), int(max(info.st_mtime, created) * 1000)


def _zstd_reader(raw):
    try:
        # novermin -- 3.14; the ImportError fallback to zstandard below is the guard, which vermin cannot see.
        from compression import zstd  # novermin
        return zstd.ZstdFile(raw)
    except ImportError:
        import zstandard  # ImportError here means the caller skips the file
        return zstandard.ZstdDecompressor().stream_reader(raw)


def read_jsonl(path: Path) -> Iterator[dict]:
    """Records of a JSONL(.zst) file; a half-written last line is normal for a live session."""
    with path.open("rb") as raw:
        stream = _zstd_reader(raw) if path.suffix == ".zst" else raw
        for line in io.TextIOWrapper(stream, encoding = "utf-8", errors = "replace"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def iso_ms(value) -> Optional[int]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo = timezone.utc)
    return int(parsed.timestamp() * 1000)


def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tool_call(tool_id: str, block: dict) -> dict:
    arguments = block.get("input")
    return {
        "type": "tool-call",
        "toolCallId": tool_id,
        "toolName": str(block.get("name") or "unknown"),
        "args": arguments if isinstance(arguments, dict) else {},
    }


def first_user_text(messages: list[dict]) -> str:
    return next(
        (
            part["text"]
            for message in messages
            if message["role"] == "user"
            for part in message["content"]
            if part.get("type") == "text"
        ),
        "",
    )


def _thread_row(transcript: Transcript, existing: dict, project_id: str) -> dict:
    # upsert_chat_thread nulls columns it is not handed, so the stored row is the base:
    # sandbox container, compare pair and fork origin survive a re-import.
    return {
        **existing,
        "id": transcript.thread_id,
        "title": existing.get("title") or transcript.title,
        # An empty model id lets the user pick one when they continue the chat.
        "modelType": existing.get("modelType") or "base",
        "modelId": existing.get("modelId") or "",
        # None for a chat the user moved to Recents.
        "projectId": existing.get("projectId") if existing else project_id,
        "archived": bool(existing.get("archived")),
        "createdAt": existing.get("createdAt") or transcript.created_at_ms,
        "updatedAt": max(transcript.updated_at_ms, int(existing.get("updatedAt") or 0)),
    }


def _pending_messages(source: Source, transcript: Transcript, existing: bool) -> list[dict]:
    # Only turns appended since the last import. The ledger count, not the message rows,
    # says which those are: a message deleted in Studio leaves no row behind.
    if not existing:
        return list(transcript.messages)
    stored = studio_db.list_chat_messages(transcript.thread_id)
    mark = studio_db.get_external_import_mark(source.key, transcript.session_id)
    if mark is None:
        # No rows and no mark = shell thread from an interrupted import.
        return [] if stored else list(transcript.messages)
    turns, revision = mark
    pending = transcript.messages[
        turns if revision == transcript.revision else transcript.inherited :
    ]
    # Hang each new turn off its nearest ancestor that exists, so a parent deleted in Studio
    # does not orphan it (the frontend refuses a thread with a missing parent).
    present = {message["id"] for message in stored}
    by_id = {message["id"]: message for message in transcript.messages}
    reseated = []
    for message in pending:
        parent = message.get("parentId")
        while parent and parent not in present:
            parent = by_id.get(parent, {}).get("parentId")
        reseated.append({**message, "parentId": parent})
        present.add(message["id"])
    return reseated


def _late_tool_results(transcript: Transcript) -> list[dict]:
    """Stored messages whose tool calls were imported while open and have a result now."""
    stored = {m["id"]: m for m in studio_db.list_chat_messages(transcript.thread_id)}
    patched = []
    for message in transcript.messages:
        current = stored.get(message["id"])
        results = {
            part["toolCallId"]: part["result"]
            for part in message["content"]
            if part.get("type") == "tool-call" and "result" in part
        }
        if current is None or not results:
            continue
        content = [
            {**part, "result": results[part["toolCallId"]]}
            if part.get("type") == "tool-call"
            and "result" not in part
            and part.get("toolCallId") in results
            else part
            for part in current["content"]
        ]
        if content != current["content"]:
            patched.append({**current, "content": content})
    return patched


def _import_session(source: Source, path: Path, project_id: str, summary: ImportSummary) -> bool:
    session_id = source.session_id(path)
    thread_id = thread_id_for(source, session_id)
    try:
        transcript = source.read_transcript(path, thread_id, session_id)
    except OSError as exc:
        summary.warnings.append(f"{path.name}: could not be read ({exc.strerror or exc}).")
        return False
    except ImportError:
        summary.warnings.append(f"{path.name}: compressed; install zstandard to import it.")
        return False
    if not transcript.messages:
        summary.skipped += 1
        return False

    existing = studio_db.get_chat_thread(thread_id) or {}
    try:
        studio_db.upsert_chat_thread(_thread_row(transcript, existing, project_id))
    except studio_db.ChatThreadDeletedError:
        summary.skipped += 1
        return False
    pending = _pending_messages(source, transcript, bool(existing))
    if existing:
        pending = _late_tool_results(transcript) + pending
    if pending:
        studio_db.sync_chat_messages(thread_id, pending, prune_missing = False)
    studio_db.record_external_import_mark(
        source.key, session_id, len(transcript.messages), transcript.revision
    )
    summary.new_chats += 0 if existing else 1
    summary.messages += len(pending)
    return True


def _import_project(
    source: Source, project: SourceProject, summary: ImportSummary, now_ms: int
) -> None:
    project_id = project_id_for(source, project.slug)
    existing = studio_db.get_chat_project(project_id) or {}
    studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": existing.get("name") or f"{source.label} · {project.name}",
            "instructions": existing.get("instructions") or "",
            "archived": bool(existing.get("archived")),
            "createdAt": existing.get("createdAt") or now_ms,
            "updatedAt": existing.get("updatedAt") or now_ms,
        }
    )
    before = (summary.new_chats, summary.messages)
    imported = sum(_import_session(source, path, project_id, summary) for path in project.sessions)
    if studio_db.list_chat_threads(project_id = project_id):
        # updatedAt only moves when something landed, so a no-op click does not re-sort the sidebar.
        if (summary.new_chats, summary.messages) != before:
            studio_db.update_chat_project(project_id, {"updatedAt": now_ms})
        summary.projects += 1
        summary.chats += imported
    elif not existing:
        studio_db.delete_chat_project(project_id)


def run_import(source: Source, *, home: Optional[Path] = None) -> ImportSummary:
    summary = ImportSummary()
    now_ms = int(time.time() * 1000)
    projects = source.list_projects(source.home(home))
    # An empty Studio (clear-all, fresh database) is a blank slate for this source's chats only:
    # other tombstones still stop a stale tab resurrecting a chat the user deleted.
    if not studio_db.list_chat_threads():
        studio_db.lift_chat_thread_tombstones(
            thread_id_for(source, source.session_id(p))
            for project in projects
            for p in project.sessions
        )
    for project in projects:
        _import_project(source, project, summary, now_ms)
    logger.info(
        "external_import_finished",
        source = source.key,
        projects = summary.projects,
        chats = summary.chats,
        new_chats = summary.new_chats,
        messages = summary.messages,
        skipped = summary.skipped,
    )
    return summary


def sources() -> dict[str, Source]:
    from core.external_import import claude, codex, cursor
    return {"cursor": cursor.SOURCE, "claude": claude.SOURCE, "codex": codex.SOURCE}
