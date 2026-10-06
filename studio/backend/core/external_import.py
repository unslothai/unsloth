# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Write another tool's local conversations into Studio, one Studio project per source project.

Re-imports are expected (that is how new turns arrive), so whatever the user changed in Studio
wins: titles, project placement, archived flags, edited or deleted messages, deleted chats.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from loggers import get_logger
from storage import studio_db

logger = get_logger(__name__)

# An empty model id lets the user pick a model when they continue the chat.
_IMPORTED_MODEL_TYPE = "base"
_IMPORTED_MODEL_ID = ""


@dataclass
class ImportSummary:
    projects: int = 0
    chats: int = 0
    # Zero on a second import, which is how the UI says "up to date".
    new_chats: int = 0
    messages: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory = list)


@dataclass(frozen = True)
class ExternalSource:
    # Ledger source key and id prefix; changing it duplicates every imported chat.
    key: str
    label: str
    read_transcript: Callable[..., object]


@dataclass
class SourceProject:
    slug: str
    name: str
    sessions: list[Path]


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:12]}"


def project_id_for(source: ExternalSource, slug: str) -> str:
    return _stable_id(source.key, slug)


def thread_id_for(source: ExternalSource, session_id: str) -> str:
    return _stable_id(f"{source.key}-thread", session_id)


def _import_transcript(
    source: ExternalSource, path: Path, *, project_id: str, summary: ImportSummary, dry_run: bool
) -> bool:
    session_id = path.stem
    thread_id = thread_id_for(source, session_id)
    try:
        transcript = source.read_transcript(path, thread_id, session_id = session_id)
    except OSError as exc:
        summary.warnings.append(f"{path.name}: could not be read ({exc.strerror or exc}).")
        return False

    if transcript.is_empty:
        summary.skipped += 1
        return False

    existing = studio_db.get_chat_thread(thread_id) or {}
    if dry_run:
        summary.new_chats += 0 if existing else 1
        summary.messages += len(_messages_to_write(source, transcript, existing))
        return True

    try:
        studio_db.upsert_chat_thread(_thread_row(transcript, existing, project_id = project_id))
    except studio_db.ChatThreadDeletedError:
        # A targeted delete stays deleted; an empty Studio (clear-all) is a blank slate.
        if studio_db.list_chat_threads():
            summary.skipped += 1
            return False
        studio_db.lift_chat_thread_tombstone(thread_id)
        existing = {}
        studio_db.upsert_chat_thread(_thread_row(transcript, existing, project_id = project_id))

    if existing:
        _merge_late_tool_results(transcript)
    pending = _reseat_pending(
        _messages_to_write(source, transcript, existing),
        thread_id,
        transcript.messages,
    )
    if pending:
        studio_db.sync_chat_messages(thread_id, pending, prune_missing = False)
    studio_db.record_external_import_mark(
        source.key, session_id, transcript.updated_at_ms, len(transcript.messages)
    )
    if not existing:
        summary.new_chats += 1
    summary.messages += len(pending)
    return True


def _thread_row(transcript, existing: dict, *, project_id: str) -> dict:
    # upsert_chat_thread nulls columns it is not handed, so the stored row is the base:
    # sandbox container, compare pair and fork origin survive a re-import.
    return {
        **existing,
        "id": transcript.thread_id,
        "title": existing.get("title") or transcript.title,
        "modelType": existing.get("modelType") or _IMPORTED_MODEL_TYPE,
        "modelId": existing.get("modelId") or _IMPORTED_MODEL_ID,
        # None for a chat the user moved to Recents.
        "projectId": existing.get("projectId") if existing else project_id,
        "archived": bool(existing.get("archived")),
        "createdAt": existing.get("createdAt") or transcript.created_at_ms,
        "updatedAt": max(transcript.updated_at_ms, int(existing.get("updatedAt") or 0)),
    }


def _messages_to_write(source: ExternalSource, transcript, existing_thread: dict) -> list[dict]:
    # Only turns appended since the last import. The ledger count, not the message rows,
    # says which those are: a message deleted in Studio leaves no row behind.
    if not existing_thread:
        return list(transcript.messages)
    mark = studio_db.get_external_import_mark(source.key, transcript.session_id)
    if mark is None:
        # No rows and no mark = shell thread from an interrupted import; rows but no mark = leave alone.
        if not studio_db.list_chat_messages(transcript.thread_id):
            return list(transcript.messages)
        return []
    # Count, not mtime: some filesystems do not bump mtime on append.
    return transcript.messages[mark["turnsImported"] :]


def _reseat_pending(pending: list[dict], thread_id: str, all_messages: list[dict]) -> list[dict]:
    """Hang new turns off the nearest ancestor still in Studio, so a deleted parent does not orphan them."""
    if not pending:
        return pending
    stored = {message["id"] for message in studio_db.list_chat_messages(thread_id)}
    parent = pending[0].get("parentId")
    if parent is None or parent in stored:
        return pending
    by_id = {message["id"]: message for message in all_messages}
    while parent and parent not in stored:
        ancestor = by_id.get(parent)
        parent = ancestor.get("parentId") if ancestor else None
    reseated = dict(pending[0])
    reseated["parentId"] = parent
    return [reseated, *pending[1:]]


def _merge_late_tool_results(transcript) -> None:
    """Fill in results for tool calls imported while still open; the rest of the stored message stays."""
    stored = {
        message["id"]: message for message in studio_db.list_chat_messages(transcript.thread_id)
    }
    patched = []
    for message in transcript.messages:
        if message["role"] != "assistant":
            continue
        current = stored.get(message["id"])
        if current is None:
            continue
        results = {
            part["toolCallId"]: part["result"]
            for part in message["content"]
            if part.get("type") == "tool-call" and part.get("result")
        }
        if not results:
            continue
        changed = False
        content = []
        for part in current["content"]:
            if (
                part.get("type") == "tool-call"
                and not part.get("result")
                and part.get("toolCallId") in results
            ):
                content.append({**part, "result": results[part["toolCallId"]]})
                changed = True
            else:
                content.append(part)
        if changed:
            patched.append({**current, "content": content})
    if patched:
        studio_db.sync_chat_messages(transcript.thread_id, patched, prune_missing = False)


def _import_project(
    source: ExternalSource,
    project: SourceProject,
    *,
    summary: ImportSummary,
    now_ms: int,
    dry_run: bool,
) -> None:
    project_id = project_id_for(source, project.slug)
    existing = studio_db.get_chat_project(project_id) or {}
    if not dry_run:
        # updatedAt only moves when something lands, so a no-op click does not re-sort the sidebar.
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

    imported = 0
    new_before, messages_before = summary.new_chats, summary.messages
    for path in project.sessions:
        if _import_transcript(
            source, path, project_id = project_id, summary = summary, dry_run = dry_run
        ):
            imported += 1
    added = summary.new_chats > new_before or summary.messages > messages_before

    housed = (
        bool(studio_db.list_chat_threads(project_id = project_id)) if not dry_run else bool(imported)
    )
    if housed:
        if not dry_run and added:
            studio_db.update_chat_project(project_id, {"updatedAt": now_ms})
        summary.projects += 1
        summary.chats += imported
    elif not dry_run and not existing:
        # Nothing landed here: drop the row created above instead of leaving an empty project.
        studio_db.delete_chat_project(project_id)


def import_projects(
    source: ExternalSource,
    projects: Iterable[SourceProject],
    *,
    dry_run: bool = False,
) -> ImportSummary:
    summary = ImportSummary()
    now_ms = int(time.time() * 1000)
    # Lift all tombstones up front: one at a time, the second chat would see a nonempty
    # history and treat its tombstone as a targeted delete.
    if not dry_run and not studio_db.list_chat_threads():
        studio_db.lift_all_chat_thread_tombstones()
    for project in projects:
        _import_project(source, project, summary = summary, now_ms = now_ms, dry_run = dry_run)

    logger.info(
        f"{source.key}_import_finished",
        projects = summary.projects,
        chats = summary.chats,
        new_chats = summary.new_chats,
        messages = summary.messages,
        skipped = summary.skipped,
        dry_run = dry_run,
    )
    return summary
