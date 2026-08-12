# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Copy every Cursor conversation into Studio, grouped by the project it came from.

One Cursor project becomes one Studio project holding that project's chats.
Identity is the Cursor state slug rather than the folder path, because the slug
is what survives: a project whose folder was renamed or deleted still has its
transcripts, and keying on the slug means a second import updates the same rows
instead of duplicating them.

A re-import is expected -- it is how new conversations arrive -- so anything the
user has since changed in Studio wins: a renamed chat keeps its name, an
archived project stays archived, and a chat deleted after an earlier import is
not resurrected.

Writes go through the storage layer rather than the HTTP API, so the import
works the same whether or not a server is in front of it.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.cursor_import.discovery import (
    NO_FOLDER_SLUG,
    CursorWorkspace,
    list_cursor_workspaces,
)
from core.cursor_import.transcripts import read_transcript
from loggers import get_logger
from storage import studio_db

logger = get_logger(__name__)

# Imported threads have no model of their own: the user picks one when they
# continue the conversation, which is what an empty base model id means to the UI.
_IMPORTED_MODEL_TYPE = "base"
_IMPORTED_MODEL_ID = ""

# How an imported project is labelled, so it reads as Cursor's in a list that
# also holds projects made here.
_PROJECT_PREFIX = "Cursor"


@dataclass
class CursorImportSummary:
    """What the import did, or would do when ``dry_run`` is set."""

    projects: int = 0
    chats: int = 0
    # Conversations Studio had not seen before. Zero on a second import, which
    # is how the UI knows to say "up to date" rather than repeat the total.
    new_chats: int = 0
    messages: int = 0
    # Conversations left out: empty ones, and ones deleted in Studio after an
    # earlier import.
    skipped: int = 0
    warnings: list[str] = field(default_factory = list)

    @property
    def imported_anything(self) -> bool:
        return bool(self.chats)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:12]}"


def project_id_for(slug: str) -> str:
    """Deterministic project id, so a re-import updates the same project."""
    return _stable_id("cursor", slug)


def thread_id_for(session_id: str) -> str:
    """Deterministic thread id, so a re-import updates the same conversation."""
    return _stable_id("cursor-thread", session_id)


def _import_transcript(
    path: Path, *, project_id: str, summary: CursorImportSummary, dry_run: bool
) -> bool:
    """Write one session as a thread. False when there was nothing to write."""
    session_id = path.stem
    thread_id = thread_id_for(session_id)
    try:
        transcript = read_transcript(path, thread_id, session_id = session_id)
    except OSError as exc:
        summary.warnings.append(f"{path.name}: could not be read ({exc.strerror or exc}).")
        return False

    if transcript.is_empty:
        summary.skipped += 1
        return False

    # A title, a rename and an archive are the user's to keep: the transcript's
    # first prompt never changes, so a re-import has nothing better to say than
    # whatever the chat is already called in Studio.
    existing = studio_db.get_chat_thread(thread_id) or {}
    if dry_run:
        summary.new_chats += 0 if existing else 1
        summary.messages += len(transcript.messages)
        return True

    try:
        studio_db.upsert_chat_thread(
            {
                "id": thread_id,
                "title": existing.get("title") or transcript.title,
                "modelType": existing.get("modelType") or _IMPORTED_MODEL_TYPE,
                "modelId": existing.get("modelId") or _IMPORTED_MODEL_ID,
                "projectId": project_id,
                "archived": bool(existing.get("archived")),
                "createdAt": transcript.created_at_ms,
                "updatedAt": transcript.updated_at_ms,
            }
        )
    except studio_db.ChatThreadDeletedError:
        # Deleted in Studio after an earlier import. Recreating it would undo a
        # deliberate deletion.
        summary.skipped += 1
        return False

    studio_db.sync_chat_messages(thread_id, transcript.messages, prune_missing = False)
    if not existing:
        summary.new_chats += 1
    summary.messages += len(transcript.messages)
    return True


def _import_workspace(
    workspace: CursorWorkspace,
    *,
    summary: CursorImportSummary,
    claimed_sessions: set[str],
    now_ms: int,
    dry_run: bool,
) -> None:
    """Write one Cursor project and the conversations that belong to it."""
    # Cursor files a session that began before a folder was opened under both
    # that folder and the no-folder window, so a run over every project meets
    # the same transcript twice. A conversation is one chat, so the first
    # project to claim it keeps it.
    transcripts = [path for path in workspace.transcripts if path.stem not in claimed_sessions]
    if not transcripts:
        return
    claimed_sessions.update(path.stem for path in transcripts)

    project_id = project_id_for(workspace.slug)
    existing = studio_db.get_chat_project(project_id) or {}
    if not dry_run:
        # An upsert overwrites every column it is given, so the name, the
        # instructions and the archived flag are carried over: the user may have
        # renamed this project or filed it away since the last import.
        studio_db.upsert_chat_project(
            {
                "id": project_id,
                "name": existing.get("name") or f"{_PROJECT_PREFIX} · {workspace.name}",
                "instructions": existing.get("instructions") or "",
                "archived": bool(existing.get("archived")),
                "createdAt": existing.get("createdAt") or now_ms,
                "updatedAt": now_ms,
            }
        )

    imported = 0
    for path in transcripts:
        if _import_transcript(path, project_id = project_id, summary = summary, dry_run = dry_run):
            imported += 1

    if imported:
        summary.projects += 1
        summary.chats += imported
    elif not dry_run and not existing:
        # Every conversation here turned out empty, or was deleted in Studio
        # after an earlier import. The project row was written to hang them off
        # and now has nothing in it, so it goes rather than sitting in the
        # sidebar as an empty entry the user never made.
        studio_db.delete_chat_project(project_id)


def import_cursor_chats(
    *, home: Optional[Path] = None, dry_run: bool = False
) -> CursorImportSummary:
    """Import every conversation Cursor has on this machine."""
    workspaces = list_cursor_workspaces(home)
    # The no-folder window shares sessions with the projects a folder was later
    # opened in, and those belong to the folder, so it goes last and keeps only
    # the sessions no real project claimed.
    workspaces.sort(key = lambda workspace: workspace.slug == NO_FOLDER_SLUG)

    summary = CursorImportSummary()
    claimed: set[str] = set()
    now_ms = int(time.time() * 1000)
    for workspace in workspaces:
        _import_workspace(
            workspace,
            summary = summary,
            claimed_sessions = claimed,
            now_ms = now_ms,
            dry_run = dry_run,
        )

    logger.info(
        "cursor_import_finished",
        projects = summary.projects,
        chats = summary.chats,
        new_chats = summary.new_chats,
        messages = summary.messages,
        skipped = summary.skipped,
        dry_run = dry_run,
    )
    return summary


__all__ = [
    "CursorImportSummary",
    "import_cursor_chats",
    "project_id_for",
    "thread_id_for",
]
