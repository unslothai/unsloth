# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Copy every Claude Code conversation into Studio, grouped by its project.

One Claude Code project becomes one Studio project holding that project's
chats. Identity is the project's encoded path rather than the folder path,
because the encoding is what survives: a project whose folder was renamed or
deleted still has its transcripts, and keying on the slug means a second import
updates the same rows instead of duplicating them.

A re-import is expected -- it is how new conversations arrive -- so anything the
user has since changed in Studio wins. A renamed chat keeps its name, an
archived project stays archived, a chat moved to Recents or to another project
stays where it was put, a chat deleted after an earlier import is not
resurrected, and an edited or deleted message is left as Studio has it: what a
second import writes is the turns Claude appended since the first.

Writes go through the storage layer rather than the HTTP API, so the import
works the same whether or not a server is in front of it.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.claude_import.discovery import ClaudeProject, list_claude_projects
from core.claude_import.transcripts import ClaudeTranscript, read_transcript
from loggers import get_logger
from storage import studio_db

logger = get_logger(__name__)

# Imported threads have no model of their own: the user picks one when they
# continue the conversation, which is what an empty base model id means to the UI.
_IMPORTED_MODEL_TYPE = "base"
_IMPORTED_MODEL_ID = ""

# How an imported project is labelled, so it reads as Claude's in a list that
# also holds projects made here.
_PROJECT_PREFIX = "Claude"

# The ledger's source key, keeping these sessions apart from Cursor's.
_SOURCE = "claude"


@dataclass
class ClaudeImportSummary:
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


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:12]}"


def project_id_for(slug: str) -> str:
    """Deterministic project id, so a re-import updates the same project."""
    return _stable_id("claude", slug)


def thread_id_for(session_id: str) -> str:
    """Deterministic thread id, so a re-import updates the same conversation."""
    return _stable_id("claude-thread", session_id)


def _import_transcript(
    path: Path, *, project_id: str, summary: ClaudeImportSummary, dry_run: bool
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

    existing = studio_db.get_chat_thread(thread_id) or {}
    if dry_run:
        summary.new_chats += 0 if existing else 1
        summary.messages += len(_messages_to_write(transcript, existing))
        return True

    try:
        studio_db.upsert_chat_thread(_thread_row(transcript, existing, project_id = project_id))
    except studio_db.ChatThreadDeletedError:
        # A targeted delete while other chats remain is the user's to keep.
        # An empty Studio is a blank slate -- clear-all, or every chat gone --
        # and Import from Claude Code is how the history comes back.
        if studio_db.list_chat_threads():
            summary.skipped += 1
            return False
        studio_db.lift_chat_thread_tombstone(thread_id)
        existing = {}
        studio_db.upsert_chat_thread(_thread_row(transcript, existing, project_id = project_id))

    if existing:
        _merge_late_tool_results(transcript)
    pending = _reseat_pending(
        _messages_to_write(transcript, existing),
        thread_id,
        transcript.messages,
    )
    if pending:
        studio_db.sync_chat_messages(thread_id, pending, prune_missing = False)
    studio_db.record_external_import_mark(
        _SOURCE, session_id, transcript.updated_at_ms, len(transcript.messages)
    )
    if not existing:
        summary.new_chats += 1
    summary.messages += len(pending)
    return True


def _thread_row(transcript: ClaudeTranscript, existing: dict, *, project_id: str) -> dict:
    """The thread to store, with everything Studio owns carried over.

    ``upsert_chat_thread`` writes every column it is handed and nulls the ones
    it is not, so an existing row is the base rather than a set of fields to
    pick from: a chat continued here can hold a code-execution container, a
    compare pair or a fork origin, none of which the transcript knows about and
    all of which a re-import would otherwise drop.

    What the transcript decides is therefore only what a first import needs. The
    title stays as Studio has it, since the opening prompt never changes and the
    user may have renamed the chat; the project stays as Studio has it too,
    including ``None`` for a chat moved to Recents; and the time is the later of
    the two, so continuing a conversation here does not send it back down the
    sidebar on the next import.
    """
    return {
        **existing,
        "id": transcript.thread_id,
        "title": existing.get("title") or transcript.title,
        "modelType": existing.get("modelType") or _IMPORTED_MODEL_TYPE,
        "modelId": existing.get("modelId") or _IMPORTED_MODEL_ID,
        "projectId": existing.get("projectId") if existing else project_id,
        "archived": bool(existing.get("archived")),
        "createdAt": existing.get("createdAt") or transcript.created_at_ms,
        "updatedAt": max(transcript.updated_at_ms, int(existing.get("updatedAt") or 0)),
    }


def _messages_to_write(transcript: ClaudeTranscript, existing_thread: dict) -> list[dict]:
    """The turns Claude has appended since the last import, and only those.

    A turn already imported is left alone. Rewriting it would undo an edit made
    in Studio, and there is nothing to gain: a session's earlier turns are
    settled by the time Claude appends the next one.

    Which turns those are cannot be read off the message rows, since a turn the
    user deleted here leaves nothing behind to tell it from one never imported.
    The ledger holds the count instead, so a deleted message stays deleted while
    genuinely new turns still arrive. It is only trusted while the chat is still
    in Studio: with the thread gone -- history cleared, or a fresh database --
    the session is imported whole again.
    """
    if not existing_thread:
        return list(transcript.messages)
    mark = studio_db.get_external_import_mark(_SOURCE, transcript.session_id)
    if mark is None:
        # Imported before this ledger existed. Its rows are the only record of
        # how far it got, and rewriting them would overwrite any edit made
        # since, so the chat keeps what it has and picks up new turns from here.
        return []
    # Count, not mtime: a filesystem that does not bump the file's clock when
    # Claude appends would otherwise record the new length as imported without
    # writing the turns, and they would never arrive.
    return transcript.messages[mark["turnsImported"] :]


def _reseat_pending(pending: list[dict], thread_id: str, all_messages: list[dict]) -> list[dict]:
    """Hang new turns off a parent that still exists in Studio.

    The transcript chain points at the message before each turn. If the user
    deleted that one here, writing the recorded parent would leave the new turn
    hanging off a missing row, and the chat would load as an orphan.
    """
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


def _merge_late_tool_results(transcript: ClaudeTranscript) -> None:
    """Attach results that arrived after a tool-call was already imported.

    A first import can land while a call is still open. The assistant row is
    stored without a ``result``, and an append-only second import would start
    after that row, so the output would never reach the card. Only the missing
    result is written; any edit the user made to the rest of the message stays.
    """
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
    project: ClaudeProject, *, summary: ClaudeImportSummary, now_ms: int, dry_run: bool
) -> None:
    """Write one Claude Code project and the conversations that belong to it."""
    project_id = project_id_for(project.slug)
    existing = studio_db.get_chat_project(project_id) or {}
    if not dry_run:
        # An upsert overwrites every column it is given, so the name, the
        # instructions and the archived flag are carried over: the user may have
        # renamed this project or filed it away since the last import. The
        # timestamp stays put until something actually lands here -- a no-op
        # click must not send a stale imported project back to the top of the
        # sidebar.
        studio_db.upsert_chat_project(
            {
                "id": project_id,
                "name": existing.get("name") or f"{_PROJECT_PREFIX} · {project.name}",
                "instructions": existing.get("instructions") or "",
                "archived": bool(existing.get("archived")),
                "createdAt": existing.get("createdAt") or now_ms,
                "updatedAt": existing.get("updatedAt") or now_ms,
            }
        )

    imported = 0
    new_before, messages_before = summary.new_chats, summary.messages
    for path in project.sessions:
        if _import_transcript(path, project_id = project_id, summary = summary, dry_run = dry_run):
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
        # Every conversation here turned out empty, was deleted in Studio, or
        # was moved to Recents / another project. The row was written to hang
        # new chats off and now has nothing in it, so it goes rather than
        # sitting in the sidebar as an empty entry the user never made -- or
        # already deleted.
        studio_db.delete_chat_project(project_id)


def import_claude_chats(
    *, home: Optional[Path] = None, dry_run: bool = False
) -> ClaudeImportSummary:
    """Import every Claude Code conversation on this machine."""
    projects = list_claude_projects(home)

    summary = ClaudeImportSummary()
    now_ms = int(time.time() * 1000)
    for project in projects:
        _import_project(project, summary = summary, now_ms = now_ms, dry_run = dry_run)

    logger.info(
        "claude_import_finished",
        projects = summary.projects,
        chats = summary.chats,
        new_chats = summary.new_chats,
        messages = summary.messages,
        skipped = summary.skipped,
        dry_run = dry_run,
    )
    return summary


__all__ = [
    "ClaudeImportSummary",
    "import_claude_chats",
    "project_id_for",
    "thread_id_for",
]
