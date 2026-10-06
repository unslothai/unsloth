# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import every Cursor conversation, keyed by Cursor's state slug (survives a rename or delete)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core import external_import
from core.cursor_import.discovery import NO_FOLDER_SLUG, list_cursor_workspaces
from core.cursor_import.transcripts import read_transcript
from core.external_import import ExternalSource, ImportSummary, SourceProject

SOURCE = ExternalSource(key = "cursor", label = "Cursor", read_transcript = read_transcript)


def project_id_for(slug: str) -> str:
    return external_import.project_id_for(SOURCE, slug)


def thread_id_for(session_id: str) -> str:
    return external_import.thread_id_for(SOURCE, session_id)


def import_cursor_chats(*, home: Optional[Path] = None, dry_run: bool = False) -> ImportSummary:
    # Cursor files a session started before a folder was opened under both that folder and
    # the no-folder window; the folder keeps it, so the no-folder window goes last.
    workspaces = sorted(
        list_cursor_workspaces(home), key = lambda workspace: workspace.slug == NO_FOLDER_SLUG
    )
    claimed: set[str] = set()
    projects = []
    for workspace in workspaces:
        sessions = [path for path in workspace.transcripts if path.stem not in claimed]
        if sessions:
            claimed.update(path.stem for path in sessions)
            projects.append(SourceProject(workspace.slug, workspace.name, sessions))
    return external_import.import_projects(SOURCE, projects, dry_run = dry_run)


__all__ = ["import_cursor_chats", "project_id_for", "thread_id_for"]
