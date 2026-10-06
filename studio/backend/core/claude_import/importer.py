# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import every Claude Code conversation, keyed by the project's encoded path (survives a rename or delete)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core import external_import
from core.claude_import.discovery import list_claude_projects
from core.claude_import.transcripts import read_transcript
from core.external_import import ExternalSource, ImportSummary, SourceProject

SOURCE = ExternalSource(key = "claude", label = "Claude", read_transcript = read_transcript)


def project_id_for(slug: str) -> str:
    return external_import.project_id_for(SOURCE, slug)


def thread_id_for(session_id: str) -> str:
    return external_import.thread_id_for(SOURCE, session_id)


def import_claude_chats(*, home: Optional[Path] = None, dry_run: bool = False) -> ImportSummary:
    projects = [
        SourceProject(project.slug, project.name, project.sessions)
        for project in list_claude_projects(home)
    ]
    return external_import.import_projects(SOURCE, projects, dry_run = dry_run)


__all__ = ["import_claude_chats", "project_id_for", "thread_id_for"]
