# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import Claude Code conversations into Studio."""

from core.claude_import.discovery import (
    CLAUDE_HOME_ENV,
    ClaudeProject,
    claude_home,
    list_claude_projects,
    read_project,
)
from core.claude_import.importer import (
    ClaudeImportSummary,
    import_claude_chats,
    project_id_for,
    thread_id_for,
)
from core.claude_import.transcripts import ClaudeTranscript, read_transcript

__all__ = [
    "CLAUDE_HOME_ENV",
    "ClaudeImportSummary",
    "ClaudeProject",
    "ClaudeTranscript",
    "claude_home",
    "import_claude_chats",
    "list_claude_projects",
    "project_id_for",
    "read_project",
    "read_transcript",
    "thread_id_for",
]
