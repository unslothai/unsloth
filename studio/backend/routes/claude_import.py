# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import Claude Code conversations into Studio.

Two endpoints, because the settings row has two things to say: whether this
computer has any Claude Code history at all, and then the result of bringing it
in. Both read the local disk only.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.authentication import get_current_subject
from core.claude_import import import_claude_chats, list_claude_projects
from loggers import get_logger
from utils.utils import log_and_http_error

router = APIRouter()

logger = get_logger(__name__)


class ClaudeImportStatus(BaseModel):
    available: bool = False
    projects: int = 0
    chats: int = 0


class ClaudeImportResult(BaseModel):
    projects: int = 0
    chats: int = 0
    new_chats: int = 0
    messages: int = 0
    skipped: int = 0
    warnings: list[str] = []


@router.get("/claude/status", response_model = ClaudeImportStatus)
def claude_status(current_subject: str = Depends(get_current_subject)):
    """What is waiting to be imported, so the UI can offer it or stay quiet."""
    try:
        # Listing sessions only stats files; nothing here resolves paths, so the
        # probe is cheap enough to run every time the settings open.
        projects = list_claude_projects()
    except OSError as exc:
        # An unreadable Claude Code directory is not an error worth a red toast:
        # the row simply does not appear.
        logger.warning("claude_status_failed", error = str(exc))
        return ClaudeImportStatus()

    chats = sum(len(project.sessions) for project in projects)
    return ClaudeImportStatus(
        available = bool(chats),
        projects = len(projects),
        chats = chats,
    )


@router.post("/claude", response_model = ClaudeImportResult)
def claude_import(current_subject: str = Depends(get_current_subject)):
    """Copy every Claude Code conversation in, grouped by the project it came from."""
    try:
        summary = import_claude_chats()
    except OSError as exc:
        raise log_and_http_error(
            exc,
            500,
            "Could not read Claude Code's conversations.",
            event = "claude_import_failed",
            log = logger,
        ) from exc

    return ClaudeImportResult(
        projects = summary.projects,
        chats = summary.chats,
        new_chats = summary.new_chats,
        messages = summary.messages,
        skipped = summary.skipped,
        warnings = summary.warnings,
    )
