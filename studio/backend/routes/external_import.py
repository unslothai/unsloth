# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import Cursor / Claude Code conversations from this machine's disk into Studio."""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.authentication import get_current_subject
from auth.policy import OwnerOnly
from core.claude_import import import_claude_chats, list_claude_projects
from core.cursor_import import import_cursor_chats, list_cursor_workspaces
from loggers import get_logger
from utils.account_context import is_owner_context
from utils.utils import log_and_http_error

router = APIRouter()

logger = get_logger(__name__)

Source = Literal["cursor", "claude"]

_LABELS = {"cursor": "Cursor", "claude": "Claude Code"}


class ExternalImportStatus(BaseModel):
    available: bool = False
    projects: int = 0
    chats: int = 0


class ExternalImportResult(BaseModel):
    projects: int = 0
    chats: int = 0
    new_chats: int = 0
    messages: int = 0
    skipped: int = 0
    warnings: list[str] = []


def _session_counts(source: Source) -> list[int]:
    if source == "cursor":
        # resolve_paths only buys display names; a count needs none.
        return [len(w.transcripts) for w in list_cursor_workspaces(resolve_paths = False)]
    return [len(p.sessions) for p in list_claude_projects()]


@router.get("/{source}/status", response_model = ExternalImportStatus)
def external_import_status(source: Source, current_subject: str = Depends(get_current_subject)):
    # The host home is the owner's: a managed account never sees its histories.
    if not is_owner_context():
        return ExternalImportStatus()
    try:
        counts = _session_counts(source)
    except OSError as exc:
        logger.warning("external_import_status_failed", source = source, error = str(exc))
        return ExternalImportStatus()
    chats = sum(counts)
    return ExternalImportStatus(available = bool(chats), projects = len(counts), chats = chats)


@router.post("/{source}", response_model = ExternalImportResult, dependencies = [OwnerOnly])
def external_import(source: Source, current_subject: str = Depends(get_current_subject)):
    run = import_cursor_chats if source == "cursor" else import_claude_chats
    try:
        summary = run()
    except OSError as exc:
        raise log_and_http_error(
            exc,
            500,
            f"Could not read {_LABELS[source]}'s conversations.",
            event = "external_import_failed",
            log = logger,
        ) from exc
    return ExternalImportResult(
        projects = summary.projects,
        chats = summary.chats,
        new_chats = summary.new_chats,
        messages = summary.messages,
        skipped = summary.skipped,
        warnings = summary.warnings,
    )
