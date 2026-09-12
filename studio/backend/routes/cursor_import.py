# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import Cursor conversations into Studio.

Two endpoints, because the settings row has two things to say: whether this
computer has any Cursor history at all, and then the result of bringing it in.
Both read the local disk only.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.authentication import get_current_subject
from core.cursor_import import import_cursor_chats, list_cursor_workspaces
from loggers import get_logger
from utils.utils import log_and_http_error

router = APIRouter()

logger = get_logger(__name__)


class CursorImportStatus(BaseModel):
    available: bool = False
    projects: int = 0
    chats: int = 0


class CursorImportResult(BaseModel):
    projects: int = 0
    chats: int = 0
    new_chats: int = 0
    messages: int = 0
    skipped: int = 0
    warnings: list[str] = []


@router.get("/cursor/status", response_model = CursorImportStatus)
def cursor_status(current_subject: str = Depends(get_current_subject)):
    """What is waiting to be imported, so the UI can offer it or stay quiet."""
    try:
        # Names are what resolving a slug back to a folder buys, and a count
        # needs none of them, so the expensive half is skipped here.
        workspaces = list_cursor_workspaces(resolve_paths = False)
    except OSError as exc:
        # An unreadable Cursor directory is not an error worth a red toast: the
        # row simply does not appear.
        logger.warning("cursor_status_failed", error = str(exc))
        return CursorImportStatus()

    chats = sum(len(workspace.transcripts) for workspace in workspaces)
    return CursorImportStatus(
        available = bool(chats),
        projects = len(workspaces),
        chats = chats,
    )


@router.post("/cursor", response_model = CursorImportResult)
def cursor_import(current_subject: str = Depends(get_current_subject)):
    """Copy every Cursor conversation in, grouped by the project it came from."""
    try:
        summary = import_cursor_chats()
    except OSError as exc:
        raise log_and_http_error(
            exc,
            500,
            "Could not read Cursor's conversations.",
            event = "cursor_import_failed",
            log = logger,
        ) from exc

    return CursorImportResult(
        projects = summary.projects,
        chats = summary.chats,
        new_chats = summary.new_chats,
        messages = summary.messages,
        skipped = summary.skipped,
        warnings = summary.warnings,
    )
