# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import Cursor / Claude Code / Codex conversations from this machine's disk into Studio."""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.authentication import get_current_subject
from auth.policy import OwnerOnly
from core.external_import import run_import, sources
from loggers import get_logger
from utils.account_context import is_owner_context
from utils.utils import log_and_http_error

router = APIRouter()

logger = get_logger(__name__)

SourceKey = Literal["cursor", "claude", "codex"]


class ExternalImportStatus(BaseModel):
    available: bool = False
    chats: int = 0


class ExternalImportResult(BaseModel):
    projects: int = 0
    chats: int = 0
    new_chats: int = 0
    messages: int = 0
    skipped: int = 0
    warnings: list[str] = []


@router.get("/{source}/status", response_model = ExternalImportStatus)
def external_import_status(source: SourceKey, current_subject: str = Depends(get_current_subject)):
    # The host home is the owner's: a managed account never sees its histories.
    if not is_owner_context():
        return ExternalImportStatus()
    try:
        chats = sources()[source].session_count()
    except OSError as exc:
        logger.warning("external_import_status_failed", source = source, error = str(exc))
        return ExternalImportStatus()
    return ExternalImportStatus(available = bool(chats), chats = chats)


# Authenticate before the owner check: decorator dependencies run before parameter ones, and until the
# bearer is resolved the request still carries the default (owner) account.
@router.post(
    "/{source}",
    response_model = ExternalImportResult,
    dependencies = [Depends(get_current_subject), OwnerOnly],
)
def external_import(source: SourceKey, current_subject: str = Depends(get_current_subject)):
    try:
        summary = run_import(sources()[source])
    except OSError as exc:
        raise log_and_http_error(
            exc,
            500,
            f"Could not read {sources()[source].label}'s conversations.",
            event = "external_import_failed",
            log = logger,
        ) from exc
    return ExternalImportResult(**vars(summary))
