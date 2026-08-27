# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated workspace identity for request-scoped storage routing."""

from __future__ import annotations

import hashlib
import re
from contextvars import ContextVar, Token


LEGACY_WORKSPACE_SUBJECT = "unsloth"

_workspace_subject: ContextVar[str] = ContextVar(
    "unsloth_workspace_subject",
    default = LEGACY_WORKSPACE_SUBJECT,
)


def current_workspace_subject() -> str:
    """Return the authenticated subject whose private workspace is active."""
    return _workspace_subject.get()


def set_workspace_subject(subject: str) -> Token[str]:
    """Bind storage lookups in the current async/thread context to ``subject``."""
    if not subject:
        raise ValueError("Workspace subject cannot be empty")
    return _workspace_subject.set(subject)


def reset_workspace_subject(token: Token[str]) -> None:
    """Restore a previous workspace binding (primarily useful to tests)."""
    _workspace_subject.reset(token)


def workspace_key(subject: str | None = None) -> str:
    """Filesystem-safe, stable directory key for a non-legacy account.

    The short readable prefix helps operators identify a workspace while the
    digest prevents two differently-spelled usernames from collapsing onto the
    same directory after sanitisation.
    """
    value = subject or current_workspace_subject()
    readable = re.sub(r"[^a-z0-9._-]+", "-", value.casefold()).strip("-.")
    readable = (readable or "user")[:32]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"
