# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated workspace identity for request-scoped storage routing."""

from __future__ import annotations

import hashlib
import re
import threading
from contextvars import ContextVar, Token
from typing import Any, Callable


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


def run_in_workspace(subject: str, target: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run ``target`` with an explicit workspace binding.

    ``ContextVar`` values follow asyncio tasks and ``asyncio.to_thread`` calls,
    but Python deliberately does not copy them into newly-created threads or
    spawned processes. Long-running jobs must therefore carry their account
    identity as data and bind it at their execution boundary.
    """
    token = set_workspace_subject(subject)
    try:
        return target(*args, **kwargs)
    finally:
        reset_workspace_subject(token)


def workspace_thread(
    *,
    target: Callable[..., Any],
    subject: str | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    **thread_kwargs: Any,
) -> threading.Thread:
    """Create a thread that is pinned to one authenticated workspace."""
    bound_subject = subject or current_workspace_subject()
    return threading.Thread(
        target = run_in_workspace,
        args = (bound_subject, target, *args),
        kwargs = kwargs or {},
        **thread_kwargs,
    )


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
