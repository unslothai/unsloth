# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated workspace identity for request-scoped storage routing."""

from __future__ import annotations

import hashlib
import re
import threading
from contextvars import ContextVar, Token
from typing import Any, Callable, NamedTuple


LEGACY_WORKSPACE_SUBJECT = "unsloth"

_workspace_subject: ContextVar[str] = ContextVar(
    "unsloth_workspace_subject",
    default = LEGACY_WORKSPACE_SUBJECT,
)


# How many times each account has been retired. Deleting an account bumps its
# entry, which fences every binding taken before the bump.
_workspace_generations: dict[str, int] = {}
_workspace_generations_lock = threading.Lock()

# The (subject, generation) a binding was taken at. Compared against the live
# generation, so a request admitted before its account was deleted cannot go on
# to recreate the workspace it authenticated into.
_workspace_admission: ContextVar[tuple[str, int] | None] = ContextVar(
    "unsloth_workspace_admission",
    default = None,
)


class WorkspaceBinding(NamedTuple):
    """The tokens a single ``set_workspace_subject`` call has to undo."""

    subject: Token[str]
    admission: Token["tuple[str, int] | None"]


def workspace_generation(subject: str) -> int:
    """How many times this account has been retired. 0 until the first deletion."""
    return _workspace_generations.get(subject, 0)


def note_workspace_retired(subject: str) -> None:
    """Fence every binding taken before now for ``subject``.

    Deletion quiesces what is RUNNING, which cannot see a request admitted a
    moment earlier and paused before it started work. That request resumes and
    recreates the directory the retirement renamed away, and a recreated name
    then inherits its writes. So the generation is bumped first, and an older
    binding stops being able to name a workspace at all.
    """
    with _workspace_generations_lock:
        _workspace_generations[subject] = _workspace_generations.get(subject, 0) + 1


def current_workspace_subject() -> str:
    """Return the authenticated subject whose private workspace is active."""
    return _workspace_subject.get()


def workspace_binding_is_stale() -> bool:
    """Whether this context authenticated into an account since deleted. False for
    anything that never bound explicitly, so the legacy owner and every
    process-wide job are unaffected."""
    admitted = _workspace_admission.get()
    if admitted is None:
        return False
    subject, generation = admitted
    if subject != _workspace_subject.get():
        # An inner binding this context has since left; it fences nothing.
        return False
    return generation != workspace_generation(subject)


def assert_workspace_binding_current() -> None:
    """Refuse to name a workspace from a binding the account outlived."""
    if workspace_binding_is_stale():
        raise RetiredWorkspaceError("This account was deleted while the request was in flight.")


def set_workspace_subject(subject: str) -> WorkspaceBinding:
    """Bind storage lookups in the current async/thread context to ``subject``."""
    if not subject:
        raise ValueError("Workspace subject cannot be empty")
    generation = workspace_generation(subject)
    return WorkspaceBinding(
        _workspace_subject.set(subject),
        _workspace_admission.set((subject, generation)),
    )


def reset_workspace_subject(token: WorkspaceBinding | Token[str]) -> None:
    """Restore a previous workspace binding (primarily useful to tests)."""
    if isinstance(token, WorkspaceBinding):
        _workspace_admission.reset(token.admission)
        _workspace_subject.reset(token.subject)
        return
    _workspace_subject.reset(token)


def run_in_workspace(subject: str, target: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run ``target`` with an explicit workspace binding.

    ContextVars follow asyncio tasks and to_thread but are deliberately not
    copied into new threads or spawned processes, so a long-running job carries
    its account identity as data and binds it at its execution boundary.
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

    Readable prefix for operators, digest so two differently-spelled usernames
    cannot collapse onto one directory after sanitisation.

    Dots are folded away: Windows resolves a reserved device name plus any
    extension back to the device, so ``con.txt`` would key to a directory whose
    base name is ``con`` and refuse to be created. The digest hashes the
    original, so ``a.b`` and ``a-b`` still differ.
    """
    value = subject or current_workspace_subject()
    readable = re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-")
    readable = (readable or "user")[:32]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def known_workspace_subjects() -> list[str]:
    """Every account whose workspace may hold state, owner included.

    Imported lazily: auth.storage reaches utils.paths, which reaches this module.
    """
    from auth.storage import list_users

    subjects = {LEGACY_WORKSPACE_SUBJECT}
    subjects.update(account["username"] for account in list_users())
    return sorted(subjects)


class RetiredWorkspaceError(RuntimeError):
    """A request outlived the account it authenticated into. Answered as 401: the
    credential it presented no longer exists."""


class ForeignWorkspaceActiveError(RuntimeError):
    """A shared singleton is busy with work that belongs to another account.

    Raised where refusing is the only honest answer, because the resource is one
    per install and the caller cannot be shown what is on it. The routes turn it
    into a 409.
    """
