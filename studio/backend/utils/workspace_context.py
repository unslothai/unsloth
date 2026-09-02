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

    Deletion quiesces what is *running*, which is a point-in-time answer. A
    request that authenticated a moment earlier and had not yet reached the code
    that starts work is invisible to that sweep: it resumes afterwards, opens
    this account's databases under the same username-derived pathnames, and
    recreates the directory the retirement just renamed away. If the name has
    since been recreated, those writes land in the new account's workspace.

    So the generation is bumped first, and any binding older than the bump stops
    being able to name a workspace at all.
    """
    with _workspace_generations_lock:
        _workspace_generations[subject] = _workspace_generations.get(subject, 0) + 1


def current_workspace_subject() -> str:
    """Return the authenticated subject whose private workspace is active."""
    return _workspace_subject.get()


def workspace_binding_is_stale() -> bool:
    """Whether this context authenticated into an account that has since been deleted.

    False for anything that never bound explicitly: module-level defaults, the
    legacy owner and every process-wide job keep working exactly as before. Only
    a binding whose account was retired *after* it was taken is stale.
    """
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

    Dots are folded away rather than kept. Usernames may legally contain them
    (``^[a-z0-9][a-z0-9._-]*$``), but Windows resolves a reserved device name
    followed by any extension back to the device: ``con.txt`` would key to
    ``con.txt-<digest>``, whose base name is ``con``, and Windows refuses to
    create it. With no dot the key is a single component ending in the digest,
    which can never equal a device name. Distinct usernames still get distinct
    keys because the digest hashes the original value, so ``a.b`` and ``a-b``
    share a readable prefix but not a key.
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
    """A request outlived the account it authenticated into.

    Raised where a stale binding would otherwise recreate a deleted account's
    workspace. The routes turn it into a 401: from the caller's point of view
    the credential it presented no longer exists.
    """


class ForeignWorkspaceActiveError(RuntimeError):
    """A shared singleton is busy with work that belongs to another account.

    Raised where refusing is the only honest answer, because the resource is one
    per install and the caller cannot be shown what is on it. The routes turn it
    into a 409.
    """
