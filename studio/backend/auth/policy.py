# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide account policy, cached against an account generation bumped on create/delete so a one-user install queries nothing."""

from __future__ import annotations

import contextlib
import threading
from typing import Optional

from fastapi import Depends, HTTPException, status

LOGIN_MODE_SINGLE = "single"
LOGIN_MODE_MULTI = "multi"

_lock = threading.Lock()
_generation = 0
_cached: Optional[tuple[int, int, int]] = None
# Depth of in-flight account mutations; the cache is bypassed entirely while any is open.
_mutating = 0


def invalidate_account_cache() -> None:
    """Call after any account is created, deleted, activated or deactivated."""
    global _generation, _cached
    with _lock:
        _generation += 1
        _cached = None


@contextlib.contextmanager
def account_mutation():
    """Wrap an account write, invalidation included. Invalidating only after the commit leaves a
    window where the row is durable but the cache still answers the pre-write verdict; bypassing
    the cache for the whole write makes every reader recompute. Counted, so nested mutations work
    and a slow write never blocks a policy read."""
    global _mutating, _generation, _cached
    with _lock:
        _mutating += 1
    try:
        yield
    finally:
        # One critical section: dropping the suppression before invalidating would reopen the window.
        with _lock:
            _mutating -= 1
            _generation += 1
            _cached = None


def account_generation() -> int:
    return _generation


def _account_counts() -> tuple[int, int]:
    global _cached
    with _lock:
        if not _mutating and _cached is not None and _cached[0] == _generation:
            return _cached[1], _cached[2]
        generation = _generation
    from auth import storage

    try:
        active, managed = storage.account_counts()
    except Exception:  # noqa: BLE001 - an unreadable auth.db is a one-user install
        # Never cached: a transient read error would hold full access off until restart.
        from utils.account_context import is_owner_context

        # A bound managed account proves a multi-user install, so isolation stays on.
        return (1 if is_owner_context() else 2), 1
    with _lock:
        if not _mutating and generation == _generation:
            _cached = (generation, active, managed)
    return active, managed


def active_account_count() -> int:
    return _account_counts()[0]


def managed_account_count() -> int:
    """Managed accounts of any state; a deactivated one still counts."""
    return _account_counts()[1]


def installation_is_multi_user() -> bool:
    return active_account_count() > 1


def login_mode() -> str:
    return LOGIN_MODE_MULTI if installation_is_multi_user() else LOGIN_MODE_SINGLE


def installation_has_managed_accounts() -> bool:
    """Whether any managed account exists; gates on this rather than the login mode, since a deactivated account's files stay on disk."""
    return installation_is_multi_user() or managed_account_count() > 0


def full_access_permitted() -> bool:
    """Whether the unsandboxed tool modes may run; refused install-wide, owner included, once any managed account exists."""
    return not installation_has_managed_accounts()


def _forbid(detail: str) -> HTTPException:
    return HTTPException(status_code = status.HTTP_403_FORBIDDEN, detail = detail)


async def require_owner() -> None:
    from utils.account_context import current_account
    if not current_account().is_owner:
        raise _forbid("Only the installation owner can do this")


def require_account_scope(resource_account_id: Optional[str]) -> None:
    """Refuse a resource owned by another account; ``None`` predates accounts and is the owner's."""
    from utils.account_context import OWNER_ACCOUNT_ID, current_account_id

    owner_of = resource_account_id or OWNER_ACCOUNT_ID
    if owner_of != current_account_id():
        # 404 rather than 403: the resource's existence is itself information.
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = "Not found")


OwnerOnly = Depends(require_owner)
