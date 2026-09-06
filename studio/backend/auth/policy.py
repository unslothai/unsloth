# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide account policy.

Everything here answers one question in one place: is this a one-user install,
where nothing changes, or a multi-user install, where isolation rules apply?
The answer is cached against an account generation that account creation and
deletion bump, so the hot path of a one-user install performs no query.
"""

from __future__ import annotations

import threading
from typing import Optional

from fastapi import Depends, HTTPException, status

LOGIN_MODE_SINGLE = "single"
LOGIN_MODE_MULTI = "multi"

_lock = threading.Lock()
_generation = 0
_cached: Optional[tuple[int, int]] = None  # (generation, active account count)


def invalidate_account_cache() -> None:
    """Call after any account is created, deleted, activated or deactivated."""
    global _generation, _cached
    with _lock:
        _generation += 1
        _cached = None


def active_account_count() -> int:
    global _cached
    with _lock:
        if _cached is not None and _cached[0] == _generation:
            return _cached[1]
        generation = _generation
    from auth import storage

    try:
        count = storage.count_active_accounts()
    except Exception:  # noqa: BLE001 - an unreadable auth.db is a one-user install
        count = 1
    with _lock:
        if generation == _generation:
            _cached = (generation, count)
    return count


def installation_is_multi_user() -> bool:
    return active_account_count() > 1


def login_mode() -> str:
    return LOGIN_MODE_MULTI if installation_is_multi_user() else LOGIN_MODE_SINGLE


def full_access_permitted() -> bool:
    """Whether the unsandboxed tool modes may run at all.

    Full access reads and writes the host as the server user, which in a
    multi-user install means every other account's workspace. It is a
    single-user feature, so it is refused install-wide the moment a second
    account exists, for the owner too: the owner's own sandbox is also where
    another account's uploads would be reachable from.
    """
    return not installation_is_multi_user()


def _forbid(detail: str) -> HTTPException:
    return HTTPException(status_code = status.HTTP_403_FORBIDDEN, detail = detail)


async def require_owner() -> None:
    """Dependency for installation-wide operations: accounts, updates, shutdown,
    executables, network exposure, global caches."""
    from utils.account_context import current_account
    if not current_account().is_owner:
        raise _forbid("Only the installation owner can do this")


def require_account_scope(resource_account_id: Optional[str]) -> None:
    """Refuse a resource that belongs to another account. ``None`` means the
    resource predates accounts and belongs to the owner."""
    from utils.account_context import OWNER_ACCOUNT_ID, current_account_id

    owner_of = resource_account_id or OWNER_ACCOUNT_ID
    if owner_of != current_account_id():
        # 404 rather than 403: the existence of another account's resource is
        # itself information.
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = "Not found")


OwnerOnly = Depends(require_owner)
