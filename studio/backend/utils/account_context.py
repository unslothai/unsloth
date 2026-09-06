# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which account a request, job or thread is acting for.

One ContextVar, bound by the auth dependency once a credential resolves and
carried by asyncio tasks from there. It is deliberately NOT inherited by new
threads or spawned processes: long-running work carries the account as data
and rebinds at its execution boundary through ``run_as`` / ``account_thread``.

The default is the installation owner. An install that never created a second
account therefore resolves every request, every background loop and every test
exactly as before accounts existed, which is the compatibility guarantee the
rest of the feature rests on.

Accounts are identified by an immutable ``account_id``, never by username: a
username is a login and display attribute that can be renamed or reused, and a
storage key derived from it would let the next holder of a name inherit the
previous holder's files.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

OWNER_ACCOUNT_ID = "owner"
OWNER_USERNAME = "unsloth"
ROLE_OWNER = "owner"
ROLE_USER = "user"

T = TypeVar("T")


@dataclass(frozen = True, slots = True)
class AccountContext:
    """The acting account. ``account_id`` is the storage key; ``username`` is not."""

    account_id: str
    username: str
    role: str = ROLE_USER

    @property
    def is_owner(self) -> bool:
        return self.role == ROLE_OWNER


OWNER = AccountContext(OWNER_ACCOUNT_ID, OWNER_USERNAME, ROLE_OWNER)

_current: ContextVar[AccountContext] = ContextVar("unsloth_account", default = OWNER)


def current_account() -> AccountContext:
    return _current.get()


def current_account_id() -> str:
    return _current.get().account_id


def is_owner_context() -> bool:
    return _current.get().is_owner


def bind_account(account: AccountContext) -> Token[AccountContext]:
    """Bind the current async task or thread to ``account``; returns the reset token."""
    if not isinstance(account, AccountContext) or not account.account_id:
        raise ValueError("bind_account needs an AccountContext with an account_id")
    return _current.set(account)


def reset_account(token: Token[AccountContext]) -> None:
    _current.reset(token)


def run_as(account: AccountContext, target: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Call ``target`` bound to ``account``. Synchronous targets only: a coroutine
    returned here would run AFTER the binding is reset, under whatever account the
    awaiting task holds. Use ``arun_as`` for those."""
    token = bind_account(account)
    try:
        result = target(*args, **kwargs)
    finally:
        reset_account(token)
    if hasattr(result, "__await__"):
        raise TypeError(
            "run_as was given an awaitable; use arun_as so the binding outlives the await"
        )
    return result


async def arun_as(account: AccountContext, awaitable: Awaitable[T]) -> T:
    """Await ``awaitable`` bound to ``account`` for its whole lifetime."""
    token = bind_account(account)
    try:
        return await awaitable
    finally:
        reset_account(token)


def account_thread(
    *,
    target: Callable[..., Any],
    account: AccountContext | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    **thread_kwargs: Any,
) -> threading.Thread:
    """A thread pinned to one account, captured at creation rather than at start."""
    bound = account or current_account()
    return threading.Thread(
        target = run_as,
        args = (bound, target, *args),
        kwargs = kwargs or {},
        **thread_kwargs,
    )
