# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memo for lookups a model inventory scan repeats per repo (root path resolves, the Hugging Face
cache setting, registered scan folders). Scoped to one scan, never kept across requests, so a
change is seen on the next one; outside a scope every lookup runs as before."""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Hashable, Iterator, Optional, TypeVar

_T = TypeVar("_T")

# asyncio.to_thread copies the context, so worker threads of one scan share its dict.
_scope: ContextVar[Optional[dict]] = ContextVar("studio_scan_memo", default = None)
# Set when the scope ends: a task or thread that copied the context and outlives it stops using it.
_CLOSED = object()
# Bumped by invalidate_hf_cache_scans: an inventory retry in the same scope must not reuse old roots.
_generation = 0


def invalidate_scan_memos() -> None:
    global _generation
    _generation += 1


@contextmanager
def scan_memo() -> Iterator[None]:
    """Memoize lookups inside; a nested scope joins the outer one."""
    current = _scope.get()
    if current is not None and _CLOSED not in current:
        yield
        return
    memo: dict = {}
    token = _scope.set(memo)
    try:
        yield
    finally:
        memo[_CLOSED] = True
        _scope.reset(token)


def scan_memo_active() -> bool:
    """Whether the caller runs inside an open scan_memo() scope."""
    memo = _scope.get()
    return memo is not None and _CLOSED not in memo


def memoized(key: Hashable, compute: Callable[[], _T]) -> _T:
    """compute() once per scope for key; a raise is not cached."""
    memo = _scope.get()
    if memo is None or _CLOSED in memo:
        return compute()
    key = (_generation, key)
    try:
        return memo[key]
    except KeyError:
        pass
    value = compute()
    memo[key] = value
    return value


def in_scan_memo(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a sync or async function inside scan_memo()."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def run_async(*args, **kwargs):
            with scan_memo():
                return await fn(*args, **kwargs)

        return run_async

    @functools.wraps(fn)
    def run(*args, **kwargs):
        with scan_memo():
            return fn(*args, **kwargs)

    return run
