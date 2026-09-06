# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registry of in-flight chat generations, keyed by conversation.

New Chat leaves the previous conversation streaming, so /load and /unload need
to know which chats a reload would interrupt: they refuse with 409 unless the
caller opts in to cancelling them, and GET /inference/active-generations lets
the UI name them. A frontend guard alone would miss a second tab or a REST call.

Entries hold the same threading.Event as the per-run cancel registry in
routes/inference.py, so cancel_all() closes each generation's own upstream
stream and never signals llama-server itself.

A plain dict plus a threading.Lock: no signals, no process groups, no event loop
affinity, so it behaves identically on Linux, macOS, Windows and WSL.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional

from utils.account_context import current_account_id

# Keyed by handle, not thread_id: a tool continuation can register before the previous leg
# unregisters, and one key would drop the other.
_ACTIVE: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


class ActiveGeneration:
    """Registers one in-flight generation for the duration of the block.

    Each __enter__ mints its own handle, so overlapping uses never clobber.
    """

    __slots__ = (
        "thread_id",
        "run_id",
        "cancel_event",
        "model",
        "kind",
        "account_id",
        "_handle",
        "_borrowed",
    )

    def __init__(
        self,
        cancel_event: threading.Event,
        *,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model: Optional[str] = None,
        kind: str = "chat",
        account_id: Optional[str] = None,
    ):
        # Captured at registration: the account this generation belongs to, so a
        # cancel or a swap can be scoped to one account instead of everyone.
        self.account_id = account_id or current_account_id()
        self.thread_id = thread_id or None
        self.run_id = run_id or None
        self.cancel_event = cancel_event
        self.model = model or None
        self.kind = kind
        self._handle: Optional[str] = None
        self._borrowed = False

    def __enter__(self) -> "ActiveGeneration":
        with _LOCK:
            # A durable supervisor registers before model loading starts and the route later enters its normal
            # tracker with that same event and run, so borrow the outer registration.
            if self.run_id:
                for entry in _ACTIVE.values():
                    if entry["run_id"] != self.run_id or entry["event"] is not self.cancel_event:
                        continue
                    if self.thread_id:
                        entry["thread_id"] = self.thread_id
                    if self.model:
                        entry["model"] = self.model
                    if self.kind:
                        entry["kind"] = self.kind
                    self._borrowed = True
                    return self
            self._handle = uuid.uuid4().hex
            _ACTIVE[self._handle] = {
                "handle": self._handle,
                "thread_id": self.thread_id,
                "run_id": self.run_id,
                "model": self.model,
                "kind": self.kind,
                "account_id": self.account_id,
                "started_at": time.time(),
                "event": self.cancel_event,
            }
        return self

    def __exit__(self, *exc) -> bool:
        if self._borrowed:
            self._borrowed = False
            return False
        handle, self._handle = self._handle, None
        if handle is not None:
            with _LOCK:
                _ACTIVE.pop(handle, None)
        return False


def snapshot(account_id: Optional[str] = None) -> list[dict[str, Any]]:
    """In-flight generations, newest last. Drops the Event: this is a response.

    ``account_id`` narrows to one account; None is every account, which only
    installation-wide callers (shutdown, the arbiter) should ask for.
    """
    with _LOCK:
        entries = [e for e in _ACTIVE.values() if account_id is None or e["account_id"] == account_id]
    entries.sort(key = lambda e: e["started_at"])
    return [
        {
            "handle": e["handle"],
            "thread_id": e["thread_id"],
            "run_id": e["run_id"],
            "model": e["model"],
            "kind": e["kind"],
            "account_id": e["account_id"],
            "started_at": e["started_at"],
        }
        for e in entries
    ]


def active_thread_ids(account_id: Optional[str] = None) -> list[str]:
    """Distinct conversation ids with a generation in flight, in start order.

    A first turn that races persistence has no thread id yet: count() sees it,
    this cannot name it.
    """
    seen: list[str] = []
    for e in snapshot(account_id):
        tid = e["thread_id"]
        if tid and tid not in seen:
            seen.append(tid)
    return seen


def count(account_id: Optional[str] = None) -> int:
    """Number of generations currently in flight, for one account or for all."""
    with _LOCK:
        if account_id is None:
            return len(_ACTIVE)
        return sum(1 for e in _ACTIVE.values() if e["account_id"] == account_id)


def foreign_count(account_id: str) -> int:
    """Generations in flight that belong to OTHER accounts. What a load or
    unload by ``account_id`` is not allowed to interrupt."""
    with _LOCK:
        return sum(1 for e in _ACTIVE.values() if e["account_id"] != account_id)


def cancel_all(account_id: Optional[str] = None) -> int:
    """Signal in-flight generations to stop. Returns how many were signalled.

    ``account_id`` limits the cancel to one account's generations, which is what
    every request-driven caller must pass: a user's forced reload stops their own
    chats, never somebody else's. None is everyone, for shutdown only.

    Only sets the cancel events; each stream tears itself down. Entries are
    removed by their own __exit__, so one mid-cleanup is neither lost nor double
    counted.
    """
    with _LOCK:
        events = [
            e["event"] for e in _ACTIVE.values()
            if account_id is None or e["account_id"] == account_id
        ]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def cancel_thread(thread_id: str, account_id: Optional[str] = None) -> int:
    """Signal only the generations belonging to ``thread_id``.

    Thread ids are client-chosen, so ``account_id`` (default: the acting
    account) keeps one account from stopping the same id in another."""
    if not thread_id:
        return 0
    scope = account_id or current_account_id()
    with _LOCK:
        events = [
            e["event"] for e in _ACTIVE.values()
            if e["thread_id"] == thread_id and e["account_id"] == scope
        ]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def cancel_run(run_id: str, account_id: Optional[str] = None) -> int:
    """Signal only the generation registered for a durable Studio run, scoped
    to the acting account unless told otherwise."""
    if not run_id:
        return 0
    scope = account_id or current_account_id()
    with _LOCK:
        events = [
            e["event"] for e in _ACTIVE.values()
            if e["run_id"] == run_id and e["account_id"] == scope
        ]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def reset_for_tests() -> None:
    """Drop every entry. Test-only; never called from request paths."""
    with _LOCK:
        _ACTIVE.clear()
