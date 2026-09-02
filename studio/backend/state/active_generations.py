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

from utils.workspace_context import current_workspace_subject

# handle id -> entry. Keyed by handle, not thread_id: a tool continuation can register
# before the previous leg unregisters, and one key would drop the other.
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
        "subject",
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
    ):
        self.thread_id = thread_id or None
        self.run_id = run_id or None
        self.cancel_event = cancel_event
        self.model = model or None
        self.kind = kind
        # Bound at registration: the cancel and snapshot routes are per account,
        # and a handle with no owner is cancellable by anyone who can read it.
        self.subject = current_workspace_subject()
        self._handle: Optional[str] = None
        self._borrowed = False

    def __enter__(self) -> "ActiveGeneration":
        with _LOCK:
            # A durable supervisor registers before model loading starts. The
            # route later enters its normal tracker with that exact event/run;
            # borrow the outer registration so lifecycle counts stay truthful.
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
                "subject": self.subject,
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


def snapshot(subject: Optional[str] = None) -> list[dict[str, Any]]:
    """In-flight generations, newest last. Drops the Event: this is a response.

    ``subject`` limits the result to one workspace; None keeps the install-wide
    view the VRAM and lifecycle counters need.
    """
    with _LOCK:
        entries = [e for e in _ACTIVE.values() if subject is None or e.get("subject") == subject]
    entries.sort(key = lambda e: e["started_at"])
    return [
        {
            "handle": e["handle"],
            "thread_id": e["thread_id"],
            "run_id": e["run_id"],
            "model": e["model"],
            "kind": e["kind"],
            "started_at": e["started_at"],
        }
        for e in entries
    ]


def active_thread_ids(subject: Optional[str] = None) -> list[str]:
    """Distinct conversation ids with a generation in flight, in start order.

    A first turn that races persistence has no thread id yet: count() sees it,
    this cannot name it. ``subject`` limits the naming to one workspace.
    """
    seen: list[str] = []
    for e in snapshot(subject):
        tid = e["thread_id"]
        if tid and tid not in seen:
            seen.append(tid)
    return seen


def count(subject: Optional[str] = None) -> int:
    """Number of generations currently in flight.

    ``subject`` limits the count to one workspace; None keeps the install-wide
    view the VRAM and lifecycle counters need. The difference between the two is
    how many belong to somebody else.
    """
    with _LOCK:
        if subject is None:
            return len(_ACTIVE)
        return sum(1 for e in _ACTIVE.values() if e.get("subject") == subject)


def cancel_all(subject: Optional[str] = None) -> int:
    """Signal every in-flight generation to stop. Returns how many were signalled.

    Only sets the cancel events; each stream tears itself down. Entries are
    removed by their own __exit__, so one mid-cleanup is neither lost nor double
    counted.

    ``subject`` restricts the sweep to one workspace. None stays install-wide,
    which is what the sidecar swap needs: it replaces the runtime underneath
    every stream, so leaving another account's running would be worse than
    stopping it.
    """
    with _LOCK:
        events = [
            e["event"] for e in _ACTIVE.values() if subject is None or e.get("subject") == subject
        ]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def cancel_thread(thread_id: str, subject: Optional[str] = None) -> int:
    """Signal only the generations belonging to ``thread_id``.

    ``subject`` restricts the cancel to one workspace, so a conversation id
    guessed or observed by another account cannot stop this one.
    """
    if not thread_id:
        return 0
    with _LOCK:
        events = [
            e["event"]
            for e in _ACTIVE.values()
            if e["thread_id"] == thread_id and (subject is None or e.get("subject") == subject)
        ]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def cancel_run(run_id: str, subject: Optional[str] = None) -> int:
    """Signal only the generation registered for a durable Studio run."""
    if not run_id:
        return 0
    with _LOCK:
        events = [
            e["event"]
            for e in _ACTIVE.values()
            if e["run_id"] == run_id and (subject is None or e.get("subject") == subject)
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
