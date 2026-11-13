# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-chat computer use state: the user's go-ahead, the last screenshot's frame, and which chat
holds the one mouse and keyboard. Memory only, so a restart asks again."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# A go-ahead lapses once the chat has left the computer alone this long.
GRANT_IDLE_S = 30 * 60.0
# How long after its last action a chat still holds the mouse and keyboard.
_CONTROL_HOLD_S = 20.0
_CONTROL_WAIT_S = 30.0
_MAX_SESSIONS = 64

COORDINATES = ("pixels", "grid")

Key = tuple[str, str]


@dataclass
class Session:
    granted: bool = False
    # Casefolded names of the apps the user let this chat act in.
    apps: set = field(default_factory = set)
    used: float = 0.0
    coordinates: str = "pixels"
    frame: Any = None
    display: Optional[int] = None
    # Where the last action left the pointer, to tell the user's own move to a corner from ours.
    pointer: Optional[tuple[float, float]] = None
    # The user took back control: "pointer" (a screen corner) or "switch" (computer use turned
    # off). Only they can lift it (an answered approval, Allow again, or switching it back on),
    # so a chat with Full access can't simply carry on.
    stopped: str = ""


_lock = threading.RLock()
_sessions: "OrderedDict[Key, Session]" = OrderedDict()
_control = threading.Lock()
_holder: Optional[Key] = None
_held_at = 0.0
# Bumped to end every chain of actions that is running now.
_halts = 0


def session_key(session_id: Optional[str]) -> Key:
    from utils.account_context import current_account_id
    return (current_account_id() or "", session_id or "")


def session(key: Key) -> Session:
    with _lock:
        found = _sessions.get(key)
        if found is None:
            found = _sessions[key] = Session()
            while len(_sessions) > _MAX_SESSIONS:
                _sessions.popitem(last = False)
        else:
            _sessions.move_to_end(key)
        return found


def is_granted(key: Key) -> bool:
    with _lock:
        found = _sessions.get(key)
        if found is None or not found.granted:
            return False
        if time.monotonic() - found.used > GRANT_IDLE_S:
            found.granted = False
            found.apps.clear()
            return False
        return True


def grant(key: Key) -> None:
    with _lock:
        found = session(key)
        found.granted = True
        found.used = time.monotonic()


def touch(key: Key) -> None:
    with _lock:
        session(key).used = time.monotonic()


def revoke(key: Key) -> None:
    with _lock:
        found = _sessions.get(key)
        if found is not None:
            found.granted = False
            found.apps.clear()
            found.frame = None


def stop(key: Key) -> None:
    """The user took back control of this chat: its go-ahead goes, and it stays stopped."""
    with _lock:
        revoke(key)
        session(key).stopped = "pointer"


def is_stopped(key: Key) -> bool:
    with _lock:
        found = _sessions.get(key)
        return found is not None and bool(found.stopped)


def resume(key: Key) -> None:
    with _lock:
        found = _sessions.get(key)
        if found is not None:
            found.stopped = ""


def stopped_count(account_id: str) -> int:
    with _lock:
        return sum(
            1
            for (account, _session), found in _sessions.items()
            if account == (account_id or "") and found.stopped == "pointer"
        )


def resume_account(account_id: str) -> int:
    """Allow again, or computer use switched back on: stopped chats may use the computer again
    on the terms of their permission mode. Their go-ahead stays withdrawn."""
    with _lock:
        count = 0
        for (account, _session), found in _sessions.items():
            if account == (account_id or "") and found.stopped:
                found.stopped = ""
                count += 1
        return count


def halts() -> int:
    return _halts


def halt_running() -> None:
    """End the chain of actions that is running now, between two of its actions."""
    global _halts
    with _lock:
        _halts += 1


def revoke_account(account_id: str) -> int:
    """Every chat of this account asks again, and what is running ends; the composer switch and
    the setting going off call this. A turn that is running keeps the tool it started with, so
    the chats also stay stopped until computer use is switched back on."""
    halt_running()
    with _lock:
        count = 0
        for (account, _session), found in _sessions.items():
            if account != (account_id or ""):
                continue
            found.stopped = found.stopped or "switch"
            if found.granted:
                found.granted = False
                found.apps.clear()
                found.frame = None
                count += 1
        return count


def set_coordinates(key: Key, mode: str) -> None:
    with _lock:
        session(key).coordinates = mode if mode in COORDINATES else "pixels"


class ControlBusy(RuntimeError):
    pass


@contextlib.contextmanager
def control(
    key: Key,
    cancel_event = None,
    *,
    take_over: bool = False,
) -> Iterator[None]:
    """One chat acts at a time, and another can't cut in between a chat's steps. `take_over`: the
    user just allowed this chat's call, which says which chat they mean to have the computer."""
    global _holder, _held_at
    deadline = time.monotonic() + _CONTROL_WAIT_S
    while not _control.acquire(timeout = 0.25):
        if cancel_event is not None and cancel_event.is_set():
            raise ControlBusy("the computer action was stopped.")
        if time.monotonic() > deadline:
            raise ControlBusy("another chat is using the computer; try again when it is done.")
    try:
        with _lock:
            now = time.monotonic()
            if not take_over and _holder not in (None, key) and now - _held_at < _CONTROL_HOLD_S:
                raise ControlBusy("another chat is using the computer; try again when it is done.")
            _holder, _held_at = key, now
        try:
            yield
        finally:
            with _lock:
                _held_at = time.monotonic()
    finally:
        _control.release()


def reset() -> None:
    """Test hook."""
    global _holder, _held_at, _halts
    with _lock:
        _sessions.clear()
        _holder, _held_at, _halts = None, 0.0, 0
