# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop durable runs whose owner asked for them to end once no tab is watching.

A durable chat run or Deep Research run outlives its tab by design (#10365), which is what most
people want from a long research task. Some want the opposite: closing the tab should stop the
run, as the browser-owned stream always did. The client says so per run (``stopWhenUnattended``
on admission), and this watchdog cancels such a run once ``run_subscribers`` has heard from no
follower for ``_GRACE_S``.

Attendance, not the closing request: a phone that kills a background tab sends nothing at all.
The grace is what lets a reload, a follower reconnecting after a network blip (its backoff tops
out at 8s), or a second tab pick the run up before it is stopped. A run nobody ever attaches to
counts its grace from when it was watched, so a tab that closes between admission and its first
event stream still stops the run.

In memory, like ``run_subscribers``: a chat run does not survive a restart anyway, and a Deep
Research run re-arms its watch from its stored config when a follower reattaches.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from loggers import get_logger
from state import run_subscribers

logger = get_logger(__name__)

# Comfortably past a reload and the follower's longest reconnect backoff, short enough that a run
# the user meant to stop does not keep its model busy for long.
_GRACE_DEFAULT_S = 30.0
_CHECK_INTERVAL_S = 5.0


def _grace_from_env() -> float:
    """Read the grace once, at import. A value nobody can parse falls back rather than raising on the chat path."""
    raw = os.environ.get("UNSLOTH_STUDIO_UNATTENDED_RUN_GRACE_S")
    if raw is None or not raw.strip():
        return _GRACE_DEFAULT_S
    try:
        value = float(raw)
    except ValueError:
        return _GRACE_DEFAULT_S
    if not math.isfinite(value) or value < 0:
        return _GRACE_DEFAULT_S
    return value


_GRACE_S = _grace_from_env()


@dataclass
class _Watch:
    attendance_id: str
    account_id: str
    # Blocking (it reads the run's database), so it is called off the loop.
    settled: Callable[[], bool]
    # Runs on the serving loop: a supervisor's cancel can touch its asyncio tasks.
    cancel: Callable[[], Awaitable[None]]
    unattended_since: float


_LOCK = threading.Lock()
# (account_id, attendance_id) -> watch
_WATCHES: dict[tuple[str, str], _Watch] = {}
_task: asyncio.Task | None = None


def _key(account_id: str, attendance_id: str) -> tuple[str, str]:
    return (account_id or "", attendance_id)


def watch(
    attendance_id: str,
    account_id: str = "",
    *,
    settled: Callable[[], bool],
    cancel: Callable[[], Awaitable[None]],
) -> None:
    """Stop ``attendance_id`` once it has gone ``_GRACE_S`` with no follower.

    ``attendance_id`` is the id its followers stamp in ``run_subscribers``. Idempotent: a run
    already watched keeps the watch it has, so re-arming on every reattach cannot reset its grace.
    Safe from any thread; nothing here touches the loop.
    """
    if not attendance_id:
        return
    key = _key(account_id, attendance_id)
    with _LOCK:
        if key in _WATCHES:
            return
        _WATCHES[key] = _Watch(
            attendance_id = attendance_id,
            account_id = account_id or "",
            settled = settled,
            cancel = cancel,
            unattended_since = time.monotonic(),
        )


def _drop(key: tuple[str, str], entry: _Watch) -> None:
    with _LOCK:
        # Only this entry: a run re-armed after it settled holds a newer one under the same key.
        if _WATCHES.get(key) is entry:
            del _WATCHES[key]


async def check_once(now: float | None = None) -> list[str]:
    """One pass over the watched runs. Returns the attendance ids it stopped."""
    now = time.monotonic() if now is None else now
    with _LOCK:
        entries = list(_WATCHES.items())
    stopped: list[str] = []
    for key, entry in entries:
        try:
            if await asyncio.to_thread(entry.settled):
                _drop(key, entry)
                continue
        except Exception as exc:
            # A locked database or a deleted account must not end the watch for good.
            logger.warning("Unattended run check failed for %s: %s", entry.attendance_id, exc)
            continue
        if run_subscribers.is_attended(entry.attendance_id, entry.account_id):
            entry.unattended_since = now
            continue
        if now - entry.unattended_since < _GRACE_S:
            continue
        _drop(key, entry)
        try:
            await entry.cancel()
        except Exception as exc:
            logger.warning("Stopping unattended run %s failed: %s", entry.attendance_id, exc)
            continue
        logger.info(
            "Stopped run %s: no tab has watched it for %.0fs", entry.attendance_id, _GRACE_S
        )
        stopped.append(entry.attendance_id)
    return stopped


async def _run() -> None:
    while True:
        await asyncio.sleep(_CHECK_INTERVAL_S)
        try:
            await check_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Unattended run watchdog pass failed: %s", exc)


def start() -> None:
    """Start the watchdog on the running loop. Idempotent within one lifespan."""
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.create_task(_run(), name = "unattended-run-watchdog")


async def stop() -> None:
    global _task
    task, _task = _task, None
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def watched_for_tests(attendance_id: str, account_id: str = "") -> bool:
    with _LOCK:
        return _key(account_id, attendance_id) in _WATCHES


def reset_for_tests() -> None:
    with _LOCK:
        _WATCHES.clear()
