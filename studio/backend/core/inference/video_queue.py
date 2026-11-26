# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The video queue's scheduling: one order for every entry point (Video page, /v1/videos, the chat
tool), FIFO within an account and round-robin across accounts, positions and ETAs, and the wake-up
the engine sends when a clip ends. In process; lost on server restart. The entries themselves live
in routes/video.py, which starts them."""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

# (family, width, height, frames): clips of one shape cost about the same per step.
ClipKey = tuple[Optional[str], Optional[int], Optional[int], Optional[int]]


class FairOrder:
    """Round-robin turns across accounts, FIFO within one.

    An account takes its turn when it first has a waiting video (behind every account already
    waiting) and again each time one of its videos starts, so a flood from one account never pushes
    another account's waiting video back."""

    def __init__(self) -> None:
        self._turn: dict[str, int] = {}
        self._tick = itertools.count()
        self._lock = threading.Lock()

    def joined(
        self, account_id: str, waiting: Iterable[Any], account_of: Callable[[Any], str]
    ) -> None:
        """Called before an entry is added; ``waiting`` is the queue without it."""
        with self._lock:
            if not any(account_of(entry) == account_id for entry in waiting):
                self._turn[account_id] = next(self._tick)

    def started(self, account_id: Optional[str]) -> None:
        if account_id is None:
            return
        with self._lock:
            self._turn[account_id] = next(self._tick)
            # Bounded by accounts that ever queued; keep only recent ones.
            if len(self._turn) > 4096:
                for stale in sorted(self._turn, key = self._turn.get)[:1024]:
                    self._turn.pop(stale, None)

    def order(self, entries: Sequence[Any], account_of: Callable[[Any], str]) -> list[Any]:
        rounds: dict[str, int] = {}
        keyed = []
        with self._lock:
            for index, entry in enumerate(entries):
                account_id = account_of(entry)
                turn = self._turn.get(account_id)
                if turn is None:
                    turn = self._turn[account_id] = next(self._tick)
                r = rounds.get(account_id, 0)
                rounds[account_id] = r + 1
                keyed.append(((r, turn, index), entry))
        # No pruning here: readers on other threads order a snapshot, and must not drop the turn
        # an account just took; joined() renews a returning account's turn anyway.
        keyed.sort(key = lambda item: item[0])
        return [entry for _, entry in keyed]

    def reset(self) -> None:
        with self._lock:
            self._turn.clear()


@dataclass
class _Running:
    key: ClipKey
    steps: Optional[int]
    started_at: float


class ClipClock:
    """Running average seconds per step, keyed by (family, width, height, frames)."""

    ALPHA = 0.5

    def __init__(self) -> None:
        self._per_step: dict[ClipKey, float] = {}
        self._running: Optional[_Running] = None
        self._lock = threading.Lock()

    def started(self, key: ClipKey, steps: Optional[int]) -> None:
        with self._lock:
            self._running = _Running(key, steps, time.monotonic())

    def finished(
        self,
        ok: bool,
        steps: Optional[int] = None,
    ) -> None:
        with self._lock:
            running, self._running = self._running, None
            if not ok or running is None:
                return
            steps = steps or running.steps
            if not steps:
                return
            per_step = (time.monotonic() - running.started_at) / float(steps)
            old = self._per_step.get(running.key)
            self._per_step[running.key] = (
                per_step if old is None else self.ALPHA * per_step + (1 - self.ALPHA) * old
            )

    def estimate(self, key: ClipKey, steps: Optional[int]) -> Optional[float]:
        """Seconds a clip of ``key`` and ``steps`` takes; None without a clip to go on."""
        if not steps:
            return None
        with self._lock:
            per_step = self._per_step.get(key)
            if per_step is None:
                # Same family, other shape: scale by the pixels x frames ratio.
                family, width, height, frames = key
                size = (width or 0) * (height or 0) * (frames or 0)
                for (f, w, h, n), seconds in self._per_step.items():
                    other = (w or 0) * (h or 0) * (n or 0)
                    if f == family and size and other:
                        per_step = seconds * size / other
                        break
        return None if per_step is None else per_step * steps

    def remaining(self, progress: Optional[dict]) -> Optional[float]:
        """Seconds left on the running clip: the engine's own ETA while it denoises, else the
        average for its shape minus the time it has run."""
        progress = progress or {}
        if not progress.get("active"):
            return 0.0
        eta = progress.get("eta_seconds")
        if isinstance(eta, (int, float)) and eta >= 0:
            return float(eta)
        with self._lock:
            running = self._running
        if running is None:
            return None
        total = self.estimate(running.key, running.steps)
        if total is None:
            return None
        return max(0.0, total - (time.monotonic() - running.started_at))

    def reset(self) -> None:
        with self._lock:
            self._per_step.clear()
            self._running = None


def queue_view(
    ordered: Sequence[Any],
    *,
    remaining: Optional[float],
    estimate: Callable[[Any], Optional[float]],
) -> dict[int, tuple[int, Optional[float]]]:
    """id(entry) -> (1-based position, seconds until it starts). The ETA is the time left on the
    running clip plus each clip ahead's estimate; None once any of those is unknown."""
    view: dict[int, tuple[int, Optional[float]]] = {}
    eta = remaining
    for index, entry in enumerate(ordered):
        view[id(entry)] = (index + 1, None if eta is None else round(eta, 1))
        if eta is not None:
            cost = estimate(entry)
            eta = None if cost is None else eta + cost
    return view


order = FairOrder()
clock = ClipClock()

_wakers: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Event]" = (
    weakref.WeakKeyDictionary()
)
_wakers_lock = threading.Lock()


def wake_event() -> asyncio.Event:
    """This loop's wake-up: set when a clip ends, so the next video starts without a poll gap."""
    loop = asyncio.get_running_loop()
    with _wakers_lock:
        event = _wakers.get(loop)
        if event is None:
            event = _wakers[loop] = asyncio.Event()
    return event


def notify() -> None:
    """Thread-safe: the engine calls this as a clip reaches its terminal state."""
    with _wakers_lock:
        pending = list(_wakers.items())
    for loop, event in pending:
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(event.set)
        except RuntimeError:  # closed between the check and the call
            pass


def clip_started(key: ClipKey, steps: Optional[int]) -> None:
    clock.started(key, steps)


def clip_finished(ok: bool, steps: Optional[int] = None) -> None:
    clock.finished(ok, steps)
    notify()


def reset() -> None:
    """Test hook."""
    order.reset()
    clock.reset()
