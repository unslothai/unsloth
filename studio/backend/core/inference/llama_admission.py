# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Admission control for local llama-server generation requests.

The helpers in this module deliberately know nothing about FastAPI, SSE, or the
OpenAI-compatible route shape. They only coordinate how many upstream generation
requests may be active for one llama-server backend and provide a cancellable
FIFO queue for excess requests.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


ADMISSION_CONTROL_ENV = "UNSLOTH_LLAMA_ADMISSION_CONTROL"
ADMISSION_QUEUE_TIMEOUT_ENV = "UNSLOTH_LLAMA_ADMISSION_QUEUE_TIMEOUT"
ADMISSION_KEEPALIVE_INTERVAL_ENV = "UNSLOTH_LLAMA_ADMISSION_KEEPALIVE_INTERVAL"
ADMISSION_MAX_QUEUE_ENV = "UNSLOTH_LLAMA_ADMISSION_MAX_QUEUE"
ADMISSION_QUEUE_PER_SLOT_ENV = "UNSLOTH_LLAMA_ADMISSION_QUEUE_PER_SLOT"

# The UNSLOTH_OPENAI_COMPAT_* spellings predate this queue being shared with the
# Anthropic /v1/messages route (same llama-server slots). Still honored; the
# neutral name above wins when both are set.
_LEGACY_ENV = {
    ADMISSION_CONTROL_ENV: "UNSLOTH_OPENAI_COMPAT_ADMISSION_CONTROL",
    ADMISSION_QUEUE_TIMEOUT_ENV: "UNSLOTH_OPENAI_COMPAT_ADMISSION_QUEUE_TIMEOUT",
    ADMISSION_KEEPALIVE_INTERVAL_ENV: "UNSLOTH_OPENAI_COMPAT_ADMISSION_KEEPALIVE_INTERVAL",
    ADMISSION_MAX_QUEUE_ENV: "UNSLOTH_OPENAI_COMPAT_ADMISSION_MAX_QUEUE",
}

DEFAULT_ADMISSION_ENABLED = True
# None: a queued request waits for its slot indefinitely rather than timing out.
DEFAULT_ADMISSION_QUEUE_TIMEOUT_S = None
DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S = 5.0
# None: no absolute cap, the wait line is sized from the pool instead.
DEFAULT_ADMISSION_MAX_QUEUE = None
# Wait line = 16 x the serving slots, so it tracks --parallel (4 slots -> 64
# waiters, 8 -> 128). Purely a memory guard; waiting itself is never timed out.
DEFAULT_ADMISSION_QUEUE_PER_SLOT = 16
# Floor for the scaled line, so a 1-slot backend (plain `unsloth studio`, or any
# load downshifted to fit VRAM) keeps the depth it had before scaling existed
# rather than dropping to 16 and rejecting callers that used to queue.
DEFAULT_ADMISSION_MIN_QUEUE = 64


@dataclass(frozen = True)
class LlamaAdmissionConfig:
    enabled: bool = DEFAULT_ADMISSION_ENABLED
    queue_timeout_s: Optional[float] = DEFAULT_ADMISSION_QUEUE_TIMEOUT_S
    keepalive_interval_s: float = DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S
    max_queue: Optional[int] = DEFAULT_ADMISSION_MAX_QUEUE
    queue_per_slot: Optional[int] = DEFAULT_ADMISSION_QUEUE_PER_SLOT
    # Only floors the default multiplier, so an operator who sets QUEUE_PER_SLOT
    # deliberately gets exactly the depth they asked for. None disables it.
    min_queue: Optional[int] = DEFAULT_ADMISSION_MIN_QUEUE

    def queue_limit(self, capacity: int) -> Optional[int]:
        """How many callers may line up for a pool of ``capacity`` slots.

        An explicit ``max_queue`` wins; otherwise the line scales with the slots
        so it follows ``--parallel``. The default multiplier is floored, so a
        1-slot backend does not end up shallower than it was before scaling. None
        (or any non-positive setting) means an unbounded line.
        """
        if self.max_queue is not None:
            return self.max_queue if self.max_queue > 0 else None
        if not self.queue_per_slot or self.queue_per_slot <= 0:
            return None
        scaled = self.queue_per_slot * max(1, capacity)
        return max(self.min_queue, scaled) if self.min_queue else scaled


@dataclass(frozen = True)
class LlamaAdmissionSnapshot:
    key: str
    capacity: int
    active: int
    queued: int
    free: int = 0


class LlamaAdmissionError(Exception):
    def __init__(
        self,
        message: str,
        *,
        snapshot: Optional[LlamaAdmissionSnapshot] = None,
    ):
        super().__init__(message)
        self.snapshot = snapshot


class LlamaAdmissionQueueFull(LlamaAdmissionError):
    pass


class LlamaAdmissionTimeout(LlamaAdmissionError):
    pass


class LlamaAdmissionCancelled(LlamaAdmissionError):
    pass


def _raw_env(name: str) -> Optional[str]:
    """Value for a canonical name, falling back to its legacy spelling."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        legacy = _LEGACY_ENV.get(name)
        value = os.environ.get(legacy) if legacy else None
    return value


def _bool_env(name: str, default: bool) -> bool:
    value = _raw_env(name)
    if value is None or not value.strip():
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _optional_positive_float_env(name: str, default: Optional[float]) -> Optional[float]:
    value = _raw_env(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else None


def _positive_float_env(name: str, default: float) -> float:
    value = _raw_env(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _optional_positive_int_env(name: str, default: Optional[int]) -> Optional[int]:
    value = _raw_env(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else None


def _queue_limits_from_env() -> tuple[Optional[int], Optional[int], Optional[int]]:
    """(max_queue, queue_per_slot, min_queue) from the environment.

    An absolute MAX_QUEUE wins outright; MAX_QUEUE=0 asks for an unbounded line.
    Unset leaves the per-slot multiplier in charge (itself 0 for unbounded). The
    floor applies only to the default multiplier: setting QUEUE_PER_SLOT means
    the operator wants that exact depth, however shallow.
    """
    raw_per_slot = _raw_env(ADMISSION_QUEUE_PER_SLOT_ENV)
    explicit_per_slot = bool(raw_per_slot and raw_per_slot.strip())
    per_slot = _optional_positive_int_env(
        ADMISSION_QUEUE_PER_SLOT_ENV,
        DEFAULT_ADMISSION_QUEUE_PER_SLOT,
    )
    min_queue = None if explicit_per_slot else DEFAULT_ADMISSION_MIN_QUEUE
    raw = _raw_env(ADMISSION_MAX_QUEUE_ENV)
    if raw is None or not raw.strip():
        return None, per_slot, min_queue
    try:
        parsed = int(raw.strip())
    except ValueError:
        return None, per_slot, min_queue
    return (parsed, None, None) if parsed > 0 else (None, None, None)


def llama_admission_config_from_env() -> LlamaAdmissionConfig:
    max_queue, queue_per_slot, min_queue = _queue_limits_from_env()
    return LlamaAdmissionConfig(
        queue_per_slot = queue_per_slot,
        min_queue = min_queue,
        enabled = _bool_env(ADMISSION_CONTROL_ENV, DEFAULT_ADMISSION_ENABLED),
        queue_timeout_s = _optional_positive_float_env(
            ADMISSION_QUEUE_TIMEOUT_ENV,
            DEFAULT_ADMISSION_QUEUE_TIMEOUT_S,
        ),
        keepalive_interval_s = _positive_float_env(
            ADMISSION_KEEPALIVE_INTERVAL_ENV,
            DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S,
        ),
        max_queue = max_queue,
    )


@dataclass
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    cancelled: bool = False
    granted_lease: Optional["LlamaAdmissionLease"] = None


class LlamaAdmissionLease:
    __slots__ = ("_queue", "_slot", "_released", "_release_lock")

    def __init__(
        self,
        queue: Optional["LlamaAdmissionQueue"],
        slot: Optional[int] = None,
    ):
        self._queue = queue
        self._slot = slot
        self._released = False
        self._release_lock = threading.Lock()

    @property
    def slot(self) -> Optional[int]:
        """Pool slot this lease holds, or None when admission is disabled."""
        return self._slot

    def release(self) -> None:
        queue = None
        with self._release_lock:
            if self._released:
                return
            self._released = True
            queue = self._queue
        if queue is not None:
            queue.release(self._slot)

    async def __aenter__(self) -> "LlamaAdmissionLease":
        return self

    async def __aexit__(self, *_args) -> None:
        self.release()


class LlamaAdmissionReservation:
    __slots__ = ("_queue", "_lease", "_waiter", "snapshot")

    def __init__(
        self,
        *,
        queue: Optional["LlamaAdmissionQueue"],
        lease: Optional[LlamaAdmissionLease] = None,
        waiter: Optional[_Waiter] = None,
        snapshot: Optional[LlamaAdmissionSnapshot] = None,
    ):
        self._queue = queue
        self._lease = lease
        self._waiter = waiter
        self.snapshot = snapshot

    @property
    def is_cancelled(self) -> bool:
        return self._lease is None and self._waiter is None

    def lease_nowait(self) -> Optional[LlamaAdmissionLease]:
        if self._lease is not None:
            return self._lease
        if self._waiter is None or not self._waiter.future.done():
            return None
        if self._waiter.future.cancelled():
            self._waiter.cancelled = True
            self._waiter = None
            return None
        self._lease = self._waiter.future.result()
        self._waiter = None
        return self._lease

    async def wait(self, timeout_s: float) -> Optional[LlamaAdmissionLease]:
        """Wait up to ``timeout_s`` for a slot.

        A timeout leaves this reservation queued so the caller can poll again.
        Any exit that abandons the wait for good must call ``cancel()``, or the
        slot granted later is delivered to a future nobody reads and is never
        released.
        """
        lease = self.lease_nowait()
        if lease is not None:
            return lease
        if self._waiter is None:
            return None
        waiter = self._waiter
        try:
            await asyncio.wait_for(asyncio.shield(waiter.future), timeout = timeout_s)
        except asyncio.CancelledError:
            if waiter.future.cancelled():
                waiter.cancelled = True
                if self._waiter is waiter:
                    self._waiter = None
                return None
            raise
        return self.lease_nowait()

    def cancel(self) -> None:
        lease = self.lease_nowait()
        if lease is not None:
            lease.release()
            self._lease = None
            return
        if self._queue is not None and self._waiter is not None:
            self._queue.cancel(self._waiter)
        self._waiter = None

    def snapshot_now(self) -> Optional[LlamaAdmissionSnapshot]:
        if self._queue is None:
            return self.snapshot
        return self._queue.snapshot()


class LlamaAdmissionQueue:
    """A fixed pool of generation slots for one llama-server, plus a FIFO wait line.

    The pool mirrors llama-server's own ``--parallel`` slots: ``capacity`` slot ids
    are each either free or held by exactly one caller. A caller that finds every
    slot busy waits in arrival order and is handed the next slot to free, so the
    backend never sees more concurrent generations than it has slots and no caller
    can be starved. Waiting is unbounded in time by default (``queue_timeout_s``
    None); the wait line itself is bounded, and only how many may line up before
    new arrivals are rejected. By default that is ``16 x slots`` floored at 64,
    not unlimited: an unbounded line takes ``max_queue`` or ``queue_per_slot``
    set to 0. See ``LlamaAdmissionConfig.queue_limit``.
    """

    __slots__ = ("key", "_lock", "_capacity", "_free", "_in_use", "_held", "_waiters")

    def __init__(self, key: str):
        self.key = key
        self._lock = threading.Lock()
        self._capacity = 1
        self._free: list[int] = [0]
        # Held slots as a bitmask: one int instead of a set, so the pool costs the
        # same whether it is idle or saturated. _held is its popcount, kept as a
        # counter because int.bit_count() is 3.10+ and this package targets 3.9.
        self._in_use = 0
        self._held = 0
        self._waiters: Deque[_Waiter] = deque()

    def _resize_pool_locked(self, capacity: int) -> None:
        # Slots past a shrunk capacity retire when their holder releases them.
        if capacity == self._capacity:
            return
        self._capacity = capacity
        self._free = [slot for slot in range(capacity) if not self._in_use >> slot & 1]

    def _can_admit_locked(self) -> bool:
        # Slots still held above a shrunk capacity keep occupying the backend, so
        # count every held slot against the ceiling, not just the ids below it.
        return bool(self._free) and self._held < self._capacity

    def _take_slot_locked(self) -> Optional[int]:
        if not self._can_admit_locked():
            return None
        slot = self._free.pop()
        self._in_use |= 1 << slot
        self._held += 1
        return slot

    def reserve(self, *, capacity: int, config: LlamaAdmissionConfig) -> LlamaAdmissionReservation:
        capacity = max(1, int(capacity or 1))
        if not config.enabled:
            return LlamaAdmissionReservation(
                queue = None,
                lease = LlamaAdmissionLease(None),
                snapshot = LlamaAdmissionSnapshot(self.key, capacity, 0, 0, capacity),
            )

        loop = asyncio.get_running_loop()
        with self._lock:
            self._resize_pool_locked(capacity)
            self._grant_waiters_locked()
            if not self._waiters:
                slot = self._take_slot_locked()
                if slot is not None:
                    # No snapshot here: callers read it through snapshot_now(),
                    # which re-reads the queue, so building one per admitted
                    # request would be pure allocation on the hot path.
                    return LlamaAdmissionReservation(
                        queue = self,
                        lease = LlamaAdmissionLease(self, slot),
                    )
            limit = config.queue_limit(self._capacity)
            if limit is not None and self._live_waiters_locked() >= limit:
                raise LlamaAdmissionQueueFull(
                    "llama-server generation queue is full",
                    snapshot = self._snapshot_locked(),
                )
            waiter = _Waiter(
                loop = loop,
                future = loop.create_future(),
            )
            self._waiters.append(waiter)
            return LlamaAdmissionReservation(
                queue = self,
                waiter = waiter,
            )

    def _release_slot_locked(self, slot: Optional[int]) -> None:
        # A slot id at or past a shrunk capacity retires instead of returning.
        if slot is None or not self._in_use >> slot & 1:
            return
        self._in_use &= ~(1 << slot)
        self._held -= 1
        if slot < self._capacity:
            self._free.append(slot)

    def release(self, slot: Optional[int]) -> None:
        with self._lock:
            self._release_slot_locked(slot)
            self._grant_waiters_locked()

    def cancel(self, waiter: _Waiter) -> None:
        lease_to_release = None
        with self._lock:
            waiter.cancelled = True
            try:
                self._waiters.remove(waiter)
            except ValueError:
                pass
            if waiter.granted_lease is not None:
                lease_to_release = waiter.granted_lease
                waiter.granted_lease = None
            if not waiter.future.done():
                waiter.loop.call_soon_threadsafe(waiter.future.cancel)
        if lease_to_release is not None:
            lease_to_release.release()

    def snapshot(self) -> LlamaAdmissionSnapshot:
        with self._lock:
            self._prune_waiters_locked()
            return self._snapshot_locked()

    def is_idle(self) -> bool:
        with self._lock:
            self._prune_waiters_locked()
            return self._in_use == 0 and not self._waiters

    def _grant_waiters_locked(self) -> None:
        # Dead waiters are skipped as they are popped, so no prune is needed here.
        while self._waiters and self._can_admit_locked():
            waiter = self._waiters.popleft()
            if waiter.cancelled or waiter.future.done():
                continue
            slot = self._take_slot_locked()
            lease = LlamaAdmissionLease(self, slot)
            waiter.granted_lease = lease
            try:
                waiter.loop.call_soon_threadsafe(self._deliver_lease, waiter, lease)
            except RuntimeError:
                # Waiter's loop is gone. Reclaim the slot; leaving the bit set
                # would strand it, since _free is rebuilt from the bitmask.
                waiter.granted_lease = None
                self._release_slot_locked(slot)

    def _deliver_lease(self, waiter: _Waiter, lease: LlamaAdmissionLease) -> None:
        # Runs on the waiter's own loop thread, which is also the only thread that
        # cancels that reservation, so waiter state is safe to touch unlocked here.
        # release() may be called from any thread, but only reaches this via
        # call_soon_threadsafe. Cancelling off-loop would need this under _lock.
        if waiter.cancelled or waiter.future.done():
            waiter.granted_lease = None
            if not waiter.future.done():
                waiter.future.cancel()
            lease.release()
            return
        try:
            waiter.future.set_result(lease)
            waiter.granted_lease = None
        except asyncio.InvalidStateError:
            waiter.granted_lease = None
            lease.release()

    def _prune_waiters_locked(self) -> None:
        # Rebuilding the deque on every reserve/release dominated the hot path, so
        # only pay it when a waiter actually died out of band (an externally
        # cancelled future); cancel() already drops its own waiter eagerly.
        for waiter in self._waiters:
            if waiter.cancelled or waiter.future.done():
                break
        else:
            return
        self._waiters = deque(
            waiter for waiter in self._waiters if not waiter.cancelled and not waiter.future.done()
        )

    def _live_waiters_locked(self) -> int:
        self._prune_waiters_locked()
        return len(self._waiters)

    def _snapshot_locked(self) -> LlamaAdmissionSnapshot:
        return LlamaAdmissionSnapshot(
            key = self.key,
            capacity = self._capacity,
            active = self._held,
            queued = len(self._waiters),
            # What another caller could actually take, so the admission log never
            # shows free slots next to queued requests: after a shrink, ids below
            # the new capacity can be free while holdovers still fill the ceiling.
            free = min(len(self._free), max(0, self._capacity - self._held)),
        )


_QUEUES_LOCK = threading.Lock()
_QUEUES: dict[str, LlamaAdmissionQueue] = {}


def get_llama_admission_queue(key: str) -> LlamaAdmissionQueue:
    with _QUEUES_LOCK:
        queue = _QUEUES.get(key)
        if queue is None:
            queue = LlamaAdmissionQueue(key)
            _QUEUES[key] = queue
            # base_url carries a fresh ephemeral port on every model load, so
            # each load registers a new key. Drop the now-idle queues from prior
            # loads so the registry can't grow without bound on a long-running
            # server. Queues with in-flight requests are kept until they drain.
            for stale_key in [k for k in _QUEUES if k != key and _QUEUES[k].is_idle()]:
                del _QUEUES[stale_key]
        return queue


def reset_llama_admission_queues() -> None:
    with _QUEUES_LOCK:
        _QUEUES.clear()
