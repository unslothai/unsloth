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


ADMISSION_CONTROL_ENV = "UNSLOTH_OPENAI_COMPAT_ADMISSION_CONTROL"
ADMISSION_QUEUE_TIMEOUT_ENV = "UNSLOTH_OPENAI_COMPAT_ADMISSION_QUEUE_TIMEOUT"
ADMISSION_KEEPALIVE_INTERVAL_ENV = "UNSLOTH_OPENAI_COMPAT_ADMISSION_KEEPALIVE_INTERVAL"
ADMISSION_MAX_QUEUE_ENV = "UNSLOTH_OPENAI_COMPAT_ADMISSION_MAX_QUEUE"

DEFAULT_ADMISSION_ENABLED = True
DEFAULT_ADMISSION_QUEUE_TIMEOUT_S = None
DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S = 5.0
DEFAULT_ADMISSION_MAX_QUEUE = 64


@dataclass(frozen = True)
class LlamaAdmissionConfig:
    enabled: bool = DEFAULT_ADMISSION_ENABLED
    queue_timeout_s: Optional[float] = DEFAULT_ADMISSION_QUEUE_TIMEOUT_S
    keepalive_interval_s: float = DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S
    max_queue: Optional[int] = DEFAULT_ADMISSION_MAX_QUEUE


@dataclass(frozen = True)
class LlamaAdmissionSnapshot:
    key: str
    capacity: int
    active: int
    queued: int


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


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _optional_positive_float_env(name: str, default: Optional[float]) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else None


def _positive_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _optional_positive_int_env(name: str, default: Optional[int]) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else None


def llama_admission_config_from_env() -> LlamaAdmissionConfig:
    return LlamaAdmissionConfig(
        enabled = _bool_env(ADMISSION_CONTROL_ENV, DEFAULT_ADMISSION_ENABLED),
        queue_timeout_s = _optional_positive_float_env(
            ADMISSION_QUEUE_TIMEOUT_ENV,
            DEFAULT_ADMISSION_QUEUE_TIMEOUT_S,
        ),
        keepalive_interval_s = _positive_float_env(
            ADMISSION_KEEPALIVE_INTERVAL_ENV,
            DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S,
        ),
        max_queue = _optional_positive_int_env(
            ADMISSION_MAX_QUEUE_ENV,
            DEFAULT_ADMISSION_MAX_QUEUE,
        ),
    )


@dataclass
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    cancelled: bool = False
    granted_lease: Optional["LlamaAdmissionLease"] = None


class LlamaAdmissionLease:
    def __init__(self, queue: Optional["LlamaAdmissionQueue"]):
        self._queue = queue
        self._released = False
        self._release_lock = threading.Lock()

    def release(self) -> None:
        queue = None
        with self._release_lock:
            if self._released:
                return
            self._released = True
            queue = self._queue
        if queue is not None:
            queue.release()

    async def __aenter__(self) -> "LlamaAdmissionLease":
        return self

    async def __aexit__(self, *_args) -> None:
        self.release()


class LlamaAdmissionReservation:
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

    @property
    def queue(self) -> Optional["LlamaAdmissionQueue"]:
        """The queue this reservation was taken from.

        Callers that park and unpark must use this rather than re-resolving by
        key: queues are keyed by ``base_url``, which carries a fresh ephemeral
        port on every model load, so a reload between park and unpark would
        unpark a different queue and hand a stolen slot to whoever holds it.
        """
        return self._queue

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
    def __init__(self, key: str):
        self.key = key
        self._lock = threading.Lock()
        self._active = 0
        self._capacity = 1
        # Holders parked on a tool approval prompt keep their lease but must not count
        # against capacity, or unanswered prompts wedge every other chat.
        self._parked = 0
        # FIFO tickets for holders resuming from a park (see unpark_async). A bare count
        # deadlocked: every approved holder blocked every other one.
        self._unpark_tickets: Deque[int] = deque()
        self._unpark_seq = 0
        self._waiters: Deque[_Waiter] = deque()

    def reserve(self, *, capacity: int, config: LlamaAdmissionConfig) -> LlamaAdmissionReservation:
        capacity = max(1, int(capacity or 1))
        if not config.enabled:
            return LlamaAdmissionReservation(
                queue = None,
                lease = LlamaAdmissionLease(None),
                snapshot = LlamaAdmissionSnapshot(self.key, capacity, 0, 0),
            )

        loop = asyncio.get_running_loop()
        with self._lock:
            self._capacity = capacity
            self._prune_waiters_locked()
            self._grant_waiters_locked()
            # Same reservation as _grant_waiters_locked: a parked chat holding an unpark ticket
            # has already been promised the next slot, and without counting those here a stream
            # of new arrivals took it first, forever.
            if (
                self._active - self._parked + len(self._unpark_tickets)
            ) < self._capacity and not self._waiters:
                self._active += 1
                return LlamaAdmissionReservation(
                    queue = self,
                    lease = LlamaAdmissionLease(self),
                    snapshot = self._snapshot_locked(),
                )
            if config.max_queue is not None and len(self._waiters) >= config.max_queue:
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
                snapshot = self._snapshot_locked(),
            )

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1
            self._grant_waiters_locked()

    def park(self) -> None:
        """Free this holder's slot while it waits on something outside the GPU."""
        with self._lock:
            self._parked += 1
            self._grant_waiters_locked()

    def unpark(self) -> None:
        """Give up the parked state without reclaiming a slot.

        For a holder that is tearing down: it will not decode again, and its
        lease is released separately. Resuming holders must use unpark_async,
        which waits for room instead of pushing the count over capacity.
        """
        with self._lock:
            if self._parked > 0:
                self._parked -= 1

    async def unpark_async(
        self,
        *,
        cancel_event = None,
        poll_s: float = 0.02,
    ) -> None:
        """Take the slot back, waiting until capacity allows it.

        park() hands the freed slot to a waiter, so by the time the user answers
        an approval prompt someone else may be decoding in it. Decrementing the
        counter regardless put two holders on a one-slot server and the resumed
        tool loop went straight back to llama-server, past the admission limit.

        Gives up waiting if the caller is cancelled, since the holder is then
        leaving anyway and must not be stuck in this loop.
        """
        with self._lock:
            if self._parked <= 0:
                return
            # Take a ticket, so arrivals after the approval queue behind this holder. Ordered,
            # not counted: a count made every approved holder block every other, and with
            # nothing decoding it never resolved.
            self._unpark_seq += 1
            ticket = self._unpark_seq
            self._unpark_tickets.append(ticket)
        try:
            while True:
                with self._lock:
                    if self._parked <= 0:
                        return
                    ahead = 0
                    for queued in self._unpark_tickets:
                        if queued == ticket:
                            break
                        ahead += 1
                    # Effective count is (_active - _parked); un-parking adds one, and so
                    # would every approval ahead of this one.
                    if (self._active - self._parked + ahead) < self._capacity:
                        self._parked -= 1
                        return
                    if cancel_event is not None and cancel_event.is_set():
                        self._parked -= 1
                        return
                await asyncio.sleep(poll_s)
        finally:
            with self._lock:
                try:
                    self._unpark_tickets.remove(ticket)
                except ValueError:
                    pass

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
            return self._active == 0 and not self._waiters

    def _grant_waiters_locked(self) -> None:
        self._prune_waiters_locked()
        # A pending unpark reserves a slot for a holder that has been approved and is waiting
        # to resume. Without it, release() grants the freed slot to the next waiter under this
        # same lock, so an approved chat is overtaken by every later arrival and starves.
        while (
            self._waiters
            and (self._active - self._parked + len(self._unpark_tickets)) < self._capacity
        ):
            waiter = self._waiters.popleft()
            if waiter.cancelled or waiter.future.done():
                continue
            self._active += 1
            lease = LlamaAdmissionLease(self)
            waiter.granted_lease = lease
            waiter.loop.call_soon_threadsafe(self._deliver_lease, waiter, lease)

    def _deliver_lease(self, waiter: _Waiter, lease: LlamaAdmissionLease) -> None:
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
        self._waiters = deque(
            waiter for waiter in self._waiters if not waiter.cancelled and not waiter.future.done()
        )

    def _snapshot_locked(self) -> LlamaAdmissionSnapshot:
        return LlamaAdmissionSnapshot(
            key = self.key,
            capacity = self._capacity,
            active = self._active,
            queued = len(self._waiters),
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
