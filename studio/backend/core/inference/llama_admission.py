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
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


# dataclass(slots = True) halves per-instance overhead. Measured as perf-neutral
# here, not a speed win: it costs a little on construction and gains it back on
# access. It is 3.10+ and this package declares >=3.9, so gate it rather than
# dropping it outright. Empty on 3.9 means a plain dataclass.
_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


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


def _executor_workers() -> int:
    """Threads asyncio's default executor runs to_thread work on.

    Mirrors ThreadPoolExecutor's own default sizing, which is what
    ``run_in_executor(None, ...)`` builds. 3.13 sizes it from
    ``process_cpu_count()``, which honours CPU affinity and cgroup quotas;
    ``cpu_count()`` would budget from the whole host inside a one-core container.
    """
    cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return min(32, cpus + 4)


def _executor_reserve(workers: int) -> int:
    """Threads kept clear of parked approvals, for generation steps, stream
    teardown and unrelated to_thread work. Scaled rather than flat: a flat count
    would leave a 5-worker executor (one usable CPU) no budget at all.
    """
    return max(2, workers // 8)


def _max_parked(capacity: int) -> int:
    """How many holders may sit on an approval prompt with their slot given back.

    A pending prompt parks an executor thread (the loop blocks inside
    to_thread(next, gen)) whether or not it parked its slot, the pool already
    permits `capacity` of those, and every park admits one more, so budget only
    what the executor has left over. Zero on a backend whose --parallel alone
    fills it: the prompt then holds its slot, as it did before parking existed.
    """
    workers = _executor_workers()
    spare = workers - _executor_reserve(workers) - max(0, capacity)
    # A quarter of the executor, floored at two while `spare` allows: a quarter of
    # five is one, and one park cannot cover the two simultaneous prompts #7455
    # exists for.
    return max(0, min(max(2, workers // 4), spare))


# Process-wide, not per queue: there is one executor, and base_url takes a fresh
# port on every load, so a per-queue budget would hand the same allowance to each
# backend and to every reload, blind to the approvals parked on the old queue.
_PARK_LOCK = threading.Lock()
_parked_total = 0


def _claim_park(limit: int) -> bool:
    global _parked_total
    with _PARK_LOCK:
        if _parked_total >= limit:
            return False
        _parked_total += 1
        return True


def _drop_park() -> None:
    global _parked_total
    with _PARK_LOCK:
        _parked_total = max(0, _parked_total - 1)


def _live_capacity(current: "LlamaAdmissionQueue") -> int:
    """Slots across every backend still serving requests.

    One queue's capacity is the wrong denominator for a budget sized against the
    one executor: a reload drains the old queue alongside the new one, and
    prompts on both park threads. Idle queues hold nothing and are about to be
    evicted.
    """
    with _QUEUES_LOCK:
        queues = list(_QUEUES.values())
    # is_idle takes each queue's own lock, so never while holding _QUEUES_LOCK.
    total = sum(queue._capacity for queue in queues if queue is current or not queue.is_idle())
    return total if any(queue is current for queue in queues) else total + current._capacity


@dataclass(frozen = True, **_SLOTS)
class LlamaAdmissionConfig:
    enabled: bool = DEFAULT_ADMISSION_ENABLED
    queue_timeout_s: Optional[float] = DEFAULT_ADMISSION_QUEUE_TIMEOUT_S
    keepalive_interval_s: float = DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S
    max_queue: Optional[int] = DEFAULT_ADMISSION_MAX_QUEUE
    queue_per_slot: Optional[int] = DEFAULT_ADMISSION_QUEUE_PER_SLOT
    # Unconditional floor on the scaled line. The env path clears it when the
    # operator sets QUEUE_PER_SLOT, so only the default multiplier is floored.
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


@dataclass(frozen = True, **_SLOTS)
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


def _queue_limits_from_env() -> tuple[Optional[int], Optional[int], Optional[int]]:
    """(max_queue, queue_per_slot, min_queue) from the environment.

    An absolute MAX_QUEUE wins outright; MAX_QUEUE=0 asks for an unbounded line.
    Unset leaves the per-slot multiplier in charge (itself 0 for unbounded). The
    floor applies only to the default multiplier: setting QUEUE_PER_SLOT means
    the operator wants that exact depth, however shallow.
    """
    # Explicit means it parsed, not just that something was set: a typo falls back
    # to the default multiplier, so it has to keep the default's floor too.
    raw_per_slot = _raw_env(ADMISSION_QUEUE_PER_SLOT_ENV)
    try:
        per_slot = int((raw_per_slot or "").strip())
    except ValueError:
        per_slot, min_queue = DEFAULT_ADMISSION_QUEUE_PER_SLOT, DEFAULT_ADMISSION_MIN_QUEUE
    else:
        per_slot, min_queue = (per_slot if per_slot > 0 else None), None
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


@dataclass(**_SLOTS)
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    cancelled: bool = False
    granted_lease: Optional["LlamaAdmissionLease"] = None


class LlamaAdmissionLease:
    __slots__ = ("_queue", "_slot", "_released", "_release_lock", "_parked", "_budgeted")

    def __init__(
        self,
        queue: Optional["LlamaAdmissionQueue"],
        slot: Optional[int] = None,
    ):
        self._queue = queue
        self._slot = slot
        self._released = False
        self._release_lock = threading.Lock()
        self._parked = False
        self._budgeted = False

    @property
    def slot(self) -> Optional[int]:
        """Pool slot this lease holds, or None when admission is disabled."""
        return self._slot

    def park(self) -> bool:
        """Hand the slot back while this holder waits on something off the GPU.

        A run stopped on a tool approval prompt is not decoding, so holding its
        slot would let unanswered prompts fill the pool while llama-server idles.
        The lease itself stays valid: releasing it after a park is still correct.

        False when the park budget is spent and nothing was given back: the
        caller keeps its slot across the prompt, as it did before parking
        existed. Slower for whoever is behind it, but each freed slot admits
        another run that can park too, on the executor the generators run on.
        """
        queue = self._queue
        with self._release_lock:
            if queue is None or self._released or self._parked:
                return False
            # Under the lease lock so the decision and the handover cannot split.
            # Nothing takes the queue lock then a lease lock, so this order is
            # the only one in play.
            if not queue.try_park(self._slot):
                return False
            self._parked = True
            self._budgeted = True
            self._slot = None
        return True

    def _drop_budget(self) -> None:
        """Give the executor budget back now the prompt wait is over.

        Separate from the queue's parked count, which lasts until the slot is
        back: the executor thread is free the moment the answer arrives. Holding
        the budget until the resume lands would refuse someone else's park for a
        finished wait, and that someone holds the slot the resumer wants.
        """
        with self._release_lock:
            if not self._budgeted:
                return
            self._budgeted = False
        _drop_park()

    def unpark(self) -> None:
        """Drop the parked state without reclaiming a slot.

        For a holder that is tearing down: it will not decode again. Resuming
        holders must use ``unpark_async``, which waits for a slot instead of
        going back to llama-server past the admission limit.
        """
        with self._release_lock:
            if not self._parked:
                return
            self._parked = False
        self._drop_budget()
        if self._queue is not None:
            self._queue.unpark()

    async def unpark_async(
        self,
        *,
        cancel_event = None,
        poll_s: float = 0.02,
    ) -> None:
        """Take a slot back, waiting until the pool has room.

        ``park`` gave the slot to a waiter, so by the time the user answers the
        prompt someone else may be decoding in it. Resuming regardless put two
        holders on a one-slot server. Gives up if the caller is cancelled, since
        the holder is then leaving anyway and must not be stuck here.
        """
        queue = self._queue
        if queue is None or not self._parked:
            return
        # Before the wait, not after: the prompt is answered, so this holder is
        # already off the executor and must not keep anyone else off it.
        self._drop_budget()
        slot = await queue.acquire_parked_slot(cancel_event = cancel_event, poll_s = poll_s)
        stranded = None
        with self._release_lock:
            # release() may have run during the wait; it clears the flag and does
            # the unpark itself, so only the caller that clears it here repeats one.
            parked, self._parked = self._parked, False
            if self._released:
                # Released while waiting: this lease will never hand the slot
                # back, so return it here rather than strand it for good.
                stranded = slot
            else:
                self._slot = slot
        if parked:
            queue.unpark()
        if stranded is not None:
            queue.release(stranded)

    def release(self) -> None:
        queue = None
        parked = False
        with self._release_lock:
            if self._released:
                return
            self._released = True
            queue = self._queue
            parked, self._parked = self._parked, False
        self._drop_budget()
        if queue is not None:
            if parked:
                queue.unpark()
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
    slot busy waits in arrival order and is handed the next slot to free, so no
    caller is starved. This bounds only the callers that reserve: chat completions
    and messages do, while /v1/completions, Studio's own chat endpoint and RAG
    captioning all reach llama-server directly, so it is not a global cap.
    Waiting is unbounded in time by default (``queue_timeout_s``
    None); the wait line itself is bounded, and only how many may line up before
    new arrivals are rejected. By default that is ``16 x slots`` floored at 64,
    not unlimited: an unbounded line takes ``max_queue`` or ``queue_per_slot``
    set to 0. See ``LlamaAdmissionConfig.queue_limit``.
    """

    __slots__ = (
        "key",
        "_lock",
        "_capacity",
        "_free",
        "_in_use",
        "_held",
        "_waiters",
        "_parked",
        "_unpark_tickets",
        "_unpark_seq",
    )

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
        # Holders parked on a tool approval prompt. They hold no slot, so this only
        # keeps the queue off the idle-eviction list while they are away.
        self._parked = 0
        # FIFO tickets for holders resuming from a park (see acquire_parked_slot). A
        # bare count deadlocked: every approved holder blocked every other one.
        self._unpark_tickets: Deque[int] = deque()
        self._unpark_seq = 0

    def _resize_pool_locked(self, capacity: int) -> None:
        # Slots past a shrunk capacity retire when their holder releases them.
        if capacity == self._capacity:
            return
        self._capacity = capacity
        self._free = [slot for slot in range(capacity) if not self._in_use >> slot & 1]

    def _can_admit_locked(self, reserved: int) -> bool:
        # Slots still held above a shrunk capacity keep occupying the backend, so
        # count every held slot against the ceiling, not just the ids below it.
        # ``reserved`` holds slots back for approved holders waiting to resume;
        # without it a stream of new arrivals took the next slot, forever.
        return bool(self._free) and (self._held + reserved) < self._capacity

    def _take_slot_locked(self, reserved: int) -> Optional[int]:
        if not self._can_admit_locked(reserved):
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
                slot = self._take_slot_locked(len(self._unpark_tickets))
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

    def try_park(self, slot: Optional[int]) -> bool:
        """Return a parked holder's slot to the pool. See ``LlamaAdmissionLease.park``.

        False leaves the slot with its holder, so a refused park costs nothing to
        undo. The per-queue count is only what ``is_idle`` reads; the budget and
        the capacity it is sized from are both process-wide.
        """
        if not _claim_park(_max_parked(_live_capacity(self))):
            return False
        with self._lock:
            self._parked += 1
            self._release_slot_locked(slot)
            self._grant_waiters_locked()
        return True

    def unpark(self) -> None:
        with self._lock:
            if self._parked > 0:
                self._parked -= 1

    async def acquire_parked_slot(
        self,
        *,
        cancel_event = None,
        poll_s: float = 0.02,
    ) -> Optional[int]:
        """Wait for a slot for a holder resuming from a park, None if cancelled.

        Ordered by ticket rather than counted, so approvals resume in the order
        they came back: counting them made every approved holder block every
        other one, and with nothing decoding that never resolved.
        """
        with self._lock:
            self._unpark_seq += 1
            ticket = self._unpark_seq
            self._unpark_tickets.append(ticket)
        try:
            while True:
                with self._lock:
                    ahead = 0
                    for queued in self._unpark_tickets:
                        if queued == ticket:
                            break
                        ahead += 1
                    # Only the approvals ahead of this one hold slots back from it.
                    slot = self._take_slot_locked(ahead)
                    if slot is not None:
                        return slot
                    if cancel_event is not None and cancel_event.is_set():
                        return None
                await asyncio.sleep(poll_s)
        finally:
            with self._lock:
                try:
                    self._unpark_tickets.remove(ticket)
                except ValueError:
                    pass
                # This ticket was holding a slot back from the wait line.
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
                try:
                    waiter.loop.call_soon_threadsafe(waiter.future.cancel)
                except RuntimeError:
                    # Loop gone. Routes call cancel() from finally blocks, so
                    # raising here would both mask their exception and skip the
                    # release below, stranding the slot for the process lifetime.
                    pass
        if lease_to_release is not None:
            lease_to_release.release()

    def snapshot(self) -> LlamaAdmissionSnapshot:
        with self._lock:
            self._prune_waiters_locked()
            return self._snapshot_locked()

    def is_idle(self) -> bool:
        with self._lock:
            self._prune_waiters_locked()
            # A parked holder owns no slot but is coming back to this queue, so
            # evicting it here would resume it against a fresh 1-slot pool.
            return self._in_use == 0 and not self._waiters and not self._parked

    def _grant_waiters_locked(self) -> None:
        # Dead waiters are skipped as they are popped, so no prune is needed here.
        while self._waiters and self._can_admit_locked(len(self._unpark_tickets)):
            waiter = self._waiters.popleft()
            if waiter.cancelled or waiter.future.done():
                continue
            slot = self._take_slot_locked(len(self._unpark_tickets))
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
            # Resume tickets are approved tool continuations holding no slot yet; omitting
            # them would show a full, idle-looking queue while one is still waiting.
            queued = len(self._waiters) + len(self._unpark_tickets),
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


def peek_llama_admission_snapshot(key: str) -> Optional[LlamaAdmissionSnapshot]:
    """Read-only view of one queue's state; never creates a queue."""
    with _QUEUES_LOCK:
        queue = _QUEUES.get(key)
    return queue.snapshot() if queue is not None else None


def reset_llama_admission_queues() -> None:
    global _parked_total
    with _QUEUES_LOCK:
        _QUEUES.clear()
    # The budget outlives the queues it was claimed against, so dropping them
    # without it leaks the count and shrinks the budget for good.
    with _PARK_LOCK:
        _parked_total = 0
