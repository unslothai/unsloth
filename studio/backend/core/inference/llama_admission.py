# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Admission control for local llama-server generation requests.

These helpers deliberately know nothing about FastAPI, SSE or the OpenAI-compatible route shape.
They only coordinate how many upstream generation requests may be active for one llama-server
backend and provide a cancellable FIFO queue for excess requests.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


# slots needs 3.10+; this package supports 3.9.
_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


ADMISSION_CONTROL_ENV = "UNSLOTH_LLAMA_ADMISSION_CONTROL"
ADMISSION_QUEUE_TIMEOUT_ENV = "UNSLOTH_LLAMA_ADMISSION_QUEUE_TIMEOUT"
ADMISSION_KEEPALIVE_INTERVAL_ENV = "UNSLOTH_LLAMA_ADMISSION_KEEPALIVE_INTERVAL"
ADMISSION_MAX_QUEUE_ENV = "UNSLOTH_LLAMA_ADMISSION_MAX_QUEUE"
ADMISSION_QUEUE_PER_SLOT_ENV = "UNSLOTH_LLAMA_ADMISSION_QUEUE_PER_SLOT"
# off restores slot-only admission, for a backend whose context length mismatches its cache
ADMISSION_KV_BUDGET_ENV = "UNSLOTH_LLAMA_ADMISSION_KV_BUDGET"

# Legacy UNSLOTH_OPENAI_COMPAT_* names still honored; the neutral name wins when both are set.
_LEGACY_ENV = {
    ADMISSION_CONTROL_ENV: "UNSLOTH_OPENAI_COMPAT_ADMISSION_CONTROL",
    ADMISSION_QUEUE_TIMEOUT_ENV: "UNSLOTH_OPENAI_COMPAT_ADMISSION_QUEUE_TIMEOUT",
    ADMISSION_KEEPALIVE_INTERVAL_ENV: "UNSLOTH_OPENAI_COMPAT_ADMISSION_KEEPALIVE_INTERVAL",
    ADMISSION_MAX_QUEUE_ENV: "UNSLOTH_OPENAI_COMPAT_ADMISSION_MAX_QUEUE",
}

DEFAULT_ADMISSION_ENABLED = True
DEFAULT_ADMISSION_QUEUE_TIMEOUT_S = None
DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S = 5.0
DEFAULT_ADMISSION_MAX_QUEUE = None
# Wait line = 16 x serving slots; purely a memory guard, waiting is never timed out.
DEFAULT_ADMISSION_QUEUE_PER_SLOT = 16
# Floor so a 1-slot backend keeps its previous queue depth.
DEFAULT_ADMISSION_MIN_QUEUE = 64
# --kv-unified allocates ONE n_ctx cache but reports n_ctx to every slot, so slot-only
# admission overcommits; llama.cpp kills every colliding task.
DEFAULT_ADMISSION_KV_BUDGET = True
# Bounded because a reparker holds the wait line shut for every other caller. See recost_waiting.
DEFAULT_RECOST_WAIT_TIMEOUT_S = 300.0


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
    # Floor at two: one park cannot cover the two simultaneous prompts #7455 needs.
    return max(0, min(max(2, workers // 4), spare))


# Process-wide: one executor, and base_url changes port on every load.
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
    # Only the default multiplier is floored; the env path clears it.
    min_queue: Optional[int] = DEFAULT_ADMISSION_MIN_QUEUE
    kv_budget: bool = DEFAULT_ADMISSION_KV_BUDGET

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
    # budget 0 means token accounting is off, so committed carries no meaning.
    committed: int = 0
    budget: int = 0


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
    # Explicit means it parsed: a typo falls back to the default multiplier and its floor.
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
        kv_budget = _bool_env(ADMISSION_KV_BUDGET_ENV, DEFAULT_ADMISSION_KV_BUDGET),
    )


@dataclass(**_SLOTS)
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    cancelled: bool = False
    granted_lease: Optional["LlamaAdmissionLease"] = None
    # Read at the head of the line, so a large request cannot be overtaken by small ones.
    tokens: int = 0


class LlamaAdmissionLease:
    __slots__ = (
        "_queue",
        "_slot",
        "_released",
        "_release_lock",
        "_parked",
        "_budgeted",
        "_tokens",
        "_parked_tokens",
        "_park_id",
    )

    def __init__(
        self,
        queue: Optional["LlamaAdmissionQueue"],
        slot: Optional[int] = None,
        tokens: int = 0,
    ):
        self._queue = queue
        self._slot = slot
        self._released = False
        self._release_lock = threading.Lock()
        self._parked = False
        self._budgeted = False
        self._tokens = max(0, int(tokens or 0))
        self._parked_tokens = 0
        self._park_id = None

    @property
    def slot(self) -> Optional[int]:
        """Pool slot this lease holds, or None when admission is disabled."""
        return self._slot

    def park(self) -> bool:
        """Hand the slot back while this holder waits on something off the GPU.

        Tokens stay committed: llama-server holds this holder's KV until something erases it.
        See ``release_parked_cache``.

        A run stopped on a tool approval prompt is not decoding, so holding its slot would let
        unanswered prompts fill the pool while llama-server idles. The lease itself stays valid:
        releasing it after a park is still correct.

        False when the park budget is spent and nothing was given back: the caller keeps its slot
        across the prompt, as it did before parking existed. Slower for whoever is behind it, but
        each freed slot admits another run that can park too.
        """
        queue = self._queue
        with self._release_lock:
            if queue is None or self._released or self._parked:
                return False
            # Under the lease lock; nothing takes the queue lock then a lease lock, so no deadlock.
            park_id = queue.try_park(self._slot, tokens = self._tokens)
            if park_id is None:
                return False
            self._park_id = park_id
            self._parked = True
            self._budgeted = True
            self._slot = None
        return True

    @property
    def parked_tokens(self) -> int:
        with self._release_lock:
            if self._released or not self._parked:
                return 0
            return self._tokens

    def reclaim_would_admit(self) -> bool:
        queue = self._queue
        tokens = self.parked_tokens
        if queue is None or tokens <= 0:
            return False
        return queue.reclaim_would_admit(tokens)

    def release_parked_cache(self) -> int:
        """Hand back a parked holder's commitment, once llama-server confirms the erasure."""
        queue = self._queue
        with self._release_lock:
            if queue is None or self._released or not self._parked or self._tokens <= 0:
                return 0
            returned, self._tokens = self._tokens, 0
            self._parked_tokens += returned
            queue.reclaim_parked_tokens(returned, park_id = self._park_id)
        return returned

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
            park_id = self._park_id
        self._drop_budget()
        if self._queue is not None:
            self._queue.unpark(park_id)

    async def unpark_async(
        self,
        *,
        cancel_event = None,
        poll_s: float = 0.02,
        timeout_s: Optional[float] = DEFAULT_RECOST_WAIT_TIMEOUT_S,
    ) -> None:
        """Take a slot back, waiting until the pool has room.

        ``park`` gave the slot to a waiter, so by the time the user answers the
        prompt someone else may be decoding in it. Resuming regardless put two
        holders on a one-slot server. A holder whose cache was erased must win its
        commitment back too, or it decodes against room the queue gave away. Gives
        up if the caller is cancelled, since the holder is then leaving anyway and
        must not be stuck here.
        """
        queue = self._queue
        if queue is None or not self._parked:
            return
        # Before the wait: this holder is already off the executor and must not keep others off it.
        self._drop_budget()
        tokens = self._parked_tokens
        park_id = self._park_id
        slot = await queue.acquire_parked_slot(
            cancel_event = cancel_event,
            poll_s = poll_s,
            tokens = tokens,
            timeout_s = timeout_s,
            abandoned = lambda: self._released,
        )
        stranded = None
        with self._release_lock:
            # release() may have run during the wait and done the unpark itself.
            parked, self._parked = self._parked, False
            if self._released:
                # Released while waiting: return the slot here or it is stranded for good.
                stranded = slot
            else:
                self._slot = slot
                if slot is not None:
                    self._tokens += tokens
                    self._parked_tokens = 0
        if parked:
            queue.unpark(park_id)
        if stranded is not None:
            queue.release(stranded, tokens)

    def recost(self, tokens: int) -> bool:
        """Re-state what this lease actually occupies as its conversation grows.

        True when the new figure is in force (including with no budget to account
        against). False means the cache cannot back the growth, the previous commitment
        still stands, and the caller is over its reservation.
        """
        want = max(0, int(tokens or 0))
        # Held across try_recost or an interleaved release strands the difference as committed.
        # release() takes this lock then the queue's, so the order cannot deadlock.
        with self._release_lock:
            if self._released or self._queue is None:
                return True
            if want == self._tokens:
                return True
            if not self._queue.try_recost(self._tokens, want):
                return False
            self._tokens = want
            return True

    def recost_waiting(
        self,
        tokens: int,
        *,
        cancel_event = None,
        poll_s: float = 0.05,
        timeout_s: Optional[float] = DEFAULT_RECOST_WAIT_TIMEOUT_S,
        allow_yield: bool = True,
    ) -> bool:
        """Re-state this lease's cost, waiting for room rather than running over it.

        ``recost`` declines when the cache is full and the caller carries on at its old figure, so
        the next round sends a bigger prompt than the pool was told about, and enough of those is
        the ``Context size has been exceeded`` that kills every decoding slot at once.

        Call this only between rounds, and only with ``allow_yield`` true where an idle slot's cells
        actually come back. Being between rounds makes the slot IDLE; what makes its cells REUSABLE
        under ``--kv-unified`` is ``prompt_clear()``, which llama-server runs only under
        ``--cache-idle-slots`` (``server-context.cpp``), force-disabled by ``--cache-ram 0`` and
        absent on older servers. Studio emits ``--cache-ram 0`` on Windows under full GPU offload
        (#5692) alongside ``--kv-unified``, and there a yielded round's cells stay resident, so
        yielding would hand the same capacity to a second caller; with yielding off this degrades to
        plain ``recost``, which declines rather than overcommits. Where clearing IS active the cache
        changes only the PRICE: reclaiming costs a prefix hit if the cells were spilled to host RAM.

        False means this lease still holds the figure it came in with: declined, cancelled,
        released, or waited past ``timeout_s``. The timeout is the blast radius, since a reparker
        holds the wait line shut for everyone (see ``yield_commitment``); giving up restores the old
        commitment and the decline-and-continue behaviour that predates this.
        """
        want = max(0, int(tokens or 0))
        if self.recost(want):
            return True
        if not allow_yield:
            return False
        queue = self._queue
        with self._release_lock:
            if self._released or queue is None:
                return True
            held, self._tokens = self._tokens, 0
        queue.yield_commitment(held)
        deadline = None if not timeout_s or timeout_s <= 0 else time.monotonic() + timeout_s
        try:
            while True:
                if queue.try_reclaim_commitment(want):
                    with self._release_lock:
                        if self._released:
                            # release() already gave back 0 and will not run again, so return the commitment here.
                            queue.release(None, want)
                            return True
                        self._tokens = want
                    return True
                # Every pass: release() from route teardown does not set the cancel event.
                if self._released:
                    queue.abandon_repark()
                    return False
                if cancel_event is not None and cancel_event.is_set():
                    return self._give_up_repark(queue, held, cancelled = True)
                if deadline is not None and time.monotonic() >= deadline:
                    return self._give_up_repark(queue, held, cancelled = False)
                time.sleep(poll_s)
        except BaseException:
            self._give_up_repark(queue, held, cancelled = True)
            raise

    def _give_up_repark(self, queue, held: int, *, cancelled: bool) -> bool:
        """Stop waiting and go back to holding ``held``.

        Both halves under one queue lock (``abandon_repark(restore = held)``): the wait line reopens
        and the old figure is re-committed together. This lease still occupies that much of
        llama-server's cache, so re-committing it is a correction, not a request -- asking through
        ``try_recost`` let a full cache REFUSE it, after which release() subtracted a commitment
        that was never restored and handed the next arrival that much phantom room.

        ``_release_lock`` is held across the queue call in release()'s lock order, so the commitment
        cannot be released out from under the restore.
        """
        with self._release_lock:
            if self._released:
                queue.abandon_repark()
                return False
            self._tokens = held
            queue.abandon_repark(restore = held)
        return False

    def release(self) -> None:
        queue = None
        parked = False
        with self._release_lock:
            if self._released:
                return
            self._released = True
            queue = self._queue
            parked, self._parked = self._parked, False
            park_id = self._park_id
        self._drop_budget()
        if queue is not None:
            if parked:
                queue.unpark(park_id)
            queue.release(self._slot, self._tokens)

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

    The pool mirrors llama-server's own ``--parallel`` slots: ``capacity`` slot ids are each either
    free or held by exactly one caller, and a caller that finds every slot busy waits in arrival
    order and is handed the next slot to free, so no caller is starved. This bounds only the callers
    that reserve: chat completions and messages do, while /v1/completions, Unsloth's own chat
    endpoint and RAG captioning all reach llama-server directly, so it is not a global cap.

    Waiting is unbounded in time by default (``queue_timeout_s`` None); the wait line itself is
    bounded, and only in how many may line up before new arrivals are rejected. By default that is
    ``16 x slots`` floored at 64; an unbounded line takes ``max_queue`` or ``queue_per_slot`` set to
    0. See ``LlamaAdmissionConfig.queue_limit``.
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
        "_unpark_tokens",
        "_unpark_lapsed",
        "_unpark_seq",
        "_parked_reclaimable",
        "_park_seq",
        "_committed",
        "_budget",
        "_reparking",
    )

    def __init__(self, key: str):
        self.key = key
        self._lock = threading.Lock()
        self._capacity = 1
        self._free: list[int] = [0]
        # Bitmask of held slots; _held is its popcount since int.bit_count() is 3.10+.
        self._in_use = 0
        self._held = 0
        self._waiters: Deque[_Waiter] = deque()
        # Parked holders own no slot; this only keeps the queue off the idle-eviction list.
        self._parked = 0
        # FIFO tickets: a bare count deadlocked, every approved holder blocked every other one.
        self._unpark_tickets: Deque[int] = deque()
        self._unpark_tokens: dict[int, int] = {}
        self._unpark_lapsed: set = set()
        self._unpark_seq = 0
        self._parked_reclaimable: dict[int, int] = {}
        self._park_seq = 0
        # 0 budget disables the check.
        self._committed = 0
        self._budget = 0
        # Reparkers still hold a slot, so they are not in _waiters. See yield_commitment.
        self._reparking = 0

    def _resize_pool_locked(self, capacity: int) -> None:
        if capacity == self._capacity:
            return
        self._capacity = capacity
        self._free = [slot for slot in range(capacity) if not self._in_use >> slot & 1]

    def _fits_against_locked(self, committed: int, tokens: int) -> bool:
        """Whether ``tokens`` fit a cache already holding ``committed``.

        A caller is always admitted when nothing else holds KV, however large it is. Its request may
        still be refused by llama-server, but that refusal names both token counts and the setting
        to change, whereas refusing here would strand it forever.

        The escape asks whether anything is COMMITTED, not whether a slot is held. ``try_park``
        hands a slot back while its holder waits on a tool approval, which drops ``_held`` to zero
        even though llama-server still holds that lease's KV, so a held-slot test admitted the next
        caller unconditionally: park a 1500 token lease against a 2048 token budget, and the next
        1500 token one sailed through to 3000 committed, which is the collision this accounting
        exists to stop.
        """
        if self._budget <= 0 or tokens <= 0:
            return True
        if committed <= 0:
            return True
        return committed + tokens <= self._budget

    def _fits_budget_locked(
        self,
        tokens: int,
        reserved_tokens: int = 0,
    ) -> bool:
        """Whether ``tokens`` more KV fit over the ``reserved_tokens`` owed to resumes."""
        return self._fits_against_locked(self._committed + max(0, reserved_tokens), tokens)

    def _reserved_tokens_locked(self) -> int:
        return sum(
            tokens
            for ticket, tokens in self._unpark_tokens.items()
            if ticket not in self._unpark_lapsed
        )

    def _can_admit_locked(
        self,
        reserved: int,
        tokens: int = 0,
        reserved_tokens: int = 0,
    ) -> bool:
        # Count every held slot against the ceiling; reserved holds slots back for resuming holders.
        if not (bool(self._free) and (self._held + reserved) < self._capacity):
            return False
        # With --kv-unified every slot reports full n_ctx, so also check the token budget.
        return self._fits_budget_locked(tokens, reserved_tokens)

    def _take_slot_locked(
        self,
        reserved: int,
        tokens: int = 0,
        reserved_tokens: int = 0,
    ) -> Optional[int]:
        if not self._can_admit_locked(reserved, tokens, reserved_tokens):
            return None
        slot = self._free.pop()
        self._in_use |= 1 << slot
        self._held += 1
        self._committed += max(0, tokens)
        return slot

    def reserve(
        self,
        *,
        capacity: int,
        config: LlamaAdmissionConfig,
        tokens: Optional[int] = None,
        budget: Optional[int] = None,
    ) -> LlamaAdmissionReservation:
        """Take a serving slot, waiting in FIFO order when none is available.

        ``tokens`` is the KV this caller will occupy and ``budget`` the size of the
        cache it is drawn from. Leaving either unset (or ``kv_budget`` off) keeps
        the slot-only behaviour every caller had before token accounting existed.
        """
        capacity = max(1, int(capacity or 1))
        if not config.enabled:
            return LlamaAdmissionReservation(
                queue = None,
                lease = LlamaAdmissionLease(None),
                snapshot = LlamaAdmissionSnapshot(self.key, capacity, 0, 0, capacity),
            )

        if not config.kv_budget:
            tokens = budget = None
        cost = max(0, int(tokens or 0))

        loop = asyncio.get_running_loop()
        with self._lock:
            self._resize_pool_locked(capacity)
            # Re-read every reserve: a reload can relaunch llama-server at a different -c.
            self._budget = max(0, int(budget or 0))
            self._grant_waiters_locked()
            # A pending repark closes the fast path, or arrivals take the room the reparker gave back.
            if not self._waiters and self._reparking == 0:
                slot = self._take_slot_locked(
                    len(self._unpark_tickets), cost, self._reserved_tokens_locked()
                )
                if slot is not None:
                    return LlamaAdmissionReservation(
                        queue = self,
                        lease = LlamaAdmissionLease(self, slot, cost),
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
                tokens = cost,
            )
            self._waiters.append(waiter)
            return LlamaAdmissionReservation(
                queue = self,
                waiter = waiter,
            )

    def _release_slot_locked(self, slot: Optional[int]) -> None:
        if slot is None or not self._in_use >> slot & 1:
            return
        self._in_use &= ~(1 << slot)
        self._held -= 1
        if slot < self._capacity:
            self._free.append(slot)

    def release(
        self,
        slot: Optional[int],
        tokens: int = 0,
    ) -> None:
        with self._lock:
            self._release_slot_locked(slot)
            # Floored at 0 so a manual double release cannot let the budget overadmit.
            self._committed = max(0, self._committed - max(0, int(tokens or 0)))
            self._grant_waiters_locked()

    def try_recost(self, tokens_from: int, tokens_to: int) -> bool:
        """Move a live commitment to a new size. False leaves it exactly as it was.

        A tool loop's cost is unknown at admission: each round appends its results and re-sends the
        conversation, so a run that opened small can grow into the cache while its commitment stays
        at the opening estimate. #9392 closed that by reserving the whole cache for any tool loop,
        which serialises every tool chat; this is the re-costing that PR named as the alternative.

        Never blocks and never overcommits, so it is safe to call from inside a generator: growth
        that does not fit is refused and the caller keeps what it holds, rather than waiting on
        holders that may be waiting on it.
        """
        tokens_from = max(0, int(tokens_from or 0))
        tokens_to = max(0, int(tokens_to or 0))
        with self._lock:
            if self._budget <= 0:
                return True
            if tokens_to <= tokens_from:
                self._committed = max(0, self._committed - (tokens_from - tokens_to))
                self._grant_waiters_locked()
                return True
            delta = tokens_to - tokens_from
            # A lone holder goes past the budget: refusing it stalls a conversation nothing can unblock.
            alone = (self._committed - tokens_from) <= 0
            if alone or self._committed + delta <= self._budget:
                self._committed += delta
                return True
            return False

    def yield_commitment(self, tokens: int) -> None:
        """Hand a live commitment back before asking for a bigger one.

        Half of ``LlamaAdmissionLease.recost_waiting``. Unconditional by design: a holder that
        blocked while still holding its old commitment would be waiting on holders waiting on it --
        four runs at a quarter of the cache each, all wanting half, never resolve. Letting go first
        cannot deadlock, since ``_committed`` strictly falls. The caller counts as reparking until
        it reclaims, so a new arrival cannot take the room it just released: an in-flight
        conversation beats one that has not started.
        """
        with self._lock:
            self._committed = max(0, self._committed - max(0, int(tokens or 0)))
            self._reparking += 1
            self._grant_waiters_locked()

    def try_reclaim_commitment(self, tokens: int) -> bool:
        """The other half: take a commitment of ``tokens``, or report that it does not fit.

        Test and commit under one lock, so only one caller can win the ``_committed == 0``
        escape. Otherwise four reparkers racing an empty cache each see zero and each
        admit themselves.
        """
        want = max(0, int(tokens or 0))
        with self._lock:
            if self._budget > 0 and not self._fits_budget_locked(want):
                return False
            self._committed += want
            self._reparking = max(0, self._reparking - 1)
            # The decrement may have dropped the last repark barrier, and nothing else re-runs the grant.
            self._grant_waiters_locked()
            return True

    def abandon_repark(self, restore: int = 0) -> None:
        """Stop counting a reparker that gave up, and re-commit what it takes back.

        One lock for both. ``restore`` is what ``yield_commitment`` handed back and the lease still
        occupies at llama-server, so it is re-committed UNCONDITIONALLY: a correction, not a request
        for room. In two steps the wait line reopened on room the lease was about to take back;
        through ``try_recost`` a full cache could refuse it, after which release() subtracted a
        commitment that was never restored.
        """
        with self._lock:
            self._reparking = max(0, self._reparking - 1)
            self._committed += max(0, int(restore or 0))
            self._grant_waiters_locked()

    def try_park(
        self,
        slot: Optional[int],
        *,
        tokens: int = 0,
    ) -> Optional[int]:
        """Return a parked holder's slot to the pool. See ``LlamaAdmissionLease.park``.

        None leaves the slot with its holder, so a refused park costs nothing to
        undo. Otherwise an id identifying this park until the holder leaves it. The
        per-queue count is only what ``is_idle`` reads; the budget and the capacity it
        is sized from are both process-wide.

        ``tokens`` stays committed, and is recorded only so ``reclaim_would_admit``
        knows how much erasing could free.
        """
        if not _claim_park(_max_parked(_live_capacity(self))):
            return None
        with self._lock:
            self._parked += 1
            self._park_seq += 1
            park_id = self._park_seq
            self._parked_reclaimable[park_id] = max(0, int(tokens or 0))
            self._release_slot_locked(slot)
            self._grant_waiters_locked()
        return park_id

    def reclaim_parked_tokens(
        self,
        tokens: int,
        *,
        park_id: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._parked_reclaimable.pop(park_id, None)
            self._committed = max(0, self._committed - max(0, int(tokens or 0)))
            self._grant_waiters_locked()

    def _ticket_position_locked(self, ticket: int) -> tuple:
        """(slots, room) that resuming holders ahead of ``ticket`` hold back from it.

        One ahead reserves room only until its deadline lapses, but holds a slot back only
        while it can afford that room: pinning one it cannot pay for deadlocks the pool.
        """
        ahead = 0
        ahead_tokens = 0
        for queued in self._unpark_tickets:
            if queued == ticket:
                break
            needs = self._unpark_tokens.get(queued, 0)
            if self._fits_budget_locked(needs, ahead_tokens):
                ahead += 1
            if queued not in self._unpark_lapsed:
                ahead_tokens += needs
        return (ahead, ahead_tokens)

    def _claimants_locked(self):
        for ticket in self._unpark_tickets:
            yield (self._unpark_tokens.get(ticket, 0), *self._ticket_position_locked(ticket))
        self._prune_waiters_locked()
        if self._waiters:
            yield (
                self._waiters[0].tokens,
                len(self._unpark_tickets),
                self._reserved_tokens_locked(),
            )

    def reclaim_would_admit(self, tokens: int) -> bool:
        """Whether erasing parked caches would let anyone admission owes into the cache.

        Weighed in aggregate: one claimant erasing cannot fit must not hide a smaller one,
        nor two too-small holders both decline and leave the queue stalled.
        """
        if max(0, int(tokens or 0)) <= 0:
            return False
        with self._lock:
            if self._budget <= 0 or self._reparking > 0:
                return False
            reclaimable = sum(self._parked_reclaimable.values())
            for wanted, held_back, reserved in self._claimants_locked():
                if self._can_admit_locked(held_back, wanted, reserved):
                    continue
                if not (bool(self._free) and (self._held + held_back) < self._capacity):
                    continue
                if self._fits_against_locked(
                    max(0, self._committed - reclaimable) + reserved, wanted
                ):
                    return True
            return False

    def unpark(self, park_id: Optional[int] = None) -> None:
        with self._lock:
            if self._parked > 0:
                self._parked -= 1
            self._parked_reclaimable.pop(park_id, None)

    async def acquire_parked_slot(
        self,
        *,
        cancel_event = None,
        poll_s: float = 0.02,
        tokens: int = 0,
        timeout_s: Optional[float] = DEFAULT_RECOST_WAIT_TIMEOUT_S,
        abandoned = None,
    ) -> Optional[int]:
        """Wait for a slot, and for ``tokens`` of cache room, None if cancelled.

        Ordered by ticket rather than counted, so approvals resume in the order
        they came back: counting them made every approved holder block every
        other one, and with nothing decoding that never resolved.

        ``tokens`` is what a holder whose cache was erased must win back, held from arrivals
        until ``timeout_s`` lapses the claim. Lapsing keeps the ticket's place in the slot
        order and still never admits it over budget: llama.cpp answers an exhausted cache by
        clearing every slot it is decoding.
        """
        tokens = max(0, int(tokens or 0))
        with self._lock:
            self._unpark_seq += 1
            ticket = self._unpark_seq
            self._unpark_tickets.append(ticket)
            self._unpark_tokens[ticket] = tokens
        deadline = None if not timeout_s or timeout_s <= 0 else time.monotonic() + timeout_s
        try:
            while True:
                with self._lock:
                    if deadline is not None and time.monotonic() >= deadline:
                        self._unpark_lapsed.add(ticket)
                        deadline = None
                        self._grant_waiters_locked()
                    ahead, ahead_tokens = self._ticket_position_locked(ticket)
                    slot = self._take_slot_locked(ahead, tokens, ahead_tokens)
                    if slot is not None:
                        return slot
                    if cancel_event is not None and cancel_event.is_set():
                        return None
                    if abandoned is not None and abandoned():
                        return None
                await asyncio.sleep(poll_s)
        finally:
            with self._lock:
                try:
                    self._unpark_tickets.remove(ticket)
                except ValueError:
                    pass
                self._unpark_tokens.pop(ticket, None)
                self._unpark_lapsed.discard(ticket)
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
                    pass
            # A cancel frees nothing, so nothing else re-runs admission for waiters it was blocking.
            self._grant_waiters_locked()
        if lease_to_release is not None:
            lease_to_release.release()

    def snapshot(self) -> LlamaAdmissionSnapshot:
        with self._lock:
            self._prune_waiters_locked()
            return self._snapshot_locked()

    def is_idle(self) -> bool:
        with self._lock:
            self._prune_waiters_locked()
            # A parked holder will return here; evicting would resume it against a fresh 1-slot pool.
            return self._in_use == 0 and not self._waiters and not self._parked

    def _grant_waiters_locked(self) -> None:
        # Reparkers go first, or steady arrivals hold a growing run at its old size.
        if self._reparking > 0:
            return
        # Test the head's own cost (FIFO, head-of-line blocking) so a large waiter is not starved.
        while self._waiters and self._can_admit_locked(
            len(self._unpark_tickets), self._waiters[0].tokens, self._reserved_tokens_locked()
        ):
            waiter = self._waiters.popleft()
            if waiter.cancelled or waiter.future.done():
                continue
            slot = self._take_slot_locked(
                len(self._unpark_tickets), waiter.tokens, self._reserved_tokens_locked()
            )
            lease = LlamaAdmissionLease(self, slot, waiter.tokens)
            waiter.granted_lease = lease
            try:
                waiter.loop.call_soon_threadsafe(self._deliver_lease, waiter, lease)
            except RuntimeError:
                # Reclaim the slot: _free is rebuilt from the bitmask, so a set bit would strand it.
                waiter.granted_lease = None
                self._release_slot_locked(slot)
                self._committed = max(0, self._committed - max(0, waiter.tokens))

    def _deliver_lease(self, waiter: _Waiter, lease: LlamaAdmissionLease) -> None:
        # Runs on the waiter's own loop thread, the only one that cancels it, so unlocked is safe.
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
        # Only rebuild the deque when a waiter died out of band; doing it every time dominated the hot path.
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
            # Include resume tickets, or a full queue looks idle while one still waits.
            queued = len(self._waiters) + len(self._unpark_tickets),
            # What another caller could actually take, matching _can_admit_locked.
            free = min(
                len(self._free),
                max(0, self._capacity - self._held - len(self._unpark_tickets)),
            ),
            committed = self._committed,
            budget = self._budget,
        )


_QUEUES_LOCK = threading.Lock()
_QUEUES: dict[str, LlamaAdmissionQueue] = {}


def get_llama_admission_queue(key: str) -> LlamaAdmissionQueue:
    with _QUEUES_LOCK:
        queue = _QUEUES.get(key)
        if queue is None:
            queue = LlamaAdmissionQueue(key)
            _QUEUES[key] = queue
            # base_url has a fresh port per load; drop idle queues from prior loads so the registry is bounded.
            for stale_key in [k for k in _QUEUES if k != key and _QUEUES[k].is_idle()]:
                del _QUEUES[stale_key]
        return queue


def peek_llama_admission_snapshot(key: str) -> Optional[LlamaAdmissionSnapshot]:
    """Read-only view of one queue's state; never creates a queue."""
    with _QUEUES_LOCK:
        queue = _QUEUES.get(key)
    return queue.snapshot() if queue is not None else None


def estimate_gpu_retry_after() -> int:
    with _QUEUES_LOCK:
        queues = tuple(_QUEUES.values())
    waves = 1
    for queue in queues:
        snapshot = queue.snapshot()
        capacity = max(1, snapshot.capacity)
        waves = max(waves, (snapshot.active + snapshot.queued + capacity - 1) // capacity)
    return min(120, 15 * waves)


def reset_llama_admission_queues() -> None:
    global _parked_total
    with _QUEUES_LOCK:
        _QUEUES.clear()
    # The budget outlives its queues; dropping them without it leaks the count.
    with _PARK_LOCK:
        _parked_total = 0
