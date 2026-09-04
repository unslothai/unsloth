# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Randomised stress against the admission queue's invariants.

Two properties that have to hold under ANY interleaving, because the failure they guard is
a wedged Unsloth, not a wrong number:

  1. ``committed`` never exceeds ``budget``, except the single holder the escape lets past.
     Breaking this is the ``Context size has been exceeded`` that clears every decoding
     slot at once.
  2. The queue always drains. A leaked repark counter holds the wait line shut for every
     caller, freezing the pool for the life of the process.

Seeded, so a failure is reproducible from the seed in the assertion message.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time

import pytest

from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue


def _lease(queue, *, tokens, budget, capacity):
    reservation = queue.reserve(
        capacity = capacity,
        config = LlamaAdmissionConfig(),
        tokens = tokens,
        budget = budget,
    )
    return reservation.lease_nowait()


class _Ceiling:
    """Watches ``committed`` from another thread and remembers the worst it saw."""

    def __init__(self, queue):
        self.queue = queue
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target = self._run, daemon = True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.queue.snapshot().committed)
            time.sleep(0.001)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(5)
        self.peak = max(self.peak, self.queue.snapshot().committed)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.asyncio
async def test_random_traffic_never_exceeds_the_cache_and_always_drains(seed):
    """Many holders, random sizes, random growth, random cancels, all at once."""
    rng = random.Random(seed)
    capacity = rng.choice([1, 2, 4, 8])
    budget = rng.choice([2048, 4096, 65536])
    queue = LlamaAdmissionQueue(f"stress-{seed}")

    # Every holder is admitted at a size that fits alongside the others, so any excess is
    # the queue's doing, not the workload's. The escape is exercised separately.
    opening = max(1, budget // max(1, capacity))
    leases = [
        lease
        for lease in (
            _lease(queue, tokens = opening, budget = budget, capacity = capacity) for _ in range(capacity)
        )
        if lease is not None
    ]
    assert leases, f"seed={seed}: nothing was admitted at an equal share"

    def worker(lease):
        for _ in range(rng.randint(1, 4)):
            want = rng.randint(1, budget)
            cancel = threading.Event()
            if rng.random() < 0.3:
                # Cancel shortly after asking, to land inside the wait.
                threading.Timer(rng.uniform(0.0, 0.05), cancel.set).start()
            lease.recost_waiting(
                want,
                cancel_event = cancel,
                poll_s = 0.005,
                timeout_s = 5.0,
            )
            time.sleep(rng.uniform(0.0, 0.01))
        lease.release()

    with _Ceiling(queue) as ceiling:
        threads = [threading.Thread(target = worker, args = (lease,), daemon = True) for lease in leases]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(90)

    assert not any(t.is_alive() for t in threads), f"seed={seed}: a worker never finished"
    snapshot = queue.snapshot()
    assert snapshot.committed == 0, f"seed={seed}: {snapshot.committed} tokens stranded"
    assert queue._reparking == 0, f"seed={seed}: repark counter leaked, the queue is wedged"
    # One holder may sit above the budget via the alone escape, so the ceiling is the
    # budget plus the largest single request.
    assert (
        ceiling.peak <= budget * 2
    ), f"seed={seed}: committed peaked at {ceiling.peak} against a {budget} cache"


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.asyncio
async def test_new_arrivals_alongside_growing_holders_still_drain(seed):
    """Reparkers hold the wait line shut, so forgetting to reopen it shows up here as
    arrivals that are never granted."""
    rng = random.Random(1000 + seed)
    capacity = 4
    budget = 4096
    queue = LlamaAdmissionQueue(f"mixed-{seed}")
    share = budget // capacity

    holders = [
        _lease(queue, tokens = share, budget = budget, capacity = capacity) for _ in range(capacity)
    ]
    holders = [lease for lease in holders if lease is not None]

    def grower(lease):
        lease.recost_waiting(
            rng.randint(share, budget),
            poll_s = 0.005,
            timeout_s = 5.0,
        )
        time.sleep(rng.uniform(0.0, 0.02))
        lease.release()

    threads = [threading.Thread(target = grower, args = (lease,), daemon = True) for lease in holders]
    for thread in threads:
        thread.start()

    # Arrivals queue behind the growers, and must eventually be admitted.
    arrivals: list = []

    async def arrive():
        reservation = queue.reserve(
            capacity = capacity,
            config = LlamaAdmissionConfig(),
            tokens = rng.randint(1, share),
            budget = budget,
        )
        # await, never time.sleep: a granted lease is delivered with
        # loop.call_soon_threadsafe, so a synchronous poll blocks its own delivery.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            lease = reservation.lease_nowait()
            if lease is not None:
                arrivals.append(lease)
                lease.release()
                return
            await asyncio.sleep(0.01)
        reservation.cancel()
        arrivals.append(None)

    # reserve() reads the running loop, so each arrival brings its own.
    def arrive_in_loop():
        asyncio.run(arrive())

    newcomers = [threading.Thread(target = arrive_in_loop, daemon = True) for _ in range(4)]
    for thread in newcomers:
        thread.start()
    for thread in threads + newcomers:
        thread.join(120)

    assert not any(t.is_alive() for t in threads + newcomers), f"seed={seed}: did not drain"
    assert arrivals and all(
        lease is not None for lease in arrivals
    ), f"seed={seed}: an arrival was never admitted, the wait line stayed shut"
    assert queue.snapshot().committed == 0
    assert queue._reparking == 0


@pytest.mark.asyncio
async def test_a_wait_that_can_never_be_satisfied_gives_up_rather_than_wedging_the_queue():
    """The blast-radius test. A reparker holds the line shut for everyone, so a wait with
    no possible end must expire, restore its old figure and let the queue run again."""
    budget = 4096
    queue = LlamaAdmissionQueue("timeout")
    # All but one token, so the grower is still admitted alongside it and the cache is
    # exactly full. A squatter holding the whole budget would leave nothing to admit it.
    squatter = _lease(queue, tokens = budget - 1, budget = budget, capacity = 4)
    assert squatter is not None
    grower = _lease(queue, tokens = 1, budget = budget, capacity = 4)
    assert grower is not None, "the grower must be admitted before it can grow"

    start = time.monotonic()
    assert grower.recost_waiting(budget, poll_s = 0.01, timeout_s = 0.5) is False
    waited = time.monotonic() - start
    assert 0.4 <= waited < 15, f"gave up after {waited}s, expected roughly the timeout"
    assert queue._reparking == 0, "the wait line is still shut after a timeout"
    # The queue is usable again: the grower is back at its old figure, so releasing the
    # squatter leaves exactly that behind.
    squatter.release()
    assert queue.snapshot().committed == 1
    grower.release()
    assert queue.snapshot().committed == 0


@pytest.mark.asyncio
async def test_release_during_a_wait_does_not_spin_forever():
    """release() runs from the route's teardown without touching the cancel event, so a
    wait that only watched the event would spin on a dead lease and hold the line shut."""
    budget = 4096
    queue = LlamaAdmissionQueue("released")
    squatter = _lease(queue, tokens = budget - 1, budget = budget, capacity = 4)
    grower = _lease(queue, tokens = 1, budget = budget, capacity = 4)
    assert squatter is not None and grower is not None

    out: list = []

    def grow():
        out.append(grower.recost_waiting(budget, poll_s = 0.01, timeout_s = 60))

    thread = threading.Thread(target = grow, daemon = True)
    thread.start()
    time.sleep(0.2)
    assert thread.is_alive(), "expected it to be waiting"
    grower.release()
    thread.join(15)
    assert not thread.is_alive(), "a released lease kept waiting"
    assert out == [False]
    assert queue._reparking == 0
    squatter.release()
    assert queue.snapshot().committed == 0
