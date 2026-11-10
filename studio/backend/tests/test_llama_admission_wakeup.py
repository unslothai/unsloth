# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Waits for room wake when the room appears, not on their next poll; the poll only bounds what
does not notify (the cancel event, a release-only predicate, the deadline)."""

from __future__ import annotations

import asyncio
import threading
import time

from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue

# Far longer than any wake below may take, so a wait that only polls fails these tests.
_SLOW_POLL_S = 5.0


def _lease(
    queue,
    *,
    capacity,
    tokens = None,
    budget = None,
):
    reservation = queue.reserve(
        capacity = capacity, config = LlamaAdmissionConfig(), tokens = tokens, budget = budget
    )
    lease = reservation.lease_nowait()
    assert lease is not None
    return lease


def test_a_growth_waiting_for_room_wakes_when_a_holder_releases():
    async def scenario():
        queue = LlamaAdmissionQueue("wake-recost")
        first = _lease(queue, capacity = 4, tokens = 2000, budget = 4096)
        second = _lease(queue, capacity = 4, tokens = 2000, budget = 4096)
        done: list = []

        def grow():
            done.append(second.recost_waiting(3500, poll_s = _SLOW_POLL_S))

        thread = threading.Thread(target = grow, daemon = True)
        thread.start()
        thread.join(0.2)
        assert thread.is_alive(), "expected the growth to wait"
        start = time.monotonic()
        first.release()
        thread.join(_SLOW_POLL_S * 2)
        waited = time.monotonic() - start
        assert done == [True]
        assert waited < 1.0, f"woke {waited:.2f}s after the release, on the poll"
        assert queue.snapshot().committed == 3500

    asyncio.run(scenario())


def test_a_cancel_still_ends_the_growth_wait_within_the_poll():
    async def scenario():
        queue = LlamaAdmissionQueue("wake-recost-cancel")
        _lease(queue, capacity = 4, tokens = 2000, budget = 4096)
        second = _lease(queue, capacity = 4, tokens = 2000, budget = 4096)
        cancel = threading.Event()
        done: list = []

        def grow():
            done.append(second.recost_waiting(3500, cancel_event = cancel, poll_s = 0.05))

        thread = threading.Thread(target = grow, daemon = True)
        thread.start()
        thread.join(0.2)
        assert thread.is_alive()
        cancel.set()
        thread.join(5)
        assert done == [False]
        assert queue.snapshot().committed == 4000

    asyncio.run(scenario())


def test_a_parked_holder_resumes_when_the_slot_frees():
    async def scenario():
        queue = LlamaAdmissionQueue("wake-unpark")
        config = LlamaAdmissionConfig()
        a = queue.reserve(capacity = 1, config = config)
        a_lease = a.lease_nowait()
        b = queue.reserve(capacity = 1, config = config)
        a_lease.park()
        b_lease = await asyncio.wait_for(b.wait(timeout_s = 1), timeout = 2)
        assert b_lease is not None

        resumed = asyncio.ensure_future(a_lease.unpark_async(poll_s = _SLOW_POLL_S))
        await asyncio.sleep(0.05)
        assert not resumed.done()
        start = time.monotonic()
        threading.Thread(target = b_lease.release).start()
        await asyncio.wait_for(resumed, timeout = _SLOW_POLL_S * 2)
        waited = time.monotonic() - start
        assert a_lease.slot is not None
        assert waited < 1.0, f"woke {waited:.2f}s after the release, on the poll"
        assert queue._change_events == {}, "a finished wait must not stay registered"

    asyncio.run(scenario())


def test_a_parked_holder_still_gives_up_on_cancel_within_the_poll():
    async def scenario():
        queue = LlamaAdmissionQueue("wake-unpark-cancel")
        config = LlamaAdmissionConfig()
        a = queue.reserve(capacity = 1, config = config)
        a_lease = a.lease_nowait()
        b = queue.reserve(capacity = 1, config = config)
        a_lease.park()
        assert await asyncio.wait_for(b.wait(timeout_s = 1), timeout = 2) is not None
        cancel = threading.Event()
        waiting = asyncio.ensure_future(a_lease.unpark_async(cancel_event = cancel, poll_s = 0.05))
        await asyncio.sleep(0.1)
        assert not waiting.done()
        cancel.set()
        await asyncio.wait_for(waiting, timeout = 2)
        assert a_lease.slot is None

    asyncio.run(scenario())
