# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A paused chat's resume is bounded once, across the room wait and the slot line.

The room wait has an absolute bound of 80 times its stall window. Room found near it, with
every slot still taken, started the admission wait on a fresh bound of its own, so one
resume could stay paused nearly twice the documented limit.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue
from core.inference.llama_preemption import (
    MAX_RESUME_WAIT_MULTIPLE,
    ControllerPreemptionPolicy,
    ParticipantState,
    PreemptionController,
    PreemptSignal,
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _ticking():
    box = {"n": 0}

    def _read():
        box["n"] += 1
        return (box["n"],)

    return _read


def _lease(queue, tokens):
    reservation = queue.reserve(
        capacity = 1, config = LlamaAdmissionConfig(), tokens = tokens, budget = 16384
    )
    lease = reservation.lease_nowait()
    assert lease is not None
    return lease


@pytest.mark.asyncio
async def test_the_slot_line_stops_at_the_bound_it_is_given():
    queue = LlamaAdmissionQueue("shared-deadline")
    paused = _lease(queue, 1000)
    assert paused.preempt() is True
    holder = _lease(queue, 1000)
    started = time.monotonic()
    # A moving backend renews the stall window for ever; only the absolute bound ends it.
    got = await asyncio.wait_for(
        paused.resume_async(
            1000,
            poll_s = 0.01,
            timeout_s = 10.0,
            progress = _ticking(),
            hard_deadline_at = started + 0.2,
        ),
        timeout = 5.0,
    )
    assert got is False
    assert time.monotonic() - started < 1.0, "the slot line started a fresh bound"
    holder.release()
    paused.release()


def test_the_policy_hands_the_lease_the_room_waits_bound():
    controller = PreemptionController("http://127.0.0.1:65085")
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    seen: dict = {}

    class _Lease:
        is_released = False
        tokens = 0

        async def resume_async(self, want, **kwargs):
            seen.update(kwargs)
            return True

    loop = asyncio.new_event_loop()
    threading.Thread(target = loop.run_forever, daemon = True).start()
    try:
        signal = PreemptSignal()
        controller.register("chat", lease = _Lease(), tokens = 1000, signal = signal)
        controller.set_state("chat", ParticipantState.PAUSED)
        policy = ControllerPreemptionPolicy(controller, "chat", signal, loop = loop)
        started = time.monotonic()
        assert policy.await_resume(timeout = 1.0) is True
        bound = seen.get("hard_deadline_at")
        assert bound is not None, f"the lease wait got no bound of the room wait's: {seen}"
        assert started < bound <= time.monotonic() + 1.0 * MAX_RESUME_WAIT_MULTIPLE
    finally:
        loop.call_soon_threadsafe(loop.stop)
