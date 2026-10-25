# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A raw relay cannot be paused and decodes without reporting tokens, so the cache reading is
the only sign it is still working. A paused chat waiting behind it must not call that a stall."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.inference.llama_preemption import (
    ControllerPreemptionPolicy,
    ParticipantState,
    PreemptionController,
    PreemptSignal,
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    is_released = False
    tokens = 0

    async def resume_async(self, want, **kwargs):
        return True


def _raw_beside_a_paused_chat(key):
    controller = PreemptionController(key)
    controller.configure(budget = 8192, kv_unified = True)
    controller.register("raw", tokens = 6000, state = ParticipantState.STREAMING_RAW)
    controller.note_measured("raw")
    controller.note_resident(1000)
    loop = asyncio.new_event_loop()
    threading.Thread(target = loop.run_forever, daemon = True).start()
    controller.register("chat", lease = _Lease(), tokens = 3000, signal = PreemptSignal())
    controller.set_state("chat", ParticipantState.PAUSED)
    policy = ControllerPreemptionPolicy(controller, "chat", PreemptSignal(), loop = loop)
    return controller, policy, loop


def test_residency_growth_counts_as_progress():
    controller, _policy, loop = _raw_beside_a_paused_chat("raw-sig")
    try:
        before = controller.progress_signature()
        controller.note_resident(1100)
        assert controller.progress_signature() != before, (
            "a raw holder filling the cache is progress the waiter must see"
        )
        steady = controller.progress_signature()
        controller.note_resident(1100)
        assert controller.progress_signature() == steady, "an unchanged reading is not progress"
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_a_paused_chat_outwaits_a_raw_holder_that_is_still_filling():
    timeout = 0.3
    controller, policy, loop = _raw_beside_a_paused_chat("raw-wait")
    assert not controller.room_for("chat", 3000), "the fixture must start without room"
    stop = threading.Event()
    grew = {"steps": 0}

    def fill_then_finish():
        resident = 1000
        deadline = time.monotonic() + timeout * 3
        while time.monotonic() < deadline and not stop.is_set():
            time.sleep(timeout / 3)
            resident += 100
            controller.note_resident(resident)
            grew["steps"] += 1
        controller.unregister("raw")
        controller.note_resident(0)

    worker = threading.Thread(target = fill_then_finish, daemon = True)
    worker.start()
    try:
        started = time.monotonic()
        resumed = policy.await_resume(timeout = timeout)
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        worker.join(timeout = 5)
        loop.call_soon_threadsafe(loop.stop)

    assert grew["steps"] >= 3, "the raw holder never grew; the test proves nothing"
    assert resumed is True, (
        f"gave up after {elapsed:.2f}s while the raw holder's residency kept growing"
    )
    assert elapsed > timeout * 2


def test_a_genuinely_stalled_cache_still_times_out():
    timeout = 0.3
    controller, policy, loop = _raw_beside_a_paused_chat("raw-stall")
    stop = threading.Event()

    def same_reading():
        while not stop.is_set():
            time.sleep(0.05)
            controller.note_resident(1000)

    worker = threading.Thread(target = same_reading, daemon = True)
    worker.start()
    try:
        started = time.monotonic()
        resumed = policy.await_resume(timeout = timeout)
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        worker.join(timeout = 5)
        loop.call_soon_threadsafe(loop.stop)
    assert resumed is False
    assert elapsed < timeout * 4, f"a stalled cache must still end near the timeout, took {elapsed}s"
