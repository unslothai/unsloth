# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Arming reads ``/slots`` before planning. The route bodies that arm run on the event loop, and
the read used to be joined there for up to 250 ms, so every other request on the loop stalled
behind one admission. Now the async arm awaits the read in a worker and the loop keeps serving."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import PreemptSignal, get_preemption_controller
import routes.inference as inference_route


# The read cap stretched well past scheduler noise, so a loop held for it shows as a gap no
# descheduled worker produces, and the bound below stays falsifiable under -n 4.
_READ_CAP_S = 1.5
_MAX_GAP_S = _READ_CAP_S / 2


@pytest.fixture(autouse = True)
def _long_read_cap(monkeypatch):
    monkeypatch.setattr(inference_route, "_ARM_RESIDENCY_READ_S", _READ_CAP_S)


@pytest.fixture(autouse = True)
def _studio_owned(monkeypatch):
    monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
    monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, preemption.PREEMPT_MODE_STUDIO)
    preemption.note_server_parking(False)
    yield
    preemption.note_server_parking(False)


_ids = iter(range(1_000_000))


def _backend():
    return SimpleNamespace(
        base_url = f"http://loop-{next(_ids)}.test",
        context_length = 16384,
        _kv_cache_context_total = 16384,
        effective_parallel_slots = 4,
        _kv_cache_unified = True,
        server_preempts_kv = False,
    )


def _payload():
    return SimpleNamespace(
        messages = [{"role": "user", "content": "hi"}], max_tokens = None, max_completion_tokens = None
    )


async def _ticks_while(work, *, period = 0.01):
    """Run ``work`` beside a ticker; return (result, longest gap between ticks, elapsed)."""
    gaps = []
    done = asyncio.Event()

    async def ticker():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(period)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    ticking = asyncio.create_task(ticker())
    await asyncio.sleep(period * 3)
    started = time.monotonic()
    try:
        result = await work()
    finally:
        elapsed = time.monotonic() - started
        done.set()
        await ticking
    return result, max(gaps), elapsed


def _slow_probe(backend, released):
    controller = get_preemption_controller(inference_route._preempt_key(backend))

    def probe():
        released.wait(_READ_CAP_S * 2)

    controller.set_residency_probe(probe)
    return controller


def test_arming_awaits_the_slow_probe_without_stalling_the_loop():
    backend = _backend()
    released = threading.Event()

    async def run():
        _slow_probe(backend, released)
        reservation, _config = inference_route._openai_llama_admission_reserve(
            request = None, llama_backend = backend, payload = _payload(), counted_prompt_tokens = 100
        )

        async def arm():
            return await inference_route._openai_llama_preemption_arm_async(
                request = None,
                llama_backend = backend,
                reservation = reservation,
                gen_id = "chat-1",
                signal = PreemptSignal(),
                loop = asyncio.get_running_loop(),
            )

        try:
            return await _ticks_while(arm)
        finally:
            released.set()
            reservation.cancel()

    policy, worst_gap, elapsed = asyncio.run(run())
    assert policy is not None, "the chat must still be armed"
    # A tenth off for timer granularity: Windows' 15.6 ms tick ends the wait a tick early.
    assert elapsed >= inference_route._ARM_RESIDENCY_READ_S * 0.9, "the read was not waited for"
    assert worst_gap < _MAX_GAP_S, f"the loop stalled for {worst_gap:.3f}s while arming"


def test_counting_a_raw_holder_awaits_the_slow_probe_without_stalling_the_loop():
    backend = _backend()
    released = threading.Event()

    async def run():
        controller = _slow_probe(backend, released)
        reservation, _config = inference_route._openai_llama_admission_reserve(
            request = None, llama_backend = backend, payload = _payload(), counted_prompt_tokens = 100
        )

        async def count():
            await inference_route._openai_llama_count_raw_holder_async(
                llama_backend = backend, lease = reservation.lease_nowait(), gen_id = "raw-1"
            )
            return controller.participant("raw-1")

        try:
            return await _ticks_while(count)
        finally:
            released.set()
            reservation.cancel()

    holder, worst_gap, _elapsed = asyncio.run(run())
    assert holder is not None
    assert worst_gap < _MAX_GAP_S, f"the loop stalled for {worst_gap:.3f}s while counting"


def test_the_synchronous_read_on_the_loop_returns_at_once():
    """A caller on the loop that cannot await gets the read started, not joined."""
    backend = _backend()
    released = threading.Event()

    async def run():
        controller = _slow_probe(backend, released)
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        started = time.monotonic()
        inference_route._refresh_residency_before_planning(controller)
        return time.monotonic() - started

    try:
        assert asyncio.run(run()) < _MAX_GAP_S
    finally:
        released.set()
