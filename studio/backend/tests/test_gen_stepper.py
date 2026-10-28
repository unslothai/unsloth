# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ThreadStepper matches asyncio.to_thread(next, gen, default) step for step."""

import asyncio
import contextvars
import threading
import time

import pytest

from core.inference import gen_stepper
from core.inference.gen_stepper import ThreadStepper

_VAR = contextvars.ContextVar("_VAR", default = "unset")


def _drive(step, n):
    async def run():
        return [await step(None) for _ in range(n)]

    return asyncio.run(run())


def test_steps_match_to_thread():
    def gen():
        yield from ("a", "b", "c")

    async def via_to_thread(default):
        return await asyncio.to_thread(next, base, default)

    base = gen()
    assert (
        _drive(ThreadStepper(gen()).next, 5)
        == _drive(via_to_thread, 5)
        == ["a", "b", "c", None, None]
    )


def test_one_thread_runs_every_step_in_order():
    seen = []

    def gen():
        for i in range(50):
            seen.append(threading.get_ident())
            yield i

    assert _drive(ThreadStepper(gen()).next, 51) == list(range(50)) + [None]
    assert len(set(seen)) == 1 and seen[0] != threading.get_ident()


def test_exception_propagates_and_generator_finishes():
    def gen():
        yield 1
        raise KeyError("boom")

    async def run():
        step = ThreadStepper(gen()).next
        first = await step(None)
        with pytest.raises(KeyError):
            await step(None)
        return first, await step("done")

    assert asyncio.run(run()) == (1, "done")


def test_context_is_copied_per_step():
    def gen():
        while True:
            yield _VAR.get()

    async def run():
        step = ThreadStepper(gen()).next
        _VAR.set("one")
        a = await step(None)
        _VAR.set("two")
        return a, await step(None)

    assert asyncio.run(run()) == ("one", "two")


def test_cancelled_wait_abandons_the_step_only():
    release = threading.Event()

    def gen():
        release.wait(5)
        yield "slow"
        yield "next"

    async def run():
        step = ThreadStepper(gen()).next
        task = asyncio.create_task(step(None))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        # The abandoned step still ran; its value is dropped, as with to_thread.
        return await step(None)

    assert asyncio.run(run()) == "next"


def test_idle_thread_exits_and_restarts(monkeypatch):
    monkeypatch.setattr(gen_stepper, "_IDLE_EXIT_S", 0.05)

    def gen():
        yield from range(3)

    stepper = ThreadStepper(gen())

    async def run():
        first = await stepper.next(None)
        deadline = time.monotonic() + 5
        while stepper._running and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert not stepper._running
        return first, await stepper.next(None)

    assert asyncio.run(run()) == (0, 1)


def test_idle_thread_does_not_keep_a_dropped_generator_alive():
    closed = threading.Event()

    def gen():
        try:
            yield from range(10)
        finally:
            closed.set()

    async def run():
        stepper = ThreadStepper(gen())
        first = await stepper.next(None)
        del stepper
        return first

    assert asyncio.run(run()) == 0
    import gc

    gc.collect()
    # The idle worker waits up to _IDLE_EXIT_S, but must not hold the generator meanwhile.
    assert closed.wait(2)


def test_dropping_a_stepper_frees_its_thread_before_the_idle_timeout():
    import gc

    def workers():
        return [t for t in threading.enumerate() if t.name == "studio-gen-step"]

    # By identity, not count: a worker woken by an earlier test may still be exiting.
    before = set(workers())
    started = []

    async def run():
        steppers = [ThreadStepper(iter(range(3))) for _ in range(32)]
        assert [await stepper.next(None) for stepper in steppers] == [0] * 32
        started.extend(t for t in workers() if t not in before)
        assert len(started) == 32
        steppers.clear()

    asyncio.run(run())
    gc.collect()
    # _IDLE_EXIT_S is 30 s; a dropped stepper's worker is woken instead of left to wait it out.
    deadline = time.monotonic() + 5
    for t in started:
        t.join(max(0.0, deadline - time.monotonic()))
    alive = [t for t in started if t.is_alive()]
    assert not alive, len(alive)
