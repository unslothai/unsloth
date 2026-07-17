# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for generator-close cleanup in the tool-streaming routes.

Tool streams run ``next(gen)`` in an ``asyncio.to_thread`` worker. Closing the
generator while that worker is still inside ``next`` raises ``ValueError:
generator already executing`` and skips the generator's ``finally`` (tool
cleanup); the routes drain the pending task first (``_drain_pending_next_task``),
which these tests exercise.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from routes.inference import _drain_pending_next_task


def test_drain_before_close_avoids_generator_already_executing():
    cancel_event = threading.Event()
    entered = threading.Event()
    finally_ran = threading.Event()

    def blocking_gen():
        try:
            entered.set()
            # Blocking call inside next(gen) that respects the cancel flag.
            cancel_event.wait()
            yield "value"
        finally:
            finally_ran.set()

    async def scenario():
        gen = blocking_gen()
        next_task = asyncio.create_task(asyncio.to_thread(next, gen, object()))
        await asyncio.to_thread(entered.wait)  # worker now inside next(gen)

        # Closing mid-next races and raises, leaving the finally unrun.
        with pytest.raises(ValueError):
            gen.close()
        assert not finally_ran.is_set()

        # Draining sets the cancel flag so the worker returns; then close is
        # clean and the generator's finally runs.
        await _drain_pending_next_task(next_task, cancel_event)
        gen.close()
        return

    asyncio.run(scenario())
    assert finally_ran.is_set()


def test_drain_pending_next_task_is_noop_without_task():
    # None (task already consumed): draining is a no-op, cancel flag untouched.
    cancel_event = threading.Event()

    asyncio.run(_drain_pending_next_task(None, cancel_event))
    assert not cancel_event.is_set()


def test_drain_pending_next_task_returns_when_worker_finishes():
    # A worker finishing on its own drains without error; the cancel flag stays
    # set (the caller is tearing the stream down).
    cancel_event = threading.Event()
    release = threading.Event()

    def gen():
        release.wait()
        yield "done"

    async def scenario():
        g = gen()
        task = asyncio.create_task(asyncio.to_thread(next, g, object()))
        release.set()  # let the worker complete before draining
        await _drain_pending_next_task(task, cancel_event)
        assert task.done()

    asyncio.run(scenario())
    assert cancel_event.is_set()
