# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for generator-close cleanup in the tool-streaming routes.

The safetensors and Anthropic tool streams run ``next(gen)`` in a worker thread
via ``asyncio.to_thread``. On disconnect/cancellation that worker may still be
inside ``next(gen)`` (blocked in a web/MCP/generator call). Closing the
generator while the thread is executing it raises
``ValueError: generator already executing`` and leaves the generator's
``finally`` (tool/resource cleanup) unrun -- the exact GGUF-vs-others asymmetry
these routes now fix by draining the pending task first
(``_drain_pending_next_task``). These tests exercise the shared helper against a
generator that blocks inside ``next`` exactly as the routes do.
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
            # Simulate a blocking call inside next(gen) that respects the
            # generation cancel flag (as a real web/MCP tool step would).
            cancel_event.wait()
            yield "value"
        finally:
            finally_ran.set()

    async def scenario():
        gen = blocking_gen()
        next_task = asyncio.create_task(asyncio.to_thread(next, gen, object()))
        # Wait until the worker is actually inside next(gen).
        await asyncio.to_thread(entered.wait)

        # Regression guard: closing the generator while the worker is inside
        # next(gen) races and raises, leaving the finally-cleanup unrun.
        with pytest.raises(ValueError):
            gen.close()
        assert not finally_ran.is_set()

        # The fix: drain the pending next(gen) task (which sets the cancel flag
        # so the worker returns), THEN close cleanly. No ValueError, and the
        # generator's finally runs, releasing tool/resources.
        await _drain_pending_next_task(next_task, cancel_event)
        gen.close()
        return

    asyncio.run(scenario())
    assert finally_ran.is_set()


def test_drain_pending_next_task_is_noop_without_task():
    # The happy path passes None (the task reference was cleared after its
    # result was consumed); draining must be a safe no-op that never touches the
    # cancel flag.
    cancel_event = threading.Event()

    asyncio.run(_drain_pending_next_task(None, cancel_event))
    assert not cancel_event.is_set()


def test_drain_pending_next_task_returns_when_worker_finishes():
    # A worker that finishes on its own (StopIteration -> sentinel) is drained
    # without error and the cancel flag is still set (the caller is tearing the
    # stream down).
    cancel_event = threading.Event()
    release = threading.Event()

    def gen():
        release.wait()
        yield "done"

    async def scenario():
        g = gen()
        task = asyncio.create_task(asyncio.to_thread(next, g, object()))
        # Let the worker complete on its own before draining.
        release.set()
        await _drain_pending_next_task(task, cancel_event)
        assert task.done()

    asyncio.run(scenario())
    assert cancel_event.is_set()
