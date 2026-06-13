# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the FastAPI lifespan shutdown cleanup.

On an abrupt shutdown (closing the Windows console window, or interpreter
teardown racing uvicorn's graceful stop) the event loop's default thread-pool
executor can already be shut down by the time the lifespan shutdown runs, so
``asyncio.to_thread()`` raises ``RuntimeError: cannot schedule new futures after
shutdown``. Before the fix that raise was the first statement after ``yield``
and aborted the rest of the shutdown chain (propagating through every nested
``merged_lifespan`` ``__aexit__``), producing "Application shutdown failed.
Exiting." ``run_lifespan_shutdown`` must absorb that and still run the later
cleanup steps.

The helper is dependency-injected and free of the heavy backend import graph,
so these tests need only ``structlog`` — no torch / unsloth / fastapi install.
"""

import asyncio
import types

from utils.lifespan_shutdown import run_lifespan_shutdown


def _counter():
    box = {"n": 0}

    def _fn():
        box["n"] += 1

    return box, _fn


def test_run_lifespan_shutdown_survives_dead_default_executor():
    term_box, terminate = _counter()
    clear_box, clear = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    async def _drive():
        loop = asyncio.get_running_loop()
        # Instantiate then kill the default executor to mimic the teardown race
        # that made asyncio.to_thread() raise in the field report.
        await asyncio.to_thread(lambda: None)
        loop._default_executor.shutdown(wait = True)
        # Must NOT raise even though the executor backing to_thread is gone.
        await run_lifespan_shutdown(terminate, clear, hw)

    asyncio.run(_drive())

    # Inline fallback still terminated downloads, and later cleanup still ran.
    assert term_box["n"] == 1, "terminate must run via inline fallback"
    assert clear_box["n"] == 1, "clear must still run after the to_thread failure"
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_normal_path():
    """With a healthy executor the cleanup runs exactly once via the thread."""
    term_box, terminate = _counter()
    clear_box, clear = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    asyncio.run(run_lifespan_shutdown(terminate, clear, hw))

    assert term_box["n"] == 1
    assert clear_box["n"] == 1
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_swallows_terminate_errors():
    """A failure terminating downloads must not prevent later cleanup (no raise)."""
    clear_box, clear = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    def _boom():
        raise ValueError("boom")

    asyncio.run(run_lifespan_shutdown(_boom, clear, hw))

    assert clear_box["n"] == 1, "later cleanup must run even when terminate raises"
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_swallows_clear_errors():
    """A failure in the final cleanup step must not raise out of shutdown."""
    term_box, terminate = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    def _boom():
        raise ValueError("boom")

    # Should complete without raising.
    asyncio.run(run_lifespan_shutdown(terminate, _boom, hw))

    assert term_box["n"] == 1
    assert hw.DEVICE is None
