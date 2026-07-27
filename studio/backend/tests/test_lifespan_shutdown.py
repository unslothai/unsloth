# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for run_lifespan_shutdown: a dead default executor (the
abrupt-shutdown teardown race) must not abort the remaining cleanup. The helper
is dependency-injected, so these need only structlog."""

import asyncio
import contextvars
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
        # Kill the default executor to mimic the teardown race.
        await asyncio.to_thread(lambda: None)
        loop._default_executor.shutdown(wait = True)
        await run_lifespan_shutdown(terminate, clear, hw)

    asyncio.run(_drive())

    assert term_box["n"] == 1, "terminate must run via inline fallback"
    assert clear_box["n"] == 1, "clear must still run after the to_thread failure"
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_survives_shutdown_default_executor():
    """Production path: loop.shutdown_default_executor() makes run_in_executor raise
    'Executor shutdown has been called'; the helper must still recover inline."""
    term_box, terminate = _counter()
    clear_box, clear = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    async def _drive():
        await asyncio.get_running_loop().shutdown_default_executor()
        await run_lifespan_shutdown(terminate, clear, hw)

    asyncio.run(_drive())

    assert term_box["n"] == 1, "terminate must run via inline fallback"
    assert clear_box["n"] == 1
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_normal_path():
    """Healthy executor: each step runs exactly once."""
    term_box, terminate = _counter()
    clear_box, clear = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    asyncio.run(run_lifespan_shutdown(terminate, clear, hw))

    assert term_box["n"] == 1
    assert clear_box["n"] == 1
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_swallows_terminate_errors():
    """A terminate failure must not block later cleanup."""
    clear_box, clear = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    def _boom():
        raise ValueError("boom")

    asyncio.run(run_lifespan_shutdown(_boom, clear, hw))

    assert clear_box["n"] == 1, "later cleanup must run even when terminate raises"
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_swallows_clear_errors():
    """A clear failure must not raise out of shutdown."""
    term_box, terminate = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    def _boom():
        raise ValueError("boom")

    asyncio.run(run_lifespan_shutdown(terminate, _boom, hw))

    assert term_box["n"] == 1
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_does_not_retry_body_runtime_error():
    """A body-side RuntimeError (healthy executor) must run terminate once, not retry inline."""
    term_box, _ = _counter()
    clear_box, clear = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    def _boom():
        term_box["n"] += 1
        raise RuntimeError("body failed")

    asyncio.run(run_lifespan_shutdown(_boom, clear, hw))

    assert term_box["n"] == 1, "body RuntimeError must not be retried inline"
    assert clear_box["n"] == 1, "later cleanup must still run"
    assert hw.DEVICE is None


def test_run_lifespan_shutdown_preserves_contextvars():
    """terminate runs in a copy of the caller's context (parity with asyncio.to_thread)."""
    cv = contextvars.ContextVar("unsloth_test_cv")
    cv.set("bound-value")
    seen = []
    clear_box, clear = _counter()
    hw = types.SimpleNamespace(DEVICE = "cuda:0")

    def terminate():
        seen.append(cv.get("UNSET"))

    asyncio.run(run_lifespan_shutdown(terminate, clear, hw))

    assert seen == ["bound-value"], "terminate must run with the caller's contextvars"
    assert clear_box["n"] == 1
    assert hw.DEVICE is None
