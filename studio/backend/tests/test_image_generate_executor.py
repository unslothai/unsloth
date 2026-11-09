# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Queued image generations wait on their own pool, not the default executor every other route uses."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import routes.inference as inference

ENV = "UNSLOTH_DIFFUSION_GENERATE_EXECUTOR"
_ACCOUNT = contextvars.ContextVar("account", default = None)


def _queued_renders_then_a_status_read(n_renders: int, workers: int) -> tuple[float, list]:
    """``n_renders`` generations queued behind one held render slot, then a default-executor read (status)."""
    slot = threading.Lock()
    slot.acquire()
    seen = []

    def generate(i):
        seen.append((i, _ACCOUNT.get(), threading.current_thread().name))
        with slot:
            return i

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers = workers))
        _ACCOUNT.set("acct")
        renders = [
            asyncio.ensure_future(inference._image_generate_in_thread(generate, i))
            for i in range(n_renders)
        ]
        while len(seen) < min(n_renders, workers):
            await asyncio.sleep(0.01)
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(asyncio.to_thread(lambda: None), timeout = 0.5)
            waited = time.perf_counter() - t0
        except asyncio.TimeoutError:
            waited = float("inf")
        slot.release()
        assert sorted(await asyncio.gather(*renders)) == list(range(n_renders))
        return waited

    return asyncio.run(main()), seen


def test_queued_image_generations_leave_the_default_executor_free(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    waited, seen = _queued_renders_then_a_status_read(n_renders = 4, workers = 4)
    assert waited < 0.25
    assert all(
        account == "acct" for _i, account, _t in seen
    )  # the caller's context reaches the render
    assert all(name.startswith("image-generate") for _i, _a, name in seen)


def test_the_executor_switch_restores_asyncio_to_thread(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    assert inference._IMAGE_GENERATE_EXECUTOR_ENV == ENV
    waited, seen = _queued_renders_then_a_status_read(n_renders = 4, workers = 4)
    assert waited == float("inf")  # every default worker waits on the render slot
    assert all(account == "acct" for _i, account, _t in seen)


@pytest.mark.parametrize("raises", [False, True])
def test_results_and_errors_pass_through(monkeypatch, raises):
    monkeypatch.delenv(ENV, raising = False)

    def generate(x, *, y):
        if raises:
            raise RuntimeError("cancelled")
        return x + y

    async def main():
        return await inference._image_generate_in_thread(generate, 1, y = 2)

    if raises:
        with pytest.raises(RuntimeError, match = "cancelled"):
            asyncio.run(main())
    else:
        assert asyncio.run(main()) == 3
