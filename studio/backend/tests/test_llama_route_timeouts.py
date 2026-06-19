# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import os
import sys
import time
from types import SimpleNamespace

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod  # noqa: E402


def test_non_streaming_generation_timeout_has_read_deadline():
    timeout = inf_mod._llama_non_streaming_generation_timeout()
    assert timeout.read == inf_mod._DEFAULT_FIRST_TOKEN_TIMEOUT_S


def test_stream_first_item_deadline_after_headers():
    async def _run():
        class _Never:
            async def __anext__(self):
                await asyncio.Future()

        started = time.monotonic()
        try:
            async for _ in inf_mod._aiter_llama_stream_items(
                _Never(),
                first_token_deadline = started + 0.02,
            ):
                pass
        except inf_mod.httpx.ReadTimeout:
            pass
        else:
            raise AssertionError("first item deadline did not fire")
        assert time.monotonic() - started < 0.5

    asyncio.run(_run())


def test_stream_first_item_deadline_does_not_hop_tasks():
    async def _run():
        outer_task = asyncio.current_task()
        seen_tasks = []

        class _One:
            def __init__(self):
                self.done = False

            async def __anext__(self):
                seen_tasks.append(asyncio.current_task())
                if self.done:
                    raise StopAsyncIteration
                self.done = True
                return "data: {}"

        out = []
        async for item in inf_mod._aiter_llama_stream_items(
            _One(),
            first_token_deadline = time.monotonic() + 1,
        ):
            out.append(item)

        assert out == ["data: {}"]
        assert seen_tasks == [outer_task, outer_task]

    asyncio.run(_run())


def test_preheader_send_cleanup_on_disconnect_and_cancel():
    async def _run(cancel_parent):
        state = SimpleNamespace(disconnected = False, closed = False, cancelled = False)
        started = asyncio.Event()

        class _Client:
            async def send(
                self,
                req,
                stream = False,
            ):
                started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    state.cancelled = True
                    raise

            async def aclose(self):
                state.closed = True

        class _Request:
            async def is_disconnected(self):
                return state.disconnected

        task = asyncio.create_task(
            inf_mod._send_stream_with_preheader_cancel(_Client(), object(), request = _Request())
        )
        await started.wait()
        if cancel_parent:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("helper cancellation did not propagate")
        else:
            state.disconnected = True
            assert await task is None
        assert state.closed
        assert state.cancelled

    asyncio.run(_run(False))
    asyncio.run(_run(True))
