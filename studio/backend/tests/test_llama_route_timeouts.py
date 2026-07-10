# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import os
import sys
import time
import threading
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


def test_stream_first_item_deadline_uses_compat_timeout_without_task_hop(monkeypatch):
    monkeypatch.setattr(inf_mod.asyncio, "timeout", None, raising = False)

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


def test_stream_wait_stops_on_known_disconnect_before_read():
    async def _run():
        state = SimpleNamespace(disconnect_checks = 0)
        cancel_event = threading.Event()

        class _Request:
            async def is_disconnected(self):
                state.disconnect_checks += 1
                return True

        class _Unread:
            async def __anext__(self):
                raise AssertionError("stream should stop before reading upstream")

        async for _ in inf_mod._aiter_llama_stream_items(
            _Unread(),
            cancel_event = cancel_event,
            request = _Request(),
            first_token_deadline = time.monotonic() + 1,
        ):
            raise AssertionError("stream should stop after disconnect")

        assert cancel_event.is_set()
        assert state.disconnect_checks == 1

    asyncio.run(_run())


def test_stream_wait_does_not_shorten_upstream_read_for_disconnect_poll():
    async def _run():
        response = SimpleNamespace(request = SimpleNamespace(extensions = {"timeout": {}}))
        seen_read_timeouts = []

        class _Request:
            async def is_disconnected(self):
                return False

        class _NoItem:
            async def __anext__(self):
                seen_read_timeouts.append(response.request.extensions["timeout"]["read"])
                raise StopAsyncIteration

        async for _ in inf_mod._aiter_llama_stream_items(
            _NoItem(),
            cancel_event = threading.Event(),
            request = _Request(),
            response = response,
            first_token_deadline = time.monotonic() + 1,
        ):
            raise AssertionError("stream should end")

        assert seen_read_timeouts
        assert seen_read_timeouts[0] > inf_mod._STREAM_DISCONNECT_POLL_TIMEOUT_S

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


def test_stream_stall_timeout_callable_re_resolved_each_read(monkeypatch):
    # The OpenAI passthrough passes a callable so the stall bound can switch to
    # the short post-terminal grace mid-stream; it must be re-resolved per read,
    # not captured once at generator start. Enforcement is the in-task
    # _same_task_timeout wrap around each read, so capture the bound it arms.
    armed = []
    real_timeout = inf_mod._same_task_timeout

    def _record(timeout_s):
        armed.append(timeout_s)
        return real_timeout(timeout_s)

    monkeypatch.setattr(inf_mod, "_same_task_timeout", _record)

    async def _run():
        values = iter([100.0, 2.0])

        class _Request:
            async def is_disconnected(self):
                return False

        class _Items:
            def __init__(self):
                self.count = 0

            async def __anext__(self):
                self.count += 1
                if self.count > 3:
                    raise StopAsyncIteration
                return "data: {}"

        out = []
        async for item in inf_mod._aiter_llama_stream_items(
            _Items(),
            cancel_event = threading.Event(),
            request = _Request(),
            first_token_deadline = time.monotonic() + 1,
            post_first_item_read_timeout_s = lambda: next(values, 5.0),
        ):
            out.append(item)

        assert out == ["data: {}"] * 3

    asyncio.run(_run())

    # First read is the first-token wait (~1s remaining). Each of the three
    # post-first reads re-resolves the callable, consuming 100.0, 2.0, then the
    # 5.0 default; a once-captured bound would repeat a single value.
    assert len(armed) == 4
    assert armed[0] <= 1.0
    assert armed[1:] == [100.0, 2.0, 5.0]


def test_stream_stall_timeout_disabled_skips_post_first_wrap(monkeypatch):
    # UNSLOTH_OPENAI_COMPAT_STREAM_STALL_TIMEOUT=0 disables the stall guard, so
    # the callable returns None and post-first reads run unwrapped: only the
    # first-token wait is bounded, no stall deadline the operator turned off.
    armed = []
    real_timeout = inf_mod._same_task_timeout

    def _record(timeout_s):
        armed.append(timeout_s)
        return real_timeout(timeout_s)

    monkeypatch.setattr(inf_mod, "_same_task_timeout", _record)

    async def _run():
        class _Request:
            async def is_disconnected(self):
                return False

        class _Items:
            def __init__(self):
                self.count = 0

            async def __anext__(self):
                self.count += 1
                if self.count > 2:
                    raise StopAsyncIteration
                return "data: {}"

        out = []
        async for item in inf_mod._aiter_llama_stream_items(
            _Items(),
            cancel_event = threading.Event(),
            request = _Request(),
            first_token_deadline = time.monotonic() + 5,
            post_first_item_read_timeout_s = lambda: None,
        ):
            out.append(item)

        assert out == ["data: {}"] * 2

    asyncio.run(_run())

    # Only the first-token read is wrapped; the disabled guard leaves every
    # post-first read unwrapped, so exactly one timeout is armed.
    assert len(armed) == 1
    assert armed[0] <= 5.0
