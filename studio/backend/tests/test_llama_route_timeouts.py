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


class _Request:
    async def is_disconnected(self):
        return False


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


def test_stream_read_is_never_cancelled_to_implement_a_deadline():
    """Replaces a pair that asserted `__anext__` ran in the request task.

    That targeted `asyncio.wait_for`, which times out by CANCELLING the read --
    and httpcore closes the body on any streaming exception, so a cancelled read
    is a dead stream that surfaces as a truncated 200. Cancellation was the
    hazard, task identity only its proxy; emitting while a read is outstanding
    needs a task, so pin the hazard itself.
    """

    async def _run():
        cancels = []
        reads = []

        class _Items:
            def __init__(self):
                self.count = 0

            async def __anext__(self):
                reads.append(1)
                try:
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    cancels.append(1)
                    raise
                self.count += 1
                if self.count > 2:
                    raise StopAsyncIteration
                return "data: {}"

        out = []
        async for item in inf_mod._aiter_llama_stream_items(
            _Items(),
            first_token_deadline = time.monotonic() + 5,
            keepalive_interval_s = 0.01,
        ):
            if item is inf_mod._LLAMA_STREAM_KEEPALIVE:
                continue
            out.append(item)

        assert out == ["data: {}", "data: {}"]
        assert cancels == [], "a keepalive tick must never cancel the in-flight read"
        # One per item plus the StopAsyncIteration: same read, never restarted.
        assert len(reads) == 3, reads

    asyncio.run(_run())


def test_stream_pump_does_not_require_asyncio_timeout(monkeypatch):
    """The repo floor is 3.9, which has no `asyncio.timeout`. The rewrite needs
    only `ensure_future` + `wait`, so removing it must change nothing."""
    monkeypatch.setattr(inf_mod.asyncio, "timeout", None, raising = False)

    async def _run():
        class _One:
            def __init__(self):
                self.done = False

            async def __anext__(self):
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

    asyncio.run(_run())

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


def test_latched_read_ceiling_covers_the_stall_guard():
    """A lowered first-token env must not cap the whole body under it.

    httpcore reads `extensions["timeout"]["read"]` once, before the body loop
    (`_async/http11.py::_receive_response_body`), so the value armed before the
    FIRST read is the ceiling for every later one and the post-token re-arm
    cannot raise it. With UNSLOTH_OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT below the
    stall timeout, a healthy mid-stream gap would then be cut at the smaller
    first-token value instead of the configured stall guard.
    """
    async def _run():
        response = SimpleNamespace(request = SimpleNamespace(extensions = {"timeout": {}}))
        armed = []

        class _Items:
            def __init__(self):
                self.count = 0

            async def __anext__(self):
                armed.append(response.request.extensions["timeout"].get("read"))
                self.count += 1
                if self.count > 1:
                    raise StopAsyncIteration
                return "data: {}"

        async for _ in inf_mod._aiter_llama_stream_items(
            _Items(),
            response = response,
            first_token_deadline = time.monotonic() + 30.0,
            post_first_item_read_timeout_s = 120.0,
        ):
            pass

        # The latched (first) ceiling is the stall bound, not the 30s first-token
        # budget; the wall clock still enforces the 30s on its own.
        assert armed[0] == 120.0, armed

    asyncio.run(_run())


def test_deadline_does_not_discard_a_read_that_already_landed():
    """A token that arrives while the pump is suspended on a keepalive yield.

    Downstream backpressure can hold the generator past the deadline after the
    read has already completed. Timing out then would throw away a valid item
    and report a stall that did not happen, so the deadline is only allowed to
    fire on an empty `asyncio.wait`.
    """
    async def _run():
        gate = asyncio.Event()

        class _Items:
            def __init__(self):
                self.count = 0

            async def __anext__(self):
                self.count += 1
                if self.count > 1:
                    raise StopAsyncIteration
                await gate.wait()
                return "data: {}"

        out = []
        agen = inf_mod._aiter_llama_stream_items(
            _Items(),
            first_token_deadline = time.monotonic() + 0.2,
            keepalive_interval_s = 0.05,
        )
        async for item in agen:
            if item is inf_mod._LLAMA_STREAM_KEEPALIVE:
                # Let the read finish, then stay suspended here until the
                # first-token deadline has certainly gone by.
                gate.set()
                await asyncio.sleep(0.4)
                continue
            out.append(item)

        assert out == ["data: {}"], out

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


def test_stream_stall_timeout_callable_re_resolved_each_read():
    # The OpenAI passthrough passes a callable so the stall bound can switch to
    # the short post-terminal grace mid-stream; it must be re-resolved per read,
    # not captured once at generator start.
    async def _run():
        response = SimpleNamespace(request = SimpleNamespace(extensions = {"timeout": {}}))
        # The leading value is consumed by the pre-first-read ceiling, which now
        # resolves the post-token bound too so the latched socket timeout cannot
        # come in under it. The rest is the sequence this test is about.
        values = iter([50.0, 100.0, 2.0])
        seen = []

        class _Items:
            def __init__(self):
                self.count = 0

            async def __anext__(self):
                self.count += 1
                if self.count > 3:
                    raise StopAsyncIteration
                return "data: {}"

        async for _ in inf_mod._aiter_llama_stream_items(
            _Items(),
            cancel_event = threading.Event(),
            request = _Request(),
            response = response,
            first_token_deadline = time.monotonic() + 1,
            post_first_item_read_timeout_s = lambda: next(values, 5.0),
        ):
            seen.append(response.request.extensions["timeout"].get("read"))

        assert len(seen) == 3
        # The callable is resolved right after the first item (arming the
        # post-first window) and again before each later read, consuming
        # successive values.
        assert seen[0] == 100.0
        assert 1.0 <= seen[1] <= 2.0
        assert 4.0 <= seen[2] <= 5.0

    asyncio.run(_run())


def test_stream_stall_timeout_disabled_clears_read_timeout():
    # UNSLOTH_OPENAI_COMPAT_STREAM_STALL_TIMEOUT=0 disables the stall guard, so
    # the callable returns None. Once a chunk has arrived the leftover
    # first-token read timeout must be cleared, else a long post-first-chunk gap
    # trips a stale deadline the operator asked to turn off.
    async def _run():
        response = SimpleNamespace(request = SimpleNamespace(extensions = {"timeout": {}}))
        seen = []

        class _Items:
            def __init__(self):
                self.count = 0

            async def __anext__(self):
                self.count += 1
                if self.count > 2:
                    raise StopAsyncIteration
                return "data: {}"

        async for _ in inf_mod._aiter_llama_stream_items(
            _Items(),
            cancel_event = threading.Event(),
            request = _Request(),
            response = response,
            first_token_deadline = time.monotonic() + 5,
            post_first_item_read_timeout_s = lambda: None,
        ):
            seen.append(response.request.extensions["timeout"].get("read"))

        # With the guard off there is no socket ceiling at any point, including
        # the latched first read: the first-token wall clock is what bounds the
        # prefill, and a leftover finite read timeout would trip a deadline the
        # operator asked to turn off.
        assert seen == [None, None], seen

    asyncio.run(_run())
