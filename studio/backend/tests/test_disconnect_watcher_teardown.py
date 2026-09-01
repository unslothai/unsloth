# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A disconnect watcher that never settles must not block a stream's teardown.

Starlette's Request.is_disconnected() awaits receive() inside an anyio CancelScope it has
already cancelled, and under uvicorn's h11 protocol that receive() genuinely parks. A cancel()
arriving in that window counts as a second cancellation, so the scope's __exit__ uncancel()s
and swallows the CancelledError as its own: the watcher survives cancel() and keeps polling.

Teardown that awaited such a watcher outright therefore blocked until the client itself
disconnected, holding the ASGI response open and stranding the llama-server admission lease
released behind it. #7617.

These tests wait on a task rather than asyncio.wait_for: a stub that ignores cancel() cannot
be unblocked by cancelling it, so a wait_for here would hang instead of failing.
"""

import ast
import asyncio
import inspect
import textwrap
import threading

import httpx
import pytest
from starlette.requests import Request

import routes.inference as inference_route

_TEARDOWN_CALLS = ("_aclose_send_task", "_aclose_stream_resources")


class _Closeable:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


def _bounded_stop(monkeypatch, collected):
    """Keep the real bounded stop, record what it was handed, shorten its 5s default."""
    real_stop = inference_route._stop_local_disconnect_cancel_watcher

    def _stop(task, timeout_s = 0.05):
        collected.append(task)
        return real_stop(task, timeout_s)

    monkeypatch.setattr(inference_route, "_stop_local_disconnect_cancel_watcher", _stop)


def _swallowing_watcher(release, created):
    """Stands in for a watcher suspended inside is_disconnected()'s cancelled scope.

    Records its own task so cleanup has a handle even when the teardown never hands it
    to the bounded stop.
    """

    async def _poll(*_args, **_kwargs):
        created.append(asyncio.current_task())
        while not release.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue

    return _poll


async def _finished(coro, timeout_s = 5.0):
    """Run coro to completion without cancelling it on timeout; report whether it ended."""
    task = asyncio.create_task(coro)
    done, _pending = await asyncio.wait({task}, timeout = timeout_s)
    return task, bool(done)


async def _release(release, tasks):
    """Must run even on failure: a stub that swallows cancel() wedges loop shutdown."""
    release.set()
    for task in tasks:
        if task is None:
            continue
        await asyncio.wait({task}, timeout = 2.0)
        if not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout = 2.0)


def test_aclose_stream_resources_is_not_blocked_by_a_wedged_watcher(monkeypatch):
    """_aclose_stream_resources runs in the stream's finally; it must stay bounded."""
    stopped = []
    _bounded_stop(monkeypatch, stopped)

    async def _run():
        release = asyncio.Event()
        watcher = asyncio.create_task(_swallowing_watcher(release, [])())
        teardown = None
        try:
            await asyncio.sleep(0)
            iterator, resp, client = _Closeable(), _Closeable(), _Closeable()

            teardown, finished = await _finished(
                inference_route._aclose_stream_resources(
                    watchers = (watcher,),
                    iterator = iterator,
                    resp = resp,
                    client = client,
                )
            )

            assert finished, "teardown blocked on a watcher that ignores cancel()"
            assert stopped == [watcher], "the watcher must go through the bounded stop"
            assert not watcher.done(), "watcher should have been abandoned, not awaited"
            assert (
                iterator.closed and resp.closed and client.closed
            ), "teardown must still close the upstream resources after abandoning it"
        finally:
            await _release(release, [watcher, teardown])

    asyncio.run(_run())


def test_send_stream_preheader_finally_is_not_blocked_by_a_wedged_watcher(monkeypatch):
    """The preheader finally runs on the success path too, before the first byte."""
    stopped = []
    _bounded_stop(monkeypatch, stopped)

    async def _run():
        release = asyncio.Event()
        created = []
        sent = httpx.Response(200)

        class _Client:
            async def send(
                self,
                req,
                stream = False,
            ):
                return sent

            async def aclose(self):
                pass

        class _Request:
            async def is_disconnected(self):
                return False

        monkeypatch.setattr(
            inference_route, "_wait_preheader_cancel", _swallowing_watcher(release, created)
        )
        send = None
        try:
            send, finished = await _finished(
                inference_route._send_stream_with_preheader_cancel(
                    _Client(),
                    httpx.Request("POST", "http://llama.test/v1/chat/completions"),
                    threading.Event(),
                    request = _Request(),
                )
            )

            assert finished, "preheader send blocked on a cancel task that ignores cancel()"
            assert send.result() is sent, "the upstream response must still be returned"
            assert stopped, "the preheader cancel task must go through the bounded stop"
            assert not stopped[0].done(), "cancel task should have been abandoned"
        finally:
            await _release(release, created + [send])

    asyncio.run(_run())


def test_real_disconnect_watcher_can_survive_cancel_inside_is_disconnected():
    """Pins the upstream behaviour the bounded teardown exists for.

    A real watcher on a real starlette Request, cancelled while suspended inside
    is_disconnected()'s already-cancelled anyio scope, keeps running. The window is a
    single loop iteration, so sweep the ticks around the poll rather than assume one.

    If this stops reproducing, the unbounded awaits are still wrong to restore: the point
    is that cancel() is not guaranteed to land here.
    """

    async def _survives_cancel_at(offset):
        parked = asyncio.Queue()
        release = asyncio.Event()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }

        async def receive():
            parked.put_nowait(None)
            await release.wait()
            return {"type": "http.disconnect"}

        watcher = asyncio.create_task(
            inference_route._await_disconnect_then_close(
                Request(scope, receive), _Closeable(), threading.Event()
            )
        )
        try:
            await parked.get()
            for _ in range(offset):
                await asyncio.sleep(0)
            watcher.cancel()
            done, _pending = await asyncio.wait({watcher}, timeout = 0.5)
            return not done
        finally:
            await _release(release, [watcher])

    async def _run():
        for offset in range(8):
            if await _survives_cancel_at(offset):
                return offset
        return None

    offset = asyncio.run(_run())
    if offset is None:
        pytest.skip("is_disconnected() delivered cancel() at every tick on this runtime")


def _blocks(tree):
    """Every statement list in the tree, so ordering is checked within one scope."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def _called_name(stmt):
    """The name a bare `foo(...)` / `await foo(...)` statement calls, else None.

    Structural rather than textual so an enclosing try/finally does not match on the calls
    nested inside it.
    """
    if not isinstance(stmt, ast.Expr):
        return None
    value = stmt.value
    if isinstance(value, ast.Await):
        value = value.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return value.func.id
    return None


def _first_index(block, names):
    for index, stmt in enumerate(block):
        if _called_name(stmt) in names:
            return index
    return None


@pytest.mark.parametrize(
    "func_name",
    ["_openai_passthrough_stream_admitted", "_anthropic_passthrough_stream"],
)
def test_admission_is_released_after_the_upstream_stream_is_closed(func_name):
    """The slot must be handed back only once the upstream response is closed.

    On disconnect llama-server keeps decoding until resp is closed, so releasing first
    admits another request past --parallel. The release must still always run, from the
    finally of the same try, which is bounded only because every teardown await is.
    """
    func = getattr(inference_route, func_name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))

    # a release must never sit ahead of a close in the same block
    for block in _blocks(tree):
        release_at = _first_index(block, ("_release_admission",))
        teardown_at = _first_index(block, _TEARDOWN_CALLS)
        if release_at is None or teardown_at is None:
            continue
        assert teardown_at < release_at, (
            f"{func_name}: _release_admission() runs before the upstream closes; on a "
            f"disconnect that hands the slot back while llama-server is still decoding:\n"
            + "\n".join(ast.unparse(stmt) for stmt in block)
        )

    # and every close must sit under a try whose finally releases, so a stalled close cannot
    # drop the lease. By ancestry, not direct membership: the nesting that stops a cancel in
    # _aclose_send_task's wait from skipping the later closes puts the first teardown one
    # level up, which membership would reject even though it is strictly safer.
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def _finally_releases(stmts):
        """A release anywhere under the finally counts, including nested in another try."""
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Expr) and _called_name(node) == "_release_admission":
                    return True
        return False

    def _protected(node):
        seen = node
        current = parents.get(node)
        while current is not None:
            if (
                isinstance(current, ast.Try)
                and _finally_releases(current.finalbody)
                # a teardown sitting *in* the releasing finally is not protected by it
                and not any(seen is stmt for stmt in current.finalbody)
            ):
                return True
            seen = current
            current = parents.get(current)
        return False

    teardowns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and _called_name(node) in _TEARDOWN_CALLS
    ]
    assert teardowns, f"{func_name}: found no teardown await to check"
    for node in teardowns:
        assert _protected(node), (
            f"{func_name}: teardown await is not inside a try that releases "
            f"admission in its finally:\n{ast.unparse(node)}"
        )


def test_release_admission_exits_the_tracker_even_if_the_lease_release_raises():
    """Both are process-wide; one failing must not strand the other."""
    exited = []

    class _Lease:
        def release(self):
            raise RuntimeError("queue gone")

    class _Tracker:
        def __exit__(self, *exc):
            exited.append(exc)
            return False

    with pytest.raises(RuntimeError):
        inference_route._release_admission(_Lease(), _Tracker())

    assert exited == [(None, None, None)], "tracker must still exit"
