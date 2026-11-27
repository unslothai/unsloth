# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Builds a mounted ASGI app on its first request, not at import; its lifespan runs inside the
parent's, always in one host task (anyio groups exit where entered) and exits at parent shutdown;
concurrent first requests share one build."""

from __future__ import annotations

import threading
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

import anyio
import anyio.to_thread


def combine_lifespans(*lifespans: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Enter lifespans in order, exit in reverse, merge the dicts they yield (later wins)."""

    @asynccontextmanager
    async def combined(app: Any) -> AsyncIterator[dict]:
        merged: dict = {}
        async with AsyncExitStack() as stack:
            for lifespan in lifespans:
                result = await stack.enter_async_context(lifespan(app))
                if result is not None:
                    merged.update(result)
            yield merged

    return combined


class _Cycle:
    def __init__(self) -> None:
        self.wanted = anyio.Event()
        self.ready = anyio.Event()
        self.stopping = anyio.Event()
        self.app: Any = None
        self.error: Optional[BaseException] = None


class LazyLifespanApp:
    def __init__(
        self,
        build: Callable[[], Any],
        *,
        wanted_at_startup: Callable[[], bool] = lambda: False,
    ) -> None:
        self._build = build
        self._wanted_at_startup = wanted_at_startup
        self._built: Any = None
        self._build_lock = threading.Lock()
        self._cycle: Optional[_Cycle] = None

    @property
    def built(self) -> bool:
        return self._built is not None

    def build(self) -> Any:
        """The inner app, built once whatever thread or event loop asks first."""
        built = self._built
        if built is not None:
            return built
        with self._build_lock:
            if self._built is None:
                self._built = self._build()
            return self._built

    def _startup_wants(self) -> bool:
        try:
            return bool(self._wanted_at_startup())
        except Exception:
            return False

    async def _host(self, cycle: _Cycle) -> None:
        await cycle.wanted.wait()
        if cycle.stopping.is_set():
            cycle.error = RuntimeError("The parent app is shutting down")
            cycle.ready.set()
            return
        entered = False
        try:
            app = self._built
            if app is None:
                app = await anyio.to_thread.run_sync(self.build)
            async with app.router.lifespan_context(app):
                entered = True
                cycle.app = app
                cycle.ready.set()
                await cycle.stopping.wait()
        except Exception as exc:
            # A failure while shutting down propagates like the eager lifespan's did.
            if entered:
                raise
            cycle.error = exc
            cycle.ready.set()

    async def _ready(self, cycle: _Cycle) -> Any:
        cycle.wanted.set()
        await cycle.ready.wait()
        if cycle.error is not None:
            raise RuntimeError("The mounted app failed to start") from cycle.error
        return cycle.app

    @asynccontextmanager
    async def lifespan(self, app: Any) -> AsyncIterator[None]:
        cycle = _Cycle()
        # Built already (an earlier lifespan cycle) or on at startup: build and start it now, as an
        # eagerly built app would, so startup still fails if it cannot.
        start_now = self._built is not None or await anyio.to_thread.run_sync(self._startup_wants)
        if start_now and self._built is None:
            await anyio.to_thread.run_sync(self.build)
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._host, cycle)
            self._cycle = cycle
            try:
                if start_now:
                    await self._ready(cycle)
                yield
            finally:
                if self._cycle is cycle:
                    self._cycle = None
                cycle.stopping.set()
                cycle.wanted.set()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        cycle = self._cycle
        if scope["type"] == "lifespan" or cycle is None or cycle.stopping.is_set():
            # No parent lifespan: the request passes on unstarted, as an eager build would.
            app = self._built
            if app is None:
                app = await anyio.to_thread.run_sync(self.build)
        else:
            app = await self._ready(cycle)
        await app(scope, receive, send)
