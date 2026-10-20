# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A heartbeat for tools that wait on one long Unsloth Studio call (a chat reply, a song, a transcript), so the agent's client sees the call is alive instead of timing it out."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

HEARTBEAT_S = 2.0
NOTE_TTL_S = 3 * HEARTBEAT_S

_Fn = TypeVar("_Fn", bound = Callable[..., Awaitable[Any]])


class _Relay:
    """Stands in for the tool's ctx. MCP progress may only go up, so the tool's own reports (a download fraction) become the heartbeat's message instead of going out themselves."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._note: Optional[str] = None
        self._note_at = 0.0

    async def report_progress(
        self,
        progress: float,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        percent = f" {100 * progress / total:.0f}%" if total else ""
        self._note = f"{message or 'Working'}{percent}"
        self._note_at = time.monotonic()

    @property
    def note(self) -> Optional[str]:
        # A finished phase (a model download) sends no last report, so a quiet note has ended.
        if time.monotonic() - self._note_at > NOTE_TTL_S:
            return None
        return self._note

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)


async def _beat(ctx: Any, relay: _Relay, label: str, started: float) -> None:
    last = 0.0
    while True:
        await asyncio.sleep(HEARTBEAT_S)
        elapsed = max(round(time.monotonic() - started, 1), last + 0.1)
        last = elapsed
        note = f": {relay.note}" if relay.note else ""
        try:
            await ctx.report_progress(elapsed, None, f"{label}{note}, {elapsed:.0f} s")
        except Exception:
            # A client that went away or never asked for progress is not the tool's failure.
            pass


def with_heartbeat(label: str) -> Callable[[_Fn], _Fn]:
    """Report elapsed time every 2 s while the tool runs. The tool must take ``ctx`` as a keyword; FastMCP passes it that way and still sees the Context parameter through ``functools.wraps``."""

    def decorate(fn: _Fn) -> _Fn:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = kwargs.get("ctx")
            if ctx is None:
                return await fn(*args, **kwargs)
            relay = _Relay(ctx)
            # A sibling task: the tool itself keeps running in the caller's task and context.
            beat = asyncio.create_task(_beat(ctx, relay, label, time.monotonic()))
            try:
                return await fn(*args, **{**kwargs, "ctx": relay})
            finally:
                beat.cancel()

        return wrapper  # type: ignore[return-value]

    return decorate
