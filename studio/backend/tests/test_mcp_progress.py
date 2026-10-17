# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import inspect
from types import SimpleNamespace
from typing import Optional

import pytest

from studio_mcp import progress
from studio_mcp.progress import with_heartbeat


class _Ctx:
    def __init__(self, fail: bool = False):
        self.reports = []
        self.fail = fail
        self.session_id = "session-1"

    async def report_progress(
        self,
        value,
        total = None,
        message = None,
    ):
        self.reports.append((value, total, message))
        if self.fail:
            raise RuntimeError("client went away")


@pytest.fixture
def fast_beat(monkeypatch):
    monkeypatch.setattr(progress, "HEARTBEAT_S", 0.01)


async def _until(ctx, count):
    while len(ctx.reports) < count:
        await asyncio.sleep(0.005)


def test_without_ctx_the_tool_runs_as_is():
    @with_heartbeat("Waiting")
    async def tool(x: int, ctx: Optional[object] = None) -> int:
        assert ctx is None
        return x + 1

    assert asyncio.run(tool(1)) == 2


def test_the_wrapper_keeps_the_tool_signature():
    async def tool(prompt: str, ctx: Optional[object] = None) -> str:
        """Doc."""
        return prompt

    wrapped = with_heartbeat("Waiting")(tool)
    assert inspect.signature(wrapped) == inspect.signature(tool)
    assert wrapped.__doc__ == "Doc." and wrapped.__name__ == "tool"


def test_the_heartbeat_reports_strictly_increasing_elapsed_seconds(monkeypatch, fast_beat):
    # A clock that never moves still yields increasing progress, which MCP requires.
    monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic = lambda: 100.0))
    ctx = _Ctx()

    @with_heartbeat("Waiting for the reply")
    async def tool(ctx = None):
        await _until(ctx._ctx, 4)
        return "done"

    assert asyncio.run(tool(ctx = ctx)) == "done"
    values = [value for value, _total, _message in ctx.reports]
    assert values == sorted(set(values)) and len(values) >= 4
    assert all(total is None for _value, total, _message in ctx.reports)
    assert all(message.startswith("Waiting for the reply, ") for *_, message in ctx.reports)
    assert ctx.reports[0][2].endswith(" s")


def test_the_tools_own_progress_becomes_the_heartbeat_note(fast_beat):
    ctx = _Ctx()

    @with_heartbeat("Transcribing")
    async def tool(ctx = None):
        await ctx.report_progress(0.25, 1.0, "Downloading whisper")
        await _until(ctx._ctx, 2)
        await ctx.report_progress(1, None)
        await _until(ctx._ctx, len(ctx._ctx.reports) + 2)
        # Other ctx attributes pass through.
        return ctx.session_id

    assert asyncio.run(tool(ctx = ctx)) == "session-1"
    messages = [message for *_, message in ctx.reports]
    assert any(m.startswith("Transcribing: Downloading whisper 25%, ") for m in messages)
    assert messages[-1].startswith("Transcribing: Working, ")
    # The tool's own fraction never reaches the client as progress.
    assert all(value != 0.25 for value, *_ in ctx.reports)


@pytest.mark.parametrize("raises", [False, True])
def test_the_heartbeat_stops_when_the_tool_ends(fast_beat, raises):
    ctx = _Ctx()

    @with_heartbeat("Generating audio")
    async def tool(ctx = None):
        await _until(ctx._ctx, 1)
        if raises:
            raise ValueError("route failed")
        return "ok"

    async def run():
        try:
            result = await tool(ctx = ctx)
        except ValueError:
            result = "raised"
        seen = len(ctx.reports)
        await asyncio.sleep(0.05)
        return result, seen

    result, seen = asyncio.run(run())
    assert result == ("raised" if raises else "ok")
    assert len(ctx.reports) == seen


def test_a_client_that_rejects_progress_does_not_fail_the_tool(fast_beat):
    ctx = _Ctx(fail = True)

    @with_heartbeat("Waiting")
    async def tool(ctx = None):
        await _until(ctx._ctx, 3)
        return "ok"

    assert asyncio.run(tool(ctx = ctx)) == "ok"
