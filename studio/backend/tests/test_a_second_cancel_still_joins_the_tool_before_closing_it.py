# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A provider stream cancelled again while its cleanup waits for the running tool still joins
that tool's worker before closing the tool stream. Closed while the worker is inside ``next()``,
the generator raises ``ValueError: generator already executing`` and is never closed; joined
first, the stream closes and the cancellation propagates, as the per-step drain did."""

from __future__ import annotations

import asyncio
import threading

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools

from .test_studio_tool_loop import PY, _shared_setup_1


def test_a_second_cancel_during_cleanup_still_closes_the_tool_stream(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def _execute(name, arguments, **kwargs):
        started.set()
        # Ignores the cancel event, as a tool deep in a blocking call does.
        release.wait(timeout = 10)
        return "late"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    from state import tool_policy

    real_stream = tool_policy.account_tool_stream
    closed = []

    def _tracking(factory):
        scoped = real_stream(factory)

        def _open(*args, **kwargs):
            gen = scoped(*args, **kwargs)
            real_close = gen.close

            class _Tracked:
                def __iter__(self):
                    return self

                def __next__(self):
                    return next(gen)

                def send(self, value):
                    return gen.send(value)

                def close(self):
                    real_close()
                    closed.append(True)

            return _Tracked()

        return _open

    monkeypatch.setattr(tool_policy, "account_tool_stream", _tracking)

    async def _main():
        agen = stream_with_studio_tools(
            _shared_setup_1(),
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
            ),
            policy = ToolLoopPolicy(
                tools = [PY],
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )

        async def _consume():
            async for _ in agen:
                pass

        task = asyncio.create_task(_consume())
        while not started.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_main())
    assert closed == [True]
