# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A parallel round cut short drains and closes every call it launched, not only the first.

The round launches every call and then settles them in order. A cancel while the first is
settling used to tidy that one and leave its siblings running: their pumps never awaited and
their streams never closed, so nothing joined a tool still working after the request ended.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from core.inference import studio_tool_loop as loop_mod  # noqa: E402
from core.inference.studio_tool_loop import (  # noqa: E402
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)
from test_studio_tool_loop import WEB  # noqa: E402
from test_tool_calls_within_one_turn_overlap import _two_calls  # noqa: E402


def test_a_cancel_mid_round_drains_and_closes_the_siblings(monkeypatch):
    started = {"alpha": threading.Event(), "beta": threading.Event()}
    closed: list[str] = []

    def _execute(name, arguments, **kwargs):
        query = (arguments or {}).get("query", "")
        started[query].set()
        if query == "alpha":
            # The call being settled: it stops when told to.
            kwargs["cancel_event"].wait(5)
        else:
            # A sibling still working when the response is cancelled.
            time.sleep(0.5)
        return f"RESULT<{query}>"

    real_stream = loop_mod.stream_tool_execution

    def _recording_stream(invoke, **kwargs):
        inner = real_stream(invoke, **kwargs)

        class _Stream:
            def __iter__(self):
                return self

            def __next__(self):
                return next(inner)

            def close(self):
                closed.append(kwargs.get("tool_call_id"))
                inner.close()

        return _Stream()

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "stream_tool_execution", _recording_stream)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)

    cancel_event = threading.Event()

    async def _consume():
        agen = stream_with_studio_tools(
            _two_calls(),
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
            ),
            policy = ToolLoopPolicy(
                tools = [WEB],
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = cancel_event,
        )
        async for _line in agen:
            pass

    async def _main():
        task = asyncio.create_task(_consume())
        while not (started["alpha"].is_set() and started["beta"].is_set()):
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Read the moment the response task has ended: a stream not closed by now is one
        # nothing will close.
        return list(closed)

    closed_at_end = asyncio.run(_main())
    assert sorted(closed_at_end) == [
        "call_a",
        "call_b",
    ], f"the response ended with only {closed_at_end} of the round's streams closed"
