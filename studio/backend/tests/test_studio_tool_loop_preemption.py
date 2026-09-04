# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing a tool loop must not cost a tool call, or run one twice.

The ledger that makes this delicate is unrecoverable: ``ToolLoopController`` keeps
``_completed_one_shot_tools``, ``_successful_keys`` and ``_force_final_answer`` in
memory only, built once per response. Tearing the response down to pause it and
rebuilding it on resume would re-run a one-shot tool. So a pause holds off while
tools are running, and these prove it holds.

The transport is the same fake the rest of the loop's tests use, so what is
exercised here is the loop, not a mock of it.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from core.inference import llama_preemption as preemption
from core.inference import studio_tool_loop as loop_mod
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)


def _sse(
    delta = None,
    finish = None,
    **extra,
) -> str:
    choice: dict = {"index": 0, "delta": delta or {}}
    if finish is not None:
        choice["finish_reason"] = finish
    payload: dict = {"choices": [choice]}
    payload.update(extra)
    return "data: " + json.dumps(payload)


_DONE = "data: [DONE]"


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


WEB = _tool("web_search")


def _call_turn(name: str, call_id: str = "c1") -> list[str]:
    """A turn that asks for one tool call, shaped like llama.cpp's output."""
    return [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps({"query": "q"}),
                        },
                    }
                ]
            }
        ),
        _sse({}, finish = "tool_calls"),
        _DONE,
    ]


def _answer_turn(text: str = "done") -> list[str]:
    return [_sse({"content": text}), _sse({}, finish = "stop"), _DONE]


class PausingTransport:
    """Asks for a pause at a chosen moment, and records when it was honoured.

    ``request_on_turn`` is the provider turn (0-based) during whose stream the
    pause is asked for. That is the realistic shape: the admission side notices KV
    pressure while a stream is running, not between rounds.
    """

    heals_text_tool_calls = True
    sanitizes_provider_frames = False

    def __init__(
        self,
        turns,
        signal,
        request_on_turn: int = 0,
    ):
        self.turns = [list(turn) for turn in turns]
        self.signal = signal
        self.request_on_turn = request_on_turn
        self.requests: list[dict] = []
        # Whether the signal was visible when each turn's stream began.
        self.visible_at_turn_start: list[bool] = []

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        turn_index = len(self.requests)
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        self.visible_at_turn_start.append(self.signal.is_set())
        lines = self.turns.pop(0) if self.turns else [_DONE]
        should_request = turn_index == self.request_on_turn

        async def _gen():
            for line in lines:
                yield line
            if should_request:
                # Mid-stream, exactly where KV pressure is noticed.
                self.signal.request("kv_pressure")

        return _gen()


@pytest.fixture
def executed(monkeypatch):
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments})
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return calls


def _run(
    transport,
    *,
    signal,
    tools = None,
    **policy_kwargs,
):
    fields = {
        "tools": tools if tools is not None else [WEB],
        "max_calls": 25,
        "timeout": 300,
        "permission_mode": "off",
        "confirm_calls": False,
        "bypass_permissions": False,
        "rag_scope": None,
    }
    fields.update(policy_kwargs)

    async def _collect():
        out = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
                tool_choice = None,
            ),
            policy = ToolLoopPolicy(**fields),
            cancel_event = threading.Event(),
            preempt_signal = signal,
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.run(_collect())


class TestTheToolIsRunExactlyOnce:
    def test_a_pause_during_a_tool_turn_does_not_double_execute(self, executed):
        """The ledger case. A pause asked for while the calls of turn 0 are being
        executed must not cause turn 0's tool to run again."""
        signal = preemption.PreemptSignal()
        transport = PausingTransport(
            [_call_turn("web_search"), _answer_turn()],
            signal,
            request_on_turn = 0,
        )
        _run(transport, signal = signal)
        names = [call["name"] for call in executed]
        assert names == ["web_search"], f"expected one execution, got {names}"

    def test_the_pause_is_held_off_while_the_tool_runs(self, executed, monkeypatch):
        """Not merely "the result was right": the signal must have been invisible
        for the whole execution stretch, which is what makes it right."""
        signal = preemption.PreemptSignal()
        seen_during_execution: list[bool] = []

        def _execute(name, arguments, **kwargs):
            seen_during_execution.append(signal.is_set())
            return "ok"

        monkeypatch.setattr(loop_mod, "execute_tool", _execute)
        transport = PausingTransport(
            [_call_turn("web_search"), _answer_turn()],
            signal,
            request_on_turn = 0,
        )
        _run(transport, signal = signal)

        assert seen_during_execution, "the tool never ran"
        assert not any(seen_during_execution), "a pause was visible while a tool was executing"

    def test_the_request_is_deferred_not_dropped(self, executed):
        """Deferring is only safe because nothing is lost by it."""
        signal = preemption.PreemptSignal()
        transport = PausingTransport(
            [_call_turn("web_search"), _answer_turn()],
            signal,
            request_on_turn = 0,
        )
        _run(transport, signal = signal)
        assert signal.pending, "the pause request was silently discarded"


class TestWhereThePauseLands:
    def test_it_becomes_visible_before_the_next_stream(self, executed):
        """A round boundary is a safe point; the middle of tool execution is not."""
        signal = preemption.PreemptSignal()
        transport = PausingTransport(
            [_call_turn("web_search"), _answer_turn()],
            signal,
            request_on_turn = 0,
        )
        _run(transport, signal = signal)
        assert len(transport.visible_at_turn_start) >= 2
        assert transport.visible_at_turn_start[0] is False
        assert (
            transport.visible_at_turn_start[1] is True
        ), "the deferred pause should be visible by the next round's stream"

    def test_a_turn_with_no_calls_never_opens_a_window(self, executed):
        """Nothing is executing, so there is nothing to protect and a pause may
        land immediately."""
        signal = preemption.PreemptSignal()
        transport = PausingTransport([_answer_turn()], signal, request_on_turn = 0)
        _run(transport, signal = signal)
        assert signal.is_set()
        assert not signal.deferred


class TestWithoutASignal:
    def test_the_loop_is_unchanged(self, executed):
        """The parameter is optional and every existing caller omits it."""
        signal = preemption.PreemptSignal()
        transport = PausingTransport(
            [_call_turn("web_search"), _answer_turn()],
            signal,
            request_on_turn = 99,
        )

        async def _collect():
            out = []
            agen = stream_with_studio_tools(
                transport,
                run = ToolLoopRun(
                    messages = [{"role": "user", "content": "hi"}],
                    session_id = "s1",
                    thread_id = "t1",
                    tool_choice = None,
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
                cancel_event = threading.Event(),
            )
            async for line in agen:
                out.append(line)
            return out

        asyncio.run(_collect())
        assert [call["name"] for call in executed] == ["web_search"]
