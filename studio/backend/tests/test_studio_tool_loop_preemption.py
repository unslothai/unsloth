# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing a tool loop must not cost a tool call, or run one twice."""

from __future__ import annotations

import asyncio
import json
import threading


from core.inference import llama_preemption as preemption
from core.inference import studio_tool_loop as loop_mod

from .preempt_fakes import executed  # noqa: F401
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
    """Asks for a pause at a chosen moment, and records when it was honoured."""

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
            **({} if signal is None else {"preempt_signal": signal}),
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.run(_collect())


def _one_round(signal, *, request_on_turn = 0):
    return PausingTransport(
        [_call_turn("web_search"), _answer_turn()], signal, request_on_turn = request_on_turn
    )


class TestTheToolIsRunExactlyOnce:
    def test_a_pause_during_a_tool_turn_does_not_double_execute(self, executed):
        signal = preemption.PreemptSignal()
        _run(_one_round(signal), signal = signal)
        names = [call["name"] for call in executed]
        assert names == ["web_search"], f"expected one execution, got {names}"

    def test_the_pause_is_held_off_while_the_tool_runs_and_is_not_dropped(
        self, executed, monkeypatch
    ):
        signal = preemption.PreemptSignal()
        seen_during_execution: list[bool] = []

        def _execute(name, arguments, **kwargs):
            seen_during_execution.append(signal.is_set())
            return "ok"

        monkeypatch.setattr(loop_mod, "execute_tool", _execute)
        _run(_one_round(signal), signal = signal)

        assert seen_during_execution, "the tool never ran"
        assert not any(seen_during_execution), "a pause was visible while a tool was executing"
        assert signal.pending, "the pause request was silently discarded"


class TestWhereThePauseLands:
    def test_it_becomes_visible_before_the_next_stream(self, executed):
        signal = preemption.PreemptSignal()
        transport = _one_round(signal)
        _run(transport, signal = signal)
        assert len(transport.visible_at_turn_start) >= 2
        assert transport.visible_at_turn_start[0] is False
        assert (
            transport.visible_at_turn_start[1] is True
        ), "the deferred pause should be visible by the next round's stream"


class TestWithoutASignal:
    def test_the_loop_is_unchanged(self, executed):
        signal = preemption.PreemptSignal()
        _run(_one_round(signal, request_on_turn = 99), signal = None)
        assert [call["name"] for call in executed] == ["web_search"]
