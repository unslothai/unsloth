# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two ways a provider can present a call the loop must refuse to execute.

Withdrawing the catalog on the way out only tells a well-behaved provider what
not to do. These cover what happens when one asks anyway:

* ``tool_choice: "none"``. Deep Research sets it precisely so the scraped web
  text in its prompts cannot reach ``python`` or ``terminal``, so an endpoint
  that echoes a call back regardless must not be able to run one here.
* a turn that ended early. ``length`` hit the token ceiling and
  ``content_filter`` had the output cut by the provider, so in both cases the
  arguments collected so far may be half written.

``stop`` is deliberately absent from that second set: llama.cpp and vLLM
routinely finish a perfectly good tool call with it.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)


_DONE = "data: [DONE]"

WEB = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _call_line() -> str:
    return "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "x"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )


def _finish(reason: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


class FakeTransport:
    heals_text_tool_calls = False

    def __init__(
        self,
        turns,
        *,
        max_turns = 20,
    ):
        self.turns = [list(turn) for turn in turns]
        self.requests: list[dict] = []
        self.max_turns = max_turns

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append({"tools": tools, "tool_choice": tool_choice})
        assert len(self.requests) <= self.max_turns, "loop never terminated"
        lines = self.turns.pop(0) if self.turns else [_DONE]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


@pytest.fixture
def executed(monkeypatch):
    calls: list[str] = []

    def _execute(name, arguments, **kwargs):
        calls.append(name)
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return calls


def _run(transport, *, tool_choice = None):
    async def _collect():
        out: list[str] = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
                tool_choice = tool_choice,
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

    return asyncio.run(asyncio.wait_for(_collect(), timeout = 30))


# ── tool_choice: "none" is enforced, not just advertised ─────────────


def test_tool_choice_none_refuses_a_call_the_provider_sent_anyway(executed):
    """The Deep Research containment case.

    Its hops carry scraped third-party text, so a page that talks a naive
    endpoint into emitting a python call must not get one executed.
    """
    transport = FakeTransport([[_call_line(), _finish("tool_calls")], [_DONE]])
    _run(transport, tool_choice = "none")
    assert executed == []


def test_tool_choice_none_still_withdraws_the_catalog(executed):
    """The outbound half of the same contract must not have regressed."""
    transport = FakeTransport([[_call_line(), _finish("tool_calls")], [_DONE]])
    _run(transport, tool_choice = "none")
    assert transport.requests[0]["tool_choice"] == "none"


def test_tool_choice_auto_still_executes(executed):
    """The refusal must be specific to "none"."""
    transport = FakeTransport([[_call_line(), _finish("tool_calls")], [_DONE]])
    _run(transport, tool_choice = "auto")
    assert executed == ["web_search"]


# ── a turn that ended early is described, not run ────────────────────


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_a_turn_cut_short_does_not_execute_its_call(executed, reason):
    """Both endings mean the model never finished saying what it wanted."""
    transport = FakeTransport([[_call_line(), _finish(reason)], [_DONE]])
    _run(transport, tool_choice = "auto")
    assert executed == []


@pytest.mark.parametrize("reason", ["tool_calls", "stop"])
def test_a_completed_turn_still_executes(executed, reason):
    """ "stop" is how llama.cpp and vLLM commonly end a good tool call.

    Refusing it would disable tool calling on exactly the self-hosted servers
    this path exists to serve.
    """
    transport = FakeTransport([[_call_line(), _finish(reason)], [_DONE]])
    _run(transport, tool_choice = "auto")
    assert executed == ["web_search"]
