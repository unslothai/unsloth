# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two ways the external tool loop can lose model output or desync the UI.

Both were reproduced in a browser against an OpenAI-compatible mock server.

1. ``finish_reason: "length"``. Refusing to execute a possibly-truncated call is
   right, but promotion is destructive: the healer has already cut the
   ``<tool_call>...</tool_call>`` span out of the relayed text, so dropping the
   calls as well loses the call AND the sentence that introduced it. A small
   GGUF on llama-server with a modest ``max_tokens`` hits this routinely.

2. The budget/no-op branches close a tool card with ``tool_end`` and the replay
   with a ``role="tool"`` message. Every card the loop closes has to have been
   opened, and every ``role="tool"`` message has to be declared by a preceding
   assistant ``tool_calls`` entry, or OpenAI, DeepSeek and strict vLLM answer
   400 instead of continuing the conversation.
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


def _sse(
    delta = None,
    finish = None,
    **extra,
) -> str:
    choice: dict = {"index": 0, "delta": delta if delta is not None else {}}
    if finish is not None:
        choice["finish_reason"] = finish
    payload: dict = {"choices": [choice]}
    payload.update(extra)
    return "data: " + json.dumps(payload, ensure_ascii = False)


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


class FakeTransport:
    """Replays scripted turns; records what the loop asked for each time."""

    def __init__(
        self,
        turns,
        *,
        heals = True,
        max_turns = 20,
    ):
        self.turns = [list(turn) for turn in turns]
        self.heals_text_tool_calls = heals
        self.requests: list[dict] = []
        self.max_turns = max_turns

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        assert len(self.requests) <= self.max_turns, "loop never terminated"
        lines = self.turns.pop(0) if self.turns else [_DONE]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


@pytest.fixture
def executed(monkeypatch):
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments, **kwargs})
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return calls


def _run(transport, **policy_kwargs):
    fields = {
        "tools": [WEB],
        "max_calls": 25,
        "timeout": 300,
        "permission_mode": "off",
        "confirm_calls": False,
        "bypass_permissions": False,
        "rag_scope": None,
    }
    fields.update(policy_kwargs)

    async def _collect():
        out: list[str] = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
            ),
            policy = ToolLoopPolicy(**fields),
            cancel_event = threading.Event(),
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.run(asyncio.wait_for(_collect(), timeout = 30.0))


def _payloads(lines):
    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            continue
        if isinstance(payload, dict):
            yield payload


def _events(lines, kind):
    return [payload for payload in _payloads(lines) if payload.get("type") == kind]


def _visible_text(lines) -> str:
    text = []
    for payload in _payloads(lines):
        if isinstance(payload.get("type"), str):
            continue
        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str):
                text.append(content)
    return "".join(text)


def _call_turn(
    call_id = "c1",
    name = "web_search",
    arguments = '{"query":"q"}',
):
    return [
        _sse(
            {
                "tool_calls": [
                    {"index": 0, "id": call_id, "function": {"name": name, "arguments": arguments}}
                ]
            }
        ),
        _sse(finish = "tool_calls"),
        _DONE,
    ]


# ── 1. A truncated turn must not swallow a promoted call's own text ──


_HEALED_TURN = [
    _sse({"content": "Let me compute that. "}),
    _sse({"content": '<tool_call>{"name": "web_search", '}),
    _sse({"content": '"arguments": {"query": "42"}}</tool_call>'}),
    _sse({"content": " follow-up"}),
    _sse(finish = "length"),
    _DONE,
]


def test_truncated_healed_call_releases_its_own_markup(executed):
    """The user must still see what the model was attempting.

    Promotion cut the markup out of the relayed text before the loop learned the
    turn was truncated. Discarding the call then leaves the answer reading
    "Let me compute that.  follow-up" -- no card, no execution, and the request
    the model actually wrote is gone from the transcript.
    """
    lines = _run(FakeTransport([_HEALED_TURN]))

    assert executed == [], "a truncated call must never be executed"
    visible = _visible_text(lines)
    assert "Let me compute that. " in visible
    assert " follow-up" in visible
    assert (
        '<tool_call>{"name": "web_search", "arguments": {"query": "42"}}</tool_call>' in visible
    ), f"the promoted span was lost from the stream: {visible!r}"


def test_untruncated_healed_call_still_hides_its_markup(executed):
    """The release above is for truncation only: a normal turn still executes."""
    turn = list(_HEALED_TURN[:-2]) + [_sse(finish = "stop"), _DONE]
    lines = _run(FakeTransport([turn, [_sse({"content": "done"}), _sse(finish = "stop"), _DONE]]))

    assert [call["name"] for call in executed] == ["web_search"]
    assert "<tool_call>" not in _visible_text(lines)


def test_truncated_structured_call_relays_nothing_extra(executed):
    """A provider-emitted call had no markup removed, so there is none to give back."""
    turn = [
        _sse({"content": "thinking"}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "function": {"name": "web_search", "arguments": '{"que'},
                    }
                ]
            }
        ),
        _sse(finish = "length"),
        _DONE,
    ]
    lines = _run(FakeTransport([turn]))

    assert executed == []
    assert _visible_text(lines) == "thinking"
    # The delta was relayed as it arrived, so the client has a card for c1.
    # Refusing to run it is right; leaving it open for the rest of the response
    # is not, so it is closed the way every other unrun call is.
    assert _card_ids(lines, "tool_end") == _card_ids(lines, "tool_start") == ["c1"]
    assert "output limit" in _events(lines, "tool_end")[0]["result"]


def test_a_truncated_call_never_streamed_gets_no_card(executed):
    """A call recovered from text was never announced to the client, and the
    healer's released span is what tells the user about that one. Opening a card
    for it as well would report the same attempt twice."""
    turn = [
        _sse({"content": '<tool_call>{"name": "web_search", "arg'}),
        _sse(finish = "length"),
        _DONE,
    ]
    lines = _run(FakeTransport([turn]))

    assert executed == []
    assert _events(lines, "tool_start") == []
    assert _events(lines, "tool_end") == []


# ── 2. Cards and replayed messages must stay balanced ──────────────


def _card_ids(lines, kind):
    return [event.get("tool_call_id") for event in _events(lines, kind)]


def _overflow_turns():
    """One turn asking for two calls with one budget slot left.

    The shape matters. A turn whose ONLY call overflows ends the loop, so its
    replay is built and never sent; a turn where one call runs and the next
    overflows is followed by the budget-nudge turn, which is what carries the
    broken history to the provider. Parallel calls are ordinary for every model
    the loop serves.
    """
    return [
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "function": {"name": "web_search", "arguments": '{"query":"a"}'},
                        },
                        {
                            "index": 1,
                            "id": "c2",
                            "function": {"name": "web_search", "arguments": '{"query":"b"}'},
                        },
                    ]
                }
            ),
            _sse(finish = "tool_calls"),
            _DONE,
        ],
        [_sse({"content": "final"}), _sse(finish = "stop"), _DONE],
    ]


def test_budget_exhausted_call_does_not_close_a_card_it_never_opened(executed):
    """A tool_end with no tool_start closes a card the client never drew."""
    transport = FakeTransport(_overflow_turns())
    lines = _run(transport, max_calls = 1)

    assert len(executed) == 1, "the budget must still be enforced"
    assert _card_ids(lines, "tool_end") == _card_ids(
        lines, "tool_start"
    ), "every closed card must have been opened, in order"


def test_budget_exhausted_result_is_declared_by_an_assistant_tool_call(executed):
    """An orphan role="tool" message is a 400 from OpenAI, DeepSeek and vLLM."""
    transport = FakeTransport(_overflow_turns())
    _run(transport, max_calls = 1)

    assert len(transport.requests) > 1, "the overflow must reach a follow-up turn"
    for request in transport.requests:
        declared = {
            call.get("id")
            for message in request["messages"]
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        }
        orphans = [
            message["tool_call_id"]
            for message in request["messages"]
            if message.get("role") == "tool" and message.get("tool_call_id") not in declared
        ]
        assert not orphans, f"role=tool messages with no matching assistant tool_calls: {orphans}"


def test_disabled_call_card_is_opened_before_it_is_closed(executed):
    """Same invariant on the controller's no-op branch."""
    transport = FakeTransport(
        [_call_turn(name = "terminal"), [_sse({"content": "final"}), _sse(finish = "stop"), _DONE]]
    )
    lines = _run(transport, tools = [WEB])

    assert executed == []
    assert _card_ids(lines, "tool_end") == _card_ids(lines, "tool_start")
