# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool result on a connection is sized to the window its server reports."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference import tools
from core.inference.context_window import estimate_messages_tokens_conservative
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)


_DONE = "data: [DONE]"
_LONG = "\n".join(f"row {i}: the order shipped on time" for i in range(500))

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


def _sse(delta = None, finish = None) -> str:
    choice: dict = {"index": 0, "delta": delta if delta is not None else {}}
    if finish is not None:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]})


def _call_turn(
    *call_ids,
    escaped = False,
    name = "web_search",
):
    return [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call_id,
                        "function": {
                            "name": name,
                            "arguments": json.dumps({"query": call_id}, ensure_ascii = escaped),
                        },
                    }
                    for index, call_id in enumerate(call_ids)
                ]
            }
        ),
        _sse(finish = "tool_calls"),
        _DONE,
    ]


_ANSWER_TURN = [_sse({"content": "done"}), _sse(finish = "stop"), _DONE]


class FakeTransport:
    heals_text_tool_calls = True

    def __init__(self, turns):
        self.turns = [list(turn) for turn in turns]
        self.requests: list[list[dict]] = []

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append([dict(message) for message in messages])
        self.tools = tools
        assert len(self.requests) <= 5, "loop never terminated"
        lines = self.turns.pop(0) if self.turns else [_DONE]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


@pytest.fixture(autouse = True)
def _long_search_results(monkeypatch):
    monkeypatch.setattr(tools, "_web_search", lambda *a, **k: _LONG)
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: None)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)


def _serves(window):
    async def _read():
        return window

    return _read


def _forwarded(
    turns,
    history = (),
    catalog = (WEB,),
    **run_kwargs,
):
    transport = turns if isinstance(turns, FakeTransport) else FakeTransport(turns)

    async def _collect():
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [*history, {"role": "user", "content": "which orders shipped?"}],
                session_id = "s1",
                thread_id = "t1",
                **run_kwargs,
            ),
            policy = ToolLoopPolicy(
                tools = list(catalog),
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        async for _ in agen:
            pass

    asyncio.run(asyncio.wait_for(_collect(), timeout = 30.0))
    return transport.requests[-1]


def _results(messages):
    return [message["content"] for message in messages if message.get("role") == "tool"]


def test_a_long_result_is_trimmed_to_the_served_window():
    messages = _forwarded([_call_turn("c1"), _ANSWER_TURN], served_window = _serves(4096))

    (result,) = _results(messages)
    assert len(result) < len(_LONG)
    assert result.startswith(_LONG[:200])
    assert f"{len(_LONG)} chars total" in result
    assert estimate_messages_tokens_conservative(messages) < 4096


def test_a_digit_heavy_result_fits_at_one_token_a_character(monkeypatch):
    digits = str(list(range(4000)))
    monkeypatch.setattr(tools, "_web_search", lambda *a, **k: digits)

    messages = _forwarded([_call_turn("c1"), _ANSWER_TURN], served_window = _serves(8192))

    (result,) = _results(messages)
    rest = [message for message in messages if message.get("role") != "tool"]
    assert len(result) + estimate_messages_tokens_conservative(rest) < 8192


def test_parallel_results_share_the_served_window():
    messages = _forwarded([_call_turn("c1", "c2", "c3"), _ANSWER_TURN], served_window = _serves(4096))

    results = _results(messages)
    assert len(results) == 3
    assert all(len(result) < len(_LONG) for result in results)
    assert estimate_messages_tokens_conservative(messages) < 4096


def test_a_window_reported_after_the_first_turn_still_sizes_the_result():
    transport = FakeTransport([_call_turn("c1"), _ANSWER_TURN])

    async def _loaded_by_the_first_turn():
        return 4096 if transport.requests else None

    messages = _forwarded(transport, served_window = _loaded_by_the_first_turn)

    (result,) = _results(messages)
    assert len(result) < len(_LONG)


@pytest.mark.parametrize("run_kwargs", [{}, {"served_window": _serves(None)}])
def test_a_result_is_forwarded_whole_when_the_window_is_unknown(run_kwargs):
    messages = _forwarded([_call_turn("c1"), _ANSWER_TURN], **run_kwargs)

    assert _results(messages) == [_LONG]


class CompactingTransport(FakeTransport):
    def stream(self, *, messages, **kwargs):
        self.sent_messages = messages[-1:] if messages[-1]["role"] == "user" else messages[-3:]
        return super().stream(messages = self.sent_messages, **kwargs)


def test_a_result_is_sized_to_the_history_left_after_compaction():
    history = [
        {"role": role, "content": f"turn {i} " + "x" * 2000}
        for i in range(20)
        for role in ("user", "assistant")
    ]
    transport = CompactingTransport([_call_turn("c1"), _ANSWER_TURN])

    messages = _forwarded(transport, history = history, served_window = _serves(4096))

    (result,) = _results(messages)
    assert result.startswith(_LONG[:200])
    assert estimate_messages_tokens_conservative(messages) < 4096


@pytest.mark.parametrize("escaped", [False, True])
def test_non_ascii_call_arguments_do_not_use_up_the_result_room(monkeypatch, escaped):
    reply = "字" * 600
    monkeypatch.setattr(tools, "_web_search", lambda *a, **k: reply)

    messages = _forwarded(
        [_call_turn("中" * 1000, escaped = escaped), _ANSWER_TURN], served_window = _serves(4096)
    )

    (result,) = _results(messages)
    assert result.startswith(reply[:200])
    assert estimate_messages_tokens_conservative(messages) < 4096


def test_tool_schemas_found_by_search_count_against_the_result_room(monkeypatch):
    from core.inference import tool_loop_controller as controller_mod

    def _spec(name, description = ""):
        return {
            "type": "function",
            "function": {**WEB["function"], "name": name, "description": description},
        }

    search = _spec("mcp_tool_search")
    found = _spec("mcp__srv1__ping", "pings the server " * 300)
    monkeypatch.setattr(controller_mod, "_mcp_tool_search_specs", lambda result: [found])
    monkeypatch.setattr(controller_mod, "_mcp_tool_spec", lambda name: found)
    monkeypatch.setattr(tools, "_web_search", lambda *a, **k: str(list(range(4000))))
    transport = FakeTransport(
        [_call_turn("s1", name = "mcp_tool_search"), _call_turn("c1"), _ANSWER_TURN]
    )

    messages = _forwarded(transport, catalog = (WEB, search), served_window = _serves(8192))

    assert found in transport.tools
    result = _results(messages)[-1]
    rest = [message for message in messages if message.get("content") != result]
    prompt = estimate_messages_tokens_conservative(rest + transport.tools)
    assert len(result) + prompt < 8192


def test_a_local_model_at_the_same_window_does_not_price_the_connection(monkeypatch):
    import sys
    import types

    class _LocalGguf:
        is_loaded = True
        context_length = 8192

        def count_chat_tokens(self, messages, *args, **kwargs):
            # A tokenizer that packs digits several to a token, unlike the served model's.
            return max(1, len(json.dumps(messages)) // 8)

    stub = sys.modules.get("routes.inference") or types.ModuleType("routes.inference")
    monkeypatch.setitem(sys.modules, "routes.inference", stub)
    monkeypatch.setattr(stub, "get_llama_cpp_backend", lambda: _LocalGguf(), raising = False)
    digits = str(list(range(4000)))
    monkeypatch.setattr(tools, "_web_search", lambda *a, **k: digits)

    messages = _forwarded([_call_turn("c1"), _ANSWER_TURN], served_window = _serves(8192))

    (result,) = _results(messages)
    rest = [message for message in messages if message.get("role") != "tool"]
    assert len(result) + estimate_messages_tokens_conservative(rest) < 8192
