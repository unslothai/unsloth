# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#19593: with enable_tools, a GGUF call to a request tool ends the turn as OpenAI tool_calls."""

from __future__ import annotations

import contextlib
import copy
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.inference as inference_route
from auth.authentication import get_current_subject
from core.inference.llama_cpp import LlamaCppBackend, client_tool_calls_in
from .llama_backend_double import FakeLlamaCppBackend


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }


WEB_SEARCH = _function_tool("web_search")
LIVE_CONTEXT = _function_tool("mcp_homeassistant__GetLiveContext")
CALL_ACTION = _function_tool("mcp_homeassistant__call_action")


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _tool_call(
    name: str,
    arguments: dict,
    call_id: str,
    index: int = 0,
) -> str:
    return _sse(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        }
    )


def _done() -> str:
    return "data: [DONE]\n"


def _backend(monkeypatch, streams: list[list[str]]):
    payloads: list[dict] = []
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48847
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
        payloads.append(copy.deepcopy(payload))
        yield type(
            "FakeResponse",
            (),
            {"status_code": 200, "chunks": streams.pop(0), "close": lambda self: None},
        )()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    return backend, payloads


def _executed(monkeypatch) -> list[str]:
    names: list[str] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        names.append(name)
        return "Home Assistant 2026.11 released"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)
    return names


def _loop(backend, **kwargs):
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Is the living room light on?"}],
            tools = [WEB_SEARCH, LIVE_CONTEXT, CALL_ACTION],
            client_tool_names = frozenset(
                {"mcp_homeassistant__GetLiveContext", "mcp_homeassistant__call_action"}
            ),
            max_tool_iterations = 5,
            permission_mode = "off",
            **kwargs,
        )
    )


def test_unsloth_tool_runs_then_a_client_tool_call_ends_the_turn(monkeypatch):
    backend, payloads = _backend(
        monkeypatch,
        [
            [_tool_call("web_search", {"query": "home assistant"}, "call_search"), _done()],
            [
                _sse({"content": "Checking the house. "}),
                _tool_call("mcp_homeassistant__GetLiveContext", {}, "call_ha"),
                _done(),
            ],
        ],
    )
    executed = _executed(monkeypatch)

    events = _loop(backend)

    # Both catalogs reach the model on every round.
    for payload in payloads:
        assert [tool["function"]["name"] for tool in payload["tools"]] == [
            "web_search",
            "mcp_homeassistant__GetLiveContext",
            "mcp_homeassistant__call_action",
        ]
    assert executed == ["web_search"]
    [handed_back] = [e for e in events if e["type"] == "client_tool_calls"]
    assert handed_back["tool_calls"] == [
        {
            "id": "call_ha",
            "type": "function",
            "function": {"name": "mcp_homeassistant__GetLiveContext", "arguments": "{}"},
        }
    ]
    assert handed_back["content"] == "Checking the house. "
    assert events[-1]["type"] == "metadata"
    assert events[-1]["finish_reason"] == "tool_calls"
    # The loop stops at the client call: no further round is decoded.
    assert len(payloads) == 2


def test_a_batch_with_a_client_call_runs_none_of_its_unsloth_calls(monkeypatch):
    backend, payloads = _backend(
        monkeypatch,
        [
            [
                _tool_call("web_search", {"query": "weather"}, "call_search", index = 0),
                _tool_call(
                    "mcp_homeassistant__call_action",
                    {"query": "light.living_room on"},
                    "call_action",
                    index = 1,
                ),
                _done(),
            ]
        ],
    )
    executed = _executed(monkeypatch)

    events = _loop(backend)

    assert executed == []
    [handed_back] = [e for e in events if e["type"] == "client_tool_calls"]
    assert [call["function"]["name"] for call in handed_back["tool_calls"]] == [
        "mcp_homeassistant__call_action"
    ]
    assert json.loads(handed_back["tool_calls"][0]["function"]["arguments"]) == {
        "query": "light.living_room on"
    }
    assert len(payloads) == 1


def test_a_healed_client_tool_name_goes_back_to_the_caller(monkeypatch):
    backend, payloads = _backend(
        monkeypatch,
        [[_tool_call("mcp_homeassistant__getlivecontext", {}, "call_ha"), _done()]],
    )
    executed = _executed(monkeypatch)

    events = _loop(backend)

    assert executed == []
    [handed_back] = [e for e in events if e["type"] == "client_tool_calls"]
    assert [call["function"]["name"] for call in handed_back["tool_calls"]] == [
        "mcp_homeassistant__GetLiveContext"
    ]
    assert len(payloads) == 1


def test_client_tool_results_continue_the_loop(monkeypatch):
    backend, payloads = _backend(
        monkeypatch, [[_sse({"content": "The living room light is on."}), _done()]]
    )
    _executed(monkeypatch)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "Is the living room light on?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_ha",
                            "type": "function",
                            "function": {
                                "name": "mcp_homeassistant__GetLiveContext",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_ha", "content": "light.living_room: on"},
            ],
            tools = [WEB_SEARCH, LIVE_CONTEXT],
            client_tool_names = frozenset({"mcp_homeassistant__GetLiveContext"}),
            max_tool_iterations = 5,
            permission_mode = "off",
        )
    )

    sent = payloads[0]["messages"]
    assert any(m.get("role") == "tool" and "light.living_room: on" in str(m) for m in sent)
    assert not any(e["type"] == "client_tool_calls" for e in events)
    assert any(
        e["type"] == "content" and "The living room light is on." in e["text"] for e in events
    )


def test_client_tool_calls_are_shaped_for_the_caller():
    calls = client_tool_calls_in(
        [
            {"id": "call_0", "function": {"name": "lookup", "arguments": {"q": "x"}}},
            {"id": "abc", "function": {"name": "web_search", "arguments": "{}"}},
            {"function": {"name": " lookup ", "arguments": None}},
        ],
        frozenset({"lookup"}),
    )

    assert [call["function"] for call in calls] == [
        {"name": "lookup", "arguments": '{"q": "x"}'},
        {"name": "lookup", "arguments": "{}"},
    ]
    # Per-turn text-parser ids repeat across turns, so they are replaced.
    assert all(call["id"].startswith("call_") and call["id"] != "call_0" for call in calls)
    assert calls[0]["id"] != calls[1]["id"]
    assert client_tool_calls_in([{"function": {"name": "lookup"}}], None) == []


def test_client_tools_beside_studio_tools_keeps_well_formed_new_names():
    payload = type(
        "Payload",
        (),
        {
            "tool_choice": None,
            "tools": [
                LIVE_CONTEXT,
                _function_tool("web_search"),
                {"type": "custom", "name": "apply_patch"},
                {"type": "function", "function": {"description": "no name"}},
                LIVE_CONTEXT,
            ],
        },
    )()

    kept = inference_route._client_tools_beside_studio_tools(payload, [WEB_SEARCH])

    assert [tool["function"]["name"] for tool in kept] == ["mcp_homeassistant__GetLiveContext"]
    payload.tool_choice = "none"
    assert inference_route._client_tools_beside_studio_tools(payload, [WEB_SEARCH]) == []


class _RecordingToolBackend(FakeLlamaCppBackend):
    supports_tools = True
    context_length = 8192

    def __init__(self):
        self.kwargs = None

    def generate_chat_completion_with_tools(self, **kwargs):
        self.kwargs = kwargs
        yield {"type": "content", "text": "Let me check. "}
        yield {"type": "status", "text": ""}
        yield {
            "type": "client_tool_calls",
            "content": "Let me check. ",
            "tool_calls": [
                {
                    "id": "call_ha",
                    "type": "function",
                    "function": {"name": "mcp_homeassistant__GetLiveContext", "arguments": "{}"},
                }
            ],
        }
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            "timings": {},
            "finish_reason": "tool_calls",
        }


def _route_client(monkeypatch, backend):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def fake_select(payload, **_kwargs):
        return [WEB_SEARCH]

    monkeypatch.setattr(inference_route, "_select_request_tools", fake_select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _request(stream: bool) -> dict:
    # The Open WebUI filter from the issue.
    return {
        "messages": [{"role": "user", "content": "Is the living room light on?"}],
        "tools": [LIVE_CONTEXT],
        "enable_tools": True,
        "enabled_tools": ["web_search"],
        "stream": stream,
        "permission_mode": "off",
    }


def test_non_streaming_route_offers_both_catalogs_and_returns_the_client_call(monkeypatch):
    backend = _RecordingToolBackend()

    response = _route_client(monkeypatch, backend).post(
        "/chat/completions", json = _request(stream = False)
    )

    assert response.status_code == 200
    assert [tool["function"]["name"] for tool in backend.kwargs["tools"]] == [
        "web_search",
        "mcp_homeassistant__GetLiveContext",
    ]
    assert backend.kwargs["client_tool_names"] == frozenset({"mcp_homeassistant__GetLiveContext"})
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"].strip() == "Let me check."
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_ha",
            "type": "function",
            "function": {"name": "mcp_homeassistant__GetLiveContext", "arguments": "{}"},
        }
    ]


def test_streaming_route_sends_tool_call_deltas_and_tool_calls_finish(monkeypatch):
    backend = _RecordingToolBackend()

    response = _route_client(monkeypatch, backend).post(
        "/chat/completions", json = _request(stream = True)
    )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [chunk["choices"][0] for chunk in chunks if chunk.get("choices")]
    tool_deltas = [d["delta"]["tool_calls"] for d in deltas if d["delta"].get("tool_calls")]
    assert tool_deltas == [
        [
            {
                "index": 0,
                "id": "call_ha",
                "type": "function",
                "function": {"name": "mcp_homeassistant__GetLiveContext", "arguments": "{}"},
            }
        ]
    ]
    assert deltas[-1]["finish_reason"] == "tool_calls"
    assert "Let me check." in "".join(d["delta"].get("content") or "" for d in deltas)
    assert response.text.rstrip().endswith("data: [DONE]")
