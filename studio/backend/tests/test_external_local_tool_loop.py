# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the OAI-compat external local tool loop (#7282).

Remote Connections (Ollama / llama.cpp / vLLM / Custom) stream chat
completions from a remote host while Unsloth executes Search / Code / MCP
tools locally. These tests drive ``stream_external_local_tool_loop`` with a
fake client — no network, GPU, or model required.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.external_agentic import (
    _merge_tool_call_delta,
    provider_supports_local_tool_runtime,
    stream_external_local_tool_loop,
)


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _sse_chunk(
    *,
    content = None,
    tool_calls = None,
    finish_reason = None,
    model = "remote",
):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _parse_events(lines: list[str]) -> list[dict]:
    out = []
    for line in lines:
        text = line.strip()
        if text.startswith("data:"):
            text = text[5:].strip()
        if not text or text == "[DONE]":
            continue
        out.append(json.loads(text))
    return out


class _FakeClient:
    """Yields pre-canned SSE lines per stream_chat_completion call."""

    def __init__(self, streams: list[list[str]]):
        self.streams = list(streams)
        self.requests: list[dict] = []

    async def stream_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        for line in self.streams.pop(0):
            yield line


def test_provider_supports_local_tool_runtime_allowlist():
    for name in ("ollama", "llama_cpp", "vllm", "custom"):
        assert provider_supports_local_tool_runtime(name) is True
    for name in ("openai", "anthropic", "openrouter", "gemini", "kimi", None, ""):
        assert provider_supports_local_tool_runtime(name) is False


def test_merge_tool_call_delta_accumulates_name_and_arguments():
    acc: dict[int, dict] = {}
    _merge_tool_call_delta(
        acc,
        {"index": 0, "id": "call_1", "type": "function", "function": {"name": "web_"}},
    )
    _merge_tool_call_delta(acc, {"index": 0, "function": {"name": "search", "arguments": '{"q'}})
    _merge_tool_call_delta(acc, {"index": 0, "function": {"arguments": 'uery":"hi"}'}})
    assert acc[0]["id"] == "call_1"
    assert acc[0]["function"]["name"] == "web_search"
    assert acc[0]["function"]["arguments"] == '{"query":"hi"}'


def test_stream_external_local_tool_loop_content_only():
    client = _FakeClient(
        [
            [
                _sse_chunk(content = "hello "),
                _sse_chunk(content = "world", finish_reason = "stop"),
                "data: [DONE]",
            ]
        ]
    )

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "hi"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],
        ):
            lines.append(line)
        return lines

    lines = asyncio.run(_run())
    events = _parse_events(lines)
    contents = [
        (e.get("choices") or [{}])[0].get("delta", {}).get("content")
        for e in events
        if e.get("object") == "chat.completion.chunk"
    ]
    assert "hello " in contents
    assert "world" in contents
    assert lines[-1].strip() == "data: [DONE]"
    assert len(client.requests) == 1
    assert client.requests[0]["tools"] == [WEB_SEARCH_TOOL]


def test_stream_external_local_tool_loop_executes_and_continues(monkeypatch):
    # Round 1: model emits a tool call. Round 2: final answer after tool result.
    tool_call_stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": ""},
                }
            ]
        ),
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "function": {"arguments": '{"query":"unsloth tools"}'},
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final_stream = [
        _sse_chunk(content = "Found results.", finish_reason = "stop"),
        "data: [DONE]",
    ]
    client = _FakeClient([tool_call_stream, final_stream])

    def fake_execute_tool(name, arguments, *args, **kwargs):
        assert name == "web_search"
        assert arguments == {"query": "unsloth tools"}
        return "search hits: 3"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "search please"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],
            confirm_tool_calls = False,
        ):
            lines.append(line)
        return lines

    lines = asyncio.run(_run())
    events = _parse_events(lines)
    types = [e.get("type") for e in events if "type" in e]
    assert "tool_start" in types
    assert "tool_end" in types
    start = next(e for e in events if e.get("type") == "tool_start")
    end = next(e for e in events if e.get("type") == "tool_end")
    assert start["tool_name"] == "web_search"
    assert start["arguments"] == {"query": "unsloth tools"}
    assert end["result"] == "search hits: 3"
    contents = [
        (e.get("choices") or [{}])[0].get("delta", {}).get("content")
        for e in events
        if e.get("object") == "chat.completion.chunk"
    ]
    assert "Found results." in contents
    assert len(client.requests) == 2
    # Second round must include the assistant tool_calls + tool result.
    second_msgs = client.requests[1]["messages"]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second_msgs)
    assert any(
        m.get("role") == "tool" and m.get("content") == "search hits: 3" for m in second_msgs
    )


def test_stream_external_local_tool_loop_rejects_disabled_tool(monkeypatch):
    tool_call_stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_x",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": '{"code":"print(1)"}',
                    },
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final_stream = [
        _sse_chunk(content = "ok", finish_reason = "stop"),
        "data: [DONE]",
    ]
    client = _FakeClient([tool_call_stream, final_stream])
    called = {"n": 0}

    def fake_execute_tool(*_a, **_k):
        called["n"] += 1
        return "should not run"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "hi"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],  # python not enabled
            confirm_tool_calls = False,
        ):
            lines.append(line)
        return lines

    lines = asyncio.run(_run())
    assert called["n"] == 0
    events = _parse_events(lines)
    end = next(e for e in events if e.get("type") == "tool_end")
    assert "not enabled" in end["result"]


def test_stream_external_local_tool_loop_rejects_fabricated_mcp_tool(monkeypatch):
    tool_call_stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_mcp",
                    "type": "function",
                    "function": {
                        "name": "mcp__filesystem__read_file",
                        "arguments": '{"path":"/etc/passwd"}',
                    },
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final_stream = [
        _sse_chunk(content = "ok", finish_reason = "stop"),
        "data: [DONE]",
    ]
    client = _FakeClient([tool_call_stream, final_stream])
    called = {"n": 0}

    def fake_execute_tool(*_a, **_k):
        called["n"] += 1
        return "should not run"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "hi"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],  # MCP not enabled
            confirm_tool_calls = False,
        ):
            lines.append(line)
        return lines

    lines = asyncio.run(_run())
    assert called["n"] == 0
    events = _parse_events(lines)
    end = next(e for e in events if e.get("type") == "tool_end")
    assert "not enabled" in end["result"]


def test_stream_external_local_tool_loop_awaiting_confirmation(monkeypatch):
    tool_call_stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"test"}',
                    },
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final_stream = [
        _sse_chunk(content = "done", finish_reason = "stop"),
        "data: [DONE]",
    ]
    client = _FakeClient([tool_call_stream, final_stream])

    def fake_wait_tool_decision(_slot, _approval_id, _cancel_event):
        return "allow"

    monkeypatch.setattr("core.inference.external_agentic.wait_tool_decision", fake_wait_tool_decision)
    monkeypatch.setattr(
        "core.inference.external_agentic.execute_tool",
        lambda *_a, **_k: "ok",
    )

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "search"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],
            confirm_tool_calls = True,
            permission_mode = "ask",
        ):
            lines.append(line)
        return lines

    events = _parse_events(asyncio.run(_run()))
    start = next(e for e in events if e.get("type") == "tool_start")
    assert start.get("awaiting_confirmation") is True
    assert start.get("approval_id")


def test_stream_external_local_tool_loop_forwards_rag_scope(monkeypatch):
    tool_call_stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_rag",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": '{"query":"docs"}',
                    },
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final_stream = [
        _sse_chunk(content = "answer", finish_reason = "stop"),
        "data: [DONE]",
    ]
    client = _FakeClient([tool_call_stream, final_stream])
    rag_tool = {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search docs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    seen = {"rag_scope": None}

    def fake_execute_tool(_name, _arguments, *_args, rag_scope=None, **_kwargs):
        seen["rag_scope"] = _args[4] if len(_args) > 4 else rag_scope
        return "hits"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "search docs"}],
            model = "remote-model",
            tools = [rag_tool],
            rag_scope = {"thread_id": "t-ext-1"},
            confirm_tool_calls = False,
        ):
            lines.append(line)
        return lines

    asyncio.run(_run())
    assert seen["rag_scope"] == {"thread_id": "t-ext-1"}


def test_stream_external_local_tool_loop_preserves_sse_keepalives():
    client = _FakeClient(
        [
            [
                ": ping\n",
                _sse_chunk(content = "hi", finish_reason = "stop"),
                "data: [DONE]",
            ]
        ]
    )

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "hi"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],
        ):
            lines.append(line)
        return lines

    lines = asyncio.run(_run())
    assert any(line.startswith(": ping") for line in lines)


def test_proxy_ollama_enable_tools_attaches_local_tools(monkeypatch):
    """``_proxy_to_external_provider`` must enter the local tool loop for Ollama."""
    import routes.inference as inf_mod
    from models.inference import ChatCompletionRequest, ChatMessage
    from routes.inference import _proxy_to_external_provider

    class _Req:
        url = type("U", (), {"path": "/v1/chat/completions"})()
        method = "POST"
        state = type("S", (), {"skip_api_monitor": True})()

    seen = {"tools": None}

    class DummyExternalClient:
        def __init__(self, **_kwargs):
            pass

        async def stream_chat_completion(self, **kwargs):
            seen["tools"] = kwargs.get("tools")
            assert kwargs.get("tools"), "local tool loop must attach tools"
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "hi"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
            )
            yield "data: [DONE]"

        async def close(self):
            pass

    monkeypatch.setattr(inf_mod, "ExternalProviderClient", DummyExternalClient)
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "qwen3:8b",
        provider_type = "ollama",
        provider_base_url = "http://127.0.0.1:11434/v1",
        messages = [ChatMessage(role = "user", content = "hi")],
        stream = True,
        enable_tools = True,
        enabled_tools = ["web_search"],
    )

    async def _run():
        response = await _proxy_to_external_provider(payload, _Req())
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(chunks)

    text = asyncio.run(_run())
    assert "hi" in text
    assert "data: [DONE]" in text
    names = [
        (t.get("function") or {}).get("name") for t in (seen["tools"] or []) if isinstance(t, dict)
    ]
    assert "web_search" in names


def test_proxy_openai_enable_tools_stays_on_passthrough(monkeypatch):
    """Hosted OpenAI must not enter the local Studio tool loop."""
    import routes.inference as inf_mod
    from models.inference import ChatCompletionRequest, ChatMessage
    from routes.inference import _proxy_to_external_provider

    class _Req:
        url = type("U", (), {"path": "/v1/chat/completions"})()
        method = "POST"
        state = type("S", (), {"skip_api_monitor": True})()

    seen = {"tools": "unset", "enabled_tools": "unset"}

    class DummyExternalClient:
        def __init__(self, **_kwargs):
            pass

        async def stream_chat_completion(self, **kwargs):
            seen["tools"] = kwargs.get("tools")
            seen["enabled_tools"] = kwargs.get("enabled_tools")
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "hi"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
            )
            yield "data: [DONE]"

        async def close(self):
            pass

    monkeypatch.setattr(inf_mod, "ExternalProviderClient", DummyExternalClient)
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "gpt-4.1",
        provider_type = "openai",
        provider_base_url = "https://api.openai.com/v1",
        messages = [ChatMessage(role = "user", content = "hi")],
        stream = True,
        enable_tools = True,
        enabled_tools = ["web_search"],
    )

    async def _run():
        response = await _proxy_to_external_provider(payload, _Req())
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(chunks)

    text = asyncio.run(_run())
    assert "hi" in text
    # Pure proxy: client tools stay None; hosted builtins use enabled_tools.
    assert seen["tools"] is None
    assert seen["enabled_tools"] == ["web_search"]
