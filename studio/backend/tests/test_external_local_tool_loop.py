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

    monkeypatch.setattr(
        "core.inference.external_agentic.wait_tool_decision", fake_wait_tool_decision
    )
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

    def fake_execute_tool(
        _name,
        _arguments,
        *_args,
        rag_scope = None,
        **_kwargs,
    ):
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


def _one_call_stream(
    tool_id,
    name = "web_search",
    args = '{"query":"q"}',
):
    return [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]


def test_assistant_content_retained_with_tool_call(monkeypatch):
    # Streamed explanation before a tool call must survive into the transcript.
    stream = [
        _sse_chunk(content = "Let me search. "),
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"q"}'},
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final = [_sse_chunk(content = "Done.", finish_reason = "stop"), "data: [DONE]"]
    client = _FakeClient([stream, final])
    monkeypatch.setattr(
        "core.inference.external_agentic.execute_tool",
        lambda *a, **k: "hit",
    )

    async def _run():
        async for _ in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "go"}],
            model = "m",
            tools = [WEB_SEARCH_TOOL],
        ):
            pass

    asyncio.run(_run())
    second = client.requests[1]["messages"]
    asst = next(m for m in second if m.get("role") == "assistant" and m.get("tool_calls"))
    assert asst["content"] == "Let me search. "


def test_no_limit_timeout_sentinel_passed_as_none(monkeypatch):
    captured = {}

    def fake_execute(name, arguments, cancel, timeout, *a, **k):
        captured["timeout"] = timeout
        return "ok"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute)
    client = _FakeClient(
        [_one_call_stream("c1"), [_sse_chunk(content = "x", finish_reason = "stop"), "data: [DONE]"]]
    )

    async def _run():
        async for _ in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "go"}],
            model = "m",
            tools = [WEB_SEARCH_TOOL],
            tool_call_timeout = 9999,
        ):
            pass

    asyncio.run(_run())
    assert captured["timeout"] is None


def test_parallel_calls_respect_per_message_budget(monkeypatch):
    # One completion with two parallel calls, budget = 1 -> only one executes.
    stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"a"}'},
                },
                {
                    "index": 1,
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"b"}'},
                },
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final = [_sse_chunk(content = "done", finish_reason = "stop"), "data: [DONE]"]
    client = _FakeClient([stream, final])
    calls = []
    monkeypatch.setattr(
        "core.inference.external_agentic.execute_tool",
        lambda name, arguments, *a, **k: calls.append(arguments) or "hit",
    )

    async def _run():
        async for _ in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "go"}],
            model = "m",
            tools = [WEB_SEARCH_TOOL],
            max_tool_iterations = 1,
        ):
            pass

    asyncio.run(_run())
    assert len(calls) == 1


def test_tool_events_carry_local_provenance(monkeypatch):
    client = _FakeClient(
        [_one_call_stream("c1"), [_sse_chunk(content = "x", finish_reason = "stop"), "data: [DONE]"]]
    )
    monkeypatch.setattr("core.inference.external_agentic.execute_tool", lambda *a, **k: "hit")

    async def _run():
        out = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "go"}],
            model = "m",
            tools = [WEB_SEARCH_TOOL],
        ):
            out.append(line)
        return out

    events = _parse_events(asyncio.run(_run()))
    for kind in ("tool_start", "tool_end"):
        ev = next(e for e in events if e.get("type") == kind)
        assert ev.get("provenance", {}).get("source") == "local"


def test_forced_tool_choice_downgraded_after_first_round(monkeypatch):
    client = _FakeClient(
        [_one_call_stream("c1"), [_sse_chunk(content = "done", finish_reason = "stop"), "data: [DONE]"]]
    )
    monkeypatch.setattr("core.inference.external_agentic.execute_tool", lambda *a, **k: "hit")
    forced = {"type": "function", "function": {"name": "web_search"}}

    async def _run():
        async for _ in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "go"}],
            model = "m",
            tools = [WEB_SEARCH_TOOL],
            tool_choice = forced,
        ):
            pass

    asyncio.run(_run())
    assert client.requests[0]["tool_choice"] == forced
    # Second round must be freed to synthesize the answer.
    assert client.requests[1]["tool_choice"] == "auto"


def test_stop_during_prefill_unblocks(monkeypatch):
    import threading as _t

    cancel = _t.Event()

    class _StallClient:
        requests: list = []

        async def stream_chat_completion(self, **kwargs):
            self.requests.append(kwargs)
            cancel.set()  # simulate Stop while we are blocked awaiting the next line
            # Never yields a line: the loop must abandon this read via cancel.
            while True:
                await asyncio.sleep(0.05)
            yield ""  # pragma: no cover

    client = _StallClient()

    async def _run():
        out = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "go"}],
            model = "m",
            tools = [WEB_SEARCH_TOOL],
            cancel_event = cancel,
        ):
            out.append(line)
        return out

    # Must return promptly rather than hang on the stalled read.
    out = asyncio.run(asyncio.wait_for(_run(), timeout = 5))
    assert isinstance(out, list)


def test_stop_during_final_synthesis_unblocks(monkeypatch):
    # The budget-exhausted final synthesis pass runs a distinct SSE reader. A Stop
    # delivered while that reader is blocked awaiting the remote provider's next line
    # must abandon the read immediately rather than hang until the provider responds.
    import threading as _t

    cancel = _t.Event()

    class _SynthStallClient:
        def __init__(self):
            self.calls = 0

        async def stream_chat_completion(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # First (and only budgeted) round emits one tool call, exhausting the budget.
                for chunk in _one_call_stream("c1"):
                    yield chunk
                return
            # Final synthesis pass: simulate a remote prefill stall and fire Stop. The
            # loop must abandon this read via cancel instead of blocking on __anext__.
            cancel.set()
            while True:
                await asyncio.sleep(0.05)
            yield ""  # pragma: no cover

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", lambda *a, **k: "hit")
    client = _SynthStallClient()

    async def _run():
        out = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "go"}],
            model = "m",
            tools = [WEB_SEARCH_TOOL],
            cancel_event = cancel,
            max_tool_iterations = 1,
        ):
            out.append(line)
        return out

    # Must return promptly rather than hang on the stalled final-synthesis read.
    out = asyncio.run(asyncio.wait_for(_run(), timeout = 5))
    assert isinstance(out, list)
    # Confirms the final synthesis pass was actually reached and then abandoned.
    assert client.calls == 2


RENDER_HTML_TOOL = {
    "type": "function",
    "function": {
        "name": "render_html",
        "description": "Render HTML to the canvas",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}


def _full_tool_call_stream(call_id, name, arguments_json):
    return [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_json},
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]


def test_stream_external_local_tool_loop_dedupes_repeated_tool_call(monkeypatch):
    # A remote model that repeats the same successful call in a later round must not
    # re-execute the side effect: the second identical call is an internal no-op, matching
    # the local GGUF/safetensors loops (#7282).
    client = _FakeClient(
        [
            _full_tool_call_stream("c1", "web_search", '{"query":"x"}'),
            _full_tool_call_stream("c2", "web_search", '{"query":"x"}'),  # identical
            [_sse_chunk(content = "done", finish_reason = "stop"), "data: [DONE]"],
        ]
    )
    calls = {"n": 0}

    def fake_execute_tool(name, arguments, *args, **kwargs):
        calls["n"] += 1
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
    # The duplicate call is not executed.
    assert calls["n"] == 1
    events = _parse_events(lines)
    # Only the real execution surfaces a visible tool card.
    assert len([e for e in events if e.get("type") == "tool_start"]) == 1
    assert len([e for e in events if e.get("type") == "tool_end"]) == 1
    # The third round still carries a role=tool reply for the deduped call_id so the
    # assistant tool_call stays matched, and its content is the duplicate no-op nudge.
    third_msgs = client.requests[2]["messages"]
    dup_tool_msg = next(
        m for m in third_msgs if m.get("role") == "tool" and m.get("tool_call_id") == "c2"
    )
    assert "not executed" in dup_tool_msg["content"]
    assert "identical call" in dup_tool_msg["content"]


def test_stream_external_local_tool_loop_dedupes_one_shot_render_html(monkeypatch):
    # render_html is one-shot: once it has run successfully, a later render_html call (even
    # with different args) is a no-op, mirroring the local loops (#7282).
    client = _FakeClient(
        [
            _full_tool_call_stream("r1", "render_html", '{"code":"<h1>hi</h1>"}'),
            _full_tool_call_stream("r2", "render_html", '{"code":"<h2>bye</h2>"}'),
            [_sse_chunk(content = "done", finish_reason = "stop"), "data: [DONE]"],
        ]
    )
    calls = {"n": 0}

    def fake_execute_tool(name, arguments, *args, **kwargs):
        calls["n"] += 1
        return "<rendered>"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "draw please"}],
            model = "remote-model",
            tools = [RENDER_HTML_TOOL],
            confirm_tool_calls = False,
        ):
            lines.append(line)
        return lines

    lines = asyncio.run(_run())
    assert calls["n"] == 1
    third_msgs = client.requests[2]["messages"]
    repeat_msg = next(
        m for m in third_msgs if m.get("role") == "tool" and m.get("tool_call_id") == "r2"
    )
    assert "render_html completed successfully earlier" in repeat_msg["content"]


def test_merge_system_nudge_preserves_structured_system_content():
    # Codex #7330: a leading system message with structured list content (text/image
    # parts preserved for vision-capable Connections) must stay a parts list -- the nudge
    # is appended as a text part, never str()-ed into a Python-repr string.
    from routes.inference import _merge_system_nudge

    loop_messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
        {"role": "user", "content": "hi"},
    ]
    out = _merge_system_nudge(loop_messages, "NUDGE")
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0] == {"type": "text", "text": "You are helpful."}
    assert out[0]["content"][-1] == {"type": "text", "text": "NUDGE"}
    # No Python repr of the list leaked into any text part.
    assert not any("[{" in str(p.get("text", "")) for p in out[0]["content"])
    # Later messages are untouched.
    assert out[1] == {"role": "user", "content": "hi"}


def test_merge_system_nudge_appends_to_string_system_content():
    from routes.inference import _merge_system_nudge
    out = _merge_system_nudge([{"role": "system", "content": "Base."}], "NUDGE")
    assert out[0]["content"] == "Base.\n\nNUDGE"


def test_merge_system_nudge_prepends_when_no_leading_system_message():
    from routes.inference import _merge_system_nudge

    out = _merge_system_nudge([{"role": "user", "content": "hi"}], "NUDGE")
    assert out[0] == {"role": "system", "content": "NUDGE"}
    assert out[1] == {"role": "user", "content": "hi"}


TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

RENDER_HTML_TOOL = {
    "type": "function",
    "function": {
        "name": "render_html",
        "description": "Render HTML",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}


def test_stream_external_local_tool_loop_executes_with_healed_arguments(monkeypatch):
    # A scalar/malformed function.arguments must run with the controller-healed
    # shape (terminal -> {"command": raw}), not the empty {} lenient JSON yields.
    tool_call_stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_term",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "ls -la"},
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

    seen = {}

    def fake_execute_tool(name, arguments, *args, **kwargs):
        seen["name"] = name
        seen["arguments"] = arguments
        return "listing"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "list files"}],
            model = "remote-model",
            tools = [TERMINAL_TOOL],
            confirm_tool_calls = False,
        ):
            lines.append(line)
        return lines

    events = _parse_events(asyncio.run(_run()))
    assert seen["name"] == "terminal"
    assert seen["arguments"] == {"command": "ls -la"}
    start = next(e for e in events if e.get("type") == "tool_start")
    assert start["arguments"] == {"command": "ls -la"}


def test_stream_external_local_tool_loop_stops_tools_after_terminal_noop(monkeypatch):
    # A repeated render_html (one-shot) declares the answer final; the loop must
    # switch to the tools-free synthesis pass, so no other tool round runs.
    render_call = lambda: [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_html",
                    "type": "function",
                    "function": {"name": "render_html", "arguments": '{"code":"<h1>hi</h1>"}'},
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final_stream = [
        _sse_chunk(content = "final answer", finish_reason = "stop"),
        "data: [DONE]",
    ]
    client = _FakeClient([render_call(), render_call(), final_stream])

    calls = {"count": 0}

    def fake_execute_tool(name, arguments, *args, **kwargs):
        calls["count"] += 1
        return "<h1>hi</h1>"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "render"}],
            model = "remote-model",
            tools = [RENDER_HTML_TOOL],
            confirm_tool_calls = False,
            max_tool_iterations = 5,
        ):
            lines.append(line)
        return lines

    asyncio.run(_run())
    # render_html ran once; the repeat was a controller no-op, not a second execution.
    assert calls["count"] == 1
    # Three remote calls: two tool rounds then the tools-free synthesis pass.
    assert len(client.requests) == 3
    assert client.requests[2]["tools"] is None
    assert client.requests[2]["tool_choice"] == "none"


def test_stream_external_local_tool_loop_unique_fallback_ids_across_rounds(monkeypatch):
    # A server that omits tool-call ids must not restart at call_0 every round;
    # duplicate ids collide in the transcript and the frontend tool-event map.
    def round_stream(query):
        return [
            _sse_chunk(
                tool_calls = [
                    {
                        "index": 0,
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": query}),
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
    client = _FakeClient([round_stream("a"), round_stream("b"), final_stream])

    monkeypatch.setattr(
        "core.inference.external_agentic.execute_tool",
        lambda name, arguments, *a, **k: f"hits for {arguments.get('query')}",
    )

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "search"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],
            confirm_tool_calls = False,
            max_tool_iterations = 5,
        ):
            lines.append(line)
        return lines

    events = _parse_events(asyncio.run(_run()))
    ids = [e["tool_call_id"] for e in events if e.get("type") == "tool_start"]
    assert ids == ["call_0_0", "call_1_0"]
    assert len(set(ids)) == len(ids)


def test_stream_external_local_tool_loop_honors_disabled_healing(monkeypatch):
    # auto_heal_tool_calls=False must leave a scalar/malformed function.arguments
    # unhealed (terminal -> {"raw": ...}) instead of recovering {"command": ...} and
    # executing it, so the caller's explicit opt-out is respected (GGUF parity).
    tool_call_stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_term",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "ls -la"},
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

    seen = {}

    def fake_execute_tool(name, arguments, *args, **kwargs):
        seen["name"] = name
        seen["arguments"] = arguments
        return "listing"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "list files"}],
            model = "remote-model",
            tools = [TERMINAL_TOOL],
            confirm_tool_calls = False,
            auto_heal_tool_calls = False,
        ):
            lines.append(line)
        return lines

    events = _parse_events(asyncio.run(_run()))
    # Not healed to {"command": "ls -la"}: the opt-out keeps the raw scalar shape.
    assert seen["arguments"] == {"raw": "ls -la"}
    start = next(e for e in events if e.get("type") == "tool_start")
    assert start["arguments"] == {"raw": "ls -la"}
    assert start["provenance"].get("healed") in (None, False)


def test_stream_external_local_tool_loop_budget_counts_only_executed(monkeypatch):
    # A disabled/no-op call must not consume the caller's tool budget: only real
    # executions count, so a later round can still run a genuine tool (GGUF parity).
    round0 = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_dis",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
                },
                {
                    "index": 1,
                    "id": "call_s1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"q1"}'},
                },
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    round1 = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_s2",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"q2"}'},
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
    client = _FakeClient([round0, round1, final_stream])

    executed = []

    def fake_execute_tool(name, arguments, *args, **kwargs):
        executed.append((name, arguments.get("query")))
        return f"hits for {arguments.get('query')}"

    monkeypatch.setattr("core.inference.external_agentic.execute_tool", fake_execute_tool)

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "search"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],  # python is not enabled -> a no-op
            confirm_tool_calls = False,
            max_tool_iterations = 2,
        ):
            lines.append(line)
        return lines

    asyncio.run(_run())
    # The disabled python no-op did not eat a budget slot, so both real searches ran.
    assert executed == [("web_search", "q1"), ("web_search", "q2")]


def test_stream_external_local_tool_loop_feeds_error_nudge(monkeypatch):
    # A failed tool result must carry the shared retry nudge back to the model so it
    # does not repeat the same bad call until the budget is exhausted (GGUF parity).
    from core.inference.tool_call_parser import TOOL_ERROR_NUDGE

    tool_call_stream = [
        _sse_chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_err",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"boom"}'},
                }
            ],
            finish_reason = "tool_calls",
        ),
        "data: [DONE]",
    ]
    final_stream = [
        _sse_chunk(content = "recovered", finish_reason = "stop"),
        "data: [DONE]",
    ]
    client = _FakeClient([tool_call_stream, final_stream])

    monkeypatch.setattr(
        "core.inference.external_agentic.execute_tool",
        lambda name, arguments, *a, **k: "Error: search backend unavailable",
    )

    async def _run():
        lines = []
        async for line in stream_external_local_tool_loop(
            client = client,
            messages = [{"role": "user", "content": "search"}],
            model = "remote-model",
            tools = [WEB_SEARCH_TOOL],
            confirm_tool_calls = False,
        ):
            lines.append(line)
        return lines

    asyncio.run(_run())
    # The follow-up round's tool message must include the retry guidance.
    second_msgs = client.requests[1]["messages"]
    tool_msgs = [m for m in second_msgs if m.get("role") == "tool"]
    assert tool_msgs
    assert any(TOOL_ERROR_NUDGE in m["content"] for m in tool_msgs)
    # The raw error text is still present for context.
    assert any("search backend unavailable" in m["content"] for m in tool_msgs)
