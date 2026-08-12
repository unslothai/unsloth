# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the external-provider local tool loop.

Self-hosted OpenAI-compatible connections (llama.cpp / vLLM / Ollama / custom)
have no server-side tools, so ``run_external_provider_tool_loop`` streams a
chat completion carrying the local tool schemas, executes any ``tool_calls``
with Unsloth's local executor, and re-enters the model until a final answer.
These tests drive the loop against a mocked ``/chat/completions`` endpoint and
assert the flat local-loop event stream.
"""

import asyncio
import json
import threading
import time

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_agentic import (
    EXTERNAL_LOCAL_TOOL_PROVIDERS,
    run_external_provider_tool_loop,
)
from core.inference.external_provider import (
    ExternalProviderClient,
)
from state.tool_approvals import TOOL_REJECTED_MESSAGE, resolve_tool_decision

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _sse(events: list[dict]) -> bytes:
    chunks: list[str] = []
    for event in events:
        chunks.append(f"data: {json.dumps(event)}")
    return ("\n".join(chunks) + "\ndata: [DONE]\n").encode("utf-8")


def _mock_http_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


def _llama_cpp_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "llama_cpp",
        base_url = "http://127.0.0.1:8080/v1",
        api_key = "",
    )


async def _collect(loop):
    out = []
    async for event in loop:
        out.append(event)
    return out


# ── structured tool_calls ───────────────────────────────────────────


def test_structured_tool_call_executes_and_continues(monkeypatch):
    requests_seen = []

    def handler(request):
        body = json.loads(request.content)
        requests_seen.append(body)
        if len(requests_seen) == 1:
            # Model decides to call web_search; arguments arrive as fragments.
            events = [
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query": "current wea',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"arguments": 'ther"}'},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c2",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "It is sunny today."},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c2",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    executed = []

    def execute_tool(name, arguments, **kwargs):
        executed.append((name, arguments))
        return "WEATHER: sunny, 22C"

    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "What is the weather?"}],
                model = "qwen3:8b",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = execute_tool,
            )
        )
    )

    # One real tool execution, arguments healed across the streamed fragments.
    assert executed == [("web_search", {"query": "current weather"})]

    types = [event["type"] for event in events]
    assert types.count("tool_start") == 1
    assert types.count("tool_end") == 1
    assert "content" in types

    # The tool result must round-trip into the second request's history.
    second = requests_seen[1]["messages"]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second)
    assert any(m.get("role") == "tool" and "WEATHER" in (m.get("content") or "") for m in second)

    # The final answer streams to the client.
    final = "".join(event["text"] for event in events if event["type"] == "content")
    assert "sunny" in final


# ── text tool-call healing ──────────────────────────────────────────


def test_heals_tool_call_written_as_text(monkeypatch):
    requests_seen = []

    def handler(request):
        body = json.loads(request.content)
        requests_seen.append(body)
        if len(requests_seen) == 1:
            # Weak template: the model writes the call as <tool_call> markup.
            content = (
                'Let me search.<tool_call>{"name": "web_search", '
                '"arguments": {"query": "population of france"}}</tool_call>'
            )
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"content": content[:30]}, "finish_reason": None}
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"content": content[30:]}, "finish_reason": None}
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "France has ~68M people."},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    executed = []

    def execute_tool(name, arguments, **kwargs):
        executed.append((name, arguments))
        return "population ~68 million"

    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "Population of France?"}],
                model = "llama-3.2-3b",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = execute_tool,
            )
        )
    )

    assert executed == [("web_search", {"query": "population of france"})]
    types = [event["type"] for event in events]
    assert "tool_start" in types and "tool_end" in types
    final = "".join(event["text"] for event in events if event["type"] == "content")
    assert "68M" in final


# ── plain final answer, no tools ────────────────────────────────────


def test_final_answer_without_tool_calls(monkeypatch):
    def handler(request):
        events = [
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "Hello!"}, "finish_reason": None}],
            },
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    executed = []
    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "hi"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, arguments, **kwargs: (
                    executed.append((name, arguments)) or "x"
                ),
            )
        )
    )

    assert executed == []
    assert [event["type"] for event in events].count("content") == 1


# ── live stdout bridging ────────────────────────────────────────────


def test_streams_live_tool_output(monkeypatch):
    requests_seen = []

    def handler(request):
        requests_seen.append(json.loads(request.content))
        if len(requests_seen) == 1:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "python",
                                            "arguments": '{"code": "print(1+1)"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": "2"}, "finish_reason": None}],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    def execute_tool(
        name,
        arguments,
        output_callback = None,
        **kwargs,
    ):
        assert name == "python"
        if output_callback:
            output_callback("2")
        return "2"

    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "compute 1+1"}],
                model = "m",
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "python",
                            "description": "Run Python.",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                            },
                        },
                    }
                ],
                execute_tool = execute_tool,
            )
        )
    )

    assert any(event["type"] == "tool_output" and event["text"] == "2" for event in events)


# ── budget / provider set ───────────────────────────────────────────


def test_provider_set_matches_registry_studio_tools():
    # The backend loop gate must cover exactly the self-hosted OAI-compat
    # family, derived from the registry's provider-level studio_tools
    # capability (the "*" wildcard). The frontend reads the same flag via
    # providerModelSupportsStudioTools, so the gate and the UI pills share one
    # source of truth.
    from core.inference.providers import PROVIDER_REGISTRY

    expected = frozenset(
        pt
        for pt, info in PROVIDER_REGISTRY.items()
        if (info.get("model_capabilities") or {}).get("*", {}).get("studio_tools")
    )
    assert EXTERNAL_LOCAL_TOOL_PROVIDERS == expected
    assert EXTERNAL_LOCAL_TOOL_PROVIDERS == frozenset({"llama_cpp", "vllm", "ollama", "custom"})


def test_budget_exhausted_still_yields_final_answer(monkeypatch):
    requests_seen = []

    def handler(request):
        body = json.loads(request.content)
        requests_seen.append(body)
        # Once the loop runs out of budget it drops the tool schemas and asks
        # for a plain answer; serve that final pass here.
        if not body.get("tools"):
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "done after budget"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        else:
            # Distinct query per turn so the duplicate guard never short-circuits
            # the budget path.
            query = f"x{len(requests_seen)}"
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": f"call_{len(requests_seen)}",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": json.dumps({"query": query}),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    def execute_tool(name, arguments, **kwargs):
        return "some result"

    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "keep searching"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = execute_tool,
                max_tool_iterations = 2,
            )
        )
    )

    # Budget of 2 -> two tool turns then a plain final pass.
    assert len([e for e in events if e["type"] == "tool_start"]) == 2
    final = "".join(e["text"] for e in events if e["type"] == "content")
    assert "done after budget" in final


# ── security: disabled tool names are never executed ───────────────


def test_disabled_tool_name_never_executes(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        if body.get("tools"):
            # Model calls "python", which is NOT in the allow-list (only
            # web_search was enabled).
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "python",
                                            "arguments": '{"code": "rm -rf /"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "I cannot run code."},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    executed = []
    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "delete everything"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, arguments, **kwargs: (
                    executed.append((name, arguments)) or "x"
                ),
            )
        )
    )

    # The out-of-allow-list call is a no-op (never executed) and forces a
    # final no-tools pass.
    assert executed == []
    assert not any(e["type"] == "tool_start" for e in events)
    final = "".join(e["text"] for e in events if e["type"] == "content")
    assert "cannot run code" in final


# ── tool exception becomes a tool result, loop continues ───────────


def test_tool_exception_is_surfaced_as_result(monkeypatch):
    requests_seen = []

    def handler(request):
        body = json.loads(request.content)
        requests_seen.append(body)
        if len(requests_seen) == 1:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query": "x"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"content": "got it"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    def execute_tool(name, arguments, **kwargs):
        raise RuntimeError("boom")

    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "q"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = execute_tool,
            )
        )
    )

    # The exception is surfaced as a tool result (loop keeps going) and a
    # role=tool message carrying it reaches the next request.
    tool_end = [e for e in events if e["type"] == "tool_end"]
    assert len(tool_end) == 1
    assert "Error: tool raised an exception" in tool_end[0]["result"]
    second = requests_seen[1]["messages"]
    assert any(
        m.get("role") == "tool" and "Error: tool raised an exception" in (m.get("content") or "")
        for m in second
    )
    final = "".join(e["text"] for e in events if e["type"] == "content")
    assert "got it" in final


def test_tool_exception_with_cancel_event_does_not_abort_loop(monkeypatch):
    # Regression: a HANDLED tool exception must not set the request-scoped
    # cancel_event, or the loop would stop before the model's final answer.
    cancel_event = threading.Event()
    requests_seen = []

    def handler(request):
        requests_seen.append(json.loads(request.content))
        if len(requests_seen) == 1:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query": "x"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"content": "final answer"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    def execute_tool(name, arguments, **kwargs):
        raise RuntimeError("boom")

    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "q"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = execute_tool,
                cancel_event = cancel_event,
            )
        )
    )

    assert not cancel_event.is_set()
    assert len(requests_seen) == 2
    final = "".join(e["text"] for e in events if e["type"] == "content")
    assert "final answer" in final


# ── provider error SSE surfaces instead of hanging ─────────────────


def test_provider_error_sse_raises(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            content = 'data: {"error": {"message": "upstream 500", "type": "server_error"}}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def main():
        events = []
        with pytest.raises(RuntimeError, match = "upstream 500"):
            async for ev in run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "q"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, arguments, **kwargs: "x",
            ):
                events.append(ev)
        return events

    asyncio.run(main())


# ── duplicate calls within a turn are no-ops ───────────────────────


def test_duplicate_call_within_turn_is_noop(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        if body.get("tools"):
            # The model emits the SAME web_search call twice in one message.
            tc = {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "same"}'},
            }
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [tc]},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{**tc, "index": 1, "id": "call_2"}]},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"content": "answer"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    executed = []
    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "q"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, arguments, **kwargs: (
                    executed.append((name, arguments)) or "r"
                ),
            )
        )
    )

    assert executed == [("web_search", {"query": "same"})]
    # Exactly one visible card for the identical pair.
    assert len([e for e in events if e["type"] == "tool_start"]) == 1


# ── parallel tool calls accumulate per index ───────────────────────


def test_parallel_tool_calls(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        if body.get("tools"):

            def tc(idx, q):
                return {
                    "index": idx,
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": q})},
                }

            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [tc(0, "a"), tc(1, "b")]},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": None}],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    executed = []
    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "q"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, arguments, **kwargs: (
                    executed.append((name, arguments)) or "r"
                ),
            )
        )
    )

    assert sorted(executed, key = lambda item: item[1]["query"]) == [
        ("web_search", {"query": "a"}),
        ("web_search", {"query": "b"}),
    ]
    assert len([e for e in events if e["type"] == "tool_start"]) == 2


# ── structured content parts (Magistral-style deltas) ─────────────


def test_handles_structured_content_parts(monkeypatch):
    requests_seen = []

    def handler(request):
        body = json.loads(request.content)
        requests_seen.append(body)
        if len(requests_seen) == 1:
            # Content arrives as structured parts, not a plain string.
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": [
                                    {"type": "text", "text": "The weather is "},
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": [{"type": "text", "text": "sunny."}],
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Done."},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "hi"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, arguments, **kwargs: "r",
            )
        )
    )

    # The structured parts join into text without leaking a Python repr.
    final = "".join(e["text"] for e in events if e["type"] == "content")
    assert "The weather is sunny." in final
    assert "[" not in final and "{" not in final


# ── per-turn fan-out is capped like the local loops ───────────────


def test_fan_out_is_capped_per_turn(monkeypatch):
    requests_seen = []

    def handler(request):
        body = json.loads(request.content)
        requests_seen.append(body)
        if body.get("tools"):

            def tc(idx):
                return {
                    "index": idx,
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": f"q{idx}"}),
                    },
                }

            # 10 distinct parallel calls in ONE turn: the loop must cap fan-out.
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [tc(i) for i in range(10)]},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": None}],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    executed = []
    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "q"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, arguments, **kwargs: (
                    executed.append((name, arguments)) or "r"
                ),
            )
        )
    )

    # Only the first _MAX_TOOL_CALLS_PER_TURN distinct calls run.
    assert len(executed) == 8
    assert len([e for e in events if e["type"] == "tool_start"]) == 8
    final = "".join(e["text"] for e in events if e["type"] == "content")
    assert "done" in final


# ── interactive confirm gate (permission_mode=ask / confirm_tool_calls) ──


def _confirm_driver(
    monkeypatch,
    *,
    decision: str,
    session_id: str,
    confirm_tool_calls: bool = True,
    permission_mode: str = "ask",
):
    requests_seen = []

    def handler(request):
        body = json.loads(request.content)
        requests_seen.append(body)
        if len(requests_seen) == 1:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query": "current weather"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            # After the tool result round-trips (executed or denied), the model
            # answers plainly; a denied call must not loop forever.
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"content": "final answer"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    executed = []

    async def run():
        events = []
        loop = run_external_provider_tool_loop(
            client = _llama_cpp_client(),
            messages = [{"role": "user", "content": "q"}],
            model = "m",
            tools = [WEB_SEARCH_TOOL],
            execute_tool = lambda name, arguments, **kwargs: (
                executed.append((name, arguments)) or "r"
            ),
            session_id = session_id,
            confirm_tool_calls = confirm_tool_calls,
            permission_mode = permission_mode,
        )
        async for ev in loop:
            events.append(ev)
            if ev.get("type") == "tool_start" and ev.get("awaiting_confirmation"):
                resolve_tool_decision(ev["approval_id"], decision, session_id = session_id)
        return events

    events = asyncio.run(run())
    return events, executed, requests_seen


def test_confirm_gate_denies_call(monkeypatch):
    session = "sess-confirm-deny"
    events, executed, requests_seen = _confirm_driver(
        monkeypatch, decision = "deny", session_id = session
    )

    # The gated call parked on the approval slot and never executed.
    starts = [e for e in events if e["type"] == "tool_start"]
    assert len(starts) == 1
    assert starts[0]["awaiting_confirmation"] is True
    assert starts[0]["approval_id"]
    assert executed == []

    # The denial surfaces as a tool result, and the rejected message round-trips
    # into the next request so the model can adapt.
    denied = [e for e in events if e["type"] == "tool_end"]
    assert denied and TOOL_REJECTED_MESSAGE in denied[0]["result"]
    second = requests_seen[1]["messages"]
    assert any(
        m.get("role") == "tool" and TOOL_REJECTED_MESSAGE in (m.get("content") or "")
        for m in second
    )
    final = "".join(e["text"] for e in events if e["type"] == "content")
    assert "final answer" in final


def test_confirm_gate_allows_call(monkeypatch):
    session = "sess-confirm-allow"
    events, executed, _requests_seen = _confirm_driver(
        monkeypatch, decision = "allow", session_id = session
    )

    starts = [e for e in events if e["type"] == "tool_start"]
    assert len(starts) == 1
    assert starts[0]["awaiting_confirmation"] is True
    # Approved: the tool runs.
    assert executed == [("web_search", {"query": "current weather"})]
    assert len([e for e in events if e["type"] == "tool_end"]) == 1


def test_confirm_off_executes_without_pausing(monkeypatch):
    # confirm_tool_calls off -> no approval_id / awaiting_confirmation on the
    # start event, no pause, straight execution.
    session = "sess-confirm-off"
    events, executed, _requests_seen = _confirm_driver(
        monkeypatch, decision = "allow", session_id = session, confirm_tool_calls = False
    )

    starts = [e for e in events if e["type"] == "tool_start"]
    assert len(starts) == 1
    assert starts[0].get("awaiting_confirmation") is not True
    assert executed == [("web_search", {"query": "current weather"})]


# ── bypass_permissions reaches execute_tool as disable_sandbox ─────


def test_bypass_permissions_forwards_disable_sandbox(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        if body.get("tools"):
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "python",
                                            "arguments": '{"code": "print(1)"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                },
            ]
        else:
            events = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}],
                },
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    seen = {}

    def execute_tool(name, arguments, **kwargs):
        seen.update(kwargs)
        return "1"

    asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "q"}],
                model = "m",
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "python",
                            "description": "Run Python.",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                            },
                        },
                    }
                ],
                execute_tool = execute_tool,
                bypass_permissions = True,
            )
        )
    )
    assert seen.get("disable_sandbox") is True


# ── disconnect aborts a running tool via the shared cancel_event ───


def test_disconnect_aborts_running_tool(monkeypatch):
    cancel_event = threading.Event()
    tool_started = threading.Event()
    tool_returned = threading.Event()

    def handler(request):
        events = [
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "python",
                                        "arguments": '{"code": "while True: pass"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        return httpx.Response(
            200, content = _sse(events), headers = {"content-type": "text/event-stream"}
        )

    _mock_http_client(monkeypatch, handler)

    def execute_tool(
        name,
        arguments,
        cancel_event = None,
        **kwargs,
    ):
        tool_started.set()
        while cancel_event is None or not cancel_event.is_set():
            time.sleep(0.01)
        tool_returned.set()
        return "stopped early"

    loop = run_external_provider_tool_loop(
        client = _llama_cpp_client(),
        messages = [{"role": "user", "content": "q"}],
        model = "m",
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": "Run Python.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                },
            }
        ],
        execute_tool = execute_tool,
        cancel_event = cancel_event,
    )

    async def main():
        it = loop.__aiter__()
        # status("") -> status("Running Python: ...") -> tool_start
        for _ in range(3):
            ev = await it.__anext__()
        assert ev["type"] == "tool_start"
        # Drive into tool execution (the loop blocks awaiting the result) in a
        # separate task, and wait until the tool is genuinely running.
        driver = asyncio.create_task(it.__anext__())
        assert await asyncio.to_thread(tool_started.wait, 5)
        # SSE disconnect: cancel the driver. That delivers CancelledError into
        # the tool-execution bridge, which sets the shared cancel_event so the
        # running tool stops; aclose() then unwinds the generator.
        driver.cancel()
        try:
            await driver
        except asyncio.CancelledError:
            pass
        await loop.aclose()

    asyncio.run(main())

    assert cancel_event.is_set()
    # The running tool observed the cancellation and returned, instead of
    # executing "while True: pass" forever after the client left.
    assert tool_returned.is_set()


# ── pre-set cancel_event stops the loop before any request ─────────


def test_cancel_before_start_emits_no_request(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content = _sse([]), headers = {"content-type": "text/event-stream"})

    _mock_http_client(monkeypatch, handler)

    cancel_event = threading.Event()
    cancel_event.set()
    events = asyncio.run(
        _collect(
            run_external_provider_tool_loop(
                client = _llama_cpp_client(),
                messages = [{"role": "user", "content": "q"}],
                model = "m",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, arguments, **kwargs: "x",
                cancel_event = cancel_event,
            )
        )
    )
    assert events == []
    assert calls["n"] == 0
