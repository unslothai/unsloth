# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for the OpenAI `/v1/responses` translation in external_provider.

Covers:
- Request body shape: system messages collapse into `instructions`,
  user/assistant messages go into `input`, and unsupported sampling knobs
  (presence_penalty, top_k) are not forwarded.
- SSE translation: `response.output_text.delta` → Chat Completions chunks,
  `response.completed` → a `finish_reason: stop` chunk, stream ends with
  `data: [DONE]`.
- Image parts rewritten from Chat Completions
  `{type: image_url, image_url: {url}}` to Responses
  `{type: input_image, image_url: <url>}`.
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    out = []
    async for line in agen:
        out.append(line)
    return out


def _mock_http_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "openai",
        base_url = "https://api.openai.com/v1",
        api_key = "sk-test",
    )


def _responses_sse(events: list[dict]) -> bytes:
    """Serialize a list of Responses-API event dicts as an SSE byte stream."""
    chunks: list[str] = []
    for event in events:
        chunks.append(f"event: {event['type']}")
        chunks.append(f"data: {json.dumps(event)}")
        chunks.append("")
    chunks.append("data: [DONE]")
    chunks.append("")
    return ("\n".join(chunks) + "\n").encode("utf-8")


def test_responses_request_body_uses_input_and_instructions(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Hi"},
            ],
            model = "gpt-5.5",
            temperature = 0.5,
            top_p = 0.9,
            max_tokens = 512,
            enable_thinking = None,
            reasoning_effort = None,
        ):
            pass
        await client.close()

    _drive(run())

    assert captured["url"] == "https://api.openai.com/v1/responses"
    body = captured["body"]
    assert body["model"] == "gpt-5.5"
    assert body["instructions"] == "You are concise."
    assert body["input"] == [{"role": "user", "content": "Hi"}]
    assert body["max_output_tokens"] == 512
    assert body["stream"] is True
    # Responses API on reasoning-class models (gpt-5.x / o3 / gpt-4.5 — the only
    # OpenAI ids the registry allowlist exposes) rejects these as `Unsupported
    # parameter`. Never silently forward them.
    assert "temperature" not in body
    assert "top_p" not in body
    assert "presence_penalty" not in body
    assert "frequency_penalty" not in body
    assert "top_k" not in body
    assert "messages" not in body


def test_responses_translates_image_parts(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        },
                    ],
                }
            ],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = None,
            enable_thinking = None,
            reasoning_effort = None,
        ):
            pass
        await client.close()

    _drive(run())

    parts = captured["body"]["input"][0]["content"]
    assert parts[0] == {"type": "input_text", "text": "What is this?"}
    assert parts[1] == {"type": "input_image", "image_url": "data:image/png;base64,AAA"}
    # No max_output_tokens key when caller passes max_tokens=None.
    assert "max_output_tokens" not in captured["body"]


def test_responses_sse_translates_to_chat_completions_chunks(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": ", world"},
            {"type": "response.completed", "response": {}},
        ]
        return httpx.Response(
            200,
            content = _responses_sse(events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        lines = await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = None,
                enable_thinking = None,
                reasoning_effort = None,
            )
        )
        await client.close()
        return lines

    lines = _drive(run())

    # Keep only data lines for assertion clarity.
    data_lines = [line for line in lines if line.startswith("data:")]
    payloads = []
    for line in data_lines:
        raw = line[len("data:") :].strip()
        if raw == "[DONE]":
            payloads.append("[DONE]")
        else:
            payloads.append(json.loads(raw))

    # Two text deltas, one terminal chunk, then [DONE].
    assert payloads[0]["choices"][0]["delta"]["content"] == "Hello"
    assert payloads[0]["choices"][0]["finish_reason"] is None
    assert payloads[1]["choices"][0]["delta"]["content"] == ", world"
    assert payloads[2]["choices"][0]["delta"] == {}
    assert payloads[2]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1] == "[DONE]"


def test_responses_function_call_output_translates_to_delta_tool_calls(monkeypatch):
    """Round 12: function tools forwarded into /v1/responses must have their
    `function_call` output items translated back into Chat Completions
    delta.tool_calls, and the terminal chunk must emit
    finish_reason="tool_calls" (not "stop") so the frontend's accumulator runs
    the function."""

    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            {"type": "response.created"},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_abc",
                    "call_id": "call_xyz",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                },
            },
            {"type": "response.completed", "response": {}},
        ]
        return httpx.Response(
            200,
            content = _responses_sse(events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        lines = await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "weather?"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = None,
                enable_thinking = None,
                reasoning_effort = None,
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    payloads = [
        json.loads(line[len("data:") :].strip())
        for line in lines
        if line.startswith("data:") and line[len("data:") :].strip() != "[DONE]"
    ]
    tool_call_deltas = [
        p
        for p in payloads
        if isinstance(p, dict)
        and p.get("choices")
        and p["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert tool_call_deltas, payloads
    tc = tool_call_deltas[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["id"] == "call_xyz"
    assert tc["function"]["name"] == "get_weather"
    assert tc["function"]["arguments"] == '{"city":"SF"}'
    # Final chunk reports tool_calls, not stop.
    terminal = next(
        p
        for p in payloads
        if isinstance(p, dict)
        and p.get("choices")
        and p["choices"][0].get("finish_reason") in ("stop", "tool_calls")
    )
    assert terminal["choices"][0]["finish_reason"] == "tool_calls", payloads


def test_responses_parallel_function_calls_get_distinct_indices(monkeypatch):
    """Round 13: parallel function_call items must land on distinct
    delta.tool_calls[].index slots so index-keyed clients don't collapse the
    second call into the first."""

    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            {"type": "response.created"},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_a",
                    "call_id": "call_a",
                    "name": "lookup_a",
                    "arguments": "{}",
                },
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_b",
                    "call_id": "call_b",
                    "name": "lookup_b",
                    "arguments": "{}",
                },
            },
            {"type": "response.completed", "response": {}},
        ]
        return httpx.Response(
            200,
            content = _responses_sse(events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        lines = await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "x"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = None,
                enable_thinking = None,
                reasoning_effort = None,
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_a",
                            "parameters": {"type": "object"},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_b",
                            "parameters": {"type": "object"},
                        },
                    },
                ],
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    indices: list[int] = []
    for raw in lines:
        if not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
        for tc in delta.get("tool_calls") or []:
            indices.append(tc.get("index"))
    assert indices == [0, 1], indices


def test_responses_follow_up_tool_result_uses_function_call_output_items(monkeypatch):
    """Round 13: a second turn after a Responses function call must serialize
    the tool_calls history and tool result as Responses `function_call` /
    `function_call_output` input items, not Chat Completions role="tool"
    content."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse(
                [
                    {"type": "response.created"},
                    {"type": "response.completed", "response": {}},
                ]
            ),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        await _collect(
            client._stream_openai_responses(
                messages = [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"SF"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_xyz",
                        "content": "sunny",
                    },
                    {"role": "user", "content": "thanks"},
                ],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = None,
                enable_thinking = None,
                reasoning_effort = None,
            )
        )
        await client.close()

    _drive(run())
    items = captured["body"]["input"]
    types = [it.get("type") or it.get("role") for it in items]
    assert "function_call" in types, items
    assert "function_call_output" in types, items
    fc = next(it for it in items if it.get("type") == "function_call")
    assert fc["call_id"] == "call_xyz"
    assert fc["name"] == "get_weather"
    assert fc["arguments"] == '{"city":"SF"}'
    fco = next(it for it in items if it.get("type") == "function_call_output")
    assert fco["call_id"] == "call_xyz"
    assert fco["output"] == "sunny"


def test_responses_response_incomplete_maps_to_length_finish_reason(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            {"type": "response.output_text.delta", "delta": "partial"},
            {"type": "response.incomplete", "response": {}},
        ]
        return httpx.Response(
            200,
            content = _responses_sse(events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        lines = await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4,
                enable_thinking = None,
                reasoning_effort = None,
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    finish_reasons = [
        json.loads(line[len("data:") :].strip())["choices"][0]["finish_reason"]
        for line in lines
        if line.startswith("data:") and line[len("data:") :].strip() not in ("", "[DONE]")
    ]
    assert "length" in finish_reasons


def test_responses_reasoning_effort_included_when_requested(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = None,
            enable_thinking = None,
            reasoning_effort = "high",
        ):
            pass
        await client.close()

    _drive(run())
    assert captured["body"]["reasoning"] == {"effort": "high", "summary": "auto"}


def test_responses_reasoning_summary_omitted_for_o3(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "hi"}],
            model = "o3",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = None,
            enable_thinking = None,
            reasoning_effort = "high",
        ):
            pass
        await client.close()

    _drive(run())
    assert captured["body"]["reasoning"] == {"effort": "high"}


def test_responses_reasoning_summary_omitted_for_o3_with_enable_thinking(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "hi"}],
            model = "o3",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = None,
            enable_thinking = True,
            reasoning_effort = None,
        ):
            pass
        await client.close()

    _drive(run())
    assert captured["body"]["reasoning"] == {"effort": "medium"}


def test_responses_reasoning_effort_none_omits_summary(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = None,
            enable_thinking = None,
            reasoning_effort = "none",
        ):
            pass
        await client.close()

    _drive(run())
    assert captured["body"]["reasoning"] == {"effort": "none"}


def test_responses_reasoning_effort_xhigh_passthrough(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = None,
            enable_thinking = None,
            reasoning_effort = "xhigh",
        ):
            pass
        await client.close()

    _drive(run())
    assert captured["body"]["reasoning"] == {"effort": "xhigh", "summary": "auto"}


def test_responses_enable_thinking_false_maps_to_reasoning_none(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _responses_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = None,
            enable_thinking = False,
            reasoning_effort = None,
        ):
            pass
        await client.close()

    _drive(run())
    assert captured["body"]["reasoning"] == {"effort": "none"}


def test_responses_reasoning_summary_wrapped_in_think_tags(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                },
            },
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.completed", "response": {}},
        ]
        return httpx.Response(
            200,
            content = _responses_sse(events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        lines = await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = None,
                enable_thinking = None,
                reasoning_effort = None,
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    data_lines = [
        line[len("data:") :].strip()
        for line in lines
        if line.startswith("data:") and line[len("data:") :].strip() not in ("", "[DONE]")
    ]
    payloads = [json.loads(raw) for raw in data_lines]
    combined = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload["choices"][0]["delta"]
    )
    assert "<think>plan</think>answer" in combined
