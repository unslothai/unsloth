# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for OpenAI's server-side `shell` tool translation in
`_stream_openai_responses`.

Covers:
- Request body: ``enabled_tools=["code_execution"]`` on the OpenAI
  cloud base_url appends ``{"type": "shell", "environment": {"type":
  "container_auto"}}`` to ``tools``.
- Container reuse: when ``openai_code_exec_container_id`` is provided,
  the outgoing ``environment.type`` flips to ``"container_reference"``
  and the id propagates.
- Cloud guard: code_execution on a non-cloud base_url (e.g. a local
  OpenAI-compat preset / ollama / llama.cpp / vLLM) does NOT add the
  shell tool, preventing a guaranteed 400 from those servers.
- SSE translation: a `shell_call` + `shell_call_output` pair emits one
  ``_toolEvent`` `tool_start` (`tool_name="code_execution"`,
  `arguments.kind="bash"`) and one `tool_end` whose `result` contains
  the joined stdout from the shell_call_output entries.
- Container surfacing: container_id captured from
  `response.completed.container_id` is emitted as a synthetic
  `container_ready` `_toolEvent` (only when it differs from the
  inbound id).
- Stale-container handling: 400 with "container expired" body emits a
  `container_invalidated` event before propagating the error.
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


def _make_client(base_url: str = "https://api.openai.com/v1") -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "openai",
        base_url = base_url,
        api_key = "sk-test",
    )


def _openai_sse(events: list[dict]) -> bytes:
    chunks: list[str] = []
    for event in events:
        chunks.append(f"event: {event['type']}")
        chunks.append(f"data: {json.dumps(event)}")
        chunks.append("")
    return ("\n".join(chunks) + "\n").encode("utf-8")


def _tool_events(lines: list[str]) -> list[dict]:
    out: list[dict] = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "_toolEvent" in parsed:
            out.append(parsed["_toolEvent"])
    return out


def test_shell_tool_added_on_cloud_with_container_auto(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _openai_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "compute 2+2"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            enable_thinking = None,
            reasoning_effort = None,
            enabled_tools = ["code_execution"],
        ):
            pass
        await client.close()

    _drive(run())

    tools = captured["body"].get("tools") or []
    assert {
        "type": "shell",
        "environment": {"type": "container_auto"},
    } in tools


def test_shell_tool_uses_container_reference_when_id_supplied(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _openai_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "what did i write earlier"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            enable_thinking = None,
            reasoning_effort = None,
            enabled_tools = ["code_execution"],
            openai_code_exec_container_id = "cntr_abc123",
        ):
            pass
        await client.close()

    _drive(run())

    tools = captured["body"].get("tools") or []
    assert {
        "type": "shell",
        "environment": {
            "type": "container_reference",
            "container_id": "cntr_abc123",
        },
    } in tools


def test_shell_tool_refused_for_non_cloud_base_url(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = _openai_sse([{"type": "response.completed", "response": {}}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client(base_url = "http://localhost:11434/v1")
        async for _ in client._stream_openai_responses(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            enable_thinking = None,
            reasoning_effort = None,
            enabled_tools = ["code_execution"],
        ):
            pass
        await client.close()

    _drive(run())

    tools = captured["body"].get("tools") or []
    # Shell tool must NOT leak to local OpenAI-compat servers — those
    # 400 on the unknown tool type.
    assert all(t.get("type") != "shell" for t in tools)


def test_shell_call_emits_tool_start_and_end(monkeypatch):
    sse_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "shell_call",
                "id": "scall_1",
                "action": {"commands": ["ls -la"]},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "scall_1",
                "action": {"commands": ["ls -la"]},
                "status": "completed",
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call_output",
                "id": "scout_1",
                "call_id": "scall_1",
                "output": [
                    {
                        "stdout": "total 24\ndrwxr-xr-x .",
                        "stderr": "",
                        "outcome": {"type": "exit", "exit_code": 0},
                    }
                ],
            },
        },
        {"type": "response.completed", "response": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _openai_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "list files"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                enable_thinking = None,
                reasoning_effort = None,
                enabled_tools = ["code_execution"],
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)
    starts = [e for e in events if e["type"] == "tool_start"]
    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0]["tool_name"] == "code_execution"
    assert starts[0]["tool_call_id"] == "scall_1"
    assert starts[0]["arguments"] == {"kind": "bash", "command": "ls -la"}
    assert ends[0]["tool_call_id"] == "scall_1"
    assert "total 24" in ends[0]["result"]


def test_container_ready_emitted_when_new_id_surfaces(monkeypatch):
    sse_events = [
        {
            "type": "response.completed",
            "response": {"container_id": "cntr_new_456"},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _openai_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "do stuff"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                enable_thinking = None,
                reasoning_effort = None,
                enabled_tools = ["code_execution"],
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)
    ready = [e for e in events if e["type"] == "container_ready"]
    assert len(ready) == 1
    assert ready[0]["container_id"] == "cntr_new_456"


def test_container_ready_not_emitted_when_id_unchanged(monkeypatch):
    sse_events = [
        {
            "type": "response.completed",
            "response": {"container_id": "cntr_same_789"},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _openai_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "do stuff"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                enable_thinking = None,
                reasoning_effort = None,
                enabled_tools = ["code_execution"],
                openai_code_exec_container_id = "cntr_same_789",
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)
    # No churn — id matches the one already on the thread record.
    assert not any(e["type"] == "container_ready" for e in events)


def test_stale_container_emits_invalidated(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            content = json.dumps(
                {
                    "error": {
                        "message": "container has expired",
                        "type": "invalid_request_error",
                    }
                }
            ).encode("utf-8"),
            headers = {"content-type": "application/json"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                enable_thinking = None,
                reasoning_effort = None,
                enabled_tools = ["code_execution"],
                openai_code_exec_container_id = "cntr_stale_999",
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)
    invalidated = [e for e in events if e["type"] == "container_invalidated"]
    assert len(invalidated) == 1
