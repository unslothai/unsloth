# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for OpenAI's server-side `shell` tool translation in
`_stream_openai_responses`.

Covers: request body shaping (container_auto, container_reference), the cloud
guard (no shell tool on non-cloud base_urls), SSE translation of a
shell_call/shell_call_output pair into tool_start/tool_end events, container_id
surfacing as container_ready, and stale-container invalidation.
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
    assert {"type": "shell", "environment": {"type": "container_auto"}} in tools


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
    # Shell tool must not leak to local OpenAI-compat servers (they 400 on it).
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
    # `_server_tool: True` marks a synthetic builtin so the frontend can tell
    # hosted tools from user-declared functions on history replay.
    assert starts[0]["arguments"] == {"kind": "bash", "command": "ls -la", "_server_tool": True}
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


def test_expired_container_triggers_transparent_retry(monkeypatch):
    """On a 'Container is expired' 400 for a container_reference request, the
    streamer retries once with the container stripped; the user sees only
    container_invalidated then the retry stream, never an error line.
    """
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        # Inspect the shell tool's environment.type.
        shell_env_type = None
        for tool in body.get("tools", []) or []:
            if tool.get("type") == "shell":
                shell_env_type = tool.get("environment", {}).get("type")
                break
        # container_reference -> 400 expired; retry omits container -> normal stream.
        if shell_env_type == "container_reference":
            return httpx.Response(
                400,
                content = json.dumps(
                    {
                        "error": {
                            "message": "Container is expired.",
                            "type": "invalid_request_error",
                        }
                    }
                ).encode("utf-8"),
                headers = {"content-type": "application/json"},
            )
        # Successful retry: minimal SSE — completed response with a fresh
        # container_id so container_ready latches.
        sse = _openai_sse(
            [
                {
                    "type": "response.completed",
                    "response": {"container_id": "cntr_fresh_111"},
                },
            ]
        )
        return httpx.Response(
            200,
            content = sse,
            headers = {"content-type": "text/event-stream"},
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

    # Two outbound calls: the expired-container attempt, then the retry
    # without the container field.
    assert len(calls) == 2
    shell_types = []
    for body in calls:
        for tool in body.get("tools", []) or []:
            if tool.get("type") == "shell":
                shell_types.append(tool.get("environment", {}).get("type"))
    assert shell_types == ["container_reference", "container_auto"]

    # container_invalidated emitted (frontend nulls its stored id).
    assert any(e.get("type") == "container_invalidated" for e in events)
    # container_ready emitted from the retry stream with the fresh id.
    assert any(
        e.get("type") == "container_ready" and e.get("container_id") == "cntr_fresh_111"
        for e in events
    )
    # CRUCIALLY: no SSE error line surfaced to the chat.
    error_lines = [
        line
        for line in lines
        if line.startswith("data:") and '"error"' in line and '"_toolEvent"' not in line
    ]
    assert error_lines == [], f"unexpected error line(s): {error_lines}"


def test_expired_container_retries_only_once(monkeypatch):
    """If the retry ALSO fails (any 4xx), the error surfaces normally —
    no infinite retry loop.
    """
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            400,
            content = json.dumps(
                {
                    "error": {
                        "message": "Container is expired.",
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

    # Exactly two calls (first + one retry); a third would be a loop.
    assert call_count["n"] == 2
    # The second failure surfaces normally as an error SSE line.
    error_lines = [line for line in lines if '"error"' in line and "_toolEvent" not in line]
    assert len(error_lines) >= 1
