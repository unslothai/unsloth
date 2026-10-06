# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Anthropic server-side `code_execution` tool translation in
`_stream_anthropic`: request body/beta header, combined web_search,
bash/text_editor SSE tool events, and the container_expired error path."""

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
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _anthropic_sse(events: list[dict]) -> bytes:
    chunks: list[str] = []
    for event in events:
        chunks.append(f"event: {event['type']}")
        chunks.append(f"data: {json.dumps(event)}")
        chunks.append("")
    return ("\n".join(chunks) + "\n").encode("utf-8")


def _tool_events(lines: list[str]) -> list[dict]:
    """Extract `_toolEvent` payloads from emitted SSE data lines."""
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


def test_code_execution_tool_appended_to_request_body(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "compute 2 + 2"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            enabled_tools = ["code_execution"],
        ):
            pass
        await client.close()

    _drive(run())

    body = captured["body"]
    tools = body.get("tools") or []
    # Opus 4.7 gets the newer date-pinned variant with REPL persistence.
    assert {"type": "code_execution_20260120", "name": "code_execution"} in tools
    # No web_search entry when only code_execution is enabled.
    assert all("web_search" not in (t.get("type") or "") for t in tools)
    # Same beta header unlocks both _20250825 and _20260120.
    beta_header = captured["headers"].get("anthropic-beta", "")
    assert "code-execution-2025-08-25" in beta_header


def test_code_execution_with_web_search_sends_both_tools(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "look it up and chart it"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            enabled_tools = ["web_search", "code_execution"],
        ):
            pass
        await client.close()

    _drive(run())

    tools = captured["body"].get("tools") or []
    tool_types = {t.get("type") for t in tools if isinstance(t, dict)}
    # Opus 4.7 picks the newer pinned versions for both tools.
    assert "web_search_20260209" in tool_types
    assert "code_execution_20260120" in tool_types
    assert "code-execution-2025-08-25" in captured["headers"].get("anthropic-beta", "")


def test_no_code_execution_tool_when_pill_off(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = _anthropic_sse([{"type": "message_stop"}]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        async for _ in client._stream_anthropic(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
        ):
            pass
        await client.close()

    _drive(run())

    tools = captured["body"].get("tools") or []
    # Pill off: no code_execution variant on the wire.
    assert all("code_execution" not in (t.get("type") or "") for t in tools)
    # Beta header must omit code-execution when the tool is off (opt-in only).
    assert "code-execution-2025-08-25" not in captured["headers"].get("anthropic-beta", "")


def test_bash_code_execution_emits_tool_start_and_end(monkeypatch):
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "bash_code_execution",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"command": "ls -la"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "total 24\ndrwxr-xr-x 2 user user 4096 Jan 1 12:00 .",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "list files"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                enabled_tools = ["code_execution"],
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)

    assert len(events) == 2
    start, end = events
    assert start["type"] == "tool_start"
    assert start["tool_name"] == "code_execution"
    assert start["tool_call_id"] == "srvtoolu_1"
    # `_server_tool: True` marks a provider-side synthetic tool card.
    assert start["arguments"] == {"kind": "bash", "command": "ls -la", "_server_tool": True}

    assert end["type"] == "tool_end"
    assert end["tool_call_id"] == "srvtoolu_1"
    assert "total 24" in end["result"]
    # Non-zero return_code not present, so no return_code line.
    assert "return_code:" not in end["result"]


def test_text_editor_create_emits_kind_and_status(monkeypatch):
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_2",
                "name": "text_editor_code_execution",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": (
                    '{"command": "create", "path": "new_file.txt", "file_text": "hi"}'
                ),
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "text_editor_code_execution_tool_result",
                "tool_use_id": "srvtoolu_2",
                "content": {
                    "type": "text_editor_code_execution_result",
                    "is_file_update": False,
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "write a file"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                enabled_tools = ["code_execution"],
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)

    assert len(events) == 2
    start, end = events
    assert start["arguments"]["kind"] == "text_editor"
    assert start["arguments"]["command"] == "create"
    assert start["arguments"]["path"] == "new_file.txt"
    assert end["result"] == "Created"


def test_code_execution_error_renders_error_code(monkeypatch):
    sse_events = [
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_3",
                "name": "bash_code_execution",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"command": "echo broken"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_3",
                "content": {
                    "type": "bash_code_execution_tool_result_error",
                    "error_code": "container_expired",
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "run it"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 4096,
                enabled_tools = ["code_execution"],
            )
        )

    lines = _drive(run())
    events = _tool_events(lines)

    assert len(events) == 2
    end = events[1]
    assert end["type"] == "tool_end"
    assert end["result"] == "Error: container_expired"
