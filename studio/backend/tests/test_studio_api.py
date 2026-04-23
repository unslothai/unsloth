# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
End-to-end tests for Unsloth Studio's HTTP API surface.

Covers the OpenAI-compatible and Anthropic-compatible endpoints exposed
by the server that ``unsloth studio run`` boots, plus API key
authentication and the CLI's ``--help`` output:

    1. curl -- basic chat completions (non-streaming)
    2. curl -- streaming chat completions
    3. Python OpenAI SDK -- streaming completions
    4. curl -- Studio server-side tools (enable_tools=true)
    5. curl -- Standard OpenAI function calling (non-streaming)
    6. curl -- Standard OpenAI function calling (streaming)
    7. curl -- Standard OpenAI function calling (multi-turn tool loop)
    8. OpenAI Python SDK -- Standard function calling
    9. Anthropic Messages API -- basic non-streaming
    10. Anthropic Messages API -- streaming SSE
    11. Anthropic Python SDK -- non-streaming
    12. Anthropic Messages API -- streaming with tools
    13. Anthropic Messages API -- tool_choice={"type":"any"} honored

Training, export, fine-tuning, and chat-UI concerns are out of scope —
see the unit suites elsewhere under ``studio/backend/tests/`` for those.

Usage:

    # Script mode — launches its own server via ``unsloth studio run``.
    python tests/test_studio_api.py
    python tests/test_studio_api.py --model unsloth/... --gguf-variant ...

    # Pytest mode, external server — start a Studio server yourself,
    # then point pytest at it. Fastest iteration loop.
    unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --gguf-variant UD-Q4_K_XL &
    export UNSLOTH_E2E_BASE_URL=http://127.0.0.1:8080
    export UNSLOTH_E2E_API_KEY=sk-unsloth-...   # from the server banner
    pytest tests/test_studio_api.py -v

    # Pytest mode, fixture-managed server — pytest launches and tears
    # down the server itself. One-shot verification, CI-friendly.
    pytest tests/test_studio_api.py -v \\
        --unsloth-model unsloth/Qwen3-1.7B-GGUF \\
        --unsloth-gguf-variant UD-Q4_K_XL

The ``base_url`` / ``api_key`` parameters on the test functions resolve
via the ``studio_server`` session fixture in ``conftest.py``.

Requires a GPU and ~2 GB of disk for the GGUF download.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────

DEFAULT_MODEL = "unsloth/Qwen3-1.7B-GGUF"
DEFAULT_VARIANT = "UD-Q4_K_XL"
PORT = 18222  # high port unlikely to collide
HOST = "127.0.0.1"
STARTUP_TIMEOUT = 120  # seconds to wait for banner
LOG_FILE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "temp"
    / "test_studio_api.log"
)


# ── Helpers ──────────────────────────────────────────────────────────


def _http(
    method: str,
    url: str,
    *,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 60,
) -> tuple[int, str]:
    """Minimal stdlib HTTP helper.  Returns (status_code, body_text)."""
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data = data, headers = headers or {}, method = method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors = "replace")


def _stream_http(
    url: str,
    *,
    body: dict,
    headers: dict,
    timeout: int = 60,
) -> tuple[int, list[dict]]:
    """POST a streaming request and collect SSE chunks."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data = data, headers = headers, method = "POST")
    req.add_header("Content-Type", "application/json")
    chunks: list[dict] = []
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            status = resp.status
            for raw_line in resp:
                line = raw_line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunks.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
            return status, chunks
    except urllib.error.HTTPError as exc:
        return exc.code, []


# ── Test functions ───────────────────────────────────────────────────


def test_help_output():
    """``unsloth studio run --help`` should show all documented options."""
    result = subprocess.run(
        ["unsloth", "studio", "run", "--help"],
        capture_output = True,
        text = True,
        timeout = 15,
    )
    out = result.stdout
    assert result.returncode == 0, f"--help exited with {result.returncode}"

    for flag in [
        "--model",
        "--gguf-variant",
        "--max-seq-length",
        "--load-in-4bit",
        "--api-key-name",
        "--port",
        "--host",
        "--frontend",
        "--silent",
    ]:
        assert flag in out, f"Missing flag {flag!r} in --help output"
    print("  PASS  --help shows all flags")


def test_curl_basic(base_url: str, api_key: str):
    """Example 1: basic non-streaming chat completion via HTTP."""
    status, text = _http(
        "POST",
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "Say just the word hello"}],
            "stream": False,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
    )
    assert status == 200, f"Expected 200, got {status}: {text[:300]}"
    data = json.loads(text)
    assert "choices" in data, f"Missing 'choices' in response: {text[:300]}"
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0, "Empty assistant content"
    print(f"  PASS  curl basic: {content[:80]!r}")


def _collect_streamed_content(chunks: list[dict]) -> str:
    """Extract text from SSE chunks, skipping role-only and usage chunks."""
    parts = []
    for c in chunks:
        choices = c.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        part = delta.get("content")
        if part:
            parts.append(part)
    return "".join(parts)


def test_curl_streaming(base_url: str, api_key: str):
    """Example 2: streaming chat completion via HTTP SSE."""
    status, chunks = _stream_http(
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "Count from 1 to 3"}],
            "stream": True,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
    )
    assert status == 200, f"Expected 200, got {status}"
    assert len(chunks) > 0, "No SSE chunks received"
    full = _collect_streamed_content(chunks)
    assert len(full) > 0, "Streamed content is empty"
    print(f"  PASS  curl streaming: got {len(chunks)} chunks, {len(full)} chars")


def test_openai_sdk(base_url: str, api_key: str):
    """Example 3: OpenAI Python SDK streaming completion."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  SKIP  openai SDK not installed")
        return

    client = OpenAI(base_url = f"{base_url}/v1", api_key = api_key)
    response = client.chat.completions.create(
        model = "current",
        messages = [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        stream = True,
    )
    content_parts = []
    for chunk in response:
        if not chunk.choices:
            continue
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            content_parts.append(delta_content)
    full = "".join(content_parts)
    assert len(full) > 0, "OpenAI SDK returned empty content"
    print(f"  PASS  OpenAI SDK streaming: {full.strip()[:80]!r}")


def test_curl_with_tools(base_url: str, api_key: str):
    """Example 4: chat completion with tool calling enabled.

    Note: when ``enable_tools`` is set the server always returns SSE
    streaming regardless of the ``stream`` flag, so we parse SSE chunks.
    The model may or may not produce visible content -- tool orchestration
    can intercept the response -- so we only assert the endpoint succeeds.
    """
    status, chunks = _stream_http(
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 123 * 456? Use code to compute it.",
                }
            ],
            "stream": True,
            "enable_tools": True,
            "enabled_tools": ["python"],
            "session_id": "test-session",
        },
        headers = {"Authorization": f"Bearer {api_key}"},
        timeout = 120,
    )
    assert status == 200, f"Expected 200, got {status}"
    assert len(chunks) > 0, "No SSE chunks received for tools request"

    # Check that at least one chunk has the expected shape
    has_valid_chunk = any("choices" in c or "type" in c for c in chunks)
    assert has_valid_chunk, "No valid chunks in tools response"
    full = _collect_streamed_content(chunks)
    print(f"  PASS  curl with tools: {len(chunks)} chunks, {len(full)} chars content")


# ── Standard OpenAI function-calling pass-through tests ─────────────
#
# Regression coverage for unslothai/unsloth#4999: Studio's
# /v1/chat/completions used to silently strip standard OpenAI `tools`
# and `tool_choice` fields, so clients (opencode, Claude Code, Cursor,
# Continue, ...) could never get structured tool_calls back. These
# tests exercise the client-side pass-through path that forwards those
# fields to llama-server verbatim.
#
# They require a tool-capable GGUF (``supports_tools=True`` — e.g.
# Qwen3, Qwen2.5-Coder, Llama-3.1-Instruct). The default test model
# ``unsloth/Qwen3-1.7B-GGUF`` advertises tool support via its chat
# template metadata.

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. 'Paris'.",
                },
            },
            "required": ["city"],
        },
    },
}


def _collect_streamed_tool_calls(chunks: list[dict]) -> list[dict]:
    """Reassemble OpenAI streaming delta.tool_calls into full tool calls.

    OpenAI streams partial tool calls across chunks — the first chunk for
    a given index carries ``id`` + ``function.name``, and subsequent
    chunks append fragments to ``function.arguments``.
    """
    by_index: dict[int, dict] = {}
    for c in chunks:
        choices = c.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            idx = tc.get("index", 0)
            slot = by_index.setdefault(
                idx,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                },
            )
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
    return [by_index[i] for i in sorted(by_index)]


def _final_finish_reason(chunks: list[dict]) -> str | None:
    for c in reversed(chunks):
        choices = c.get("choices") or []
        if not choices:
            continue
        fr = choices[0].get("finish_reason")
        if fr is not None:
            return fr
    return None


def test_openai_tools_nonstream(base_url: str, api_key: str):
    """Standard OpenAI function calling, non-streaming, tool_choice='required'.

    Regression: before the fix, Studio silently stripped `tools` and the
    model returned plain text with finish_reason='stop'. After the fix,
    llama-server's response is forwarded verbatim so the client sees
    finish_reason='tool_calls' with a structured tool_calls array and
    non-zero usage.prompt_tokens.
    """
    status, text = _http(
        "POST",
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [_WEATHER_TOOL],
            "tool_choice": "required",
            "stream": False,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
        timeout = 120,
    )
    assert status == 200, f"Expected 200, got {status}: {text[:500]}"
    data = json.loads(text)
    assert "choices" in data, f"Missing 'choices': {text[:300]}"
    choice = data["choices"][0]
    assert (
        choice["finish_reason"] == "tool_calls"
    ), f"Expected finish_reason='tool_calls', got {choice['finish_reason']!r}"
    msg = choice["message"]
    tool_calls = msg.get("tool_calls") or []
    assert len(tool_calls) >= 1, f"No tool_calls in response: {msg}"
    first = tool_calls[0]
    assert first["type"] == "function"
    assert (
        first["function"]["name"] == "get_weather"
    ), f"Wrong tool name: {first['function']['name']!r}"
    # arguments must be valid JSON
    parsed = json.loads(first["function"]["arguments"])
    assert "city" in parsed, f"Tool call missing required 'city' arg: {parsed}"
    # Usage must be non-zero (was 0 before the fix)
    usage = data.get("usage") or {}
    assert (
        usage.get("prompt_tokens", 0) > 0
    ), f"Expected non-zero prompt_tokens; got {usage}"
    assert data.get("id"), "Missing response id"
    print(
        f"  PASS  openai tools non-stream: "
        f"tool={first['function']['name']}, args={parsed}, "
        f"prompt_tokens={usage['prompt_tokens']}"
    )


def test_openai_tools_stream(base_url: str, api_key: str):
    """Standard OpenAI function calling, streaming, tool_choice='required'."""
    status, chunks = _stream_http(
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
            "tools": [_WEATHER_TOOL],
            "tool_choice": "required",
            "stream": True,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
        timeout = 120,
    )
    assert status == 200, f"Expected 200, got {status}"
    assert len(chunks) > 0, "No SSE chunks received"
    assert _final_finish_reason(chunks) == "tool_calls", (
        f"Expected final finish_reason='tool_calls', got "
        f"{_final_finish_reason(chunks)!r}"
    )
    assembled = _collect_streamed_tool_calls(chunks)
    assert len(assembled) >= 1, "No tool_calls reassembled from stream"
    first = assembled[0]
    assert first["function"]["name"] == "get_weather"
    parsed = json.loads(first["function"]["arguments"])
    assert "city" in parsed
    print(
        f"  PASS  openai tools stream: {len(chunks)} chunks, "
        f"tool={first['function']['name']}, args={parsed}"
    )


def test_openai_tools_multiturn(base_url: str, api_key: str):
    """Multi-turn client-side tool loop: validates that role='tool' result
    messages and assistant messages carrying tool_calls are accepted.

    Regression: before the fix, ChatMessage.role was restricted to
    {system,user,assistant} and rejected role='tool' at the Pydantic
    validation stage. This test sends a full round trip so the model
    receives the simulated tool result and responds with final text.
    """
    status, text = _http(
        "POST",
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_test_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_test_1",
                    "content": '{"temperature_c": 14, "condition": "cloudy"}',
                },
            ],
            "tools": [_WEATHER_TOOL],
            "stream": False,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
        timeout = 120,
    )
    assert status == 200, f"Expected 200, got {status}: {text[:500]}"
    data = json.loads(text)
    msg = data["choices"][0]["message"]
    # The model should respond with text now that it has the tool result
    content = msg.get("content") or ""
    assert len(content) > 0 or msg.get(
        "tool_calls"
    ), f"Expected text or follow-up tool call, got empty message: {msg}"
    print(f"  PASS  openai tools multiturn: {content[:80]!r}")


def test_openai_sdk_tool_calling(base_url: str, api_key: str):
    """OpenAI Python SDK round trip — the real client shape opencode et al. use."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  SKIP  openai SDK not installed")
        return

    client = OpenAI(base_url = f"{base_url}/v1", api_key = api_key)
    resp = client.chat.completions.create(
        model = "current",
        messages = [{"role": "user", "content": "What's the weather in Berlin?"}],
        tools = [_WEATHER_TOOL],
        tool_choice = "required",
        stream = False,
    )
    assert resp.choices[0].finish_reason == "tool_calls", (
        f"Expected finish_reason='tool_calls', got "
        f"{resp.choices[0].finish_reason!r}"
    )
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls and len(tool_calls) >= 1, "No tool_calls from SDK"
    tc = tool_calls[0]
    assert tc.function.name == "get_weather"
    parsed = json.loads(tc.function.arguments)
    assert "city" in parsed
    print(
        f"  PASS  openai SDK tool calling: " f"tool={tc.function.name}, args={parsed}"
    )


def test_invalid_key_rejected(base_url: str):
    """Requests with a bad API key should be rejected."""
    status, _text = _http(
        "POST",
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
        headers = {"Authorization": "Bearer sk-unsloth-boguskey123"},
    )
    assert status == 401, f"Expected 401 for invalid key, got {status}"
    print("  PASS  invalid API key rejected (401)")


def test_no_key_rejected(base_url: str):
    """Requests without any auth header should be rejected."""
    status, _text = _http(
        "POST",
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert status == 401 or status == 403, f"Expected 401/403 for no key, got {status}"
    print(f"  PASS  no API key rejected ({status})")


# ── Anthropic SSE helper ─────────────────────────────────────────────


def _stream_anthropic_http(
    url: str,
    *,
    body: dict,
    headers: dict,
    timeout: int = 60,
) -> tuple[int, list[tuple[str, dict]]]:
    """POST a streaming request and collect Anthropic SSE events.

    Returns (status, [(event_type, data_dict), ...]).
    """
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data = data, headers = headers, method = "POST")
    req.add_header("Content-Type", "application/json")
    events: list[tuple[str, dict]] = []
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            status = resp.status
            current_event = None
            for raw_line in resp:
                line = raw_line.decode().strip()
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: ") and current_event:
                    try:
                        events.append((current_event, json.loads(line[6:])))
                    except json.JSONDecodeError:
                        pass
                    current_event = None
            return status, events
    except urllib.error.HTTPError as exc:
        return exc.code, []


def _collect_anthropic_text(events: list[tuple[str, dict]]) -> str:
    """Extract text content from Anthropic SSE events."""
    parts = []
    for etype, data in events:
        if etype == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                parts.append(delta.get("text", ""))
    return "".join(parts)


# ── Anthropic /v1/messages test functions ────────────────────────────


def test_anthropic_basic(base_url: str, api_key: str):
    """Anthropic Messages API: non-streaming."""
    status, text = _http(
        "POST",
        f"{base_url}/v1/messages",
        body = {
            "model": "default",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say just the word hello"}],
        },
        headers = {"Authorization": f"Bearer {api_key}"},
    )
    assert status == 200, f"Expected 200, got {status}: {text[:300]}"
    data = json.loads(text)
    assert data.get("type") == "message", f"Expected type 'message': {text[:300]}"
    assert data.get("role") == "assistant"
    content = data.get("content", [])
    assert len(content) > 0, "Empty content array"
    text_block = content[-1]
    assert text_block.get("type") == "text", f"Expected text block: {text_block}"
    assert len(text_block.get("text", "")) > 0, "Empty text in response"
    print(f"  PASS  anthropic basic: {text_block['text'][:80]!r}")


def test_anthropic_streaming(base_url: str, api_key: str):
    """Anthropic Messages API: streaming SSE."""
    status, events = _stream_anthropic_http(
        f"{base_url}/v1/messages",
        body = {
            "model": "default",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Count from 1 to 3"}],
            "stream": True,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
    )
    assert status == 200, f"Expected 200, got {status}"
    assert len(events) > 0, "No SSE events received"

    event_types = [e[0] for e in events]
    assert "message_start" in event_types, "Missing message_start event"
    assert "message_stop" in event_types, "Missing message_stop event"

    full = _collect_anthropic_text(events)
    assert len(full) > 0, "Streamed text content is empty"
    print(f"  PASS  anthropic streaming: {len(events)} events, {len(full)} chars")


def test_anthropic_sdk(base_url: str, api_key: str):
    """Anthropic Python SDK: non-streaming."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("  SKIP  anthropic SDK not installed")
        return

    client = Anthropic(base_url = f"{base_url}/v1", api_key = api_key)
    message = client.messages.create(
        model = "default",
        max_tokens = 100,
        messages = [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
    )
    assert message.role == "assistant"
    assert len(message.content) > 0, "Empty content"
    text = message.content[0].text
    assert len(text) > 0, "Empty text"
    print(f"  PASS  Anthropic SDK: {text.strip()[:80]!r}")


def test_anthropic_with_tools(base_url: str, api_key: str):
    """Anthropic Messages API: streaming with tools."""
    status, events = _stream_anthropic_http(
        f"{base_url}/v1/messages",
        body = {
            "model": "default",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 123 * 456? Use code to compute it.",
                }
            ],
            "tools": [
                {
                    "name": "python",
                    "description": "Execute Python code in a sandbox and return stdout/stderr.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to run",
                            },
                        },
                        "required": ["code"],
                    },
                }
            ],
            "stream": True,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
        timeout = 120,
    )
    assert status == 200, f"Expected 200, got {status}"
    assert len(events) > 0, "No SSE events received for tools request"

    event_types = [e[0] for e in events]
    assert "message_start" in event_types, "Missing message_start"
    assert "message_stop" in event_types, "Missing message_stop"

    full = _collect_anthropic_text(events)
    print(
        f"  PASS  anthropic with tools: {len(events)} events, {len(full)} chars content"
    )


def test_anthropic_tool_choice_any(base_url: str, api_key: str):
    """Anthropic Messages API: ``tool_choice: {"type": "any"}`` must be
    honored (forwarded as OpenAI ``tool_choice: "required"`` to
    llama-server). Regression for the secondary fix bundled with #4999 —
    previously this field was accepted on the request model but silently
    dropped with a warning log, so the model was free to answer from
    memory instead of using the tool.
    """
    status, events = _stream_anthropic_http(
        f"{base_url}/v1/messages",
        body = {
            "model": "default",
            "max_tokens": 256,
            "messages": [
                # A question the model could easily answer from memory if
                # tool_choice were not enforced.
                {
                    "role": "user",
                    "content": "What is the weather in London right now?",
                }
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Look up current weather for a city.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "any"},
            "stream": True,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
        timeout = 120,
    )
    assert status == 200, f"Expected 200, got {status}"
    assert len(events) > 0, "No SSE events received"

    # With tool_choice=any, stop_reason must be tool_use (not end_turn)
    stop_reason = None
    for etype, data in events:
        if etype == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason") or stop_reason
    assert stop_reason == "tool_use", (
        f"Expected stop_reason='tool_use' with tool_choice=any, got "
        f"{stop_reason!r} — tool_choice may not be forwarded to llama-server."
    )

    # And at least one tool_use content block must be emitted
    tool_use_starts = [
        e
        for e in events
        if e[0] == "content_block_start"
        and e[1].get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_use_starts) >= 1, "No tool_use content block emitted"
    print(
        f"  PASS  anthropic tool_choice=any honored: "
        f"{len(tool_use_starts)} tool_use blocks, stop_reason={stop_reason}"
    )


# ── Server lifecycle ─────────────────────────────────────────────────


def _start_server(model: str, variant: str | None) -> tuple[subprocess.Popen, str]:
    """Launch ``unsloth studio run`` and parse the API key from its banner.

    Returns (process, api_key).
    """
    cmd = [
        "unsloth",
        "studio",
        "run",
        "--model",
        model,
        "--port",
        str(PORT),
        "--host",
        HOST,
        "--api-key-name",
        "test",
    ]
    if variant:
        cmd.extend(["--gguf-variant", variant])

    LOG_FILE.parent.mkdir(parents = True, exist_ok = True)
    log_fh = open(LOG_FILE, "w")
    proc = subprocess.Popen(
        cmd,
        stdout = log_fh,
        stderr = subprocess.STDOUT,
        preexec_fn = os.setsid,
    )

    # Wait for the banner containing the API key
    api_key = None
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(2)
        if proc.poll() is not None:
            log_fh.flush()
            log_text = LOG_FILE.read_text()
            raise RuntimeError(
                f"Server exited early (code {proc.returncode}):\n{log_text[-2000:]}"
            )
        log_text = LOG_FILE.read_text()
        m = re.search(r"API Key:\s+(sk-unsloth-[a-f0-9]+)", log_text)
        if m:
            api_key = m.group(1)
            break

    if not api_key:
        log_text = LOG_FILE.read_text()
        _kill_server(proc)
        raise RuntimeError(
            f"Timed out waiting for API key in server output:\n{log_text[-2000:]}"
        )

    # Wait a moment for the model to be fully loaded
    time.sleep(2)
    return proc, api_key


def _kill_server(proc: subprocess.Popen):
    """Send SIGTERM to the process group and wait for cleanup."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout = 10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout = 5)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description = "End-to-end tests for unsloth studio run"
    )
    parser.add_argument(
        "--model",
        default = DEFAULT_MODEL,
        help = f"Model to test with (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--gguf-variant",
        default = DEFAULT_VARIANT,
        help = f"GGUF variant (default: {DEFAULT_VARIANT})",
    )
    args = parser.parse_args()

    passed = 0
    failed = 0
    skipped = 0

    def run_test(fn, *a, **kw):
        nonlocal passed, failed, skipped
        try:
            fn(*a, **kw)
            passed += 1
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")

    # ── 1. Test --help (no server needed) ────────────────────────────
    print("\n[1/16] Testing --help output")
    run_test(test_help_output)

    # ── 2-16. Start server and run API tests ─────────────────────────
    print(
        f"\nStarting server: {args.model} (variant={args.gguf_variant}) on port {PORT}..."
    )
    proc = None
    try:
        proc, api_key = _start_server(args.model, args.gguf_variant)
        base_url = f"http://{HOST}:{PORT}"
        print(f"Server ready.  API Key: {api_key[:20]}...\n")

        print("[2/16] Testing curl basic (non-streaming)")
        run_test(test_curl_basic, base_url, api_key)

        print("[3/16] Testing curl streaming")
        run_test(test_curl_streaming, base_url, api_key)

        print("[4/16] Testing OpenAI Python SDK (streaming)")
        run_test(test_openai_sdk, base_url, api_key)

        print("[5/16] Testing curl with tools (server-side enable_tools)")
        run_test(test_curl_with_tools, base_url, api_key)

        print("[6/16] Testing OpenAI standard tools (non-streaming)")
        run_test(test_openai_tools_nonstream, base_url, api_key)

        print("[7/16] Testing OpenAI standard tools (streaming)")
        run_test(test_openai_tools_stream, base_url, api_key)

        print("[8/16] Testing OpenAI standard tools (multi-turn)")
        run_test(test_openai_tools_multiturn, base_url, api_key)

        print("[9/16] Testing OpenAI SDK tool calling")
        run_test(test_openai_sdk_tool_calling, base_url, api_key)

        print("[10/16] Testing invalid API key rejection")
        run_test(test_invalid_key_rejected, base_url)

        print("[11/16] Testing no API key rejection")
        run_test(test_no_key_rejected, base_url)

        print("[12/16] Testing Anthropic basic (non-streaming)")
        run_test(test_anthropic_basic, base_url, api_key)

        print("[13/16] Testing Anthropic streaming")
        run_test(test_anthropic_streaming, base_url, api_key)

        print("[14/16] Testing Anthropic Python SDK")
        run_test(test_anthropic_sdk, base_url, api_key)

        print("[15/16] Testing Anthropic with tools")
        run_test(test_anthropic_with_tools, base_url, api_key)

        print("[16/16] Testing Anthropic tool_choice=any honored")
        run_test(test_anthropic_tool_choice_any, base_url, api_key)

    except RuntimeError as exc:
        print(f"\nFATAL: Server failed to start: {exc}")
        failed += 16  # count remaining tests as failed
    finally:
        if proc:
            print("\nStopping server...")
            _kill_server(proc)
            print("Server stopped.")

    # ── Summary ──────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"Log: {LOG_FILE}")
    print(f"{'=' * 40}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
