# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for OpenAI Responses API image_generation tool wiring.

The image_generation tool is a server-side Responses-API tool:
``{type: "image_generation"}`` in the request's tools array, and the
result comes back as an ``image_generation_call`` output item carrying
the base64 image on ``result``. Studio translates the output item
into ``_toolEvent`` chunks (``tool_start`` with `kind:"image"`,
``tool_end`` with ``image_b64`` + ``image_mime``) so the chat adapter
can render the image inline.

These tests pin: the tool is added to the outbound body only when the
caller asks for it on a cloud OpenAI base; the SSE output_item.done
for ``image_generation_call`` produces the expected _toolEvent chunks;
non-cloud bases drop the tool silently.
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _capture_body(
    monkeypatch,
    *,
    base_url: str,
    enabled_tools,
    messages: list[dict] | None = None,
) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = (
                b"event: response.completed\n"
                b'data: {"type":"response.completed",'
                b'"response":{"output":[],"usage":{"input_tokens":0,'
                b'"output_tokens":0}}}\n\n'
            ),
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = base_url,
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = (
                messages
                if messages is not None
                else [{"role": "user", "content": "draw a cat"}]
            ),
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            reasoning_effort = "medium",
            enabled_tools = enabled_tools,
        ):
            pass
        await client.close()

    _drive(run())
    return captured


def _collect_tool_events(monkeypatch) -> list[dict]:
    """Drive a Responses stream that emits one image_generation_call done
    event and return the parsed _toolEvent chunks."""

    sse = (
        b"event: response.output_item.done\n"
        b'data: {"type":"response.output_item.done",'
        b'"item":{"type":"image_generation_call",'
        b'"id":"img_abc",'
        b'"revised_prompt":"A photorealistic cat sitting",'
        b'"result":"AAAA",'
        b'"output_format":"png",'
        b'"size":"1024x1024",'
        b'"quality":"high",'
        b'"background":"opaque"}}\n\n'
        b"event: response.completed\n"
        b'data: {"type":"response.completed",'
        b'"response":{"output":[],"usage":{"input_tokens":0,'
        b'"output_tokens":0}}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = sse,
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    events: list[dict] = []

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-test",
        )
        async for line in client.stream_chat_completion(
            messages = [{"role": "user", "content": "draw a cat"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            reasoning_effort = "medium",
            enabled_tools = ["image_generation"],
        ):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "_toolEvent" in obj:
                events.append(obj["_toolEvent"])
        await client.close()

    _drive(run())
    return events


# ── tool entry appended to outbound body on cloud OpenAI ─────────────


def test_cloud_openai_appends_image_generation_tool(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        enabled_tools = ["image_generation"],
    )
    tools = captured["body"].get("tools") or []
    assert {"type": "image_generation"} in tools, tools


def test_combined_with_web_search_and_code_execution(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        enabled_tools = ["web_search", "code_execution", "image_generation"],
    )
    tools = captured["body"].get("tools") or []
    tool_types = {t["type"] for t in tools if isinstance(t, dict)}
    assert tool_types == {"web_search", "shell", "image_generation"}, tools


# ── non-cloud base silently drops the tool ──────────────────────────


def test_non_cloud_base_drops_image_generation(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        base_url = "http://127.0.0.1:11434/v1",
        enabled_tools = ["image_generation"],
    )
    tools = captured["body"].get("tools") or []
    assert {"type": "image_generation"} not in tools, tools


# ── omitted pill leaves body untouched ──────────────────────────────


def test_omitted_image_generation_pill_no_tool(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        enabled_tools = ["web_search"],
    )
    tools = captured["body"].get("tools") or []
    assert all(t.get("type") != "image_generation" for t in tools)


# ── follow-up edits forward previous image_generation_call refs ──────


def test_image_generation_reference_forwarded_for_followup_edit(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        enabled_tools = ["image_generation"],
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "image_generation_call", "id": "img_abc"},
                ],
            },
            {"role": "user", "content": "make it more realistic"},
        ],
    )
    input_items = captured["body"].get("input") or []
    assert {"type": "image_generation_call", "id": "img_abc"} in input_items
    assert {"role": "user", "content": "make it more realistic"} in input_items


# ── output translation surfaces tool_start + tool_end ────────────────


def test_image_generation_done_emits_tool_event_chunks(monkeypatch):
    events = _collect_tool_events(monkeypatch)
    image_events = [
        e
        for e in events
        if e.get("tool_name") == "image_generation"
        or (e.get("type") == "tool_end" and e.get("image_b64"))
    ]
    starts = [e for e in image_events if e.get("type") == "tool_start"]
    ends = [e for e in image_events if e.get("type") == "tool_end"]
    assert len(starts) == 1, image_events
    assert len(ends) == 1, image_events
    assert starts[0]["arguments"] == {
        "kind": "image",
        "prompt": "A photorealistic cat sitting",
        "openai_image_generation_call_id": "img_abc",
    }
    assert ends[0]["image_b64"] == "AAAA"
    assert ends[0]["image_mime"] == "image/png"
    assert ends[0]["size"] == "1024x1024"
    assert ends[0]["quality"] == "high"
    assert ends[0]["background"] == "opaque"
