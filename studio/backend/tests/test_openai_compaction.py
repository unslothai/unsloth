# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for OpenAI Responses API context_management wiring.

OpenAI's Responses API supports server-side compaction via
``context_management: [{type:"compaction", compact_threshold:N}]``. No
beta header, no dated version pin; the threshold is silently accepted and
compaction runs when the rendered prompt crosses it.

These pin: the body shape when threshold is set on cloud OpenAI, the
silent no-op on non-cloud base URLs, and the omitted-threshold
pass-through.
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _capture(monkeypatch, *, base_url: str, threshold) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        # Empty Responses-shaped SSE stream so the helper exits cleanly.
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
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 32,
            reasoning_effort = "medium",
            compaction_threshold = threshold,
        ):
            pass
        await client.close()

    _drive(run())
    return captured


def _capture_at_threshold(
    *args,
    threshold = 200_000,
    **kwargs,
):
    """_capture at the shared 200k compaction threshold."""
    return _capture(*args, threshold = threshold, **kwargs)


# ── cloud OpenAI carries the compaction field verbatim ──────────────


def test_cloud_openai_sets_compaction_block(monkeypatch):
    captured = _capture_at_threshold(monkeypatch, base_url = "https://api.openai.com/v1")
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]


def test_cloud_openai_below_default_threshold_passes_through(monkeypatch):
    # Unsloth doesn't clamp the OpenAI side -- the API accepts whatever the
    # caller sends, so a small probe like 60k still goes through.
    captured = _capture(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        threshold = 60_000,
    )
    assert captured["body"]["context_management"] == [
        {"type": "compaction", "compact_threshold": 60_000}
    ]


def test_cloud_openai_threshold_capped_at_200k(monkeypatch):
    captured = _capture(monkeypatch, base_url = "https://api.openai.com/v1", threshold = 750_000)
    assert captured["body"]["context_management"] == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]


# ── non-cloud bases drop the field ──────────────────────────────────


def test_non_cloud_base_silently_drops_compaction(monkeypatch):
    # ollama / llama.cpp / "custom" presets collapse to provider="openai"
    # but lack context_management. Sending the field would 400 them, so it
    # must NOT appear on the wire.
    captured = _capture_at_threshold(monkeypatch, base_url = "http://127.0.0.1:11434/v1")
    assert "context_management" not in captured["body"]


# ── Azure OpenAI Foundry is treated as cloud ────────────────────────


def test_azure_openai_base_url_carries_compaction_block(monkeypatch):
    # Azure OpenAI Foundry exposes the same /v1/responses extensions
    # (context_management, prompt_cache_retention, container shell) under
    # a *.openai.azure.com base URL. Treat it as cloud so the compaction
    # field reaches the API.
    captured = _capture_at_threshold(
        monkeypatch,
        base_url = "https://my-resource.openai.azure.com/openai/v1",
    )
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]
    # Sibling Azure-cloud extension: prompt_cache_retention should also
    # be set so caching works the same on Azure deployments.
    assert captured["body"].get("prompt_cache_retention") == "24h"


def test_azure_openai_mixed_case_base_url_matches(monkeypatch):
    # Case-insensitive match so URLs copy-pasted from the Azure portal
    # (which sometimes capitalise the resource name) still get the
    # cloud-only fields.
    captured = _capture(
        monkeypatch,
        base_url = "https://My-Resource.OpenAI.Azure.Com/openai/v1",
        threshold = 50_000,
    )
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 50_000}
    ]


def test_cloud_gate_uses_hostname_not_substring(monkeypatch):
    # CodeQL py/incomplete-url-substring-sanitization: an attacker
    # controlling base_url could embed `api.openai.com` or
    # an Azure managed suffix in a path or subdomain on an arbitrary host to
    # slip cloud-only body fields to their own server. The
    # hostname-anchored helper must reject both shapes.
    for evil in [
        "https://evil.com/api.openai.com/v1",
        "https://api.openai.com.attacker.com/v1",
        "https://attacker.com/.openai.azure.com/v1",
        "https://my-resource.openai.azure.com.attacker.com/openai/v1",
        "https://attacker.com/.services.ai.azure.com/openai/v1",
        "https://my-resource.services.ai.azure.com.attacker.com/openai/v1",
        "https://my-resource.services.ai.azure.com@attacker.com/openai/v1",
    ]:
        captured = _capture_at_threshold(monkeypatch, base_url = evil)
        assert "context_management" not in captured["body"], evil
        assert "prompt_cache_retention" not in captured["body"], evil


# ── omitted threshold leaves body untouched ─────────────────────────


def test_omitted_threshold_no_body_field(monkeypatch):
    captured = _capture(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        threshold = None,
    )
    assert "context_management" not in captured["body"]


# ── schema floor matches what the upstream API actually accepts ────


def test_chat_completion_request_accepts_any_positive_compaction_threshold():
    # Codex follow-up: the field is a no-op for non-cloud OpenAI bases and
    # every non-OpenAI provider, so a cross-provider schema floor would
    # 422 valid Anthropic / ollama / llama.cpp requests carrying it. Keep
    # the schema floor at ge=1 (any positive int) and let per-provider
    # helpers (_stream_openai_responses / _stream_anthropic) enforce or
    # clamp the real floor.
    import pytest as _pytest

    from models.inference import ChatCompletionRequest

    # Non-positive values rejected so blank-string posts don't sneak in.
    with _pytest.raises(Exception):
        ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "compaction_threshold": 0,
            }
        )

    # Any positive int passes schema validation, including values that
    # are no-ops on the OpenAI cloud path. Intentional -- the OpenAI
    # helper drops the field on non-cloud bases and forwards as-is on
    # cloud bases; if it's below the model's effective floor, the upstream
    # API surfaces the error.
    for v in (1, 5_000, 9_999, 10_000, 200_000):
        req = ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "compaction_threshold": v,
            }
        )
        assert req.compaction_threshold == v


def _stream(
    monkeypatch,
    handler,
    *,
    messages,
    threshold = None,
    base_url = "https://api.openai.com/v1",
):
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
        lines = [
            line
            async for line in client.stream_chat_completion(
                messages = messages,
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 32,
                compaction_threshold = threshold,
            )
        ]
        await client.close()
        return lines

    return _drive(run())


_EMPTY_COMPLETED = (
    b"event: response.completed\n"
    b'data: {"type":"response.completed",'
    b'"response":{"output":[],"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
)


def test_a_compaction_output_item_is_handed_to_the_client(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = (
                b"event: response.output_item.done\n"
                b'data: {"type":"response.output_item.done","output_index":0,'
                b'"item":{"type":"compaction","id":"cmp_1","encrypted_content":"gAAAA-opaque"}}\n\n'
                + _EMPTY_COMPLETED
            ),
            headers = {"content-type": "text/event-stream"},
        )

    lines = _stream(
        monkeypatch, handler, messages = [{"role": "user", "content": "hi"}], threshold = 200_000
    )
    events = [
        json.loads(line[len("data:") :])["_toolEvent"]
        for line in lines
        if line.startswith("data:") and "compaction_block" in line
    ]
    assert [event["encrypted_content"] for event in events] == ["gAAAA-opaque"]


def test_a_replayed_compaction_item_replaces_the_history_it_covers(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    _stream(
        monkeypatch,
        handler,
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "turn 1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "encrypted_content": "gAAAA-opaque"},
                    {"type": "text", "text": "answer 1"},
                ],
            },
            {"role": "user", "content": "turn 2"},
        ],
    )
    items = captured["body"]["input"]
    assert items[0] == {"type": "compaction", "encrypted_content": "gAAAA-opaque"}
    assert "turn 1" not in json.dumps(items)
    assert "turn 2" in json.dumps(items)
    assert captured["body"]["instructions"] == "Be brief."


def test_a_deployment_without_compaction_is_retried_without_it(monkeypatch):
    bodies: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        bodies.append(body)
        if "context_management" in body:
            return httpx.Response(
                400,
                json = {
                    "error": {
                        "message": "compact_threshold is not enabled.",
                        "type": "invalid_request_error",
                        "param": "compact_threshold",
                        "code": "unsupported_parameter",
                    }
                },
            )
        return httpx.Response(
            200, content = _EMPTY_COMPLETED, headers = {"content-type": "text/event-stream"}
        )

    lines = _stream(
        monkeypatch,
        handler,
        messages = [{"role": "user", "content": "hi"}],
        threshold = 96_000,
        base_url = "https://myres.openai.azure.com/openai/v1",
    )
    assert len(bodies) == 2
    assert "context_management" not in bodies[1]
    assert not any('"error"' in line for line in lines)


def test_build_external_messages_passes_the_compaction_item_to_openai():
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "encrypted_content": "gAAAA-opaque"},
                    {"type": "text", "text": "answer"},
                ],
            }
        )
    ]
    out = _build_external_messages(msgs, supports_vision = True, provider_type = "openai")
    assert out[0]["content"][0] == {"type": "compaction", "encrypted_content": "gAAAA-opaque"}
    anthropic = _build_external_messages(msgs, supports_vision = True, provider_type = "anthropic")
    deepseek = _build_external_messages(msgs, supports_vision = True, provider_type = "deepseek")
    for out in (anthropic, deepseek):
        assert "gAAAA-opaque" not in json.dumps(out)
