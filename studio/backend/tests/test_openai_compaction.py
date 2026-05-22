# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for OpenAI Responses API context_management wiring.

OpenAI's Responses API supports server-side compaction via
``context_management: [{type:"compaction", compact_threshold:N}]``.
There is no beta header and no dated version pin; the threshold is
silently accepted and the API runs the compaction step when the
rendered prompt crosses it.

These tests pin: the body shape when threshold is set on cloud OpenAI,
the silent no-op when the base URL is non-cloud, and the
omitted-threshold pass-through.
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
        # Send an empty Responses-shaped SSE stream so the helper exits
        # cleanly.
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


# ── cloud OpenAI carries the compaction field verbatim ──────────────


def test_cloud_openai_sets_compaction_block(monkeypatch):
    captured = _capture(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        threshold = 200_000,
    )
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]


def test_cloud_openai_below_default_threshold_passes_through(monkeypatch):
    # Studio doesn't clamp the OpenAI side -- the API accepts whatever
    # the caller sends, so a small probe like 60k still goes through.
    captured = _capture(
        monkeypatch,
        base_url = "https://api.openai.com/v1",
        threshold = 60_000,
    )
    assert captured["body"]["context_management"] == [
        {"type": "compaction", "compact_threshold": 60_000}
    ]


# ── non-cloud bases drop the field ──────────────────────────────────


def test_non_cloud_base_silently_drops_compaction(monkeypatch):
    # ollama / llama.cpp / "custom" presets collapse to provider="openai"
    # but don't implement context_management. Sending the field would
    # 400 those servers, so it must NOT appear on the wire.
    captured = _capture(
        monkeypatch,
        base_url = "http://127.0.0.1:11434/v1",
        threshold = 200_000,
    )
    assert "context_management" not in captured["body"]


# ── Azure OpenAI Foundry is treated as cloud ────────────────────────


def test_azure_openai_base_url_carries_compaction_block(monkeypatch):
    # Azure OpenAI Foundry exposes the same /v1/responses extensions
    # (context_management, prompt_cache_retention, container shell)
    # under a *.openai.azure.com base URL. Treat it as cloud so the
    # compaction field actually reaches the API.
    captured = _capture(
        monkeypatch,
        base_url = "https://my-resource.openai.azure.com/openai/v1",
        threshold = 200_000,
    )
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]
    # Sibling Azure-cloud extension: prompt_cache_retention should also
    # be set so caching works the same way on Azure deployments.
    assert captured["body"].get("prompt_cache_retention") == "24h"


def test_azure_openai_mixed_case_base_url_matches(monkeypatch):
    # The match is case-insensitive so URLs copy-pasted from the Azure
    # portal (which sometimes capitalise the resource name) still get
    # the cloud-only fields.
    captured = _capture(
        monkeypatch,
        base_url = "https://My-Resource.OpenAI.Azure.Com/openai/v1",
        threshold = 50_000,
    )
    assert captured["body"].get("context_management") == [
        {"type": "compaction", "compact_threshold": 50_000}
    ]


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
    # Codex follow-up: the field is documented as a no-op for non-cloud
    # OpenAI bases and every non-OpenAI provider, so a cross-provider
    # schema floor would 422 perfectly valid Anthropic / ollama /
    # llama.cpp requests that happen to carry the field. Keep schema
    # floor at ge=1 (any positive int) and rely on per-provider
    # helpers (_stream_openai_responses / _stream_anthropic) to
    # enforce or clamp the real floor.
    import pytest as _pytest

    from models.inference import ChatCompletionRequest

    # Non-positive values still rejected so blank-string posts don't
    # sneak through.
    with _pytest.raises(Exception):
        ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "compaction_threshold": 0,
            }
        )

    # Any positive int passes schema validation, including values that
    # would be no-ops on the OpenAI cloud path. This is intentional --
    # the OpenAI helper drops the field on non-cloud bases and
    # forwards-as-is on cloud bases; if the value is below the model's
    # effective floor, the upstream API surfaces the error.
    for v in (1, 5_000, 9_999, 10_000, 200_000):
        req = ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "compaction_threshold": v,
            }
        )
        assert req.compaction_threshold == v
