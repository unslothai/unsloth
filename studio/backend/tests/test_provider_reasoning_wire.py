# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient

_OPENAI_SSE = 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
_GEMINI_SSE = (
    'data: {"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],'
    '"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}\n\n'
)
_ANTHROPIC_SSE = 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
}
_SSE = {"gemini": _GEMINI_SSE, "anthropic": _ANTHROPIC_SSE}


def _body(provider_type: str, model: str, **kwargs) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            content = _SSE.get(provider_type, _OPENAI_SSE),
            headers = {"content-type": "text/event-stream"},
        )

    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(
        provider_type = provider_type,
        base_url = _BASE_URLS.get(provider_type, "http://127.0.0.1:8000/v1"),
        api_key = "test-key",
    )

    async def run() -> None:
        try:
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = model,
                **kwargs,
            ):
                pass
        finally:
            await client.close()
            await mock_client.aclose()

    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    ep_mod._http_client = mock_client
    try:
        loop.run_until_complete(run())
    finally:
        ep_mod._http_client = previous
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return captured["body"]


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
        ("minimal", "low"),
        ("medium", "high"),
        ("xhigh", "high"),
    ],
)
def test_deepseek_effort_is_forwarded_with_thinking_enabled(effort, expected):
    body = _body("deepseek", "deepseek-v4-flash", reasoning_effort = effort, enable_thinking = True)
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == expected


def test_deepseek_off_disables_thinking():
    body = _body("deepseek", "deepseek-v4-flash", reasoning_effort = "none", enable_thinking = False)
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    body = _body("deepseek", "deepseek-v4-flash", enable_thinking = False)
    assert body["thinking"] == {"type": "disabled"}


def test_deepseek_bare_toggle_and_silence():
    assert _body("deepseek", "deepseek-v4-flash", enable_thinking = True)["thinking"] == {
        "type": "enabled"
    }
    body = _body("deepseek", "deepseek-chat")
    assert "thinking" not in body and "reasoning_effort" not in body


def test_qwen_toggle_is_a_top_level_enable_thinking():
    assert _body("qwen", "qwen3.5-plus", enable_thinking = True)["enable_thinking"] is True
    assert _body("qwen", "qwen3.5-plus", enable_thinking = False)["enable_thinking"] is False
    body = _body("qwen", "qwen3.5-plus", reasoning_effort = "high")
    assert "reasoning_effort" not in body and "enable_thinking" not in body


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
def test_huggingface_router_takes_the_effort_verbatim(effort):
    assert (
        _body("huggingface", "openai/gpt-oss-120b", reasoning_effort = effort)["reasoning_effort"]
        == effort
    )


def test_huggingface_off_and_silence():
    assert (
        _body("huggingface", "openai/gpt-oss-120b", enable_thinking = False)["reasoning_effort"]
        == "none"
    )
    assert (
        _body("huggingface", "openai/gpt-oss-120b", reasoning_effort = "none")["reasoning_effort"]
        == "none"
    )
    assert "reasoning_effort" not in _body(
        "huggingface", "openai/gpt-oss-120b", enable_thinking = True
    )


@pytest.mark.parametrize("provider_type", ["vllm", "llama_cpp"])
def test_local_servers_get_template_toggle_and_effort(provider_type):
    body = _body(
        provider_type, "openai/gpt-oss-20b", reasoning_effort = "xhigh", enable_thinking = True
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["reasoning_effort"] == "high"
    body = _body(provider_type, "Qwen/Qwen3-14B", enable_thinking = False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # vLLM through 0.16 types reasoning_effort as low | medium | high, so off never goes top-level.
    assert "reasoning_effort" not in body
    body = _body(provider_type, "Qwen/Qwen3-14B", reasoning_effort = "none")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in body
    body = _body(provider_type, "Qwen/Qwen3-14B", reasoning_effort = "medium")
    assert body["reasoning_effort"] == "medium"
    assert "chat_template_kwargs" not in body


def test_custom_gateway_still_receives_nothing():
    body = _body("custom", "some-model", reasoning_effort = "high", enable_thinking = True)
    assert (
        "reasoning_effort" not in body
        and "chat_template_kwargs" not in body
        and "thinking" not in body
    )


def test_mistral_medium_3_5_takes_the_documented_two_value_form():
    assert (
        _body("mistral", "mistral-medium-3-5", reasoning_effort = "medium")["reasoning_effort"]
        == "high"
    )
    assert (
        _body("mistral", "mistral-medium-3-5", enable_thinking = True)["reasoning_effort"] == "high"
    )
    assert (
        _body("mistral", "mistral-medium-3-5", reasoning_effort = "none")["reasoning_effort"]
        == "none"
    )
    assert (
        _body("mistral", "mistral-medium-3-5", enable_thinking = False)["reasoning_effort"] == "none"
    )
    assert "reasoning_effort" not in _body("mistral", "mistral-large-latest")


def test_mistral_known_specs_are_unchanged():
    assert (
        _body("mistral", "magistral-medium-latest", reasoning_effort = "high")["prompt_mode"]
        == "reasoning"
    )
    assert (
        _body("mistral", "mistral-small-latest", reasoning_effort = "high")["reasoning_effort"]
        == "high"
    )


def test_gemini_families_past_three_use_thinking_level():
    body = _body(
        "gemini",
        "gemini-4.1-flash",
        reasoning_effort = "medium",
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = 64,
    )
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "medium"}
    body = _body(
        "gemini",
        "gemini-4-pro",
        reasoning_effort = "minimal",
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = 64,
    )
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    body = _body(
        "gemini",
        "gemini-2.5-flash",
        reasoning_effort = "none",
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = 64,
    )
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_anthropic_models_outside_the_spec_take_the_adaptive_shape():
    body = _body(
        "anthropic",
        "claude-opus-6",
        reasoning_effort = "xhigh",
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = 4096,
    )
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "xhigh"}
    body = _body(
        "anthropic",
        "claude-opus-6",
        reasoning_effort = "none",
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = 4096,
    )
    assert "output_config" not in body and "thinking" not in body


def test_gemma_on_gemini_toggles_with_thinking_level_not_budget():
    sampling = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 64}
    body = _body("gemini", "gemma-4-31b-it", enable_thinking = True, **sampling)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}
    body = _body("gemini", "gemma-4-26b-a4b-it", enable_thinking = False, **sampling)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    body = _body("gemini", "gemma-4-31b-it", **sampling)
    assert "thinkingConfig" not in body.get("generationConfig", {})


# Wire proof for older Claude models, run against a mocked Anthropic stream rather than the live API. Anthropic documents
# adaptive thinking as a 400 on Claude 4.5 and earlier, and budget_tokens as the only thinking mode there:
# https://platform.claude.com/docs/en/build-with-claude/extended-thinking
@pytest.mark.parametrize(
    "model,effort,budget",
    [
        ("claude-sonnet-4-20250514", "high", 4096),
        ("claude-opus-4-1-20250805", "low", 1024),
        ("claude-3-7-sonnet-20250219", "medium", 2048),
    ],
)
def test_earlier_claude_thinking_models_keep_budget_tokens(model, effort, budget):
    body = _body(
        "anthropic", model, reasoning_effort = effort, temperature = 0.7, top_p = 0.95, max_tokens = 8192
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": budget}
    assert "output_config" not in body


@pytest.mark.parametrize(
    "model", ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
)
def test_claude_models_without_extended_thinking_stream_without_a_thinking_field(model):
    sampling = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096}
    body = _body("anthropic", model, reasoning_effort = "high", **sampling)
    assert "thinking" not in body and "output_config" not in body
    body = _body("anthropic", model, **sampling)
    assert "thinking" not in body and "output_config" not in body


@pytest.mark.parametrize("model", ["claude-opus-4-10-20260901", "claude-opus-4-15-20260901"])
def test_a_two_digit_claude_minor_is_not_read_as_the_4_1_spec(model):
    """`claude-opus-4-1` is a prefix of `claude-opus-4-15`, so an unbounded match sent a model
    numbered after 4.6 the manual budget shape and dropped xhigh/max."""
    body = _body("anthropic", model, reasoning_effort = "xhigh", max_tokens = 8192)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "xhigh"}


def test_a_capitalized_claude_id_keeps_its_thinking_spec():
    """Model ids reach us as typed in a custom model field; the version helpers lowercase, so the
    spec lookup must too or the id silently loses extended thinking."""
    body = _body("anthropic", "Claude-Opus-4-1-20250805", reasoning_effort = "low", max_tokens = 8192)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}


@pytest.mark.parametrize("model", ["claude-opus-4-6", "Claude-Opus-4-6", "CLAUDE-SONNET-4-6"])
def test_the_4_6_xhigh_remap_reads_the_id_the_same_way_the_spec_lookup_does(model):
    """4.6 spells the top adaptive tier `max`; the spec lookup lowercases, so a capitalized id was
    allowed `xhigh` and then missed the remap that turns it into `max`."""
    body = _body("anthropic", model, reasoning_effort = "xhigh", max_tokens = 8192)
    assert body["output_config"] == {"effort": "max"}


# Mistral documents reasoning_effort for mistral-small-latest and mistral-medium-3-5 only, with
# values "high" and "none": https://docs.mistral.ai/capabilities/reasoning/
# mistral-large, codestral and the older mistral-medium releases are absent from that page and
# reject the parameter, so no caller-supplied effort may reach them.
@pytest.mark.parametrize(
    "model", ["mistral-large-latest", "codestral-latest", "mistral-medium-2505"]
)
@pytest.mark.parametrize(
    "controls",
    [
        {"reasoning_effort": "none"},
        {"reasoning_effort": "high"},
        {"reasoning_effort": "medium"},
        {"enable_thinking": True},
        {"enable_thinking": False},
        {"enable_thinking": False, "reasoning_effort": "none"},
    ],
)
def test_a_mistral_model_outside_the_reasoning_docs_gets_no_effort_whatever_the_caller_sends(
    model, controls
):
    assert "reasoning_effort" not in _body("mistral", model, **controls)


def _snapshot_reasoning_models(provider_type: str) -> dict[str, list[str]]:
    """The ids the committed frontend snapshot marks reasoning-capable, with their effort lists.

    Read out of the TypeScript rather than a fixture: the point is to catch the real file drifting
    away from the wire allowlist, which a copy could not do."""
    path = (
        Path(__file__).resolve().parents[2] / "frontend/src/features/chat/model-catalog-snapshot.ts"
    )
    text = path.read_text(encoding = "utf-8")
    start = text.index(f'"{provider_type}": {{')
    bucket = text[start : text.index("\n  },", start)]
    found: dict[str, list[str]] = {}
    for model_id, entry in re.findall(r'"([^"]+)":\s*(\{.*\})', bucket):
        parsed = json.loads(entry)
        if parsed.get("reasoning") and parsed.get("efforts"):
            found[model_id] = parsed["efforts"]
    return found


def test_every_mistral_model_the_snapshot_gives_an_effort_ladder_is_on_the_wire_allowlist():
    """The composer offers a Thinking control whenever the catalog says the model reasons, so an id
    the catalog lists but the wire drops renders a control that silently does nothing."""
    snapshot = _snapshot_reasoning_models("mistral")
    assert (
        snapshot
    ), "the snapshot's mistral bucket has no reasoning entries, so this proves nothing"
    for model, efforts in sorted(snapshot.items()):
        spec = ep_mod._mistral_thinking_spec(model)
        assert spec.style == "reasoning_effort", f"{model} is offered a ladder but sends nothing"
        # The frontend clamps a catalog ladder to CATALOG_REASONING_WIRE.mistral before it renders,
        # so only the values in both sets can ever be selected; those are the ones that must arrive.
        for effort in set(efforts) & set(spec.efforts):
            assert _body("mistral", model, reasoning_effort = effort)["reasoning_effort"] == effort
