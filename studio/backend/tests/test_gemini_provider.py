# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the native Gemini API translation layer.

Gemini does NOT speak OpenAI Chat Completions on its primary endpoint
(`streamGenerateContent`). `_stream_gemini` in
`core/inference/external_provider.py` translates between the two shapes:

  Request:
    OpenAI messages [{role, content}]
      -> Gemini contents [{role, parts: [{text}|{inlineData}|{functionCall}|...]}]
        + systemInstruction.parts[].text for role=system messages
        + generationConfig.{temperature,topP,topK,maxOutputTokens}
        + tools[{googleSearch:{}}] for web_search
        + tools[{codeExecution:{}}] for code_execution
        + responseModalities=[TEXT,IMAGE] for Nano Banana (gemini-2.5-flash-image)
        + cachedContent for prompt caching

  Response:
    Gemini SSE chunks { candidates:[{content:{parts:[...]}, finishReason}],
                        usageMetadata:{promptTokenCount, candidatesTokenCount} }
      -> OpenAI chat.completion.chunk frames
        (delta.content for text, delta.tool_calls for functionCall,
         _toolEvent for image_b64/web_search, usage block before [DONE])

These tests pin the outbound body shape AND the inbound translation
using httpx.MockTransport (no live network). Mirrors the structure of
test_anthropic_cache_ttl.py and test_openai_image_generation.py.
"""

import asyncio
import base64
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_gemini_client(
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
) -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "gemini",
        base_url = base_url,
        api_key = "AIza-test-key",
    )


def _mock_http(monkeypatch, handler):
    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )


def _gemini_sse(events: list[dict]) -> bytes:
    """Encode a list of dicts as Gemini-style SSE frames (`data:` lines)."""
    chunks: list[str] = []
    for event in events:
        chunks.append(f"data: {json.dumps(event)}")
        chunks.append("")
    return ("\n".join(chunks) + "\n").encode("utf-8")


def _capture_body(monkeypatch, **kwargs) -> dict:
    """Drive a single stream and return the captured outbound request body."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        captured["method"] = request.method
        # Minimal valid Gemini stream so the helper can complete.
        return httpx.Response(
            200,
            content = _gemini_sse(
                [
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "ok"}],
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 1,
                            "candidatesTokenCount": 1,
                        },
                    }
                ]
            ),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    messages = kwargs.pop("messages", [{"role": "user", "content": "hi"}])
    model = kwargs.pop("model", "gemini-2.5-flash")
    temperature = kwargs.pop("temperature", 0.7)
    top_p = kwargs.pop("top_p", 0.95)
    max_tokens = kwargs.pop("max_tokens", 64)

    async def run():
        client = _make_gemini_client()
        async for _ in client.stream_chat_completion(
            messages = messages,
            model = model,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            **kwargs,
        ):
            pass
        await client.close()

    _drive(run())
    return captured


def _collect(monkeypatch, sse_events, **kwargs) -> list[str]:
    """Drive a stream with a custom set of SSE events and return raw lines."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _gemini_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    messages = kwargs.pop("messages", [{"role": "user", "content": "hi"}])
    model = kwargs.pop("model", "gemini-2.5-flash")
    temperature = kwargs.pop("temperature", 0.7)
    top_p = kwargs.pop("top_p", 0.95)
    max_tokens = kwargs.pop("max_tokens", 64)

    out: list[str] = []

    async def run():
        client = _make_gemini_client()
        async for line in client.stream_chat_completion(
            messages = messages,
            model = model,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            **kwargs,
        ):
            out.append(line)
        await client.close()

    _drive(run())
    return out


def _parse_chunks(lines: list[str]) -> list[dict]:
    out: list[dict] = []
    for raw in lines:
        if not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return out


# ── request body translation ─────────────────────────────────────────


def test_request_body_uses_contents_and_parts_shape(monkeypatch):
    """OpenAI messages must be translated to Gemini's `contents` shape."""
    captured = _capture_body(
        monkeypatch,
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Follow up"},
        ],
    )
    body = captured["body"]
    # system -> systemInstruction
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}, body
    # user/assistant -> contents with role user/model
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi there"}]},
        {"role": "user", "parts": [{"text": "Follow up"}]},
    ], body["contents"]
    # generationConfig fields map across with Google's casing.
    gc = body["generationConfig"]
    assert gc["temperature"] == 0.7
    assert gc["topP"] == 0.95
    assert gc["maxOutputTokens"] == 64


def test_request_url_targets_stream_generate_content(monkeypatch):
    """Helper must POST to /v1beta/models/{model}:streamGenerateContent?alt=sse."""
    captured = _capture_body(monkeypatch, model = "gemini-2.5-pro")
    url = captured["url"]
    assert ":streamGenerateContent" in url, url
    assert "alt=sse" in url, url
    assert "/v1beta/models/gemini-2.5-pro" in url, url
    assert captured["method"] == "POST"


def test_request_auth_header_uses_x_goog_api_key(monkeypatch):
    """API key must be sent on `x-goog-api-key`, not Authorization."""
    captured = _capture_body(monkeypatch)
    hdrs = captured["headers"]
    assert hdrs.get("x-goog-api-key") == "AIza-test-key", hdrs
    assert "authorization" not in {k.lower() for k in hdrs}, hdrs


def test_top_k_forwarded_only_when_positive(monkeypatch):
    """top_k is opt-in; only positive integers reach the wire."""
    captured = _capture_body(monkeypatch, top_k = 40)
    assert captured["body"]["generationConfig"]["topK"] == 40

    captured = _capture_body(monkeypatch, top_k = 0)
    assert "topK" not in captured["body"]["generationConfig"]


def test_presence_penalty_forwarded_to_generation_config(monkeypatch):
    """A non-zero presence_penalty reaches generationConfig.presencePenalty."""
    captured = _capture_body(monkeypatch, presence_penalty = 0.7)
    assert captured["body"]["generationConfig"]["presencePenalty"] == 0.7

    # And the default of zero is omitted, matching top_k semantics.
    captured = _capture_body(monkeypatch, presence_penalty = 0.0)
    assert "presencePenalty" not in captured["body"]["generationConfig"]


# ── thinkingConfig translation ────────────────────────────────────────


def test_thinking_disabled_sets_budget_zero_on_flash(monkeypatch):
    """enable_thinking=False on Flash-tier sets thinkingBudget=0."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3.5-flash",
        enable_thinking = False,
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc == {"thinkingBudget": 0}, tc


def test_thinking_disabled_pro_tier_uses_small_budget(monkeypatch):
    """Pro-tier ids 400 on thinkingBudget=0 ("only works in thinking mode");
    a small positive budget is forwarded instead so the turn doesn't fail.
    """
    for model in (
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        "gemini-pro-latest",
    ):
        captured = _capture_body(
            monkeypatch,
            model = model,
            enable_thinking = False,
        )
        tc = captured["body"]["generationConfig"].get("thinkingConfig")
        assert tc is not None, f"missing thinkingConfig for {model}: {captured}"
        assert tc["thinkingBudget"] > 0, (model, tc)


def test_reasoning_effort_levels_map_to_budgets(monkeypatch):
    """The OpenAI/Anthropic effort ladder must translate to Gemini budgets."""
    cases = {
        "minimal": 512,
        "low": 2048,
        "medium": 8192,
        "high": 24576,
        "max": -1,  # dynamic
        "xhigh": -1,
    }
    for effort, expected in cases.items():
        captured = _capture_body(
            monkeypatch,
            model = "gemini-3.5-flash",
            reasoning_effort = effort,
        )
        tc = captured["body"]["generationConfig"].get("thinkingConfig")
        assert tc == {"thinkingBudget": expected}, (effort, tc)


def test_reasoning_effort_none_disables_on_flash(monkeypatch):
    """`reasoning_effort='none'` is shorthand for thinking off (Flash)."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3.5-flash",
        reasoning_effort = "none",
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc == {"thinkingBudget": 0}, tc


def test_thinking_default_omits_thinking_config(monkeypatch):
    """When neither knob is supplied, thinkingConfig is omitted entirely
    (Google's server-side default applies)."""
    captured = _capture_body(monkeypatch, model = "gemini-3.5-flash")
    gc = captured["body"]["generationConfig"]
    assert "thinkingConfig" not in gc, gc


def test_nano_banana_alias_routes_through_image_modalities(monkeypatch):
    """`nano-banana-pro-preview` is an alias for the Pro image model and
    must set responseModalities=[TEXT,IMAGE] same as the `*-image` ids."""
    captured = _capture_body(
        monkeypatch,
        model = "nano-banana-pro-preview",
    )
    gc = captured["body"]["generationConfig"]
    assert gc.get("responseModalities") == ["TEXT", "IMAGE"], gc


def test_image_models_skip_thinking_config(monkeypatch):
    """Image-tier ids do not benefit from a visible thinking knob and
    must NOT forward thinkingConfig even when stale UI state still
    sends `reasoning_effort` or `enable_thinking=False`."""
    for model in (
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "nano-banana-pro-preview",
    ):
        captured = _capture_body(
            monkeypatch,
            model = model,
            reasoning_effort = "high",
            enable_thinking = False,
        )
        gc = captured["body"]["generationConfig"]
        assert "thinkingConfig" not in gc, (model, gc)


def test_image_models_drop_text_only_tools(monkeypatch):
    """Image-tier ids reject googleSearch / codeExecution wiring; drop
    them silently so a stale `enabled_tools` array does not 400 the turn."""
    for model in (
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "nano-banana-pro-preview",
    ):
        captured = _capture_body(
            monkeypatch,
            model = model,
            enabled_tools = ["web_search", "code_execution"],
        )
        assert "tools" not in captured["body"], (
            model,
            captured["body"].get("tools"),
        )


def test_gemini_35_pro_recognized_as_pro_thinking(monkeypatch):
    """`gemini-3.5-pro` rejects thinkingBudget=0 with "only works in
    thinking mode" -- coerce to a positive budget."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3.5-pro",
        enable_thinking = False,
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc is not None and tc["thinkingBudget"] > 0, tc


def test_legacy_openai_base_url_normalized(monkeypatch):
    """Saved Gemini providers carrying the legacy `/v1beta/openai` base
    (from the pre-PR OpenAI-compat plumbing) now point at the native
    endpoint without the user re-saving the connection."""
    client = ExternalProviderClient(
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai",
        api_key = "AIza-test-key",
    )
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_finish_reason_swaps_to_tool_calls_when_function_call_emitted(monkeypatch):
    """Gemini emits finishReason="STOP" even for pure functionCall turns;
    surface as `tool_calls` so OAI clients trigger tool execution."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "lookup", "args": {"k": "v"}}}
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    finish_chunks = [
        c for c in chunks if c.get("choices", [{}])[0].get("finish_reason") is not None
    ]
    assert finish_chunks, chunks
    assert finish_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls", chunks


def test_thought_signature_round_trips_into_gemini_function_call(monkeypatch):
    """An assistant tool_call carrying `extra_content.google.thought_signature`
    must echo the value back as a sibling of the Gemini functionCall part."""
    captured = _capture_body(
        monkeypatch,
        messages = [
            {"role": "user", "content": "lookup x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "SIG-ABC"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "name": "lookup",
                "content": "{}",
            },
        ],
    )
    contents = captured["body"]["contents"]
    fc_turn = next((c for c in contents if c["role"] == "model"), None)
    assert fc_turn is not None, contents
    fc_part = next(
        (p for p in fc_turn["parts"] if "functionCall" in p),
        None,
    )
    assert fc_part is not None, fc_turn
    assert fc_part.get("thoughtSignature") == "SIG-ABC", fc_part


def test_thought_signature_emitted_in_tool_call_delta(monkeypatch):
    """A Gemini functionCall part with `thoughtSignature` must surface
    that signature on the outbound OpenAI tool_calls delta via
    `extra_content.google.thought_signature`."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "lookup",
                                    "args": {"k": "v"},
                                    "id": "call_xyz",
                                },
                                "thoughtSignature": "SIG-FROM-GEMINI",
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
    ]
    chunks = _parse_chunks(_collect(monkeypatch, sse))
    deltas = [
        tc
        for c in chunks
        for tc in (c.get("choices", [{}])[0].get("delta", {}) or {}).get(
            "tool_calls", []
        )
    ]
    assert deltas, chunks
    sig = deltas[0].get("extra_content", {}).get("google", {}).get("thought_signature")
    assert sig == "SIG-FROM-GEMINI", deltas


def test_usage_chunk_includes_thoughts_tokens(monkeypatch):
    """`thoughtsTokenCount` is the hidden-reasoning slice of output;
    roll it into `output_tokens` AND surface it on
    `output_tokens_details.reasoning_tokens` so total_tokens reflects
    the full billable spend."""
    sse = [
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "ok"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 20,
                "totalTokenCount": 35,
            },
        }
    ]
    chunks = _parse_chunks(_collect(monkeypatch, sse))
    usage_chunk = next((c for c in chunks if isinstance(c.get("usage"), dict)), None)
    assert usage_chunk is not None, chunks
    usage = usage_chunk["usage"]
    assert usage.get("prompt_tokens") == 10, usage
    # candidates 5 + thoughts 20 = 25 output tokens; total = 35.
    assert usage.get("completion_tokens") == 25, usage
    assert usage.get("total_tokens") == 35, usage


# ── web_search forwarded as googleSearch tool ────────────────────────


def test_web_search_forwarded_as_google_search_tool(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        enabled_tools = ["web_search"],
    )
    tools = captured["body"].get("tools") or []
    assert {"googleSearch": {}} in tools, tools


def test_code_execution_forwarded_as_code_execution_tool(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        enabled_tools = ["code_execution"],
    )
    tools = captured["body"].get("tools") or []
    assert {"codeExecution": {}} in tools, tools


def test_omitted_tools_leaves_body_untouched(monkeypatch):
    captured = _capture_body(monkeypatch, enabled_tools = [])
    assert "tools" not in captured["body"], captured["body"]


# ── prompt caching passthrough ───────────────────────────────────────


def test_cached_content_pass_through(monkeypatch):
    """A string cache id on enable_prompt_caching is forwarded verbatim."""
    cache_name = "cachedContents/abc123"
    captured = _capture_body(
        monkeypatch,
        enable_prompt_caching = cache_name,
    )
    assert captured["body"].get("cachedContent") == cache_name


def test_boolean_caching_does_not_set_cached_content(monkeypatch):
    """Studio's existing True/False signals shouldn't fabricate a cache id."""
    captured = _capture_body(monkeypatch, enable_prompt_caching = True)
    assert "cachedContent" not in captured["body"]


# ── image generation: request modalities + response translation ──────


def test_image_model_sets_response_modalities(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
    )
    assert captured["body"]["generationConfig"]["responseModalities"] == [
        "TEXT",
        "IMAGE",
    ]


def test_image_generation_tool_sets_response_modalities(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash",
        enabled_tools = ["image_generation"],
    )
    assert captured["body"]["generationConfig"]["responseModalities"] == [
        "TEXT",
        "IMAGE",
    ]


def test_image_response_emits_image_b64_tool_event(monkeypatch):
    """`inlineData` parts become a tool_end with image_b64 + image_mime."""
    fake_b64 = base64.b64encode(b"PNG-BYTES").decode()
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": fake_b64,
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 0,
            },
        }
    ]
    lines = _collect(
        monkeypatch,
        sse,
        model = "gemini-2.5-flash-image",
    )
    chunks = _parse_chunks(lines)
    tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
    starts = [e for e in tool_events if e.get("type") == "tool_start"]
    ends = [e for e in tool_events if e.get("type") == "tool_end"]
    image_starts = [e for e in starts if e.get("tool_name") == "image_generation"]
    image_ends = [e for e in ends if e.get("image_b64")]
    assert len(image_starts) == 1, tool_events
    assert len(image_ends) == 1, tool_events
    assert image_ends[0]["image_b64"] == fake_b64
    assert image_ends[0]["image_mime"] == "image/png"


# ── function calling round-trips both directions ─────────────────────


def test_function_call_response_translates_to_tool_calls_delta(monkeypatch):
    """Gemini `functionCall` parts become OpenAI `tool_calls` delta chunks."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Paris"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 4,
            },
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    tool_call_chunks = [
        c
        for c in chunks
        if "_toolEvent" not in c
        and any(
            (isinstance(ch.get("delta"), dict) and "tool_calls" in ch["delta"])
            for ch in c.get("choices", [])
        )
    ]
    assert len(tool_call_chunks) == 1, chunks
    tc = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    args = json.loads(tc["function"]["arguments"])
    assert args == {"location": "Paris"}


def test_tool_message_translates_to_function_response_part(monkeypatch):
    """role=tool follow-ups are rewritten to functionResponse parts."""
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Paris"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "content": json.dumps({"temp_c": 18, "summary": "Sunny"}),
        },
    ]
    captured = _capture_body(monkeypatch, messages = messages)
    contents = captured["body"]["contents"]
    # Last turn must be a functionResponse part (Gemini wraps it as a
    # role=user turn carrying the result).
    last = contents[-1]
    assert last["role"] == "user", last
    fr = last["parts"][0].get("functionResponse")
    assert fr is not None, last
    assert fr["name"] == "get_weather"
    assert fr["response"] == {"temp_c": 18, "summary": "Sunny"}
    # And the assistant turn carries the original functionCall so the
    # model sees the round-trip context.
    assistant_turn = [c for c in contents if c["role"] == "model"][0]
    fc_part = next(
        (p for p in assistant_turn["parts"] if "functionCall" in p),
        None,
    )
    assert fc_part is not None, assistant_turn
    assert fc_part["functionCall"]["name"] == "get_weather"
    assert fc_part["functionCall"]["args"] == {"location": "Paris"}


def test_parallel_function_calls_get_distinct_tool_call_indices(monkeypatch):
    """Each emitted functionCall in one assistant turn needs its own
    tool_calls[*].index. Hardcoding index=0 collapses parallel calls
    onto a single slot in OpenAI-style reassemblers."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "call_alpha",
                                    "name": "search",
                                    "args": {"q": "alpha"},
                                }
                            },
                            {
                                "functionCall": {
                                    "id": "call_beta",
                                    "name": "search",
                                    "args": {"q": "beta"},
                                }
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4,
            },
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    tool_call_chunks = [
        c
        for c in chunks
        if "_toolEvent" not in c
        and any(
            (isinstance(ch.get("delta"), dict) and "tool_calls" in ch["delta"])
            for ch in c.get("choices", [])
        )
    ]
    assert len(tool_call_chunks) == 2, tool_call_chunks
    indices = [
        c["choices"][0]["delta"]["tool_calls"][0]["index"] for c in tool_call_chunks
    ]
    assert indices == [0, 1], indices


def test_function_call_ids_forwarded_into_gemini_function_call_part(monkeypatch):
    """OpenAI tool_call id rides functionCall.id so parallel calls disambiguate."""
    messages = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_alpha",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"q": "a"}),
                    },
                },
                {
                    "id": "call_beta",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"q": "b"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_alpha",
            "content": json.dumps({"hits": ["A"]}),
        },
        {
            "role": "tool",
            "tool_call_id": "call_beta",
            "content": json.dumps({"hits": ["B"]}),
        },
    ]
    captured = _capture_body(monkeypatch, messages = messages)
    contents = captured["body"]["contents"]
    assistant_parts = next(c for c in contents if c["role"] == "model")["parts"]
    call_ids = [p["functionCall"]["id"] for p in assistant_parts if "functionCall" in p]
    assert call_ids == ["call_alpha", "call_beta"], assistant_parts
    response_ids = [
        p["functionResponse"]["id"]
        for c in contents
        for p in c["parts"]
        if "functionResponse" in p
    ]
    assert response_ids == ["call_alpha", "call_beta"], contents


def test_parse_gemini_models_translates_native_catalog():
    """Gemini's native /v1beta/models payload becomes OpenAI-shape entries."""
    payload = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "baseModelId": "gemini-2.5-flash",
                "displayName": "Gemini 2.5 Flash",
                "supportedGenerationMethods": [
                    "generateContent",
                    "streamGenerateContent",
                ],
            },
            {
                "name": "models/embedding-001",
                "supportedGenerationMethods": ["embedContent"],
            },
            {
                "name": "models/gemini-2.5-pro",
            },
        ]
    }
    out = ExternalProviderClient._parse_gemini_models(payload)
    ids = [m["id"] for m in out]
    assert "gemini-2.5-flash" in ids
    assert "gemini-2.5-pro" in ids
    assert "embedding-001" not in ids
    flash = next(m for m in out if m["id"] == "gemini-2.5-flash")
    assert flash["display_name"] == "Gemini 2.5 Flash"
    assert flash["owned_by"] == "google"


def test_code_execution_parts_translate_to_code_execution_tool_events(monkeypatch):
    """executableCode + codeExecutionResult parts emit code_execution events."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "executableCode": {
                                    "language": "PYTHON",
                                    "code": "print(2+2)",
                                }
                            },
                            {
                                "codeExecutionResult": {
                                    "outcome": "OUTCOME_OK",
                                    "output": "4\n",
                                }
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4,
            },
        }
    ]
    lines = _collect(monkeypatch, sse, enabled_tools = ["code_execution"])
    chunks = _parse_chunks(lines)
    tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
    code_starts = [
        e
        for e in tool_events
        if e.get("type") == "tool_start" and e.get("tool_name") == "code_execution"
    ]
    code_ends = [
        e
        for e in tool_events
        if e.get("type") == "tool_end" and "4" in str(e.get("result", ""))
    ]
    assert len(code_starts) == 1, tool_events
    assert code_starts[0]["arguments"]["code"] == "print(2+2)"
    assert code_starts[0]["arguments"]["language"] == "python"
    assert len(code_ends) == 1, tool_events
    # tool_start and tool_end must share the same tool_call_id so the
    # frontend pairs them onto a single CodeExecutionToolUI block.
    assert code_starts[0]["tool_call_id"] == code_ends[0]["tool_call_id"]


def test_code_execution_failure_outcome_surfaces_in_result(monkeypatch):
    """OUTCOME_FAILED is prefixed onto the result text so the UI shows it."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "executableCode": {
                                    "language": "PYTHON",
                                    "code": "1/0",
                                }
                            },
                            {
                                "codeExecutionResult": {
                                    "outcome": "OUTCOME_FAILED",
                                    "output": "ZeroDivisionError",
                                }
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
            },
        }
    ]
    lines = _collect(monkeypatch, sse, enabled_tools = ["code_execution"])
    chunks = _parse_chunks(lines)
    tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
    result_text = next(
        (e["result"] for e in tool_events if e.get("type") == "tool_end"),
        "",
    )
    assert "OUTCOME_FAILED" in result_text
    assert "ZeroDivisionError" in result_text


def test_tool_message_recovers_name_from_tool_call_id(monkeypatch):
    """When name is omitted, recover it from the matching tool_call_id."""
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Paris"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_xyz",
            "content": json.dumps({"temp_c": 18}),
        },
    ]
    captured = _capture_body(monkeypatch, messages = messages)
    contents = captured["body"]["contents"]
    last = contents[-1]
    fr = last["parts"][0].get("functionResponse")
    assert fr is not None, last
    assert (
        fr["name"] == "get_weather"
    ), "name should fall back to the prior tool_call's function name"


# ── usage chunk surfaces promptTokenCount / candidatesTokenCount ─────


def test_usage_chunk_translates_gemini_token_counts(monkeypatch):
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "ok"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1234,
                "candidatesTokenCount": 56,
                "cachedContentTokenCount": 1000,
            },
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    usage_chunks = [c for c in chunks if c.get("choices") == [] and "usage" in c]
    assert len(usage_chunks) == 1, chunks
    usage = usage_chunks[0]["usage"]
    assert usage["prompt_tokens"] == 1234
    assert usage["completion_tokens"] == 56
    assert usage["total_tokens"] == 1290
    assert usage["prompt_tokens_details"]["cached_tokens"] == 1000


# ── multimodal: vision image -> inlineData ───────────────────────────


def test_vision_data_url_translates_to_inline_data(monkeypatch):
    fake = base64.b64encode(b"JPGBYTES").decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{fake}",
                    },
                },
            ],
        }
    ]
    captured = _capture_body(monkeypatch, messages = messages)
    parts = captured["body"]["contents"][0]["parts"]
    inline_parts = [p for p in parts if "inlineData" in p]
    assert len(inline_parts) == 1, parts
    assert inline_parts[0]["inlineData"] == {
        "mimeType": "image/jpeg",
        "data": fake,
    }


# ── finish reason mapping ────────────────────────────────────────────


@pytest.mark.parametrize(
    "gemini_reason, openai_reason",
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
    ],
)
def test_finish_reason_translation(monkeypatch, gemini_reason, openai_reason):
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "x"}],
                    },
                    "finishReason": gemini_reason,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
            },
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    finish_chunks = [
        c for c in chunks if any(ch.get("finish_reason") for ch in c.get("choices", []))
    ]
    assert any(
        ch["choices"][0]["finish_reason"] == openai_reason for ch in finish_chunks
    ), finish_chunks


# ── grounding citations surface as web_search tool_end ───────────────


def test_grounding_metadata_surfaces_as_tool_end_citations(monkeypatch):
    """`groundingMetadata.groundingChunks[].web` -> tool_end result block."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Answer with sources."}],
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://example.com/a",
                                    "title": "Example A",
                                }
                            },
                            {
                                "web": {
                                    "uri": "https://example.com/b",
                                    "title": "Example B",
                                }
                            },
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 3,
            },
        }
    ]
    lines = _collect(
        monkeypatch,
        sse,
        enabled_tools = ["web_search"],
    )
    chunks = _parse_chunks(lines)
    tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
    web_search_ends = [
        e
        for e in tool_events
        if e.get("type") == "tool_end" and e.get("tool_call_id") == "gemini_web_search"
    ]
    assert len(web_search_ends) == 1, tool_events
    result = web_search_ends[0]["result"]
    assert "https://example.com/a" in result
    assert "https://example.com/b" in result
    assert "Example A" in result
    assert "Example B" in result
