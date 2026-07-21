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

These tests pin the outbound body shape AND the inbound translation via
httpx.MockTransport (no live network). Mirrors test_anthropic_cache_ttl.py
and test_openai_image_generation.py.
"""

import asyncio
import base64
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


_active_mock_clients: list[httpx.AsyncClient] = []


def _drive(coro):
    # Fresh loop per drive so tests don't share asyncio state. Close mocked
    # clients + shutdown async-generators inside this loop so Python 3.13
    # doesn't emit `Response.aiter_*.aclose was never awaited` on GC.
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(coro)
        while _active_mock_clients:
            mc = _active_mock_clients.pop()
            loop.run_until_complete(mc.aclose())
        return result
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def _make_gemini_client(
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
) -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "gemini",
        base_url = base_url,
        api_key = "AIza-test-key",
    )


def _mock_http(monkeypatch, handler):
    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    monkeypatch.setattr(ep_mod, "_http_client", mock_client)
    # `_drive` acloses this at end of run inside the same event loop, so we
    # don't leak an unawaited aclose() coroutine.
    _active_mock_clients.append(mock_client)


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
        # Minimal valid Gemini stream so the helper completes.
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

    # Default zero is omitted, matching top_k semantics.
    captured = _capture_body(monkeypatch, presence_penalty = 0.0)
    assert "presencePenalty" not in captured["body"]["generationConfig"]


# ── thinkingConfig translation ────────────────────────────────────────


def test_gemini25_flash_thinking_disabled_sets_budget_zero(monkeypatch):
    """Gemini 2.5 Flash still uses thinkingBudget; 0 = off."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash",
        enable_thinking = False,
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc == {"thinkingBudget": 0}, tc


def test_gemini3_flash_thinking_disabled_uses_minimal_level(monkeypatch):
    """Gemini 3 Flash uses thinkingLevel; "off" maps to minimal
    (Gemini 3 cannot turn thinking fully off)."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3.5-flash",
        enable_thinking = False,
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc == {"thinkingLevel": "minimal"}, tc


def test_gemini25_pro_thinking_disabled_uses_small_budget(monkeypatch):
    """Gemini 2.5 Pro 400s on thinkingBudget=0 ("only works in thinking
    mode"); coerce to a small positive budget."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-pro",
        enable_thinking = False,
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc is not None and tc.get("thinkingBudget", 0) > 0, tc


def test_gemini3_pro_thinking_disabled_uses_low_level(monkeypatch):
    """Gemini 3 Pro uses thinkingLevel and rejects 'minimal' (Pro tier), so
    'off' coerces to 'low' (lowest the API accepts)."""
    for model in (
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.5-pro",
        "gemini-pro-latest",
    ):
        captured = _capture_body(
            monkeypatch,
            model = model,
            enable_thinking = False,
        )
        tc = captured["body"]["generationConfig"].get("thinkingConfig")
        assert tc == {"thinkingLevel": "low"}, (model, tc)


def test_gemini25_flash_effort_levels_map_to_budgets(monkeypatch):
    """Gemini 2.5 Flash retains the integer thinkingBudget ladder."""
    cases = {
        "minimal": 512,
        "low": 2048,
        "medium": 8192,
        "high": 24576,
        "max": -1,
        "xhigh": -1,
    }
    for effort, expected in cases.items():
        captured = _capture_body(
            monkeypatch,
            model = "gemini-2.5-flash",
            reasoning_effort = effort,
        )
        tc = captured["body"]["generationConfig"].get("thinkingConfig")
        assert tc == {"thinkingBudget": expected}, (effort, tc)


def test_gemini3_flash_effort_levels_map_to_thinking_level(monkeypatch):
    """Gemini 3 Flash thinkingLevel ladder: minimal/low/medium/high."""
    cases = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": "high",
    }
    for effort, expected in cases.items():
        captured = _capture_body(
            monkeypatch,
            model = "gemini-3.5-flash",
            reasoning_effort = effort,
        )
        tc = captured["body"]["generationConfig"].get("thinkingConfig")
        assert tc == {"thinkingLevel": expected}, (effort, tc)


def test_gemini3_pro_passes_medium_through(monkeypatch):
    """Gemini 3.1+ Pro accepts thinkingLevel="medium" per
    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-1-pro;
    forward as-is (medium is the documented mid-tier on Gemini 3.1)."""
    for model in (
        "gemini-3.1-pro-preview",
        "gemini-pro-latest",
    ):
        captured = _capture_body(
            monkeypatch,
            model = model,
            reasoning_effort = "medium",
        )
        tc = captured["body"]["generationConfig"].get("thinkingConfig")
        assert tc == {"thinkingLevel": "medium"}, (model, tc)


def test_gemini3_pro_minimal_effort_coerces_to_low(monkeypatch):
    """Gemini 3 Pro rejects thinkingLevel="minimal"; coerce to "low"."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3.1-pro-preview",
        reasoning_effort = "minimal",
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc == {"thinkingLevel": "low"}, tc


def test_gemini3_flash_effort_none_maps_to_minimal(monkeypatch):
    """reasoning_effort='none' on Gemini 3 Flash -> thinkingLevel=minimal."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3.5-flash",
        reasoning_effort = "none",
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc == {"thinkingLevel": "minimal"}, tc


def test_thinking_default_omits_thinking_config(monkeypatch):
    """When neither knob is supplied, thinkingConfig is omitted (Google's
    server-side default applies)."""
    captured = _capture_body(monkeypatch, model = "gemini-3.5-flash")
    gc = captured["body"]["generationConfig"]
    assert "thinkingConfig" not in gc, gc


def test_nano_banana_alias_routes_through_image_modalities(monkeypatch):
    """`nano-banana-pro-preview` aliases the Pro image model; must set
    responseModalities=[TEXT,IMAGE] when the Images pill is on
    (enabled_tools includes "image_generation")."""
    captured = _capture_body(
        monkeypatch,
        model = "nano-banana-pro-preview",
        enabled_tools = ["image_generation"],
    )
    gc = captured["body"]["generationConfig"]
    assert gc.get("responseModalities") == ["TEXT", "IMAGE"], gc


def test_image_capable_model_without_image_pill_stays_text_only(monkeypatch):
    """When the Images pill is off (no image_generation in enabled_tools), an
    image-capable model id (gemini-2.5-flash-image) must force
    responseModalities=["TEXT"]. Google's image models default to text+image
    when responseModalities is omitted, so omitting it would silently bill
    image output the UI says is disabled."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        enabled_tools = [],
    )
    gc = captured["body"]["generationConfig"]
    assert gc.get("responseModalities") == ["TEXT"], gc


def test_image_models_skip_thinking_config(monkeypatch):
    """Image-tier ids have no visible thinking knob and must NOT forward
    thinkingConfig even when stale UI state still sends `reasoning_effort` or
    `enable_thinking=False`."""
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
            enabled_tools = ["image_generation"],
        )
        gc = captured["body"]["generationConfig"]
        assert "thinkingConfig" not in gc, (model, gc)


def test_image_models_drop_code_execution(monkeypatch):
    """All image-tier ids reject `tools: [{codeExecution: {}}]`; drop
    silently. (Gemini 3 image models DO accept googleSearch -- see
    test_gemini3_image_models_allow_google_search; older ones drop
    everything.)"""
    for model in (
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "nano-banana-pro-preview",
    ):
        captured = _capture_body(
            monkeypatch,
            model = model,
            enabled_tools = ["image_generation", "code_execution"],
        )
        tools_arr = captured["body"].get("tools") or []
        names = [list(t.keys())[0] for t in tools_arr]
        assert "codeExecution" not in names, (model, tools_arr)


def test_gemini_35_pro_uses_thinking_level(monkeypatch):
    """`gemini-3.5-pro` is Gemini 3 family and uses thinkingLevel (not
    thinkingBudget). "Off" maps to "low" since Pro tier rejects "minimal"."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3.5-pro",
        enable_thinking = False,
    )
    tc = captured["body"]["generationConfig"].get("thinkingConfig")
    assert tc == {"thinkingLevel": "low"}, tc


def test_gemini3_image_models_allow_google_search(monkeypatch):
    """Google documents Search grounding on the Gemini 3 image family
    (gemini-3-pro-image-preview, gemini-3.1-flash-image-preview,
    nano-banana-pro). codeExecution stays blocked on image mode."""
    for model in (
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "nano-banana-pro-preview",
    ):
        captured = _capture_body(
            monkeypatch,
            model = model,
            enabled_tools = ["image_generation", "web_search", "code_execution"],
        )
        tools_arr = captured["body"].get("tools") or []
        names = [list(t.keys())[0] for t in tools_arr]
        assert "googleSearch" in names, (model, tools_arr)
        assert "codeExecution" not in names, (model, tools_arr)


def test_legacy_image_models_block_google_search(monkeypatch):
    """Older Gemini image ids (gemini-2.5-flash-image) still 400 on
    `tools: [{googleSearch: {}}]`; backend keeps stripping it."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        enabled_tools = ["image_generation", "web_search", "code_execution"],
    )
    assert "tools" not in captured["body"], captured["body"].get("tools")


def test_legacy_openai_base_url_normalized(monkeypatch):
    """Saved Gemini providers with the legacy `/v1beta/openai` base (from
    pre-PR OpenAI-compat plumbing) now point at the native endpoint without
    the user re-saving the connection."""
    client = ExternalProviderClient(
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai",
        api_key = "AIza-test-key",
    )
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_finish_reason_swaps_to_tool_calls_when_function_call_emitted(monkeypatch):
    """Gemini emits finishReason="STOP" even for pure functionCall turns;
    surface as `tool_calls` so OAI clients run the tool."""
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
    must echo it back as a sibling of the Gemini functionCall part."""
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
    """A Gemini functionCall part with `thoughtSignature` must surface it on
    the outbound OpenAI tool_calls delta via
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


def test_image_models_suppress_phantom_web_search_card(monkeypatch):
    """When the image guard filters googleSearch out of the request, the
    inbound stream must NOT emit web_search tool_start / tool_end (else the UI
    shows a misleading 'Search complete' card on a turn Gemini never
    searched)."""
    sse = [
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "drawn"}]},
                    "finishReason": "STOP",
                }
            ]
        }
    ]
    lines = _collect(
        monkeypatch,
        sse,
        model = "gemini-2.5-flash-image",
        enabled_tools = ["image_generation", "web_search", "code_execution"],
    )
    chunks = _parse_chunks(lines)
    tool_evs = [
        ev
        for c in chunks
        for ev in [c.get("_toolEvent")]
        if isinstance(ev, dict) and ev.get("tool_name") == "web_search"
    ]
    assert tool_evs == [], tool_evs


def test_image_generation_tool_on_image_model_drops_text_tools(monkeypatch):
    """`enabled_tools=["image_generation", "web_search", "code_execution"]`
    on a Gemini IMAGE model flips responseModalities to TEXT+IMAGE; in that
    mode codeExecution must NOT be forwarded (Gemini rejects text code tools
    alongside image responseModalities). Older image families also drop
    googleSearch."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        enabled_tools = [
            "image_generation",
            "web_search",
            "code_execution",
        ],
    )
    assert "tools" not in captured["body"], captured["body"]
    assert captured["body"]["generationConfig"].get("responseModalities") == [
        "TEXT",
        "IMAGE",
    ]


def test_prompt_feedback_block_reason_surfaces_as_error(monkeypatch):
    """`promptFeedback.blockReason` with zero candidates must produce an error
    chunk, not a silent empty assistant reply."""
    sse = [
        {
            "promptFeedback": {"blockReason": "SAFETY"},
        }
    ]
    chunks = _parse_chunks(_collect(monkeypatch, sse))
    error_chunks = [c for c in chunks if "error" in c]
    assert error_chunks, chunks
    assert "SAFETY" in (
        error_chunks[0].get("error", {}).get("message") or ""
    ), error_chunks


def test_usage_chunk_includes_thoughts_tokens(monkeypatch):
    """`thoughtsTokenCount` is the hidden-reasoning slice of output; roll it
    into `output_tokens` AND surface it on
    `output_tokens_details.reasoning_tokens` so total_tokens reflects the full
    billable spend."""
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
    """Unsloth's existing True/False signals shouldn't fabricate a cache id."""
    captured = _capture_body(monkeypatch, enable_prompt_caching = True)
    assert "cachedContent" not in captured["body"]


# ── image generation: request modalities + response translation ──────


def test_image_model_sets_response_modalities(monkeypatch):
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        enabled_tools = ["image_generation"],
    )
    assert captured["body"]["generationConfig"]["responseModalities"] == [
        "TEXT",
        "IMAGE",
    ]


def test_image_generation_tool_sets_response_modalities_on_image_model(monkeypatch):
    """`enabled_tools=["image_generation"]` flips responseModalities
    only when the selected model is image-capable; otherwise the
    request stays plain text (text-only models 400 on
    responseModalities)."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
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
    # Last turn must be a functionResponse part (Gemini wraps it as a role=user
    # turn carrying the result).
    last = contents[-1]
    assert last["role"] == "user", last
    fr = last["parts"][0].get("functionResponse")
    assert fr is not None, last
    assert fr["name"] == "get_weather"
    assert fr["response"] == {"temp_c": 18, "summary": "Sunny"}
    # And the assistant turn carries the original functionCall so the model
    # sees the round-trip context.
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
    tool_calls[*].index. Hardcoding index=0 collapses parallel calls onto one
    slot in OpenAI-style reassemblers."""
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
    # tool_start and tool_end must share a tool_call_id so the frontend pairs
    # them onto one CodeExecutionToolUI block.
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
    assert inline_parts[0]["inlineData"] == {"mimeType": "image/jpeg", "data": fake}


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


# ── round 3 review follow-ups ─────────────────────────────────────────


def test_custom_gemini_proxy_base_url_not_rewritten():
    """Only the Google-hosted /v1beta/openai base is normalized; a custom
    gateway whose path ends in /openai must be left alone."""
    client = ExternalProviderClient(
        provider_type = "gemini",
        base_url = "https://proxy.example.com/team/openai",
        api_key = "AIza-test-key",
    )
    assert client.base_url == "https://proxy.example.com/team/openai"


def test_custom_gemini_proxy_uses_openai_dispatch():
    """Any non-Google Gemini base (LiteLLM, custom OpenAI-compat routers) must
    route through the OpenAI-compatible forwarder, not the native translator.
    Auth uses Authorization: Bearer ..., not x-goog-api-key."""
    for base in (
        "https://proxy.example.com/team/openai",
        "https://proxy.example.com/v1",
        "https://litellm.internal.example/v1",
    ):
        client = ExternalProviderClient(
            provider_type = "gemini",
            base_url = base,
            api_key = "AIza-test-key",
        )
        assert client._is_openai_compatible() is True, base
        headers = client._auth_headers()
        assert "x-goog-api-key" not in {k.lower() for k in headers}, (base, headers)
        assert headers["Authorization"] == "Bearer AIza-test-key", (base, headers)


def test_google_hosted_gemini_still_uses_native_dispatch():
    """Google-hosted Gemini keeps native dispatch + x-goog-api-key auth."""
    client = ExternalProviderClient(
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta",
        api_key = "AIza-test-key",
    )
    assert client._is_openai_compatible() is False
    headers = client._auth_headers()
    assert headers.get("x-goog-api-key") == "AIza-test-key", headers


def test_invalid_gemini_model_id_rejected_before_request(monkeypatch):
    """Path-traversal model ids must be rejected before the URL is
    interpolated, so the configured API key isn't sent to unintended Gemini
    endpoints."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content = _gemini_sse([]),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    out: list[str] = []

    async def run():
        client = _make_gemini_client()
        async for line in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "../cachedContents/leak",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
        ):
            out.append(line)
        await client.close()

    _drive(run())
    # No outbound request should have been issued.
    assert captured == [], captured
    error_lines = [line for line in out if '"error"' in line]
    assert error_lines, out


def test_top_k_omitted_when_not_explicit_default_for_gemini(monkeypatch):
    """top_k=None means "use provider default"; helper must not emit `topK` in
    generationConfig when the caller didn't pass it."""
    captured = _capture_body(monkeypatch, top_k = None)
    assert "topK" not in captured["body"]["generationConfig"], captured["body"]


def test_text_model_image_generation_tool_silently_dropped(monkeypatch):
    """A stale `enabled_tools=["image_generation"]` on a text-only Gemini
    model (e.g. gemini-2.5-flash) must NOT switch the request into image mode
    -- Google's API 400s on responseModalities for text models."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash",
        enabled_tools = ["image_generation"],
    )
    gc = captured["body"]["generationConfig"]
    assert "responseModalities" not in gc, gc


def test_empty_text_part_with_thought_signature_emits_extra_content(monkeypatch):
    """Gemini 3 can ship a content-free fragment whose only payload is
    `thoughtSignature`. The translator must still surface it on a
    delta.extra_content envelope so the next turn can replay it."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "answer"},
                            {"thoughtSignature": "SIG-FINAL"},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 1,
            },
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    extra_carriers = [
        c
        for c in chunks
        if c.get("choices")
        and c["choices"][0]["delta"].get("extra_content")
        == {"google": {"thought_signature": "SIG-FINAL"}}
    ]
    assert extra_carriers, chunks


def test_enable_prompt_caching_false_string_coerces_to_bool():
    """Pre-PR the field was Optional[bool]; widening to Union[bool,str] must
    preserve historical coercion so callers sending `"false"` still opt out of
    caching."""
    from models.inference import ChatCompletionRequest

    msg = {"role": "user", "content": "hi"}
    req = ChatCompletionRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [msg],
            "enable_prompt_caching": "false",
        }
    )
    assert req.enable_prompt_caching is False, req.enable_prompt_caching

    req = ChatCompletionRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [msg],
            "enable_prompt_caching": "true",
        }
    )
    assert req.enable_prompt_caching is True

    # An actual cache resource name passes through untouched.
    req = ChatCompletionRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [msg],
            "enable_prompt_caching": "cachedContents/abc123",
        }
    )
    assert req.enable_prompt_caching == "cachedContents/abc123"


def test_legacy_google_openai_base_url_is_rewritten():
    """The Google-hosted /v1beta/openai legacy base IS still rewritten."""
    client = ExternalProviderClient(
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai",
        api_key = "AIza-test-key",
    )
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_remote_image_url_downloads_and_inlines_as_base64(monkeypatch):
    """Round 14: arbitrary public HTTPS image URLs cannot be sent as Gemini
    fileData (reserved for Files API URIs and YouTube). The translator must
    fetch the bytes server-side and inline them as base64 inlineData."""
    image_bytes = b"FAKEPNGBYTES"

    async def fake_fetch(
        url,
        fallback_mime,
        max_bytes = None,
    ):
        assert url == "https://cdn.example.com/diagram.png"
        return ("image/png", base64.b64encode(image_bytes).decode("ascii"))

    monkeypatch.setattr(ep_mod, "_safe_fetch_image_for_gemini", fake_fetch)
    captured = _capture_body(
        monkeypatch,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://cdn.example.com/diagram.png",
                        },
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["contents"][-1]["parts"]
    inline = next((p for p in parts if "inlineData" in p), None)
    assert inline is not None, parts
    assert inline["inlineData"]["mimeType"] == "image/png"
    assert inline["inlineData"]["data"] == base64.b64encode(image_bytes).decode()
    assert not any("fileData" in p for p in parts), parts


def test_remote_image_url_dropped_when_fetch_returns_none(monkeypatch):
    """Round 15: if the SSRF guard rejects the URL (private host, non-https,
    oversize, non-image), the helper returns None and the image part is
    silently dropped, not forwarded as raw bytes or a fileData fallback."""

    async def fake_fetch_reject(
        url,
        fallback_mime,
        max_bytes = None,
    ):
        return None

    monkeypatch.setattr(ep_mod, "_safe_fetch_image_for_gemini", fake_fetch_reject)
    captured = _capture_body(
        monkeypatch,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://10.0.0.5/private.png"},
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["contents"][-1]["parts"]
    assert not any("inlineData" in p for p in parts), parts
    assert not any("fileData" in p for p in parts), parts


def test_safe_fetch_image_rejects_non_https():
    """SSRF guard: only https URLs may be fetched."""
    res = asyncio.new_event_loop().run_until_complete(
        ep_mod._safe_fetch_image_for_gemini("http://cdn.example.com/x.png", "image/png")
    )
    assert res is None


def test_safe_fetch_image_rejects_loopback_ip_literal():
    """SSRF guard: refuse loopback / private IP literals before any network
    call."""
    for url in (
        "https://127.0.0.1/x.png",
        "https://[::1]/x.png",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.5/x.png",
        "https://192.168.1.1/x.png",
    ):
        res = asyncio.new_event_loop().run_until_complete(
            ep_mod._safe_fetch_image_for_gemini(url, "image/png")
        )
        assert res is None, url


def test_safe_fetch_image_rejects_resolved_private_host(monkeypatch):
    """SSRF guard: if a hostname resolves to a private IP, refuse."""
    import socket

    def fake_getaddrinfo(host, *_args, **_kwargs):
        return [(socket.AF_INET, None, None, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    res = asyncio.new_event_loop().run_until_complete(
        ep_mod._safe_fetch_image_for_gemini(
            "https://internal.example/x.png", "image/png"
        )
    )
    assert res is None


def test_youtube_and_files_api_uris_stay_as_file_data(monkeypatch):
    """Round 14: YouTube URLs and generativelanguage.googleapis.com Files API
    URIs are the documented `fileData.fileUri` paths and must NOT be
    downloaded; arbitrary public URLs do get fetched."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
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

    async def run():
        client = _make_gemini_client()
        async for _ in client.stream_chat_completion(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "explain"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://www.youtube.com/watch?v=abc123",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://generativelanguage.googleapis.com/v1beta/files/abc",
                            },
                        },
                    ],
                }
            ],
            model = "gemini-2.5-flash",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
        ):
            pass
        await client.close()

    _drive(run())
    parts = captured["body"]["contents"][-1]["parts"]
    file_uris = [p["fileData"]["fileUri"] for p in parts if "fileData" in p]
    assert "https://www.youtube.com/watch?v=abc123" in file_uris, parts
    assert (
        "https://generativelanguage.googleapis.com/v1beta/files/abc" in file_uris
    ), parts


def test_tool_use_prompt_tokens_added_to_input_tokens(monkeypatch):
    """`toolUsePromptTokenCount` must roll into the OpenAI prompt total --
    else tool turns silently undercount input tokens."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "result"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "toolUsePromptTokenCount": 100,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 2,
            },
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == 1, chunks
    usage = usage_chunks[0]["usage"]
    assert usage["prompt_tokens"] == 110, usage
    assert usage["completion_tokens"] == 7, usage
    assert usage["total_tokens"] == 117, usage
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 2, usage


def test_usage_chunk_reasoning_tokens_surfaced(monkeypatch):
    """thoughtsTokenCount must surface as
    completion_tokens_details.reasoning_tokens in the emitted OpenAI usage
    chunk."""
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
                "promptTokenCount": 8,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 20,
            },
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == 1, chunks
    usage = usage_chunks[0]["usage"]
    assert usage["completion_tokens"] == 25, usage
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 20, usage


def test_prompt_block_pairs_web_search_tool_end(monkeypatch):
    """When `promptFeedback.blockReason` triggers after the synthetic
    web_search tool_start, the helper must emit a matching tool_end so the UI
    doesn't leave a "searching..." spinner stuck on screen."""
    sse = [
        {"promptFeedback": {"blockReason": "SAFETY"}},
    ]
    lines = _collect(
        monkeypatch,
        sse,
        enabled_tools = ["web_search"],
    )
    chunks = _parse_chunks(lines)
    tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
    starts = [e for e in tool_events if e.get("type") == "tool_start"]
    ends = [e for e in tool_events if e.get("type") == "tool_end"]
    assert len(starts) == 1, tool_events
    assert len(ends) == 1, tool_events
    assert ends[0]["tool_call_id"] == "gemini_web_search"
    assert "aborted" in ends[0]["result"]
    error_chunks = [c for c in chunks if c.get("error")]
    assert error_chunks, chunks


def test_code_execution_tool_events_stow_native_part(monkeypatch):
    """executableCode / codeExecutionResult must round-trip native ids and
    thoughtSignature in google.native_part so follow-up turns can replay
    Gemini's required history shape."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "executableCode": {
                                    "id": "code_a",
                                    "language": "PYTHON",
                                    "code": "print(1+1)",
                                },
                                "thoughtSignature": "SIG-CODE",
                            },
                            {
                                "codeExecutionResult": {
                                    "id": "result_a",
                                    "outcome": "OUTCOME_OK",
                                    "output": "2\n",
                                },
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 4,
            },
        }
    ]
    lines = _collect(
        monkeypatch,
        sse,
        enabled_tools = ["code_execution"],
    )
    chunks = _parse_chunks(lines)
    tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
    starts = [e for e in tool_events if e.get("type") == "tool_start"]
    ends = [e for e in tool_events if e.get("type") == "tool_end"]
    code_start = next(
        (e for e in starts if e.get("tool_name") == "code_execution"),
        None,
    )
    code_end = next(iter(ends), None)
    assert code_start is not None, starts
    assert code_start["tool_call_id"] == "code_a", code_start
    native = code_start["arguments"]["google"]["native_part"]
    # Round 21: native_part uses an ordered `parts` list so per-part
    # `thoughtSignature` survives a frontend merge of executableCode +
    # codeExecutionResult into one tool-call card.
    start_parts = native["parts"]
    assert start_parts[0]["executableCode"]["id"] == "code_a"
    assert start_parts[0]["thoughtSignature"] == "SIG-CODE"
    assert code_end is not None, ends
    assert code_end["tool_call_id"] == "code_a", code_end
    native_end = code_end["google"]["native_part"]
    end_parts = native_end["parts"]
    assert end_parts[0]["codeExecutionResult"]["id"] == "result_a"


def test_inline_image_tool_end_carries_thought_signature(monkeypatch):
    """Inline image parts with thoughtSignature must persist it on the emitted
    tool_end so Gemini 3 image editing can echo it back."""
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
                                    "data": base64.b64encode(b"PNG").decode(),
                                },
                                "thoughtSignature": "SIG-IMG",
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 1,
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
    image_ends = [
        e for e in tool_events if e.get("type") == "tool_end" and e.get("image_b64")
    ]
    assert image_ends, tool_events
    assert image_ends[0]["google"]["thought_signature"] == "SIG-IMG"
    # Multi-turn image edit must replay the original inlineData part with its
    # thoughtSignature; the outbound translator reads
    # google.native_part.parts[].inlineData, so stow it on the tool_end too.
    # Round 21 made native_part an ordered parts list so a per-part signature
    # stays attached to inlineData only.
    native = image_ends[0]["google"]["native_part"]
    image_parts = native["parts"]
    assert image_parts[0]["inlineData"]["mimeType"] == "image/png"
    assert image_parts[0]["inlineData"]["data"] == base64.b64encode(b"PNG").decode()
    assert image_parts[0]["thoughtSignature"] == "SIG-IMG"


def test_code_execution_plot_attaches_inline_image_native_part(monkeypatch):
    """A code_execution turn that returns a matplotlib plot must stow the
    plot's inlineData on the secondary tool_end so the follow-up turn can
    replay the image alongside executableCode and codeExecutionResult."""
    plot_data = base64.b64encode(b"PLOT").decode()
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "executableCode": {
                                    "id": "code_a",
                                    "language": "PYTHON",
                                    "code": "plt.plot([0,1])",
                                },
                            },
                            {
                                "codeExecutionResult": {
                                    "id": "result_a",
                                    "outcome": "OUTCOME_OK",
                                    "output": "",
                                },
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": plot_data,
                                },
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 4,
            },
        }
    ]
    lines = _collect(
        monkeypatch,
        sse,
        enabled_tools = ["code_execution"],
    )
    chunks = _parse_chunks(lines)
    tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
    code_ends = [
        e
        for e in tool_events
        if e.get("type") == "tool_end" and e.get("tool_call_id") == "code_a"
    ]
    # Two tool_end events on the same id: one for codeExecutionResult, one
    # merging in the inlineData plot. The plot one must carry the native
    # inlineData under google.native_part so the frontend tool_end merge union
    # joins it with the prior executableCode and codeExecutionResult parts on
    # the same card.
    assert len(code_ends) == 2, code_ends
    image_end = next(
        (e for e in code_ends if "__IMAGES__:" in (e.get("result") or "")),
        None,
    )
    assert image_end is not None, code_ends
    native = image_end["google"]["native_part"]
    plot_parts = native["parts"]
    assert plot_parts[0]["inlineData"]["mimeType"] == "image/png"
    assert plot_parts[0]["inlineData"]["data"] == plot_data


def test_text_chunk_carries_thought_signature(monkeypatch):
    """Text parts with thoughtSignature surface it on delta.extra_content so
    frontend persistence can replay it on the follow-up turn."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": "hello",
                                "thoughtSignature": "SIG-TEXT",
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 1,
            },
        }
    ]
    lines = _collect(monkeypatch, sse)
    chunks = _parse_chunks(lines)
    text_chunks = [
        c
        for c in chunks
        if c.get("choices") and c["choices"][0]["delta"].get("content") == "hello"
    ]
    assert text_chunks, chunks
    extra = text_chunks[0]["choices"][0]["delta"].get("extra_content")
    assert extra == {"google": {"thought_signature": "SIG-TEXT"}}, text_chunks


def test_openai_tools_translated_into_function_declarations(monkeypatch):
    """Standard ChatCompletionRequest.tools must be forwarded into Gemini's
    tools[].functionDeclarations envelope."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice = {"type": "function", "function": {"name": "get_weather"}},
    )
    tools_arr = captured["body"].get("tools") or []
    fn_decls = [t for t in tools_arr if "functionDeclarations" in t]
    assert fn_decls, captured["body"]
    decls = fn_decls[0]["functionDeclarations"]
    assert decls[0]["name"] == "get_weather"
    assert decls[0]["parameters"]["properties"]["city"]["type"] == "string"
    tool_config = captured["body"].get("toolConfig")
    assert tool_config is not None, captured["body"]
    fcc = tool_config["functionCallingConfig"]
    assert fcc["mode"] == "ANY"
    assert fcc["allowedFunctionNames"] == ["get_weather"]


def test_tool_choice_auto_maps_to_function_calling_mode_auto(monkeypatch):
    """tool_choice="auto" maps to toolConfig.functionCallingConfig.mode."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {"name": "noop", "parameters": {"type": "object"}},
            }
        ],
        tool_choice = "auto",
    )
    fcc = captured["body"]["toolConfig"]["functionCallingConfig"]
    assert fcc["mode"] == "AUTO"
    assert "allowedFunctionNames" not in fcc


def test_code_exec_inline_image_attaches_to_code_execution_card(monkeypatch):
    """A codeExecution sandbox plot (matplotlib) ships as an inline image part
    right after the codeExecutionResult. Instead of a separate empty
    image_generation card, attach to the same code_execution tool_end via the
    `__IMAGES__:` marker the chat adapter already understands."""
    sse = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "executableCode": {
                                    "id": "code_plot",
                                    "language": "PYTHON",
                                    "code": "import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.savefig('out.png')",
                                },
                            },
                            {
                                "codeExecutionResult": {
                                    "outcome": "OUTCOME_OK",
                                    "output": "saved",
                                },
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"PNGDATA").decode(),
                                },
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 4,
            },
        }
    ]
    lines = _collect(
        monkeypatch,
        sse,
        enabled_tools = ["code_execution"],
    )
    chunks = _parse_chunks(lines)
    tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
    # No standalone image_generation card should have been emitted.
    image_starts = [
        e
        for e in tool_events
        if e.get("type") == "tool_start" and e.get("tool_name") == "image_generation"
    ]
    assert not image_starts, tool_events
    # The code_execution tool_end should now carry the inline image
    # via the `__IMAGES__:` marker.
    code_ends = [
        e
        for e in tool_events
        if e.get("type") == "tool_end" and e.get("tool_call_id") == "code_plot"
    ]
    assert code_ends, tool_events
    final_result = code_ends[-1]["result"]
    assert "__IMAGES__:" in final_result, code_ends
    assert "data:image/png;base64," in final_result, code_ends


def test_code_execution_tool_call_replays_native_executable_code(monkeypatch):
    """An assistant tool_call with toolName=code_execution and
    extra_content.google.native_part holding the originally-emitted
    `executableCode` + `codeExecutionResult` must round-trip as native Gemini
    parts (not a generic functionCall) on the next turn."""
    captured = _capture_body(
        monkeypatch,
        messages = [
            {"role": "user", "content": "compute 2+2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "code_a",
                        "type": "function",
                        "function": {
                            "name": "code_execution",
                            "arguments": "{}",
                        },
                        "extra_content": {
                            "google": {
                                "native_part": {
                                    "executableCode": {
                                        "id": "code_a",
                                        "language": "PYTHON",
                                        "code": "print(2+2)",
                                    },
                                    "codeExecutionResult": {
                                        "outcome": "OUTCOME_OK",
                                        "output": "4\n",
                                    },
                                    "thoughtSignature": "SIG-CODE",
                                },
                            },
                        },
                    },
                ],
            },
            {"role": "user", "content": "what was that result"},
        ],
    )
    assistant_turn = captured["body"]["contents"][1]
    assert assistant_turn["role"] == "model"
    parts = assistant_turn["parts"]
    native_keys = [list(p.keys())[0] for p in parts if isinstance(p, dict)]
    assert "executableCode" in native_keys, parts
    assert "codeExecutionResult" in native_keys, parts
    assert not any(
        "functionCall" in p
        and (p["functionCall"] or {}).get("name") == "code_execution"
        for p in parts
    ), parts
    exec_part = next(p for p in parts if "executableCode" in p)
    assert exec_part.get("thoughtSignature") == "SIG-CODE", exec_part


def test_image_generation_tool_call_replays_native_inline_data(monkeypatch):
    """An assistant tool_call with toolName=image_generation and
    extra_content.google.native_part.inlineData must replay the prior image as
    a native Gemini inlineData part (not a generic functionCall) so multi-turn
    image editing keeps the image context."""
    pixel = base64.b64encode(b"PNG").decode()
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        messages = [
            {"role": "user", "content": "make a circle"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "img_a",
                        "type": "function",
                        "function": {
                            "name": "image_generation",
                            "arguments": "{}",
                        },
                        "extra_content": {
                            "google": {
                                "native_part": {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": pixel,
                                    },
                                    "thoughtSignature": "SIG-IMG",
                                },
                            },
                        },
                    },
                ],
            },
            {"role": "user", "content": "now make it blue"},
        ],
    )
    assistant_turn = captured["body"]["contents"][1]
    assert assistant_turn["role"] == "model"
    parts = assistant_turn["parts"]
    inline_parts = [p for p in parts if "inlineData" in p]
    assert inline_parts, parts
    assert inline_parts[0]["inlineData"]["mimeType"] == "image/png"
    assert inline_parts[0]["inlineData"]["data"] == pixel
    assert inline_parts[0].get("thoughtSignature") == "SIG-IMG", inline_parts
    assert not any(
        "functionCall" in p
        and (p["functionCall"] or {}).get("name") == "image_generation"
        for p in parts
    ), parts


def test_assistant_text_thought_signature_replays_on_outbound_text_part(monkeypatch):
    """Assistant text with extra_content.google.thought_signature must attach
    `thoughtSignature` to the LAST text part of the replayed Gemini history.
    Gemini 3 strict function-calling rejects history that drops returned
    signatures, so the frontend stows the latest signed-text signature and the
    backend pins it on the next turn."""
    captured = _capture_body(
        monkeypatch,
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello"},
                ],
                "extra_content": {
                    "google": {"thought_signature": "SIG-TEXT"},
                },
            },
            {"role": "user", "content": "again"},
        ],
    )
    assistant_turn = captured["body"]["contents"][1]
    assert assistant_turn["role"] == "model"
    parts = assistant_turn["parts"]
    text_parts = [p for p in parts if "text" in p]
    assert text_parts, parts
    assert text_parts[-1].get("thoughtSignature") == "SIG-TEXT", text_parts


def test_function_declarations_strip_openai_only_schema_keys(monkeypatch):
    """OpenAI strict tools commonly include `additionalProperties`, `$schema`,
    `$defs`, `strict`, etc. Gemini's Schema rejects those with
    INVALID_ARGUMENT, so the translator must strip them while keeping
    properties.<field>.type intact."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "additionalProperties": False,
                        "strict": True,
                        "properties": {
                            "key": {
                                "type": "string",
                                "additionalProperties": False,
                            },
                        },
                        "required": ["key"],
                    },
                },
            }
        ],
    )
    tools_arr = captured["body"].get("tools") or []
    decls = next(
        (
            t.get("functionDeclarations")
            for t in tools_arr
            if "functionDeclarations" in t
        ),
        None,
    )
    assert decls is not None, captured["body"]
    params = decls[0]["parameters"]
    assert "additionalProperties" not in params
    assert "$schema" not in params
    assert "strict" not in params
    assert params["type"] == "object"
    assert params["properties"]["key"]["type"] == "string"
    assert "additionalProperties" not in params["properties"]["key"]
    assert params["required"] == ["key"]


def test_function_declarations_inline_local_refs_into_gemini_schema(monkeypatch):
    """Round 25: Pydantic-generated tool schemas hoist nested object shapes
    into `$defs` and reference them with `{"$ref": "#/$defs/..."}`. Gemini's
    OpenAPI subset has no $ref, so a naive allowlist sanitizer drops the
    reference and reduces the nested property to `{}`, losing its type, fields,
    and required keys. The sanitizer must resolve local `#/...` pointers and
    inline the referenced schema."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_user",
                    "description": "Persist a user.",
                    "parameters": {
                        "type": "object",
                        "$defs": {
                            "Address": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "zip": {"type": "string"},
                                },
                                "required": ["street", "zip"],
                            },
                        },
                        "properties": {
                            "name": {"type": "string"},
                            "address": {"$ref": "#/$defs/Address"},
                        },
                        "required": ["name", "address"],
                    },
                },
            }
        ],
    )
    tools_arr = captured["body"].get("tools") or []
    decls = next(
        (
            t.get("functionDeclarations")
            for t in tools_arr
            if "functionDeclarations" in t
        ),
        None,
    )
    assert decls is not None, captured["body"]
    params = decls[0]["parameters"]
    assert "$defs" not in params
    address = params["properties"]["address"]
    assert address.get("type") == "object", address
    assert address.get("properties", {}).get("street", {}).get("type") == "string"
    assert address.get("properties", {}).get("zip", {}).get("type") == "string"
    assert address.get("required") == ["street", "zip"]


def test_function_declarations_inline_local_refs_in_anyof_and_items(monkeypatch):
    """The recursive inliner must reach through `anyOf` branches and `items`
    (array element schemas), not just top-level property refs."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bulk_set",
                    "parameters": {
                        "type": "object",
                        "$defs": {
                            "Address": {
                                "type": "object",
                                "properties": {"zip": {"type": "string"}},
                                "required": ["zip"],
                            },
                        },
                        "properties": {
                            "primary": {
                                "anyOf": [
                                    {"$ref": "#/$defs/Address"},
                                    {"type": "null"},
                                ],
                            },
                            "extras": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/Address"},
                            },
                        },
                    },
                },
            }
        ],
    )
    tools_arr = captured["body"].get("tools") or []
    decls = next(
        (
            t.get("functionDeclarations")
            for t in tools_arr
            if "functionDeclarations" in t
        ),
        None,
    )
    assert decls is not None
    params = decls[0]["parameters"]
    primary = params["properties"]["primary"]
    # anyOf with single non-null branch + null collapses to inline +
    # nullable: true; the inlined branch must contain the resolved Address
    # shape.
    assert primary.get("nullable") is True
    assert primary.get("type") == "object"
    assert primary.get("properties", {}).get("zip", {}).get("type") == "string"
    extras = params["properties"]["extras"]
    assert extras.get("type") == "array"
    assert extras.get("items", {}).get("type") == "object"
    assert (
        extras.get("items", {}).get("properties", {}).get("zip", {}).get("type")
        == "string"
    )


def test_function_declarations_self_referential_schema_terminates(monkeypatch):
    """Self-referential / cyclic JSON Schemas (a `Node` with `children:
    [Node]`) must not infinite-loop. The inliner tracks the set of refs in
    flight and short-circuits to `{}` on a cycle."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_tree",
                    "parameters": {
                        "type": "object",
                        "$defs": {
                            "Node": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                    "children": {
                                        "type": "array",
                                        "items": {"$ref": "#/$defs/Node"},
                                    },
                                },
                            },
                        },
                        "properties": {
                            "root": {"$ref": "#/$defs/Node"},
                        },
                    },
                },
            }
        ],
    )
    tools_arr = captured["body"].get("tools") or []
    decls = next(
        (
            t.get("functionDeclarations")
            for t in tools_arr
            if "functionDeclarations" in t
        ),
        None,
    )
    assert decls is not None
    root = decls[0]["parameters"]["properties"]["root"]
    assert root.get("type") == "object"
    assert root.get("properties", {}).get("value", {}).get("type") == "string"


def test_gemini_native_skips_orphan_function_response_for_dropped_builtin(monkeypatch):
    """Round 26: when the assistant-side synthetic web_search/web_fetch
    tool_call is dropped from native Gemini history, the matching role="tool"
    follow-up must also be dropped. Otherwise the outbound body carries an
    orphan functionResponse with no preceding functionCall, which 400s the
    Gemini turn."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    req = ChatCompletionRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": "search please"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_s",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": ('{"_server_tool": true, "query": "x"}'),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_s",
                    "content": "[search result]",
                },
                {"role": "user", "content": "again"},
            ],
            "max_tokens": 64,
            "stream": True,
        }
    )
    built = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta",
    )
    captured = _capture_body(monkeypatch, messages = built)
    contents = captured["body"].get("contents") or []
    for entry in contents:
        for part in entry.get("parts", []):
            fr = part.get("functionResponse")
            if isinstance(fr, dict):
                assert fr.get("name") != "web_search", contents


def test_gemini_native_skips_orphan_function_response_for_native_part_replay(
    monkeypatch,
):
    """Round 26: code_execution / image_generation tool_calls are replayed as
    Gemini-native executableCode / codeExecutionResult / inlineData parts. The
    matching role="tool" follow-up must NOT then be emitted as a
    functionResponse named code_execution -- there is no declared user
    function with that name, and Gemini's history rules already attribute the
    result to the native parts above."""
    captured = _capture_body(
        monkeypatch,
        messages = [
            {"role": "user", "content": "plot something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "code_execution",
                            "arguments": "{}",
                        },
                        "extra_content": {
                            "google": {
                                "native_part": {
                                    "parts": [
                                        {
                                            "executableCode": {
                                                "language": "PYTHON",
                                                "code": "print(2)",
                                            }
                                        },
                                        {
                                            "codeExecutionResult": {
                                                "outcome": "OUTCOME_OK",
                                                "output": "2\n",
                                            }
                                        },
                                    ]
                                }
                            }
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_a",
                "name": "code_execution",
                "content": "2",
            },
            {"role": "user", "content": "next"},
        ],
    )
    contents = captured["body"].get("contents") or []
    saw_native = False
    for entry in contents:
        for part in entry.get("parts", []):
            if "executableCode" in part or "codeExecutionResult" in part:
                saw_native = True
            fr = part.get("functionResponse")
            if isinstance(fr, dict):
                assert fr.get("name") != "code_execution", contents
    assert saw_native, contents


def test_gemini_native_part_falls_back_to_args_google(monkeypatch):
    """Round 27: a direct OpenAI-compat API caller (or imported third-party
    thread) cannot use Unsloth's non-standard `tool_calls[].extra_content`
    field, so the native_part payload round-trips through `function.arguments`
    as `{"google": {"native_part": {...}}}`. The synthetic-builtin detector
    recognizes that location, but the replay branch was only reading from
    `tc.extra_content.google.native_part`. Result: the round-25 guard saw a
    synthetic builtin with no _native_part and dropped the entire assistant
    turn, losing the prior code/image context. The translator must fall back
    to args.google.native_part and still emit the native executableCode /
    inlineData parts."""
    import json as _json

    captured = _capture_body(
        monkeypatch,
        messages = [
            {"role": "user", "content": "draw a cat"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_img",
                        "type": "function",
                        "function": {
                            "name": "image_generation",
                            "arguments": _json.dumps(
                                {
                                    "google": {
                                        "native_part": {
                                            "parts": [
                                                {
                                                    "inlineData": {
                                                        "mimeType": "image/png",
                                                        "data": "AAAA",
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                }
                            ),
                        },
                    }
                ],
            },
            {"role": "user", "content": "now make it a dog"},
        ],
    )
    contents = captured["body"].get("contents") or []
    saw_inline = False
    for entry in contents:
        for part in entry.get("parts", []):
            if "inlineData" in part:
                saw_inline = True
    assert saw_inline, contents


def test_gemini_native_skips_synthetic_server_builtin_replay(monkeypatch):
    """Round 25: Marked server-side builtin tool_calls (web_search /
    web_fetch with `_server_tool` or `args.google.native_part`) must not fall
    through to the generic Gemini `functionCall` replay path when no replayable
    native part exists. Without this guard the outbound body contains a fake
    `functionCall` whose name isn't a declared user function, and the Gemini
    turn 400s."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    req = ChatCompletionRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": "search please"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_s",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": ('{"_server_tool": true, "query": "x"}'),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_s",
                    "content": "[search result]",
                },
                {"role": "user", "content": "again"},
            ],
            "max_tokens": 64,
            "stream": True,
        }
    )
    built = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta",
    )
    captured = _capture_body(monkeypatch, messages = built)
    contents = captured["body"].get("contents") or []
    for entry in contents:
        for part in entry.get("parts", []):
            fc = part.get("functionCall")
            if isinstance(fc, dict):
                assert fc.get("name") != "web_search", contents


def test_chat_message_extra_content_round_trips_through_validation():
    """Round 9: ChatMessage was missing `extra_content`, so Pydantic discarded
    it during request validation and the text-part signature replay path read
    nothing. The field must survive model_validate and pass through
    _build_external_messages."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    req = ChatCompletionRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hello"},
                    ],
                    "extra_content": {
                        "google": {"thought_signature": "SIG-TEXT"},
                    },
                },
                {"role": "user", "content": "again"},
            ],
            "max_tokens": 64,
            "stream": True,
        }
    )
    assistant_msg = req.messages[1]
    assert assistant_msg.extra_content == {"google": {"thought_signature": "SIG-TEXT"}}
    built = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta",
    )
    assistant_out = built[1]
    assert assistant_out["extra_content"] == {
        "google": {"thought_signature": "SIG-TEXT"}
    }
    # Non-Gemini providers must NOT receive extra_content; Google's
    # thought_signature is unknown to OpenAI / Mistral / etc.
    built_openai = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "openai",
    )
    assert "extra_content" not in built_openai[1], built_openai[1]
    # Custom non-Google Gemini bases (LiteLLM / OAI-compat gateways) also must
    # not receive Gemini-only extra_content -- the backend dispatches them
    # through /chat/completions.
    built_custom = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "gemini",
        base_url = "https://litellm.example/v1",
    )
    assert "extra_content" not in built_custom[1], built_custom[1]


def test_parallel_tool_results_group_into_one_user_block(monkeypatch):
    """Round 14: Gemini docs group parallel functionResponses in a single
    subsequent user content with multiple functionResponse parts. Consecutive
    OpenAI role="tool" messages must merge into one Gemini user block, not
    split into separate user turns."""
    captured = _capture_body(
        monkeypatch,
        messages = [
            {"role": "user", "content": "compute"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"x":1}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "mul", "arguments": '{"x":2}'},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_a",
                "name": "add",
                "content": "2",
            },
            {
                "role": "tool",
                "tool_call_id": "call_b",
                "name": "mul",
                "content": "4",
            },
        ],
    )
    contents = captured["body"]["contents"]
    # Initial user, model with two functionCalls, ONE user with two
    # functionResponses.
    tool_result_users = [
        c
        for c in contents
        if c.get("role") == "user"
        and all(
            isinstance(p, dict) and "functionResponse" in p
            for p in (c.get("parts") or [])
        )
    ]
    assert len(tool_result_users) == 1, contents
    fr_parts = tool_result_users[0]["parts"]
    assert len(fr_parts) == 2, fr_parts
    names = [p["functionResponse"]["name"] for p in fr_parts]
    assert names == ["add", "mul"], names


def test_function_schema_nullable_type_array_flattens(monkeypatch):
    """Round 14: OpenAI strict tools commonly use `"type": ["string", "null"]`
    for optional fields. Gemini's OpenAPI-style Schema rejects union types and
    expects `"type": "string"` with `"nullable": true`. The sanitizer must
    translate the union form."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": ["string", "null"]},
                            "score": {"type": ["number", "null"]},
                        },
                    },
                },
            }
        ],
    )
    decls = next(
        t["functionDeclarations"]
        for t in captured["body"].get("tools") or []
        if "functionDeclarations" in t
    )
    params = decls[0]["parameters"]["properties"]
    assert params["city"]["type"] == "string"
    assert params["city"]["nullable"] is True
    assert params["score"]["type"] == "number"
    assert params["score"]["nullable"] is True


def test_image_picker_model_with_search_off_pill_strips_text_tools(monkeypatch):
    """Round 11: image-tier model ids reject text-only tools and
    thinkingConfig at the model level regardless of the Images pill. Selecting
    gemini-2.5-flash-image + enabled_tools=["web_search"] with no
    image_generation must NOT forward googleSearch or thinkingConfig (Gemini
    400s on text tools for legacy image ids)."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        enabled_tools = ["web_search"],
        reasoning_effort = "high",
    )
    body = captured["body"]
    assert "tools" not in body, body.get("tools")
    assert "thinkingConfig" not in body.get("generationConfig", {}), body[
        "generationConfig"
    ]


def test_image_models_drop_function_declarations(monkeypatch):
    """Image-mode requests cannot mix tools with responseModalities, so
    user-supplied function declarations must be dropped."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        enabled_tools = ["image_generation"],
        tools = [
            {
                "type": "function",
                "function": {"name": "noop", "parameters": {"type": "object"}},
            }
        ],
    )
    assert captured["body"].get("tools") is None
    assert captured["body"]["generationConfig"]["responseModalities"] == [
        "TEXT",
        "IMAGE",
    ]


def test_safe_fetch_image_rejects_malformed_bracketed_url():
    """Round 17: bracketed IPv6 garbage like `https://[bad/x.png` makes
    urlparse raise ValueError. The fetch helper must catch it and drop the
    image rather than crashing the request mid-build."""
    res = _drive(ep_mod._safe_fetch_image_for_gemini("https://[bad/x.png", "image/png"))
    assert res is None


def test_safe_fetch_image_pins_validated_ip_no_hostname_in_request(monkeypatch):
    """Round 17: the fetch helper must pin the validated IP into the outgoing
    request URL (with a Host header carrying the original hostname). A second
    hostname-style getaddrinfo after validate would be a DNS-rebinding gap, so
    we assert the urllib opener is called with an IP-rewritten URL."""
    import socket

    captured: dict = {"requests": []}

    # Public IP during validate; record every getaddrinfo call.
    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, *args, **kwargs):
        captured.setdefault("dns", []).append(host)
        if host == "cdn.example.com":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("8.8.8.8", 0),
                )
            ]
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    class _StubResp:
        status = 200
        headers = {"content-type": "image/png", "content-length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n = None):
            return b"PNG"

    class _StubOpener:
        def open(
            self,
            req,
            timeout = None,
        ):
            captured["requests"].append(
                {
                    "url": req.full_url,
                    "host_header": req.get_header("Host"),
                }
            )
            return _StubResp()

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_args, **_kw: _StubOpener()
    )

    res = _drive(
        ep_mod._safe_fetch_image_for_gemini(
            "https://cdn.example.com/x.png", "image/png"
        )
    )
    assert res is not None
    assert res[0] == "image/png"
    # Outgoing URL must use the pinned IP literal, not the hostname.
    assert any("8.8.8.8" in r["url"] for r in captured["requests"]), captured
    assert all(
        "cdn.example.com" not in r["url"] for r in captured["requests"]
    ), captured
    # Host header still carries the original hostname for vhost/SNI.
    assert captured["requests"][0]["host_header"] == "cdn.example.com"


def test_safe_fetch_image_redirect_to_private_host_rejected(monkeypatch):
    """Round 17: each redirect hop must re-validate the new host. A public hop
    that redirects to an internal address must be dropped."""
    import socket
    import urllib.error

    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, *args, **kwargs):
        if host == "cdn.example.com":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("1.1.1.1", 0),
                )
            ]
        if host == "internal.bad":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("10.0.0.5", 0),
                )
            ]
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    class _StubOpener:
        def open(
            self,
            req,
            timeout = None,
        ):
            # Simulate a 302 to a private host.
            raise urllib.error.HTTPError(
                req.full_url,
                302,
                "Found",
                {"Location": "https://internal.bad/secret.png"},
                None,
            )

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_args, **_kw: _StubOpener()
    )

    res = _drive(
        ep_mod._safe_fetch_image_for_gemini(
            "https://cdn.example.com/x.png", "image/png"
        )
    )
    assert res is None


def test_files_api_substring_url_not_misclassified_as_filedata(monkeypatch):
    """Round 17: a CDN URL whose path/query merely contains the Files API
    substring must NOT be sent as `fileData.fileUri`; route it through the
    safe-fetch path. The old substring check
    `"generativelanguage.googleapis.com/" in url.lower()` matched any URL
    carrying that text anywhere."""
    captured_outbound: dict = {}
    fetch_calls: list[str] = []

    async def fake_fetch(
        url,
        fallback_mime,
        max_bytes = None,
    ):
        fetch_calls.append(url)
        return "image/png", base64.b64encode(b"DATA").decode("ascii")

    monkeypatch.setattr(ep_mod, "_safe_fetch_image_for_gemini", fake_fetch)

    def handler(request: httpx.Request) -> httpx.Response:
        captured_outbound["body"] = json.loads(request.content.decode("utf-8"))
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

    async def run():
        client = _make_gemini_client()
        async for _ in client.stream_chat_completion(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                # Files-API-looking path, but host is an
                                # attacker CDN.
                                "url": "https://evil.example/path/generativelanguage.googleapis.com/v1beta/files/abc.png",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # Looks YouTube-ish in the path.
                                "url": "https://cdn.example.com/youtube.com/cat.png",
                            },
                        },
                    ],
                }
            ],
            model = "gemini-2.5-flash",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
        ):
            pass
        await client.close()

    _drive(run())

    parts = captured_outbound["body"]["contents"][-1]["parts"]
    assert not any("fileData" in p for p in parts), parts
    inline_count = sum(1 for p in parts if "inlineData" in p)
    assert inline_count == 2, parts
    assert len(fetch_calls) == 2, fetch_calls


def test_function_schema_anyof_null_variant_flattens_to_nullable(monkeypatch):
    """Round 17: OpenAI/Pydantic emit `anyOf: [{X}, {"type":"null"}]` for
    Optional[X]. Gemini's OpenAPI subset rejects `"type":"null"` inside anyOf.
    The sanitizer must collapse a singleton-plus-null union back to the
    non-null branch with `nullable: true`."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"},
                                ]
                            },
                            "count": {
                                "anyOf": [
                                    {"type": "integer"},
                                    {"type": "null"},
                                ]
                            },
                        },
                    },
                },
            }
        ],
    )
    decls = next(
        t["functionDeclarations"]
        for t in captured["body"].get("tools") or []
        if "functionDeclarations" in t
    )
    params = decls[0]["parameters"]["properties"]
    assert params["label"]["type"] == "string"
    assert params["label"]["nullable"] is True
    assert "anyOf" not in params["label"]
    assert params["count"]["type"] == "integer"
    assert params["count"]["nullable"] is True


def test_legacy_gemini3_pro_medium_coerced_to_high(monkeypatch):
    """Round 17: legacy `gemini-3-pro*` (incl. `-preview`, shut down
    2026-03-09) only accepted low/high. 3.1+ Pro added medium. The backend
    must coerce medium → high for the legacy model so stale UI state doesn't
    400 the request."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3-pro-preview",
        reasoning_effort = "medium",
    )
    assert captured["body"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "high"
    }


def test_gemini_3_1_pro_medium_passes_through(monkeypatch):
    """Round 17 regression: 3.1+ Pro accepts medium; coercion must NOT apply
    when the model id is gemini-3.1-pro*."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-3.1-pro-preview",
        reasoning_effort = "medium",
    )
    assert captured["body"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "medium"
    }


def test_tool_calls_extra_content_stripped_for_non_native_gemini():
    """Round 17: per-tool-call `extra_content` (Gemini thoughtSignature
    carrier) must not leak through `_build_external_messages` to
    non-native-Gemini providers; OpenAI / Anthropic / custom Gemini OAI-compat
    gateways would 400 on the unknown key."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                        "extra_content": {
                            "google": {"thought_signature": "SIG"},
                        },
                    }
                ],
            }
        ],
        "stream": True,
    }
    req = ChatCompletionRequest.model_validate(payload)

    # Non-native providers (openai, custom Gemini OAI-compat proxy) must have
    # extra_content stripped from the tool_call entry.
    for provider_type, base_url in [
        ("openai", None),
        ("gemini", "https://litellm.example/v1"),
    ]:
        result = _build_external_messages(
            req.messages,
            supports_vision = True,
            provider_type = provider_type,
            base_url = base_url,
        )
        assert len(result) == 1
        tc = result[0]["tool_calls"][0]
        assert "extra_content" not in tc, (provider_type, tc)

    # Native Gemini still receives extra_content for the round-trip.
    result_native = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta",
    )
    tc_native = result_native[0]["tool_calls"][0]
    assert tc_native["extra_content"]["google"]["thought_signature"] == "SIG"


def test_user_function_named_with_server_tool_arg_not_dropped(monkeypatch):
    """Round 17: the OpenAI Responses translator must NOT drop a user function
    whose JSON arguments contain `_server_tool: true` UNLESS the function name
    is also a canonical builtin name. Otherwise a user schema with an
    `_server_tool` field becomes invisible to the model."""
    captured: dict = {"input_items": None}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["input_items"] = body.get("input")
        return httpx.Response(
            200,
            content = b'data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_user",
                            "type": "function",
                            "function": {
                                "name": "user_function",
                                "arguments": json.dumps(
                                    {"_server_tool": True, "q": "x"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "result",
                    "tool_call_id": "call_user",
                    "name": "user_function",
                },
                {"role": "user", "content": "continue"},
            ],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 1.0,
            max_tokens = 16,
        ):
            pass
        await client.close()

    _drive(run())

    items = captured["input_items"] or []
    fn_calls = [i for i in items if i.get("type") == "function_call"]
    fn_outs = [i for i in items if i.get("type") == "function_call_output"]
    # User function call must survive (call + output).
    assert any(c.get("name") == "user_function" for c in fn_calls), items
    assert len(fn_outs) == 1, items


def test_builtin_named_with_server_tool_marker_dropped(monkeypatch):
    """Round 17 control: a builtin (web_search) tagged with `_server_tool:
    true` continues to be filtered from outbound history."""
    captured: dict = {"input_items": None}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["input_items"] = body.get("input")
        return httpx.Response(
            200,
            content = b'data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "search please"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps(
                                    {"_server_tool": True, "query": "x"}
                                ),
                            },
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 1.0,
            max_tokens = 16,
        ):
            pass
        await client.close()

    _drive(run())

    items = captured["input_items"] or []
    fn_calls = [i for i in items if i.get("type") == "function_call"]
    # Builtin server-side tool call must be filtered out.
    assert all(c.get("name") != "web_search" for c in fn_calls), items


def test_gemini_tool_choice_none_disables_hosted_builtins(monkeypatch):
    """Round 18: `tool_choice="none"` must drop hosted Google Search / code
    execution from the Gemini body, not just user function declarations.
    Otherwise an API client that opted out of tool use still triggers grounded
    search (privacy + billing)."""
    captured = _capture_body(
        monkeypatch,
        enabled_tools = ["web_search", "code_execution"],
        tool_choice = "none",
    )
    assert captured["body"].get("tools") is None, captured["body"]


def test_gemini_tool_choice_none_disables_function_declarations(monkeypatch):
    """Round 18: `tool_choice="none"` must drop user function declarations as
    well as hosted builtins from the Gemini body."""
    captured = _capture_body(
        monkeypatch,
        tool_choice = "none",
        tools = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    )
    assert captured["body"].get("tools") is None, captured["body"]


def test_schema_anyof_multitype_with_null_keeps_anyof_and_nullable(monkeypatch):
    """Round 18: multi-branch unions with null (e.g. `Union[str, int, None]`)
    must keep the slim anyOf without the null branch and add `nullable: true`;
    Gemini rejects `{"type":"null"}` inside anyOf."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "either": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "integer"},
                                    {"type": "null"},
                                ]
                            },
                        },
                    },
                },
            }
        ],
    )
    decls = next(
        t["functionDeclarations"]
        for t in captured["body"].get("tools") or []
        if "functionDeclarations" in t
    )
    either = decls[0]["parameters"]["properties"]["either"]
    assert either.get("nullable") is True
    inner = either.get("anyOf")
    assert isinstance(inner, list) and len(inner) == 2, either
    assert all(
        not (isinstance(b, dict) and b.get("type") == "null") for b in inner
    ), inner


def test_safe_fetch_image_redirect_malformed_url_no_crash(monkeypatch):
    """Round 18: when the upstream 302 Location is a malformed bracketed-IPv6
    URL, the helper must return None instead of letting a urlparse ValueError
    abort the chat stream."""
    import socket
    import urllib.error

    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, *args, **kwargs):
        if host == "cdn.example.com":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("1.1.1.1", 0),
                )
            ]
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    class _StubOpener:
        def open(
            self,
            req,
            timeout = None,
        ):
            raise urllib.error.HTTPError(
                req.full_url,
                302,
                "Found",
                {"Location": "https://[bad/x.png"},
                None,
            )

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_args, **_kw: _StubOpener()
    )

    res = _drive(
        ep_mod._safe_fetch_image_for_gemini(
            "https://cdn.example.com/x.png", "image/png"
        )
    )
    assert res is None


def test_safe_fetch_image_malformed_port_no_crash():
    """Round 18: a URL with a non-numeric port (`https://h:bad/x.png`) must
    not raise; urlparse's port property lazily ValueErrors."""
    res = _drive(
        ep_mod._safe_fetch_image_for_gemini(
            "https://example.com:bad/x.png", "image/png"
        )
    )
    assert res is None


def test_safe_fetch_image_missing_content_type_uses_fallback(monkeypatch):
    """Round 18: when the server returns image bytes but no Content-Type
    header, the helper must use the caller-provided fallback MIME (guessed from
    URL extension) instead of dropping the image as `non-image
    content-type=<none>`."""
    import socket

    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, *args, **kwargs):
        if host == "cdn.example.com":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("1.1.1.1", 0),
                )
            ]
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    class _StubResp:
        status = 200
        headers = {"content-length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n = None):
            return b"PNG"

    class _StubOpener:
        def open(
            self,
            req,
            timeout = None,
        ):
            return _StubResp()

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_args, **_kw: _StubOpener()
    )

    res = _drive(
        ep_mod._safe_fetch_image_for_gemini(
            "https://cdn.example.com/cat.png", "image/png"
        )
    )
    assert res is not None
    assert res[0] == "image/png"


def test_anthropic_translates_openai_tool_calls_into_tool_use_blocks(monkeypatch):
    """Round 18: an assistant turn with OpenAI-style top-level `tool_calls`
    must be translated into Anthropic native `{type:"tool_use", id, name,
    input}` content blocks before forwarding. The OpenAI `role="tool"`
    follow-up must become a `role:"user"` message with a `tool_result`
    block."""
    captured: dict = {"messages": None}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["messages"] = body.get("messages")
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com",
            api_key = "sk-ant-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "look up X"},
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"x"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "result_text",
                    "tool_call_id": "call_a",
                    "name": "lookup",
                },
                {"role": "user", "content": "summarise"},
            ],
            model = "claude-sonnet-4-5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
        ):
            pass
        await client.close()

    _drive(run())

    msgs = captured["messages"] or []
    # No top-level tool_calls should remain.
    assert all("tool_calls" not in m for m in msgs), msgs
    # The assistant turn must now have content blocks including a tool_use
    # block.
    asst = [m for m in msgs if m.get("role") == "assistant"]
    assert asst and isinstance(asst[0]["content"], list), asst
    tool_uses = [b for b in asst[0]["content"] if b.get("type") == "tool_use"]
    assert len(tool_uses) == 1, asst[0]
    assert tool_uses[0]["name"] == "lookup"
    assert tool_uses[0]["input"] == {"q": "x"}
    # The role="tool" message must become a user/tool_result message.
    tool_results: list[dict] = []
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            tool_results.extend(
                b for b in m["content"] if b.get("type") == "tool_result"
            )
    assert any(
        tr.get("tool_use_id") == "call_a" and tr.get("content") == "result_text"
        for tr in tool_results
    ), msgs


def test_unmarked_user_web_search_function_survives_serialization():
    """Round 18: a user-defined function literally named `web_search` with NO
    `_server_tool` marker must survive `_build_external_messages` when
    forwarded to a non-native provider; only marked synthetic builtin cards may
    be dropped."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_user",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "x"}',
                        },
                    }
                ],
            }
        ],
        "stream": True,
    }
    req = ChatCompletionRequest.model_validate(payload)
    result = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "openai",
        base_url = None,
    )
    assert len(result) == 1, result
    tcs = result[0].get("tool_calls") or []
    assert len(tcs) == 1, result
    assert tcs[0]["function"]["name"] == "web_search"


def test_marked_server_builtin_dropped_from_build_external_messages():
    """Round 18: when a Gemini-native turn carrying a marked `image_generation`
    server-tool card is forwarded to OpenAI / a custom Gemini OAI-compat proxy,
    the tool_call must be dropped, not just have its extra_content stripped.
    Forwarding an orphan `image_generation` tool_call would 400 the receiving
    API."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    marked_args = json.dumps({"_server_tool": True, "kind": "image"})
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "image_generation",
                            "arguments": marked_args,
                        },
                    }
                ],
            }
        ],
        "stream": True,
    }
    req = ChatCompletionRequest.model_validate(payload)
    # Non-native providers: marked builtin tool_call must be dropped, and if it
    # was the only payload, the whole message disappears.
    for provider_type, base_url in [
        ("openai", None),
        ("gemini", "https://litellm.example/v1"),
    ]:
        result = _build_external_messages(
            req.messages,
            supports_vision = True,
            provider_type = provider_type,
            base_url = base_url,
        )
        # Empty assistant turn with only synthetic tool_call dropped.
        assert result == [] or all(not (m.get("tool_calls") or []) for m in result), (
            provider_type,
            result,
        )

    # Native Gemini preserves it (round-trips via extra_content).
    result_native = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "gemini",
        base_url = "https://generativelanguage.googleapis.com/v1beta",
    )
    assert len(result_native) == 1
    assert result_native[0]["tool_calls"][0]["function"]["name"] == "image_generation"


def test_openai_responses_tool_choice_none_drops_hosted_tools(monkeypatch):
    """Round 18: `tool_choice="none"` must also drop hosted OpenAI Responses
    builtins (web_search, code execution shell, image generation), not just
    user function tools."""
    captured: dict = {"body": None}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 1.0,
            max_tokens = 16,
            enabled_tools = ["web_search", "code_execution", "image_generation"],
            tool_choice = "none",
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"] or {}
    assert body.get("tools") in (None, []), body


def test_anthropic_tool_choice_none_drops_hosted_tools(monkeypatch):
    """Round 19: tool_choice="none" must opt out of Anthropic hosted builtins
    (web_search, web_fetch, code_execution) like it does for Gemini and OpenAI
    Responses."""
    captured: dict = {"body": None}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com",
            api_key = "sk-ant-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-sonnet-4-5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search", "web_fetch", "code_execution"],
            tool_choice = "none",
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"] or {}
    assert body.get("tools") in (None, []), body


def test_openrouter_tool_choice_none_drops_web_plugin(monkeypatch):
    """Round 19: tool_choice="none" must drop the OpenRouter web plugin so a
    request that opted out of tool use doesn't still trigger hosted web
    search."""
    captured: dict = {"body": None}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b"data: [DONE]\n\n",
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openrouter",
            base_url = "https://openrouter.ai/api/v1",
            api_key = "sk-or-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "openai/gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search"],
            tool_choice = "none",
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"] or {}
    assert body.get("plugins") in (None, []), body


def test_kimi_tool_choice_none_skips_web_search_helper(monkeypatch):
    """Round 19: when tool_choice="none" plus enabled_tools=["web_search"] on
    Kimi, the dispatcher must NOT route into `_stream_kimi_web_search`. Falling
    through to the generic OAI-compat path is expected."""
    routed_to_helper = {"called": False}

    real_helper = ExternalProviderClient._stream_kimi_web_search

    async def fake_helper(self, *args, **kwargs):  # noqa: ARG001
        routed_to_helper["called"] = True
        if False:
            yield ""  # pragma: no cover

    monkeypatch.setattr(
        ExternalProviderClient,
        "_stream_kimi_web_search",
        fake_helper,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = b"data: [DONE]\n\n",
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi",
            base_url = "https://api.moonshot.ai/v1",
            api_key = "sk-kimi-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "kimi-k2.6",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search"],
            tool_choice = "none",
        ):
            pass
        await client.close()

    _drive(run())
    assert routed_to_helper["called"] is False

    monkeypatch.setattr(
        ExternalProviderClient,
        "_stream_kimi_web_search",
        real_helper,
    )


def test_user_code_execution_function_not_dropped():
    """Round 19: a user-declared function literally named `code_execution` with
    normal `code` arguments must survive `_build_external_messages` -- round
    17's shape heuristic dropped it, breaking function-calling round-trips."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_user",
                        "type": "function",
                        "function": {
                            "name": "code_execution",
                            "arguments": '{"code": "print(1)"}',
                        },
                    }
                ],
            }
        ],
        "stream": True,
    }
    req = ChatCompletionRequest.model_validate(payload)
    result = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "openai",
        base_url = None,
    )
    assert len(result) == 1, result
    tcs = result[0].get("tool_calls") or []
    assert len(tcs) == 1, result
    assert tcs[0]["function"]["name"] == "code_execution"


def test_native_part_code_execution_treated_as_server_side():
    """Round 19: a Gemini `code_execution` card persists its replay payload at
    `args.google.native_part` (no `_server_tool` marker on pre-PR cards). The
    backend filter must still drop it for non-native providers because it's a
    synthetic card, not a real user function."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    args_with_native_part = json.dumps(
        {
            "google": {
                "native_part": {
                    "executableCode": {
                        "language": "PYTHON",
                        "code": "print(1)",
                    }
                }
            }
        }
    )
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {
                            "name": "code_execution",
                            "arguments": args_with_native_part,
                        },
                    }
                ],
            }
        ],
        "stream": True,
    }
    req = ChatCompletionRequest.model_validate(payload)
    result = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "openai",
        base_url = None,
    )
    assert result == [] or all(not (m.get("tool_calls") or []) for m in result), result


def test_remote_image_fetch_attempt_cap_includes_failures(monkeypatch):
    """Round 19: the per-request image fetch count cap must count ATTEMPTS,
    not just successes. Otherwise a request with 100 failing/slow URLs runs 100
    fetches each up to the 15s timeout."""
    fetch_calls: list[str] = []

    async def fake_fetch(
        url,
        fallback_mime,
        max_bytes = None,
    ):
        fetch_calls.append(url)
        return None

    monkeypatch.setattr(ep_mod, "_safe_fetch_image_for_gemini", fake_fetch)

    def handler(request: httpx.Request) -> httpx.Response:
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

    async def run():
        client = _make_gemini_client()
        image_parts = [
            {
                "type": "image_url",
                "image_url": {"url": f"https://cdn.example.com/img{idx}.png"},
            }
            for idx in range(20)
        ]
        async for _ in client.stream_chat_completion(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        *image_parts,
                    ],
                }
            ],
            model = "gemini-2.5-flash",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
        ):
            pass
        await client.close()

    _drive(run())
    assert len(fetch_calls) <= 8, len(fetch_calls)


def test_orphan_function_call_output_dropped_when_call_skipped(monkeypatch):
    """Round 19: when a marked server-side builtin `function_call` is dropped
    from OpenAI Responses input items, the matching role=tool follow-up must
    also be dropped to avoid an orphan `function_call_output`."""
    captured: dict = {"input_items": None}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["input_items"] = body.get("input")
        return httpx.Response(
            200,
            content = b'data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "search please"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps(
                                    {"_server_tool": True, "query": "x"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "result_text",
                    "tool_call_id": "call_b",
                    "name": "web_search",
                },
                {"role": "user", "content": "continue"},
            ],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 1.0,
            max_tokens = 16,
        ):
            pass
        await client.close()

    _drive(run())

    items = captured["input_items"] or []
    fn_calls = [i for i in items if i.get("type") == "function_call"]
    fn_outs = [i for i in items if i.get("type") == "function_call_output"]
    assert all(c.get("call_id") != "call_b" for c in fn_calls), items
    assert all(o.get("call_id") != "call_b" for o in fn_outs), items


def test_schema_multitype_union_with_null_preserves_anyof(monkeypatch):
    """Round 19: a JSON Schema `"type": ["string","integer","null"]` must be
    sanitized to anyOf:[{string},{integer}] + nullable:true. Flattening to just
    `{"type":"string"}` silently drops the integer branch and changes the
    function contract."""
    captured = _capture_body(
        monkeypatch,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "either": {"type": ["string", "integer", "null"]},
                        },
                    },
                },
            }
        ],
    )
    decls = next(
        t["functionDeclarations"]
        for t in captured["body"].get("tools") or []
        if "functionDeclarations" in t
    )
    either = decls[0]["parameters"]["properties"]["either"]
    assert either.get("nullable") is True
    inner = either.get("anyOf")
    assert isinstance(inner, list) and len(inner) == 2, either
    types = sorted(
        b.get("type") for b in inner if isinstance(b, dict) and b.get("type")
    )
    assert types == ["integer", "string"], inner


def test_invalid_gemini_model_rejected_before_image_fetch(monkeypatch):
    """Round 19: invalid Gemini model IDs are rejected at the top of
    `_stream_gemini`, BEFORE any user-controlled remote image fetch runs."""
    fetch_calls: list[str] = []

    async def fake_fetch(
        url,
        fallback_mime,
        max_bytes = None,
    ):
        fetch_calls.append(url)
        return None

    monkeypatch.setattr(ep_mod, "_safe_fetch_image_for_gemini", fake_fetch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = b"",
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = _make_gemini_client()
        async for _ in client.stream_chat_completion(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/x.png"},
                        },
                    ],
                }
            ],
            model = "../cachedContents/leak",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
        ):
            pass
        await client.close()

    _drive(run())
    assert fetch_calls == [], fetch_calls


def test_empty_assistant_turn_skipped_after_synthetic_tool_calls_dropped():
    """Round 20: when `_filter_tool_calls` drops every synthetic server-builtin
    tool_call on an empty-content assistant turn, the whole message must be
    skipped. Several providers reject `{"role":"assistant","content":""}` as an
    empty assistant turn."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    marked_args = json.dumps({"_server_tool": True, "kind": "image"})
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "image_generation",
                            "arguments": marked_args,
                        },
                    }
                ],
            }
        ],
        "stream": True,
    }
    req = ChatCompletionRequest.model_validate(payload)
    for provider_type, base_url in [
        ("openai", None),
        ("gemini", "https://litellm.example/v1"),
    ]:
        result = _build_external_messages(
            req.messages,
            supports_vision = True,
            provider_type = provider_type,
            base_url = base_url,
        )
        # The empty assistant turn (only a synthetic builtin) must NOT appear
        # in the output at all.
        assert result == [], (provider_type, result)


def test_role_tool_dropped_when_matching_synthetic_call_filtered():
    """Round 20: `_build_external_messages` drops the matching role=tool
    follow-up when its tool_call was a synthetic builtin that
    `_filter_tool_calls` removed. Otherwise the receiving provider sees an
    orphan tool_result with no tool_call."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_external_messages

    marked_args = json.dumps({"_server_tool": True, "query": "x"})
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": marked_args,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "result_text",
                "tool_call_id": "call_b",
                "name": "web_search",
            },
            {"role": "user", "content": "continue"},
        ],
        "stream": True,
    }
    req = ChatCompletionRequest.model_validate(payload)
    result = _build_external_messages(
        req.messages,
        supports_vision = True,
        provider_type = "openai",
        base_url = None,
    )
    # Only the user "continue" message survives.
    roles = [m.get("role") for m in result]
    assert roles == ["user"], result


def test_openrouter_no_synthetic_web_search_event_on_tool_choice_none(monkeypatch):
    """Round 20: OpenRouter dispatcher must not emit synthetic web_search
    tool_start / tool_end events when tool_choice="none"; otherwise the chat UI
    shows a search card for a search that never happened."""
    captured_events: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = b"data: [DONE]\n\n",
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openrouter",
            base_url = "https://openrouter.ai/api/v1",
            api_key = "sk-or-test",
        )
        async for line in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "openai/gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search"],
            tool_choice = "none",
        ):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            # Backend emits synthetic tool events as a top-level `_toolEvent`
            # on the SSE payload (not nested inside `delta`). Read both shapes
            # so a future format change can't mask this regression.
            evt = obj.get("_toolEvent")
            if isinstance(evt, dict):
                captured_events.append(evt)
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                nested = delta.get("_toolEvent") if isinstance(delta, dict) else None
                if isinstance(nested, dict):
                    captured_events.append(nested)
        await client.close()

    _drive(run())
    # No synthetic web_search tool_start / tool_end emitted.
    assert all(
        e.get("tool_name") != "web_search" for e in captured_events
    ), captured_events


def test_anthropic_role_tool_list_content_translates_to_tool_result(monkeypatch):
    """Round 20: an OpenAI-shape role=tool message with list content
    (`content=[{"type":"text","text":"result"}]`) must be translated into
    Anthropic's native tool_result block, not forwarded as an invalid role=tool
    message."""
    captured: dict = {"messages": None}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["messages"] = body.get("messages")
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com",
            api_key = "sk-ant-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "look up X"},
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"x"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": [{"type": "text", "text": "result_text"}],
                    "tool_call_id": "call_a",
                    "name": "lookup",
                },
                {"role": "user", "content": "summarise"},
            ],
            model = "claude-sonnet-4-5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
        ):
            pass
        await client.close()

    _drive(run())

    msgs = captured["messages"] or []
    assert all(m.get("role") != "tool" for m in msgs), msgs
    tool_results: list[dict] = []
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            tool_results.extend(
                b for b in m["content"] if b.get("type") == "tool_result"
            )
    assert any(
        tr.get("tool_use_id") == "call_a" and tr.get("content") == "result_text"
        for tr in tool_results
    ), msgs


def test_data_url_non_image_mime_dropped(monkeypatch):
    """Round 20: a `data:text/html;base64,...` image_url must be dropped from
    the Gemini body, not forwarded as `inlineData.mimeType="text/html"` which
    Gemini rejects."""
    captured = _capture_body(
        monkeypatch,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:text/html;base64,PGgxPmhpPC9oMT4=",
                        },
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["contents"][-1]["parts"]
    assert not any("inlineData" in p for p in parts), parts


def test_youtube_filedata_uses_video_mime(monkeypatch):
    """Round 20: YouTube `fileData.fileUri` must declare a video mimeType, not
    `image/jpeg` guessed from the URL path."""
    captured = _capture_body(
        monkeypatch,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarise"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.youtube.com/watch?v=abc",
                        },
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["contents"][-1]["parts"]
    yt = next((p for p in parts if "fileData" in p), None)
    assert yt is not None, parts
    assert yt["fileData"]["mimeType"].startswith("video/"), yt


def test_openai_responses_assistant_text_serialized_before_function_call(monkeypatch):
    """Round 20: in OpenAI Responses history, the assistant's visible text for
    a turn that ALSO emitted a function_call must serialize BEFORE the
    function_call item, matching the prior response.output sequence. Otherwise
    function_call_output (the role=tool follow-up) appears to follow an
    unrelated assistant message."""
    captured: dict = {"input_items": None}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["input_items"] = body.get("input")
        return httpx.Response(
            200,
            content = b'data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check that.",
                    "tool_calls": [
                        {
                            "id": "call_w",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "sunny",
                    "tool_call_id": "call_w",
                    "name": "get_weather",
                },
                {"role": "user", "content": "thanks"},
            ],
            model = "gpt-5.5",
            temperature = 0.7,
            top_p = 1.0,
            max_tokens = 16,
        ):
            pass
        await client.close()

    _drive(run())

    items = captured["input_items"] or []
    types = [i.get("type") or i.get("role") for i in items]
    # Expected order:
    #   user ("weather?")
    #   assistant ("Let me check that.")
    #   function_call (get_weather)
    #   function_call_output (sunny)
    #   user ("thanks")
    assert types == [
        "user",
        "assistant",
        "function_call",
        "function_call_output",
        "user",
    ], items


def test_gemini_tool_choice_none_disables_image_generation(monkeypatch):
    """Round 21: `tool_choice="none"` must also flip the implicit
    image-generation hosted tool off on image-tier models. Otherwise
    `responseModalities=["TEXT","IMAGE"]` still rides on the body and the
    provider can generate (and bill for) image output despite the explicit
    OpenAI tool opt-out."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        enabled_tools = ["image_generation"],
        tool_choice = "none",
    )
    body = captured["body"]
    assert body["generationConfig"].get("responseModalities") == ["TEXT"], body


def test_gemini_forced_function_tool_choice_drops_hosted_builtins(monkeypatch):
    """Round 21: forced-function `tool_choice` (e.g.
    `{"type":"function","function":{"name":"lookup"}}`) must suppress hosted
    Google Search / code execution. Gemini's toolConfig only constrains
    function declarations, not hosted tools, so leaving
    `googleSearch`/`codeExecution` in `tools[]` lets them fire despite the
    caller pinning a specific user function."""
    captured = _capture_body(
        monkeypatch,
        enabled_tools = ["web_search", "code_execution"],
        tools = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        tool_choice = {
            "type": "function",
            "function": {"name": "lookup"},
        },
    )
    body = captured["body"]
    tool_kinds = [list(t.keys())[0] for t in (body.get("tools") or [])]
    assert "googleSearch" not in tool_kinds, body
    assert "codeExecution" not in tool_kinds, body
    # User function declaration still survives.
    assert "functionDeclarations" in tool_kinds, body


def test_gemini_forced_function_tool_choice_drops_image_generation(monkeypatch):
    """Round 21: forced-function `tool_choice` must also flip the implicit
    image-generation hosted tool off on image-tier models."""
    captured = _capture_body(
        monkeypatch,
        model = "gemini-2.5-flash-image",
        enabled_tools = ["image_generation"],
        tool_choice = {
            "type": "function",
            "function": {"name": "lookup"},
        },
        tools = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    )
    body = captured["body"]
    assert body["generationConfig"].get("responseModalities") == ["TEXT"], body


def test_gemini_code_execution_native_part_list_replays_per_part_signatures(
    monkeypatch,
):
    """Round 21: merged code-execution history must replay per-part
    `thoughtSignature`s, not fan one top-level signature across every native
    subpart. Gemini 3 strict validators reject a signature on the wrong
    part."""
    history = [
        {"role": "user", "content": "plot 1+1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "code_execution",
                        "arguments": "{}",
                    },
                    "extra_content": {
                        "google": {
                            "native_part": {
                                "parts": [
                                    {
                                        "executableCode": {
                                            "id": "code_a",
                                            "language": "PYTHON",
                                            "code": "print(1+1)",
                                        },
                                        "thoughtSignature": "SIG-EXEC",
                                    },
                                    {
                                        "codeExecutionResult": {
                                            "id": "res_a",
                                            "outcome": "OUTCOME_OK",
                                            "output": "2\n",
                                        },
                                    },
                                ],
                            },
                        },
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "name": "code_execution",
            "content": "2",
        },
        {"role": "user", "content": "next"},
    ]
    captured = _capture_body(monkeypatch, messages = history)
    contents = captured["body"]["contents"]
    # Find the assistant turn replayed as native code-exec parts.
    assistant_turn = next(c for c in contents if c["role"] == "model")
    parts = assistant_turn["parts"]
    exec_parts = [p for p in parts if "executableCode" in p]
    result_parts = [p for p in parts if "codeExecutionResult" in p]
    assert exec_parts and result_parts, parts
    assert exec_parts[0].get("thoughtSignature") == "SIG-EXEC", exec_parts[0]
    # codeExecutionResult had no signature -- must NOT inherit one.
    assert "thoughtSignature" not in result_parts[0], result_parts[0]


def test_gemini_code_execution_legacy_merged_signature_only_on_executable(monkeypatch):
    """Round 21: backward compat for pre-round-21 persisted history that stored
    merged `native_part` as a single object plus a top-level
    `thoughtSignature`. The replay branch must attach that signature only to
    `executableCode` (where Gemini 3 emits it), not fan it across
    `codeExecutionResult` / `inlineData`."""
    history = [
        {"role": "user", "content": "plot 1+1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "code_execution",
                        "arguments": "{}",
                    },
                    "extra_content": {
                        "google": {
                            "native_part": {
                                "executableCode": {
                                    "id": "code_b",
                                    "language": "PYTHON",
                                    "code": "print(1+1)",
                                },
                                "codeExecutionResult": {
                                    "id": "res_b",
                                    "outcome": "OUTCOME_OK",
                                    "output": "2\n",
                                },
                                "thoughtSignature": "LEGACY-SIG",
                            },
                        },
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "name": "code_execution",
            "content": "2",
        },
        {"role": "user", "content": "next"},
    ]
    captured = _capture_body(monkeypatch, messages = history)
    contents = captured["body"]["contents"]
    assistant_turn = next(c for c in contents if c["role"] == "model")
    exec_parts = [p for p in assistant_turn["parts"] if "executableCode" in p]
    result_parts = [p for p in assistant_turn["parts"] if "codeExecutionResult" in p]
    assert exec_parts[0].get("thoughtSignature") == "LEGACY-SIG", exec_parts[0]
    assert "thoughtSignature" not in result_parts[0], result_parts[0]


def test_gemini_role_tool_list_content_flattens_to_result_text(monkeypatch):
    """Round 21: OpenAI-shape role=tool messages may carry list content like
    `[{"type":"text","text":"result"}]`. Forwarding those parts verbatim into
    `functionResponse.response.result` yields a list of content-part objects
    instead of the actual tool output text."""
    history = [
        {"role": "user", "content": "look up"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"q": "x"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "lookup",
            "content": [{"type": "text", "text": "answer-text"}],
        },
        {"role": "user", "content": "next"},
    ]
    captured = _capture_body(monkeypatch, messages = history)
    contents = captured["body"]["contents"]
    fn_response = None
    for c in contents:
        for p in c.get("parts") or []:
            if isinstance(p, dict) and "functionResponse" in p:
                fn_response = p["functionResponse"]
                break
        if fn_response:
            break
    assert fn_response is not None, contents
    assert fn_response["response"] == {"result": "answer-text"}, fn_response


def test_safe_fetch_image_threads_per_request_byte_budget(monkeypatch):
    """Round 21: the aggregate per-request byte cap must be passed into
    `_safe_fetch_image_for_gemini` so an oversize URL is refused via
    Content-Length (short-circuit) rather than fully downloaded then
    discarded."""
    import socket

    captured: dict = {"reads": 0, "content_length_seen": None}

    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, *args, **kwargs):
        if host == "cdn.example.com":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("8.8.8.8", 0),
                )
            ]
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    class _StubResp:
        status = 200
        # Declared 5 MiB, but caller passes a 1 MiB remaining budget.
        headers = {
            "content-type": "image/png",
            "content-length": str(5 * 1024 * 1024),
        }

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n = None):
            captured["reads"] += 1
            return b"\x00" * (5 * 1024 * 1024)

    class _StubOpener:
        def open(
            self,
            req,
            timeout = None,
        ):
            return _StubResp()

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_args, **_kw: _StubOpener()
    )

    res = _drive(
        ep_mod._safe_fetch_image_for_gemini(
            "https://cdn.example.com/big.png",
            "image/png",
            max_bytes = 1 * 1024 * 1024,
        )
    )
    assert res is None
    # Refused via Content-Length pre-check, never read.
    assert captured["reads"] == 0


def test_openai_chat_delta_type_includes_tool_calls_and_extra_content():
    """Round 21: the frontend `OpenAIChatDelta` interface must expose
    `tool_calls` and `extra_content` so TypeScript callers can consume the
    Gemini-native stream fields without `any` casts. A static-string assertion
    against the .ts source; mirrors how other frontend wire-contract tests are
    pinned from the backend suite."""
    import os

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    types_path = os.path.join(
        here, "frontend", "src", "features", "chat", "types", "api.ts"
    )
    with open(types_path, "r", encoding = "utf-8") as f:
        src = f.read()
    assert "tool_calls?: OpenAIToolCallPart[]" in src, src[:200]
    assert "extra_content?: Record<string, unknown>" in src, src[:200]
    assert "boolean | string | null" in src, src[:200]


def test_anthropic_forced_function_tool_choice_drops_hosted_tools(monkeypatch):
    """Round 22: forced-function tool_choice must suppress Anthropic hosted
    builtins like it does for Gemini. Pinning a user function
    (`tool_choice={"type":"function","function":{"name":...}}`) while passing
    `enabled_tools=["web_search","web_fetch","code_execution"]` should not still
    fire those server-side."""
    captured: dict = {"body": None}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com",
            api_key = "sk-ant-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-sonnet-4-5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search", "web_fetch", "code_execution"],
            tool_choice = {
                "type": "function",
                "function": {"name": "lookup_record"},
            },
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"] or {}
    # No hosted tools in the body — only the caller's user-function
    # declarations (none passed here).
    tools = body.get("tools") or []
    hosted_tool_names = {"web_search", "web_fetch", "code_execution"}
    for tool in tools:
        assert tool.get("name") not in hosted_tool_names, body


def test_openrouter_forced_function_tool_choice_drops_web_plugin(monkeypatch):
    """Round 22: forced-function tool_choice must drop the OpenRouter web
    plugin too — caller pinned a user function, so OpenRouter must not attach
    the hosted web-search plugin."""
    captured: dict = {"body": None}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b"data: [DONE]\n\n",
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openrouter",
            base_url = "https://openrouter.ai/api/v1",
            api_key = "sk-or-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "openai/gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search"],
            tool_choice = {
                "type": "function",
                "function": {"name": "lookup_record"},
            },
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"] or {}
    assert body.get("plugins") in (None, []), body


def test_kimi_forced_function_tool_choice_skips_web_search_helper(monkeypatch):
    """Round 22: forced-function tool_choice plus enabled_tools=["web_search"]
    on Kimi must NOT route into `_stream_kimi_web_search`. Caller pinned a user
    function; hosted $web_search should be suppressed for the same
    privacy/billing reason."""
    routed_to_helper = {"called": False}

    async def fake_helper(self, *args, **kwargs):  # noqa: ARG001
        routed_to_helper["called"] = True
        if False:
            yield ""  # pragma: no cover

    monkeypatch.setattr(
        ExternalProviderClient,
        "_stream_kimi_web_search",
        fake_helper,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = b"data: [DONE]\n\n",
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi",
            base_url = "https://api.moonshot.ai/v1",
            api_key = "sk-kimi-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "kimi-k2.6",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search"],
            tool_choice = {
                "type": "function",
                "function": {"name": "lookup_record"},
            },
        ):
            pass
        await client.close()

    _drive(run())
    assert not routed_to_helper["called"]


def test_openai_responses_forced_function_tool_choice_drops_hosted_tools(monkeypatch):
    """Round 23: forced-function tool_choice on the OpenAI Responses path must
    suppress hosted builtins (web_search, shell, image_generation) like it does
    for Gemini / Anthropic / OpenRouter / Kimi. User-defined function tools
    still flow through so the pinned function can resolve."""
    captured: dict = {"body": None}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b"event: response.completed\ndata: {}\n\n",
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-openai-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search", "code_execution", "image_generation"],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_record",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            tool_choice = {
                "type": "function",
                "function": {"name": "lookup_record"},
            },
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"] or {}
    tools = body.get("tools") or []
    hosted_types = {"web_search", "shell", "image_generation"}
    hosted_seen = {t.get("type") for t in tools if isinstance(t, dict)}
    assert not (hosted_seen & hosted_types), body
    # The user function declaration must still be present so the pin has a
    # target.
    user_function_seen = any(
        isinstance(t, dict) and t.get("type") == "function" for t in tools
    )
    assert user_function_seen, body
    # And the forced-function tool_choice must be forwarded in Responses shape:
    # `{type:"function", name:"..."}`.
    tc = body.get("tool_choice")
    assert isinstance(tc, dict) and tc.get("type") == "function", body
    assert tc.get("name") == "lookup_record", body


def test_strip_provider_synthetic_tool_history_drops_text_only_extra_content():
    """Round 24: a plain text Gemini reply (no tool_calls) carrying
    `extra_content.google.thought_signature` must still have that metadata
    stripped before being forwarded to a local llama-server backend. Without
    it, switching a Gemini thread mid-stream to a local GGUF model leaks
    Gemini-only fields to llama-server."""
    from routes.inference import _strip_provider_synthetic_tool_history

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Hello!",
            "extra_content": {"google": {"thought_signature": "SIG_ABC"}},
        },
        {"role": "user", "content": "now in pirate voice"},
    ]
    out = _strip_provider_synthetic_tool_history(messages)
    # Same three turns, but the assistant's `extra_content` is gone.
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assistant = out[1]
    assert "extra_content" not in assistant, assistant
    assert assistant["content"] == "Hello!"


def test_validate_and_resolve_host_blocks_shared_address_space():
    """Round 24 SSRF P1: 100.64.0.0/10 carrier-grade NAT addresses are
    `is_private=False` AND `is_global=False` per Python's ipaddress docs. The
    old denylist (is_private/loopback/link_local/etc.) missed them. Adding `not
    ip.is_global` as the primary gate covers all non-public ranges, current and
    future."""
    import socket as _socket
    from core.inference import tools as _tools

    orig_getaddrinfo = _socket.getaddrinfo

    def fake_getaddrinfo(hostname, port, *args, **kwargs):
        if hostname == "shared.example":
            return [
                (
                    _socket.AF_INET,
                    _socket.SOCK_STREAM,
                    0,
                    "",
                    ("100.64.0.1", port),
                ),
            ]
        return orig_getaddrinfo(hostname, port, *args, **kwargs)

    _socket.getaddrinfo = fake_getaddrinfo
    try:
        ok, reason, _ip = _tools._validate_and_resolve_host("shared.example", 443)
    finally:
        _socket.getaddrinfo = orig_getaddrinfo
    assert ok is False, (ok, reason)
    assert "non-public" in reason.lower() or "100.64.0.1" in reason


def test_gemini_custom_oai_compat_base_skips_native_allowlist():
    """Round 24: a custom Gemini OAI-compatible base (LiteLLM/proxy) must NOT
    have its model list filtered through the native Gemini allowlist regex. A
    LiteLLM gateway returning
    `["google/gemini-2.5-flash", "my-team/gemini", "gemini-2.5-flash"]` should
    pass through; the native filter would strip the prefixed IDs even though
    chat dispatch routes them via the OpenAI-compatible client."""
    import asyncio as _asyncio

    from routes import providers as _providers
    from routes.providers import (
        ProviderModelsRequest,
        list_provider_models,
    )

    captured: dict = {"base": None}

    class _FakeClient:
        def __init__(self, *, base_url, **kwargs):
            captured["base"] = base_url

        async def list_models(self):
            return [
                {"id": "google/gemini-2.5-flash"},
                {"id": "my-team/gemini"},
                {"id": "gemini-2.5-flash"},
            ]

        async def close(self):
            return None

    orig = _providers.ExternalProviderClient
    _providers.ExternalProviderClient = _FakeClient
    try:
        req = ProviderModelsRequest(
            provider_type = "gemini",
            base_url = "https://litellm.example/v1",
        )
        result = _asyncio.run(list_provider_models(req, current_subject = "unsloth"))
    finally:
        _providers.ExternalProviderClient = orig
    ids = {m.id for m in result}
    # All three IDs survive — native allowlist bypassed.
    assert "google/gemini-2.5-flash" in ids, ids
    assert "my-team/gemini" in ids, ids
    assert "gemini-2.5-flash" in ids, ids


def test_strip_provider_synthetic_tool_history_drops_synthetic_only():
    """Round 22: switching a thread from native Gemini (code_execution /
    image_generation tool_cards in history) to a local GGUF backend must strip
    the synthetic tool_calls + matching role=tool replies before llama-server
    sees them. Real user-function tool_calls and their matching tool replies
    must survive."""
    from routes.inference import _strip_provider_synthetic_tool_history

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "let me run it",
            "tool_calls": [
                {
                    "id": "synth_ce_1",
                    "type": "function",
                    "function": {
                        "name": "code_execution",
                        "arguments": json.dumps(
                            {
                                "_server_tool": True,
                                "google": {"native_part": {"parts": []}},
                            }
                        ),
                    },
                    "extra_content": {"google": {"thought_signature": "abc"}},
                },
                {
                    "id": "real_lookup",
                    "type": "function",
                    "function": {
                        "name": "lookup_user",
                        "arguments": json.dumps({"id": 42}),
                    },
                },
            ],
            "extra_content": {"google": {"thought_signature": "msglevel"}},
        },
        {
            "role": "tool",
            "tool_call_id": "synth_ce_1",
            "content": "Gemini-only result text",
        },
        {
            "role": "tool",
            "tool_call_id": "real_lookup",
            "content": '{"name": "alice"}',
        },
    ]
    out = _strip_provider_synthetic_tool_history(messages)
    assistant = next(m for m in out if m.get("role") == "assistant")
    tcs = assistant["tool_calls"]
    assert len(tcs) == 1, tcs
    assert tcs[0]["id"] == "real_lookup"
    assert "extra_content" not in tcs[0]
    assert "extra_content" not in assistant
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "real_lookup"


def test_strip_provider_synthetic_tool_history_drops_empty_assistant():
    """If every tool_call was synthetic and the assistant turn had no content,
    the entire turn must be dropped (llama-server rejects empty assistant
    messages with no tool_calls)."""
    from routes.inference import _strip_provider_synthetic_tool_history

    messages = [
        {"role": "user", "content": "draw a sloth"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "synth_imggen",
                    "type": "function",
                    "function": {
                        "name": "image_generation",
                        "arguments": json.dumps(
                            {
                                "google": {
                                    "native_part": {
                                        "parts": [
                                            {
                                                "inlineData": {
                                                    "mimeType": "image/png",
                                                    "data": "Zm9v",
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "synth_imggen", "content": "(image)"},
        {"role": "user", "content": "now try in pirate voice"},
    ]
    out = _strip_provider_synthetic_tool_history(messages)
    roles = [m.get("role") for m in out]
    # Synthetic assistant + its tool reply are both gone; only the two user
    # turns survive.
    assert roles == ["user", "user"], out


def test_openrouter_no_synthetic_web_search_event_on_forced_function_tool_choice(
    monkeypatch,
):
    """Round 22 sibling of the round-20 `tool_choice='none'` test: when the
    caller forces a specific function via `tool_choice={"type":"function", ...}`
    AND passes `enabled_tools=["web_search"]`, the OpenRouter path must NOT
    synthesize a fake `web_search` tool card. The plugin wasn't attached
    upstream, so the UI must not see a server-tool card."""
    captured_events: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = (
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openrouter",
            base_url = "https://openrouter.ai/api/v1",
            api_key = "sk-or-test",
        )
        async for line in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "openai/gpt-5.5",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 16,
            enabled_tools = ["web_search"],
            tool_choice = {
                "type": "function",
                "function": {"name": "lookup_record"},
            },
        ):
            payload = line.strip().removeprefix("data: ")
            if payload and payload != "[DONE]":
                try:
                    captured_events.append(json.loads(payload))
                except Exception:
                    pass
        await client.close()

    _drive(run())
    for evt in captured_events:
        for choice in evt.get("choices") or []:
            delta = choice.get("delta") or {}
            extra = delta.get("extra_content") or {}
            tool_event = extra.get("toolEvent") if isinstance(extra, dict) else None
            if isinstance(tool_event, dict):
                assert tool_event.get("tool_name") != "web_search", evt
