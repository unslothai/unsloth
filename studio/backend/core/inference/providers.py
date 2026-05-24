# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Static registry of supported external LLM providers.

All providers expose OpenAI-compatible /v1/chat/completions endpoints
with Bearer token authentication and SSE streaming support.
"""

import re
from typing import Any

PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "openai": {
        "display_name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_models": [
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "o3",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        # Keep the model picker scoped to the current generation. The remote
        # /v1/models listing returns dozens of historical snapshots, fine-tunes
        # and non-chat models (embeddings, TTS, image, moderation) that we
        # never want to surface in the chat UI. Filtering here so backend
        # is the single source of truth.
        "model_id_allowlist": re.compile(r"^(gpt-5\.[345]|gpt-4\.5|o3)(?:[-.]|$)"),
        # Hide dated snapshots and the retired plain gpt-5.3 id.
        "model_id_denylist": re.compile(r"^(gpt-5\.3)$|-\d{4}-\d{2}-\d{2}$"),
    },
    "anthropic": {
        "display_name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "default_models": [
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
        # Anthropic /v1/models returns dated snapshot ids alongside the
        # canonical names (e.g. claude-3-5-sonnet-20241022). Hide the
        # YYYYMMDD-suffixed variants from the picker — same intent as the
        # OpenAI denylist, just a different date format (no dashes between
        # year/month/day).
        "model_id_denylist": re.compile(r"-\d{8}$"),
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": False,
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "extra_headers": {
            "anthropic-version": "2023-06-01",
        },
        "openai_compatible": False,
        "notes": "Native Anthropic Messages API. Uses x-api-key header and /v1/messages endpoint with SSE translation.",
    },
    "gemini": {
        "display_name": "Google Gemini",
        # Native Gemini REST endpoint -- the Gemini API does NOT speak
        # OpenAI Chat Completions on this base. Requests/responses are
        # translated in `_stream_gemini` in external_provider.py.
        # API reference: https://ai.google.dev/gemini-api/docs
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        # Curated lineup -- the live ListModels response returns dozens
        # of historical / experimental / embedding ids. Cap to the
        # current chat-capable Gemini families (3.5 / 3.1 / 3 / 2.5)
        # plus the Nano Banana image trio and the rolling `*-latest`
        # aliases. `gemini-2.0-flash*` were retired by Google in 2026
        # and are intentionally excluded; the allowlist below blocks
        # them from re-appearing through the live ListModels fetch.
        # Verified against the live `/v1beta/models` catalog 2026-05-24.
        "default_models": [
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-pro-image-preview",
            "gemini-3.1-flash-image-preview",
            "gemini-2.5-flash-image",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        # The native API takes the API key on the `x-goog-api-key`
        # header. An empty `auth_prefix` ensures we send the bare key.
        "auth_header": "x-goog-api-key",
        "auth_prefix": "",
        "openai_compatible": False,
        "notes": (
            "Native Gemini API. Translation lives in _stream_gemini. "
            "API key from https://aistudio.google.com/apikey. "
            "See https://ai.google.dev/gemini-api/docs for endpoint shapes."
        ),
        # Matches the chat-capable 3.5 / 3.1 / 3 / 2.5 families plus the
        # rolling *-latest aliases (which Google rolls forward as new
        # generations ship). Image-tier ids (`-image`, `-image-preview`,
        # `nano-banana-pro-preview`) flow through the Nano Banana
        # `responseModalities` path in `_stream_gemini`. Retired 2.0
        # ids ARE NOT in this regex on purpose -- Google's ListModels
        # would otherwise re-surface them and they 404 on use.
        "model_id_allowlist": re.compile(
            r"^("
            r"gemini-3\.5-(?:flash|pro)(?:-preview)?|"
            r"gemini-3\.1-(?:flash|pro|flash-lite)(?:-preview)?(?:-customtools)?|"
            r"gemini-3\.1-flash-image-preview|"
            r"gemini-3-(?:flash|pro)(?:-preview)?|"
            r"gemini-3-pro-image-preview|"
            r"nano-banana-pro-preview|"
            r"gemini-2\.5-pro|gemini-2\.5-flash|gemini-2\.5-flash-lite|"
            r"gemini-2\.5-flash-image|"
            r"gemini-pro-latest|gemini-flash-latest|gemini-flash-lite-latest"
            r")$"
        ),
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "supports_streaming": True,
        "supports_vision": False,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "OpenAI-compatible API. deepseek-chat = V3, deepseek-reasoner = R1 thinking mode.",
    },
    "mistral": {
        "display_name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "default_models": [
            "codestral-latest",
            "devstral-latest",
            "devstral-medium-latest",
            "magistral-medium-latest",
            "ministral-14b-latest",
            "ministral-3b-latest",
            "ministral-8b-latest",
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "mistral-tiny-latest",
            "mistral-vibe-cli-latest",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "model_id_allowlist": re.compile(
            r"^(codestral-latest|devstral-latest|devstral-medium-latest|"
            r"magistral-medium-latest|ministral-(?:14b|3b|8b)-latest|"
            r"mistral-(?:large|medium|small|tiny)-latest|"
            r"mistral-vibe-cli-latest)$"
        ),
    },
    "kimi": {
        "display_name": "Kimi",
        "base_url": "https://api.moonshot.ai/v1",
        # Current Kimi model lineup per the official docs:
        #   https://platform.kimi.ai/docs/models
        # Listing/overview endpoints used to enumerate them:
        #   https://platform.kimi.ai/docs/api/list-models
        #   https://platform.kimi.ai/docs/api/overview
        # kimi-k2.6 and kimi-k2.5 are the two SoTA multimodal models we
        # surface in the picker; everything else (moonshot-v1-*, dated
        # k2 previews) is filtered out by model_id_allowlist below.
        "default_models": [
            "kimi-k2.6",
            "kimi-k2.5",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "Moonshot API key. China: use base URL https://api.moonshot.cn/v1",
        "model_id_allowlist": re.compile(r"^kimi-k2\.[56]$"),
        # Both k2.6 and k2.5 are reasoning-class. The API rejects custom
        # sampling: "invalid temperature: only 1 is allowed for this model"
        # (and the same shape for top_p). Strip both fields from the
        # outbound body so the server falls back to its required defaults.
        "body_omit": ("temperature", "top_p"),
    },
    "qwen": {
        "display_name": "Qwen",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_models": [
            "qwen-plus",
            "qwen-turbo",
            "qwen-max",
            "qwen2.5-72b-instruct",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "DashScope API key. China mainland: override base URL to https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "huggingface": {
        "display_name": "Hugging Face",
        "base_url": "https://router.huggingface.co/v1",
        # Seed the picker with a few popular ids so something is selectable
        # before the live /v1/models call resolves. The remote listing is
        # the source of truth — see model_list_mode below.
        "default_models": [
            "openai/gpt-oss-120b",
            "deepseek-ai/DeepSeek-V3",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": (
            "HF token from huggingface.co/settings/tokens. Uses the "
            "OpenAI-compatible router at /v1/chat/completions; /v1/models "
            "returns the cross-provider chat catalog. See "
            "https://huggingface.co/docs/inference-providers/index."
        ),
        # /v1/models works on the HF router and returns the full chat-model
        # catalog (state.org/model[:policy] ids). Switch to remote so users
        # see live availability — the picker has a search box, and
        # loadModels() merges defaults so default_models entries remain
        # visible if the remote call fails.
        "model_list_mode": "remote",
        # Scope the catalog to first-party org repos we trust as primary
        # sources. The HF /v1/models response is otherwise hundreds of
        # ids long (community fine-tunes, mirrors, fp8 variants, etc.).
        "model_id_allowlist": re.compile(
            r"^(openai|deepseek-ai|google|meta-llama|Qwen|moonshotai|"
            r"mistralai|zai-org)/"
        ),
        # Cap the post-filter list. /v1/models has no server-side limit
        # or popularity sort, so this is just "first N matches" — pair it
        # with the default_models seed so the most useful flagship ids
        # are always among the top regardless of the API's order.
        "model_id_limit": 15,
    },
    "vllm": {
        "display_name": "vLLM",
        # User-supplied via provider_base_url; the route layer already falls
        # back to the payload's base_url when the registry entry has none.
        "base_url": "",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        # Force /v1/chat/completions in stream_chat_completion — vLLM's
        # /v1/responses rebuilds messages and runs them through the loaded
        # model's chat template, which 400s on strict-alternation templates
        # (Gemma 3 raises "Conversation roles must alternate user/assistant
        # /user/assistant/..."). The chat-completions path takes messages
        # verbatim and avoids that template gauntlet.
        "notes": "Self-hosted vLLM server. Always routed to /v1/chat/completions.",
        # Surfaced through the frontend's CUSTOM_PROVIDER_PRESETS, not the
        # /api/providers/registry dropdown — see list_available_providers.
        "hidden": True,
    },
    "ollama": {
        "display_name": "Ollama",
        "base_url": "http://localhost:11434/v1",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": (
            "Local Ollama server. OpenAI-compatible /v1/chat/completions; "
            "no API key. Surfaced via CUSTOM_PROVIDER_PRESETS in the frontend."
        ),
        "hidden": True,
    },
    "llama_cpp": {
        "display_name": "llama.cpp",
        "base_url": "http://localhost:8080/v1",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": (
            "Local llama.cpp server (llama-server). OpenAI-compatible "
            "/v1/chat/completions. Surfaced via CUSTOM_PROVIDER_PRESETS."
        ),
        "hidden": True,
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        # Curated list for Studio's picker (explicitly locked, not live /models).
        "default_models": [
            "openrouter/free",
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4-5",
            "google/gemini-2.5-flash",
            "mistralai/mistral-large-2411",
            "deepseek/deepseek-r1",
            "mistralai/mistral-small-3.1-24b-instruct",
            "perceptron/perceptron-mk1",
            "inclusionai/ring-2.6-1t:free",
            "google/gemini-3.1-flash-lite",
            "baidu/cobuddy:free",
            "openai/gpt-chat-latest",
            "x-ai/grok-4.3",
            "ibm-granite/granite-4.1-8b",
            "openrouter/owl-alpha",
            "poolside/laguna-xs.2:free",
            "~google/gemini-pro-latest",
            "~moonshotai/kimi-latest",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {
            "HTTP-Referer": "https://unsloth.ai",
            "X-Title": "Unsloth Studio",
        },
        "notes": "Unified gateway to 300+ models across all major providers. HTTP-Referer and X-Title headers sent for attribution.",
        "model_list_mode": "curated",
    },
}


def get_provider_info(provider_type: str) -> dict[str, Any] | None:
    """Return the registry entry for a provider type, or None if unknown."""
    return PROVIDER_REGISTRY.get(provider_type)


def get_base_url(provider_type: str) -> str | None:
    """Return the default base URL for a provider type."""
    info = PROVIDER_REGISTRY.get(provider_type)
    return info["base_url"] if info else None


def list_available_providers() -> list[dict[str, Any]]:
    """Return all registered providers (for the /registry endpoint).

    Hidden entries (``"hidden": True``) are filtered out — they exist in the
    registry only for backend lookups (e.g. ``supports_vision`` for vLLM) and
    are surfaced in the frontend via ``CUSTOM_PROVIDER_PRESETS`` instead of
    the cloud-provider dropdown.
    """
    result = []
    for provider_type, info in PROVIDER_REGISTRY.items():
        if info.get("hidden"):
            continue
        result.append(
            {
                "provider_type": provider_type,
                "display_name": info["display_name"],
                "base_url": info["base_url"],
                "default_models": info["default_models"],
                "supports_streaming": info["supports_streaming"],
                "supports_vision": info.get("supports_vision", False),
                "supports_tool_calling": info.get("supports_tool_calling", False),
                "model_list_mode": info.get("model_list_mode", "remote"),
            }
        )
    return result
