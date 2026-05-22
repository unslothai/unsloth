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
        # The remote /v1/models listing returns the full account catalog,
        # including non-chat models (embeddings, TTS, image, moderation,
        # whisper, dall-e) and fine-tunes. Previously we used a hardcoded
        # family allowlist (`gpt-5\.[345]|gpt-4\.5|o3`) which silently
        # dropped every new family OpenAI shipped, including the gpt-5.5
        # generation today. Switch to a non-chat denylist instead so any
        # new chat family auto-appears the moment OpenAI lists it.
        #
        # Pattern strategy:
        # - Unambiguous *feature* indicators (tts/whisper/transcribe/
        #   image/embedding/moderation/sora) match anywhere in the id
        #   with `(?:^|-)` because OpenAI uses them as the primary
        #   qualifier on every variant -- e.g. canonical `tts-1` AND
        #   family variants `gpt-4o-tts`, `gpt-4o-mini-tts`. These
        #   words don't appear in legitimate chat model ids mid-string,
        #   so the mid-id match is intentional and safe.
        # - `audio` and `realtime` are INTENTIONALLY NOT in the
        #   mid-id set. Both the `gpt-audio*` and `gpt-realtime*`
        #   families are chat/responses-capable today -- per OpenAI's
        #   audio and realtime guides they accept text in via
        #   /v1/chat/completions and /v1/responses, not only the
        #   specialised /v1/realtime WebSocket transport. Dropping
        #   them hid supported chat models from the picker. The
        #   audio-only endpoint families (`tts-*`, `whisper-*`,
        #   `*-transcribe`) are still caught above.
        # - `search` is INTENTIONALLY NOT in the mid-id set because
        #   `gpt-4o-search-preview` and `gpt-4o-mini-search-preview` are
        #   chat-with-retrieval models that absolutely belong in the
        #   picker. The standalone search API is caught separately by
        #   `-search-api` (suffix) which never appears on a chat id.
        # - Legacy completion bases (babbage / davinci / ada / curie)
        #   are ^-anchored: they ONLY ever begin a legacy id like
        #   `babbage-002`, `davinci-002`, `text-davinci-003`. A future
        #   hypothetical `gpt-7-davinci-edition` chat model would NOT
        #   be dropped, which is the right default for a denylist.
        # - Prefix-only patterns (`^dall-e`, `^computer-use`, `^ft:`,
        #   `^gpt-image`) cover canonical non-chat families without
        #   risk of mid-id false positives.
        #
        # Verified against the live /v1/models listing 2026-05-22.
        "model_id_denylist": re.compile(
            # Feature suffixes that mark a non-chat variant on any base.
            # Note: `audio` and `realtime` are omitted on purpose --
            # see docstring above.
            r"(?:^|-)(?:embedding|tts|whisper|moderation|image|"
            r"transcribe|sora)\b"
            # Legacy completion bases -- ^-anchored to avoid false
            # positives on hypothetical future chat ids containing
            # those words mid-string.
            r"|^(?:babbage|davinci|ada|curie)\b"
            r"|^text-(?:embedding|moderation|davinci|curie|babbage|ada)\b"
            # Standalone search API (separate endpoint shape).
            # `gpt-4o-search-preview` (chat-with-search) is intentionally
            # NOT matched here.
            r"|-search-api(?:-\d{4}-\d{2}-\d{2})?$"
            # Canonical non-chat prefixes.
            r"|^dall-e\b"
            r"|^computer-use\b"
            # Fine-tunes carry the user's tenant in the id.
            r"|^ft:"
            # Dated snapshot suffixes hide behind the canonical id
            # which is also in the listing.
            r"|-\d{4}-\d{2}-\d{2}$"
        ),
    },
    "anthropic": {
        "display_name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "default_models": [
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
        ],
        # Anthropic's /v1/models returns dated ids for every model in the
        # pre-4.6 generation -- `claude-opus-4-5-20251101`,
        # `claude-haiku-4-5-20251001`, `claude-sonnet-4-5-20250929`,
        # `claude-opus-4-1-20250805`. Per the models overview, those
        # dated ids ARE the canonical names for that generation, not
        # snapshots to hide. Dropping the previous `-\d{8}$` denylist
        # so every live model the API returns reaches the picker.
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
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        # Curated lineup — Google's /v1beta/openai/models returns dozens
        # of historical / experimental / embedding ids. Cap to the current
        # 3.x family plus the rolling `*-latest` aliases.
        "default_models": [
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-pro-latest",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "OpenAI-compatible endpoint. API key from https://aistudio.google.com/apikey.",
        "model_id_allowlist": re.compile(
            r"^(gemini-3\.1-flash-lite|gemini-3-flash-preview|"
            r"gemini-3\.1-pro-preview|gemini-pro-latest|"
            r"gemini-flash-latest|gemini-flash-lite-latest)$"
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
