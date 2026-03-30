# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Static registry of supported external LLM providers.

All providers expose OpenAI-compatible /v1/chat/completions endpoints
with Bearer token authentication and SSE streaming support.
"""

from typing import Any

PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "openai": {
        "display_name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o3-mini",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "mistral": {
        "display_name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "default_models": [
            "mistral-large-latest",
            "mistral-small-latest",
            "codestral-latest",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "google": {
        "display_name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "OpenAI compatibility layer (beta). Uses native Google API key as Bearer token.",
    },
    "cohere": {
        "display_name": "Cohere",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "default_models": [
            "command-a-03-2025",
            "command-r-plus-08-2024",
            "command-r-08-2024",
        ],
        "supports_streaming": True,
        "supports_vision": False,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "OpenAI compatibility layer over native Cohere API.",
    },
    "together": {
        "display_name": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "default_models": [
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "Qwen/Qwen3-235B-A22B",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "fireworks": {
        "display_name": "Fireworks AI",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_models": [
            "accounts/fireworks/models/deepseek-v3-0324",
            "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "accounts/fireworks/models/qwen3-30b",
            "accounts/fireworks/models/llama-3.3-70b",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "Model IDs use 'accounts/fireworks/models/' prefix. Usage stats included in streaming.",
    },
    "perplexity": {
        "display_name": "Perplexity",
        "base_url": "https://api.perplexity.ai",
        "default_models": [
            "sonar-pro",
            "sonar",
            "sonar-reasoning",
            "sonar-reasoning-pro",
        ],
        "supports_streaming": True,
        "supports_vision": False,
        "supports_tool_calling": False,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "Web-grounded responses with built-in search.",
    },
    "anthropic": {
        "display_name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "default_models": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "extra_headers": {
            "anthropic-version": "2023-06-01",
        },
        "openai_compatible": False,
        "notes": "Native Anthropic Messages API. Uses x-api-key header and /v1/messages endpoint with SSE translation.",
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
    "openrouter": {
        "display_name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_models": [
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4-5",
            "google/gemini-2.5-flash",
            "meta-llama/llama-4-maverick",
            "mistralai/mistral-small-3.1-24b-instruct",
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
    """Return all registered providers (for the /registry endpoint)."""
    result = []
    for provider_type, info in PROVIDER_REGISTRY.items():
        result.append(
            {
                "provider_type": provider_type,
                "display_name": info["display_name"],
                "base_url": info["base_url"],
                "default_models": info["default_models"],
                "supports_streaming": info["supports_streaming"],
                "supports_vision": info.get("supports_vision", False),
                "supports_tool_calling": info.get("supports_tool_calling", False),
            }
        )
    return result
