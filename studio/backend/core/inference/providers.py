# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static registry of supported external LLM providers. All expose OpenAI-compatible
/v1/chat/completions endpoints with Bearer token auth and SSE streaming."""

import ipaddress
import os
import re
import threading
import time
from typing import Any
from urllib.parse import quote, urlsplit

PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "openai_codex": {
        "display_name": "ChatGPT / Codex subscription",
        "base_url": "https://chatgpt.com/backend-api",
        "default_models": [
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-astra",
            "gpt-6-luna",
            "gpt-6-sol",
        ],
        "model_capabilities": {
            "gpt-5.4": {"vision": True, "studio_tools": True},
            "gpt-5.4-mini": {"vision": True, "studio_tools": True},
            "gpt-5.5": {"vision": True, "studio_tools": True},
            "gpt-5.6-luna": {"vision": True, "studio_tools": True},
            "gpt-5.6-sol": {"vision": True, "studio_tools": True},
            "gpt-5.6-terra": {"vision": True, "studio_tools": True},
            "gpt-6-astra": {"vision": True, "studio_tools": True},
            "gpt-6-luna": {"vision": True, "studio_tools": True},
            "gpt-6-sol": {"vision": True, "studio_tools": True},
        },
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_kind": "chatgpt_oauth",
        "base_url_editable": False,
        "model_ids_editable": False,
        "model_list_mode": "curated",
        "notes": "Personal ChatGPT subscription via the Codex Responses endpoint.",
    },
    "openai": {
        "display_name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_models": [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "o3",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "hosted_tools": ("web_search", "code_execution", "image_generation"),
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        # Denylist, not allowlist: new families must appear. Bar is "servable on /v1/responses",
        # the only OpenAI transport. Feature words match mid-id; bases stay ^-anchored.
        "model_id_denylist": re.compile(
            r"(?:^|-)(?:embedding|tts|whisper|moderation|image|"
            r"transcribe|translate|instruct|sora)\b"
            # Chat Completions only per OpenAI model docs; `-search-api` is the standalone endpoint.
            r"|(?:^|-)(?:audio|realtime)\b"
            r"|(?:^|-)search-(?:preview|api)\b"
            # Needs a data source and a Studio turn sends no tools.
            r"|(?:^|-)deep-research\b"
            r"|^o1-(?:mini|preview)\b"
            # Retired canonical id retained by /v1/models.
            r"|^gpt-5\.3$"
            # `^(?:text|code)-` needs the hyphen, so `codex-mini-latest` stays.
            r"|^(?:babbage|davinci|ada|curie)\b"
            r"|^(?:text|code)-(?:embedding|moderation|search|similarity"
            r"|davinci|curie|babbage|ada|cushman)\b"
            r"|^dall-e\b"
            r"|^computer-use\b"
            # Fine-tunes carry the user's tenant in the id.
            r"|^ft:"
            # Snapshots: modern `-YYYY-MM-DD` and legacy `-MMDD` (`gpt-4-1106-preview`).
            r"|-\d{4}-\d{2}-\d{2}$"
            r"|-\d{4}(?:-preview)?$"
        ),
    },
    "anthropic": {
        "display_name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "default_models": [
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
        ],
        # No denylist: `-YYYYMMDD` ids are canonical for pre-4.6 Claude models.
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "hosted_tools": ("web_search", "web_fetch", "code_execution"),
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
        # Native Gemini REST, not OpenAI-compatible; translated in `_stream_gemini`.
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        # Excludes retired gemini-2.0-flash* and gemini-3-pro-preview (redirects to 3.1).
        "default_models": [
            "gemini-3.1-pro-preview",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-pro-latest",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
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
        "studio_tools": True,
        "hosted_tools": ("web_search", "code_execution", "image_generation"),
        "auth_header": "x-goog-api-key",
        "auth_prefix": "",
        "openai_compatible": False,
        "notes": (
            "Native Gemini API. Translation lives in _stream_gemini. "
            "API key from https://aistudio.google.com/apikey. "
            "See https://ai.google.dev/gemini-api/docs for endpoint shapes."
        ),
        "model_id_deny_exact": ("gemini-3-pro-preview",),
        # `-preview` optional on image ids so a GA rollover does not drop them.
        "model_id_allowlist": re.compile(
            r"^("
            r"gemini-3\.6-(?:flash|pro)(?:-preview)?|"
            r"gemini-3\.5-(?:flash|pro|flash-lite)(?:-preview)?|"
            r"gemini-3\.1-(?:flash|pro|flash-lite)(?:-preview)?(?:-customtools)?|"
            r"gemini-3\.1-flash-image(?:-preview)?|"
            r"gemini-3-(?:flash|pro)(?:-preview)?|"
            r"gemini-3-pro-image(?:-preview)?|"
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
            "deepseek-flash",
            "deepseek-v4-pro",
        ],
        "supports_streaming": True,
        "supports_vision": False,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "OpenAI-compatible API. deepseek-flash and deepseek-v4-pro take thinking on/off plus reasoning_effort low/high/max.",
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
        "studio_tools": True,
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
        "default_models": [
            "kimi-k2.6",
            "kimi-k2.5",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "hosted_tools": ("web_search",),
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "Moonshot API key. China: use base URL https://api.moonshot.cn/v1",
        "model_id_allowlist": re.compile(r"^kimi-k2\.[56]$"),
        # Reasoning-class: the API rejects custom temperature/top_p ("only 1 is allowed").
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
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "DashScope API key. China mainland: override base URL to https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "huggingface": {
        "display_name": "Hugging Face",
        "base_url": "https://router.huggingface.co/v1",
        "default_models": [
            "openai/gpt-oss-120b",
            "deepseek-ai/DeepSeek-V3",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": (
            "HF token from huggingface.co/settings/tokens. Uses the "
            "OpenAI-compatible router at /v1/chat/completions; /v1/models "
            "returns the cross-provider chat catalog. See "
            "https://huggingface.co/docs/inference-providers/index."
        ),
        "model_list_mode": "remote",
        "model_id_allowlist": re.compile(
            r"^(openai|deepseek-ai|google|meta-llama|Qwen|moonshotai|mistralai|zai-org)/"
        ),
        "model_id_limit": 15,
    },
    "vllm": {
        "display_name": "vLLM",
        "base_url": "",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        # vLLM's /v1/responses re-templates messages and 400s on strict-alternation templates.
        "notes": "Self-hosted vLLM server. Always routed to /v1/chat/completions.",
        "supports_chat_template_kwargs": True,
        "hidden": True,
    },
    "custom": {
        "display_name": "Custom",
        "base_url": "",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": (
            "User-supplied OpenAI-compatible server. Routed to "
            "/v1/chat/completions; /models is optional."
        ),
        # A strict gateway 400s on unknown keys; a base_url says nothing about what serves it.
        "body_omit": ("top_k", "min_p", "repetition_penalty"),
        "hidden": True,
    },
    "ollama": {
        "display_name": "Ollama",
        "base_url": "http://localhost:11434/v1",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": (
            "Ollama server (local or cloud). OpenAI-compatible "
            "/v1/chat/completions; API key optional (required by Ollama "
            "cloud). Surfaced via CUSTOM_PROVIDER_PRESETS in the frontend."
        ),
        # Ollama's /v1 silently drops these (native /api/chat options); this route is public.
        "body_omit": ("top_k", "min_p", "repetition_penalty"),
        "hidden": True,
    },
    "llama_cpp": {
        "display_name": "llama.cpp",
        "base_url": "http://localhost:8080/v1",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": (
            "Local llama.cpp server (llama-server). OpenAI-compatible "
            "/v1/chat/completions. Surfaced via CUSTOM_PROVIDER_PRESETS."
        ),
        # llama-server reads chat_template_kwargs when started with --jinja, and ignores it otherwise.
        "supports_chat_template_kwargs": True,
        "hidden": True,
    },
    "lemonade": {
        "display_name": "AMD NPU (FastFlowLM)",
        "base_url": "",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "Unsloth-managed Lemonade serving FastFlowLM on the AMD NPU.",
        # FastFlowLM 1.0.3 parses min_p into an integer, so 0.05 arrives as 0.
        "body_omit": ("min_p",),
        "hidden": True,
        "managed": True,
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
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
        "studio_tools": True,
        "hosted_tools": ("web_search",),
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {
            "HTTP-Referer": "https://unsloth.ai",
            "X-Title": "Unsloth Studio",
        },
        "notes": "Unified gateway to 300+ models across all major providers. HTTP-Referer and X-Title headers sent for attribution.",
        "model_list_mode": "curated",
    },
    "typesafe": {
        "display_name": "TypeSafe",
        "base_url": "https://api.typesafe.ai/v1",
        "default_models": ["jev-latest", "jev-1.13"],
        "supports_streaming": False,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "System One decision models. Used by the Decision API, never by chat.",
        "model_list_mode": "curated",
        "decisions_only": True,
    },
    "liquid": {
        "display_name": "Liquid AI",
        "base_url": "https://api.liquid.ai/decisions/v1",
        "default_models": ["d1:free"],
        "supports_streaming": False,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "System One decision models. Used by the Decision API, never by chat.",
        "model_list_mode": "curated",
        "decisions_only": True,
    },
}


def get_provider_info(provider_type: str) -> dict[str, Any] | None:
    return PROVIDER_REGISTRY.get(provider_type)


def get_connectable_provider_info(provider_type: str) -> dict[str, Any] | None:
    """Return a user-configurable provider, excluding Studio-managed runtimes."""
    info = PROVIDER_REGISTRY.get(provider_type)
    return None if info is None or info.get("managed") else info


def answers_decisions_only(provider_type: str | None, api_type: str | None = None) -> bool:
    info = PROVIDER_REGISTRY.get(provider_type) if isinstance(provider_type, str) else None
    return bool(info and info.get("decisions_only")) or (
        provider_type == "custom" and api_type == "systemone"
    )


def get_base_url(provider_type: str) -> str | None:
    info = PROVIDER_REGISTRY.get(provider_type)
    return info["base_url"] if info else None


def provider_runs_local_tools(provider_type: str | None) -> bool:
    """Whether Unsloth may run its own tool loop against this provider type.

    Unsloth's tools (web_search, python, terminal, MCP, knowledge-base search) execute on the
    Unsloth host, so any provider whose wire format can carry a tool schema out and a tool result
    back can use them: the whole OpenAI-compatible family plus Gemini and Anthropic, whose
    native shapes are translated to and from OpenAI chunks in ``external_provider.py``.
    """
    # isinstance: a list/dict from the body would raise TypeError (500 instead of 400).
    if not isinstance(provider_type, str):
        return False
    info = PROVIDER_REGISTRY.get(provider_type)
    return bool(info and info.get("studio_tools"))


def provider_model_runs_local_tools(provider_type: str | None, model: str | None) -> bool:
    """``provider_runs_local_tools`` narrowed to one model.

    Gemini's image models are the exception the provider-wide flag cannot
    express: ``_stream_gemini`` sets ``text_tools_allowed = False`` for them and
    emits no ``functionDeclarations``, so the catalog never reaches the model.
    Advertising the capability there lights up the MCP and Docs pills and then
    completes the turn as if the user had selected nothing, which is worse than
    not offering them.
    """
    if not provider_runs_local_tools(provider_type):
        return False
    if provider_type == "gemini" and isinstance(model, str):
        # Kept in step with _stream_gemini.
        model_lc = model.lower()
        if "-image" in model_lc or "nano-banana" in model_lc:
            return False
    return True


def provider_hosted_tools(provider_type: str | None) -> frozenset[str]:
    """Built-in tool names this provider executes on its own side.

    These are not Unsloth's tools: they are body flags (`tools: [{type:
    "web_search"}]`, `plugins: [{id: "web"}]`, `codeExecution`) that the provider
    runs and bills, and the only thing this server does with them is forward the
    name. `provider_runs_local_tools` is orthogonal -- most providers do both,
    and a request picks a side by which names it lists.

    Empty for the self-hosted presets (llama.cpp, vLLM, Ollama, custom) and for
    openai_codex, whose `web_search` is Unsloth's own tool run by the Codex loop.
    """
    if not isinstance(provider_type, str):
        return frozenset()
    info = PROVIDER_REGISTRY.get(provider_type)
    return frozenset(info.get("hosted_tools") or ()) if info else frozenset()


# Mirrored on the frontend as _SERVER_SIDE_BUILTIN_TOOL_NAMES.
HOSTED_TOOL_NAMES: frozenset[str] = frozenset(
    name for info in PROVIDER_REGISTRY.values() for name in (info.get("hosted_tools") or ())
)


# If a request names both sides the local tool wins. code_execution has no local
# implementation, so it is never treated as replaced.
LOCAL_STANDINS_FOR_HOSTED_TOOLS: dict[str, frozenset[str]] = {
    "web_search": frozenset({"web_search"}),
    "code_execution": frozenset({"python", "terminal"}),
}


def hosted_only_tools(provider_type: str | None, enabled_tools: Any) -> list[str]:
    """The requested hosted tools Unsloth is not running in their place.

    image_generation and web_fetch have no local implementation, and their UI
    pills are independent of Search / Code / RAG, so a request that mixes one of
    them with an Unsloth tool has to carry it through to the provider or the tool
    silently disappears while its toggle stays on. code_execution has no local
    implementation either, and rides along unless the same request also asked
    for the local tools that would duplicate it.
    """
    if not isinstance(enabled_tools, list):
        return []
    hosted = provider_hosted_tools(provider_type)
    requested = list(dict.fromkeys(n for n in enabled_tools if isinstance(n, str)))
    requested_set = set(requested)
    return [
        name
        for name in requested
        if name in hosted
        and not (LOCAL_STANDINS_FOR_HOSTED_TOOLS.get(name, frozenset()) & requested_set)
    ]


# Cloud metadata hosts would leak instance credentials. Keep in sync with core/inference/tools.py.
_METADATA_HOST_NAMES = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "metadata.tencentyun.com",
        "instance-data.ec2.internal",
    }
)
# Held parsed so every spelling of the same address matches.
_METADATA_IPS = frozenset(
    ipaddress.ip_address(address)
    for address in (
        "169.254.169.254",
        "169.254.169.252",
        "169.254.170.2",
        "169.254.170.23",
        "fd00:ec2::254",
        "fd20:ce::254",
        "100.100.100.200",
        "100.100.100.110",
        # metadata.tencentyun.com; listed since the resolved-address check reads this set.
        "169.254.0.23",
        "169.254.10.10",
    )
)
# Matched as a network so names like 169.254.gateway.example.com are not mistaken.
_METADATA_NETWORK = ipaddress.ip_network("169.254.0.0/16")

# Opt-in: loopback/LAN endpoints (Ollama, llama.cpp, vLLM) are the normal case.
_BLOCK_PRIVATE_ENV = "UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS"

MANAGED_PRIVATE_URL_HINT = (
    " The installation owner can allow private and LAN addresses in Settings > Accounts."
)
MANAGED_PUBLIC_ONLY_TEXT = "Managed accounts may only use public-network provider base URLs."


def managed_private_url_hint() -> str:
    """The hint, omitted when the environment lock means the owner cannot act on it either."""
    return "" if os.environ.get(_BLOCK_PRIVATE_ENV) == "1" else MANAGED_PRIVATE_URL_HINT


def managed_public_only_reason() -> str:
    return MANAGED_PUBLIC_ONLY_TEXT + managed_private_url_hint()


# Resolvers accept decimal/octal/hex IPv4 (0xA9FEA9FE) that `ipaddress` reads as names.
_NUMERIC_HOST_PART = re.compile(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)")

# httpx IDNA-splits on these, so http://169。254。169。254/ reaches 169.254.169.254.
_IDNA_DOTS = str.maketrans({"。": ".", "．": ".", "｡": "."})


def _canonical_host(hostname: str) -> str:
    """Return the dotted-quad form of a numeric host, else ``hostname``."""
    parts = hostname.split(".")
    if len(parts) > 4 or not all(_NUMERIC_HOST_PART.fullmatch(part) for part in parts):
        return hostname
    import socket

    try:
        # inet_aton parses every legacy spelling and never touches DNS.
        return socket.inet_ntoa(socket.inet_aton(hostname))
    except OSError:
        return hostname


def _metadata_host(hostname: str) -> bool:
    """True when ``hostname`` names a cloud metadata service."""
    # Strip the IPv6 scope id: it dials the same host but compares unequal.
    hostname = hostname.translate(_IDNA_DOTS).rstrip(".").split("%")[0]
    hostname = _canonical_host(hostname)
    if hostname in _METADATA_HOST_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    # ::ffff:169.254.169.254 and 2002:a9fe:a9fe:: reach the same service.
    for candidate in (getattr(ip, "ipv4_mapped", None), getattr(ip, "sixtofour", None)):
        if candidate is not None and _metadata_host(candidate.compressed):
            return True
    return ip in _METADATA_IPS or (ip.version == 4 and ip in _METADATA_NETWORK)


# Resolve names to catch DNS aliases of metadata IPs. Short: sync validator in async handlers
# blocks the event loop; a timeout counts as unknown (allowed on the default path).
_DNS_TIMEOUT_SECONDS = 0.5
_DNS_CACHE_TTL_SECONDS = 300.0
_DNS_CACHE_MAX_ENTRIES = 512
# Only answers are cached; failures and timeouts are cheap to repeat and wrong to remember.
_dns_cache: dict[str, tuple[float, tuple[str, ...]]] = {}
_dns_cache_lock = threading.Lock()
# Timed-out lookups are abandoned, not cancelled; cap threads in flight.
_DNS_MAX_IN_FLIGHT = 32
_dns_in_flight = threading.BoundedSemaphore(_DNS_MAX_IN_FLIGHT)

# Hard-coded destinations, not caller-controlled, so learning their addresses buys nothing.
_REGISTRY_HOSTNAMES = frozenset(
    host
    for host in (
        urlsplit(info["base_url"]).hostname
        for info in PROVIDER_REGISTRY.values()
        if info.get("base_url")
    )
    if host
)


def _public_registry_hostname(host: str) -> bool:
    """A shipped public vendor hostname, usable by a managed account without a lookup."""
    host = (host or "").lower().rstrip(".")
    if host not in _REGISTRY_HOSTNAMES:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        return ipaddress.ip_address(_canonical_host(host)).is_global
    except ValueError:
        return True


def _metadata_address(address: str) -> bool:
    """``_metadata_host`` for a RESOLVED address: the exact services, no net.

    The 169.254.0.0/16 clause exists to catch a caller who types a link-local
    address at the metadata service directly, and it stays for that. It cannot
    be applied to a resolved address: 169.254/16 is the general IPv4 link-local
    range (RFC 3927), so a self-assigned host, an mDNS .local name on a network
    without DHCP, or a captive portal answering every query would all read as
    the metadata service. Those are refused today by nobody, and this change is
    not the place to start.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    for candidate in (getattr(ip, "ipv4_mapped", None), getattr(ip, "sixtofour", None)):
        if candidate is not None and _metadata_address(candidate.compressed):
            return True
    return ip in _METADATA_IPS


# Kept in step with httpx `_urlparse.encode_host`.
_HOST_SAFE_CHARS = "!$&'()*+,;=" + '"`{}%|\\'


def _transport_host(hostname: str) -> str:
    """The ASCII host httpx will dial, so the checked name is the dialled name.

    ``socket.getaddrinfo`` encodes a Unicode host with the stdlib ``idna`` codec
    (IDNA 2003), httpx with the ``idna`` package (IDNA 2008), and the two differ
    on the deviation characters: straße.de resolves as strasse.de through the
    resolver and as xn--strae-oqa.de through httpx, which are different hosts
    owned by different people. Mirrors httpx's `_urlparse.encode_host`.
    """
    try:
        # An address is dialled as written; quoting one would corrupt IPv6.
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass
    if hostname.isascii():
        # Match httpx percent-encoding so we resolve the same name it dials.
        return quote(hostname.lower(), safe = _HOST_SAFE_CHARS)
    try:
        import idna
        return idna.encode(hostname.lower()).decode("ascii")
    except Exception:
        return hostname


def _cached_addresses(hostname: str) -> tuple[str, ...] | None:
    """What ``_resolve_host`` already learned about ``hostname``, if anything."""
    now = time.monotonic()
    with _dns_cache_lock:
        cached = _dns_cache.get(_transport_host(hostname))
    return cached[1] if cached is not None and cached[0] > now else None


def _resolve_host(hostname: str, port: int | None, scheme: str) -> tuple[str, ...] | None:
    """Addresses for ``hostname``, or ``None`` when the resolver did not answer.

    One lookup and one cache entry serve both callers, which read "no answer" in
    opposite directions: the metadata check has nothing to refuse, the opt-in
    private-address check refuses.
    """
    import socket

    hostname = _transport_host(hostname)
    cached = _cached_addresses(hostname)
    if cached is not None:
        return cached
    now = time.monotonic()

    # Local binding: an abandoned worker may outlive the global; BoundedSemaphore would raise.
    in_flight = _dns_in_flight
    if not in_flight.acquire(timeout = _DNS_TIMEOUT_SECONDS):
        return None

    resolved: list[str] = []
    answered = False

    def _resolve() -> None:
        nonlocal answered
        try:
            infos = socket.getaddrinfo(
                hostname,
                port or (443 if scheme == "https" else 80),
                type = socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError, ValueError):
            return
        finally:
            # Released by the worker, so an abandoned lookup frees its slot only when the resolver does.
            in_flight.release()
        resolved.extend(str(info[4][0]) for info in infos)
        answered = True

    thread = threading.Thread(target = _resolve, daemon = True)
    thread.start()
    thread.join(_DNS_TIMEOUT_SECONDS)
    if thread.is_alive():
        # Not cached: a deliberately slow server could still answer the transport after the deadline.
        return None
    if not answered:
        # Not cached: one transient SERVFAIL would become minutes of refusal on the opt-in path.
        return None

    addresses = tuple(resolved)
    with _dns_cache_lock:
        if len(_dns_cache) >= _DNS_CACHE_MAX_ENTRIES:
            _dns_cache.clear()
        _dns_cache[hostname] = (now + _DNS_CACHE_TTL_SECONDS, addresses)
    return addresses


def _resolves_to_metadata(hostname: str, port: int | None, scheme: str) -> bool:
    """True when ``hostname`` resolves to a cloud metadata address."""
    hostname = hostname.translate(_IDNA_DOTS).rstrip(".").split("%")[0]
    if not hostname or hostname in _REGISTRY_HOSTNAMES:
        return False
    try:
        ipaddress.ip_address(_canonical_host(hostname))
        return False
    except ValueError:
        pass
    return any(
        _metadata_address(address) for address in _resolve_host(hostname, port, scheme) or ()
    )


def _managed_account_caller() -> bool:
    """True when this validation runs for a managed (non-owner) account."""
    from utils.account_context import is_owner_context
    return not is_owner_context()


def _managed_private_urls_allowed() -> bool:
    """True when the owner has opened private provider addresses to managed accounts.

    Imported here, not at module scope: tests/test_provider_base_url_validation.py loads this
    module standalone.
    """
    from utils.managed_provider_url_settings import get_managed_private_provider_urls_allowed
    return get_managed_private_provider_urls_allowed()


def _reject_non_public(hostname: str, port: int | None, scheme: str, reason: str) -> None:
    """Raise when ``hostname`` is, or resolves to, a non-public address."""
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        # Reuse the metadata check's cached answer instead of a second bounded lookup.
        resolved = _cached_addresses(hostname)
        if resolved is None:
            # Unbounded fallback: slow resolvers are normal (5s x 2), and here no answer fails closed.
            import socket
            try:
                infos = socket.getaddrinfo(
                    _transport_host(hostname),
                    port or (443 if scheme == "https" else 80),
                    type = socket.SOCK_STREAM,
                )
            except (OSError, UnicodeError) as exc:
                raise ValueError("Provider base URL hostname could not be resolved.") from exc
            resolved = tuple(str(info[4][0]) for info in infos)
        addresses = [ipaddress.ip_address(address.split("%", 1)[0]) for address in resolved]
    if not addresses or any(not ip.is_global for ip in addresses):
        raise ValueError(reason)


def public_provider_address(url: str) -> str:
    """Resolve ``url``'s host now and return one public address to dial, or raise ``ValueError``.

    Re-resolving per connection stops a name rebinding to loopback or the LAN after the
    cached check.
    """
    import socket

    parts = urlsplit(url)
    hostname = (parts.hostname or "").rstrip(".")
    if not hostname:
        raise ValueError("Provider URL must contain a hostname.")
    reason = managed_public_only_reason()
    try:
        addresses = [ipaddress.ip_address(_canonical_host(hostname))]
    except ValueError:
        try:
            infos = socket.getaddrinfo(
                _transport_host(hostname),
                parts.port or (443 if parts.scheme == "https" else 80),
                type = socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError) as exc:
            raise ValueError("Provider base URL hostname could not be resolved.") from exc
        addresses = [ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]) for info in infos]
    if not addresses or any(not ip.is_global for ip in addresses):
        raise ValueError(reason)
    return str(addresses[0])


METADATA_REFUSED_REASON = "Cloud metadata endpoints cannot be used as a provider base URL."


def provider_address_excluding_metadata(url: str) -> str:
    """Resolve ``url``'s host now and return one address to dial, refusing only metadata services.

    For a caller the owner has allowed private addresses. Re-resolving rather than trusting the
    save-time check is the point: a name is free to answer 169.254.169.254 afterwards.
    """
    import socket

    parts = urlsplit(url)
    hostname = (parts.hostname or "").rstrip(".")
    if not hostname:
        raise ValueError("Provider URL must contain a hostname.")
    literal = True
    try:
        addresses = [ipaddress.ip_address(_canonical_host(hostname))]
    except ValueError:
        literal = False
        try:
            infos = socket.getaddrinfo(
                _transport_host(hostname),
                parts.port or (443 if parts.scheme == "https" else 80),
                type = socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError) as exc:
            raise ValueError("Provider base URL hostname could not be resolved.") from exc
        addresses = [ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]) for info in infos]
    if not addresses:
        raise ValueError("Provider base URL hostname could not be resolved.")
    # Literals: all of 169.254.0.0/16 is metadata; DNS answers there may be mDNS/self-assigned.
    refuses = _metadata_host if literal else _metadata_address
    if any(refuses(str(ip)) for ip in addresses):
        raise ValueError(METADATA_REFUSED_REASON)
    return str(addresses[0])


def validate_provider_base_url(base_url: str) -> str:
    """Return a normalized provider base URL, or raise ``ValueError``.

    The backend issues outbound requests to this URL with the caller's decrypted
    API key attached, so it is caller-controlled server-side egress. Only shapes
    that can never be a real provider endpoint are refused: a non-http(s) scheme,
    control characters, a missing host, and cloud metadata services. Plain http,
    loopback, LAN hosts, odd ports, query strings and basic-auth userinfo all
    stay valid -- Ollama, llama.cpp, vLLM and custom gateways rely on them. A
    caller-supplied hostname is resolved far enough to apply the metadata block
    to DNS aliases of it; rejecting other private addresses stays opt-in for the
    owner, and is on for a managed account until the owner turns it off for the
    installation (``utils.managed_provider_url_settings``), which is how a team
    sharing one LAN model server gets to use it from more than one account.

    Normalization is strip + trailing-slash removal only (what the client did
    before), so validating an already-validated URL returns it unchanged.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Provider base URL is required.")

    raw = base_url.strip()
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in raw) or "\\" in raw:
        raise ValueError("Provider base URL contains invalid characters.")

    try:
        parts = urlsplit(raw)
        port = parts.port
        hostname = parts.hostname
    except ValueError as exc:
        raise ValueError("Provider base URL is malformed.") from exc

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("Provider base URL must use http or https.")
    # Checks read the parsed hostname, so http://api.openai.com@169.254.169.254/ is caught.
    if not hostname:
        raise ValueError("Provider base URL must contain a hostname.")

    hostname = hostname.rstrip(".")
    if _metadata_host(hostname) or _resolves_to_metadata(hostname, port, scheme):
        raise ValueError("Cloud metadata endpoints cannot be used as a provider base URL.")

    if os.environ.get(_BLOCK_PRIVATE_ENV) == "1":
        _reject_non_public(
            hostname,
            port,
            scheme,
            "Provider base URL points at a private address, which is disabled on this "
            f"server ({_BLOCK_PRIVATE_ENV}=1).",
        )
    elif (
        _managed_account_caller()
        and not _public_registry_hostname(hostname)
        and not _managed_private_urls_allowed()
    ):
        _reject_non_public(hostname, port, scheme, managed_public_only_reason())

    return raw.rstrip("/")


def list_available_providers(
    include_hidden: bool = False, include_oauth: bool = False
) -> list[dict[str, Any]]:
    """Return registered providers (for the /registry endpoint).

    Hidden entries exist only for backend lookups and are surfaced by the UI via
    ``CUSTOM_PROVIDER_PRESETS`` instead of the dropdown, so they stay filtered out by default. That
    default is load-bearing for upgrades: a browser holding a cached bundle from before this
    capability existed has no idea to filter on ``hidden``, and would render the self-hosted presets
    as duplicate dropdown entries.

    ``include_hidden`` is how a client that does know says so. The self-hosted presets are exactly
    the ones that run Unsloth's tools, so their capability has to reach a frontend that asks for it,
    and asking is opt-in.

    OAuth rows are opt-in too: a pre-OAuth bundle (v0.1.701-beta, bare request) renders them as an
    API-key form the backend then rejects (#8722). Every bundle sending ``include_hidden`` already
    renders OAuth, so either flag opts in.
    """
    result = []
    for provider_type, info in PROVIDER_REGISTRY.items():
        if (info.get("hidden") and not include_hidden) or info.get("managed"):
            continue
        if info.get("auth_kind") == "chatgpt_oauth" and not (include_hidden or include_oauth):
            continue
        result.append(
            {
                "provider_type": provider_type,
                "display_name": info["display_name"],
                "base_url": info["base_url"],
                "default_models": info["default_models"],
                "model_capabilities": info.get("model_capabilities", {}),
                "supports_streaming": info["supports_streaming"],
                "supports_vision": info.get("supports_vision", False),
                "supports_tool_calling": info.get("supports_tool_calling", False),
                "supports_studio_tools": bool(info.get("studio_tools")),
                "hidden": bool(info.get("hidden")),
                "model_list_mode": info.get("model_list_mode", "remote"),
                "auth_kind": info.get("auth_kind", "api_key"),
                "base_url_editable": info.get("base_url_editable", True),
                "model_ids_editable": info.get("model_ids_editable", True),
            }
        )
    return result
