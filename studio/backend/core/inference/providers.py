# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Static registry of supported external LLM providers.

All providers expose OpenAI-compatible /v1/chat/completions endpoints
with Bearer token auth and SSE streaming.
"""

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
        # Only seeds the picker; /codex/models is the truth once connected.
        "default_models": [
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
        ],
        "model_capabilities": {
            "gpt-5.4": {"vision": True, "studio_tools": True},
            "gpt-5.4-mini": {"vision": True, "studio_tools": True},
            "gpt-5.5": {"vision": True, "studio_tools": True},
            "gpt-5.6-luna": {"vision": True, "studio_tools": True},
            "gpt-5.6-sol": {"vision": True, "studio_tools": True},
            "gpt-5.6-terra": {"vision": True, "studio_tools": True},
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
        # Server tools OpenAI runs itself on /v1/responses (cloud base URL only;
        # `_stream_openai_responses` re-checks that). See `hosted_tools` below.
        "hosted_tools": ("web_search", "code_execution", "image_generation"),
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        # Scope the picker to the current generation. /v1/models returns many
        # historical snapshots, fine-tunes, and non-chat models we don't want.
        "model_id_allowlist": re.compile(r"^(gpt-5\.[3456]|gpt-4\.5|o3)(?:[-.]|$)"),
        # Hide dated snapshots and the retired plain gpt-5.3 id.
        "model_id_denylist": re.compile(r"^(gpt-5\.3)$|-\d{4}-\d{2}-\d{2}$"),
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
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
        # Hide YYYYMMDD-suffixed snapshot ids (e.g. claude-3-5-sonnet-20241022).
        "model_id_denylist": re.compile(r"-\d{8}$"),
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": False,
        # Anthropic's own server tools, appended by `_stream_anthropic`.
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
        # Native Gemini REST endpoint -- does NOT speak OpenAI Chat Completions;
        # translated in `_stream_gemini` (external_provider.py).
        # https://ai.google.dev/gemini-api/docs
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        # Curated lineup (ListModels returns many historical/experimental ids).
        # Excluded on purpose:
        #   - `gemini-2.0-flash*` (retired 2026-06-01; 404 on use)
        #   - `gemini-3-pro-preview` (shut down 2026-03-09; auto-redirects to
        #     `gemini-3.1-pro-preview`, so we surface 3.1 directly).
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
        # googleSearch / codeExecution grounding and the Nano Banana image path,
        # all wired natively in `_stream_gemini`.
        "hosted_tools": ("web_search", "code_execution", "image_generation"),
        # Native API takes the bare key on `x-goog-api-key`.
        "auth_header": "x-goog-api-key",
        "auth_prefix": "",
        "openai_compatible": False,
        "notes": (
            "Native Gemini API. Translation lives in _stream_gemini. "
            "API key from https://aistudio.google.com/apikey. "
            "See https://ai.google.dev/gemini-api/docs for endpoint shapes."
        ),
        # gemini-3-pro-preview was shut down 2026-03-09 and auto-aliased to
        # gemini-3.1-pro-preview; drop it so users see one canonical card.
        "model_id_deny_exact": ("gemini-3-pro-preview",),
        # Chat-capable 3.6 / 3.5 / 3.1 / 3 / 2.5 families plus rolling *-latest
        # aliases. Image-tier ids flow through the Nano Banana
        # `responseModalities` path in `_stream_gemini`. Retired 2.0 ids
        # excluded (they 404 on use). `-preview` is optional on the image ids
        # so a GA rollover does not drop them from the picker.
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
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "supports_streaming": True,
        "supports_vision": False,
        "supports_tool_calling": True,
        "studio_tools": True,
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
        # Surface only the two SoTA multimodal models (kimi-k2.6/k2.5);
        # moonshot-v1-* and dated k2 previews are filtered by the allowlist.
        # Docs: https://platform.kimi.ai/docs/models
        # Listing/overview: https://platform.kimi.ai/docs/api/list-models
        #                   https://platform.kimi.ai/docs/api/overview
        "default_models": [
            "kimi-k2.6",
            "kimi-k2.5",
        ],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        # `$web_search` builtin_function, driven by `_stream_kimi_web_search`.
        "hosted_tools": ("web_search",),
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "notes": "Moonshot API key. China: use base URL https://api.moonshot.cn/v1",
        "model_id_allowlist": re.compile(r"^kimi-k2\.[56]$"),
        # Reasoning-class: the API rejects custom temperature/top_p ("only 1
        # is allowed"). Strip both so the server uses its required defaults.
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
        # Seed the picker before the live /v1/models call resolves; the remote
        # listing (see model_list_mode) is the source of truth.
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
        # Remote so users see live availability; loadModels() merges defaults
        # so they stay visible if the remote call fails.
        "model_list_mode": "remote",
        # Scope to trusted first-party org repos (the response is otherwise
        # hundreds of community fine-tunes, mirrors, fp8 variants).
        "model_id_allowlist": re.compile(
            r"^(openai|deepseek-ai|google|meta-llama|Qwen|moonshotai|mistralai|zai-org)/"
        ),
        # Cap the post-filter list to first N matches (no server-side sort);
        # default_models keeps flagship ids near the top.
        "model_id_limit": 15,
    },
    "vllm": {
        "display_name": "vLLM",
        # User-supplied via provider_base_url; the route falls back to the
        # payload's base_url when the registry entry has none.
        "base_url": "",
        "default_models": [],
        "supports_streaming": True,
        "supports_vision": True,
        "supports_tool_calling": True,
        "studio_tools": True,
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        # Force /v1/chat/completions -- vLLM's /v1/responses rebuilds messages
        # through the chat template, 400ing on strict-alternation templates
        # (Gemma 3). The chat-completions path takes messages verbatim.
        "notes": "Self-hosted vLLM server. Always routed to /v1/chat/completions.",
        # Surfaced via the frontend's CUSTOM_PROVIDER_PRESETS, not the dropdown.
        "hidden": True,
    },
    "custom": {
        "display_name": "Custom",
        # User-supplied via provider_base_url.
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
        # Surfaced by the frontend's generic Custom option, not the dropdown.
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
        "hidden": True,
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        # Curated picker list (locked, not live /models).
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
        # The router's universal web plugin (`plugins: [{id: "web"}]`).
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
}


def get_provider_info(provider_type: str) -> dict[str, Any] | None:
    """Return the registry entry for a provider type, or None if unknown."""
    return PROVIDER_REGISTRY.get(provider_type)


def get_base_url(provider_type: str) -> str | None:
    """Return the default base URL for a provider type."""
    info = PROVIDER_REGISTRY.get(provider_type)
    return info["base_url"] if info else None


def provider_runs_local_tools(provider_type: str | None) -> bool:
    """Whether Unsloth may run its own tool loop against this provider type.

    Unsloth's tools (web_search, python, terminal, MCP, knowledge-base search)
    execute on the Unsloth host, so any provider whose wire format can carry a
    tool schema out and a tool result back can use them. That is the whole
    OpenAI-compatible family plus Gemini, whose native shape is translated to
    and from OpenAI chunks in ``external_provider.py``.

    Anthropic is deliberately absent: ``_stream_anthropic`` only appends
    Anthropic's own hosted builtins and never forwards a caller's function-tool
    schemas, so the loop would advertise a catalog the model never sees.
    Enabling it needs OpenAI -> Anthropic schema translation plus tool_use /
    tool_result message replay, which is separate work. Anthropic keeps its
    hosted web_search, web_fetch and code_execution meanwhile.
    """
    # isinstance, not a truthiness check: the value reaches here straight from a
    # request body, and a list or dict key raises TypeError inside dict.get, which
    # would turn malformed input into a 500 instead of the caller's 400.
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
        # Same test _stream_gemini applies, kept in step with it deliberately.
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


# The whole server-side builtin vocabulary, derived from the registry so a new
# provider entry extends it by declaring its own tools. Mirrored on the frontend
# as _SERVER_SIDE_BUILTIN_TOOL_NAMES (external_provider.py) for card labelling.
HOSTED_TOOL_NAMES: frozenset[str] = frozenset(
    name for info in PROVIDER_REGISTRY.values() for name in (info.get("hosted_tools") or ())
)


# Local tool names that stand in for a hosted one, per hosted name. A request
# naming BOTH sides is asking for one tool twice, so the local side wins and the
# hosted name is not forwarded: the alternative runs the provider's copy as well
# and bills for it.
#
# Which side a request wants is the request's to say, not this server's to
# assume. web_search is unambiguous -- the hosted name and Unsloth's own tool are
# spelled the same, so naming it while the loop runs can only mean Unsloth's.
# code_execution is a different name from python/terminal precisely because it
# is a different thing: it runs in the provider's sandbox, and Unsloth has no
# implementation of it at all (see ALL_TOOLS). Treating it as "already replaced"
# therefore substitutes nothing, it just drops the tool while its pill stays lit.
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


# Cloud-metadata hosts. The backend fetches the base URL on the caller's behalf,
# so one of these would hand instance credentials to whoever asked, and no LLM
# endpoint lives here. Refused on every deployment. Keep in sync with the
# tool-approval gate's list in core/inference/tools.py (function-local there).
_METADATA_HOST_NAMES = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "metadata.tencentyun.com",
        "instance-data.ec2.internal",
    }
)
# Held parsed, so every spelling of the same address matches: fd00:ec2::254,
# fd00:0ec2:0000:0000:0000:0000:0000:0254 and fd00:ec2::0.0.2.84 are one host.
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
        # metadata.tencentyun.com, on VPC and on the classic network. Listed
        # exactly because the resolved-address check reads this set rather than
        # the link-local network below.
        "169.254.0.23",
        "169.254.10.10",
    )
)
# Link-local, where the IPv4 metadata services live. Matched as a network so a
# DNS name that merely starts with those digits (169.254.gateway.example.com) is
# not mistaken for one.
_METADATA_NETWORK = ipaddress.ip_network("169.254.0.0/16")

# Opt-in for operators who expose Unsloth on a shared host: also refuse provider
# URLs that resolve to a non-public address. Off by default, because loopback and
# LAN endpoints are the normal case (Ollama, llama.cpp, vLLM, custom gateways).
_BLOCK_PRIVATE_ENV = "UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS"


# An all-numeric host is an IPv4 literal to the resolver, in decimal, octal or
# hex. `ipaddress` parses only the dotted quad, so 2852039166, 0xA9FEA9FE and
# 0251.0376.0251.0376 would read as names while getaddrinfo returns
# 169.254.169.254.
_NUMERIC_HOST_PART = re.compile(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)")

# IDNA label separators. httpx encodes a host through idna, which splits on all
# of these, so http://169。254。169。254/ reaches 169.254.169.254.
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
    # An IPv6 scope id (fd00:ec2::254%250 once the transport decodes it) keeps
    # the address unequal to the unscoped entry while dialling the same host.
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


# The block above only reads the hostname text, so a caller-controlled name
# (metadata-alias.attacker.test IN A 169.254.169.254) dials the very service it
# exists to refuse. Names are resolved on the default path too, but only far
# enough to answer "is this metadata"; refusing other private addresses stays
# opt-in. Three things keep that lookup off the endpoints people configure:
#   * registry hosts and IP literals skip it, so a real provider or
#     http://127.0.0.1:11434 touches no resolver;
#   * a name that does not resolve is allowed -- http://my_ollama:11434 may only
#     resolve in the client's network namespace, not in this one;
#   * it is bounded and cached, so a dead resolver cannot stall each request.
#     A client is built per request and the route validates the same URL again,
#     so the cache is what keeps a request to one lookup.
# Short on purpose. This validator is sync and called from async handlers, so
# the wait is the event loop's wait, and every millisecond of it is shared by
# every concurrent request. A provider hostname that a resolver can answer at
# all is answered well inside this; past it the answer is treated as unknown,
# which the default path allows and the opt-in path re-asks for without a bound.
# So a longer deadline buys accuracy for nobody and costs latency for everyone.
_DNS_TIMEOUT_SECONDS = 0.5
_DNS_CACHE_TTL_SECONDS = 300.0
_DNS_CACHE_MAX_ENTRIES = 512
# hostname -> (expiry, addresses). Only answers land here; a failure and a
# timeout are both cheap to repeat and wrong to remember.
_dns_cache: dict[str, tuple[float, tuple[str, ...]]] = {}
_dns_cache_lock = threading.Lock()
# A lookup that times out is abandoned, not cancelled, so its thread lives until
# the platform resolver gives up. Rotating hostnames defeat the cache and would
# otherwise pile those up one per request, so the number in flight is capped and
# a caller waits its turn up to the same deadline rather than being waved
# through the moment the pool is busy.
#
# Saturating the pool still ends in "no answer", which the default path allows.
# That is the same decision this file makes for a lookup that times out, and it
# is deliberate: refusing instead would mean any resolver trouble, or any caller
# willing to stall a few lookups, could stop the operator configuring a provider
# at all. The check is a bound on what a caller-supplied URL may resolve to, not
# a guarantee about what the socket will later connect to -- see the transport
# note on _resolve_host.
_DNS_MAX_IN_FLIGHT = 32
_dns_in_flight = threading.BoundedSemaphore(_DNS_MAX_IN_FLIGHT)

# Hostnames of the providers this build ships. They are hard-coded destinations,
# not caller-controlled names, so learning their addresses buys nothing.
_REGISTRY_HOSTNAMES = frozenset(
    host
    for host in (
        urlsplit(info["base_url"]).hostname
        for info in PROVIDER_REGISTRY.values()
        if info.get("base_url")
    )
    if host
)


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


# What httpx leaves unescaped in a host: RFC 3986 sub-delims plus the WHATWG
# set. Kept in step with its `_urlparse.encode_host`.
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
        # httpx percent-encodes what RFC 3986 does not allow in a reg-name, so
        # safe^alias.example is dialled as safe%5Ealias.example, a name whose
        # parent zone can answer differently. Same quoting, same name.
        return quote(hostname.lower(), safe = _HOST_SAFE_CHARS)
    try:
        import idna
        return idna.encode(hostname.lower()).decode("ascii")
    except Exception:
        # httpx raises InvalidURL here, so the request cannot happen at all;
        # resolving the name as written is then as good an answer as any.
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

    # Bound to a local: a worker abandoned at the deadline may outlive the
    # global, and BoundedSemaphore raises if it releases one it never took.
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
            # Released by the worker, not the caller, so an abandoned lookup
            # frees its slot only once the resolver actually lets it go.
            in_flight.release()
        resolved.extend(str(info[4][0]) for info in infos)
        answered = True

    # Daemon thread, so a resolver that never answers cannot hold up shutdown;
    # the validator abandons it after the timeout and treats the name the same
    # way it treats any other lookup failure.
    thread = threading.Thread(target = _resolve, daemon = True)
    thread.start()
    thread.join(_DNS_TIMEOUT_SECONDS)
    if thread.is_alive():
        # A timeout is not an answer, so it is not remembered as one: the
        # transport waits longer than this and would still get the address a
        # deliberately slow authoritative server sends after the deadline.
        return None
    if not answered:
        # A failure is not cached either. It is cheap to repeat (the resolver
        # says so immediately), and remembering it would turn one transient
        # SERVFAIL into five minutes of refusal on the opt-in path.
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
        # Literals were already classified by _metadata_host; no lookup needed.
        ipaddress.ip_address(_canonical_host(hostname))
        return False
    except ValueError:
        pass
    return any(
        _metadata_address(address) for address in _resolve_host(hostname, port, scheme) or ()
    )


def _reject_non_public(hostname: str, port: int | None, scheme: str) -> None:
    """Raise when ``hostname`` is, or resolves to, a non-public address."""
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        # The metadata check ran a moment ago and cached whatever it learned, so
        # this reads that rather than starting a second bounded lookup: on a
        # slow resolver the pair of them would each spend a deadline before the
        # fallback below spent a third.
        resolved = _cached_addresses(hostname)
        if resolved is None:
            # This path blocked on an unbounded getaddrinfo before the metadata
            # check existed, and a resolver slower than that check's deadline is
            # ordinary (the Linux default is 5s per server, twice). Falling back
            # to the same unbounded call keeps a slow-but-working resolver from
            # turning into a refusal here, where "no answer" fails closed.
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
        raise ValueError(
            "Provider base URL points at a private address, which is disabled on this "
            f"server ({_BLOCK_PRIVATE_ENV}=1)."
        )


def validate_provider_base_url(base_url: str) -> str:
    """Return a normalized provider base URL, or raise ``ValueError``.

    The backend issues outbound requests to this URL with the caller's decrypted
    API key attached, so it is caller-controlled server-side egress. Only shapes
    that can never be a real provider endpoint are refused: a non-http(s) scheme,
    control characters, a missing host, and cloud metadata services. Plain http,
    loopback, LAN hosts, odd ports, query strings and basic-auth userinfo all
    stay valid -- Ollama, llama.cpp, vLLM and custom gateways rely on them. A
    caller-supplied hostname is resolved far enough to apply the metadata block
    to DNS aliases of it; rejecting other private addresses stays opt-in.

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
    # Userinfo stays allowed for gateways behind basic auth; the checks below read
    # the parsed hostname, so http://api.openai.com@169.254.169.254/ is caught.
    if not hostname:
        raise ValueError("Provider base URL must contain a hostname.")

    hostname = hostname.rstrip(".")
    if _metadata_host(hostname) or _resolves_to_metadata(hostname, port, scheme):
        raise ValueError("Cloud metadata endpoints cannot be used as a provider base URL.")

    if os.environ.get(_BLOCK_PRIVATE_ENV) == "1":
        _reject_non_public(hostname, port, scheme)

    return raw.rstrip("/")


def list_available_providers(include_hidden: bool = False) -> list[dict[str, Any]]:
    """Return registered providers (for the /registry endpoint).

    Hidden entries exist only for backend lookups and are surfaced by the UI via
    ``CUSTOM_PROVIDER_PRESETS`` instead of the dropdown, so they stay filtered
    out by default. That default is load-bearing for upgrades: a browser holding
    a cached bundle from before this capability existed has no idea to filter on
    ``hidden``, and would render the self-hosted presets as duplicate dropdown
    entries.

    ``include_hidden`` is how a client that does know says so. The self-hosted
    presets are exactly the ones that run Unsloth's tools, so their capability
    has to reach a frontend that asks for it, and asking is opt-in.
    """
    result = []
    for provider_type, info in PROVIDER_REGISTRY.items():
        if info.get("hidden") and not include_hidden:
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
