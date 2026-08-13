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
from typing import Any
from urllib.parse import urlsplit

PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "openai_codex": {
        "display_name": "ChatGPT / Codex subscription",
        "base_url": "https://chatgpt.com/backend-api",
        "default_models": [
            "gpt-5.3-codex-spark",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
        ],
        "model_capabilities": {
            "gpt-5.3-codex-spark": {"vision": False, "studio_tools": True},
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
    """Whether Studio may run its own tool loop against this provider type.

    Studio's tools (web_search, python, terminal, MCP, knowledge-base search)
    execute on the Studio host, so any provider whose wire format can carry a
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
    info = PROVIDER_REGISTRY.get(provider_type or "")
    return bool(info and info.get("studio_tools"))


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
    )
)
# Link-local, where the IPv4 metadata services live. Matched as a network so a
# DNS name that merely starts with those digits (169.254.gateway.example.com) is
# not mistaken for one.
_METADATA_NETWORK = ipaddress.ip_network("169.254.0.0/16")

# Opt-in for operators who expose Studio on a shared host: also refuse provider
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


def _reject_non_public(hostname: str, port: int | None, scheme: str) -> None:
    """Raise when ``hostname`` is, or resolves to, a non-public address."""
    import socket

    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(
                hostname,
                port or (443 if scheme == "https" else 80),
                type = socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError) as exc:
            raise ValueError("Provider base URL hostname could not be resolved.") from exc
        addresses = [ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]) for info in infos]
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
    stay valid -- Ollama, llama.cpp, vLLM and custom gateways rely on them. No
    DNS lookup happens unless the private-address opt-in is set.

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
    if _metadata_host(hostname):
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
    presets are exactly the ones that run Studio's tools, so their capability
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
