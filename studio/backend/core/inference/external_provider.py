# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Async HTTP client for proxying chat completions to external LLM providers.

Most registry providers expose OpenAI-compatible /v1/chat/completions endpoints;
Anthropic uses native Messages API with translation in this client.
"""

import json as _json
import mimetypes
import re
import time
from typing import Any, AsyncGenerator, Literal, NamedTuple, Optional, Union
from urllib.parse import urlparse

import httpx
import structlog

# Use structlog so INFO-level diagnostics actually surface in the
# studio backend's JSON log stream. The stdlib root logger defaults to
# WARNING and is not configured with handlers, so plain
# `logging.getLogger(__name__).info(...)` was being silently dropped —
# only WARNING/ERROR made it through (because they bypassed the root
# level threshold via uvicorn's stderr capture). All existing call
# sites use printf-style positional args, which structlog accepts.
logger = structlog.get_logger(__name__)


# Claude 4.7 (Opus/Sonnet/Haiku) removed temperature, top_p, and top_k —
# the API returns 400 "<param> is deprecated for this model" if any of
# them is set to a non-default value. The "Sampling parameters removed"
# section of the 4.7 release notes is the authoritative reference:
#   https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7
# 3.x and 4.5/4.6 still accept all three; match the 4-7 line strictly so
# the knobs keep working on earlier families. The trailing -4-7[-.]/EOL
# anchor keeps future versions (e.g. claude-opus-5) unaffected.
def _is_openai_family_cloud(base_url: Optional[str]) -> bool:
    """True iff ``base_url`` points at OpenAI cloud or Azure OpenAI Foundry.

    Anchored to the URL host so an attacker can't bypass the gate with a
    path or subdomain like ``https://evil.com/api.openai.com/v1`` or
    ``https://api.openai.com.attacker.com/v1`` (CodeQL py/incomplete-url-
    substring-sanitization). Used to scope cloud-only Responses-API
    extensions (prompt_cache_retention, context_management compaction,
    container shell tool) that 400 on non-cloud OpenAI-compatible
    servers (ollama / llama.cpp / vLLM).

    Azure Foundry resources are scoped to
    ``<resource-name>.openai.azure.com``; match any subdomain via an
    `endswith` on the lowercased hostname, with the leading dot so
    `openai.azure.com` itself doesn't slip through (there is no
    apex-hosted Azure Foundry endpoint).
    """
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return host == "api.openai.com" or host.endswith(".openai.azure.com")


_ANTHROPIC_4_7_SAMPLING_REMOVED = re.compile(
    r"^claude-(?:opus|sonnet|haiku)-4-7(?:[-.]|$)"
)
_OPENAI_REASONING_SUMMARY_UNSUPPORTED = re.compile(r"^o3(?:[-.]|$)")


class _AnthropicThinkingSpec(NamedTuple):
    prefixes: tuple[str, ...]
    kind: Literal["adaptive", "manual"]
    efforts: tuple[str, ...]


_ANTHROPIC_THINKING_SPECS = (
    _AnthropicThinkingSpec(
        prefixes = ("claude-opus-4-7",),
        kind = "adaptive",
        efforts = ("none", "low", "medium", "high", "xhigh", "max"),
    ),
    _AnthropicThinkingSpec(
        prefixes = ("claude-opus-4-6", "claude-sonnet-4-6"),
        kind = "adaptive",
        efforts = ("none", "low", "medium", "high", "xhigh", "max"),
    ),
    _AnthropicThinkingSpec(
        prefixes = ("claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"),
        kind = "manual",
        efforts = ("none", "low", "medium", "high"),
    ),
)


def _anthropic_thinking_spec(model: str) -> Optional[_AnthropicThinkingSpec]:
    for spec in _ANTHROPIC_THINKING_SPECS:
        if model.startswith(spec.prefixes):
            return spec
    return None


# Anthropic ships date-pinned tool versions per model family. Per the
# tool-reference docs (https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)
# the newer `_20260209` / `_20260120` variants only run on Opus 4.6/4.7
# and Sonnet 4.6 (web_search / web_fetch) or Opus 4.5+ and Sonnet 4.5+
# (code_execution). Sending the new versions to an older model returns
# 400 "tool not supported", and sending the old versions on a new model
# misses the dynamic-filtering and free-with-search pricing path. Pick
# the newest combination the model accepts, falling back to the GA
# (`_20250305` / `_20250910` / `_20250825`) defaults for everything else.
_ANTHROPIC_NEW_WEB_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)
_ANTHROPIC_NEW_CODE_EXEC_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
)


def _anthropic_web_search_version(model: str) -> str:
    return (
        "web_search_20260209"
        if model.startswith(_ANTHROPIC_NEW_WEB_PREFIXES)
        else "web_search_20250305"
    )


def _anthropic_web_fetch_version(model: str) -> str:
    return (
        "web_fetch_20260209"
        if model.startswith(_ANTHROPIC_NEW_WEB_PREFIXES)
        else "web_fetch_20250910"
    )


def _anthropic_code_execution_version(model: str) -> str:
    return (
        "code_execution_20260120"
        if model.startswith(_ANTHROPIC_NEW_CODE_EXEC_PREFIXES)
        else "code_execution_20250825"
    )


# Anthropic's beta-header flag for code execution does NOT change with
# the tool version -- both `_20250825` and `_20260120` are unlocked by
# the same `code-execution-2025-08-25` header per the upstream docs.
_ANTHROPIC_CODE_EXECUTION_BETA = "code-execution-2025-08-25"


# Anthropic server-side context compaction (beta as of compact-2026-01-12).
# Per the docs, the compaction tool is currently supported on Opus 4.6,
# Opus 4.7, Sonnet 4.6 and Mythos Preview. The beta header is the same
# for every supported model; the dated `compact_20260112` type lives in
# the body's `context_management.edits` array. Anything sent to a model
# outside this prefix list is silently ignored so we don't 400 upstream.
_ANTHROPIC_COMPACTION_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-mythos-preview",
)
_ANTHROPIC_COMPACTION_BETA = "compact-2026-01-12"
_ANTHROPIC_COMPACTION_TYPE = "compact_20260112"
# The docs require the threshold to be at least 50K tokens; lower values
# would 400. We clamp on the way out so a UI slider can't underflow.
_ANTHROPIC_COMPACTION_MIN = 50_000


def _anthropic_supports_compaction(model: str) -> bool:
    return model.startswith(_ANTHROPIC_COMPACTION_PREFIXES)


class _MistralThinkingSpec(NamedTuple):
    models: tuple[str, ...]
    style: Literal["prompt_mode", "reasoning_effort", "disabled"]
    efforts: tuple[str, ...] = ()


_MISTRAL_THINKING_SPECS = (
    _MistralThinkingSpec(
        models = ("magistral-medium-latest",),
        style = "prompt_mode",
    ),
    _MistralThinkingSpec(
        models = ("mistral-small-latest", "mistral-vibe-cli-latest"),
        style = "reasoning_effort",
        efforts = ("none", "high"),
    ),
)

_OPENROUTER_MANDATORY_REASONING_MODELS = frozenset(
    {
        "~google/gemini-pro-latest",
        "baidu/cobuddy:free",
        "inclusionai/ring-2.6-1t:free",
        "deepseek/deepseek-r1",
    }
)


def _mistral_thinking_spec(model: str) -> _MistralThinkingSpec:
    for spec in _MISTRAL_THINKING_SPECS:
        if model in spec.models:
            return spec
    return _MistralThinkingSpec(models = (), style = "disabled")


def _apply_mistral_reasoning_controls(
    body: dict[str, Any],
    model: str,
    enable_thinking: Optional[bool],
    reasoning_effort: Optional[str],
) -> None:
    """
    Translate generic reasoning controls into Mistral's model-specific shape.

    Current contract:
      - magistral-medium-latest: baseline (no extra field) or
        `prompt_mode="reasoning"` for the explicit reasoning mode.
      - mistral-small-latest / mistral-vibe-cli-latest:
        `reasoning_effort` in {"none", "high"}.
      - all other tested Mistral models: no reasoning/thinking params.
    """
    model_for_matching = model.rsplit("/", 1)[-1].strip().lower()
    spec = _mistral_thinking_spec(model_for_matching)
    body.pop("prompt_mode", None)
    body.pop("reasoning_effort", None)

    if spec.style == "prompt_mode":
        # Magistral baseline is already reasoning-capable. The explicit
        # prompt_mode path is only used for the "high" UI selection.
        if enable_thinking is True or reasoning_effort == "high":
            body["prompt_mode"] = "reasoning"
        return

    if spec.style == "reasoning_effort":
        if reasoning_effort in spec.efforts:
            body["reasoning_effort"] = reasoning_effort
        elif enable_thinking is False:
            body["reasoning_effort"] = "none"
        elif enable_thinking is True:
            body["reasoning_effort"] = "high"


# Shared client reused across all requests for HTTP connection pooling.
# Auth headers and timeouts are passed per-request, so a single client
# handles every provider without storing credentials.
_http_client = httpx.AsyncClient()


def _build_kimi_tool_end(
    synthetic_chunk_fn: Any,
    tool_call_id: str,
    citations: list[dict[str, str]],
) -> str:
    """Format Kimi web_search citations into the tool_end payload.

    Same shape parseSourcesFromResult on the frontend expects for the
    other built-in web_search providers: `Title: ...\\nURL: ...\\n
    Snippet: ...\\n---\\n...`. If no citations were emitted, fall back
    to a generic "(search complete)" string so the UI still shows the
    tool card transitioning to a completed state.
    """
    blocks: list[str] = []
    for cit in citations:
        line = f"Title: {cit['title']}\nURL: {cit['url']}"
        if cit.get("snippet"):
            line += f"\nSnippet: {cit['snippet']}"
        blocks.append(line)
    return synthetic_chunk_fn(
        {
            "type": "tool_end",
            "tool_call_id": tool_call_id,
            "result": "\n---\n".join(blocks) if blocks else "(search complete)",
        }
    )


class ExternalProviderClient:
    """Async proxy for OpenAI-compatible external LLM APIs."""

    def __init__(
        self,
        provider_type: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
    ):
        self.provider_type = provider_type
        self.base_url = base_url.rstrip("/")
        # Legacy Gemini configs saved with the OpenAI-compatibility base
        # (`/v1beta/openai`) build broken native URLs after PR #5720
        # switched Gemini onto the native streamGenerateContent endpoint
        # (`/v1beta/openai/models/{model}:streamGenerateContent` 404s).
        # Strip the `/openai` suffix transparently so saved providers keep
        # working without a manual re-config. Gate strictly to the
        # Google-hosted base so custom proxies whose paths also end in
        # `/openai` (e.g. `https://proxy.example.com/team/openai`) are
        # left untouched.
        if self.provider_type == "gemini":
            _parsed_base = urlparse(self.base_url)
            if (
                (_parsed_base.hostname or "").lower()
                == "generativelanguage.googleapis.com"
                and _parsed_base.path.rstrip("/") == "/v1beta/openai"
            ):
                self.base_url = self.base_url[: -len("/openai")]
        self.api_key = api_key
        self._timeout = httpx.Timeout(timeout, connect = 10.0)
        # Separate timeout for SSE streams: reasoning-heavy providers
        # (Anthropic Opus 4.7 with adaptive thinking, OpenAI gpt-5.x via
        # /v1/responses) can pause for tens of seconds between bytes
        # while the model is internally thinking. httpx's read timeout is
        # the *gap* between successive reads, not a wall clock — so
        # disabling it lets long thinks complete without cutting the
        # stream prematurely. connect/write/pool keep the 10s / 120s
        # bounds so genuine network failures still surface.
        self._stream_timeout = httpx.Timeout(timeout, connect = 10.0, read = None)

    def _auth_headers(self) -> dict[str, str]:
        """Build authentication headers using the provider's registry config."""
        from core.inference.providers import get_provider_info

        provider_info = get_provider_info(self.provider_type) or {}
        auth_header = provider_info.get("auth_header", "Authorization")
        auth_prefix = provider_info.get("auth_prefix", "Bearer ")

        # Gemini connections pointed at a custom OpenAI-compat gateway
        # (non-Google host, path ending /openai) speak the OpenAI surface
        # and authenticate with Authorization: Bearer ..., not Google's
        # native x-goog-api-key. Override the registry default so the
        # OAI-compat dispatch path receives the right header.
        if self.provider_type == "gemini":
            _p = urlparse(self.base_url)
            _host = (_p.hostname or "").lower()
            _path = _p.path.rstrip("/").lower()
            if (
                _host != "generativelanguage.googleapis.com"
                and _path.endswith("/openai")
            ):
                auth_header = "Authorization"
                auth_prefix = "Bearer "

        headers = {"Content-Type": "application/json"}
        # Skip auth header when api_key is empty (optional for local providers);
        # httpx rejects an empty `Bearer ` value as "Illegal header value".
        if self.api_key:
            headers[auth_header] = f"{auth_prefix}{self.api_key}"
        # Merge any provider-specific extra headers (e.g. anthropic-version, OpenRouter attribution)
        headers.update(provider_info.get("extra_headers", {}))
        return headers

    def _is_openai_compatible(self) -> bool:
        """Return False for providers that need request/response translation (e.g. Anthropic)."""
        from core.inference.providers import get_provider_info

        info = get_provider_info(self.provider_type) or {}
        # Gemini ships an OpenAI-compatible surface at
        # `/v1beta/openai/chat/completions` (Authorization: Bearer ...)
        # and a native surface at `/v1beta/models/...:streamGenerateContent`
        # (x-goog-api-key). Google-hosted Gemini moved to native in this
        # PR for full feature coverage, but third-party gateways /
        # custom proxies that still expose only the OpenAI-compat
        # surface (e.g. https://proxy.example.com/team/openai) would
        # break if we forced them through the native translator.
        # Detect the OAI-compat suffix and fall back to OpenAI-compat
        # forwarding for non-Google bases.
        if self.provider_type == "gemini":
            _p = urlparse(self.base_url)
            _path = _p.path.rstrip("/").lower()
            _host = (_p.hostname or "").lower()
            if (
                _host != "generativelanguage.googleapis.com"
                and _path.endswith("/openai")
            ):
                return True
        return info.get("openai_compatible", True)

    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        top_k: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[Union[bool, str]] = None,
        openai_code_exec_container_id: Optional[str] = None,
        anthropic_code_exec_container_id: Optional[str] = None,
        prompt_cache_ttl: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        Yield OpenAI-format SSE lines from the external provider.

        For OpenAI-compatible providers, lines are forwarded verbatim.
        For Anthropic, the native Messages API SSE is translated to OpenAI format.

        ``top_k`` and ``presence_penalty`` are forwarded only when the caller
        supplies a value the provider accepts — the frontend's
        provider-capability map already filters these per provider, so we
        treat them as opt-in here.
        """
        if not self._is_openai_compatible():
            # Gemini speaks its own native REST shape (contents/parts);
            # `_stream_gemini` translates request/response into the OpenAI
            # Chat Completions chunk format the rest of Studio expects.
            # API reference: https://ai.google.dev/gemini-api/docs
            if self.provider_type == "gemini":
                async for line in self._stream_gemini(
                    messages,
                    model,
                    temperature,
                    top_p,
                    max_tokens,
                    top_k,
                    presence_penalty,
                    enabled_tools,
                    enable_prompt_caching,
                    enable_thinking,
                    reasoning_effort,
                    tools,
                    tool_choice,
                ):
                    yield line
                return
            async for line in self._stream_anthropic(
                messages,
                model,
                temperature,
                top_p,
                max_tokens,
                top_k,
                enable_thinking,
                reasoning_effort,
                enabled_tools,
                enable_prompt_caching,
                anthropic_code_exec_container_id,
                prompt_cache_ttl,
                compaction_threshold,
            ):
                yield line
            return

        # OpenAI moved their flagship models (gpt-5.x) off /v1/chat/completions
        # — those endpoints return 404 with "This is not a chat model" for the
        # new families. Route all OpenAI traffic through /v1/responses instead;
        # we translate the Responses SSE back into Chat Completions chunks so
        # the frontend stays endpoint-agnostic.
        if self.provider_type == "openai":
            async for line in self._stream_openai_responses(
                messages,
                model,
                temperature,
                top_p,
                max_tokens,
                enable_thinking,
                reasoning_effort,
                enabled_tools,
                enable_prompt_caching,
                openai_code_exec_container_id,
                compaction_threshold,
            ):
                yield line
            return

        # Kimi's $web_search is a builtin_function that requires a client
        # round-trip: the first call returns a tool_calls envelope with
        # function.arguments populated; the caller echoes those arguments
        # back as a role=tool message; the second call streams the final
        # answer with the search incorporated. The doc also mandates
        # disabling thinking while $web_search is active. Route to a
        # dedicated helper so the default OAI-compat path stays single-pass.
        #   https://platform.kimi.ai/docs/guide/use-web-search
        if (
            self.provider_type == "kimi"
            and enabled_tools
            and "web_search" in enabled_tools
        ):
            async for line in self._stream_kimi_web_search(
                messages,
                model,
                max_tokens,
            ):
                yield line
            return

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
        }
        if max_tokens is not None:
            # OpenAI newer models (gpt-4o, gpt-5.x) reject max_tokens
            if self.provider_type == "openai":
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens

        # Strip body fields a provider's registry entry declares unusable —
        # reasoning-class models that lock these to fixed defaults (e.g.
        # Kimi k2.5/k2.6 only accept temperature=1, top_p=1) 400 otherwise.
        # The frontend capability map already hides the matching sliders;
        # this is the matching guard for the pydantic default that the
        # route layer would otherwise still fill in.
        from core.inference.providers import get_provider_info

        provider_info = get_provider_info(self.provider_type) or {}
        for field in provider_info.get("body_omit", ()):
            body.pop(field, None)

        # Kimi (kimi-k2.6, kimi-k2-thinking) accepts a boolean thinking toggle
        # via a top-level `thinking` field (the docs show it nested under
        # extra_body, but that is an OpenAI Python SDK convention; on the
        # wire it merges into the request body).
        #   - kimi-k2.6 defaults to thinking enabled; clients can pass
        #     {"type": "disabled"} to suppress it.
        #   - kimi-k2-thinking is always on; we never send disabled there.
        # `keep: all` retains every thinking chunk through the stream, which
        # is what we need so our frontend can wrap reasoning_content into
        # the chat reasoning panel.
        if self.provider_type == "kimi" and enable_thinking is not None:
            if model == "kimi-k2-thinking":
                # Always on; ignore client toggle to avoid an API-level reject.
                pass
            elif enable_thinking:
                body["thinking"] = {"type": "enabled", "keep": "all"}
            else:
                body["thinking"] = {"type": "disabled"}
        elif self.provider_type == "mistral":
            _apply_mistral_reasoning_controls(
                body, model, enable_thinking, reasoning_effort
            )
        elif self.provider_type == "vllm" and enable_thinking is not None:
            # vLLM gates thinking via chat_template_kwargs.enable_thinking.
            tpl_kw = body.get("chat_template_kwargs")
            if not isinstance(tpl_kw, dict):
                tpl_kw = {}
            tpl_kw["enable_thinking"] = bool(enable_thinking)
            body["chat_template_kwargs"] = tpl_kw

        # OpenRouter exposes a unified `reasoning` parameter on every
        # chat-completion request — the gateway routes it to whichever
        # underlying model actually supports reasoning, and silently
        # no-ops for ones that don't. Documented at
        #   https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
        # Shape: `reasoning: {enabled?: bool, effort?: low|medium|high,
        # max_tokens?: N, exclude?: bool}` with effort and max_tokens
        # mutually exclusive. We forward either an effort level (when
        # the user picked one) or a bare {enabled: true}. A small set of
        # known routes rejects explicit disable with 400 ("Reasoning is
        # mandatory for this endpoint ..."), so only those omit "off".
        if self.provider_type == "openrouter":
            normalized_or_model = model.strip().lower()
            if reasoning_effort in ("low", "medium", "high"):
                body["reasoning"] = {"effort": reasoning_effort}
            elif enable_thinking is True:
                body["reasoning"] = {"enabled": True}
            elif enable_thinking is False:
                if normalized_or_model in _OPENROUTER_MANDATORY_REASONING_MODELS:
                    body.pop("reasoning", None)
                else:
                    body["reasoning"] = {"enabled": False}

            # OpenRouter web-search plugin — universal shape that works
            # for every model id, including the `openrouter/free` and
            # `openrouter/auto` meta-routers. Documented at
            #   https://openrouter.ai/docs/guides/features/plugins/web-search
            # The `:online` model-suffix shortcut is "exactly equivalent
            # to" this plugin per the same doc, but only works on
            # concrete model ids — meta-routers reject the suffix.
            # `plugins: [{id: "web"}]` works everywhere, no model id
            # rewrite needed, and idempotent if some future call site
            # adds the entry first.
            if enabled_tools and "web_search" in enabled_tools:
                plugins = list(body.get("plugins") or [])
                if not any(
                    isinstance(p, dict) and p.get("id") == "web" for p in plugins
                ):
                    plugins.append({"id": "web"})
                body["plugins"] = plugins
                logger.info(
                    "OpenRouter web_search: attached plugins=[{id: 'web'}] "
                    "(model=%s)",
                    body.get("model"),
                )

        url = f"{self.base_url}/chat/completions"
        logger.info(
            "Proxying chat completion to %s (provider=%s, model=%s)",
            url,
            self.provider_type,
            model,
        )

        try:
            async with _http_client.stream(
                "POST",
                url,
                json = body,
                headers = self._auth_headers(),
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    error_text = _friendly_provider_error_text(
                        self.provider_type,
                        response.status_code,
                        error_text,
                        model = model,
                    )
                    logger.error(
                        "External provider returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    yield _error_sse_line(
                        response.status_code, error_text, self.provider_type
                    )
                    return

                # NOTE: manual __anext__ loop instead of `async for` is intentional.
                # On Python 3.13 + httpcore 1.0.x, `async for` auto-calls aclose() on
                # early exit (break/return/GeneratorExit) BEFORE our finally block runs.
                # That propagates GeneratorExit into PoolByteStream.__aiter__() while it
                # calls `await self.aclose()` inside `with AsyncShieldCancellation()`,
                # triggering "RuntimeError: async generator ignored GeneratorExit".
                # Fix: call response.aclose() FIRST (sets PoolByteStream._closed=True),
                # then lines_gen.aclose() is a no-op and GeneratorExit re-raises cleanly.
                lines_gen = response.aiter_lines().__aiter__()
                # Best-effort diagnostics for the default OAI-compat path. Without
                # this, OpenRouter mid-stream errors (200 OK + error event in the
                # SSE body) and OpenRouter-router model selection were invisible
                # in the backend logs — the user only saw "Provider returned
                # error" in the UI with no trail on the server side.
                event_counts: dict[str, int] = {}
                chosen_model: Optional[str] = None
                # Web-search tool-card synthesis for OpenRouter. The gateway
                # doesn't emit structured web_search_call events — citations
                # come back as `annotations` of type=url_citation on delta /
                # message objects. Mirror the OpenAI/Anthropic UX by yielding
                # a synthetic tool_start at stream open and tool_end at
                # stream close with the collected citation list.
                web_search_active = (
                    self.provider_type == "openrouter"
                    and bool(enabled_tools)
                    and "web_search" in (enabled_tools or [])
                )
                web_search_tool_id = "openrouter_web_search"
                web_search_citations: list[dict[str, str]] = []
                web_search_tool_started = False
                web_search_tool_ended = False

                def _emit_synthetic_tool_event(payload: dict[str, Any]) -> str:
                    chunk = {
                        "id": f"chatcmpl-{self.provider_type}-synthetic",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": None,
                            }
                        ],
                        "_toolEvent": payload,
                    }
                    return f"data: {_json.dumps(chunk)}"

                def _record_or_url_citation(payload: Any) -> None:
                    if not isinstance(payload, dict):
                        return
                    if payload.get("type") != "url_citation":
                        return
                    # OpenRouter (and OpenAI Chat Completions web_search)
                    # nest the citation under url_citation; some variants
                    # ship the fields flat on the annotation itself. Accept
                    # both.
                    cit = payload.get("url_citation")
                    if not isinstance(cit, dict):
                        cit = payload
                    url = cit.get("url", "") if isinstance(cit, dict) else ""
                    if not url or not isinstance(url, str):
                        return
                    if any(c["url"] == url for c in web_search_citations):
                        return
                    title = cit.get("title") or url
                    snippet = cit.get("content") or cit.get("snippet") or ""
                    web_search_citations.append(
                        {
                            "url": url,
                            "title": title,
                            "snippet": snippet if isinstance(snippet, str) else "",
                        }
                    )

                def _build_web_search_tool_end() -> str:
                    blocks: list[str] = []
                    for cit in web_search_citations:
                        line = f"Title: {cit['title']}\nURL: {cit['url']}"
                        if cit.get("snippet"):
                            line += f"\nSnippet: {cit['snippet']}"
                        blocks.append(line)
                    return _emit_synthetic_tool_event(
                        {
                            "type": "tool_end",
                            "tool_call_id": web_search_tool_id,
                            "result": (
                                "\n---\n".join(blocks)
                                if blocks
                                else "(search complete)"
                            ),
                        }
                    )

                if web_search_active:
                    yield _emit_synthetic_tool_event(
                        {
                            "type": "tool_start",
                            "tool_name": "web_search",
                            "tool_call_id": web_search_tool_id,
                            "arguments": {},
                        }
                    )
                    web_search_tool_started = True

                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip():
                            continue
                        if line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                            if data_str == "[DONE]":
                                event_counts["done"] = event_counts.get("done", 0) + 1
                                # Emit synthetic tool_end with collected
                                # citations BEFORE forwarding [DONE], so the
                                # tool-card transitions to "complete" in the
                                # UI before the stream closes.
                                if (
                                    web_search_active
                                    and web_search_tool_started
                                    and not web_search_tool_ended
                                ):
                                    yield _build_web_search_tool_end()
                                    web_search_tool_ended = True
                            elif data_str:
                                try:
                                    parsed = _json.loads(data_str)
                                except Exception:
                                    parsed = None
                                if isinstance(parsed, dict):
                                    # Mid-stream provider error event. OpenRouter
                                    # in particular returns 200 then surfaces the
                                    # actual failure as an SSE error event.
                                    if "error" in parsed:
                                        event_counts["error"] = (
                                            event_counts.get("error", 0) + 1
                                        )
                                        logger.warning(
                                            "%s SSE error event: %s",
                                            self.provider_type,
                                            parsed.get("error"),
                                        )
                                    else:
                                        event_counts["delta"] = (
                                            event_counts.get("delta", 0) + 1
                                        )
                                    # OpenRouter (and most OAI-compat providers)
                                    # report the underlying model that handled
                                    # the request in every chunk's `model` field.
                                    # Latch the first non-empty value so the
                                    # router-picked model surfaces in logs and
                                    # is available to the proxy caller.
                                    if chosen_model is None and isinstance(
                                        parsed.get("model"), str
                                    ):
                                        chosen_model = parsed["model"]
                                    # When the user has web_search on, scan
                                    # every chunk's delta and message
                                    # objects for url_citation annotations.
                                    # Different OpenRouter upstreams place
                                    # them in different spots.
                                    if web_search_active:
                                        choices = parsed.get("choices") or []
                                        if isinstance(choices, list):
                                            for choice in choices:
                                                if not isinstance(choice, dict):
                                                    continue
                                                for envelope in (
                                                    choice.get("delta"),
                                                    choice.get("message"),
                                                ):
                                                    if not isinstance(envelope, dict):
                                                        continue
                                                    for ann in (
                                                        envelope.get("annotations")
                                                        or []
                                                    ):
                                                        _record_or_url_citation(ann)
                        yield line
                    # Stream ended without [DONE] (some upstreams just close
                    # the connection). Emit tool_end so the card doesn't
                    # stay in "running" forever.
                    if (
                        web_search_active
                        and web_search_tool_started
                        and not web_search_tool_ended
                    ):
                        yield _build_web_search_tool_end()
                        web_search_tool_ended = True
                except GeneratorExit:
                    await response.aclose()  # set PoolByteStream._closed=True FIRST
                    await lines_gen.aclose()  # now safe — aclose() is a no-op
                    raise
                finally:
                    logger.info(
                        "%s stream complete (model=%s, chosen=%s, "
                        "web_search_requested=%s, citations=%s, events=%s)",
                        self.provider_type,
                        model,
                        chosen_model,
                        web_search_active,
                        len(web_search_citations),
                        event_counts,
                    )
                    await response.aclose()
                    await lines_gen.aclose()

        except httpx.ConnectError as exc:
            logger.error("Connection error to %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Failed to connect to {self.provider_type}: {exc}",
                self.provider_type,
            )
        except httpx.ReadTimeout as exc:
            logger.error("Read timeout from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                504,
                f"Timeout waiting for {self.provider_type} response",
                self.provider_type,
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP error from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Error communicating with {self.provider_type}: {exc}",
                self.provider_type,
            )

    async def _stream_kimi_web_search(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: Optional[int],
    ) -> AsyncGenerator[str, None]:
        """
        Kimi $web_search round-trip.

        Wire flow (per https://platform.kimi.ai/docs/guide/use-web-search):
          1. POST messages with tools=[{type: "builtin_function",
             function: {name: "$web_search"}}] and thinking=disabled.
          2. Stream the first response — accumulate function.arguments
             across tool_call deltas until finish_reason="tool_calls".
             Do NOT forward those tool_call chunks to the client (they
             are an internal protocol step, not user-visible output).
          3. Build a second request: original messages + the assistant
             message carrying the tool_calls + a role=tool message that
             echoes the same arguments back verbatim (per Kimi docs,
             the caller "just needs to submit tool_call.function.arguments
             to Kimi as they are" — the server actually runs the search).
          4. Stream the second response — that is the final answer the
             user sees, with search results already incorporated.

        We synthesize tool_start (with the parsed query) when step (2)
        completes, and tool_end (with any url_citation annotations the
        second stream emits) before [DONE], so the chat UI shows the
        same web-search tool card as the other providers.
        """
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            # $web_search forbids thinking; sending the toggle silently
            # would have the server reject the request with 400.
            "thinking": {"type": "disabled"},
            "tools": [
                {"type": "builtin_function", "function": {"name": "$web_search"}}
            ],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        # Strip body fields the Kimi registry declares unusable
        # (temperature/top_p — see body_omit in providers.py).
        from core.inference.providers import get_provider_info

        provider_info = get_provider_info(self.provider_type) or {}
        for field in provider_info.get("body_omit", ()):
            body.pop(field, None)

        tool_call_id = "kimi_web_search"
        synthetic_id = f"chatcmpl-{self.provider_type}-synthetic"

        def _synthetic_chunk(payload: dict[str, Any]) -> str:
            chunk = {
                "id": synthetic_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "_toolEvent": payload,
            }
            return f"data: {_json.dumps(chunk)}"

        logger.info(
            "Kimi $web_search round-trip starting (model=%s, url=%s)",
            model,
            url,
        )

        # ---- First call: collect the model's $web_search tool_call ----
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        try:
            async with _http_client.stream(
                "POST",
                url,
                json = body,
                headers = self._auth_headers(),
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    logger.error(
                        "Kimi first-call returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    yield _error_sse_line(
                        response.status_code, error_text, self.provider_type
                    )
                    return

                lines_gen = response.aiter_lines().__aiter__()
                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip() or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = _json.loads(data_str)
                        except Exception:
                            continue
                        for choice in parsed.get("choices") or []:
                            if not isinstance(choice, dict):
                                continue
                            delta = choice.get("delta") or {}
                            for tc in delta.get("tool_calls") or []:
                                if not isinstance(tc, dict):
                                    continue
                                idx = tc.get("index", 0)
                                slot = tool_calls_acc.setdefault(
                                    idx,
                                    {
                                        "id": tc.get("id") or f"call_{idx}",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["function"]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    slot["function"]["arguments"] += fn["arguments"]
                            if choice.get("finish_reason") == "tool_calls":
                                break
                except GeneratorExit:
                    await response.aclose()
                    await lines_gen.aclose()
                    raise
                finally:
                    await response.aclose()
                    await lines_gen.aclose()
        except httpx.HTTPError as exc:
            logger.error("Kimi first-call HTTP error: %s", exc)
            yield _error_sse_line(
                502,
                f"Error communicating with kimi: {exc}",
                self.provider_type,
            )
            return

        # If the model decided not to search, fall back to a plain
        # streaming call without the builtin tool. That mirrors the UX
        # of every other provider when web_search is on but the model
        # didn't actually need it.
        search_calls = [
            tc
            for tc in tool_calls_acc.values()
            if tc["function"]["name"] == "$web_search"
        ]
        if not search_calls:
            logger.info(
                "Kimi $web_search: model did not invoke search; "
                "falling back to plain stream"
            )
            fallback_body = dict(body)
            fallback_body.pop("tools", None)
            try:
                async with _http_client.stream(
                    "POST",
                    url,
                    json = fallback_body,
                    headers = self._auth_headers(),
                    timeout = self._stream_timeout,
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_text = error_body.decode("utf-8", errors = "replace")
                        logger.error(
                            "Kimi fallback returned %d: %s",
                            response.status_code,
                            error_text[:500],
                        )
                        yield _error_sse_line(
                            response.status_code, error_text, self.provider_type
                        )
                        return
                    # Manual __anext__ loop instead of `async for` — see the
                    # comment in stream_chat_completion for the Python 3.13 +
                    # httpcore 1.0.x GeneratorExit interaction this avoids.
                    lines_gen = response.aiter_lines().__aiter__()
                    try:
                        while True:
                            try:
                                line = await lines_gen.__anext__()
                            except StopAsyncIteration:
                                break
                            if line.strip():
                                yield line
                    except GeneratorExit:
                        await response.aclose()
                        await lines_gen.aclose()
                        raise
                    finally:
                        await response.aclose()
                        await lines_gen.aclose()
            except httpx.HTTPError as exc:
                logger.error("Kimi fallback HTTP error: %s", exc)
                yield _error_sse_line(
                    502,
                    f"Error communicating with kimi: {exc}",
                    self.provider_type,
                )
            return

        # Synthesize tool_start with the parsed search query so the
        # chat UI's web-search card shows "Searching for: ...".
        first_args_raw = search_calls[0]["function"]["arguments"] or "{}"
        try:
            first_args = _json.loads(first_args_raw)
        except Exception:
            first_args = {}
        # Log the raw arguments so we can confirm the server actually
        # ran the search. The shape is documented loosely but in practice
        # the model emits `{"search_result":{"search_id":...},
        # "usage":{"total_tokens":N}}` — an opaque receipt where N is the
        # token cost of the injected search context. The query string is
        # NOT present; Kimi runs the search server-side during the first
        # call and bakes the results straight into the model's context.
        logger.info(
            "Kimi $web_search: %d tool_call(s), args[0]=%s",
            len(search_calls),
            first_args_raw[:500],
        )
        first_args_search_tokens: Optional[int] = None
        if isinstance(first_args, dict):
            usage_block = first_args.get("usage")
            if isinstance(usage_block, dict):
                tok = usage_block.get("total_tokens")
                if isinstance(tok, int):
                    first_args_search_tokens = tok
        yield _synthetic_chunk(
            {
                "type": "tool_start",
                "tool_name": "web_search",
                "tool_call_id": tool_call_id,
                "arguments": first_args if isinstance(first_args, dict) else {},
            }
        )
        # Kimi's search has already executed server-side by the time the
        # first call returns (the tool_call envelope encodes the search
        # result reference, not a query for us to dispatch). Emit
        # tool_end NOW so the UI's web-search card transitions to
        # "complete" before the second call starts streaming the
        # answer, instead of after — otherwise the card sits in
        # "running" all the way through the answer streaming and the
        # user perceives the model answering before search finishes.
        yield _build_kimi_tool_end(_synthetic_chunk, tool_call_id, [])

        # ---- Second call: echo the tool_calls back and stream answer ----
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": list(tool_calls_acc.values()),
        }
        tool_msgs = [
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["function"]["name"],
                "content": tc["function"]["arguments"],
            }
            for tc in tool_calls_acc.values()
        ]
        followup_body = dict(body)
        followup_body["messages"] = list(messages) + [assistant_msg] + tool_msgs
        # Ask the SSE stream to include a final `usage` block so we can
        # see prompt_tokens (which jumps to thousands when the server
        # injects search context). Without this, OpenAI-compat streams
        # omit usage entirely. Kimi follows the same convention.
        followup_body["stream_options"] = {"include_usage": True}
        # Keep the tool definition on the second call so the model can
        # decide to search again mid-turn if needed. Kimi's doc shows
        # the same tools array on every step.

        try:
            async with _http_client.stream(
                "POST",
                url,
                json = followup_body,
                headers = self._auth_headers(),
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    logger.error(
                        "Kimi second-call returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    yield _error_sse_line(
                        response.status_code, error_text, self.provider_type
                    )
                    return

                lines_gen = response.aiter_lines().__aiter__()
                # Diagnostics: latch usage.prompt_tokens from the final
                # chunk. The Kimi docs say search results count toward
                # prompt_tokens, so a big value here is direct evidence
                # the server actually injected results into context.
                last_usage: Optional[dict[str, Any]] = None
                annotation_shapes: set[str] = set()
                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip():
                            continue
                        if line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                            if data_str and data_str != "[DONE]":
                                try:
                                    parsed = _json.loads(data_str)
                                except Exception:
                                    parsed = None
                                if isinstance(parsed, dict):
                                    usage = parsed.get("usage")
                                    if isinstance(usage, dict):
                                        last_usage = usage
                                    # Scan annotations only for diagnostics —
                                    # Kimi today doesn't emit url_citation, but
                                    # if a future model version starts to we'll
                                    # see the type name in the final log line
                                    # and can wire it into the tool_end payload.
                                    for choice in parsed.get("choices") or []:
                                        if not isinstance(choice, dict):
                                            continue
                                        for envelope in (
                                            choice.get("delta"),
                                            choice.get("message"),
                                        ):
                                            if not isinstance(envelope, dict):
                                                continue
                                            for ann in (
                                                envelope.get("annotations") or []
                                            ):
                                                if isinstance(ann, dict):
                                                    annotation_shapes.add(
                                                        str(ann.get("type") or "?")
                                                    )
                        yield line
                except GeneratorExit:
                    await response.aclose()
                    await lines_gen.aclose()
                    raise
                finally:
                    logger.info(
                        "Kimi $web_search complete (model=%s, "
                        "search_ctx_tokens=%s, annotation_types=%s, "
                        "prompt_tokens=%s, completion_tokens=%s)",
                        model,
                        first_args_search_tokens,
                        sorted(annotation_shapes) or None,
                        (last_usage or {}).get("prompt_tokens"),
                        (last_usage or {}).get("completion_tokens"),
                    )
                    await response.aclose()
                    await lines_gen.aclose()
        except httpx.HTTPError as exc:
            logger.error("Kimi second-call HTTP error: %s", exc)
            yield _error_sse_line(
                502,
                f"Error communicating with kimi: {exc}",
                self.provider_type,
            )

    async def _stream_anthropic(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        top_k: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[bool] = None,
        anthropic_code_exec_container_id: Optional[str] = None,
        prompt_cache_ttl: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Call the Anthropic Messages API and translate its SSE to OpenAI format.

        Anthropic SSE event types:
          content_block_delta  → OpenAI chunk with delta.content
          message_delta        → OpenAI chunk with finish_reason
          message_stop         → data: [DONE]
          (all others skipped)
        """
        import json as _json

        # Extract system prompt and translate image_url parts to Anthropic format
        system: Optional[str] = None
        filtered: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                system = (
                    content
                    if isinstance(content, str)
                    else "\n".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
                )
                continue

            content = msg.get("content")
            if isinstance(content, list):
                # Translate OpenAI multimodal parts -> Anthropic native shapes.
                # - `image_url`     -> `{type:"image", source:...}`
                # - `input_document` -> `{type:"document", source:...}`
                #   (Studio extension; mirrors Anthropic's document block,
                #   which supports PDFs as base64 or URL per
                #   https://platform.claude.com/docs/en/build-with-claude/vision)
                anthropic_parts: list[dict[str, Any]] = []
                for part in content:
                    if part.get("type") == "text":
                        anthropic_parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "compaction":
                        # Round-trip the compaction block. When the
                        # prior assistant turn ran server-side
                        # compaction, that block must land back on this
                        # turn's assistant message so Anthropic skips
                        # re-compaction from scratch. Forward verbatim
                        # under the {type:"compaction", content:"..."}
                        # shape the API expects. See
                        #   https://platform.claude.com/docs/en/build-with-claude/compaction
                        summary = part.get("content") or ""
                        if isinstance(summary, str) and summary:
                            anthropic_parts.append(
                                {"type": "compaction", "content": summary}
                            )
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # data:image/png;base64,<DATA> -> split header and data
                            header, _, b64data = url.partition(",")
                            media_type = (
                                header.split(";")[0].replace("data:", "")
                                or "image/jpeg"
                            )
                            anthropic_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64data,
                                    },
                                }
                            )
                        else:
                            # Remote URL -- Anthropic supports url source type natively.
                            # See: https://docs.anthropic.com/en/docs/build-with-claude/vision#url-based-images
                            anthropic_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url,
                                    },
                                }
                            )
                    elif part.get("type") == "input_document":
                        # `input_document` is Studio's normalised content type
                        # for PDFs / docs. The frontend sends either
                        # `{type:"input_document", file_data:"data:application/pdf;base64,..."}`
                        # or `{type:"input_document", file_url:"https://..."}`,
                        # plus optional `filename` and `media_type`.
                        # Translate to Anthropic's native `document` block.
                        url = part.get("file_url") or ""
                        data_uri = part.get("file_data") or ""
                        title = part.get("filename")
                        # Treat any "data:" URI with no actual base64
                        # payload (`data:application/pdf;base64,` or
                        # whitespace-only) as missing so the file_url
                        # branch below can take over. Matches the
                        # OpenAI-side fallback so a malformed inline
                        # payload + valid remote URL still attaches.
                        data_uri_valid = False
                        b64data = ""
                        header = ""
                        if data_uri.startswith("data:"):
                            header, _, b64data = data_uri.partition(",")
                            data_uri_valid = bool(b64data.strip())
                        if data_uri_valid:
                            media_type = (
                                part.get("media_type")
                                or header.split(";")[0].replace("data:", "")
                                or "application/pdf"
                            )
                            doc_block: dict[str, Any] = {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64data,
                                },
                            }
                            if title:
                                doc_block["title"] = title
                            anthropic_parts.append(doc_block)
                        elif url:
                            doc_block = {
                                "type": "document",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                },
                            }
                            if title:
                                doc_block["title"] = title
                            anthropic_parts.append(doc_block)
                # Skip whole-message append when nothing usable survived.
                # An empty content array (e.g. user dropped only an unparseable
                # `input_document`) would 400 the Anthropic API with
                # "messages.N.content: at least one block is required".
                if anthropic_parts:
                    filtered.append({"role": msg["role"], "content": anthropic_parts})
            else:
                filtered.append(msg)

        # Claude 4.7 family removed temperature / top_p / top_k entirely.
        # The earlier guard only handled top_k; temperature is now also
        # rejected with 400 "temperature is deprecated for this model".
        # Latch the match once and reuse it everywhere temperature or
        # top_k would otherwise be set — including the thinking-mode
        # override below, which used to force temperature=1.
        sampling_removed = bool(_ANTHROPIC_4_7_SAMPLING_REMOVED.match(model))

        body: dict[str, Any] = {
            "model": model,
            "messages": filtered,
            "max_tokens": max_tokens or 1024,  # required by Anthropic
            "stream": True,
        }
        if not sampling_removed:
            body["temperature"] = temperature
        if top_k is not None and top_k > 0 and not sampling_removed:
            body["top_k"] = top_k
        # Anthropic only caches a prefix when at least one cache_control
        # marker is attached to it — the frontend defaults
        # enable_prompt_caching to True for Anthropic, so treat `None` the
        # same as True here (callers that don't set the flag still get
        # caching). Pass False explicitly to opt out.
        prompt_caching_enabled = enable_prompt_caching is not False
        # Anthropic accepts an optional `ttl` on each cache_control marker
        # (default is the 5m ephemeral pool; set "1h" to land in the 1h
        # pool instead). Per the prompt-caching docs, 1h cache writes are
        # billed at 2x base input vs 1.25x for 5m, but reads are 0.1x for
        # both. The 1h pool is the right pick when conversations span
        # multiple short bursts more than 5 minutes apart -- the read
        # discount makes up for the 1.6x write premium after a single
        # additional hit. Anything other than the known TTL strings is
        # dropped to avoid sending a malformed marker.
        #
        # The `extended-cache-ttl-2025-04-11` beta header that originally
        # gated 1h TTL has been promoted to GA: as of 2026-05 the live
        # API accepts `ttl: "1h"` without any beta opt-in. Verified
        # against api.anthropic.com on claude-opus-4-7 (status 200 +
        # `ephemeral_1h_input_tokens` populated). The test below pins
        # the contract by asserting the header is NOT on the wire so a
        # future regression that reintroduces the gate would surface
        # before users see a 400.
        cache_marker: dict[str, Any] = {"type": "ephemeral"}
        if prompt_cache_ttl in ("5m", "1h"):
            cache_marker["ttl"] = prompt_cache_ttl

        if system:
            if prompt_caching_enabled:
                # System block is the most stable prefix across turns, so
                # it gets its own breakpoint. Skipped when system is
                # empty — there's nothing to cache, and an empty marker
                # is a no-op.
                body["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": dict(cache_marker),
                    }
                ]
            else:
                body["system"] = system

        if prompt_caching_enabled and filtered:
            # Second breakpoint at the end of the conversation. Anthropic
            # caches the longest matching prefix up to a cache_control
            # marker; placing one on the latest message means turn N+1
            # rehydrates everything up through turn N from cache instead
            # of recomputing it. This is what makes caching actually work
            # when the system prompt is empty or shorter than Anthropic's
            # ~1024-token cache floor — the conversation history carries
            # the bulk of the input tokens. Anthropic allows up to 4
            # breakpoints per request; we use at most 2 (system + tail).
            last_msg = filtered[-1]
            content = last_msg.get("content")
            if isinstance(content, str):
                last_msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": dict(cache_marker),
                    }
                ]
            elif isinstance(content, list) and content:
                # Don't mutate the caller's list. Rebuild the tail with
                # cache_control attached to the final block so an
                # upstream image-bearing turn still cleanly slots into
                # the cache as part of the conversational prefix.
                head = list(content[:-1])
                tail = content[-1]
                if isinstance(tail, dict):
                    head.append({**tail, "cache_control": dict(cache_marker)})
                else:
                    head.append(tail)
                last_msg["content"] = head
        thinking_spec = _anthropic_thinking_spec(model)
        allowed_efforts = (
            thinking_spec.efforts
            if thinking_spec
            else ("none", "low", "medium", "high")
        )
        effort = reasoning_effort if reasoning_effort in allowed_efforts else None
        # Claude 4.6 Opus/Sonnet accept top-tier adaptive effort as "max" only;
        # "xhigh" is rejected (supported on Claude 4.7). Map our shared "xhigh"
        # semantic to "max" for 4.6 outbound requests while still accepting
        # both in ``allowed_efforts`` for persisted / cross-provider UI state.
        if effort == "xhigh" and model.startswith(
            ("claude-opus-4-6", "claude-sonnet-4-6")
        ):
            effort = "max"
        if effort is None:
            if enable_thinking is False:
                effort = "none"
            elif enable_thinking is True:
                effort = "medium"
        # Normalize one semantic Thinking control into Anthropic's two model-era
        # APIs: adaptive effort on Claude 4.6/4.7, manual budget_tokens on 4.5.
        if effort and effort != "none":
            # Anthropic rejects top_k whenever thinking is enabled.
            body.pop("top_k", None)
            # Earlier families (4.5/4.6) require temperature=1 when
            # thinking is enabled and forbid top_p in the same request:
            #   "temperature and top_p cannot both be specified for this
            #    model. Please use only one."
            # On Claude 4.7, temperature was removed entirely — sending
            # any value (including 1) returns 400 — so skip the override
            # there and let the model use its default sampling.
            if not sampling_removed:
                body["temperature"] = 1
            body.pop("top_p", None)
            if thinking_spec and thinking_spec.kind == "adaptive":
                # `display` defaults to "omitted" on Claude Opus 4.7 (per the
                # adaptive-thinking docs) — without an explicit opt-in the
                # API emits an empty thinking block plus a signature_delta,
                # so our SSE handler would surface a stray <think></think>
                # and the reasoning panel would stay blank. Force
                # "summarized" so 4.7 streams thinking_delta events like
                # 4.6 does. On 4.6 / Sonnet 4.6 this is the default, so
                # setting it explicitly is harmless.
                body["thinking"] = {"type": "adaptive", "display": "summarized"}
                # Per the Messages API reference, the effort knob for
                # adaptive thinking lives under `output_config.effort` —
                # NOT as a top-level field. Sending `effort: ...` directly
                # produces a 400 "effort: Extra inputs are not permitted".
                # Allowed values: low | medium | high | xhigh | max. See:
                # https://platform.claude.com/docs/en/api/messages
                body["output_config"] = {"effort": effort}
            elif thinking_spec and thinking_spec.kind == "manual":
                budget_tokens = {"low": 1024, "medium": 2048, "high": 4096}[effort]
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
                # Anthropic requires max_tokens to be strictly greater than
                # thinking.budget_tokens on the manual-thinking path.
                if body.get("max_tokens", 0) <= budget_tokens:
                    body["max_tokens"] = budget_tokens + 1024

        # Anthropic server-side web_search — see
        #   https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
        # The tool type is date-pinned per model family. Newer Opus /
        # Sonnet 4.6 + 4.7 accept `web_search_20260209` with dynamic
        # filtering (Claude writes code to filter results before they
        # reach context); everything else uses `web_search_20250305`.
        # `_anthropic_web_search_version` picks the right one. Anthropic
        # dispatches search calls server-side, returning server_tool_use
        # + web_search_tool_result blocks in the SSE stream, plus
        # url-citation annotations on text deltas. We translate all of
        # that into our local _toolEvent shape so the chat UI renders
        # web_search exactly like OpenAI's path.
        if enabled_tools and "web_search" in enabled_tools:
            anthropic_tools = list(body.get("tools") or [])
            anthropic_tools.append(
                {
                    "type": _anthropic_web_search_version(model),
                    "name": "web_search",
                    "max_uses": 5,
                }
            )
            body["tools"] = anthropic_tools

        # Anthropic server-side web_fetch — see
        #   https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
        # `web_fetch_20250910` reads a single URL (text or PDF) and
        # returns a document block in a `web_fetch_tool_result`. For
        # safety Anthropic only lets the model fetch URLs that already
        # appeared in the conversation (user message, prior tool
        # result, web_search hit) — there is no domain restriction we
        # have to apply locally. No beta header is required today; the
        # tool ships under the standard `2023-06-01` API version. We
        # mirror the web_search wiring: max_uses cap, opt in via
        # `enabled_tools=["web_fetch"]`, citations off by default
        # because the frontend already paints source pills from the
        # generic tool_end payload.
        web_fetch_enabled = bool(enabled_tools and "web_fetch" in enabled_tools)
        if web_fetch_enabled:
            anthropic_tools = list(body.get("tools") or [])
            anthropic_tools.append(
                {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "max_uses": 5,
                }
            )
            body["tools"] = anthropic_tools

        # Anthropic server-side code execution — see
        #   https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
        # The tool type is date-pinned per model family.
        # `_anthropic_code_execution_version` picks `code_execution_20260120`
        # for Opus 4.5+ / Sonnet 4.5+ / Opus 4.7 / Sonnet 4.6 (adds REPL
        # state persistence + programmatic tool calling) and falls back
        # to `code_execution_20250825` everywhere else. Both versions
        # run Python + bash + str_replace file edits inside a 5 GB
        # sandboxed container per request, with no internet access, and
        # both are unlocked by the same `code-execution-2025-08-25`
        # `anthropic-beta` header set further down. On the SSE stream
        # Anthropic emits two sub-tool names -- `bash_code_execution`
        # and `text_editor_code_execution` -- wrapped in the standard
        # server_tool_use / *_tool_result block shape.
        # v1 wires the tool only; file uploads (container_upload
        # content blocks and generated-file retrieval via the Files
        # API) are a deliberate follow-up.
        code_execution_enabled = bool(
            enabled_tools and "code_execution" in enabled_tools
        )
        if code_execution_enabled:
            anthropic_tools = list(body.get("tools") or [])
            anthropic_tools.append(
                {
                    "type": _anthropic_code_execution_version(model),
                    "name": "code_execution",
                }
            )
            body["tools"] = anthropic_tools
            # Reuse the prior turn's container so filesystem state
            # (files written, packages installed, variables set)
            # persists across turns of the same thread. Anthropic
            # exposes the container id on the Message object's
            # top-level `container.id`; on the SSE stream we latch it
            # off `message_start.message.container.id` further down
            # and emit a `container_ready` _toolEvent so the chat
            # adapter persists it on the thread record. A stale id
            # (container expired / not found) surfaces as a 4xx
            # below, where we emit `container_invalidated` and let
            # the next turn fall back to auto-create.
            if anthropic_code_exec_container_id:
                body["container"] = anthropic_code_exec_container_id

        # Server-side context compaction — see
        #   https://platform.claude.com/docs/en/build-with-claude/compaction
        # Beta as of `compact-2026-01-12`. When `compaction_threshold` is
        # provided AND the model accepts compaction (Opus 4.6+ / 4.7,
        # Sonnet 4.6, Mythos preview), attach
        # `context_management.edits[{type:"compact_20260112", trigger:
        # {type:"input_tokens", value:N}}]` to the body. Anthropic runs
        # the compaction step server-side once the rendered prompt
        # crosses the threshold and replies with a top-level
        # `context_management` block plus `usage.iterations[]` so we can
        # account per-iteration. Below-min thresholds get clamped up to
        # 50K so the request doesn't 400.
        compaction_active = (
            compaction_threshold is not None
            and compaction_threshold > 0
            and _anthropic_supports_compaction(model)
        )
        if compaction_active:
            trigger_value = max(
                int(compaction_threshold),
                _ANTHROPIC_COMPACTION_MIN,
            )
            body["context_management"] = {
                "edits": [
                    {
                        "type": _ANTHROPIC_COMPACTION_TYPE,
                        "trigger": {
                            "type": "input_tokens",
                            "value": trigger_value,
                        },
                    }
                ]
            }

        url = f"{self.base_url}/messages"
        completion_id = f"chatcmpl-anthropic-{model.replace('/', '-')}"

        # Log the outgoing config keys (not the messages themselves) so we
        # can prove which thinking/effort fields actually reached the wire.
        # If Anthropic skips reasoning despite a configured effort, this
        # tells us whether we sent the field or dropped it on the floor.
        logger.info(
            "Anthropic request shape (model=%s, has_thinking=%s, thinking=%s, "
            "output_config=%s, temperature=%s, has_top_p=%s, has_top_k=%s, "
            "max_tokens=%s)",
            model,
            "thinking" in body,
            body.get("thinking"),
            body.get("output_config"),
            body.get("temperature"),
            "top_p" in body,
            "top_k" in body,
            body.get("max_tokens"),
        )

        # Translate Anthropic stop reasons onto the OpenAI chat-completions
        # `finish_reason` vocabulary. `pause_turn` maps to None so the
        # adapter does NOT emit a finish_reason chunk: pause_turn means
        # Claude paused a long server-tool turn (web_search / web_fetch)
        # and will continue once the user (or our retry) sends back the
        # partial assistant message. Forwarding it as "stop" makes the
        # OpenAI client think the answer is done and truncates the
        # rendered message. `refusal` maps to "content_filter" as the
        # nearest semantic match. See
        #   https://platform.claude.com/docs/en/api/messages#response-stop-reason
        _finish_reason_map: dict[str, Optional[str]] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "refusal": "content_filter",
            "pause_turn": None,
        }

        logger.info("Proxying Anthropic Messages API to %s (model=%s)", url, model)

        request_headers = self._auth_headers()
        # Anthropic accepts comma-separated beta features in a single
        # `anthropic-beta` header. Merge our flags onto whatever the
        # registry's extra_headers contributed (currently nothing on
        # the beta axis, just anthropic-version) so future betas
        # added at the registry level keep working.
        existing_beta = request_headers.get("anthropic-beta", "").strip()
        beta_parts = (
            [p.strip() for p in existing_beta.split(",") if p.strip()]
            if existing_beta
            else []
        )
        if code_execution_enabled and _ANTHROPIC_CODE_EXECUTION_BETA not in beta_parts:
            beta_parts.append(_ANTHROPIC_CODE_EXECUTION_BETA)
        if compaction_active and _ANTHROPIC_COMPACTION_BETA not in beta_parts:
            beta_parts.append(_ANTHROPIC_COMPACTION_BETA)
        if beta_parts:
            request_headers["anthropic-beta"] = ",".join(beta_parts)

        try:
            async with _http_client.stream(
                "POST",
                url,
                json = body,
                headers = request_headers,
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    logger.error(
                        "Anthropic returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    # Stale container detection (mirror of the OpenAI
                    # path). When we sent a `container` field and the
                    # response is 4xx with any hint that the id is
                    # expired / missing, emit container_invalidated so
                    # the chat adapter clears the stored id and the
                    # next turn falls back to auto-create.
                    if (
                        anthropic_code_exec_container_id
                        and 400 <= response.status_code < 500
                    ):
                        lowered = error_text.lower()
                        if "container" in lowered and (
                            "expired" in lowered
                            or "not_found" in lowered
                            or "not found" in lowered
                            or "no such container" in lowered
                            or "invalid" in lowered
                        ):
                            yield (
                                f"data: "
                                f"{_json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': None}], '_toolEvent': {'type': 'container_invalidated'}})}"
                            )
                    yield _error_sse_line(
                        response.status_code, error_text, self.provider_type
                    )
                    return

                # NOTE: same manual __anext__ loop as stream_chat_completion — see comment there.
                lines_gen = response.aiter_lines().__aiter__()
                thinking_open = False
                # Diagnostic counters for the next time the user reports
                # "no thinking content" — distinguishes "Anthropic never sent
                # thinking_delta" from "frontend didn't render the chunks".
                event_counts: dict[str, int] = {}
                # web_search state. Anthropic emits the query inside an
                # `input_json_delta` stream on a `server_tool_use` content
                # block, then a separate `web_search_tool_result` block
                # with the URL list. Unlike OpenAI we get per-call results
                # directly, so each tool card carries its own citations.
                # `current_server_tool_use`: {id, name, partial_json_buffer}
                # `current_result_block`: {tool_use_id, results}
                # Both go to None when the matching content_block_stop fires.
                current_server_tool_use: Optional[dict[str, Any]] = None
                current_result_block: Optional[dict[str, Any]] = None
                web_search_calls: dict[str, dict[str, Any]] = {}
                # code_execution state. Anthropic's
                # `code_execution_20250825` tool emits the same
                # server_tool_use → *_tool_result block shape as
                # web_search, but the server_tool_use carries one of
                # two sub-tool names (`bash_code_execution` or
                # `text_editor_code_execution`) and the result block
                # type matches (`bash_code_execution_tool_result` /
                # `text_editor_code_execution_tool_result`). Kept
                # parallel to web_search state so the two paths don't
                # collide when both pills are on in the same turn.
                current_code_exec_use: Optional[dict[str, Any]] = None
                current_code_exec_result: Optional[dict[str, Any]] = None
                code_execution_calls: dict[str, dict[str, Any]] = {}
                # web_fetch state. Same server_tool_use → *_tool_result
                # block shape as web_search but the server_tool_use
                # carries name="web_fetch" and the result block is
                # `web_fetch_tool_result` with content.type=
                # `web_fetch_result` (success) or `web_fetch_tool_error`
                # (failure). Kept separate from web_search state so a
                # turn that uses both does not collide.
                current_web_fetch_use: Optional[dict[str, Any]] = None
                current_web_fetch_result: Optional[dict[str, Any]] = None
                web_fetch_calls: dict[str, dict[str, Any]] = {}
                # Compaction state. Server-side compaction emits a
                # `{type:"compaction", content:"..."}` content block
                # whenever it runs. The summary text can land on the
                # start event AND/OR via text_delta events on the same
                # block (Anthropic's wire format is permissive here).
                # Accumulate in `current_compaction["content"]` and emit
                # on content_block_stop so the chat-adapter can persist
                # it onto the assistant message for round-tripping on
                # the next turn.
                current_compaction: Optional[dict[str, Any]] = None
                compaction_blocks_seen = 0
                # Counts surfaced in the final log line so reports of
                # "Code execution did nothing" can be triaged at a
                # glance. generated_files_count is interesting for the
                # future Files API PR — when bash creates files inside
                # the container, they show up as file_id entries on
                # bash_code_execution_result.content, and v1 drops
                # them. Track the count so we know how often it would
                # have mattered.
                code_execution_generated_files = 0
                # Container id captured from `message_start.message.container.id`
                # when code_execution is enabled. Emit a `container_ready`
                # _toolEvent on first sight so the chat adapter persists it
                # on the thread record. Only emitted when the value differs
                # from the inbound id — no churn on reuse.
                latched_container_id: Optional[str] = None
                container_id_emitted = False
                # Cache usage tracking. message_start carries the input
                # accounting (incl. cache_creation_input_tokens and
                # cache_read_input_tokens); message_delta carries cumulative
                # output_tokens. Both are surfaced in the "stream complete"
                # log so prompt caching can be verified per-request without
                # opening the Anthropic dashboard.
                last_usage: dict[str, Any] = {}

                def _content_chunk(text: str) -> str:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    return f"data: {_json.dumps(chunk)}"

                def _emit_tool_event(payload: dict[str, Any]) -> str:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": None,
                            }
                        ],
                        "_toolEvent": payload,
                    }
                    return f"data: {_json.dumps(chunk)}"

                def _format_web_search_results(
                    results: list[Any],
                ) -> str:
                    blocks: list[str] = []
                    for r in results:
                        if not isinstance(r, dict):
                            continue
                        if r.get("type") != "web_search_result":
                            continue
                        url = r.get("url", "")
                        title = r.get("title") or url
                        if not url:
                            continue
                        blocks.append(f"Title: {title}\nURL: {url}")
                    return "\n---\n".join(blocks)

                def _format_web_fetch_result(inner: dict[str, Any]) -> str:
                    """Render a `web_fetch_tool_result.content` payload
                    as the Title / URL / snippet block CodeExecutionToolUI
                    and parseSourcesFromResult already expect from the
                    web_search path.

                    Success shape (text):
                        {type: web_fetch_result, url, retrieved_at,
                         content: {type: document, source: {type: text,
                                   media_type, data}, title?}}
                    Success shape (pdf): source.type=base64 + media_type=
                        application/pdf. We do not surface the base64
                        bytes; the title + url is enough for the source
                        pill, and the model still sees the document
                        contents on its side.
                    Error shape: {type: web_fetch_tool_error, error_code}.
                    """
                    inner_type = inner.get("type") or ""
                    if inner_type == "web_fetch_tool_error":
                        return f"Error: {inner.get('error_code', 'unknown')}"
                    url = inner.get("url", "")
                    document = inner.get("content") or {}
                    title = ""
                    snippet = ""
                    if isinstance(document, dict):
                        title = document.get("title") or ""
                        source = document.get("source") or {}
                        if isinstance(source, dict):
                            media_type = source.get("media_type") or ""
                            data = source.get("data") or ""
                            # Inline a short text preview so the source
                            # pill carries usable context; skip for PDFs
                            # since the body is base64-encoded.
                            if (
                                media_type.startswith("text/")
                                and isinstance(data, str)
                                and data
                            ):
                                snippet = data[:240].strip()
                    # Frontend parseSourcesFromResult only emits a source
                    # pill when both `Title:` and `URL:` are present, so
                    # fall back to the URL when Anthropic omits the
                    # document title (matches the web_search formatter).
                    if not title and url:
                        title = url
                    parts: list[str] = []
                    if title:
                        parts.append(f"Title: {title}")
                    if url:
                        parts.append(f"URL: {url}")
                    if snippet:
                        parts.append(f"Snippet: {snippet}")
                    return "\n".join(parts) if parts else "(fetch complete)"

                def _format_code_execution_result(
                    inner: dict[str, Any],
                ) -> str:
                    """Render an Anthropic code-execution result block as
                    the preformatted text payload the frontend's
                    CodeExecutionToolUI displays inside a <pre>. Handles
                    bash, text_editor (view/create/str_replace), and the
                    matching error variants.
                    """
                    inner_type = inner.get("type") or ""
                    if inner_type.endswith("_error"):
                        return f"Error: {inner.get('error_code', 'unknown')}"
                    if inner_type == "bash_code_execution_result":
                        stdout = inner.get("stdout") or ""
                        stderr = inner.get("stderr") or ""
                        return_code = inner.get("return_code")
                        parts: list[str] = []
                        if stdout:
                            parts.append(stdout)
                        if stderr:
                            parts.append(f"--- stderr ---\n{stderr}")
                        if isinstance(return_code, int) and return_code != 0:
                            parts.append(f"return_code: {return_code}")
                        return "\n".join(parts) if parts else "(no output)"
                    if inner_type == "text_editor_code_execution_result":
                        # view: file content; create: is_file_update flag;
                        # str_replace: diff `lines` list. The matching
                        # server_tool_use carries the command + path, but
                        # that's encoded into the tool_start arguments
                        # already — here we only format the result body.
                        if "lines" in inner and isinstance(inner.get("lines"), list):
                            return "\n".join(str(line) for line in inner["lines"])
                        if "is_file_update" in inner:
                            return (
                                "Updated" if inner.get("is_file_update") else "Created"
                            )
                        content_field = inner.get("content")
                        if isinstance(content_field, str):
                            return content_field
                        return "(file operation complete)"
                    return "(code execution complete)"

                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line or line.startswith("event:"):
                            continue
                        if not line.startswith("data:"):
                            continue

                        data_str = line[len("data:") :].strip()
                        if not data_str:
                            continue

                        try:
                            event = _json.loads(data_str)
                        except _json.JSONDecodeError:
                            continue

                        event_type = event.get("type")
                        if event_type == "content_block_delta":
                            delta_kind = (event.get("delta") or {}).get("type")
                            key = f"{event_type}:{delta_kind}"
                        else:
                            key = event_type or "<unknown>"
                        event_counts[key] = event_counts.get(key, 0) + 1

                        # message_start carries the input-side usage block
                        # including cache_creation_input_tokens and
                        # cache_read_input_tokens. message_delta updates
                        # output_tokens (and may overwrite the input fields
                        # with final values). Merge both into last_usage.
                        if event_type == "message_start":
                            start_usage = (event.get("message") or {}).get("usage")
                            if isinstance(start_usage, dict):
                                last_usage.update(start_usage)

                        if event_type == "content_block_start":
                            content_block = event.get("content_block") or {}
                            block_type = content_block.get("type")
                            block_name = content_block.get("name")
                            if (
                                block_type == "server_tool_use"
                                and block_name == "web_search"
                            ):
                                tool_use_id = content_block.get("id", "") or (
                                    f"ws_{len(web_search_calls)}"
                                )
                                current_server_tool_use = {
                                    "id": tool_use_id,
                                    "buffer": "",
                                }
                                web_search_calls[tool_use_id] = {
                                    "query": "",
                                    "results": [],
                                }
                            elif block_type == "web_search_tool_result":
                                tool_use_id = content_block.get("tool_use_id", "")
                                # Anthropic sometimes ships the full results
                                # list on the start event; sometimes deltas
                                # follow. Capture whatever is present and
                                # finalize on content_block_stop.
                                content = content_block.get("content") or []
                                current_result_block = {
                                    "tool_use_id": tool_use_id,
                                    "results": list(content)
                                    if isinstance(content, list)
                                    else [],
                                }
                            elif (
                                block_type == "server_tool_use"
                                and block_name == "web_fetch"
                            ):
                                tool_use_id = content_block.get("id", "") or (
                                    f"wf_{len(web_fetch_calls)}"
                                )
                                current_web_fetch_use = {
                                    "id": tool_use_id,
                                    "buffer": "",
                                }
                                web_fetch_calls[tool_use_id] = {
                                    "url": "",
                                    "result": None,
                                }
                            elif block_type == "web_fetch_tool_result":
                                tool_use_id = content_block.get("tool_use_id", "")
                                inner = content_block.get("content") or {}
                                current_web_fetch_result = {
                                    "tool_use_id": tool_use_id,
                                    "inner": inner if isinstance(inner, dict) else {},
                                }
                            elif block_type == "server_tool_use" and block_name in (
                                "bash_code_execution",
                                "text_editor_code_execution",
                            ):
                                tool_use_id = content_block.get("id", "") or (
                                    f"ce_{len(code_execution_calls)}"
                                )
                                kind = (
                                    "bash"
                                    if block_name == "bash_code_execution"
                                    else "text_editor"
                                )
                                current_code_exec_use = {
                                    "id": tool_use_id,
                                    "kind": kind,
                                    "buffer": "",
                                }
                                code_execution_calls[tool_use_id] = {
                                    "kind": kind,
                                    "arguments": {},
                                    "result": None,
                                }
                            elif block_type in (
                                "bash_code_execution_tool_result",
                                "text_editor_code_execution_tool_result",
                            ):
                                # Anthropic ships the full result content
                                # on the start event for code-exec result
                                # blocks (unlike web_search, which can
                                # split across deltas). Capture it and
                                # finalize on content_block_stop so the
                                # ordering matches the web_search path.
                                tool_use_id = content_block.get("tool_use_id", "")
                                inner = content_block.get("content") or {}
                                current_code_exec_result = {
                                    "tool_use_id": tool_use_id,
                                    "inner": inner if isinstance(inner, dict) else {},
                                }
                            elif block_type == "compaction":
                                # Server-side compaction emits a `compaction`
                                # content block on the assistant message.
                                # Anthropic may include the summary text on
                                # this start event AND/OR stream it via
                                # text_delta events on the same block. See
                                #   https://platform.claude.com/docs/en/build-with-claude/compaction
                                # Capture either form; finalize and emit
                                # on content_block_stop. The chat-adapter
                                # persists the block onto the assistant
                                # message so the next turn's request
                                # carries it back -- Anthropic then skips
                                # re-compaction from scratch.
                                seed = content_block.get("content") or ""
                                current_compaction = {
                                    "content": seed if isinstance(seed, str) else "",
                                }

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            delta_type = delta.get("type")
                            if delta_type == "thinking_delta":
                                # Anthropic streams extended-thinking content as
                                # thinking_delta events on a separate content
                                # block. Wrap as inline <think>...</think> so
                                # the frontend's parseAssistantContent lifts it
                                # into the reasoning panel — same pattern as
                                # the OpenAI Responses path.
                                thinking_text = delta.get("thinking", "")
                                if thinking_text:
                                    if not thinking_open:
                                        thinking_text = f"<think>{thinking_text}"
                                        thinking_open = True
                                    yield _content_chunk(thinking_text)
                            elif delta_type == "text_delta":
                                text = delta.get("text", "")
                                # text_deltas inside a compaction block
                                # carry the summary chunks; route them
                                # into the compaction buffer and DON'T
                                # yield them to the user-visible stream
                                # -- the summary is opaque internal
                                # state, not assistant prose.
                                if current_compaction is not None:
                                    if text:
                                        current_compaction["content"] += text
                                else:
                                    # First text after a thinking block closes the
                                    # <think> tag we opened above. Anthropic emits
                                    # a content_block_stop between blocks, but
                                    # closing on the text_delta transition is more
                                    # forgiving if events arrive out of order.
                                    if thinking_open:
                                        yield _content_chunk("</think>")
                                        thinking_open = False
                                    if text:
                                        yield _content_chunk(text)
                                    # Citations on text deltas are attached
                                    # per-call by Anthropic via the
                                    # `web_search_tool_result` block; we don't
                                    # need to scrape them off the text events.
                            elif delta_type == "input_json_delta":
                                # Streamed partial_json carrying tool inputs
                                # — the search query for web_search, or the
                                # command/path/etc. for code execution.
                                # Route to whichever buffer is open. The two
                                # state slots are exclusive in practice
                                # (Anthropic doesn't interleave tool input
                                # streams), but checking both keeps the
                                # dispatch robust if that ever changes.
                                partial = delta.get("partial_json", "")
                                if current_server_tool_use is not None:
                                    current_server_tool_use["buffer"] += partial
                                elif current_code_exec_use is not None:
                                    current_code_exec_use["buffer"] += partial
                                elif current_web_fetch_use is not None:
                                    current_web_fetch_use["buffer"] += partial
                            # signature_delta and any other delta types are
                            # intentionally skipped — they carry trust /
                            # verification metadata, not user-visible content.

                        elif event_type == "content_block_stop":
                            if current_server_tool_use is not None:
                                # End of the server_tool_use block — parse the
                                # accumulated input_json into a query and
                                # emit tool_start. The matching tool_end fires
                                # later when the web_search_tool_result block
                                # closes with the actual results.
                                buffer = current_server_tool_use["buffer"]
                                query = ""
                                if buffer:
                                    try:
                                        parsed = _json.loads(buffer)
                                        if isinstance(parsed, dict):
                                            q = parsed.get("query", "")
                                            if isinstance(q, str):
                                                query = q
                                    except Exception:
                                        query = ""
                                tool_use_id = current_server_tool_use["id"]
                                if tool_use_id in web_search_calls:
                                    web_search_calls[tool_use_id]["query"] = query
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_start",
                                        "tool_name": "web_search",
                                        "tool_call_id": tool_use_id,
                                        "arguments": (
                                            {"query": query} if query else {}
                                        ),
                                    }
                                )
                                current_server_tool_use = None
                            elif current_result_block is not None:
                                # End of a web_search_tool_result — emit
                                # tool_end carrying the search results as
                                # Title:/URL: blocks. parseSourcesFromResult
                                # on the frontend lifts these into source
                                # pills at message tail.
                                tool_use_id = current_result_block["tool_use_id"]
                                results = current_result_block["results"]
                                if tool_use_id in web_search_calls:
                                    web_search_calls[tool_use_id]["results"] = results
                                result_text = _format_web_search_results(results)
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": tool_use_id,
                                        "result": (result_text or "(search complete)"),
                                    }
                                )
                                current_result_block = None
                            elif current_code_exec_use is not None:
                                # End of a code-execution server_tool_use —
                                # parse the buffered input_json into a
                                # {command, path, ...} dict and emit
                                # tool_start. The matching tool_end fires
                                # on the result block's content_block_stop.
                                buffer = current_code_exec_use["buffer"]
                                parsed_args: dict[str, Any] = {}
                                if buffer:
                                    try:
                                        parsed_obj = _json.loads(buffer)
                                        if isinstance(parsed_obj, dict):
                                            parsed_args = parsed_obj
                                    except Exception:
                                        parsed_args = {}
                                tool_use_id = current_code_exec_use["id"]
                                kind = current_code_exec_use["kind"]
                                emit_args = {"kind": kind, **parsed_args}
                                if tool_use_id in code_execution_calls:
                                    code_execution_calls[tool_use_id]["arguments"] = (
                                        emit_args
                                    )
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_start",
                                        "tool_name": "code_execution",
                                        "tool_call_id": tool_use_id,
                                        "arguments": emit_args,
                                    }
                                )
                                current_code_exec_use = None
                            elif current_compaction is not None:
                                # End of a compaction block. Emit it as a
                                # synthetic tool_event so the chat-adapter
                                # can persist the {type:"compaction",
                                # content:"..."} payload onto the
                                # assistant message. The next turn's
                                # request body forwards the content_part
                                # verbatim and Anthropic recognises it
                                # as the prior compaction state.
                                compaction_blocks_seen += 1
                                yield _emit_tool_event(
                                    {
                                        "type": "compaction_block",
                                        "content": current_compaction["content"],
                                    }
                                )
                                current_compaction = None
                            elif current_code_exec_result is not None:
                                # End of a code-execution result block —
                                # format the inner result into the text
                                # payload CodeExecutionToolUI renders.
                                tool_use_id = current_code_exec_result["tool_use_id"]
                                inner = current_code_exec_result["inner"]
                                # Track generated-file count for the
                                # follow-up Files API PR. v1 drops them.
                                if isinstance(inner, dict):
                                    file_blocks = inner.get("content")
                                    if isinstance(file_blocks, list):
                                        for entry in file_blocks:
                                            if isinstance(entry, dict) and entry.get(
                                                "file_id"
                                            ):
                                                code_execution_generated_files += 1
                                result_text = _format_code_execution_result(
                                    inner if isinstance(inner, dict) else {}
                                )
                                if tool_use_id in code_execution_calls:
                                    code_execution_calls[tool_use_id]["result"] = (
                                        result_text
                                    )
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": tool_use_id,
                                        "result": result_text,
                                    }
                                )
                                current_code_exec_result = None
                            elif current_web_fetch_use is not None:
                                # End of the web_fetch server_tool_use —
                                # parse the buffered input_json into the
                                # URL the model asked Anthropic to fetch
                                # and emit tool_start. The matching
                                # tool_end fires on the result block's
                                # content_block_stop just below.
                                buffer = current_web_fetch_use["buffer"]
                                url = ""
                                if buffer:
                                    try:
                                        parsed = _json.loads(buffer)
                                        if isinstance(parsed, dict):
                                            probe = parsed.get("url", "")
                                            if isinstance(probe, str):
                                                url = probe
                                    except Exception:
                                        logger.debug(
                                            "Failed to parse web_fetch input_json",
                                            buffer = buffer,
                                        )
                                        url = ""
                                tool_use_id = current_web_fetch_use["id"]
                                if tool_use_id in web_fetch_calls:
                                    web_fetch_calls[tool_use_id]["url"] = url
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_start",
                                        "tool_name": "web_fetch",
                                        "tool_call_id": tool_use_id,
                                        "arguments": ({"url": url} if url else {}),
                                    }
                                )
                                current_web_fetch_use = None
                            elif current_web_fetch_result is not None:
                                # End of the web_fetch_tool_result —
                                # format Title / URL / snippet for the
                                # frontend source pill and emit tool_end.
                                # `inner` is sanitised to a dict at the
                                # matching content_block_start, and the
                                # formatter always returns a non-empty
                                # string (defaulting to "(fetch complete)"
                                # when no fields are present), so no
                                # extra fallback is needed here.
                                tool_use_id = current_web_fetch_result["tool_use_id"]
                                result_text = _format_web_fetch_result(
                                    current_web_fetch_result["inner"]
                                )
                                if tool_use_id in web_fetch_calls:
                                    web_fetch_calls[tool_use_id]["result"] = result_text
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": tool_use_id,
                                        "result": result_text,
                                    }
                                )
                                current_web_fetch_result = None
                            elif thinking_open:
                                # Close the <think> tag when the thinking block
                                # ends, in case no text_delta follows (e.g.
                                # display=omitted on Claude 4.7, or thinking-
                                # only turns).
                                yield _content_chunk("</think>")
                                thinking_open = False

                        elif event_type == "message_delta":
                            delta_usage = event.get("usage")
                            if isinstance(delta_usage, dict):
                                last_usage.update(delta_usage)
                                # When a fresh compaction has run, Anthropic
                                # publishes per-iteration token counts in
                                # `usage.iterations[]`. The top-level
                                # input_tokens / output_tokens only cover the
                                # `message` iteration, NOT the compaction
                                # passes — billing has to sum the whole
                                # array. See
                                #   https://platform.claude.com/docs/en/build-with-claude/compaction
                                # Fold the compaction iterations into
                                # `compaction_input_tokens` / `compaction_output_tokens`
                                # so the cost surface can add them without
                                # re-walking the array (and so the closing
                                # log line names the figures).
                                iterations = delta_usage.get("iterations")
                                if isinstance(iterations, list):
                                    c_in = 0
                                    c_out = 0
                                    for it in iterations:
                                        if (
                                            isinstance(it, dict)
                                            and it.get("type") == "compaction"
                                        ):
                                            c_in += int(it.get("input_tokens") or 0)
                                            c_out += int(it.get("output_tokens") or 0)
                                    if c_in or c_out:
                                        last_usage["compaction_input_tokens"] = c_in
                                        last_usage["compaction_output_tokens"] = c_out
                            # Anthropic reports the code_execution container
                            # id on `message_delta.delta.container.{id,
                            # expires_at}` (NOT on message_start — at start
                            # the container hasn't been provisioned yet).
                            # Latch on first sight and emit container_ready
                            # only when the value differs from the inbound
                            # id, so steady-state reuse doesn't re-write
                            # the same id to the thread record every turn.
                            delta_obj = event.get("delta") or {}
                            container_obj = delta_obj.get("container")
                            if (
                                isinstance(container_obj, dict)
                                and latched_container_id is None
                            ):
                                probe = container_obj.get("id")
                                if isinstance(probe, str) and probe:
                                    latched_container_id = probe
                            if (
                                latched_container_id
                                and not container_id_emitted
                                and latched_container_id
                                != anthropic_code_exec_container_id
                            ):
                                yield _emit_tool_event(
                                    {
                                        "type": "container_ready",
                                        "container_id": latched_container_id,
                                    }
                                )
                                container_id_emitted = True
                            stop_reason = event.get("delta", {}).get("stop_reason")
                            if stop_reason:
                                if thinking_open:
                                    yield _content_chunk("</think>")
                                    thinking_open = False
                                # `pause_turn` is in-progress, not terminal:
                                # the SSE stream still ends with [DONE] via
                                # message_stop but we skip emitting a
                                # finish_reason="stop" chunk that would
                                # truncate the rendered message in the UI.
                                mapped = _finish_reason_map.get(stop_reason, "stop")
                                if mapped is not None:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": mapped,
                                            }
                                        ],
                                    }
                                    yield f"data: {_json.dumps(chunk)}"

                        elif event_type == "message_stop":
                            if thinking_open:
                                yield _content_chunk("</think>")
                                thinking_open = False
                            # Final include_usage-style chunk so callers can
                            # see cache_creation / cache_read without
                            # scraping the server log.
                            usage_line = _build_usage_chunk(
                                completion_id,
                                "anthropic",
                                last_usage,
                            )
                            if usage_line:
                                yield usage_line
                            yield "data: [DONE]"
                            await (
                                response.aclose()
                            )  # set PoolByteStream._closed=True FIRST
                            break
                except GeneratorExit:
                    await response.aclose()  # set PoolByteStream._closed=True FIRST
                    await lines_gen.aclose()  # now safe — aclose() is a no-op
                    raise
                finally:
                    # Surface per-event-type counts + web_search summary so
                    # reports of "no reasoning panel content" / "Search
                    # didn't do anything" can be triaged at a glance.
                    web_search_requested = bool(
                        enabled_tools and "web_search" in enabled_tools
                    )
                    web_search_invocations = len(web_search_calls)
                    total_results = sum(
                        len(sc.get("results") or []) for sc in web_search_calls.values()
                    )
                    queries = [
                        sc["query"]
                        for sc in web_search_calls.values()
                        if sc.get("query")
                    ]
                    # cache_read_input_tokens > 0 on turn N proves the
                    # cache_control marker on the system block is doing
                    # its job — turn 1 will show cache_creation > 0
                    # instead. cache_creation tokens are billed at a
                    # small premium; cache_read tokens are billed at a
                    # discount.
                    code_execution_invocations = len(code_execution_calls)
                    code_execution_results = sum(
                        1
                        for c in code_execution_calls.values()
                        if c.get("result") is not None
                    )
                    web_fetch_requested = web_fetch_enabled
                    web_fetch_invocations = len(web_fetch_calls)
                    web_fetch_urls = [
                        wf["url"] for wf in web_fetch_calls.values() if wf.get("url")
                    ]
                    logger.info(
                        "Anthropic stream complete (model=%s, "
                        "web_search_requested=%s, web_search_invocations=%s, "
                        "results=%s, queries=%s, "
                        "web_fetch_requested=%s, web_fetch_invocations=%s, "
                        "web_fetch_urls=%s, "
                        "code_execution_requested=%s, "
                        "code_execution_invocations=%s, "
                        "code_execution_results=%s, "
                        "code_execution_generated_files=%s, "
                        "container_id_in=%s, container_id_out=%s, "
                        "input_tokens=%s, output_tokens=%s, "
                        "cache_creation_input_tokens=%s, "
                        "cache_read_input_tokens=%s, "
                        "compaction_input_tokens=%s, "
                        "compaction_output_tokens=%s, "
                        "compaction_blocks_seen=%s, events=%s)",
                        model,
                        web_search_requested,
                        web_search_invocations,
                        total_results,
                        queries,
                        web_fetch_requested,
                        web_fetch_invocations,
                        web_fetch_urls,
                        code_execution_enabled,
                        code_execution_invocations,
                        code_execution_results,
                        code_execution_generated_files,
                        anthropic_code_exec_container_id,
                        latched_container_id,
                        last_usage.get("input_tokens"),
                        last_usage.get("output_tokens"),
                        last_usage.get("cache_creation_input_tokens"),
                        last_usage.get("cache_read_input_tokens"),
                        last_usage.get("compaction_input_tokens"),
                        last_usage.get("compaction_output_tokens"),
                        compaction_blocks_seen,
                        event_counts,
                    )
                    await response.aclose()
                    await lines_gen.aclose()

        except httpx.ConnectError as exc:
            logger.error("Connection error to %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Failed to connect to {self.provider_type}: {exc}",
                self.provider_type,
            )
        except httpx.ReadTimeout as exc:
            logger.error("Read timeout from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                504,
                f"Timeout waiting for {self.provider_type} response",
                self.provider_type,
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP error from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Error communicating with {self.provider_type}: {exc}",
                self.provider_type,
            )

    async def _stream_gemini(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        top_k: Optional[int] = None,
        presence_penalty: float = 0.0,
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[Any] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Call Google's native Gemini API and translate its streaming
        ``streamGenerateContent`` response into OpenAI Chat Completions
        chunk format.

        Gemini does NOT speak the OpenAI Chat Completions contract on
        its primary endpoint. The wire shape is:

          POST /v1beta/models/{model}:streamGenerateContent?alt=sse
          {
            "contents": [{"role": "user|model", "parts": [{"text": "..."}]}],
            "systemInstruction": {"parts": [{"text": "..."}]},
            "generationConfig": {"temperature": 0.7, "topP": 0.95, "topK": 40,
                                  "maxOutputTokens": 1024},
            "tools": [{"googleSearch": {}}, {"codeExecution": {}}],
            "cachedContent": "<cache name>"  // optional, see caching docs
          }

        Streamed responses are SSE frames carrying partial
        ``GenerateContentResponse`` objects:

          {"candidates": [{"content": {"parts": [{"text": "Hello"}]},
                            "finishReason": "STOP"}],
           "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}

        Image generation uses the same endpoint with model
        ``gemini-2.5-flash-image`` (also called Nano Banana); the
        response carries an ``inlineData`` part with the base64 PNG
        bytes and a ``mimeType``. We surface that through the same
        ``tool_start`` / ``tool_end`` ``image_b64`` envelope the OpenAI
        image_generation path uses, so the chat UI renders the image
        inline with no extra plumbing.

        References:
          - https://ai.google.dev/gemini-api/docs/text-generation
          - https://ai.google.dev/gemini-api/docs/function-calling
          - https://ai.google.dev/gemini-api/docs/grounding
          - https://ai.google.dev/gemini-api/docs/caching
          - https://ai.google.dev/gemini-api/docs/image-generation
        """
        import json as _json

        # Translate OpenAI messages -> Gemini contents. The `system`
        # role becomes a top-level `systemInstruction`; user / assistant
        # turns map to role="user" / role="model" with `parts` carrying
        # text (and for vision turns, inline image data via
        # `inlineData`).
        system_text_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        # OpenAI sometimes drops ``name`` from the role="tool" follow-up
        # and only carries ``tool_call_id``. Remember the function names
        # we emitted on prior assistant turns so the matching response
        # can recover its name (Gemini rejects an empty functionResponse
        # name with HTTP 400).
        tool_call_names: dict[str, str] = {}
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if isinstance(content, str):
                    if content:
                        system_text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "text"
                            and part.get("text")
                        ):
                            system_text_parts.append(part["text"])
                continue
            # Map OpenAI roles to Gemini's two-role contract.
            gemini_role = "model" if role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        text = part.get("text", "")
                        if text:
                            parts.append({"text": text})
                    elif ptype == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            header, _, b64data = url.partition(",")
                            media_type = (
                                header.split(";")[0].replace("data:", "")
                                or "image/jpeg"
                            )
                            if b64data:
                                parts.append(
                                    {
                                        "inlineData": {
                                            "mimeType": media_type,
                                            "data": b64data,
                                        }
                                    }
                                )
                        elif url:
                            # Remote image. Gemini's `fileData` part takes
                            # a `fileUri` for both Files-API uploads and
                            # public https URLs. Guess MIME from the URL
                            # path so PNG/WebP/GIF inputs are not mislabeled
                            # as JPEG (Gemini rejects mismatched mime).
                            _path = urlparse(url).path
                            _guessed, _ = mimetypes.guess_type(_path)
                            _media_type = (
                                _guessed
                                if isinstance(_guessed, str)
                                and _guessed.startswith("image/")
                                else "image/jpeg"
                            )
                            parts.append(
                                {
                                    "fileData": {
                                        "fileUri": url,
                                        "mimeType": _media_type,
                                    }
                                }
                            )
            # OpenAI may attach tool_calls on an assistant message.
            # Translate into Gemini's functionCall part so the prior
            # turn's tool request round-trips back to the model.
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    if not isinstance(fn, dict):
                        continue
                    args_raw = fn.get("arguments") or "{}"
                    if isinstance(args_raw, str):
                        try:
                            args = _json.loads(args_raw)
                        except Exception:
                            args = {"_raw": args_raw}
                    elif isinstance(args_raw, dict):
                        args = args_raw
                    else:
                        args = {}
                    fn_name = fn.get("name", "")
                    tc_id = tc.get("id")
                    if fn_name and isinstance(tc_id, str) and tc_id:
                        tool_call_names[tc_id] = fn_name
                    # Forward the OpenAI tool_call id into Gemini's
                    # functionCall.id so a follow-up turn that issues
                    # multiple calls to the same function (different
                    # args, same name) can be disambiguated on the
                    # response side. Gemini accepts the field per
                    # https://ai.google.dev/gemini-api/docs/function-calling.
                    function_call_part: dict[str, Any] = {
                        "name": fn_name,
                        "args": args,
                    }
                    if isinstance(tc_id, str) and tc_id:
                        function_call_part["id"] = tc_id
                    # Gemini 3 function-calling requires the prior
                    # thoughtSignature to be echoed back as a sibling
                    # of the functionCall part. The translator stows
                    # it on the assistant tool_call via
                    # `extra_content.google.thought_signature` (see
                    # the inbound emit below).
                    fc_part: dict[str, Any] = {"functionCall": function_call_part}
                    extra = tc.get("extra_content")
                    if isinstance(extra, dict):
                        google_extra = extra.get("google") or {}
                        if isinstance(google_extra, dict):
                            sig = google_extra.get(
                                "thought_signature"
                            ) or google_extra.get("thoughtSignature")
                            if isinstance(sig, str) and sig:
                                fc_part["thoughtSignature"] = sig
                    parts.append(fc_part)
            if role == "tool":
                # OpenAI's role="tool" follow-up carries the function
                # result. Gemini's matching shape is a role="user" turn
                # with a functionResponse part. When the caller dropped
                # ``name``, recover it from the matching assistant
                # tool_call so Gemini doesn't 400 on an empty name.
                tool_name = msg.get("name") or msg.get("tool_name") or ""
                if not tool_name:
                    tc_id = msg.get("tool_call_id")
                    if isinstance(tc_id, str) and tc_id in tool_call_names:
                        tool_name = tool_call_names[tc_id]
                response_payload: Any
                if isinstance(content, str):
                    try:
                        response_payload = _json.loads(content)
                    except Exception:
                        response_payload = {"result": content}
                else:
                    response_payload = content or {}
                function_response_part: dict[str, Any] = {
                    "name": tool_name,
                    "response": (
                        response_payload
                        if isinstance(response_payload, dict)
                        else {"result": response_payload}
                    ),
                }
                # Mirror tool_call_id onto functionResponse.id so
                # Gemini can match the result to the originating
                # functionCall when multiple parallel calls were made.
                tc_id = msg.get("tool_call_id")
                if isinstance(tc_id, str) and tc_id:
                    function_response_part["id"] = tc_id
                parts = [{"functionResponse": function_response_part}]
                gemini_role = "user"
            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        body: dict[str, Any] = {"contents": contents}
        if system_text_parts:
            body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_text_parts)}]
            }

        # Generation config -- temperature / topP / topK / maxOutputTokens
        # map straight across. The frontend capability matrix restricts
        # the sliders the UI exposes for Gemini to this set.
        gen_config: dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["topP"] = top_p
        if top_k is not None and top_k > 0:
            gen_config["topK"] = top_k
        # Gemini accepts ``presencePenalty`` on generationConfig with the
        # same sign convention as the OpenAI knob (positive discourages
        # repetition). Forward when the caller bothers to set it.
        if presence_penalty:
            gen_config["presencePenalty"] = presence_penalty
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens

        # Nano Banana image generation. When the user picked an image
        # model (id contains `-image` or `nano-banana`) or asked for
        # `image_generation` as a tool, tell Gemini to return image
        # bytes via `responseModalities`. The response carries an
        # `inlineData` part on each candidate which we translate into a
        # tool_end with image_b64/image_mime so the chat UI renders the
        # picture inline. See
        # https://ai.google.dev/gemini-api/docs/image-generation.
        model_lc = model.lower()
        is_image_picker_model = "-image" in model_lc or "nano-banana" in model_lc
        is_image_model = is_image_picker_model or bool(
            enabled_tools and "image_generation" in enabled_tools
        )
        if is_image_model:
            gen_config["responseModalities"] = ["TEXT", "IMAGE"]

        # Thinking control. The Gemini 3 family migrated to a string
        # `thinkingLevel` (LOW/MEDIUM/HIGH/MINIMAL) and rejects sending
        # both `thinkingLevel` + `thinkingBudget`. Gemini 3 also cannot
        # turn thinking fully off -- the "off" position is "minimal" on
        # Flash and "low" on Pro (Pro does not even accept "minimal").
        # https://ai.google.dev/gemini-api/docs/thinking
        # Gemini 2.5 stays on `thinkingBudget` (int; 0 = off on Flash,
        # -1 = dynamic, N > 0 = hard cap). Image models do not benefit
        # from a visible thinking knob and we skip the field entirely
        # so stale UI state does not leak through.
        _GEMINI3_THINKING_PREFIXES = (
            "gemini-3.5-",
            "gemini-3.1-",
            "gemini-3-",
            "gemini-pro-latest",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        )
        _GEMINI3_PRO_PREFIXES = (
            "gemini-3.5-pro",
            "gemini-3.1-pro",
            "gemini-3-pro",
            "gemini-pro-latest",
        )
        _PRO_THINKING_PREFIXES = ("gemini-2.5-pro",)
        is_gemini3_thinking = any(
            model_lc.startswith(p) for p in _GEMINI3_THINKING_PREFIXES
        )
        is_gemini3_pro = any(model_lc.startswith(p) for p in _GEMINI3_PRO_PREFIXES)
        _is_pro_thinking_only = any(
            model_lc == p or model_lc.startswith(p + "-")
            for p in _PRO_THINKING_PREFIXES
        )
        effort_lc = (reasoning_effort or "").strip().lower()
        if not is_image_model and is_gemini3_thinking:
            # Gemini 3.x: thinkingLevel only. Per Google's docs
            # (https://ai.google.dev/gemini-api/docs/thinking):
            #   - Gemini 3 Pro accepts only "low" or "high" (default high).
            #   - Gemini 3 Flash + Flash-Lite accept minimal/low/medium/high
            #     (default medium).
            # Pro tier rejects "minimal" AND "medium"; coerce both to a
            # valid neighbouring level so the UI's medium slider does
            # not 400 on Pro.
            _G3_LEVELS = {"minimal", "low", "medium", "high"}
            level: Optional[str] = None
            if effort_lc in ("none", "off"):
                level = "low" if is_gemini3_pro else "minimal"
            elif effort_lc == "max":
                level = "high"
            elif effort_lc in _G3_LEVELS:
                if is_gemini3_pro:
                    if effort_lc == "minimal":
                        level = "low"
                    elif effort_lc == "medium":
                        level = "high"
                    else:
                        level = effort_lc
                else:
                    level = effort_lc
            elif enable_thinking is True:
                level = "high"
            elif enable_thinking is False:
                level = "low" if is_gemini3_pro else "minimal"
            if level is not None:
                gen_config["thinkingConfig"] = {"thinkingLevel": level}
        elif not is_image_model:
            # Gemini 2.5 / older: thinkingBudget int. Effort -> budget
            # mirrors the OpenAI minimal/low/medium/high ladder so the
            # existing frontend picker maps cleanly.
            # NOTE: gemini-2.5-flash-lite rejects positive budgets below
            # 512 with HTTP 400, so minimal=512 sits at that floor.
            _EFFORT_TO_BUDGET: dict[str, int] = {
                "minimal": 512,
                "low": 2048,
                "medium": 8192,
                "high": 24576,
                "xhigh": -1,
                "max": -1,
            }
            thinking_budget: Optional[int] = None
            if effort_lc == "none" or enable_thinking is False:
                # Pro-tier 2.5 rejects budget=0 (400 "only works in
                # thinking mode"), so coerce to a small positive value.
                thinking_budget = 128 if _is_pro_thinking_only else 0
            elif effort_lc in _EFFORT_TO_BUDGET:
                thinking_budget = _EFFORT_TO_BUDGET[effort_lc]
            elif enable_thinking is True:
                thinking_budget = -1
            if thinking_budget is not None:
                gen_config["thinkingConfig"] = {
                    "thinkingBudget": thinking_budget,
                }

        if gen_config:
            body["generationConfig"] = gen_config

        # Server-side tool wiring.
        # - `{googleSearch: {}}` -- grounded web search; citations come
        #   back on `candidates[0].groundingMetadata.groundingChunks[].web`.
        #   https://ai.google.dev/gemini-api/docs/grounding
        # - `{codeExecution: {}}` -- sandboxed Python tool.
        #   https://ai.google.dev/gemini-api/docs/code-execution
        # Image-mode (responseModalities=[TEXT,IMAGE]) rejects code
        # execution. Google Search grounding is documented as supported
        # on the Gemini 3 image picker family (Nano Banana Pro =
        # gemini-3-pro-image-preview, gemini-3.1-flash-image-preview)
        # but NOT on the older 2.5-flash-image family. Allow Search
        # only on the documented image models; older image models keep
        # the strict "no text tools" gate.
        def _gemini_image_model_allows_google_search(_m: str) -> bool:
            return (
                _m.startswith("gemini-3-pro-image")
                or _m.startswith("gemini-3.1-flash-image")
                or _m.startswith("nano-banana-pro")
                or _m.startswith("nano-banana-2")
            )

        google_search_allowed = (
            not is_image_model or _gemini_image_model_allows_google_search(model_lc)
        )
        code_execution_allowed = not is_image_model
        text_tools_allowed = not is_image_model
        tools_array: list[dict[str, Any]] = []
        if enabled_tools and "web_search" in enabled_tools and google_search_allowed:
            tools_array.append({"googleSearch": {}})
        if (
            enabled_tools
            and "code_execution" in enabled_tools
            and code_execution_allowed
        ):
            tools_array.append({"codeExecution": {}})
        # OpenAI-style function declarations -> Gemini functionDeclarations.
        # https://ai.google.dev/gemini-api/docs/function-calling#step_1
        function_declarations: list[dict[str, Any]] = []
        if tools and text_tools_allowed:
            for _tool in tools:
                if not isinstance(_tool, dict) or _tool.get("type") != "function":
                    continue
                _fn = _tool.get("function")
                if not isinstance(_fn, dict) or not _fn.get("name"):
                    continue
                _decl: dict[str, Any] = {
                    "name": _fn["name"],
                    "description": _fn.get("description") or "",
                }
                _params = _fn.get("parameters")
                if isinstance(_params, dict):
                    _decl["parameters"] = _params
                function_declarations.append(_decl)
        if function_declarations:
            tools_array.append({"functionDeclarations": function_declarations})
        if tools_array:
            body["tools"] = tools_array
        # Tool-choice mapping: OpenAI "auto"/"none"/"required"/{name=...}
        # -> Gemini toolConfig.functionCallingConfig.mode + allowedFunctionNames.
        if tool_choice is not None and function_declarations and text_tools_allowed:
            _mode: Optional[str] = None
            _allowed: Optional[list[str]] = None
            if isinstance(tool_choice, str):
                _tc_lc = tool_choice.strip().lower()
                if _tc_lc == "auto":
                    _mode = "AUTO"
                elif _tc_lc == "none":
                    _mode = "NONE"
                elif _tc_lc in ("required", "any"):
                    _mode = "ANY"
            elif (
                isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
            ):
                _fn_pick = tool_choice.get("function") or {}
                _name = _fn_pick.get("name") if isinstance(_fn_pick, dict) else None
                if isinstance(_name, str) and _name:
                    _mode = "ANY"
                    _allowed = [_name]
            if _mode is not None:
                _fcc: dict[str, Any] = {"mode": _mode}
                if _allowed:
                    _fcc["allowedFunctionNames"] = _allowed
                body["toolConfig"] = {"functionCallingConfig": _fcc}

        # Prompt caching. The Gemini caching contract is "create a
        # CachedContent resource, then pass its name on
        # `cachedContent`". The cache itself is created out of band by
        # the caller via POST /cachedContents; here we forward an
        # explicit cache id when the dispatcher hands us one (a string
        # value on enable_prompt_caching means "use this cache name").
        # https://ai.google.dev/gemini-api/docs/caching
        if isinstance(enable_prompt_caching, str) and enable_prompt_caching:
            body["cachedContent"] = enable_prompt_caching

        url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse"
        completion_id = f"chatcmpl-gemini-{model.replace('/', '-')}"

        logger.info(
            "Proxying Gemini streamGenerateContent to %s (model=%s, "
            "tools=%s, image=%s)",
            url,
            model,
            [list(t.keys())[0] for t in tools_array] if tools_array else [],
            is_image_model,
        )

        def _emit_tool_event(payload: dict[str, Any]) -> str:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None,
                    }
                ],
                "_toolEvent": payload,
            }
            return f"data: {_json.dumps(chunk)}"

        def _text_chunk(
            text: str, extra_content: Optional[dict[str, Any]] = None
        ) -> str:
            delta: dict[str, Any] = {"content": text}
            if extra_content:
                delta["extra_content"] = extra_content
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {_json.dumps(chunk)}"

        def _gemini_part_extra(part: dict[str, Any]) -> Optional[dict[str, Any]]:
            """Return ``{"google": {"thought_signature": ...}}`` when the
            Gemini stream part carries a `thoughtSignature` we need to
            replay on a follow-up turn (Gemini 3 image editing + tool
            contexts both require an exact signature echo)."""
            sig = part.get("thoughtSignature") or part.get("thought_signature")
            if isinstance(sig, str) and sig:
                return {"google": {"thought_signature": sig}}
            return None

        # Gemini finish reasons -> OpenAI vocabulary. Reference:
        # https://ai.google.dev/api/rest/v1beta/Candidate#FinishReason
        _finish_reason_map: dict[str, Optional[str]] = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "PROHIBITED_CONTENT": "content_filter",
            "BLOCKLIST": "content_filter",
            "MALFORMED_FUNCTION_CALL": "stop",
            "OTHER": "stop",
            "FINISH_REASON_UNSPECIFIED": None,
        }

        last_usage: Optional[dict[str, Any]] = None
        emitted_function_call_ids: set[str] = set()
        # True once any Gemini functionCall part has been emitted so the
        # final finish_reason swaps STOP -> tool_calls (matches the
        # OpenAI Chat Completions contract; an OAI client that sees a
        # tool_calls delta followed by finish_reason="stop" never
        # executes the tool).
        emitted_any_function_call = False
        # web_search_active drives the tool_start / tool_end envelope.
        # Track on whether `googleSearch` was actually forwarded above,
        # not the raw caller intent -- image-mode requests filter the
        # tool out, and emitting a phantom "search complete" card on a
        # turn where Gemini was never told to search confuses the UI.
        web_search_active = any("googleSearch" in t for t in tools_array)
        web_search_tool_id = "gemini_web_search"
        web_search_tool_started = False
        web_search_tool_ended = False
        web_search_citations: list[dict[str, str]] = []
        # Tracks the tool_call_id minted on the most recent
        # executableCode part so the matching codeExecutionResult can
        # close out the same envelope. None between rounds.
        gemini_code_exec_pending_id: Optional[str] = None
        # The most recently emitted code_execution id + result text. Kept
        # *after* the tool_end so a following inline image (matplotlib
        # plot rendered by codeExecution) can attach to the same card
        # via a `__IMAGES__:` marker instead of spawning a separate
        # image_generation event.
        last_code_exec_tool_id: Optional[str] = None
        last_code_exec_result_text: str = ""

        try:
            async with _http_client.stream(
                "POST",
                url,
                json = body,
                headers = self._auth_headers(),
                timeout = self._stream_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors = "replace")
                    logger.error(
                        "Gemini returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
                    yield _error_sse_line(
                        response.status_code, error_text, self.provider_type
                    )
                    return

                if web_search_active:
                    yield _emit_tool_event(
                        {
                            "type": "tool_start",
                            "tool_name": "web_search",
                            "tool_call_id": web_search_tool_id,
                            "arguments": {},
                        }
                    )
                    web_search_tool_started = True

                # NOTE: same manual __anext__ loop pattern as the other
                # streaming helpers (see stream_chat_completion for the
                # Python 3.13 + httpcore 1.0.x GeneratorExit ordering).
                lines_gen = response.aiter_lines().__aiter__()
                final_finish_reason: Optional[str] = None
                try:
                    while True:
                        try:
                            line = await lines_gen.__anext__()
                        except StopAsyncIteration:
                            break
                        if not line.strip():
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            event = _json.loads(data_str)
                        except Exception:
                            logger.warning(
                                "Gemini: failed to parse SSE chunk: %s",
                                data_str[:200],
                            )
                            continue
                        if not isinstance(event, dict):
                            continue

                        # Latch usageMetadata across deltas -- the final
                        # fragment carries the complete totals.
                        usage_meta = event.get("usageMetadata")
                        if isinstance(usage_meta, dict):
                            last_usage = usage_meta

                        # Prompt-level safety block: Gemini ships zero
                        # candidates plus a `promptFeedback.blockReason`
                        # (e.g. SAFETY). The downstream OAI client would
                        # otherwise see an empty successful assistant
                        # response. Surface as a content_filter error
                        # event so the UI can render the block reason.
                        prompt_feedback = event.get("promptFeedback")
                        if isinstance(prompt_feedback, dict) and prompt_feedback.get(
                            "blockReason"
                        ):
                            block_reason = str(prompt_feedback.get("blockReason"))
                            # Close out the synthetic web_search start so
                            # the UI does not show a spinner stuck on
                            # "searching..." after the error toast lands.
                            if (
                                web_search_active
                                and web_search_tool_started
                                and not web_search_tool_ended
                            ):
                                yield _emit_tool_event(
                                    {
                                        "type": "tool_end",
                                        "tool_call_id": web_search_tool_id,
                                        "result": (
                                            "(search aborted: Gemini blocked "
                                            f"prompt: {block_reason})"
                                        ),
                                    }
                                )
                                web_search_tool_ended = True
                            yield _error_sse_line(
                                400,
                                f"Gemini blocked prompt: {block_reason}",
                                self.provider_type,
                            )
                            return

                        candidates = event.get("candidates") or []
                        if not isinstance(candidates, list):
                            continue
                        for cand in candidates:
                            if not isinstance(cand, dict):
                                continue
                            # Citations / grounding metadata.
                            # `groundingMetadata.groundingChunks[].web`
                            # carries `uri` + `title`. Collect for the
                            # tool_end emission at stream close.
                            gm = cand.get("groundingMetadata")
                            if isinstance(gm, dict) and web_search_active:
                                chunks_list = gm.get("groundingChunks") or []
                                if isinstance(chunks_list, list):
                                    for ch in chunks_list:
                                        if not isinstance(ch, dict):
                                            continue
                                        web = ch.get("web") or {}
                                        if not isinstance(web, dict):
                                            continue
                                        u = web.get("uri") or ""
                                        if not u or not isinstance(u, str):
                                            continue
                                        if any(
                                            c["url"] == u for c in web_search_citations
                                        ):
                                            continue
                                        web_search_citations.append(
                                            {
                                                "url": u,
                                                "title": (web.get("title") or u),
                                                "snippet": "",
                                            }
                                        )

                            content_obj = cand.get("content") or {}
                            parts = (
                                content_obj.get("parts")
                                if isinstance(content_obj, dict)
                                else None
                            )
                            if isinstance(parts, list):
                                for part in parts:
                                    if not isinstance(part, dict):
                                        continue
                                    # Text delta. Stow part-level
                                    # `thoughtSignature` on the delta so
                                    # Gemini 3 turns that need an exact
                                    # signature echo round-trip cleanly.
                                    text = part.get("text")
                                    if isinstance(text, str) and text:
                                        yield _text_chunk(
                                            text,
                                            extra_content = _gemini_part_extra(part),
                                        )
                                    # functionCall -> OpenAI tool_calls
                                    # delta envelope.
                                    fc = part.get("functionCall")
                                    if isinstance(fc, dict):
                                        fc_name = fc.get("name") or ""
                                        fc_args = fc.get("args") or {}
                                        fc_id = (
                                            fc.get("id")
                                            or f"call_{fc_name}_{time.time_ns()}"
                                        )
                                        if fc_id in emitted_function_call_ids:
                                            continue
                                        emitted_function_call_ids.add(fc_id)
                                        # Each distinct functionCall in an
                                        # assistant turn needs its own
                                        # tool_calls[*].index. Consumers
                                        # that reassemble tool_calls by
                                        # index collapse all calls onto
                                        # the same slot when this is
                                        # hardcoded to 0, breaking
                                        # parallel/multi-tool turns.
                                        tc_index = len(emitted_function_call_ids) - 1
                                        tool_call_delta: dict[str, Any] = {
                                            "index": tc_index,
                                            "id": fc_id,
                                            "type": "function",
                                            "function": {
                                                "name": fc_name,
                                                "arguments": _json.dumps(fc_args),
                                            },
                                        }
                                        # Gemini 3 function-calling: the
                                        # part-level `thoughtSignature`
                                        # must be echoed back on the
                                        # next turn or the model rejects
                                        # the tool-result envelope. Stow
                                        # it on `extra_content.google`
                                        # so the frontend can persist it
                                        # and our outbound translator
                                        # (below) can replay it.
                                        thought_sig = part.get(
                                            "thoughtSignature"
                                        ) or part.get("thought_signature")
                                        if isinstance(thought_sig, str) and thought_sig:
                                            tool_call_delta["extra_content"] = {
                                                "google": {
                                                    "thought_signature": thought_sig,
                                                }
                                            }
                                        emitted_any_function_call = True
                                        tool_chunk = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "tool_calls": [tool_call_delta]
                                                    },
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                        yield f"data: {_json.dumps(tool_chunk)}"
                                    # executableCode + codeExecutionResult
                                    # parts surface as the standard
                                    # code_execution tool_start/tool_end
                                    # envelope (same shape OpenAI and
                                    # Anthropic emit) so the chat
                                    # adapter can render Gemini sandbox
                                    # output through CodeExecutionToolUI.
                                    # https://ai.google.dev/gemini-api/docs/code-execution
                                    exec_code = part.get("executableCode")
                                    if isinstance(exec_code, dict):
                                        code_str = exec_code.get("code") or ""
                                        if code_str:
                                            code_tool_id = (
                                                exec_code.get("id")
                                                or f"gemini_code_exec_{time.time_ns()}"
                                            )
                                            gemini_code_exec_pending_id = code_tool_id
                                            # Stow the raw Gemini part so
                                            # follow-up turns can replay
                                            # the native `executableCode`
                                            # (Gemini rejects a generic
                                            # functionCall echo for code
                                            # execution history).
                                            _exec_thought_sig = part.get(
                                                "thoughtSignature"
                                            ) or part.get("thought_signature")
                                            _exec_native: dict[str, Any] = {
                                                "executableCode": exec_code,
                                            }
                                            if (
                                                isinstance(_exec_thought_sig, str)
                                                and _exec_thought_sig
                                            ):
                                                _exec_native["thoughtSignature"] = (
                                                    _exec_thought_sig
                                                )
                                            yield _emit_tool_event(
                                                {
                                                    "type": "tool_start",
                                                    "tool_name": "code_execution",
                                                    "tool_call_id": code_tool_id,
                                                    "arguments": {
                                                        "kind": "code_execution",
                                                        "language": (
                                                            (
                                                                exec_code.get(
                                                                    "language"
                                                                )
                                                                or "PYTHON"
                                                            ).lower()
                                                        ),
                                                        "code": code_str,
                                                        "google": {
                                                            "native_part": _exec_native,
                                                        },
                                                    },
                                                }
                                            )
                                    exec_result = part.get("codeExecutionResult")
                                    if isinstance(exec_result, dict):
                                        outcome = exec_result.get("outcome") or ""
                                        output = exec_result.get("output") or ""
                                        # Gemini returns
                                        # OUTCOME_OK / OUTCOME_FAILED /
                                        # OUTCOME_DEADLINE_EXCEEDED. Treat
                                        # non-OK outcomes as stderr so the
                                        # UI surfaces the error.
                                        if outcome and outcome != "OUTCOME_OK":
                                            result_text = (
                                                f"[{outcome}]\n{output}".rstrip()
                                            )
                                        else:
                                            result_text = output
                                        # Pair with the most recent
                                        # executableCode tool_start when
                                        # present; otherwise mint a fresh
                                        # id so the UI still renders the
                                        # output as a code_execution event.
                                        # Pair the tool_end with the most
                                        # recent code_exec tool_start id so
                                        # the UI matches start/end. Fall back
                                        # to exec_result.id (no preceding
                                        # executableCode part), then mint a
                                        # fresh id as a last resort.
                                        pair_id = (
                                            gemini_code_exec_pending_id
                                            or exec_result.get("id")
                                            or f"gemini_code_exec_{time.time_ns()}"
                                        )
                                        if gemini_code_exec_pending_id is None:
                                            yield _emit_tool_event(
                                                {
                                                    "type": "tool_start",
                                                    "tool_name": "code_execution",
                                                    "tool_call_id": pair_id,
                                                    "arguments": {
                                                        "kind": "code_execution",
                                                        "code": "",
                                                    },
                                                }
                                            )
                                        _result_thought_sig = part.get(
                                            "thoughtSignature"
                                        ) or part.get("thought_signature")
                                        _result_native: dict[str, Any] = {
                                            "codeExecutionResult": exec_result,
                                        }
                                        if (
                                            isinstance(_result_thought_sig, str)
                                            and _result_thought_sig
                                        ):
                                            _result_native["thoughtSignature"] = (
                                                _result_thought_sig
                                            )
                                        yield _emit_tool_event(
                                            {
                                                "type": "tool_end",
                                                "tool_call_id": pair_id,
                                                "result": result_text,
                                                "google": {
                                                    "native_part": _result_native,
                                                },
                                            }
                                        )
                                        last_code_exec_tool_id = pair_id
                                        last_code_exec_result_text = result_text
                                        gemini_code_exec_pending_id = None
                                    # inlineData -> image bytes. Two
                                    # paths:
                                    #  (a) On a Nano Banana / image
                                    #      picker turn this is the
                                    #      generated image; emit the
                                    #      standard image_generation
                                    #      tool envelope.
                                    #  (b) On a text turn that wired
                                    #      codeExecution, this is the
                                    #      sandbox's matplotlib output
                                    #      shipped alongside the result.
                                    #      Attach to the SAME
                                    #      code_execution card via the
                                    #      `__IMAGES__:` marker the
                                    #      chat-adapter understands so
                                    #      the UI shows one combined
                                    #      tool event instead of a
                                    #      bonus empty image_generation
                                    #      card.
                                    inline = part.get("inlineData")
                                    if isinstance(inline, dict):
                                        b64 = inline.get("data") or ""
                                        mime = inline.get("mimeType") or "image/png"
                                        if b64:
                                            image_uri = f"data:{mime};base64,{b64}"
                                            attached_to_code_exec = (
                                                not is_image_model
                                                and last_code_exec_tool_id is not None
                                                and bool(enabled_tools)
                                                and "code_execution"
                                                in (enabled_tools or [])
                                            )
                                            if attached_to_code_exec:
                                                updated_result = (
                                                    last_code_exec_result_text
                                                    + "\n__IMAGES__:"
                                                    + _json.dumps([image_uri])
                                                )
                                                yield _emit_tool_event(
                                                    {
                                                        "type": "tool_end",
                                                        "tool_call_id": (
                                                            last_code_exec_tool_id
                                                        ),
                                                        "result": updated_result,
                                                    }
                                                )
                                                last_code_exec_result_text = (
                                                    updated_result
                                                )
                                            else:
                                                img_id = f"img_{time.time_ns()}"
                                                yield _emit_tool_event(
                                                    {
                                                        "type": "tool_start",
                                                        "tool_name": "image_generation",
                                                        "tool_call_id": img_id,
                                                        "arguments": {
                                                            "kind": "image",
                                                            "prompt": "",
                                                        },
                                                    }
                                                )
                                                # Gemini 3 image editing
                                                # requires the prior
                                                # turn's
                                                # `thoughtSignature` to
                                                # be echoed back on the
                                                # inline image part of
                                                # the user message;
                                                # persist it on the
                                                # tool_end so the
                                                # frontend can replay
                                                # it.
                                                _img_thought_sig = part.get(
                                                    "thoughtSignature"
                                                ) or part.get("thought_signature")
                                                _img_tool_end: dict[str, Any] = {
                                                    "type": "tool_end",
                                                    "tool_call_id": img_id,
                                                    "result": "",
                                                    "image_b64": b64,
                                                    "image_mime": mime,
                                                }
                                                if (
                                                    isinstance(_img_thought_sig, str)
                                                    and _img_thought_sig
                                                ):
                                                    _img_tool_end["google"] = {
                                                        "thought_signature": (
                                                            _img_thought_sig
                                                        ),
                                                    }
                                                yield _emit_tool_event(_img_tool_end)
                            finish_reason = cand.get("finishReason")
                            if isinstance(finish_reason, str):
                                mapped = _finish_reason_map.get(finish_reason, "stop")
                                if mapped is not None:
                                    final_finish_reason = mapped

                    # End-of-stream emission order: web_search tool_end
                    # (with citations) -> finish_reason chunk -> usage
                    # chunk -> [DONE]. Matches the Anthropic / OpenAI
                    # helpers' contract so the frontend handler does
                    # not need provider-specific ordering knowledge.
                    if (
                        web_search_active
                        and web_search_tool_started
                        and not web_search_tool_ended
                    ):
                        blocks: list[str] = []
                        for cit in web_search_citations:
                            line_out = f"Title: {cit['title']}\nURL: {cit['url']}"
                            if cit.get("snippet"):
                                line_out += f"\nSnippet: {cit['snippet']}"
                            blocks.append(line_out)
                        yield _emit_tool_event(
                            {
                                "type": "tool_end",
                                "tool_call_id": web_search_tool_id,
                                "result": (
                                    "\n---\n".join(blocks)
                                    if blocks
                                    else "(search complete)"
                                ),
                            }
                        )
                        web_search_tool_ended = True

                    if final_finish_reason:
                        # OpenAI clients trigger tool execution when
                        # finish_reason="tool_calls". Gemini emits
                        # "STOP" even when the turn was a pure
                        # functionCall request, so override after the
                        # fact to match the OAI contract.
                        if emitted_any_function_call and final_finish_reason == "stop":
                            final_finish_reason = "tool_calls"
                        finish_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": final_finish_reason,
                                }
                            ],
                        }
                        yield f"data: {_json.dumps(finish_chunk)}"

                    # Translate Gemini's usageMetadata into the OpenAI
                    # include_usage shape so the existing
                    # `_build_usage_chunk` emitter handles wire
                    # formatting (and downstream cost calculators
                    # already understand the shape).
                    # `thoughtsTokenCount` is the hidden-reasoning slice
                    # of output, billed alongside `candidatesTokenCount`;
                    # roll both into `output_tokens` so total_tokens
                    # equals promptToken + candidatesToken + thoughtsToken
                    # and the cost calculator does not undercount.
                    if isinstance(last_usage, dict):
                        thought_tokens = last_usage.get("thoughtsTokenCount") or 0
                        candidate_tokens = last_usage.get("candidatesTokenCount") or 0
                        prompt_tokens = last_usage.get("promptTokenCount") or 0
                        # Gemini bills tool-call prompt slices separately
                        # via `toolUsePromptTokenCount`. Fold into input
                        # so total_tokens does not undercount tool turns.
                        tool_use_prompt_tokens = (
                            last_usage.get("toolUsePromptTokenCount") or 0
                        )
                        translated_usage = {
                            "input_tokens": prompt_tokens + tool_use_prompt_tokens,
                            "output_tokens": candidate_tokens + thought_tokens,
                            "input_tokens_details": {
                                "cached_tokens": (
                                    last_usage.get("cachedContentTokenCount") or 0
                                ),
                                "tool_use_prompt_tokens": tool_use_prompt_tokens,
                            },
                            "output_tokens_details": {
                                "reasoning_tokens": thought_tokens,
                            },
                        }
                        usage_line = _build_usage_chunk(
                            completion_id, "openai", translated_usage
                        )
                        if usage_line:
                            yield usage_line

                    yield "data: [DONE]"
                finally:
                    # Close BOTH the upstream response and the manual
                    # aiter_lines() iterator on every exit path -- normal
                    # [DONE], prompt-block return, and GeneratorExit on
                    # client cancellation. response.aclose() FIRST so
                    # PoolByteStream._closed=True and lines_gen.aclose()
                    # is a no-op (avoids the httpcore 1.0.x
                    # "async generator ignored GeneratorExit" path).
                    # Skipping lines_gen.aclose() emits
                    # `RuntimeWarning: coroutine method 'aclose' of
                    # 'Response.aiter_lines' was never awaited`.
                    await response.aclose()
                    await lines_gen.aclose()

        except httpx.ConnectError as exc:
            logger.error("Connection error to %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Failed to connect to {self.provider_type}: {exc}",
                self.provider_type,
            )
        except httpx.ReadTimeout as exc:
            logger.error("Read timeout from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                504,
                f"Timeout waiting for {self.provider_type} response",
                self.provider_type,
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP error from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Error communicating with {self.provider_type}: {exc}",
                self.provider_type,
            )

    async def _stream_openai_responses(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        enable_thinking: Optional[bool],
        reasoning_effort: Optional[str],
        enabled_tools: Optional[list[str]] = None,
        enable_prompt_caching: Optional[bool] = None,
        openai_code_exec_container_id: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Call OpenAI's /v1/responses endpoint and translate its SSE stream back
        into OpenAI Chat Completions chunk format.

        The Responses API uses a different request shape (``input`` instead of
        ``messages``, ``instructions`` for system prompts, ``max_output_tokens``
        for the budget) and emits event-typed SSE frames (e.g.
        ``response.output_text.delta``) rather than chat-completion chunks.
        ``presence_penalty`` / ``top_k`` are not part of the Responses contract
        and are dropped here intentionally.
        """
        import json as _json

        # Split system messages out into a single `instructions` string and
        # translate user/assistant messages into the Responses input shape.
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    if content:
                        instructions_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text" and part.get("text"):
                            instructions_parts.append(part["text"])
                continue

            if isinstance(content, str):
                input_items.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                translated_parts: list[dict[str, Any]] = []
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        translated_parts.append(
                            {"type": "input_text", "text": part.get("text", "")}
                        )
                    elif part_type == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url:
                            # Responses takes image_url as a flat string (both
                            # https:// URLs and data: URLs are accepted).
                            translated_parts.append(
                                {"type": "input_image", "image_url": url}
                            )
                    elif part_type == "input_document":
                        # OpenAI Responses accepts PDFs / docs as
                        # `{type:"input_file", file_data:"data:application/pdf;base64,..."}`
                        # or `{type:"input_file", file_url:"https://..."}`,
                        # with optional `filename`. See
                        # https://developers.openai.com/api/docs/guides/images-vision
                        # Map Studio's normalised `input_document` shape
                        # straight onto Responses' `input_file`.
                        file_url = part.get("file_url")
                        file_data = part.get("file_data")
                        filename = part.get("filename")
                        # Mirror the Anthropic-side guard: any "data:" URI
                        # without an actual base64 payload (`data:application/pdf;base64,`
                        # or whitespace-only) would otherwise be forwarded
                        # to OpenAI as `file_data=""`, which 400s the whole
                        # turn. Treat such payloads as missing AND fall
                        # back to file_url if one is also present, so a
                        # recoverable remote URL doesn't get discarded in
                        # favour of a malformed inline payload.
                        file_data_valid = bool(
                            isinstance(file_data, str)
                            and file_data
                            and (
                                not file_data.startswith("data:")
                                or file_data.partition(",")[2].strip()
                            )
                        )
                        block: dict[str, Any] = {"type": "input_file"}
                        if file_data_valid:
                            block["file_data"] = file_data
                        elif file_url:
                            block["file_url"] = file_url
                        else:
                            continue
                        if filename:
                            block["filename"] = filename
                        translated_parts.append(block)
                if translated_parts:
                    input_items.append({"role": role, "content": translated_parts})

        # NOTE: gpt-5.x / o3 / gpt-4.5 are reasoning-class models. They reject
        # temperature and top_p with `Unsupported parameter` 400s on
        # /v1/responses (and on /v1/chat/completions for the same families).
        # The PROVIDER_REGISTRY['openai'] model_id_allowlist already scopes
        # the picker to those families, so we never need to send sampling
        # knobs here. ``reasoning.effort`` defaults to "medium" server-side
        # if omitted — surface it in a future commit if a knob is wanted.
        del temperature, top_p  # explicit drop — params are accepted for
        # API symmetry with the other stream methods but not forwarded.

        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
        }
        # `summary: "auto"` is what makes /v1/responses emit reasoning
        # summary events — without it OpenAI returns no thinking text on
        # most reasoning models, the SSE handler has no <think>…</think>
        # to wrap, and the chat reasoning panel stays blank. Always pair
        # an explicit effort with summary except for the explicit "off"
        # case (effort: "none"), where summaries are pointless.
        summary_unsupported = bool(
            _OPENAI_REASONING_SUMMARY_UNSUPPORTED.match(model.strip().lower())
        )
        if reasoning_effort in (
            "minimal",
            "low",
            "medium",
            "high",
            "max",
            "xhigh",
        ):
            body["reasoning"] = {"effort": reasoning_effort}
            if not summary_unsupported:
                body["reasoning"]["summary"] = "auto"
        elif reasoning_effort == "none" or enable_thinking is False:
            body["reasoning"] = {"effort": "none"}
        elif enable_thinking is True:
            body["reasoning"] = {"effort": "medium"}
            if not summary_unsupported:
                body["reasoning"]["summary"] = "auto"
        if instructions_parts:
            body["instructions"] = "\n\n".join(instructions_parts)
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens

        # Prompt caching on /v1/responses is automatic and free, but the
        # default in-memory policy only survives ~5-10 min of inactivity
        # (up to ~1 hr). Opt into the 24-hour retention policy so a chat
        # left idle overnight still hits the cache on the next turn.
        # Pricing is identical to in_memory per OpenAI's docs.
        #
        # Gated on the base URL because ollama / llama.cpp / "custom"
        # presets all collapse to provider_type="openai" in
        # toExternalBackendProviderType, so they also land in this
        # helper. Those servers expose /v1/responses-shaped routes in
        # some configurations but don't implement
        # prompt_cache_retention — sending the field unconditionally
        # would 400 them. Match the public OpenAI host strictly so the
        # field only goes to OpenAI cloud. Studio's openai model picker
        # is registry-scoped to gpt-5.x / o3 / gpt-4.5, all of which
        # accept this parameter (gpt-5.5+ already defaults to "24h" and
        # rejects "in_memory", so it's a safe no-op there).
        # OpenAI-family cloud: api.openai.com OR Azure OpenAI Foundry
        # (*.openai.azure.com). Both expose the same Responses-API
        # extensions used below -- prompt_cache_retention,
        # context_management compaction, container shell tool -- so
        # treat them uniformly. Non-cloud OpenAI-compatible servers
        # (ollama / llama.cpp / vLLM / "custom" preset) hit /v1/responses
        # without these extensions and would 400 on the unknown body
        # fields, so they intentionally fall outside this gate.
        is_openai_cloud = _is_openai_family_cloud(self.base_url)
        if is_openai_cloud and enable_prompt_caching is not False:
            body["prompt_cache_retention"] = "24h"

        # OpenAI server-side context compaction — see
        #   https://developers.openai.com/api/docs/guides/compaction
        # When `compaction_threshold` is provided on a cloud OpenAI
        # request, attach `context_management: [{type:"compaction",
        # compact_threshold:N}]` so the API runs server-side
        # compaction when the rendered prompt crosses the threshold.
        # No beta header is required; no dated version pin. The field
        # is silently dropped for non-cloud backends because ollama /
        # llama.cpp / "custom" presets land in this helper and would
        # 400 on an unknown body field.
        if (
            is_openai_cloud
            and compaction_threshold is not None
            and compaction_threshold > 0
        ):
            body["context_management"] = [
                {
                    "type": "compaction",
                    "compact_threshold": int(compaction_threshold),
                }
            ]

        # OpenAI server-side tools — see
        #   https://developers.openai.com/api/docs/guides/tools
        #   https://developers.openai.com/api/docs/guides/tools-shell
        # The frontend's Search/Code buttons map to the unified
        # enabled_tools shorthand; translate that into the Responses-API
        # tool schema. Other built-in tools (file_search,
        # code_interpreter, image_generation, computer_use_preview) can
        # be added with the same pattern when we surface their toggles.
        code_execution_enabled_openai = bool(
            enabled_tools and "code_execution" in enabled_tools and is_openai_cloud
        )
        # OpenAI's image_generation tool is a Responses-API server tool.
        # See https://developers.openai.com/api/docs/guides/tools-image-generation
        # The model picks size / quality / background server-side and
        # delegates rendering to a gpt-image-* family model; the result
        # comes back inline as an `image_generation_call` output item
        # with a base64 image. Available on every gpt-5.x family member
        # plus gpt-4.1 / gpt-4o / o3 per the docs; restrict to cloud
        # OpenAI because the local llama.cpp / ollama backends don't
        # implement it and would 400.
        image_generation_enabled_openai = bool(
            enabled_tools and "image_generation" in enabled_tools and is_openai_cloud
        )
        if enabled_tools:
            tools_array: list[dict[str, Any]] = []
            if "web_search" in enabled_tools:
                tools_array.append({"type": "web_search"})
            if code_execution_enabled_openai:
                # `container_auto` lets OpenAI auto-create a fresh
                # container per request; we capture the resulting
                # container_id off the SSE stream and the chat-adapter
                # persists it onto the thread record. Subsequent turns
                # in the same thread pass it back as
                # `openai_code_exec_container_id`, which we translate to
                # `container_reference` here so the model sees
                # filesystem state from prior turns. Container expires
                # after ~20 min of inactivity per OpenAI's default
                # policy — a stale id 400s, the chat-adapter clears it
                # via container_invalidated, and the next turn falls
                # back to auto-create.
                shell_env: dict[str, Any]
                if openai_code_exec_container_id:
                    shell_env = {
                        "type": "container_reference",
                        "container_id": openai_code_exec_container_id,
                    }
                else:
                    shell_env = {"type": "container_auto"}
                tools_array.append({"type": "shell", "environment": shell_env})
            if image_generation_enabled_openai:
                tools_array.append({"type": "image_generation"})
            if tools_array:
                body["tools"] = tools_array

        url = f"{self.base_url}/responses"
        completion_id = f"chatcmpl-openai-{model.replace('/', '-')}"

        logger.info("Proxying OpenAI Responses API to %s (model=%s)", url, model)

        def _build_body(container_id_for_this_attempt: Optional[str]) -> dict[str, Any]:
            """Snapshot of the request body. Called once for the initial
            attempt and again with ``None`` for the post-expiry retry.
            Returns a fresh dict so the retry doesn't share state with the
            first attempt.
            """
            attempt_body = dict(body)
            if enabled_tools:
                tools_array_attempt: list[dict[str, Any]] = []
                if "web_search" in enabled_tools:
                    tools_array_attempt.append({"type": "web_search"})
                if code_execution_enabled_openai:
                    if container_id_for_this_attempt:
                        env_attempt: dict[str, Any] = {
                            "type": "container_reference",
                            "container_id": container_id_for_this_attempt,
                        }
                    else:
                        env_attempt = {"type": "container_auto"}
                    tools_array_attempt.append(
                        {"type": "shell", "environment": env_attempt}
                    )
                if image_generation_enabled_openai:
                    tools_array_attempt.append({"type": "image_generation"})
                if tools_array_attempt:
                    attempt_body["tools"] = tools_array_attempt
                else:
                    attempt_body.pop("tools", None)
            return attempt_body

        def _is_openai_container_expired_error(error_text: str) -> bool:
            """Match the substring patterns OpenAI uses for expired / missing
            code-exec containers. There's no official error code in the public
            docs, so we substring-match a small set.
            """
            lowered = error_text.lower()
            if "container" not in lowered:
                return False
            return (
                "expired" in lowered
                or "not_found" in lowered
                or "not found" in lowered
                or "no such container" in lowered
            )

        try:
            retried = False
            attempt_container_id = openai_code_exec_container_id
            while True:
                attempt_body = _build_body(attempt_container_id)
                async with _http_client.stream(
                    "POST",
                    url,
                    json = attempt_body,
                    headers = self._auth_headers(),
                    timeout = self._stream_timeout,
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_text = error_body.decode("utf-8", errors = "replace")
                        logger.error(
                            "OpenAI Responses returned %d: %s",
                            response.status_code,
                            error_text[:500],
                        )
                        expired_container_4xx = (
                            attempt_container_id
                            and 400 <= response.status_code < 500
                            and _is_openai_container_expired_error(error_text)
                        )
                        if expired_container_4xx and not retried:
                            yield (
                                f"data: "
                                f"{_json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': None}], '_toolEvent': {'type': 'container_invalidated'}})}"
                            )
                            retried = True
                            attempt_container_id = None
                            continue
                        yield _error_sse_line(
                            response.status_code, error_text, self.provider_type
                        )
                        return

                    # NOTE: same manual __anext__ loop as stream_chat_completion —
                    # see comment there for the GeneratorExit / aclose ordering.
                    lines_gen = response.aiter_lines().__aiter__()
                    done_emitted = False
                    reasoning_open = False
                    reasoning_emitted = False
                    # Latched from response.completed / response.incomplete so
                    # the final log can surface input_tokens_details.cached_tokens —
                    # the field that proves prompt_cache_retention="24h" is
                    # actually hitting OpenAI's cache instead of recomputing
                    # the prefix every turn.
                    last_usage: Optional[dict[str, Any]] = None
                    # Per-call state for OpenAI's server-side web_search tool. Mapped
                    # back into our local _toolEvent shape so the existing chat-UI
                    # renderer surfaces web_search the same way it does for local
                    # tool calls: a "Searching…" tool-call card, then a `tool_end`
                    # carrying citations formatted as
                    #   Title: …\nURL: …\nSnippet: …\n---\n…
                    # blocks (which the frontend's parseSourcesFromResult lifts
                    # into source content parts at end of stream).
                    # web_search_calls preserves insertion order so we can apply
                    # the aggregated citation list onto the *last* call's
                    # tool_end — that's the one the frontend's source-pill
                    # extraction reads (parseSourcesFromResult flatMaps every
                    # web_search result, so a single non-empty result is enough
                    # to surface all sources at message tail).
                    # OpenAI emits url_citation annotations on text deltas, not
                    # per call — there's no wire field linking a citation back
                    # to a specific search invocation. Hence the shared list.
                    # web_search_calls: { item_id -> {query} }
                    web_search_calls: dict[str, dict[str, Any]] = {}
                    all_url_citations: list[dict[str, str]] = []
                    # Shell-tool (code execution) state. OpenAI emits
                    # `shell_call` items (model requesting a command list)
                    # paired with `shell_call_output` items (execution
                    # results). We mirror the Anthropic code-execution UX
                    # by emitting one `_toolEvent` tool_start per
                    # shell_call and one tool_end per shell_call_output;
                    # they're linked via `shell_call_output.call_id`
                    # matching `shell_call.id`. Items are independent of
                    # web_search (different keyed map).
                    # shell_calls: { call_id -> {commands, output} }
                    shell_calls: dict[str, dict[str, Any]] = {}
                    # Container id captured from the response stream. When
                    # it differs from the inbound id, emit a synthetic
                    # `container_ready` _toolEvent so the frontend can
                    # persist it onto the thread record for the next turn.
                    # Where OpenAI surfaces it is documented loosely; we
                    # probe two known fields (response.container_id on
                    # response.completed, item.environment.container_id on
                    # shell_call output items) and latch the first one we
                    # see.
                    latched_container_id: Optional[str] = None
                    container_id_emitted = False

                    def _emit_tool_event(payload: dict[str, Any]) -> str:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None,
                                }
                            ],
                            "_toolEvent": payload,
                        }
                        return f"data: {_json.dumps(chunk)}"

                    def _format_shell_output(output: Any) -> str:
                        """Render an OpenAI `shell_call_output.output` list
                        as the preformatted text payload the frontend's
                        CodeExecutionToolUI displays inside a <pre>. Each
                        entry has stdout/stderr/outcome — concatenate them
                        with a separator block per entry and append
                        `return_code` / `(timeout)` annotations only when
                        they convey information beyond "succeeded".
                        """
                        if not isinstance(output, list):
                            return ""
                        parts: list[str] = []
                        for entry in output:
                            if not isinstance(entry, dict):
                                continue
                            stdout = entry.get("stdout") or ""
                            stderr = entry.get("stderr") or ""
                            outcome = entry.get("outcome") or {}
                            chunk_parts: list[str] = []
                            if stdout:
                                chunk_parts.append(stdout)
                            if stderr:
                                chunk_parts.append(f"--- stderr ---\n{stderr}")
                            if isinstance(outcome, dict):
                                outcome_type = outcome.get("type")
                                if outcome_type == "exit":
                                    exit_code = outcome.get("exit_code")
                                    if isinstance(exit_code, int) and exit_code != 0:
                                        chunk_parts.append(f"return_code: {exit_code}")
                                elif outcome_type == "timeout":
                                    chunk_parts.append("(timeout)")
                            if chunk_parts:
                                parts.append("\n".join(chunk_parts))
                        return (
                            "\n--- next command ---\n".join(parts)
                            if parts
                            else "(no output)"
                        )

                    def _record_url_citation(payload: dict[str, Any]) -> None:
                        """Append a url_citation onto the shared all_url_citations
                        list. Dedup by URL — the same source can be cited multiple
                        times across deltas. We do NOT try to attribute citations
                        to individual web_search_call invocations because OpenAI's
                        annotation events don't carry that linkage."""
                        if payload.get("type") != "url_citation":
                            return
                        url = payload.get("url", "")
                        if not url:
                            return
                        if any(c["url"] == url for c in all_url_citations):
                            return
                        title = payload.get("title") or url
                        snippet = payload.get("snippet") or payload.get("quote") or ""
                        all_url_citations.append(
                            {
                                "url": url,
                                "title": title,
                                "snippet": snippet,
                            }
                        )

                    def _extract_reasoning_text(payload: Any) -> str:
                        if payload is None:
                            return ""
                        if isinstance(payload, str):
                            return payload
                        if isinstance(payload, list):
                            out: list[str] = []
                            for item in payload:
                                text = _extract_reasoning_text(item)
                                if text:
                                    out.append(text)
                            return "".join(out)
                        if isinstance(payload, dict):
                            # OpenAI responses may carry reasoning summaries in
                            # different envelope fields across event variants.
                            for key in ("text", "delta", "content", "summary"):
                                if key in payload:
                                    text = _extract_reasoning_text(payload.get(key))
                                    if text:
                                        return text
                            if payload.get("type") == "summary_text":
                                return _extract_reasoning_text(payload.get("text"))
                        return ""

                    def _chunk_with_text(text: str) -> str:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        return f"data: {_json.dumps(chunk)}"

                    try:
                        while True:
                            try:
                                line = await lines_gen.__anext__()
                            except StopAsyncIteration:
                                break
                            if not line or line.startswith("event:"):
                                continue
                            if not line.startswith("data:"):
                                continue

                            data_str = line[len("data:") :].strip()
                            if not data_str:
                                continue
                            if data_str == "[DONE]":
                                if not done_emitted:
                                    yield "data: [DONE]"
                                    done_emitted = True
                                break

                            try:
                                event = _json.loads(data_str)
                            except _json.JSONDecodeError:
                                continue

                            event_type = event.get("type")

                            if event_type == "response.output_text.delta":
                                delta_text = event.get("delta", "")
                                if delta_text:
                                    if reasoning_open:
                                        yield _chunk_with_text("</think>")
                                        reasoning_open = False
                                    yield _chunk_with_text(delta_text)
                                # Some API versions inline url citations on the
                                # delta event itself rather than as a separate
                                # response.output_text.annotation.added event.
                                for ann in event.get("annotations") or []:
                                    if isinstance(ann, dict):
                                        _record_url_citation(ann)

                            elif event_type == "response.output_text.annotation.added":
                                ann = event.get("annotation")
                                if isinstance(ann, dict):
                                    _record_url_citation(ann)

                            elif event_type == "response.output_item.added":
                                # Track the call early but do NOT emit tool_start
                                # yet — action.query is not reliably populated on
                                # added across OpenAI API versions, and the
                                # frontend's tool_start is a one-shot push (no
                                # update mechanism). Wait for output_item.done.
                                item = event.get("item", {})
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "web_search_call"
                                ):
                                    item_id = item.get("id", "") or (
                                        f"ws_{len(web_search_calls)}"
                                    )
                                    web_search_calls.setdefault(item_id, {"query": ""})
                                # Shell-tool: register the call eagerly so
                                # the matching shell_call_output can link
                                # back even if `done` arrives out of order.
                                # Also probe for container_id on the
                                # environment field — when container_auto
                                # auto-creates one, this is the first place
                                # the new id might surface (OpenAI doesn't
                                # promise this in docs, but the field is
                                # cheap to scan and lets us emit
                                # container_ready earlier than
                                # response.completed).
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "shell_call"
                                ):
                                    item_id = item.get("id", "") or (
                                        f"sc_{len(shell_calls)}"
                                    )
                                    shell_calls.setdefault(
                                        item_id,
                                        {"commands": [], "output": None},
                                    )
                                    env = item.get("environment")
                                    if isinstance(env, dict):
                                        probe = env.get("container_id") or env.get("id")
                                        if (
                                            isinstance(probe, str)
                                            and probe.startswith("cntr_")
                                            and latched_container_id is None
                                        ):
                                            latched_container_id = probe

                            elif event_type == "response.output_item.done":
                                item = event.get("item", {})
                                if not isinstance(item, dict):
                                    continue
                                if item.get("type") == "reasoning":
                                    summary_text = _extract_reasoning_text(
                                        item.get("summary")
                                    )
                                    if summary_text and not reasoning_emitted:
                                        if not reasoning_open:
                                            summary_text = f"<think>{summary_text}"
                                            reasoning_open = True
                                        yield _chunk_with_text(summary_text)
                                        reasoning_emitted = True
                                elif item.get("type") == "web_search_call":
                                    # done is the canonical place to read the
                                    # query, so emit both tool_start and tool_end
                                    # here. Frontend then renders a card per call
                                    # with the proper "Searching: <query>" label.
                                    # Citations are aggregated separately and the
                                    # *last* call's result is overwritten at
                                    # response.completed with the citation list
                                    # (so the source-pill extraction at message
                                    # tail surfaces them once).
                                    item_id = item.get("id", "") or (
                                        f"ws_{len(web_search_calls)}"
                                    )
                                    action = item.get("action")
                                    query = (
                                        action.get("query", "")
                                        if isinstance(action, dict)
                                        else ""
                                    )
                                    web_search_calls[item_id] = {"query": query}
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_start",
                                            "tool_name": "web_search",
                                            "tool_call_id": item_id,
                                            "arguments": (
                                                {"query": query} if query else {}
                                            ),
                                        }
                                    )
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": item_id,
                                            # Empty result — the last call gets
                                            # overwritten with citations at
                                            # response.completed.
                                            "result": "",
                                        }
                                    )
                                elif item.get("type") == "shell_call":
                                    # OpenAI ships the commands array on the
                                    # action field. Join them onto one
                                    # command string for the tool card —
                                    # the renderer is shared with Anthropic
                                    # bash, which only carries a single
                                    # `command`. Multiple commands in one
                                    # shell_call get joined with newlines so
                                    # they still render as one card.
                                    item_id = item.get("id", "") or (
                                        f"sc_{len(shell_calls)}"
                                    )
                                    action = item.get("action") or {}
                                    commands = (
                                        action.get("commands")
                                        if isinstance(action, dict)
                                        else None
                                    ) or []
                                    joined_command = (
                                        "\n".join(str(c) for c in commands)
                                        if isinstance(commands, list)
                                        else ""
                                    )
                                    shell_calls.setdefault(
                                        item_id,
                                        {"commands": [], "output": None},
                                    )
                                    shell_calls[item_id]["commands"] = (
                                        list(commands)
                                        if isinstance(commands, list)
                                        else []
                                    )
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_start",
                                            "tool_name": "code_execution",
                                            "tool_call_id": item_id,
                                            "arguments": {
                                                "kind": "bash",
                                                "command": joined_command,
                                            },
                                        }
                                    )
                                elif item.get("type") == "shell_call_output":
                                    # `call_id` links back to the shell_call's
                                    # `id`, which is what we used as the
                                    # tool_call_id on tool_start. Match on
                                    # call_id when present so the matching
                                    # card transitions to complete.
                                    call_id = (
                                        item.get("call_id") or item.get("id") or ""
                                    )
                                    output = item.get("output") or []
                                    if call_id in shell_calls:
                                        shell_calls[call_id]["output"] = output
                                    result_text = _format_shell_output(output)
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": call_id,
                                            "result": result_text,
                                        }
                                    )
                                elif item.get("type") == "image_generation_call":
                                    # OpenAI's image_generation tool returns
                                    # a single output item with the base64
                                    # PNG/WebP/JPEG on `result` (sometimes
                                    # `b64_json` depending on output_format).
                                    # `revised_prompt` is what the gpt-image
                                    # backbone actually used after refinement
                                    # of the assistant's request. Emit
                                    # tool_start + tool_end so the chat card
                                    # renders the prompt + the generated
                                    # image inline. The frontend chat-adapter
                                    # decides how to render the base64 blob
                                    # (likely an <img src="data:image/...">)
                                    # based on the `kind: "image"` hint we
                                    # set on tool_start arguments.
                                    # `time_ns()` (nanoseconds) instead of
                                    # millisecond resolution so synthesised
                                    # ids stay unique even when two image
                                    # generations resolve in the same ms.
                                    item_id = item.get("id", "") or (
                                        f"img_{time.time_ns()}"
                                    )
                                    prompt_in = (
                                        item.get("revised_prompt")
                                        or item.get("prompt")
                                        or ""
                                    )
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_start",
                                            "tool_name": "image_generation",
                                            "tool_call_id": item_id,
                                            "arguments": {
                                                "kind": "image",
                                                "prompt": prompt_in,
                                            },
                                        }
                                    )
                                    b64 = (
                                        item.get("result") or item.get("b64_json") or ""
                                    )
                                    output_format = item.get("output_format") or "png"
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": item_id,
                                            "result": "",
                                            "image_b64": b64,
                                            "image_mime": (f"image/{output_format}"),
                                            "size": item.get("size"),
                                            "quality": item.get("quality"),
                                            "background": item.get("background"),
                                        }
                                    )

                            elif (
                                isinstance(event_type, str)
                                and "reasoning" in event_type
                            ):
                                reasoning_delta = _extract_reasoning_text(event)
                                if reasoning_delta:
                                    if not reasoning_open:
                                        reasoning_delta = f"<think>{reasoning_delta}"
                                        reasoning_open = True
                                    yield _chunk_with_text(reasoning_delta)
                                    reasoning_emitted = True

                            elif event_type == "response.completed":
                                completed_usage = (event.get("response") or {}).get(
                                    "usage"
                                )
                                if isinstance(completed_usage, dict):
                                    last_usage = completed_usage
                                if reasoning_open:
                                    yield _chunk_with_text("</think>")
                                    reasoning_open = False
                                # Probe response.container_id (top-level) and
                                # response.container.id for the shell-tool
                                # container id. OpenAI's docs don't pin the
                                # exact field, so we scan both. Emit
                                # `container_ready` only when the value
                                # differs from the inbound one — no churn on
                                # reuse.
                                response_obj = event.get("response") or {}
                                if isinstance(response_obj, dict):
                                    probe_id = response_obj.get("container_id")
                                    if not probe_id:
                                        container_field = response_obj.get("container")
                                        if isinstance(container_field, dict):
                                            probe_id = container_field.get("id")
                                    if (
                                        isinstance(probe_id, str)
                                        and probe_id.startswith("cntr_")
                                        and latched_container_id is None
                                    ):
                                        latched_container_id = probe_id
                                if (
                                    latched_container_id
                                    and not container_id_emitted
                                    and latched_container_id
                                    != openai_code_exec_container_id
                                ):
                                    yield _emit_tool_event(
                                        {
                                            "type": "container_ready",
                                            "container_id": latched_container_id,
                                        }
                                    )
                                    container_id_emitted = True
                                # Apply the aggregated citation list onto the
                                # *last* web_search call by overwriting its
                                # tool_end result. The frontend's
                                # parseSourcesFromResult flatMaps every
                                # web_search tool-call result, so a single
                                # non-empty result is enough to surface the
                                # whole source-pill set at the message tail —
                                # no need to fan out across every card (which
                                # would just duplicate the same pills).
                                if web_search_calls and all_url_citations:
                                    last_id = list(web_search_calls.keys())[-1]
                                    blocks: list[str] = []
                                    for cit in all_url_citations:
                                        line = (
                                            f"Title: {cit['title']}\n"
                                            f"URL: {cit['url']}"
                                        )
                                        if cit.get("snippet"):
                                            line += f"\nSnippet: {cit['snippet']}"
                                        blocks.append(line)
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": last_id,
                                            "result": "\n---\n".join(blocks),
                                        }
                                    )
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "stop",
                                        }
                                    ],
                                }
                                yield f"data: {_json.dumps(chunk)}"
                                # Emit include_usage-style chunk after the
                                # finish_reason so callers can surface
                                # cached_tokens in their UI.
                                usage_line = _build_usage_chunk(
                                    completion_id,
                                    "openai",
                                    last_usage,
                                )
                                if usage_line:
                                    yield usage_line

                            elif event_type == "response.incomplete":
                                incomplete_usage = (event.get("response") or {}).get(
                                    "usage"
                                )
                                if isinstance(incomplete_usage, dict):
                                    last_usage = incomplete_usage
                                if reasoning_open:
                                    yield _chunk_with_text("</think>")
                                    reasoning_open = False
                                # Same backfill as response.completed — apply
                                # whatever citations we managed to gather
                                # before truncation onto the last call. All
                                # earlier tool cards already have their proper
                                # query + empty placeholder result from the
                                # output_item.done emissions above.
                                if web_search_calls and all_url_citations:
                                    last_id = list(web_search_calls.keys())[-1]
                                    blocks = []
                                    for cit in all_url_citations:
                                        line = (
                                            f"Title: {cit['title']}\n"
                                            f"URL: {cit['url']}"
                                        )
                                        if cit.get("snippet"):
                                            line += f"\nSnippet: {cit['snippet']}"
                                        blocks.append(line)
                                    yield _emit_tool_event(
                                        {
                                            "type": "tool_end",
                                            "tool_call_id": last_id,
                                            "result": "\n---\n".join(blocks),
                                        }
                                    )
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "length",
                                        }
                                    ],
                                }
                                yield f"data: {_json.dumps(chunk)}"
                                # Emit include_usage-style chunk after the
                                # length-truncated finish_reason too, so
                                # incomplete responses still report
                                # cached_tokens.
                                usage_line = _build_usage_chunk(
                                    completion_id,
                                    "openai",
                                    last_usage,
                                )
                                if usage_line:
                                    yield usage_line

                            elif event_type in ("response.failed", "error"):
                                # Surface the failure to the client; let the
                                # outer route emit [DONE] as part of its cleanup.
                                error_payload = event.get("response", {}).get(
                                    "error", {}
                                ) or {
                                    "message": event.get("message", "Unknown error"),
                                    "code": event.get("code"),
                                }
                                yield _error_sse_line(
                                    502,
                                    _json.dumps(error_payload),
                                    self.provider_type,
                                )
                                break
                    except GeneratorExit:
                        await response.aclose()
                        await lines_gen.aclose()
                        raise
                    finally:
                        # Summarise what the model actually did this turn so
                        # support reports of "I clicked Search and got nothing"
                        # can be triaged at a glance: was the tool requested,
                        # did OpenAI invoke it, and how many sources came back?
                        web_search_requested = bool(
                            enabled_tools and "web_search" in enabled_tools
                        )
                        web_search_invocations = len(web_search_calls)
                        total_citations = len(all_url_citations)
                        queries = [
                            sc["query"]
                            for sc in web_search_calls.values()
                            if sc.get("query")
                        ]
                        # cached_input_tokens > 0 on turn N proves
                        # prompt_cache_retention="24h" is letting the previous
                        # turn's prefix hit the cache instead of being
                        # recomputed. On /v1/responses the field is nested as
                        # usage.input_tokens_details.cached_tokens (not
                        # prompt_tokens_details, which is the /v1/chat/completions
                        # shape).
                        cached_input_tokens = None
                        if isinstance(last_usage, dict):
                            details = last_usage.get("input_tokens_details")
                            if isinstance(details, dict):
                                cached_input_tokens = details.get("cached_tokens")
                        code_execution_requested = code_execution_enabled_openai
                        code_execution_invocations = len(shell_calls)
                        code_execution_results = sum(
                            1
                            for sc in shell_calls.values()
                            if sc.get("output") is not None
                        )
                        logger.info(
                            "OpenAI Responses stream complete (model=%s, "
                            "web_search_requested=%s, web_search_invocations=%s, "
                            "citations=%s, queries=%s, reasoning_emitted=%s, "
                            "code_execution_requested=%s, "
                            "code_execution_invocations=%s, "
                            "code_execution_results=%s, "
                            "container_id_in=%s, container_id_out=%s, "
                            "input_tokens=%s, output_tokens=%s, "
                            "cached_input_tokens=%s)",
                            model,
                            web_search_requested,
                            web_search_invocations,
                            total_citations,
                            queries,
                            reasoning_emitted,
                            code_execution_requested,
                            code_execution_invocations,
                            code_execution_results,
                            openai_code_exec_container_id,
                            latched_container_id,
                            (last_usage or {}).get("input_tokens"),
                            (last_usage or {}).get("output_tokens"),
                            cached_input_tokens,
                        )
                        await response.aclose()
                        await lines_gen.aclose()
                    return

        except httpx.ConnectError as exc:
            logger.error("Connection error to %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Failed to connect to {self.provider_type}: {exc}",
                self.provider_type,
            )
        except httpx.ReadTimeout as exc:
            logger.error("Read timeout from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                504,
                f"Timeout waiting for {self.provider_type} response",
                self.provider_type,
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP error from %s: %s", self.provider_type, exc)
            yield _error_sse_line(
                502,
                f"Error communicating with {self.provider_type}: {exc}",
                self.provider_type,
            )

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full response dict.

        Note: only valid for OpenAI-compatible providers. Anthropic requires its
        own Messages API; use stream_chat_completion (with stream=False) instead
        if a non-streaming Anthropic path is needed in the future.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
        }
        if max_tokens is not None:
            if self.provider_type == "openai":
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens

        response = await _http_client.post(
            f"{self.base_url}/chat/completions",
            json = body,
            headers = self._auth_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        return response.json()

    async def list_models(self) -> list[dict[str, Any]]:
        """
        Call GET /models on the provider to discover available models.

        Returns a list of model dicts with at least 'id' and optionally
        'created', 'owned_by', etc.

        All supported providers expose a /models endpoint:
        - OpenAI-compatible: standard {"data": [...]} response
        - Anthropic: https://api.anthropic.com/v1/models — same {"data": [...]} shape
        """
        try:
            response = await _http_client.get(
                f"{self.base_url}/models",
                headers = self._auth_headers(),
                timeout = self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI format: {"data": [{"id": "...", ...}, ...]}
            # Some local servers (Ollama with no models) return data: null.
            models: list[dict[str, Any]] = []
            if isinstance(data, dict):
                raw_models = data.get("data") or []
                if isinstance(raw_models, list):
                    models = [model for model in raw_models if isinstance(model, dict)]
            if not models and self.provider_type == "ollama":
                models = await self._list_ollama_native_models()
            # Gemini's native /v1beta/models returns
            # {"models": [{"name": "models/gemini-2.5-flash", ...}]}
            # -- repackage into the OpenAI-compatible shape the rest
            # of Studio expects so dynamic model discovery works.
            if not models and self.provider_type == "gemini":
                models = self._parse_gemini_models(data)
            return models
        except httpx.HTTPError as exc:
            logger.error("Failed to list models from %s: %s", self.provider_type, exc)
            raise

    @staticmethod
    def _parse_gemini_models(payload: Any) -> list[dict[str, Any]]:
        """Translate Gemini's native /v1beta/models payload to OpenAI shape.

        Native response:
          {"models": [{"name": "models/gemini-2.5-flash",
                       "baseModelId": "gemini-2.5-flash",
                       "displayName": "Gemini 2.5 Flash",
                       "supportedGenerationMethods": [...]}]}

        We only keep entries that advertise
        ``generateContent`` / ``streamGenerateContent`` so the picker
        does not surface embedding-only models the chat path can't
        drive.
        """
        if not isinstance(payload, dict):
            return []
        entries = payload.get("models") or []
        if not isinstance(entries, list):
            return []
        out: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            methods = entry.get("supportedGenerationMethods") or []
            if (
                isinstance(methods, list)
                and methods
                and not any(
                    m in methods for m in ("generateContent", "streamGenerateContent")
                )
            ):
                continue
            base_id = entry.get("baseModelId")
            name = entry.get("name") or ""
            # ``name`` arrives as ``"models/gemini-2.5-flash"``; the
            # chat path uses the bare id.
            short_id = (
                base_id
                if isinstance(base_id, str) and base_id
                else (name.split("/", 1)[1] if "/" in name else name)
            )
            if not short_id:
                continue
            out.append(
                {
                    "id": short_id,
                    "owned_by": "google",
                    "display_name": entry.get("displayName") or short_id,
                }
            )
        return out

    async def _list_ollama_native_models(self) -> list[dict[str, Any]]:
        """Fallback when Ollama's /v1/models returns an empty or null catalog."""
        root = self.base_url.removesuffix("/v1").rstrip("/")
        response = await _http_client.get(
            f"{root}/api/tags",
            headers = self._auth_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return []
        raw_models = payload.get("models") or []
        if not isinstance(raw_models, list):
            return []
        return [
            {"id": entry.get("name", "").strip(), "owned_by": "ollama"}
            for entry in raw_models
            if isinstance(entry, dict) and entry.get("name", "").strip()
        ]

    async def verify_models_endpoint_lightweight(self) -> None:
        """
        Confirm GET /models returns 200 without buffering the full response body.

        Used for providers with enormous catalogs (e.g. OpenRouter, Hugging Face router)
        where downloading the full JSON would be prohibitive.
        """
        url = f"{self.base_url}/models"
        try:
            async with _http_client.stream(
                "GET",
                url,
                headers = self._auth_headers(),
                timeout = self._timeout,
            ) as response:
                if response.status_code != 200:
                    response.raise_for_status()
                async for _chunk in response.aiter_bytes(chunk_size = 2048):
                    break
        except httpx.HTTPError as exc:
            logger.error(
                "Lightweight /models check failed for %s: %s",
                self.provider_type,
                exc,
            )
            raise

    def _container_headers(self) -> dict[str, str]:
        """Auth headers plus the OpenAI-Beta opt-in for /v1/containers.

        OpenAI's containers API requires ``OpenAI-Beta: containers=v1``.
        Without it, DELETE silently no-ops: the API returns 200 with a
        ``{"deleted": true}`` body but does not actually remove the
        container (verified 2026-05-15). The header is required for
        list / create / delete to behave consistently.
        """
        headers = self._auth_headers()
        headers["OpenAI-Beta"] = "containers=v1"
        return headers

    async def list_openai_containers(self) -> list[dict[str, Any]]:
        """
        GET /v1/containers on the user's OpenAI account.

        Returns the raw container records (id, name, created_at,
        last_active_at, expires_after, status). The route layer
        reshapes these into the UI summary shape.

        Only valid against api.openai.com — non-cloud OpenAI-compat
        servers don't implement /v1/containers and would 404 here.
        Caller is responsible for the is_openai_cloud guard.
        """
        response = await _http_client.get(
            f"{self.base_url}/containers",
            headers = self._container_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        containers = data.get("data") if isinstance(data, dict) else None
        result = list(containers) if isinstance(containers, list) else []
        logger.info(
            "openai_container_list.response count=%s items=%s",
            len(result),
            [
                {"id": c.get("id"), "status": c.get("status")}
                for c in result
                if isinstance(c, dict)
            ],
        )
        return result

    async def create_openai_container(
        self,
        name: str,
        ttl_minutes: int,
    ) -> dict[str, Any]:
        """
        POST /v1/containers with ``expires_after.anchor="last_active_at"``.
        ``ttl_minutes`` is the idle timeout — every API call that
        touches the container resets the timer.
        """
        body = {
            "name": name,
            "expires_after": {
                "anchor": "last_active_at",
                "minutes": ttl_minutes,
            },
        }
        response = await _http_client.post(
            f"{self.base_url}/containers",
            json = body,
            headers = self._container_headers(),
            timeout = self._timeout,
        )
        response.raise_for_status()
        return response.json()

    async def delete_openai_container(self, container_id: str) -> None:
        """DELETE /v1/containers/{id}. 404s are surfaced as HTTPError.

        Uses a fresh httpx client (not the shared ``_http_client``) so
        connection-pool state from earlier chat requests cannot
        interfere — observed in the wild that DELETEs over the shared
        pool returned ``deleted: true`` while the container persisted
        in subsequent /containers list calls, even though the same
        DELETE issued from a fresh client genuinely removed it.

        Verifies the response body reports ``deleted: true``. OpenAI
        returns a 2xx ``deleted: true`` body even when the request is
        silently rejected (e.g. missing OpenAI-Beta header), so a
        status-only check is not sufficient.
        """
        url = f"{self.base_url}/containers/{container_id}"
        headers = self._container_headers()
        logger.info(
            "openai_container_delete.outbound url=%s has_auth=%s openai_beta=%s",
            url,
            "Authorization" in headers,
            headers.get("OpenAI-Beta"),
        )
        async with httpx.AsyncClient(timeout = self._timeout) as fresh_client:
            response = await fresh_client.delete(url, headers = headers)
        logger.info(
            "openai_container_delete.response status=%s cf_ray=%s "
            "request_id=%s organization=%s project=%s processing_ms=%s body=%s",
            response.status_code,
            response.headers.get("cf-ray"),
            response.headers.get("x-request-id"),
            response.headers.get("openai-organization"),
            response.headers.get("openai-project"),
            response.headers.get("openai-processing-ms"),
            response.text[:300],
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not (isinstance(payload, dict) and payload.get("deleted") is True):
            raise httpx.HTTPError(
                f"OpenAI did not confirm container deletion: {response.text[:200]}"
            )

    async def close(self) -> None:
        """No-op — the underlying client is shared across requests."""


def _provider_display_name(provider_type: str) -> str:
    from core.inference.providers import get_provider_info

    info = get_provider_info(provider_type) or {}
    return str(info.get("display_name") or provider_type)


def _friendly_provider_error_text(
    provider_type: str,
    status_code: int,
    raw_message: str,
    *,
    model: str | None = None,
) -> str:
    """Rewrite common provider errors into actionable Studio copy."""
    if status_code == 404 and model:
        lowered = raw_message.lower()
        if "not found" in lowered or "not_found" in lowered:
            if provider_type == "ollama":
                label = _provider_display_name(provider_type)
                return (
                    f"Model '{model}' is not installed in {label}. "
                    f"Run `ollama pull {model}` in a terminal, then retry."
                )
            if provider_type in ("vllm", "llama_cpp"):
                label = _provider_display_name(provider_type)
                return (
                    f"Model '{model}' is not available on the {label} server. "
                    "Check that the server is running and the model is loaded, "
                    "then retry."
                )
    return raw_message


def _error_sse_line(status_code: int, message: str, provider_type: str) -> str:
    """Format an error as an SSE data line in OpenAI error format."""
    import json

    error_obj = {
        "error": {
            "message": message,
            "type": "provider_error",
            "code": str(status_code),
            "provider": provider_type,
        }
    }
    return f"data: {json.dumps(error_obj)}"


def _build_usage_chunk(
    completion_id: str,
    provider: Literal["anthropic", "openai"],
    last_usage: Optional[dict],
) -> Optional[str]:
    """Build an OpenAI ``include_usage``-style SSE chunk that carries the
    upstream prompt-cache accounting back to the client.

    Until now Studio captured ``cache_creation_input_tokens`` /
    ``cache_read_input_tokens`` (Anthropic) and
    ``input_tokens_details.cached_tokens`` (OpenAI Responses) on
    ``last_usage`` and only wrote them to the structlog stream.
    Browser / SDK clients had no way to see how many tokens hit the cache
    -- so the "you saved $X" UX in the chat panel was impossible without
    scraping the server log.

    This helper emits the standard OpenAI chunk shape -- ``choices: []``
    with a populated ``usage`` block -- so any client that already
    consumes ``stream_options={"include_usage": true}`` keeps working,
    and the Anthropic-native counts are surfaced as extra keys on the
    same ``usage`` dict:

        usage.prompt_tokens_details.cached_tokens
            normalised cache-read count, present for both providers.
        usage.cache_creation_input_tokens
            Anthropic-only; tokens billed at the cache-write premium.
        usage.cache_read_input_tokens
            Anthropic-only; same value as cached_tokens, kept for
            callers that already key off the native Anthropic name.

    Anthropic's ``input_tokens`` excludes the cache buckets -- the
    real prompt size is ``input_tokens + cache_creation_input_tokens
    + cache_read_input_tokens``. Emitting ``input_tokens`` alone as
    ``prompt_tokens`` undercounts cache-heavy turns and breaks
    downstream context / cost displays, so we add all three input
    buckets together. OpenAI Responses already folds cached tokens
    into ``input_tokens`` so no extra arithmetic is needed there.

    Returns ``None`` when there are no usage numbers to report (e.g. an
    upstream error before ``message_start`` / ``response.completed``).
    """
    if not isinstance(last_usage, dict):
        return None

    completion_tokens = last_usage.get("output_tokens") or 0

    if provider == "anthropic":
        uncached_input = last_usage.get("input_tokens") or 0
        cache_creation = last_usage.get("cache_creation_input_tokens") or 0
        cache_read = last_usage.get("cache_read_input_tokens") or 0
        prompt_tokens = uncached_input + cache_creation + cache_read
        if not (prompt_tokens or completion_tokens):
            return None
        usage_block: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cache_read},
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        }
    else:
        prompt_tokens = last_usage.get("input_tokens") or 0
        cached = 0
        details = last_usage.get("input_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens") or 0
        if not (prompt_tokens or completion_tokens or cached):
            return None
        usage_block = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        # Surface OpenAI Responses / Gemini reasoning-token detail. The
        # caller pre-populates last_usage["output_tokens_details"] with
        # at least {"reasoning_tokens": ...}; mirror it into the OAI
        # `completion_tokens_details` shape so SDKs can render the
        # hidden-thoughts slice.
        out_details = last_usage.get("output_tokens_details")
        if isinstance(out_details, dict) and out_details:
            usage_block["completion_tokens_details"] = {
                "reasoning_tokens": out_details.get("reasoning_tokens") or 0,
            }
            usage_block["output_tokens_details"] = out_details

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": usage_block,
    }
    return f"data: {_json.dumps(chunk)}"
