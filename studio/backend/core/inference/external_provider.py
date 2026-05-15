# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Async HTTP client for proxying chat completions to external LLM providers.

Most registry providers expose OpenAI-compatible /v1/chat/completions endpoints;
Anthropic uses native Messages API with translation in this client.
"""

import json as _json
import re
from typing import Any, AsyncGenerator, Literal, NamedTuple, Optional

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

# Claude 4.7 (Opus/Sonnet/Haiku) deprecated top_k and returns 400
# "top_k is deprecated for this model" when it is set. 3.x and 4.5/4.6
# still accept it. Match the 4-7 line specifically so we keep the knob
# live on every other Claude generation.
_ANTHROPIC_TOP_K_DEPRECATED = re.compile(r"^claude-(?:opus|sonnet|haiku)-4-7(?:[-.]|$)")


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

        headers = {
            "Content-Type": "application/json",
            auth_header: f"{auth_prefix}{self.api_key}",
        }
        # Merge any provider-specific extra headers (e.g. anthropic-version, OpenRouter attribution)
        headers.update(provider_info.get("extra_headers", {}))
        return headers

    def _is_openai_compatible(self) -> bool:
        """Return False for providers that need request/response translation (e.g. Anthropic)."""
        from core.inference.providers import get_provider_info

        info = get_provider_info(self.provider_type) or {}
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
                # Translate OpenAI image_url parts → Anthropic native image format
                anthropic_parts: list[dict[str, Any]] = []
                for part in content:
                    if part.get("type") == "text":
                        anthropic_parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # data:image/png;base64,<DATA> → split header and data
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
                            # Remote URL — Anthropic supports url source type natively.
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
                filtered.append({"role": msg["role"], "content": anthropic_parts})
            else:
                filtered.append(msg)

        body: dict[str, Any] = {
            "model": model,
            "messages": filtered,
            "max_tokens": max_tokens or 1024,  # required by Anthropic
            "temperature": temperature,
            "stream": True,
        }
        # top_k is deprecated on Claude 4.7 (Opus/Sonnet/Haiku) — the API
        # returns 400 "top_k is deprecated for this model" when it is set.
        # 3.x and 4.5/4.6 still accept it, so gate strictly on the 4.7 ids.
        if (
            top_k is not None
            and top_k > 0
            and not _ANTHROPIC_TOP_K_DEPRECATED.match(model)
        ):
            body["top_k"] = top_k
        if system:
            body["system"] = system
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
            # Anthropic requires temperature=1 whenever thinking is enabled,
            # AND forbids top_p in the same request: setting both produces
            #   "temperature and top_p cannot both be specified for this
            #    model. Please use only one."
            # The base body never sets top_p, but pop defensively in case
            # an upstream edit ever adds it before this branch runs.
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
        #   https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
        # The tool type is date-pinned (web_search_20250305 today) and
        # Anthropic dispatches search calls server-side, returning
        # server_tool_use + web_search_tool_result blocks in the SSE
        # stream, plus url-citation annotations on text deltas. We
        # translate all of that into our local _toolEvent shape so the
        # chat UI renders web_search exactly like OpenAI's path.
        if enabled_tools and "web_search" in enabled_tools:
            anthropic_tools = list(body.get("tools") or [])
            anthropic_tools.append(
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                }
            )
            body["tools"] = anthropic_tools

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

        _finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }

        logger.info("Proxying Anthropic Messages API to %s (model=%s)", url, model)

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
                        "Anthropic returned %d: %s",
                        response.status_code,
                        error_text[:500],
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

                        if event_type == "content_block_start":
                            content_block = event.get("content_block") or {}
                            block_type = content_block.get("type")
                            if (
                                block_type == "server_tool_use"
                                and content_block.get("name") == "web_search"
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
                                # First text after a thinking block closes the
                                # <think> tag we opened above. Anthropic emits
                                # a content_block_stop between blocks, but
                                # closing on the text_delta transition is more
                                # forgiving if events arrive out of order.
                                if thinking_open:
                                    yield _content_chunk("</think>")
                                    thinking_open = False
                                text = delta.get("text", "")
                                if text:
                                    yield _content_chunk(text)
                                # Citations on text deltas are attached
                                # per-call by Anthropic via the
                                # `web_search_tool_result` block; we don't
                                # need to scrape them off the text events.
                            elif (
                                delta_type == "input_json_delta"
                                and current_server_tool_use is not None
                            ):
                                # Streamed partial_json carrying the search
                                # query. Buffer until content_block_stop.
                                current_server_tool_use["buffer"] += delta.get(
                                    "partial_json", ""
                                )
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
                            elif thinking_open:
                                # Close the <think> tag when the thinking block
                                # ends, in case no text_delta follows (e.g.
                                # display=omitted on Claude 4.7, or thinking-
                                # only turns).
                                yield _content_chunk("</think>")
                                thinking_open = False

                        elif event_type == "message_delta":
                            stop_reason = event.get("delta", {}).get("stop_reason")
                            if stop_reason:
                                if thinking_open:
                                    yield _content_chunk("</think>")
                                    thinking_open = False
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": _finish_reason_map.get(
                                                stop_reason, "stop"
                                            ),
                                        }
                                    ],
                                }
                                yield f"data: {_json.dumps(chunk)}"

                        elif event_type == "message_stop":
                            if thinking_open:
                                yield _content_chunk("</think>")
                                thinking_open = False
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
                    logger.info(
                        "Anthropic stream complete (model=%s, "
                        "web_search_requested=%s, web_search_invocations=%s, "
                        "results=%s, queries=%s, events=%s)",
                        model,
                        web_search_requested,
                        web_search_invocations,
                        total_results,
                        queries,
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
        if reasoning_effort in (
            "minimal",
            "low",
            "medium",
            "high",
            "max",
            "xhigh",
        ):
            body["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        elif reasoning_effort == "none" or enable_thinking is False:
            body["reasoning"] = {"effort": "none"}
        elif enable_thinking is True:
            body["reasoning"] = {"effort": "medium", "summary": "auto"}
        if instructions_parts:
            body["instructions"] = "\n\n".join(instructions_parts)
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens

        # OpenAI server-side tools — see
        #   https://developers.openai.com/api/docs/guides/tools
        # The frontend's Search button maps to the unified
        # enabled_tools=["web_search"] shorthand; translate that into the
        # Responses-API tool schema. Other built-in tools (file_search,
        # code_interpreter, image_generation, computer_use_preview) can be
        # added with the same pattern when we surface their toggles.
        if enabled_tools:
            tools_array: list[dict[str, Any]] = []
            if "web_search" in enabled_tools:
                tools_array.append({"type": "web_search"})
            if tools_array:
                body["tools"] = tools_array

        url = f"{self.base_url}/responses"
        completion_id = f"chatcmpl-openai-{model.replace('/', '-')}"

        logger.info("Proxying OpenAI Responses API to %s (model=%s)", url, model)

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
                        "OpenAI Responses returned %d: %s",
                        response.status_code,
                        error_text[:500],
                    )
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

                        elif isinstance(event_type, str) and "reasoning" in event_type:
                            reasoning_delta = _extract_reasoning_text(event)
                            if reasoning_delta:
                                if not reasoning_open:
                                    reasoning_delta = f"<think>{reasoning_delta}"
                                    reasoning_open = True
                                yield _chunk_with_text(reasoning_delta)
                                reasoning_emitted = True

                        elif event_type == "response.completed":
                            if reasoning_open:
                                yield _chunk_with_text("</think>")
                                reasoning_open = False
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
                                        f"Title: {cit['title']}\n" f"URL: {cit['url']}"
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

                        elif event_type == "response.incomplete":
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
                                        f"Title: {cit['title']}\n" f"URL: {cit['url']}"
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
                    logger.info(
                        "OpenAI Responses stream complete (model=%s, "
                        "web_search_requested=%s, web_search_invocations=%s, "
                        "citations=%s, queries=%s, reasoning_emitted=%s)",
                        model,
                        web_search_requested,
                        web_search_invocations,
                        total_citations,
                        queries,
                        reasoning_emitted,
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
            models = data.get("data", [])
            return models
        except httpx.HTTPError as exc:
            logger.error("Failed to list models from %s: %s", self.provider_type, exc)
            raise

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

    async def close(self) -> None:
        """No-op — the underlying client is shared across requests."""


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
