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
        efforts = ("none", "low", "medium", "high", "xhigh"),
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
                        yield line
                except GeneratorExit:
                    await response.aclose()  # set PoolByteStream._closed=True FIRST
                    await lines_gen.aclose()  # now safe — aclose() is a no-op
                    raise
                finally:
                    logger.info(
                        "%s stream complete (model=%s, chosen=%s, events=%s)",
                        self.provider_type,
                        model,
                        chosen_model,
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

                        if event_type == "content_block_delta":
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
                            # signature_delta and any other delta types are
                            # intentionally skipped — they carry trust /
                            # verification metadata, not user-visible content.

                        elif event_type == "content_block_stop":
                            # Close the <think> tag when the thinking block
                            # ends, in case no text_delta follows (e.g.
                            # display=omitted on Claude 4.7, or thinking-only
                            # turns).
                            if thinking_open:
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
                    # Surface per-event-type counts so reports of "no
                    # reasoning panel content" can be triaged at a glance:
                    # zero `content_block_delta:thinking_delta` entries
                    # means Anthropic skipped thinking for this prompt
                    # (adaptive can choose to); non-zero means thinking
                    # arrived and we wrapped it — any visual gap is then
                    # on the frontend.
                    logger.info(
                        "Anthropic stream event counts (model=%s): %s",
                        model,
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

                        elif event_type == "response.output_item.done":
                            item = event.get("item", {})
                            if (
                                isinstance(item, dict)
                                and item.get("type") == "reasoning"
                            ):
                                summary_text = _extract_reasoning_text(
                                    item.get("summary")
                                )
                                if summary_text and not reasoning_emitted:
                                    if not reasoning_open:
                                        summary_text = f"<think>{summary_text}"
                                        reasoning_open = True
                                    yield _chunk_with_text(summary_text)
                                    reasoning_emitted = True

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
