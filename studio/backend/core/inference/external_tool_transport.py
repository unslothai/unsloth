# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool loop over ExternalProviderClient.stream_chat_completion, which every external provider uses"""

from __future__ import annotations

import asyncio
import contextlib
import threading

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from core.inference.cloud_compaction import fit_steps as _fit_steps, stream_with_overflow_retry
from core.inference.external_provider import ExternalProviderClient, _is_kimi_k3


# The cancel paths only set a threading.Event, so poll it (same interval as the Codex client).
_CANCEL_POLL_S = 0.05


async def _await_cancel(cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(_CANCEL_POLL_S)


class OAICompatTransport:
    """One provider turn as OpenAI-shaped SSE lines.

    Sampling parameters and the provider-hosted tool flags (``enabled_tools``,
    the code-execution container ids, prompt caching, thinking) are captured
    once and replayed on every follow-up turn. Dropping them after the first
    turn would silently change sampling mid-answer and revoke the provider's own
    hosted tools as soon as a local tool ran.
    """

    heals_text_tool_calls = True
    # The client already sanitized upstream lines; what remains here is our own synthesized frames.
    sanitizes_provider_frames = True

    def __init__(
        self,
        client: ExternalProviderClient,
        *,
        model: str,
        continue_final_message: bool | None = None,
        message_fitter: Callable[
            [list[dict[str, Any]], list[dict[str, Any]] | None],
            Awaitable[tuple[list[dict[str, Any]], int | None, str | None]],
        ]
        | None = None,
        overflow_refitter: Callable[..., Any] | None = None,
        first_turn_overrides: dict[str, Any] | None = None,
        **request_kwargs: Any,
    ) -> None:
        """``message_fitter`` may be a coroutine function or an async generator function that
        yields progress lines and then its ``(messages, max_tokens, line)`` result.
        ``overflow_refitter`` has the same shape and compacts harder: it runs once per request,
        when the provider refuses a turn as too long before any output.
        ``first_turn_overrides`` replace request kwargs on the first provider turn only."""
        self._client = client
        self._model = model
        self._continue_final_message = continue_final_message
        self._message_fitter = message_fitter
        self._overflow_refitter = overflow_refitter
        self._first_turn_overrides = dict(first_turn_overrides or {})
        self._request_kwargs = request_kwargs
        # Anthropic can leave a hosted call pending beside a client call; its continuation accepts only tool results.
        self.tool_result_only_continuation = client.provider_type == "anthropic"
        self._initial_message_count: int | None = None
        self.sent_messages: list[dict[str, Any]] | None = None
        self._last_tools: list[dict[str, Any]] | None = None
        # Always, as the local GGUF loop does: preserve_thinking hides earlier turns, never this turn's tool rounds.
        self.preserves_reasoning = client.provider_type in (
            "llama_cpp",
            "vllm",
            "minimax",
            "deepseek",
        ) or _is_kimi_k3(client.provider_type, model)

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        cancel_event: threading.Event,
    ) -> AsyncIterator[str]:
        if tools:
            self._last_tools = tools
        elif self.tool_result_only_continuation and self._last_tools:
            tools, tool_choice = self._last_tools, "none"
        if self._initial_message_count is not None and self.tool_result_only_continuation:
            # Promoted MCP images join the tool result: a new user turn would end a pending server-tool turn.
            normalized = list(messages[: self._initial_message_count])
            for message in messages[self._initial_message_count :]:
                if (
                    message.get("role") == "user"
                    and normalized
                    and normalized[-1].get("role") == "tool"
                ):
                    parts = []
                    for source in (normalized[-1], message):
                        content = source.get("content")
                        if isinstance(content, list):
                            parts.extend(content)
                        elif isinstance(content, str) and content:
                            parts.append({"type": "text", "text": content})
                    normalized[-1] = {**normalized[-1], "content": parts}
                else:
                    normalized.append(message)
            messages = normalized
        elif self._initial_message_count is None:
            self._initial_message_count = len(messages)
        return self._cancellable(
            self._stream_provider(messages, tools, tool_choice),
            cancel_event,
        )

    async def _stream_provider(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, tool_choice: Any
    ) -> AsyncIterator[str]:
        request_kwargs = dict(self._request_kwargs)
        unfitted = messages
        if self._first_turn_overrides:
            request_kwargs.update(self._first_turn_overrides)
            self._first_turn_overrides = {}
        if self._message_fitter is not None:
            fit = None
            async for step in _fit_steps(self._message_fitter(messages, tools)):
                if isinstance(step, str):
                    yield step
                else:
                    fit = step
            messages, max_tokens, truncation_line = fit
            request_kwargs["max_tokens"] = max_tokens
            if truncation_line:
                yield truncation_line
        self.sent_messages = messages

        def _send(messages: list[dict[str, Any]], request_kwargs: dict[str, Any]):
            # Re-derived each turn: a tool or user message at the end must not be resumed as an
            # assistant turn.
            continue_final_message = bool(self._continue_final_message) and bool(
                messages
                and isinstance(messages[-1], dict)
                and messages[-1].get("role") == "assistant"
            )
            return self._client.stream_chat_completion(
                messages = messages,
                model = self._model,
                tools = tools,
                tool_choice = tool_choice,
                continue_final_message = continue_final_message,
                **request_kwargs,
            )

        async def _recover():
            refitter, self._overflow_refitter = self._overflow_refitter, None
            fit = None
            async for step in _fit_steps(refitter(unfitted, tools)):
                if not isinstance(step, str):
                    fit = step
            if fit is None or fit[0] is None:
                return None
            refit_messages, max_tokens, truncation_line = fit
            self.sent_messages = refit_messages

            async def _resend():
                if truncation_line:
                    yield truncation_line
                async for line in _send(
                    refit_messages, {**request_kwargs, "max_tokens": max_tokens}
                ):
                    yield line

            return _resend

        if self._overflow_refitter is not None:
            upstream = stream_with_overflow_retry(lambda: _send(messages, request_kwargs), _recover)
        else:
            upstream = _send(messages, request_kwargs)
        try:
            async for line in upstream:
                yield line
        finally:
            aclose = getattr(upstream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(RuntimeError, GeneratorExit, StopAsyncIteration):
                    await aclose()

    @staticmethod
    async def _cancellable(
        upstream: AsyncIterator[str], cancel_event: threading.Event
    ) -> AsyncIterator[str]:
        """Races each read against cancel_event; do not aclose() here, it raises generator already executing."""
        iterator = upstream.__aiter__()
        watcher = asyncio.ensure_future(_await_cancel(cancel_event))
        try:
            while not cancel_event.is_set():
                read = asyncio.ensure_future(iterator.__anext__())
                try:
                    await asyncio.wait({read, watcher}, return_when = asyncio.FIRST_COMPLETED)
                except BaseException:
                    read.cancel()
                    raise
                if not read.done():
                    read.cancel()
                    with contextlib.suppress(BaseException):
                        await read
                    return
                try:
                    line = read.result()
                except StopAsyncIteration:
                    return
                yield line
        finally:
            watcher.cancel()
            # Teardown path: the route or loop closing this generator still reaches the provider stream here.
            aclose = getattr(upstream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(RuntimeError, GeneratorExit, StopAsyncIteration):
                    await aclose()
