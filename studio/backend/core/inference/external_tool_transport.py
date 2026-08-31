# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Transport that drives the shared Unsloth tool loop over any external provider.

``ExternalProviderClient.stream_chat_completion`` is the single point every
external provider passes through: the OpenAI-compatible route yields its lines
directly, and the Anthropic, Gemini, Responses and Kimi sub-streams are each
translated into OpenAI chunk shape and re-yielded through it. Wrapping that one
generator therefore covers self-hosted llama.cpp / vLLM / Ollama / custom
endpoints and the hosted APIs alike.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading

from collections.abc import AsyncIterator
from typing import Any

from core.inference.external_provider import ExternalProviderClient


# /inference/cancel and the model-load path only set a threading.Event, so an
# asyncio consumer has to poll it. Same interval as the Codex client's watcher.
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
    # ExternalProviderClient sanitizes every raw upstream line at the point it
    # arrives, before any translation. What it yields on top of that is this
    # server's own synthesized frames (a provider-hosted image or web-search
    # result), which a second pass in the loop could no longer tell apart.
    sanitizes_provider_frames = True

    def __init__(
        self,
        client: ExternalProviderClient,
        *,
        model: str,
        continue_final_message: bool | None = None,
        **request_kwargs: Any,
    ) -> None:
        self._client = client
        self._model = model
        self._continue_final_message = continue_final_message
        self._request_kwargs = request_kwargs

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        cancel_event: threading.Event,
    ) -> AsyncIterator[str]:
        # "Resume the trailing assistant turn", so it is only ever true of the
        # first request. Once a tool runs the conversation ends with a role="tool"
        # result (or a role="user" no-op note), and vLLM / llama.cpp would splice
        # the generation prompt off the end of *that* message: the model continues
        # the tool output instead of answering it, or the chat template raises and
        # the server 400s. Re-read the tail every turn rather than replaying the
        # flag the transport was constructed with.
        continue_final_message = bool(self._continue_final_message) and bool(
            messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant"
        )
        return self._cancellable(
            self._client.stream_chat_completion(
                messages = messages,
                model = self._model,
                tools = tools,
                tool_choice = tool_choice,
                continue_final_message = continue_final_message,
                **self._request_kwargs,
            ),
            cancel_event,
        )

    @staticmethod
    async def _cancellable(
        upstream: AsyncIterator[str], cancel_event: threading.Event
    ) -> AsyncIterator[str]:
        """Relay ``upstream``, ending as soon as ``cancel_event`` is set.

        ``stream_chat_completion`` takes no cancel_event, and /inference/cancel
        and the model-load path only set the flag: nothing else can reach the
        provider socket. Relying on the route closing this generator does not
        cover them, because a closed generator is only noticed at the next yield
        and Stop arrives while the read is parked. Without this race the read
        stays parked until the provider emits again, so Stop keeps consuming
        billed tokens and a model load waits behind a stalled upstream.
        Cancelling the pending read raises inside the upstream generator at its
        own await, which runs the ``finally`` that closes the httpx response;
        calling aclose() here instead would hit "generator already executing".
        """
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
            # The pre-existing teardown: the route or the loop closing this
            # generator still reaches the provider stream through here.
            aclose = getattr(upstream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(RuntimeError, GeneratorExit, StopAsyncIteration):
                    await aclose()
