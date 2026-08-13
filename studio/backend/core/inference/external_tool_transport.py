# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Transport that drives the shared Studio tool loop over any external provider.

``ExternalProviderClient.stream_chat_completion`` is the single point every
external provider passes through: the OpenAI-compatible route yields its lines
directly, and the Anthropic, Gemini, Responses and Kimi sub-streams are each
translated into OpenAI chunk shape and re-yielded through it. Wrapping that one
generator therefore covers self-hosted llama.cpp / vLLM / Ollama / custom
endpoints and the hosted APIs alike.
"""

from __future__ import annotations

import threading

from collections.abc import AsyncIterator
from typing import Any

from core.inference.external_provider import ExternalProviderClient


class OAICompatTransport:
    """One provider turn as OpenAI-shaped SSE lines.

    Sampling parameters and the provider-hosted tool flags (``enabled_tools``,
    the code-execution container ids, prompt caching, thinking) are captured
    once and replayed on every follow-up turn. Dropping them after the first
    turn would silently change sampling mid-answer and revoke the provider's own
    hosted tools as soon as a local tool ran.
    """

    heals_text_tool_calls = True

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
        # stream_chat_completion has no cancel_event parameter: cancellation is
        # driven by the route closing this generator, which propagates as
        # GeneratorExit through the httpx stream context.
        return self._client.stream_chat_completion(
            messages = messages,
            model = self._model,
            tools = tools,
            tool_choice = tool_choice,
            continue_final_message = self._continue_final_message,
            **self._request_kwargs,
        )
