# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio-owned tool execution loop for the ChatGPT Codex subscription transport.

The loop itself now lives in ``core.inference.studio_tool_loop`` and is shared
with every other external provider. What stays here is the Codex transport: the
app-server call signature, its conversation-affinity ids, and the encrypted
reasoning replay that only this provider emits.
"""

from __future__ import annotations

import threading

from dataclasses import dataclass
from collections.abc import AsyncIterator
from typing import Any

from core.inference.openai_codex_client import OpenAICodexClient
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)


@dataclass(frozen = True)
class CodexRunContext:
    provider_id: str
    thread_id: str | None
    session_id: str | None
    messages: list[dict[str, Any]]
    model: str
    reasoning_effort: str | None
    response_format: dict[str, Any] | None = None
    tool_choice: Any = None
    continue_final_message: bool = False


@dataclass(frozen = True)
class CodexToolPolicy:
    tools: list[dict[str, Any]]
    max_calls: int
    timeout: int
    permission_mode: str
    confirm_calls: bool
    bypass_permissions: bool
    rag_scope: dict[str, Any] | None


class CodexTransport:
    """One Codex turn as OpenAI-shaped SSE lines.

    Codex emits structured ``delta.tool_calls`` and never writes a call as text,
    so text-form healing stays off: there is nothing to repair, and running the
    healer would only add a buffering window to a stream that does not need one.
    """

    heals_text_tool_calls = False

    def __init__(self, client: OpenAICodexClient, run: CodexRunContext) -> None:
        self._client = client
        self._run = run

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        cancel_event: threading.Event,
    ) -> AsyncIterator[str]:
        return self._client.stream(
            provider_id = self._run.provider_id,
            thread_id = self._run.thread_id,
            messages = messages,
            model = self._run.model,
            max_tokens = None,
            reasoning_effort = self._run.reasoning_effort,
            response_format = self._run.response_format,
            tools = tools,
            tool_choice = tool_choice,
            cancel_event = cancel_event,
        )


def stream_codex_with_studio_tools(
    client: OpenAICodexClient,
    *,
    run: CodexRunContext,
    policy: CodexToolPolicy,
    cancel_event: threading.Event,
) -> AsyncIterator[str]:
    """Stream Codex, execute requested Studio tools, and continue until a final answer."""
    return stream_with_studio_tools(
        CodexTransport(client, run),
        run = ToolLoopRun(
            messages = run.messages,
            session_id = run.session_id,
            thread_id = run.thread_id,
            tool_choice = run.tool_choice,
            continue_final_message = run.continue_final_message,
        ),
        policy = ToolLoopPolicy(
            tools = policy.tools,
            max_calls = policy.max_calls,
            timeout = policy.timeout,
            permission_mode = policy.permission_mode,
            confirm_calls = policy.confirm_calls,
            bypass_permissions = policy.bypass_permissions,
            rag_scope = policy.rag_scope,
            auto_heal = False,
        ),
        cancel_event = cancel_event,
    )
