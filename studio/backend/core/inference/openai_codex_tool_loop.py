# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio-owned tool execution loop for the ChatGPT Codex subscription transport."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any

from core.inference.openai_codex_client import OpenAICodexClient
from core.inference.tools import execute_tool, is_high_risk_tool_call
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":"))


def _chunk_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    raw = line[5:].strip()
    if not raw or raw == "[DONE]":
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _normalized_call(call: dict[str, Any]) -> dict[str, Any] | None:
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
        return None
    name = function.get("name")
    arguments = function.get("arguments", "")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = {"_raw": arguments}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments or "{}"},
        "arguments": parsed,
    }


async def stream_codex_with_studio_tools(
    client: OpenAICodexClient,
    *,
    provider_id: str,
    thread_id: str | None,
    session_id: str | None,
    messages: list[dict[str, Any]],
    model: str,
    reasoning_effort: str | None,
    tools: list[dict[str, Any]],
    max_tool_calls: int,
    tool_call_timeout: int,
    permission_mode: str,
    confirm_tool_calls: bool,
    bypass_permissions: bool,
    rag_scope: dict[str, Any] | None,
    cancel_event: threading.Event,
) -> AsyncIterator[str]:
    """Stream Codex, execute requested Studio tools, and continue until a final answer."""
    conversation = [dict(message) for message in messages]
    remaining = max_tool_calls
    unlimited = remaining >= 9999

    while not cancel_event.is_set():
        by_index: dict[int, dict[str, Any]] = {}
        assistant_text: list[str] = []
        reasoning_extra: dict[str, Any] | None = None
        finish_reason: str | None = None

        generator = client.stream(
            provider_id=provider_id,
            thread_id=thread_id,
            messages=conversation,
            model=model,
            max_tokens=None,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice="auto",
            cancel_event=cancel_event,
        )
        async for line in generator:
            payload = _chunk_payload(line)
            if payload:
                choices = payload.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else {}
                if isinstance(choice, dict):
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str):
                            assistant_text.append(content)
                        extra = delta.get("extra_content")
                        if isinstance(extra, dict):
                            reasoning_extra = extra
                        raw_calls = delta.get("tool_calls")
                        if isinstance(raw_calls, list):
                            for raw_call in raw_calls:
                                if not isinstance(raw_call, dict):
                                    continue
                                index = raw_call.get("index")
                                if not isinstance(index, int):
                                    index = len(by_index)
                                current = by_index.setdefault(
                                    index,
                                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                                )
                                if isinstance(raw_call.get("id"), str):
                                    current["id"] = raw_call["id"]
                                function = raw_call.get("function")
                                if isinstance(function, dict):
                                    if isinstance(function.get("name"), str):
                                        current["function"]["name"] = function["name"]
                                    if isinstance(function.get("arguments"), str):
                                        current["function"]["arguments"] += function["arguments"]
                    if isinstance(choice.get("finish_reason"), str):
                        finish_reason = choice["finish_reason"]
            yield line

        calls = [
            normalized
            for _, call in sorted(by_index.items())
            if (normalized := _normalized_call(call)) is not None
        ]
        if finish_reason != "tool_calls" or not calls or (not unlimited and remaining <= 0):
            return

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(assistant_text),
            "tool_calls": [
                {"id": call["id"], "type": "function", "function": call["function"]}
                for call in calls
            ],
        }
        if reasoning_extra:
            assistant_message["extra_content"] = reasoning_extra
        conversation.append(assistant_message)

        for call in calls:
            if not unlimited and remaining <= 0:
                break
            if not unlimited:
                remaining -= 1
            call_id = call["id"]
            name = call["function"]["name"]
            arguments = call["arguments"]
            needs_confirmation = (
                confirm_tool_calls
                and not bypass_permissions
                and permission_mode != "off"
            )
            if needs_confirmation and permission_mode == "auto":
                needs_confirmation = is_high_risk_tool_call(name, arguments)
            approval_id = new_approval_id() if needs_confirmation else ""
            decision_slot = (
                begin_tool_decision(session_id, approval_id) if needs_confirmation else None
            )
            yield _sse(
                {
                    "type": "tool_start",
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "arguments": arguments,
                    "approval_id": approval_id,
                    "awaiting_confirmation": needs_confirmation,
                }
            )
            try:
                decision = (
                    await asyncio.to_thread(
                        wait_tool_decision,
                        decision_slot,
                        approval_id,
                        cancel_event=cancel_event,
                    )
                    if decision_slot is not None
                    else None
                )
                if decision == "deny":
                    decision_slot = None
                    result = TOOL_REJECTED_MESSAGE
                else:
                    decision_slot = None
                    timeout = None if tool_call_timeout >= 9999 else tool_call_timeout
                    result = await asyncio.to_thread(
                        execute_tool,
                        name,
                        arguments,
                        cancel_event=cancel_event,
                        timeout=timeout,
                        session_id=session_id,
                        thread_id=thread_id,
                        rag_scope=rag_scope,
                        disable_sandbox=bypass_permissions,
                    )
            finally:
                if decision_slot is not None:
                    abort_tool_decision(decision_slot, approval_id)
            result_text = result if isinstance(result, str) else json.dumps(result)
            yield _sse(
                {
                    "type": "tool_end",
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "arguments": arguments,
                    "result": result_text,
                }
            )
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result_text,
                }
            )
