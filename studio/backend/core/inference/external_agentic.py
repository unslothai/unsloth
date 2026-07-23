# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local tool loop for OpenAI-compatible external Connections (#7282).

Ollama / llama.cpp / vLLM / Custom Connections speak `/v1/chat/completions` on
a remote host, but Unsloth's Search / Code / MCP tools still execute on the
Studio machine. This module drives that loop:

1. Stream a chat completion with OpenAI function tools attached.
2. Forward content deltas as OpenAI SSE chunks.
3. On ``tool_calls``, confirm (optional) + execute locally, emit
   ``tool_start`` / ``tool_end`` events the chat UI already understands.
4. Append the assistant tool-call message + tool results and continue.

Hosted providers (OpenAI / Anthropic / …) stay on the pure-proxy +
``supportsBuiltin*`` path — they must not enter this loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from core.inference.tool_loop_controller import strip_result_for_model
from core.inference.tools import (
    execute_tool,
    is_always_safe_tool,
    is_potentially_unsafe_tool_call,
)
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)

logger = logging.getLogger(__name__)

# OAI-compat Connections that can drive Unsloth's local tool runtime.
LOCAL_TOOL_RUNTIME_PROVIDER_TYPES = frozenset({"ollama", "llama_cpp", "vllm", "custom"})


def provider_supports_local_tool_runtime(provider_type: Optional[str]) -> bool:
    """True for OAI-compat Connections that may use Studio's local tools."""
    return (provider_type or "") in LOCAL_TOOL_RUNTIME_PROVIDER_TYPES


def _parse_sse_data_line(line: str) -> Optional[dict[str, Any]]:
    """Parse one ``data: {...}`` SSE line into a dict; ignore heartbeats/[DONE]."""
    if not line:
        return None
    text = line.strip()
    if text.startswith("data:"):
        text = text[5:].strip()
    if not text or text == "[DONE]":
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _merge_tool_call_delta(acc: dict[int, dict[str, Any]], tc_delta: dict[str, Any]) -> None:
    """Accumulate a streaming ``delta.tool_calls[]`` fragment into ``acc``."""
    try:
        idx = int(tc_delta.get("index", 0))
    except (TypeError, ValueError):
        idx = 0
    slot = acc.setdefault(
        idx,
        {
            "id": "",
            "type": "tool",
            "function": {"name": "", "arguments": ""},
        },
    )
    if tc_delta.get("id"):
        slot["id"] = str(tc_delta["id"])
    if tc_delta.get("type"):
        slot["type"] = str(tc_delta["type"])
    func = tc_delta.get("function") or {}
    if isinstance(func, dict):
        if func.get("name"):
            slot["function"]["name"] += str(func["name"])
        if func.get("arguments"):
            slot["function"]["arguments"] += str(func["arguments"])


def _coerce_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse tool-call argument JSON; fall back to ``{}`` on malformed input."""
    if not raw or not str(raw).strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _openai_content_chunk_line(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
    finish_reason: Optional[str] = None,
) -> str:
    """Build one OpenAI chat-completion SSE ``data:`` line for a content delta."""
    choice: dict[str, Any] = {
        "index": 0,
        "delta": {"content": content} if content else {},
        "finish_reason": finish_reason,
    }
    if not content and finish_reason is None:
        choice["delta"] = {}
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload, ensure_ascii = False)}"


async def stream_external_local_tool_loop(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    top_k: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    tool_choice: Optional[Any] = None,
    confirm_tool_calls: bool = False,
    bypass_permissions: bool = False,
    permission_mode: Optional[str] = None,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    tool_call_timeout: int = 300,
    max_tool_iterations: int = 25,
    rag_scope: Optional[dict] = None,
    cancel_event: Optional[threading.Event] = None,
    completion_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Drive a local tool loop against an OpenAI-compatible remote model.

    Yields SSE ``data:`` lines (without the trailing blank line) matching the
    shapes the chat frontend already handles for local GGUF tool loops.
    """
    if permission_mode == "full":
        bypass_permissions = True
    elif bypass_permissions:
        permission_mode = "full"
    elif permission_mode not in ("ask", "auto", "off"):
        permission_mode = "ask"

    conversation = [dict(m) for m in messages]
    completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    cancel_event = cancel_event or threading.Event()

    # 9999 is the "no limit" sentinel; pass None so execute_tool never times out
    # (mirrors the local GGUF loop).
    effective_tool_timeout = None if tool_call_timeout >= 9999 else tool_call_timeout
    # Total tool calls executed across all rounds must not exceed the caller's
    # per-message budget, even when a single completion returns parallel calls.
    calls_remaining = max_tool_iterations

    enabled_names = {
        (t.get("function") or {}).get("name")
        for t in tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    enabled_names.discard(None)

    for _iteration in range(max_tool_iterations):
        if cancel_event.is_set():
            break

        tool_calls_acc: dict[int, dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        saw_content = False
        assistant_content = ""

        gen = client.stream_chat_completion(
            messages = conversation,
            model = model,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            presence_penalty = presence_penalty,
            top_k = top_k,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            tools = tools,
            tool_choice = tool_choice,
            stream = True,
        )
        # Watcher: Stop may fire while the remote is still in prefill and the
        # iterator is blocked awaiting the next SSE line. Close the generator
        # so the `async for` unblocks immediately instead of waiting for a
        # chunk or the read timeout.
        async def _watch_cancel(active_gen) -> None:
            try:
                while not cancel_event.is_set():
                    await asyncio.sleep(0.1)
                await active_gen.aclose()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        watcher = asyncio.ensure_future(_watch_cancel(gen))
        try:
            async for line in gen:
                if cancel_event.is_set():
                    break
                payload = _parse_sse_data_line(line)
                if payload is None:
                    # Preserve SSE comments/fields (e.g. `: ping`, `event:`) as-is.
                    if line and line.strip() and line.strip() != "data: [DONE]":
                        yield line
                    continue
                if payload.get("error"):
                    yield line if line.startswith("data:") else f"data: {json.dumps(payload)}"
                    return

                choices = payload.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}

                content = delta.get("content")
                if content:
                    saw_content = True
                    assistant_content += str(content)
                    yield _openai_content_chunk_line(
                        completion_id = completion_id,
                        created = created,
                        model = model,
                        content = str(content),
                    )

                # Reasoning-style fields some OAI-compat servers emit.
                for key in ("reasoning_content", "reasoning"):
                    reasoning = delta.get(key)
                    if reasoning:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": str(reasoning)},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii = False)}"

                for tc_delta in delta.get("tool_calls") or []:
                    if isinstance(tc_delta, dict):
                        _merge_tool_call_delta(tool_calls_acc, tc_delta)
        finally:
            watcher.cancel()
            try:
                await watcher
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await gen.aclose()
            except Exception:
                pass

        ordered_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc) if tool_calls_acc[i]]
        # Drop empty / nameless fragments.
        ordered_calls = [tc for tc in ordered_calls if (tc.get("function") or {}).get("name")]

        if not ordered_calls or finish_reason not in (None, "tool_calls", "stop"):
            # No tool calls — emit a terminal finish if we streamed content.
            if saw_content or finish_reason:
                yield _openai_content_chunk_line(
                    completion_id = completion_id,
                    created = created,
                    model = model,
                    content = "",
                    finish_reason = finish_reason or "stop",
                )
            yield "data: [DONE]"
            return

        # Enforce the per-message call budget across parallel calls in a round.
        if calls_remaining <= 0:
            break
        ordered_calls = ordered_calls[:calls_remaining]

        # Normalize ids for the conversation transcript.
        assistant_tool_calls = []
        for i, tc in enumerate(ordered_calls):
            tc_id = tc.get("id") or f"call_{i}"
            tc["id"] = tc_id
            assistant_tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments") or "{}",
                    },
                }
            )

        conversation.append(
            {
                "role": "assistant",
                # Retain any streamed explanation so the follow-up round keeps
                # the model's own context for interpreting tool results.
                "content": assistant_content or None,
                "tool_calls": assistant_tool_calls,
            }
        )

        for tc in assistant_tool_calls:
            if cancel_event.is_set():
                break
            calls_remaining -= 1
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            arguments = _coerce_tool_arguments(raw_args)
            tool_call_id = tc["id"]

            # Skip tools the caller did not enable (defense in depth).
            if enabled_names and name not in enabled_names:
                result = f"Error: tool '{name}' is not enabled for this request."
                yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': name, 'tool_call_id': tool_call_id, 'arguments': arguments})}"
                yield f"data: {json.dumps({'type': 'tool_end', 'tool_name': name, 'tool_call_id': tool_call_id, 'result': result})}"
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": strip_result_for_model(result),
                    }
                )
                continue

            needs_confirm = bool(confirm_tool_calls) and not bypass_permissions
            if needs_confirm and permission_mode == "auto":
                # auto only pauses calls flagged unsafe; always-safe tools skip.
                if is_always_safe_tool(name) or not is_potentially_unsafe_tool_call(
                    name, arguments
                ):
                    needs_confirm = False
            elif needs_confirm and permission_mode == "off":
                needs_confirm = False

            approval_id = new_approval_id() if needs_confirm else ""
            decision_slot = begin_tool_decision(session_id, approval_id) if needs_confirm else None
            start_event: dict[str, Any] = {
                "type": "tool_start",
                "tool_name": name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
            }
            if approval_id:
                start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirm
            result = ""
            try:
                yield f"data: {json.dumps(start_event, ensure_ascii = False)}"

                denied = False
                if needs_confirm and decision_slot is not None:
                    decision = await asyncio.to_thread(
                        wait_tool_decision,
                        decision_slot,
                        approval_id,
                        cancel_event,
                    )
                    if decision != "allow":
                        denied = True
                        result = TOOL_REJECTED_MESSAGE
                    else:
                        decision_slot = None
                if not denied:
                    try:
                        result = await asyncio.to_thread(
                            execute_tool,
                            name,
                            arguments,
                            cancel_event,
                            effective_tool_timeout,
                            session_id,
                            thread_id,
                            rag_scope,
                            bypass_permissions,
                            None,
                        )
                    except Exception as exc:
                        logger.exception("external_agentic.tool_failed name=%s", name)
                        result = f"Error executing tool '{name}': {exc}"
            finally:
                if decision_slot is not None:
                    abort_tool_decision(decision_slot, approval_id)

            model_result = result if isinstance(result, str) else str(result)
            yield f"data: {json.dumps({'type': 'tool_end', 'tool_name': name, 'tool_call_id': tool_call_id, 'result': model_result}, ensure_ascii = False)}"
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": strip_result_for_model(model_result),
                }
            )

        continue

    # Budget exhausted after tool rounds — one final synthesis pass without tools.
    final_finish_reason = None
    if not cancel_event.is_set():
        if max_tool_iterations > 0:
            from core.inference.tool_call_parser import BUDGET_EXHAUSTED_NUDGE
            conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})
        final_gen = client.stream_chat_completion(
            messages = conversation,
            model = model,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            presence_penalty = presence_penalty,
            top_k = top_k,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            tools = None,
            tool_choice = "none",
            stream = True,
        )
        try:
            async for line in final_gen:
                if cancel_event.is_set():
                    break
                payload = _parse_sse_data_line(line)
                if payload is None:
                    if line and line.strip() and line.strip() != "data: [DONE]":
                        yield line
                    continue
                if payload.get("error"):
                    yield line if line.startswith("data:") else f"data: {json.dumps(payload)}"
                    return

                choices = payload.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    # Surface truncation (length/content_filter) to clients.
                    final_finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}

                content = delta.get("content")
                if content:
                    yield _openai_content_chunk_line(
                        completion_id = completion_id,
                        created = created,
                        model = model,
                        content = str(content),
                    )

                for key in ("reasoning_content", "reasoning"):
                    reasoning = delta.get(key)
                    if reasoning:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": str(reasoning)},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii = False)}"
        finally:
            try:
                await final_gen.aclose()
            except Exception:
                pass

    yield _openai_content_chunk_line(
        completion_id = completion_id,
        created = created,
        model = model,
        content = "",
        finish_reason = final_finish_reason or "stop",
    )
    yield "data: [DONE]"
