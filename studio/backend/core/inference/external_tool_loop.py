# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth-managed tools for normalized external-provider streams.

ExternalProviderClient translates OpenAI Responses, Anthropic Messages, Gemini,
and OpenAI-compatible chat streams into the same ``delta.tool_calls`` shape.
This module executes those calls through Studio's existing tool catalog,
approval gate, sandbox, cancellation, duplicate protection, and live-output
events, then continues the provider conversation with the results.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, Optional

from loggers import get_logger
from core.inference.tool_loop_controller import (
    ToolLoopController,
    append_deferred_nudges,
    awaiting_approval_status,
)
from core.inference.tool_stream_exec import accepts_output_callback, stream_tool_execution
from core.inference.tools import execute_tool, is_high_risk_tool_call
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)


logger = get_logger(__name__)
_KEEPALIVE_LINE = ": keep-alive"
_MAX_TOOL_CALLS_PER_TURN = 8


def _sse_data(line: str) -> object:
    if not isinstance(line, str) or not line.startswith("data:"):
        return None
    raw = line[len("data:") :].strip()
    if raw == "[DONE]":
        return _DONE
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, RecursionError):
        return None


_DONE = object()


def _merge_tool_call_delta(
    accumulators: dict[int, dict[str, Any]], raw_call: Mapping[str, Any]
) -> None:
    """Merge one OpenAI ``delta.tool_calls`` fragment by its stable index."""

    raw_index = raw_call.get("index", 0)
    index = raw_index if isinstance(raw_index, int) and raw_index >= 0 else 0
    entry = accumulators.setdefault(
        index,
        {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    call_id = raw_call.get("id")
    if isinstance(call_id, str) and call_id:
        # IDs are normally sent once. Repeated full IDs must not concatenate.
        if not entry["id"]:
            entry["id"] = call_id
    call_type = raw_call.get("type")
    if isinstance(call_type, str) and call_type:
        entry["type"] = call_type
    raw_function = raw_call.get("function")
    if not isinstance(raw_function, Mapping):
        return
    name = raw_function.get("name")
    if isinstance(name, str) and name:
        entry["function"]["name"] += name
    arguments = raw_function.get("arguments")
    if isinstance(arguments, str):
        entry["function"]["arguments"] += arguments
    elif isinstance(arguments, Mapping):
        # Defensive support for Ollama-native-shaped arguments leaking through
        # an OpenAI-compatible proxy.
        entry["function"]["arguments"] = json.dumps(arguments, ensure_ascii = False)


def _collect_round_delta(
    payload: object,
    tool_calls: dict[int, dict[str, Any]],
    assistant_text: list[str],
) -> None:
    if not isinstance(payload, Mapping):
        return
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            assistant_text.append(content)
        raw_calls = delta.get("tool_calls")
        if isinstance(raw_calls, list):
            for raw_call in raw_calls:
                if isinstance(raw_call, Mapping):
                    _merge_tool_call_delta(tool_calls, raw_call)


def _without_tool_transport(payload: object, *, has_tool_calls: bool) -> Optional[str]:
    """Return a forwardable SSE line with raw function-call deltas removed.

    Studio renders its authoritative ``tool_start``/``tool_end`` events instead
    of exposing upstream fragments. Intermediate ``finish_reason=tool_calls``
    must also stay hidden or the browser will treat the first model pass as the
    end of the response.
    """

    if not isinstance(payload, Mapping):
        return None
    cloned = copy.deepcopy(dict(payload))
    choices = cloned.get("choices")
    if not isinstance(choices, list):
        return "data: " + json.dumps(cloned, ensure_ascii = False)
    kept_choices: list[dict[str, Any]] = []
    for raw_choice in choices:
        if not isinstance(raw_choice, dict):
            continue
        choice = raw_choice
        delta = choice.get("delta")
        if isinstance(delta, dict):
            delta.pop("tool_calls", None)
        if has_tool_calls and choice.get("finish_reason") is not None:
            choice["finish_reason"] = None
        meaningful_delta = isinstance(delta, dict) and bool(delta)
        if meaningful_delta or choice.get("finish_reason") is not None:
            kept_choices.append(choice)
    cloned["choices"] = kept_choices
    # Intermediate usage belongs to an internal tool-selection pass, not the
    # final browser-visible answer. Error payloads must still pass through.
    if has_tool_calls and not kept_choices and "error" not in cloned:
        return None
    # Final-pass usage-only chunks remain useful. Stripped fragments do not.
    if not kept_choices and "usage" not in cloned and "error" not in cloned:
        return None
    return "data: " + json.dumps(cloned, ensure_ascii = False)


def _next_sync(generator) -> tuple[bool, Any]:
    try:
        return False, next(generator)
    except StopIteration as stop:
        return True, stop.value


def _event_line(event: Mapping[str, Any]) -> str:
    if event.get("type") == "heartbeat":
        return _KEEPALIVE_LINE
    if event.get("type") == "status":
        return "data: " + json.dumps(
            {"type": "tool_status", "content": event.get("text", "")},
            ensure_ascii = False,
        )
    return "data: " + json.dumps(dict(event), ensure_ascii = False)


async def stream_external_chat_with_tools(
    *,
    client,
    messages: Sequence[Mapping[str, Any]],
    model: str,
    tools: Sequence[Mapping[str, Any]],
    request_kwargs: Mapping[str, Any],
    tool_choice: Any = None,
    max_tool_iterations: int = 25,
    tool_call_timeout: int = 300,
    session_id: str | None = None,
    thread_id: str | None = None,
    rag_scope: dict | None = None,
    cancel_event: threading.Event | None = None,
    confirm_tool_calls: bool = False,
    bypass_permissions: bool = False,
    permission_mode: str | None = None,
    auto_heal_tool_calls: bool = True,
) -> AsyncGenerator[str, None]:
    """Stream an external chat while executing Studio tools server-side."""

    conversation = [copy.deepcopy(dict(message)) for message in messages]
    max_rounds = max(0, int(max_tool_iterations))
    final_pass = max_rounds == 0
    tool_controller = ToolLoopController(
        tools = [] if final_pass else tools,
        auto_heal_tool_calls = auto_heal_tool_calls,
    )
    timeout = None if tool_call_timeout >= 9999 else max(1, int(tool_call_timeout))
    cancel = cancel_event or threading.Event()
    rounds_with_calls = 0
    next_tool_choice = tool_choice

    while True:
        if cancel.is_set():
            return
        active_tools = tool_controller.active_tools()
        round_calls: dict[int, dict[str, Any]] = {}
        assistant_text: list[str] = []
        upstream = client.stream_chat_completion(
            messages = conversation,
            model = model,
            tools = active_tools or None,
            tool_choice = next_tool_choice if active_tools else "none",
            stream = True,
            enabled_tools = None,
            **dict(request_kwargs),
        )
        try:
            async for line in upstream:
                payload = _sse_data(line)
                if payload is _DONE:
                    if not round_calls:
                        yield line
                    continue
                _collect_round_delta(payload, round_calls, assistant_text)
                forward = _without_tool_transport(payload, has_tool_calls = bool(round_calls))
                if forward is not None:
                    yield forward
        finally:
            try:
                await upstream.aclose()
            except RuntimeError:
                pass

        if not round_calls:
            return
        # The no-tools final pass should not contain calls, but a non-conforming
        # proxy must not turn that response into an unbounded loop.
        if final_pass:
            return

        rounds_with_calls += 1
        normalized_calls: list[dict[str, Any]] = []
        seen_call_keys: set[tuple[str, str]] = set()
        for index in sorted(round_calls):
            call = round_calls[index]
            if not call.get("id"):
                call["id"] = f"call_external_{rounds_with_calls}_{index}"
            key = (
                (call.get("function") or {}).get("name", ""),
                (call.get("function") or {}).get("arguments", ""),
            )
            if key in seen_call_keys:
                continue
            seen_call_keys.add(key)
            normalized_calls.append(call)
            if len(normalized_calls) >= _MAX_TOOL_CALLS_PER_TURN:
                break

        decisions = [tool_controller.prepare_call(call) for call in normalized_calls]
        executable = [decision for decision in decisions if decision.should_execute]
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(assistant_text),
        }
        if executable:
            assistant_message["tool_calls"] = [
                decision.as_assistant_tool_call() for decision in executable
            ]
        if executable or assistant_message["content"]:
            conversation.append(assistant_message)

        deferred_noops: list[dict[str, Any]] = []
        for decision in decisions:
            if not decision.should_execute:
                completion = tool_controller.record_noop(decision)
                deferred_noops.append(completion.model_message())
                continue

            needs_confirm = (
                bool(confirm_tool_calls)
                and not bypass_permissions
                and permission_mode != "off"
            )
            if needs_confirm and permission_mode == "auto":
                needs_confirm = is_high_risk_tool_call(decision.tool_name, decision.arguments)
            approval_id = new_approval_id() if needs_confirm else ""
            decision_slot = (
                begin_tool_decision(session_id, approval_id) if needs_confirm else None
            )
            start_event = decision.tool_start_event()
            start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirm
            try:
                yield _event_line(
                    {
                        "type": "status",
                        "text": (
                            awaiting_approval_status(decision.tool_name)
                            if needs_confirm
                            else decision.status_text
                        ),
                    }
                )
                yield _event_line(start_event)
                approval = (
                    await asyncio.to_thread(
                        wait_tool_decision,
                        decision_slot,
                        approval_id,
                        cancel,
                    )
                    if decision_slot is not None
                    else None
                )
                if approval is not None and approval != "deny":
                    yield _event_line({"type": "status", "text": decision.status_text})
                if approval == "deny":
                    decision_slot = None
                    yield _event_line(
                        {
                            "type": "tool_end",
                            "tool_name": decision.tool_name,
                            "tool_call_id": decision.tool_call_id,
                            "result": TOOL_REJECTED_MESSAGE,
                            "provenance": decision.provenance,
                        }
                    )
                    denied: dict[str, Any] = {
                        "role": "tool",
                        "name": decision.tool_name,
                        "content": TOOL_REJECTED_MESSAGE,
                    }
                    if decision.tool_call_id:
                        denied["tool_call_id"] = decision.tool_call_id
                    conversation.append(denied)
                    continue
                decision_slot = None
            finally:
                if decision_slot is not None:
                    abort_tool_decision(decision_slot, approval_id)

            def _invoke_tool(output_callback, _decision = decision):
                kwargs = dict(
                    cancel_event = cancel,
                    timeout = timeout,
                    session_id = session_id,
                    thread_id = thread_id,
                    rag_scope = rag_scope,
                    disable_sandbox = bypass_permissions,
                )
                if accepts_output_callback(execute_tool):
                    kwargs["output_callback"] = output_callback
                return execute_tool(_decision.tool_name, _decision.arguments, **kwargs)

            tool_stream = stream_tool_execution(
                _invoke_tool,
                tool_name = decision.tool_name,
                tool_call_id = decision.tool_call_id,
                cancel_event = cancel,
            )
            try:
                while True:
                    next_task = asyncio.create_task(
                        asyncio.to_thread(_next_sync, tool_stream)
                    )
                    try:
                        finished, event_or_result = await asyncio.shield(next_task)
                    except asyncio.CancelledError:
                        # Let the bounded poll finish before closing the sync
                        # generator; closing it while ``next`` runs in another
                        # thread raises ``ValueError: generator already executing``.
                        cancel.set()
                        await next_task
                        raise
                    if finished:
                        result = event_or_result
                        break
                    yield _event_line(event_or_result)
            except Exception as exc:
                logger.exception("External tool %s raised: %s", decision.tool_name, exc)
                result = f"Error: tool raised an exception: {exc}"
            finally:
                tool_stream.close()
            completion = tool_controller.record_result(decision, result)
            yield _event_line(completion.tool_end_event())
            conversation.append(completion.tool_message())

        append_deferred_nudges(conversation, deferred_noops)
        yield _event_line({"type": "status", "text": ""})

        if rounds_with_calls >= max_rounds or tool_controller.force_final_answer:
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "You have used all available tool calls. Based on the tool results "
                        "above, provide the final answer now without calling more tools."
                    ),
                }
            )
            tool_controller = ToolLoopController(tools = [])
            final_pass = True
        next_tool_choice = "auto"

