# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio-owned tool execution loop, shared by every external provider transport.

The loop owns the parts that do not depend on how bytes reach the provider:
turn cycling, the tool budget, the approval handshake, local execution through
``execute_tool``, and the conversation replay that carries results back. A
``ToolLoopTransport`` supplies the one thing that does differ -- an async
iterator of OpenAI-shaped SSE lines for one turn.

Two transports exist: ``CodexTransport`` (the ChatGPT subscription, which speaks
its own app-server protocol) and ``OAICompatTransport`` (everything routed
through ``ExternalProviderClient``, which already normalises Anthropic, Gemini
and the Responses API into OpenAI chunk shape).

Self-hosted models frequently write tool calls as text rather than emitting
structured ``delta.tool_calls``. A transport that sets ``heals_text_tool_calls``
gets the content stream routed through ``StreamToolCallHealer``, the same
bounded buffer the client-tool passthrough uses: only a trailing partial-signal
window or a suspected tool block is ever withheld, and a block that cannot
become a declared call flushes verbatim. Nothing here invents a cap on how much
model output may be held.
"""

from __future__ import annotations

import asyncio
import json
import threading

from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any, Protocol

from core.inference.passthrough_healing import StreamToolCallHealer, heal_gate
from core.inference.tool_loop_controller import strip_result_for_model
from core.inference.tools import build_rag_autoinject, execute_tool, is_high_risk_tool_call
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)


_TOOL_BUDGET_EXHAUSTED = (
    "Studio did not execute this tool call because the per-message tool-call limit was reached. "
    "Continue with the available results and answer without calling another tool."
)

_TOOL_DISABLED = "Studio did not execute this tool call because the tool is disabled."


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, separators = (",", ":"))


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


def _tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    return {
        name
        for tool in tools or []
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and isinstance((name := tool["function"].get("name")), str)
        and name
    }


class ToolLoopTransport(Protocol):
    """One turn of provider inference, as OpenAI-shaped SSE lines."""

    # Whether the provider may write tool calls as text instead of emitting
    # structured delta.tool_calls. Codex never does; a self-hosted GGUF often does.
    heals_text_tool_calls: bool

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        cancel_event: threading.Event,
    ) -> AsyncIterator[str]: ...


@dataclass(frozen = True)
class ToolLoopRun:
    """Everything about the request that the loop, not the transport, needs."""

    messages: list[dict[str, Any]]
    session_id: str | None = None
    thread_id: str | None = None
    tool_choice: Any = None
    continue_final_message: bool = False


@dataclass(frozen = True)
class ToolLoopPolicy:
    tools: list[dict[str, Any]]
    max_calls: int
    timeout: int
    permission_mode: str
    confirm_calls: bool
    bypass_permissions: bool
    rag_scope: dict[str, Any] | None
    # None means "follow the process default"; False disables text-form healing.
    auto_heal: bool | None = None


@dataclass
class _Turn:
    """Accumulated state for one provider turn."""

    by_index: dict[int, dict[str, Any]] = field(default_factory = dict)
    healed: list[dict[str, Any]] = field(default_factory = list)
    text: list[str] = field(default_factory = list)
    reasoning_extra: dict[str, Any] | None = None
    finish_reason: str | None = None

    def merge_structured(self, raw_calls: list[Any]) -> None:
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            index = raw_call.get("index")
            if not isinstance(index, int):
                index = len(self.by_index)
            current = self.by_index.setdefault(
                index,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if isinstance(raw_call.get("id"), str):
                current["id"] = raw_call["id"]
            function = raw_call.get("function")
            if isinstance(function, dict):
                # Assignment, not concatenation: llama-server re-sends the whole
                # name as it grows, so appending turns "web" then "web_search"
                # into "webweb_search" and the call silently never runs.
                if isinstance(function.get("name"), str):
                    current["function"]["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    current["function"]["arguments"] += function["arguments"]

    def calls(self) -> list[dict[str, Any]]:
        structured = [
            normalized
            for _, call in sorted(self.by_index.items())
            if (normalized := _normalized_call(call)) is not None
        ]
        if not self.healed:
            return structured
        # Healed ids are minted locally (call_0, call_1, ...) and could collide
        # with provider-supplied ids. The healed call never came off the wire,
        # so its id is ours to rename.
        taken = {call["id"] for call in structured}
        healed: list[dict[str, Any]] = []
        for position, call in enumerate(self.healed):
            normalized = _normalized_call(call)
            if normalized is None:
                continue
            if normalized["id"] in taken:
                normalized["id"] = f"healed_{position}_{normalized['id']}"
            taken.add(normalized["id"])
            healed.append(normalized)
        return structured + healed


def _rewrite_content(payload: dict[str, Any], choice: dict[str, Any], text: str) -> str:
    """Re-emit a chunk with its content replaced by what the healer released."""
    new_delta = {key: value for key, value in choice.get("delta", {}).items() if key != "content"}
    if text:
        new_delta["content"] = text
    new_choice = {key: value for key, value in choice.items() if key != "delta"}
    new_choice["delta"] = new_delta
    new_payload = {key: value for key, value in payload.items() if key != "choices"}
    new_payload["choices"] = [new_choice] + list(payload.get("choices", [])[1:])
    return _sse(new_payload)


async def stream_with_studio_tools(
    transport: ToolLoopTransport,
    *,
    run: ToolLoopRun,
    policy: ToolLoopPolicy,
    cancel_event: threading.Event,
) -> AsyncIterator[str]:
    """Stream a provider, execute requested Studio tools, continue to a final answer."""
    conversation = [dict(message) for message in run.messages]
    remaining = policy.max_calls
    unlimited = remaining >= 9999
    session_id = run.session_id
    thread_id = run.thread_id
    tools = policy.tools
    tool_choice = run.tool_choice if run.tool_choice is not None else "auto"
    allowed_tool_names = _tool_names(tools)
    tool_call_timeout = policy.timeout
    permission_mode = policy.permission_mode
    confirm_tool_calls = policy.confirm_calls
    bypass_permissions = policy.bypass_permissions
    rag_scope = policy.rag_scope

    # The promotion allowlist is the selected catalog, never None: an
    # unrestricted parse re-opens markerless tool-call promotion.
    heal_names = (
        heal_gate(policy.auto_heal, tools, tool_choice) if transport.heals_text_tool_calls else None
    )

    skip_autoinject = run.continue_final_message or (
        confirm_tool_calls and not bypass_permissions and permission_mode not in ("auto", "off")
    )
    autoinject = (
        None
        if skip_autoinject
        else await asyncio.to_thread(build_rag_autoinject, conversation, rag_scope)
    )
    if autoinject:
        for event in autoinject["events"]:
            yield _sse(event)
        conversation.extend(autoinject["messages"])

    round_id = 0
    executed_any = False

    while not cancel_event.is_set():
        turn = _Turn()
        healer = StreamToolCallHealer(heal_names, tools) if heal_names else None

        tools_available = tool_choice != "none" and (unlimited or remaining > 0)
        # Withdrawing the catalog and pinning "none" together: this path owns the
        # tool surface (the route withholds enabled_tools), so there are no
        # provider-hosted builtins left for "none" to revoke, and saying it
        # explicitly stops a model from calling a tool it was just denied.
        turn_tool_choice = tool_choice if tools_available else "none"
        # A forced choice applies until the model actually calls something; the
        # result follow-up must be free to answer in prose.
        if executed_any and turn_tool_choice not in ("auto", "none"):
            turn_tool_choice = "auto"

        generator = transport.stream(
            messages = conversation,
            tools = tools if tools_available else None,
            tool_choice = turn_tool_choice,
            cancel_event = cancel_event,
        )
        async for line in generator:
            payload = _chunk_payload(line)
            if payload is None:
                yield line
                continue
            choices = payload.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else {}
            if not isinstance(choice, dict):
                yield line
                continue

            delta = choice.get("delta")
            delta = delta if isinstance(delta, dict) else {}
            content = delta.get("content")
            raw_calls = delta.get("tool_calls")
            extra = delta.get("extra_content")
            if isinstance(extra, dict):
                turn.reasoning_extra = extra
            if isinstance(choice.get("finish_reason"), str):
                turn.finish_reason = choice["finish_reason"]

            if isinstance(raw_calls, list):
                if healer is not None and not healer.dormant:
                    # Grammar mode worked; stop second-guessing the text stream.
                    # Whatever was being held was ordinary prose after all, so it
                    # has to reach the client, not just the conversation replay.
                    for kind, value in healer.structured_tool_call_seen():
                        if kind == "text" and value:
                            turn.text.append(value)
                            yield _sse({"choices": [{"index": 0, "delta": {"content": value}}]})
                turn.merge_structured(raw_calls)

            if healer is None or healer.dormant or not isinstance(content, str) or not content:
                if isinstance(content, str):
                    turn.text.append(content)
                yield line
                continue

            released: list[str] = []
            for kind, value in healer.feed(content):
                if kind == "text":
                    if value:
                        released.append(value)
                elif kind == "tool_call":
                    turn.healed.append(value)
            visible = "".join(released)
            if visible:
                turn.text.append(visible)
            if visible == content:
                # Nothing was held back, which is the case for almost every chunk
                # of ordinary prose. Relay the provider's own bytes rather than
                # paying a re-encode per chunk to reproduce them.
                yield line
                continue
            # Withholding everything is normal mid-block. Only drop the chunk when
            # it carries nothing else worth relaying.
            if visible or turn.finish_reason is not None or len(delta) > 1:
                yield _rewrite_content(payload, choice, visible)

        if healer is not None:
            for kind, value in healer.finalize():
                if kind == "text":
                    if value:
                        turn.text.append(value)
                        yield _sse({"choices": [{"index": 0, "delta": {"content": value}}]})
                elif kind == "tool_call":
                    turn.healed.append(value)

        calls = turn.calls()
        if not calls or (turn.finish_reason != "tool_calls" and not turn.healed):
            return

        round_id += 1

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(turn.text),
            "tool_calls": [
                {"id": call["id"], "type": "function", "function": call["function"]}
                for call in calls
            ],
        }
        if turn.reasoning_extra:
            assistant_message["extra_content"] = turn.reasoning_extra
        conversation.append(assistant_message)

        for call in calls:
            call_id = call["id"]
            name = call["function"]["name"]
            arguments = call["arguments"]
            allowed_call = tool_choice != "none" and name in allowed_tool_names
            has_budget = unlimited or remaining > 0
            within_budget = allowed_call and has_budget
            needs_confirmation = (
                allowed_call
                and confirm_tool_calls
                and not bypass_permissions
                and permission_mode != "off"
            )
            if needs_confirmation and permission_mode == "auto":
                needs_confirmation = is_high_risk_tool_call(name, arguments)
            approval_id = new_approval_id() if needs_confirmation else ""
            decision_slot = (
                begin_tool_decision(session_id, approval_id)
                if needs_confirmation and within_budget
                else None
            )
            yield _sse(
                {
                    "type": "tool_start",
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "arguments": arguments,
                    "provenance": {"source": "local", "round_id": round_id},
                    "approval_id": approval_id if within_budget else "",
                    "awaiting_confirmation": needs_confirmation and within_budget,
                }
            )
            if not has_budget:
                result = _TOOL_BUDGET_EXHAUSTED
            elif not allowed_call:
                result = _TOOL_DISABLED
            else:
                try:
                    decision = (
                        await asyncio.to_thread(
                            wait_tool_decision,
                            decision_slot,
                            approval_id,
                            cancel_event = cancel_event,
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
                        try:
                            result = await asyncio.to_thread(
                                execute_tool,
                                name,
                                arguments,
                                cancel_event = cancel_event,
                                timeout = timeout,
                                session_id = session_id,
                                thread_id = thread_id,
                                rag_scope = rag_scope,
                                disable_sandbox = bypass_permissions,
                            )
                        except Exception as exc:
                            if cancel_event.is_set():
                                return
                            result = f"Error: tool raised an exception: {exc}"
                        else:
                            # Only a turn that really ran a tool spends an iteration,
                            # so a duplicate or a denial cannot exhaust the catalog.
                            executed_any = True
                            if not unlimited:
                                remaining -= 1
                finally:
                    if decision_slot is not None:
                        abort_tool_decision(decision_slot, approval_id)
            result_text = result if isinstance(result, str) else json.dumps(result)

            model_result = strip_result_for_model(result_text, name)
            yield _sse(
                {
                    "type": "tool_end",
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "arguments": arguments,
                    "result": result_text,
                    "provenance": {"source": "local", "round_id": round_id},
                }
            )
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": model_result,
                }
            )
