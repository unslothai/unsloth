# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Safetensors/transformers agentic tool loop.

Wraps a single-turn cumulative-text generator with the same tool-calling,
thinking-block, status, and metadata event protocol the GGUF path uses, so
the front-end SSE shape is identical across backends.

Unlike the GGUF path (``llama_cpp.py``), which uses llama-server's structured
``delta.tool_calls``, native transformers has no such channel, so this loop
parses tool calls from the cumulative text and dispatches via
``core.inference.tools``.
"""

import re
import threading
from typing import Callable, Generator, Optional

from loggers import get_logger

from core.inference.tool_call_parser import (
    _TOOL_ALL_PATS,
    _strip_gemma_native_spans,
    BUDGET_EXHAUSTED_NUDGE,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    TOOL_XML_SIGNALS,
    parse_tool_calls_from_text,
    strip_tool_markup,
    strip_tool_patterns,
)
from core.inference.tool_loop_controller import (
    ToolLoopController,
    coerce_tool_arguments,
    status_for_tool,
    tool_event_provenance,
)
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)


logger = get_logger(__name__)


# Buffer cap while disambiguating a possible tool-call prefix.
_MAX_BUFFER_CHARS = 32


def strip_tool_markup_streaming(
    text: str,
    *,
    auto_heal_tool_calls: bool = True,
    tool_protocol_active: bool = False,
) -> str:
    """Strip open-ended tool XML from display text without trimming whitespace."""
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text
    # Quote-aware Gemma spans first, else a literal <tool_call|> inside a quoted
    # argument truncates the regex match and leaks the suffix into display.
    text = _strip_gemma_native_spans(text, final = True)
    return strip_tool_patterns(text, _TOOL_ALL_PATS)


def _strip_tool_markup_final(
    text: str,
    *,
    auto_heal_tool_calls: bool,
    tool_protocol_active: bool = False,
) -> str:
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text
    return strip_tool_markup(text, final = True)


def _status_for_tool(tool_name: str, arguments: dict) -> str:
    """Return a human-readable status line matching the GGUF path."""
    return status_for_tool(tool_name, arguments)


_FUNCTION_SIGNAL_RE = re.compile(r"<function=([\w-]+)>")
_TOOL_CALL_NAME_RE = re.compile(r'"name"\s*:\s*"([\w-]+)"')


def _detect_render_html_tool_start(content: str) -> bool:
    """Return True when the first drained tool call is clearly render_html."""
    function_match = _FUNCTION_SIGNAL_RE.search(content)
    tool_call_index = content.find("<tool_call>")
    if not function_match and tool_call_index < 0:
        return False

    if function_match and (tool_call_index < 0 or function_match.start() < tool_call_index):
        return function_match.group(1) == "render_html"

    if tool_call_index >= 0:
        name_match = _TOOL_CALL_NAME_RE.search(content[tool_call_index:])
        return bool(name_match and name_match.group(1) == "render_html")

    return False


def _coerce_arguments_with_provenance(
    raw_args,
    *,
    heal: bool,
    tool_name: str = "",
):
    """Normalise tool ``arguments`` and report whether healing was applied."""
    coerced = coerce_tool_arguments(raw_args, heal = heal, tool_name = tool_name)
    return coerced.arguments, coerced.healed


def _coerce_arguments(
    raw_args,
    *,
    heal: bool,
    tool_name: str = "",
) -> dict:
    arguments, _ = _coerce_arguments_with_provenance(
        raw_args,
        heal = heal,
        tool_name = tool_name,
    )
    return arguments


def _tool_event_provenance(**flags: object) -> dict[str, object]:
    return tool_event_provenance(**flags)


def _call_single_turn(single_turn, conversation: list, active_tools: list[dict]):
    """Call a single-turn generator with active tool schemas when supported."""
    try:
        return single_turn(conversation, active_tools = active_tools)
    except TypeError as exc:
        if "active_tools" not in str(exc):
            raise
        return single_turn(conversation)


def run_safetensors_tool_loop(
    *,
    single_turn: Callable[[list], Generator[str, None, None]],
    messages: list[dict],
    tools: list[dict],
    execute_tool: Callable[..., str],
    cancel_event: Optional[threading.Event] = None,
    auto_heal_tool_calls: bool = True,
    max_tool_iterations: int = 25,
    tool_call_timeout: int = 300,
    session_id: Optional[str] = None,
    rag_scope: Optional[dict] = None,
    confirm_tool_calls: bool = False,
    bypass_permissions: bool = False,
) -> Generator[dict, None, None]:
    """Drive an agentic tool loop on top of a cumulative-text generator.

    ``single_turn(messages)`` must yield cumulative assistant text (each
    yield is a snapshot of all tokens so far). The loop:

    * Buffers each turn's leading chars to decide whether a tool call is
      coming. Plain content streams once the buffer rules it out.
    * On ``<tool_call>`` or ``<function=`` in the cumulative text, drains
      the rest of the turn silently and parses tool calls from the content.
    * Executes each tool via ``execute_tool``, appends the assistant
      tool-call message and tool result, and re-enters ``single_turn``.
    * After ``max_tool_iterations`` turns without a final answer, asks once
      more for a final answer with no tools.

    Yields event dicts matching the GGUF path:

    * ``{"type": "status", "text": ...}`` -- empty string clears the badge.
    * ``{"type": "content", "text": ...}`` -- cumulative cleaned text for
      the current turn (consumer diffs against its own ``prev_text`` cursor).
    * ``{"type": "tool_start", "tool_name", "tool_call_id", "arguments"}``
    * ``{"type": "tool_end", "tool_name", "tool_call_id", "result"}``
    """
    conversation = list(messages)

    # Forced first-pass RAG (mirrors the GGUF loop) so doc Qs don't lose to web_search.
    from core.inference.tools import build_rag_autoinject

    _auto = None if confirm_tool_calls else build_rag_autoinject(conversation, rag_scope)
    if _auto:
        for _ev in _auto["events"]:
            yield _ev
        conversation.extend(_auto["messages"])

    unrestricted_tools = not tools
    tool_controller = ToolLoopController(
        tools = None if unrestricted_tools else tools,
        auto_heal_tool_calls = auto_heal_tool_calls,
    )
    # RAG: cap knowledge-base searches per assistant turn (controller-agnostic).
    kb_search_count = 0
    final_attempt_done = False
    next_call_id = 0

    def _tool_succeeded(tool_name: str) -> bool:
        key_prefix = f"{tool_name}:"
        return any(
            record.executed and not record.is_error and record.key.startswith(key_prefix)
            for record in tool_controller.history
        )

    if max_tool_iterations <= 0:
        # 0 = disabled (same contract as the GGUF loop).
        yield {"type": "status", "text": ""}
        return

    _state_buffering = 0
    _state_streaming = 1
    _state_draining = 2

    for iteration in range(max_tool_iterations + 1):
        if cancel_event is not None and cancel_event.is_set():
            return

        if final_attempt_done:
            active_tools: list[dict] = []
        else:
            active_tools = tool_controller.active_tools()
            if not active_tools and not unrestricted_tools:
                final_attempt_done = True
                active_tools = []

        tool_protocol_active = not final_attempt_done and (unrestricted_tools or bool(active_tools))
        tool_xml_signals = TOOL_XML_SIGNALS if tool_protocol_active else ()

        detect_state = _state_buffering
        content_buffer = ""
        content_accum = ""
        cumulative_display = ""
        last_emitted = ""
        provisional_render_html_started = False
        provisional_resolved = False
        provisional_render_html_id = f"call_{next_call_id}"
        # When a human confirmation gate is active the real tool_start is keyed
        # by an approval id and carries awaiting_confirmation, so an early
        # provisional card (keyed by tool_call_id, no approval) would show the
        # tool as "running" before the user has approved it. Suppress the early
        # card in that case and let the gated tool_start be the first signal.
        _provisional_confirm_gated = bool(confirm_tool_calls) and not bypass_permissions

        gen = _call_single_turn(single_turn, conversation, active_tools)
        prev_cumulative = ""

        _gen_iter = iter(gen)
        while True:
            try:
                cumulative = next(_gen_iter)
            except StopIteration:
                break
            except Exception:
                # The model pipeline raised mid-stream. If a provisional
                # render_html card was already surfaced, close it as errored so
                # the UI never leaves a tool card spinning after the turn fails.
                if provisional_render_html_started and not provisional_resolved:
                    provisional_resolved = True
                    yield {
                        "type": "tool_end",
                        "tool_name": "render_html",
                        "tool_call_id": provisional_render_html_id,
                        "result": "Error: generation was interrupted before the tool call completed.",
                        "provenance": _tool_event_provenance(provisional = True),
                    }
                raise

            if cancel_event is not None and cancel_event.is_set():
                return

            if not isinstance(cumulative, str):
                continue  # defensive: pipeline yields only strings

            delta = cumulative[len(prev_cumulative) :]
            prev_cumulative = cumulative
            if not delta:
                continue
            content_accum += delta

            if detect_state == _state_draining:
                if (
                    not _tool_succeeded("render_html")
                    and not _provisional_confirm_gated
                    and any(
                        ((tool.get("function") or {}).get("name") == "render_html")
                        for tool in active_tools
                    )
                    and not provisional_render_html_started
                    and _detect_render_html_tool_start(content_accum)
                ):
                    provisional_render_html_started = True
                    yield {
                        "type": "tool_start",
                        "tool_name": "render_html",
                        "tool_call_id": provisional_render_html_id,
                        "arguments": {},
                        "provenance": _tool_event_provenance(provisional = True),
                    }
                continue

            if detect_state == _state_streaming:
                candidate = cumulative_display + delta
                signal_pos = -1
                for sig in tool_xml_signals:
                    p = candidate.find(sig)
                    if p >= 0 and (signal_pos < 0 or p < signal_pos):
                        signal_pos = p
                if signal_pos >= 0:
                    before_tool = candidate[:signal_pos]
                    cleaned_before = strip_tool_markup_streaming(
                        before_tool,
                        auto_heal_tool_calls = auto_heal_tool_calls,
                        tool_protocol_active = tool_protocol_active,
                    )
                    if len(cleaned_before) > len(last_emitted):
                        last_emitted = cleaned_before
                        yield {"type": "content", "text": cleaned_before}
                    cumulative_display = candidate
                    detect_state = _state_draining
                    if (
                        not _tool_succeeded("render_html")
                        and not _provisional_confirm_gated
                        and any(
                            ((tool.get("function") or {}).get("name") == "render_html")
                            for tool in active_tools
                        )
                        and not provisional_render_html_started
                        and _detect_render_html_tool_start(content_accum)
                    ):
                        provisional_render_html_started = True
                        yield {
                            "type": "tool_start",
                            "tool_name": "render_html",
                            "tool_call_id": provisional_render_html_id,
                            "arguments": {},
                            "provenance": _tool_event_provenance(provisional = True),
                        }
                    continue
                cumulative_display = candidate
                cleaned = strip_tool_markup_streaming(
                    cumulative_display,
                    auto_heal_tool_calls = auto_heal_tool_calls,
                    tool_protocol_active = tool_protocol_active,
                )
                if len(cleaned) > len(last_emitted):
                    last_emitted = cleaned
                    yield {"type": "content", "text": cleaned}
                continue

            # BUFFERING: hold until we know it is not a tool call.
            content_buffer += delta
            stripped = content_buffer.lstrip()
            if not stripped:
                continue

            is_match = False
            is_prefix = False
            for sig in tool_xml_signals:
                if stripped.startswith(sig):
                    is_match = True
                    break
                if sig.startswith(stripped):
                    is_prefix = True
                    break

            if is_match:
                # Tool signal -- flush any visible prefix before DRAINING
                # so the route sends it before tool_start.
                cumulative_display += content_buffer
                cleaned = strip_tool_markup_streaming(
                    cumulative_display,
                    auto_heal_tool_calls = auto_heal_tool_calls,
                    tool_protocol_active = tool_protocol_active,
                )
                if len(cleaned) > len(last_emitted):
                    last_emitted = cleaned
                    yield {"type": "content", "text": cleaned}
                detect_state = _state_draining
                if (
                    not _tool_succeeded("render_html")
                    and not _provisional_confirm_gated
                    and any(
                        ((tool.get("function") or {}).get("name") == "render_html")
                        for tool in active_tools
                    )
                    and not provisional_render_html_started
                    and _detect_render_html_tool_start(content_accum)
                ):
                    provisional_render_html_started = True
                    yield {
                        "type": "tool_start",
                        "tool_name": "render_html",
                        "tool_call_id": provisional_render_html_id,
                        "arguments": {},
                        "provenance": _tool_event_provenance(provisional = True),
                    }
            elif is_prefix and len(stripped) < _MAX_BUFFER_CHARS:
                continue
            else:
                detect_state = _state_streaming
                cumulative_display += content_buffer
                cleaned = strip_tool_markup_streaming(
                    cumulative_display,
                    auto_heal_tool_calls = auto_heal_tool_calls,
                    tool_protocol_active = tool_protocol_active,
                )
                if len(cleaned) > len(last_emitted):
                    last_emitted = cleaned
                    yield {"type": "content", "text": cleaned}

        # Stream finished -- resolve what we collected.
        if cancel_event is not None and cancel_event.is_set():
            return

        if detect_state == _state_buffering:
            # Buffer never resolved -- tool XML or plain content?
            stripped = content_buffer.lstrip()
            if (
                stripped
                and tool_protocol_active
                and any(sig in stripped for sig in tool_xml_signals)
            ):
                detect_state = _state_draining
            else:
                if content_buffer:
                    cumulative_display += content_buffer
                    yield {
                        "type": "content",
                        "text": _strip_tool_markup_final(
                            cumulative_display,
                            auto_heal_tool_calls = auto_heal_tool_calls,
                            tool_protocol_active = False,
                        ),
                    }
                yield {"type": "status", "text": ""}
                return

        if detect_state == _state_streaming:
            # No tool detected mid-stream -- check for late tool XML.
            safety_tc = None
            saw_tool_signal = tool_protocol_active and any(
                sig in content_accum for sig in tool_xml_signals
            )
            if saw_tool_signal:
                safety_tc = parse_tool_calls_from_text(
                    content_accum,
                    id_offset = next_call_id,
                    allow_incomplete = auto_heal_tool_calls,
                )
            if not safety_tc:
                # Final answer: if a literal tool marker in prose was stripped
                # during streaming but did not parse as a real call, restore the
                # raw cumulative text for core callers. Route-level cleanup can
                # still apply the Auto-Heal display policy.
                if saw_tool_signal and content_accum:
                    yield {"type": "content", "text": content_accum}
                yield {"type": "status", "text": ""}
                return
            tool_calls = safety_tc
            content_text = _strip_tool_markup_final(
                content_accum,
                auto_heal_tool_calls = auto_heal_tool_calls,
                tool_protocol_active = True,
            )
            logger.info(
                "Safetensors safety net: parsed %d tool call(s) from streamed content",
                len(tool_calls),
            )
        else:
            # DRAINING: parse tool calls out of full content.
            tool_calls = parse_tool_calls_from_text(
                content_accum,
                id_offset = next_call_id,
                allow_incomplete = auto_heal_tool_calls,
            )
            if not tool_calls:
                # Parser found nothing. Auto-Heal-enabled display cleanup
                # strips unparseable tool XML; disabled Auto-Heal preserves
                # the raw text so literal/malformed markup stays visible.
                if content_accum:
                    yield {
                        "type": "content",
                        "text": _strip_tool_markup_final(
                            content_accum,
                            auto_heal_tool_calls = auto_heal_tool_calls,
                            tool_protocol_active = False,
                        ),
                    }
                if provisional_render_html_started and not provisional_resolved:
                    provisional_resolved = True
                    yield {
                        "type": "tool_end",
                        "tool_name": "render_html",
                        "tool_call_id": provisional_render_html_id,
                        "result": "Error: render_html tool call could not be parsed.",
                        "provenance": _tool_event_provenance(provisional = True),
                    }
                yield {"type": "status", "text": ""}
                return
            content_text = _strip_tool_markup_final(
                content_accum,
                auto_heal_tool_calls = auto_heal_tool_calls,
                tool_protocol_active = True,
            )

        if tool_calls:
            next_call_id += len(tool_calls)

        if final_attempt_done:
            # Final-answer turn re-called a tool -- stop the loop.
            if content_text:
                yield {"type": "content", "text": content_text}
            yield {"type": "status", "text": ""}
            return

        assistant_msg: dict = {"role": "assistant", "content": content_text}
        assistant_appended = False

        for tc in tool_calls or []:
            func = tc.get("function", {}) or {}
            tool_name = func.get("name", "") or ""
            provisional_match = (
                provisional_render_html_started
                and tool_name == "render_html"
                and tc.get("id", "") == provisional_render_html_id
            )
            decision = tool_controller.prepare_call(tc, provisional = provisional_match)

            if not decision.should_execute:
                if content_text and not assistant_appended:
                    conversation.append(assistant_msg)
                    assistant_appended = True
                if provisional_match and not provisional_resolved:
                    # A provisional render_html card is already on screen for
                    # this id; close it so it never dangles when the controller
                    # turns the call into an internal no-op (duplicate / repeat).
                    provisional_resolved = True
                    yield {
                        "type": "tool_end",
                        "tool_name": decision.tool_name,
                        "tool_call_id": decision.tool_call_id,
                        "result": "",
                        "provenance": decision.provenance,
                    }
                completion = tool_controller.record_noop(decision)
                conversation.append(completion.model_message())
                logger.info(
                    "Suppressed local safetensors tool call as internal no-op: "
                    f"action={decision.action} tool={decision.tool_name}"
                )
                break

            if not assistant_appended:
                assistant_msg["tool_calls"] = [decision.as_assistant_tool_call()]
                conversation.append(assistant_msg)
                assistant_appended = True
            else:
                assistant_msg.setdefault("tool_calls", []).append(decision.as_assistant_tool_call())

            # Bypass wins over the confirm gate at the loop level too, so a
            # direct internal caller passing both flags never prompts.
            needs_confirm = bool(confirm_tool_calls) and not bypass_permissions
            approval_id = new_approval_id() if needs_confirm else ""
            decision_slot = begin_tool_decision(session_id, approval_id) if needs_confirm else None
            start_event = decision.tool_start_event()
            start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirm

            try:
                yield {"type": "status", "text": decision.status_text}
                yield start_event

                if (
                    decision_slot is not None
                    and wait_tool_decision(
                        decision_slot,
                        approval_id,
                        cancel_event = cancel_event,
                    )
                    == "deny"
                ):
                    decision_slot = None
                    if provisional_match:
                        provisional_resolved = True
                    yield {
                        "type": "tool_end",
                        "tool_name": decision.tool_name,
                        "tool_call_id": decision.tool_call_id,
                        "result": TOOL_REJECTED_MESSAGE,
                        "provenance": decision.provenance,
                    }
                    denied_message = {
                        "role": "tool",
                        "name": decision.tool_name,
                        "content": TOOL_REJECTED_MESSAGE,
                    }
                    if decision.tool_call_id:
                        denied_message["tool_call_id"] = decision.tool_call_id
                    conversation.append(denied_message)
                    continue
                decision_slot = None
            finally:
                if decision_slot is not None:
                    abort_tool_decision(decision_slot, approval_id)

            eff_timeout = None if tool_call_timeout >= 9999 else tool_call_timeout
            # RAG: cap paraphrased KB re-searches that slip past the dup guard.
            if (
                decision.tool_name == "search_knowledge_base"
                and kb_search_count >= RAG_MAX_SEARCHES_PER_TURN
            ):
                result = RAG_SEARCH_CAP_NUDGE
            else:
                try:
                    result = execute_tool(
                        decision.tool_name,
                        decision.arguments,
                        cancel_event = cancel_event,
                        timeout = eff_timeout,
                        session_id = session_id,
                        rag_scope = rag_scope,
                        disable_sandbox = bypass_permissions,
                    )
                except Exception as exc:
                    logger.exception("Tool %s raised: %s", decision.tool_name, exc)
                    result = f"Error: tool raised an exception: {exc}"
                if decision.tool_name == "search_knowledge_base":
                    kb_search_count += 1

            completion = tool_controller.record_result(decision, result)
            if provisional_match:
                provisional_resolved = True
            yield completion.tool_end_event()
            conversation.append(completion.tool_message())

        # Clear the status badge before the next turn.
        yield {"type": "status", "text": ""}

        if tool_controller.force_final_answer:
            final_attempt_done = True
            continue
        if not unrestricted_tools and not tool_controller.active_tools():
            final_attempt_done = True
            continue
        if iteration + 1 >= max_tool_iterations and not final_attempt_done:
            # Budget exhausted; nudge a final plain answer.
            final_attempt_done = True
            conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})

    yield {"type": "status", "text": ""}
