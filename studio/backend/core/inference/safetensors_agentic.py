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
    _GEMMA_BARE_TC_PREFIX_RE,
    _GEMMA_BARE_TC_RE,
    _TOOL_ALL_PATS,
    _balanced_brace_end,
    _strip_function_xml_calls,
    _strip_gemma_wrapperless_calls,
    _strip_glm_calls,
    _strip_mistral_closed_calls,
    _strip_mistral_reasoning,
    BUDGET_EXHAUSTED_NUDGE,
    MAX_ACT_REPROMPTS,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    TOOL_XML_SIGNALS,
    is_short_intent_without_action,
    parse_tool_calls_from_text,
    reprompt_to_act_message,
    strip_leading_bare_json_call,
    strip_llama3_leading_sentinels,
    strip_tool_markup,
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

# Memory bound for holding a leading bare-JSON object whose top-level "{" never balances.
_MAX_BARE_JSON_BUFFER = 16384


# No grammar constraint here (unlike llama-server's lazy grammar): collapse
# exact-duplicate calls and cap the count so a runaway turn cannot fan out.
_MAX_TOOL_CALLS_PER_TURN = 8


def _active_tool_names(active_tools: list[dict]) -> list[str]:
    names = [
        (tool.get("function") or {}).get("name")
        for tool in active_tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    return [name for name in names if name]


def strip_tool_markup_streaming(
    text: str,
    *,
    auto_heal_tool_calls: bool = True,
    tool_protocol_active: bool = False,
    enabled_tool_names: Optional[set] = None,
) -> str:
    """Strip open-ended tool XML from display text without trimming whitespace.
    ``enabled_tool_names`` gates the markerless Gemma ``call:NAME{...}`` strip so a
    disabled/example name in prose is kept (mirrors the parser gate)."""
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text
    # Mirror the final strip's scan order so streaming and final display agree:
    # balanced strips first (nested JSON removed whole), then the guarded
    # function-XML/GLM scans that close at each call's REAL terminator, so literal
    # markup inside argument values is data and trailing prose survives. No final
    # trim so streaming length comparisons hold. Leading Magistral [THINK]...[/THINK]
    # is dropped (bracket form, not the reasoning channel's <think>); an unclosed
    # [THINK] holds until [/THINK] so the cleaned text stays monotonic.
    text = _strip_mistral_reasoning(text)
    text = _strip_mistral_closed_calls(text)
    text = _strip_gemma_wrapperless_calls(text, enabled_tool_names)
    text = _strip_function_xml_calls(text, final = True)
    text = _strip_glm_calls(text, final = True)
    for pat in _TOOL_ALL_PATS:
        text = pat.sub("", text)
    return text


def _strip_tool_markup_final(
    text: str,
    *,
    auto_heal_tool_calls: bool,
    tool_protocol_active: bool = False,
    enabled_tool_names: Optional[set] = None,
) -> str:
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text
    return strip_tool_markup(text, final = True, enabled_tool_names = enabled_tool_names)


def _status_for_tool(tool_name: str, arguments: dict) -> str:
    """Return a human-readable status line matching the GGUF path."""
    return status_for_tool(tool_name, arguments)


def _looks_like_enabled_bare_json(text: str, enabled_tool_names: Optional[set]) -> bool:
    """True when ``text`` opens with an ENABLED markerless bare-JSON call; an ordinary JSON answer returns False."""
    probe = strip_llama3_leading_sentinels(text.lstrip())
    if not (probe.startswith("{") and ('"name"' in probe or '"function"' in probe)):
        return False
    return strip_leading_bare_json_call(probe, enabled_tool_names) != probe


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
    nudge_tool_calls: Optional[bool] = None,
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
    # Autoinject ran a KB search outside the controller, so it counts as an
    # executed tool for the plan-without-action gate.
    rag_autoinjected = bool(_auto)

    unrestricted_tools = not tools
    tool_controller = ToolLoopController(
        tools = None if unrestricted_tools else tools,
        auto_heal_tool_calls = auto_heal_tool_calls,
    )
    # RAG: cap knowledge-base searches per assistant turn (controller-agnostic).
    kb_search_count = 0
    final_attempt_done = False
    next_call_id = 0
    reprompt_count = 0
    # A denied tool confirmation must not be answered with a plan-without-action
    # re-prompt (which would raise the confirmation gate again).
    tool_denied = False
    # Real tool-call turns completed. Only turns that actually executed a tool count
    # against ``max_tool_iterations``; a duplicate/disabled no-op correction turn (and a
    # plan-without-action re-prompt) must not consume budget, matching the GGUF loop.
    _executed_tool_iters = 0

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

    # Reserve re-prompt slots so they don't eat the caller's tool budget.
    _extra_iters = MAX_ACT_REPROMPTS if max_tool_iterations > 0 else 0
    for iteration in range(max_tool_iterations + _extra_iters + 1):
        if cancel_event is not None and cancel_event.is_set():
            return
        # Whether this turn ran a tool; a no-op-only turn stays False and doesn't consume budget.
        _turn_executed_real_tool = False

        if final_attempt_done:
            active_tools: list[dict] = []
        else:
            active_tools = tool_controller.active_tools()
            if not active_tools and not unrestricted_tools:
                final_attempt_done = True
                active_tools = []

        tool_protocol_active = not final_attempt_done and (unrestricted_tools or bool(active_tools))
        tool_xml_signals = TOOL_XML_SIGNALS if tool_protocol_active else ()
        # Gate the markerless bare-JSON form on enabled names so an ordinary JSON answer isn't misread as a call.
        _enabled_tool_names = None if unrestricted_tools else set(_active_tool_names(active_tools))

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
                        enabled_tool_names = _enabled_tool_names,
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
                    enabled_tool_names = _enabled_tool_names,
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

            # Llama-3.2 ``custom_tools`` emits a bare ``{"name":..,"parameters":..}`` with no XML
            # signal. Hold a leading ``{`` (after any sentinel) until it closes: drain if it parses
            # as a call, else stream as content. Non-call text is always recovered downstream.
            bare_probe = strip_llama3_leading_sentinels(stripped)
            if (
                not is_match
                and not is_prefix
                and tool_protocol_active
                and bare_probe.startswith("{")
            ):
                if _balanced_brace_end(bare_probe, 0) is None:
                    if len(stripped) < _MAX_BARE_JSON_BUFFER:
                        continue  # object still open -- keep buffering
                    elif _looks_like_enabled_bare_json(bare_probe, _enabled_tool_names):
                        # Oversized still-open ENABLED-tool call: stop holding (memory bound) but
                        # DRAIN instead of leaking the raw prefix; a giant ordinary JSON answer still streams.
                        detect_state = _state_draining
                        continue
                elif parse_tool_calls_from_text(
                    content_buffer,
                    id_offset = next_call_id,
                    allow_incomplete = auto_heal_tool_calls,
                    enabled_tool_names = _enabled_tool_names,
                ):
                    # Closed object that parses as a bare-JSON call -- drain silently.
                    detect_state = _state_draining
                    continue
                # Closed non-call object (or oversized non-call) -- stream as text.

            # Gemma wrapper-less ``call:NAME{...}`` has no tool_xml_signals entry:
            # buffer it here or it streams raw until the end-of-turn safety net.
            # ``(?<!\w)`` keeps "recall:" out; the prefix regex is whitespace-tolerant.
            if (
                not is_match
                and not is_prefix
                and tool_protocol_active
                and (
                    "call:".startswith(stripped)
                    or _GEMMA_BARE_TC_PREFIX_RE.match(stripped) is not None
                    or _GEMMA_BARE_TC_RE.match(stripped) is not None
                )
            ):
                if _GEMMA_BARE_TC_RE.match(stripped):
                    detect_state = _state_draining
                    continue
                # A ``call:`` / ``call:partial_name`` prefix with no ``{`` yet: keep
                # buffering the variable-length name instead of leaking ``call:longname``.
                # Names can exceed 32 chars (OpenAI 64, MCP longer), so a fixed cap would
                # flush real calls raw. The prefix regex self-terminates on ordinary prose
                # and the ``{`` drains above; bound generously like the bare-JSON path.
                if _GEMMA_BARE_TC_PREFIX_RE.match(stripped) is not None:
                    if len(stripped) < _MAX_BARE_JSON_BUFFER:
                        continue
                    detect_state = _state_draining
                    continue
                if len(stripped) < _MAX_BUFFER_CHARS:
                    continue  # bare "call:" prefix still forming

            if is_match:
                # Tool signal -- flush any visible prefix before DRAINING
                # so the route sends it before tool_start.
                cumulative_display += content_buffer
                cleaned = strip_tool_markup_streaming(
                    cumulative_display,
                    auto_heal_tool_calls = auto_heal_tool_calls,
                    tool_protocol_active = tool_protocol_active,
                    enabled_tool_names = _enabled_tool_names,
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
                    enabled_tool_names = _enabled_tool_names,
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
            _bare_eos = strip_llama3_leading_sentinels(stripped)
            if (
                stripped
                and tool_protocol_active
                and any(sig in stripped for sig in tool_xml_signals)
            ):
                detect_state = _state_draining
            elif tool_protocol_active and _looks_like_enabled_bare_json(
                _bare_eos, _enabled_tool_names
            ):
                # A held bare-JSON ENABLED-tool fragment has no XML signal; DRAIN it (an ordinary
                # JSON answer falls through to the else and streams as content, GGUF parity).
                detect_state = _state_draining
            else:
                # Drain and fall through to STREAMING so the intent re-prompt + safety-net parser
                # still fire on short emissions like "Let me search." that never exit BUFFERING.
                if content_buffer:
                    cumulative_display += content_buffer
                    cleaned = strip_tool_markup(
                        cumulative_display, final = True, enabled_tool_names = _enabled_tool_names
                    )
                    if len(cleaned) > len(last_emitted):
                        last_emitted = cleaned
                        yield {"type": "content", "text": cleaned}
                detect_state = _state_streaming

        if detect_state == _state_streaming:
            # Run the parser even with no XML signal (the Llama-3.2 bare-JSON form carries none); it's
            # strict so plain answers stay untouched. Mirrors GGUF.
            safety_tc = parse_tool_calls_from_text(
                content_accum,
                id_offset = next_call_id,
                allow_incomplete = auto_heal_tool_calls,
                enabled_tool_names = _enabled_tool_names,
            )
            if not safety_tc:
                # Re-prompt once on plan-without-action, before any tool runs
                # (GGUF loop parity). The retry is gated on nudge_tool_calls so
                # Studio callers (which send True) always nudge, while API callers
                # who omit the flag keep today's no-reprompt behavior (opt-in).
                stripped_answer = content_accum.strip()
                if (
                    auto_heal_tool_calls
                    and nudge_tool_calls
                    and active_tools
                    and reprompt_count < MAX_ACT_REPROMPTS
                    and not rag_autoinjected
                    and not tool_denied
                    and not any(record.executed for record in tool_controller.history)
                    and is_short_intent_without_action(stripped_answer)
                ):
                    reprompt_count += 1
                    logger.info(
                        "Safetensors re-prompt %d/%d: model responded without "
                        "calling tools (%d chars)",
                        reprompt_count,
                        MAX_ACT_REPROMPTS,
                        len(stripped_answer),
                    )
                    conversation.append({"role": "assistant", "content": stripped_answer})
                    tool_hint = " or ".join(_active_tool_names(active_tools)) or "an available tool"
                    conversation.append(
                        {
                            "role": "user",
                            "content": reprompt_to_act_message(tool_hint),
                        }
                    )
                    # Empty status clears the badge and resets the route's
                    # per-turn text cursor before the re-prompted turn streams.
                    yield {"type": "status", "text": ""}
                    continue

                # Final answer. If a literal tool marker in prose was buffered but
                # never parsed as a call, restore the raw text so the prose surfaces
                # in full; route-level cleanup still applies the Auto-Heal policy.
                if content_accum and any(sig in content_accum for sig in tool_xml_signals):
                    yield {"type": "content", "text": content_accum}
                yield {"type": "status", "text": ""}
                return
            tool_calls = safety_tc
            content_text = _strip_tool_markup_final(
                content_accum,
                auto_heal_tool_calls = auto_heal_tool_calls,
                tool_protocol_active = True,
                enabled_tool_names = _enabled_tool_names,
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
                enabled_tool_names = _enabled_tool_names,
            )
            if not tool_calls:
                # Parser found nothing. Auto-Heal-enabled display cleanup
                # strips unparseable tool XML; disabled Auto-Heal preserves
                # the raw text so literal/malformed markup stays visible.
                if content_accum:
                    _drain_text = _strip_tool_markup_final(
                        content_accum,
                        auto_heal_tool_calls = auto_heal_tool_calls,
                        tool_protocol_active = False,
                        enabled_tool_names = _enabled_tool_names,
                    )
                    # Drained bare-JSON call that didn't parse: with Auto-Heal on, drop the fragment
                    # (plain JSON answers are left untouched); off keeps it visible per the strict contract.
                    if tool_protocol_active and auto_heal_tool_calls:
                        _drain_text = strip_leading_bare_json_call(_drain_text, _enabled_tool_names)
                    if _drain_text:
                        yield {"type": "content", "text": _drain_text}
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
                enabled_tool_names = _enabled_tool_names,
            )

        if tool_calls:
            next_call_id += len(tool_calls)
            # Strip a leading bare-JSON call from the kept content so it isn't replayed as text or
            # next-turn history (``_strip_tool_markup_final`` only knows XML). No-op for plain JSON answers.
            content_text = strip_leading_bare_json_call(content_text, _enabled_tool_names)

        if final_attempt_done:
            # Final-answer turn re-called a tool -- stop the loop.
            if content_text:
                yield {"type": "content", "text": content_text}
            yield {"type": "status", "text": ""}
            return

        # Collapse exact-duplicate calls and cap the count (runaway-turn guard).
        if tool_calls:
            seen_keys: set = set()
            deduped: list = []
            for _tc in tool_calls:
                _fn = _tc.get("function", {}) or {}
                _key = (_fn.get("name", ""), str(_fn.get("arguments", "")))
                if _key in seen_keys:
                    continue
                seen_keys.add(_key)
                deduped.append(_tc)
                if len(deduped) >= _MAX_TOOL_CALLS_PER_TURN:
                    break
            if len(deduped) != len(tool_calls):
                logger.info(
                    "Safetensors: collapsed %d repeated tool call(s) in one turn to %d",
                    len(tool_calls),
                    len(deduped),
                )
            tool_calls = deduped

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
                    tool_denied = True
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
            # A tool ran this turn, so it counts against the caller's budget.
            _turn_executed_real_tool = True
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
        # Count only turns that executed a tool against the cap; a no-op correction turn doesn't
        # consume budget so the model gets its nudge and another tool-enabled turn (GGUF parity).
        if _turn_executed_real_tool:
            _executed_tool_iters += 1
        if _executed_tool_iters >= max_tool_iterations and not final_attempt_done:
            # Budget exhausted; nudge a final plain answer.
            final_attempt_done = True
            conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})

    yield {"type": "status", "text": ""}
