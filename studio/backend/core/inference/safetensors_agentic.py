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
    _GEMMA_BARE_TC_RE,
    _TOOL_ALL_PATS,
    _balanced_brace_end,
    _strip_gemma_wrapperless_calls,
    _strip_mistral_closed_calls,
    BUDGET_EXHAUSTED_NUDGE,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    TOOL_XML_SIGNALS,
    parse_tool_calls_from_text,
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

# Hard cap for holding a leading bare-JSON object ({"name":..,"parameters":..})
# while we wait for it to close. Bounds memory if a top-level "{" never balances.
_MAX_BARE_JSON_BUFFER = 16384

# Forward-looking intent ("I'll", "First,", "Step 1:") => model is planning, not
# answering; used to nudge a tool call. Excludes "I can/should/want", "let's"
# (they also appear in plain answers). Mirrors GGUF.
_INTENT_SIGNAL = re.compile(
    r"(?i)("
    r"\b(i['’](ll|m going to|m gonna)|i am (going to|gonna)|i will|i shall|let me|allow me)\b"
    r"|\b(?:first\b|step \d+:?|here['’]?s (?:my |the |a )?(?:plan|approach))"
    r"|\b(?:now i|next i)\b"
    r")"
)
_MAX_REPROMPTS = 3
_REPROMPT_MAX_CHARS = 2000
# Templated so the nudge names the tools the caller actually enabled. Hardcoding
# web_search/python pushed the model toward calls that are rejected when only a
# subset (or custom/MCP tools) is active. Mirrors the GGUF path's tool_hint.
_REPROMPT_INSTRUCTION_TEMPLATE = (
    "STOP. Do NOT write code or explain. You MUST call a tool NOW. Call {tool_hint} immediately."
)

# Without a grammar constraint a small model can loop, emitting the same tool
# call dozens of times in one turn (llama-server's lazy grammar prevents this
# on the GGUF side). Collapse exact-duplicate calls within a turn and cap the
# number kept so one runaway turn cannot fan out into many tool executions.
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
) -> str:
    """Strip open-ended tool XML from display text without trimming whitespace."""
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text
    # Balanced strip first so nested Mistral / Gemma JSON ({"a":{"b":1}},
    # call:f{a:{b:1}}) is removed whole instead of the non-greedy pattern arms
    # truncating at the first ``}`` and leaving a trailing brace visible.
    text = _strip_mistral_closed_calls(text)
    text = _strip_gemma_wrapperless_calls(text)
    for pat in _TOOL_ALL_PATS:
        text = pat.sub("", text)
    return text


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


def _looks_like_enabled_bare_json(text: str, enabled_tool_names: Optional[set]) -> bool:
    """True when ``text`` opens with a markerless bare-JSON tool call whose name is an
    ENABLED tool (the gated strip would remove it). An ordinary JSON answer whose name
    is not a tool ({"name":"Alice",...}) returns False so it streams as content instead
    of being drained -- the GGUF loop applies the same gate. ``enabled_tool_names`` is
    ``None`` in unrestricted mode, where any bare ``{...,"name",...}`` is a potential
    call (matching the prior behaviour)."""
    probe = strip_llama3_leading_sentinels(text.lstrip())
    if not (probe.startswith("{") and '"name"' in probe):
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
    reprompt_count = 0

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
    _extra_iters = _MAX_REPROMPTS if max_tool_iterations > 0 else 0
    for iteration in range(max_tool_iterations + _extra_iters + 1):
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
        # Gate the markerless bare-JSON form on the enabled tool names so an ordinary
        # JSON answer ({"name":"Alice",...}) is not misread as a disabled-tool call
        # and dropped. ``None`` in unrestricted mode -- any name may be a tool there.
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

            # Llama-3.2 ``custom_tools`` emits a bare ``{"name":..,"parameters":..}``
            # object with no XML signal. Without this, the loop streams that raw JSON
            # to the client before the end-of-turn safety net recognises it as a call.
            # Strip any leading Llama sentinel (e.g. a prior turn's ``<|eot_id|>``)
            # first so a sentinel-prefixed object is held too. Hold a leading ``{``
            # until its top-level object closes: drain silently if it parses as a tool
            # call, else fall through and stream it as content (the DRAINING/STREAMING
            # resolvers both recover non-call text, so this can never drop a plain
            # JSON answer).
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
                        # Oversized but still-open bare-JSON call for an ENABLED tool:
                        # stop holding (memory bound) yet DRAIN rather than leak the raw
                        # JSON prefix. The safety net recovers a complete call; the
                        # DRAINING resolver drops a truncated one. Gated on the enabled
                        # tool name so a giant ordinary JSON answer
                        # ({"name":"Alice",...}) still streams (GGUF parity).
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

            # Gemma 4 wrapper-less ``call:NAME{...}`` (special tokens stripped) has no
            # entry in tool_xml_signals, so without this it streams raw and is only
            # caught by the end-of-turn safety net. The ``(?<!\w)`` guard in
            # _GEMMA_BARE_TC_RE keeps words like "recall:" out; the safety net still
            # recovers the text if it turns out not to be a call.
            if (
                not is_match
                and not is_prefix
                and tool_protocol_active
                and ("call:".startswith(stripped) or stripped.startswith("call:"))
            ):
                if _GEMMA_BARE_TC_RE.match(stripped):
                    detect_state = _state_draining
                    continue
                if len(stripped) < _MAX_BUFFER_CHARS:
                    continue  # "call:" prefix / "call:partialname" -- keep buffering

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
                # A held bare-JSON ENABLED-tool call fragment carries no XML signal.
                # DRAIN it so a complete object parses/executes and a truncated one is
                # dropped by the DRAINING resolver. Gated on the enabled tool name so
                # an ordinary JSON answer whose name is not a tool falls through to the
                # else below and is streamed as content (GGUF parity).
                detect_state = _state_draining
            else:
                # Drain the buffer and fall through to STREAMING so the intent
                # re-prompt + safety-net parser still fire on short emissions like
                # "Let me search." that never exit BUFFERING (else the loop ends).
                if content_buffer:
                    cumulative_display += content_buffer
                    cleaned = strip_tool_markup(cumulative_display, final = True)
                    if len(cleaned) > len(last_emitted):
                        last_emitted = cleaned
                        yield {"type": "content", "text": cleaned}
                detect_state = _state_streaming

        if detect_state == _state_streaming:
            # Run the parser even with no XML signal: the Llama-3.2 bare-JSON form
            # ``{"name":..,"parameters":..}`` carries none, so gating on
            # has_tool_signal() dropped real calls. parse_tool_calls_from_text is
            # strict (fires only on a valid shape), so plain answers stay untouched.
            # Mirrors GGUF.
            safety_tc = parse_tool_calls_from_text(
                content_accum,
                id_offset = next_call_id,
                allow_incomplete = auto_heal_tool_calls,
                enabled_tool_names = _enabled_tool_names,
            )
            if not safety_tc:
                # Re-prompt only when the model planned without acting (intent
                # signal); "4" / "Hello!" never trigger. Mirrors GGUF.
                _stripped = content_accum.strip()
                if (
                    tools
                    and auto_heal_tool_calls
                    and reprompt_count < _MAX_REPROMPTS
                    and 0 < len(_stripped) < _REPROMPT_MAX_CHARS
                    and _INTENT_SIGNAL.search(_stripped)
                    and not final_attempt_done
                ):
                    reprompt_count += 1
                    logger.info(
                        "Safetensors re-prompt %d/%d: model planned without "
                        "calling tools (%d chars)",
                        reprompt_count,
                        _MAX_REPROMPTS,
                        len(_stripped),
                    )
                    tool_hint = " or ".join(_active_tool_names(active_tools)) or "an available tool"
                    conversation.append({"role": "assistant", "content": _stripped})
                    conversation.append(
                        {
                            "role": "user",
                            "content": _REPROMPT_INSTRUCTION_TEMPLATE.format(tool_hint = tool_hint),
                        }
                    )
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
                    )
                    # A truncated/oversized bare-JSON tool call (``{"name":..``)
                    # was drained here but did not parse. Drop it rather than
                    # leaking the raw fragment; a plain JSON answer (no ``"name"``)
                    # is left untouched by the helper.
                    if tool_protocol_active:
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
            )

        if tool_calls:
            next_call_id += len(tool_calls)
            # Strip a leading bare-JSON tool call from the content kept for the
            # assistant turn so the executed call is not replayed as visible text
            # or fed back as next-turn history. ``_strip_tool_markup_final`` only
            # knows XML/bracket markup; this also covers the Llama-3.2 bare-JSON
            # form. No-op for plain JSON answers (no ``"name"`` key).
            content_text = strip_leading_bare_json_call(content_text, _enabled_tool_names)

        if final_attempt_done:
            # Final-answer turn re-called a tool -- stop the loop.
            if content_text:
                yield {"type": "content", "text": content_text}
            yield {"type": "status", "text": ""}
            return

        # Collapse exact-duplicate tool calls emitted in a single turn (a
        # runaway model can repeat one call many times) and cap the count, so
        # one turn cannot fan out into dozens of identical executions/events.
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
        # Track against the caller-requested cap, excluding re-prompt
        # slots so a stalling model still gets a final-answer attempt.
        _tool_iters_done = iteration + 1 - reprompt_count
        if _tool_iters_done >= max_tool_iterations and not final_attempt_done:
            # Budget exhausted; nudge a final plain answer.
            final_attempt_done = True
            conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})

    yield {"type": "status", "text": ""}
