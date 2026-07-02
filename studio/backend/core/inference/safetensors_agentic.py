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

import bisect
import re
import threading
from typing import Callable, Generator, Optional

from loggers import get_logger

from core.inference.tool_call_parser import (
    _TOOL_ALL_PATS,
    BUDGET_EXHAUSTED_NUDGE,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    TOOL_XML_SIGNALS,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from core.tool_healing import (
    _TOOL_CLOSED_PATS,
    _strip_bracket_tag_calls,
    _think_spans_outside_tool_markup,
    apply_tool_strip_patterns,
    strip_outside_think,
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


def _active_tool_names(active_tools: list[dict]) -> list[str]:
    names = [
        (tool.get("function") or {}).get("name")
        for tool in active_tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    return [name for name in names if name]


# Unrestricted mode ships no tool list, so any bare identifier may be the NAME of
# a ``NAME[ARGS]`` rehearsal (optionally part-way into ``[ARGS``); the complete
# ``NAME[ARGS]`` is caught as a match before this prefix check runs.
_UNRESTRICTED_REHEARSAL_RE = re.compile(r"[\w-]+(?:\[A(?:R(?:G(?:S)?)?)?)?")


def _is_rehearsal_prefix(
    stripped: str,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> bool:
    """True if ``stripped`` is a (possibly partial) prefix of a ``NAME[ARGS]``
    rehearsal split across chunks (``web_search`` then ``[ARGS]{...}``). A space
    means prose. Unrestricted mode accepts any identifier; else NAME must be active."""
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    if unrestricted:
        return _UNRESTRICTED_REHEARSAL_RE.fullmatch(stripped) is not None
    for name in _active_tool_names(active_tools):
        if stripped == name or f"{name}[ARGS]".startswith(stripped):
            return True
    return False


def _held_rehearsal_tail_len(
    text: str,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> int:
    """Length of a trailing bare tool-name token that may be a split rehearsal call
    (``...web_search`` with ``[ARGS]{...}`` still to arrive), so STREAMING can hold it
    instead of leaking the name. Returns 0 for ordinary prose."""
    i = len(text)
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    tail = text[i:]
    return (
        len(tail)
        if tail and _is_rehearsal_prefix(tail, active_tools, unrestricted = unrestricted)
        else 0
    )


def _rehearsal_name_start(
    candidate: str,
    signal_pos: int,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> int:
    """For an ``[ARGS]`` signal at ``signal_pos``, return the start of the preceding
    bare tool-name token (``NAME[ARGS]``), else ``signal_pos`` unchanged when the
    signal is not ``[ARGS]`` or NAME is not an active tool (restricted mode)."""
    if not candidate.startswith("[ARGS]", signal_pos):
        return signal_pos
    j = signal_pos
    while j > 0 and (candidate[j - 1].isalnum() or candidate[j - 1] in "_-"):
        j -= 1
    if j < signal_pos and (
        unrestricted or candidate[j:signal_pos] in _active_tool_names(active_tools)
    ):
        return j
    return signal_pos


def _earliest_tool_signal(
    candidate: str,
    signals,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> int:
    """Index where the turn's first genuine tool-call boundary begins, or -1.

    Non-``[ARGS]`` markup wins on first occurrence. An ``[ARGS]`` hit is a rehearsal
    only when an active tool name (any name in unrestricted mode) precedes it, so a
    literal ``foo[ARGS]`` in prose is skipped rather than draining the turn; for a
    real ``NAME[ARGS]`` the boundary is pulled back to NAME."""
    best = -1
    for sig in signals:
        if sig != "[ARGS]":
            p = candidate.find(sig)
            if p >= 0 and (best < 0 or p < best):
                best = p
            continue
        from_idx = 0
        while True:
            p = candidate.find("[ARGS]", from_idx)
            if p < 0:
                break
            name_start = _rehearsal_name_start(
                candidate, p, active_tools, unrestricted = unrestricted
            )
            if name_start < p:
                # Genuine ``NAME[ARGS]``: the boundary is the start of NAME.
                if best < 0 or name_start < best:
                    best = name_start
                break
            # Bare/prose ``[ARGS]`` (no active name in front): skip it and keep
            # looking so a later real call in the same chunk is still detected.
            from_idx = p + len("[ARGS]")
    return best


def _has_genuine_tool_signal(
    candidate: str,
    signals,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> bool:
    """True when ``candidate`` holds a genuine tool-call boundary for one of ``signals``.

    Non-``[ARGS]`` markers count on a substring hit; an ``[ARGS]`` hit is genuine only
    when an active tool name (any in unrestricted mode) precedes it. Mirrors the
    ``_earliest_tool_signal`` name-gating so BUFFERING / end-of-stream checks do not
    drain inactive-name prose."""
    for sig in signals:
        if sig == "[ARGS]":
            if (
                _earliest_tool_signal(
                    candidate, ("[ARGS]",), active_tools, unrestricted = unrestricted
                )
                >= 0
            ):
                return True
            continue
        if sig in candidate:
            return True
    return False


def strip_tool_markup_streaming(
    text: str,
    *,
    auto_heal_tool_calls: bool = True,
    tool_protocol_active: bool = False,
    enabled_tool_names = None,
) -> str:
    """Strip open-ended tool XML from display text without trimming whitespace.

    ``enabled_tool_names`` keeps an inactive-name ``foo[ARGS]{..}`` example visible (it
    is prose, not a call), matching the parse / detection active-tool gate."""
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text

    def _seg(segment: str, is_last: bool) -> str:
        # Balanced-brace strip first (any JSON nesting depth), then XML regexes. The
        # open-ended tail arms in _TOOL_ALL_PATS are end-of-text anchored, so run them
        # only on the last segment: a bare ``foo[ARGS]`` before <think> is prose, not a
        # truncated call. Rehearsal strip is name-gated, keeping inactive-name examples.
        segment = _strip_bracket_tag_calls(segment, enabled_tool_names = enabled_tool_names)
        patterns = _TOOL_ALL_PATS if is_last else _TOOL_CLOSED_PATS
        return apply_tool_strip_patterns(segment, patterns, enabled_tool_names = enabled_tool_names)

    # Preserve <think>/[THINK] verbatim: stripping a call rehearsed inside a block
    # would emit a shorter cumulative text that grows back once the block closes,
    # corrupting append-by-length consumers and the visible reasoning.
    return strip_outside_think(text, _seg)


def _strip_tool_markup_final(
    text: str,
    *,
    auto_heal_tool_calls: bool,
    tool_protocol_active: bool = False,
    enabled_tool_names = None,
) -> str:
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text
    return strip_tool_markup(text, final = True, enabled_tool_names = enabled_tool_names)


def _status_for_tool(tool_name: str, arguments: dict) -> str:
    """Return a human-readable status line matching the GGUF path."""
    return status_for_tool(tool_name, arguments)


_FUNCTION_SIGNAL_RE = re.compile(r"<function=([\w-]+)>")
_TOOL_CALL_NAME_RE = re.compile(r'"name"\s*:\s*"([\w-]+)"')
# Mistral ``[TOOL_CALLS]name{json}`` / ``[TOOL_CALLS]name[CALL_ID]id[ARGS]{json}`` and
# rehearsal ``name[ARGS]{json}``, aligned with the parser regexes so the provisional
# render-html card fires for the bracket-tag serializations too, not only XML.
_MISTRAL_RENDER_NAME_RE = re.compile(
    r"\[TOOL_CALLS\]\s*([\w-]+)(?:\[CALL_ID\][\w-]+)?(?:\[ARGS\])?\s*(?=\{)"
)
_REHEARSAL_RENDER_NAME_RE = re.compile(r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]\s*(?=\{)")


def _detect_render_html_tool_start(content: str) -> bool:
    """Return True when the FIRST tool call in ``content`` is clearly render_html.

    Covers every serialization the loop executes (XML ``<function=>`` / ``<tool_call>``,
    Mistral ``[TOOL_CALLS]``, rehearsal ``NAME[ARGS]``); the earliest marker wins so a
    render_html marker inside another call's argument is treated as data. Markers inside
    a ``<think>`` / ``[THINK]`` block are dropped since the parser skips them."""
    think_spans = _think_spans_outside_tool_markup(content)
    _think_starts = [s for s, _e in think_spans]

    def _in_think(pos: int) -> bool:
        if not think_spans:
            return False
        i = bisect.bisect_right(_think_starts, pos) - 1
        return i >= 0 and think_spans[i][0] <= pos < think_spans[i][1]

    def _first_outside(start: int, finder) -> int:
        # First occurrence at/after ``start`` that is not inside a think span.
        pos = finder(start)
        while pos >= 0 and _in_think(pos):
            pos = finder(pos + 1)
        return pos

    candidates: list[tuple[int, str]] = []
    for fm in _FUNCTION_SIGNAL_RE.finditer(content):
        if not _in_think(fm.start()):
            candidates.append((fm.start(), fm.group(1)))
            break
    tc = _first_outside(0, lambda i: content.find("<tool_call>", i))
    if tc >= 0:
        nm = _TOOL_CALL_NAME_RE.search(content[tc:])
        candidates.append((tc, nm.group(1) if nm else ""))
    mt = _first_outside(0, lambda i: content.find("[TOOL_CALLS]", i))
    if mt >= 0:
        mm = _MISTRAL_RENDER_NAME_RE.match(content, mt)
        if mm:
            candidates.append((mt, mm.group(1)))
        else:
            # Array / single-object shape: ``[TOOL_CALLS] [{"name":...}]``. A bare
            # ``"name"`` search can latch onto an argument key before the real tool
            # name (``[{"arguments":{"name":"render_html"},"name":"python"}]``), so
            # resolve the first call through the parser, which reads top-level names.
            arr_calls = parse_tool_calls_from_text(content[mt:])
            if arr_calls:
                candidates.append((mt, (arr_calls[0].get("function") or {}).get("name") or ""))
    for rm in _REHEARSAL_RENDER_NAME_RE.finditer(content):
        if not _in_think(rm.start(1)):
            candidates.append((rm.start(1), rm.group(1)))
            break

    if not candidates:
        return False
    _pos, name = min(candidates, key = lambda c: c[0])
    return name == "render_html"


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
    # Gate that tells a genuine ``NAME[ARGS]`` rehearsal from an inactive-name example
    # in prose, keeping parse / display strip symmetric with the streaming guard.
    # ``None`` in unrestricted mode keeps the any-name behaviour. Built from the ORIGINAL
    # tools list so a spent one-shot tool still reads as a tool name, not prose.
    _enabled_names_gate = None if unrestricted_tools else set(_active_tool_names(tools))
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
                # Earliest genuine boundary: a bare ``[ARGS]`` in prose is skipped, and a
                # real ``NAME[ARGS]`` is pulled back to NAME so the name is not flushed.
                signal_pos = _earliest_tool_signal(
                    candidate, tool_xml_signals, active_tools, unrestricted = unrestricted_tools
                )
                if signal_pos >= 0:
                    before_tool = candidate[:signal_pos]
                    cleaned_before = strip_tool_markup_streaming(
                        before_tool,
                        auto_heal_tool_calls = auto_heal_tool_calls,
                        tool_protocol_active = tool_protocol_active,
                        enabled_tool_names = _enabled_names_gate,
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
                    enabled_tool_names = _enabled_names_gate,
                )
                # Hold a trailing bare active-tool-name (split rehearsal NAME whose
                # [ARGS] has not arrived) so it is not streamed before the call drains;
                # released once more prose follows, or by the end-of-stream flush.
                if tool_protocol_active:
                    _hold = _held_rehearsal_tail_len(
                        cleaned, active_tools, unrestricted = unrestricted_tools
                    )
                    emit = cleaned[: len(cleaned) - _hold] if _hold else cleaned
                else:
                    emit = cleaned
                if len(emit) > len(last_emitted):
                    last_emitted = emit
                    yield {"type": "content", "text": emit}
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
                # Bracket-tag forms arrive mid-buffer (``name[ARGS]{...}``,
                # ``[TOOL_CALLS]...``), so also do a substring check (mirrors the GGUF
                # loop). ``[ARGS]`` counts only as a ``NAME[ARGS]`` rehearsal with an
                # active NAME; a bare/inactive-name ``foo[ARGS]`` in prose is gated out
                # like the STREAMING state, so prose is not drained into a no-op.
                if sig == "[ARGS]":
                    if (
                        _earliest_tool_signal(
                            stripped,
                            ("[ARGS]",),
                            active_tools,
                            unrestricted = unrestricted_tools,
                        )
                        >= 0
                    ):
                        is_match = True
                        break
                elif sig.startswith("[") and sig in stripped:
                    is_match = True
                    break

            # Rehearsal split across chunks (``web_search`` then ``[ARGS]{...}``): hold
            # the bare name prefix until the ``[ARGS]`` arm arrives and matches above.
            is_rehearsal_prefix = False
            if (
                not is_match
                and not is_prefix
                and tool_protocol_active
                and _is_rehearsal_prefix(stripped, active_tools, unrestricted = unrestricted_tools)
            ):
                is_prefix = True
                is_rehearsal_prefix = True

            if is_match:
                # Tool signal -- flush any visible prefix before DRAINING
                # so the route sends it before tool_start.
                cumulative_display += content_buffer
                cleaned = strip_tool_markup_streaming(
                    cumulative_display,
                    auto_heal_tool_calls = auto_heal_tool_calls,
                    tool_protocol_active = tool_protocol_active,
                    enabled_tool_names = _enabled_names_gate,
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
            elif is_prefix and (is_rehearsal_prefix or len(stripped) < _MAX_BUFFER_CHARS):
                # A rehearsal prefix is self-bounded (stops matching past ``NAME[ARGS]``),
                # so the _MAX_BUFFER_CHARS cap must not cut it short for MCP names >31 chars.
                continue
            else:
                detect_state = _state_streaming
                cumulative_display += content_buffer
                cleaned = strip_tool_markup_streaming(
                    cumulative_display,
                    auto_heal_tool_calls = auto_heal_tool_calls,
                    tool_protocol_active = tool_protocol_active,
                    enabled_tool_names = _enabled_names_gate,
                )
                # Same trailing-name hold as STREAMING: this first flush out of BUFFERING
                # must not emit a bare name whose ``[ARGS]`` arrives next chunk.
                if tool_protocol_active:
                    _hold = _held_rehearsal_tail_len(
                        cleaned, active_tools, unrestricted = unrestricted_tools
                    )
                    emit = cleaned[: len(cleaned) - _hold] if _hold else cleaned
                else:
                    emit = cleaned
                if len(emit) > len(last_emitted):
                    last_emitted = emit
                    yield {"type": "content", "text": emit}

        # Stream finished -- resolve what we collected.
        if cancel_event is not None and cancel_event.is_set():
            return

        if detect_state == _state_buffering:
            # Buffer never resolved: gate ``[ARGS]`` on an active tool name so a
            # whole-turn prose answer with a literal ``foo[ARGS]{...}`` is not parsed.
            stripped = content_buffer.lstrip()
            if (
                stripped
                and tool_protocol_active
                and _has_genuine_tool_signal(
                    stripped,
                    tool_xml_signals,
                    active_tools,
                    unrestricted = unrestricted_tools,
                )
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
                            enabled_tool_names = _enabled_names_gate,
                        ),
                    }
                yield {"type": "status", "text": ""}
                return

        if detect_state == _state_streaming:
            # No tool mid-stream: check for late tool XML. ``[ARGS]`` is name-gated so an
            # inactive-name ``foo[ARGS]`` in prose is not parsed into a no-op extra turn.
            safety_tc = None
            saw_tool_signal = tool_protocol_active and _has_genuine_tool_signal(
                content_accum,
                tool_xml_signals,
                active_tools,
                unrestricted = unrestricted_tools,
            )
            if saw_tool_signal:
                safety_tc = parse_tool_calls_from_text(
                    content_accum,
                    id_offset = next_call_id,
                    allow_incomplete = auto_heal_tool_calls,
                    enabled_tool_names = _enabled_names_gate,
                )
            if not safety_tc:
                # Final answer: if a literal tool marker in prose was stripped
                # during streaming but did not parse as a real call, restore the
                # raw cumulative text for core callers. Route-level cleanup can
                # still apply the Auto-Heal display policy.
                if saw_tool_signal and content_accum:
                    yield {"type": "content", "text": content_accum}
                else:
                    # Release any rehearsal tail held during streaming: the turn ended
                    # as a plain answer (no ``[ARGS]`` followed), so the held token is
                    # real prose and must not be dropped.
                    final_clean = strip_tool_markup_streaming(
                        cumulative_display,
                        auto_heal_tool_calls = auto_heal_tool_calls,
                        tool_protocol_active = tool_protocol_active,
                        enabled_tool_names = _enabled_names_gate,
                    )
                    if len(final_clean) > len(last_emitted):
                        yield {"type": "content", "text": final_clean}
                yield {"type": "status", "text": ""}
                return
            tool_calls = safety_tc
            content_text = _strip_tool_markup_final(
                content_accum,
                auto_heal_tool_calls = auto_heal_tool_calls,
                tool_protocol_active = True,
                enabled_tool_names = _enabled_names_gate,
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
                enabled_tool_names = _enabled_names_gate,
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
                            enabled_tool_names = _enabled_names_gate,
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
                enabled_tool_names = _enabled_names_gate,
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
