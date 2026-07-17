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
import inspect
import re
import threading
from typing import Callable, Generator, Optional

from loggers import get_logger

from core.inference.tool_call_parser import (
    _GEMMA_BARE_TC_PREFIX_RE,
    _GEMMA_BARE_TC_RE,
    _TOOL_ALL_PATS as _PARSER_TOOL_ALL_PATS,
    _TOOL_CLOSED_PATS as _PARSER_TOOL_CLOSED_PATS,
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

# The healer owns the bracket-tag + rehearsal strip helpers and their name-gated
# pattern lists, so the safetensors streaming strip stays aligned with the parser.
from core.tool_healing import (
    _REHEARSAL_TAIL_STRIP_RE,
    _strip_bracket_tag_calls,
    _think_spans_outside_tool_markup,
    apply_tool_strip_patterns,
    strip_outside_think,
)
from core.inference.tool_loop_controller import (
    ToolLoopController,
    append_deferred_nudges,
    coerce_tool_arguments,
    status_for_tool,
    tool_event_provenance,
)
from core.inference.tool_stream_exec import stream_tool_execution
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


# Unrestricted mode has no tool list, so any identifier may open a NAME[ARGS] rehearsal;
# ``[`` and each ARGS letter stay optional so a chunk split after ``NAME[`` is still held.
_UNRESTRICTED_REHEARSAL_RE = re.compile(r"[\w-]+(?:\[(?:A(?:R(?:G(?:S)?)?)?)?)?")


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
            # Bare/prose [ARGS]: skip it so a later real call in the same chunk is still found.
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
    enabled_tool_names: Optional[set] = None,
) -> str:
    """Strip open-ended tool XML from display text without trimming whitespace.

    Mirrors the parser-side ``strip_tool_markup`` segment scan (minus the final trim) so
    streaming and final display agree: balanced strips first (nested JSON removed whole),
    then the guarded function-XML / GLM scans that close at each call's REAL terminator so
    literal markup inside argument values is data and trailing prose survives. Reasoning
    ``<think>`` / ``[THINK]`` blocks are preserved verbatim (a rehearsed call inside one must
    not be deleted, else the cumulative text shrinks then regrows). ``enabled_tool_names``
    keeps an inactive-name ``foo[ARGS]{..}`` / ``call:NAME{..}`` example visible (it is prose,
    not a call), matching the parse / detection active-tool gate."""
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text

    # Drop a leading Magistral ``[THINK]...[/THINK]`` block (bracket reasoning form, not the
    # ``<think>`` channel) so raw reasoning does not leak into streamed display; an unclosed
    # leading block is held (dropped to EOF) until its closer streams in.
    text = _strip_mistral_reasoning(text)

    def _seg(segment: str, is_last: bool) -> str:
        # Same scan order as the parser's _strip_segment (seg_final -> is_last): balanced
        # strips first, then the guarded function-XML / GLM scans, then the regex arms
        # (DeepSeek / Kimi / closed forms). EOS-anchored tail arms run only on the last
        # segment (a bare ``foo[ARGS]`` before <think> is prose). Rehearsal strips are name-gated.
        seg = _strip_mistral_closed_calls(segment)
        seg = _strip_bracket_tag_calls(seg, enabled_tool_names = enabled_tool_names)
        if is_last:
            seg = _strip_gemma_wrapperless_calls(seg, enabled_tool_names)
        seg = _strip_function_xml_calls(seg, final = is_last)
        seg = _strip_glm_calls(seg, final = is_last)
        pats = _PARSER_TOOL_ALL_PATS if is_last else _PARSER_TOOL_CLOSED_PATS
        for pat in pats:
            seg = pat.sub("", seg)
        if is_last:
            seg = apply_tool_strip_patterns(
                seg, [_REHEARSAL_TAIL_STRIP_RE], enabled_tool_names = enabled_tool_names
            )
        return seg

    # Preserve think blocks verbatim: stripping a rehearsed call inside one shrinks then
    # regrows the cumulative text, corrupting append-by-length consumers.
    return strip_outside_think(text, _seg)


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
# Mistral name/v11 and rehearsal forms, aligned with the parser so the provisional
# render-html card fires for bracket-tag serializations too.
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
            # Array shape: a bare ``"name"`` search can latch onto an argument key, so resolve the
            # first call through the parser (it reads top-level names).
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


def _accepts_output_callback(func: Callable[..., str]) -> bool:
    """Whether an injectable ``execute_tool`` supports ``output_callback``.

    The loop's ``execute_tool`` is a parameter (tests inject fakes), so forward
    the live-output kwarg only when the callable declares it or takes ``**kwargs``.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "output_callback" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


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
    thread_id: Optional[str] = None,
    rag_scope: Optional[dict] = None,
    confirm_tool_calls: bool = False,
    bypass_permissions: bool = False,
    permission_mode: Optional[str] = None,
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

    # Normalize the mode (mirrors the GGUF loop): "full" and
    # bypass_permissions are the same switch; unset/unknown behaves as "ask".
    # "off" keeps the sandbox but never prompts.
    if permission_mode == "full":
        bypass_permissions = True
    elif bypass_permissions:
        permission_mode = "full"
    elif permission_mode not in ("ask", "auto", "off"):
        permission_mode = "ask"

    # Forced first-pass RAG (mirrors the GGUF loop) so doc Qs don't lose to
    # web_search. Skip only when a retrieval call would actually prompt (ask
    # mode); auto never gates the safe search_knowledge_base tool.
    from core.inference.tools import build_rag_autoinject

    # off never prompts, so (like auto) it must not lose first-pass retrieval
    # even if a direct caller passes a stale confirm_tool_calls flag.
    _skip_autoinject = (
        confirm_tool_calls and not bypass_permissions and permission_mode not in ("auto", "off")
    )
    _auto = None if _skip_autoinject else build_rag_autoinject(conversation, rag_scope)
    if _auto:
        for _ev in _auto["events"]:
            yield _ev
        conversation.extend(_auto["messages"])
    # Autoinject ran a KB search outside the controller, so it counts as an
    # executed tool for the plan-without-action gate.
    rag_autoinjected = bool(_auto)

    unrestricted_tools = not tools
    # Gate telling a genuine NAME[ARGS] rehearsal from inactive-name prose; built from the
    # ORIGINAL tools list so a spent one-shot still reads as a tool name. None = unrestricted.
    _enabled_names_gate = None if unrestricted_tools else set(_active_tool_names(tools))
    # Detection must see the same names as the strip gate (ORIGINAL list, incl. a spent
    # one-shot), else its repeat is stripped but never drained and the turn ends blank.
    _detect_tools = [] if unrestricted_tools else list(tools or [])
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
        # Live-args offset for the provisional render_html card: the drained call
        # text streams as tool_args so the canvas shows the HTML being written.
        _live_args_streamed_upto = -1
        # When a human confirmation gate is active the real tool_start is keyed
        # by an approval id and carries awaiting_confirmation, so an early
        # provisional card (keyed by tool_call_id, no approval) would show the
        # tool as "running" before the user has approved it. Suppress the early
        # card in that case and let the gated tool_start be the first signal.
        # In auto mode render_html is always safe and never prompts, so keep its
        # early canvas card (the frontend sends confirm_tool_calls=true alongside
        # auto); mirrors the GGUF path's _confirm_gated exemption.
        from core.inference.tools import is_always_safe_tool

        _provisional_confirm_gated = (
            bool(confirm_tool_calls)
            and not bypass_permissions
            and not (permission_mode == "auto" and is_always_safe_tool("render_html"))
        )

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
                    # Backlog first: everything drained so far.
                    yield {
                        "type": "tool_args",
                        "tool_call_id": provisional_render_html_id,
                        "tool_name": "render_html",
                        "text": content_accum,
                    }
                    _live_args_streamed_upto = len(content_accum)
                elif (
                    provisional_render_html_started
                    and not provisional_resolved
                    and _live_args_streamed_upto >= 0
                    and len(content_accum) > _live_args_streamed_upto
                ):
                    # Still writing the call: stream the fragment so the canvas
                    # renders live. Display only; content_accum still feeds the
                    # stream-end parser verbatim.
                    yield {
                        "type": "tool_args",
                        "tool_call_id": provisional_render_html_id,
                        "tool_name": "render_html",
                        "text": content_accum[_live_args_streamed_upto:],
                    }
                    _live_args_streamed_upto = len(content_accum)
                continue

            if detect_state == _state_streaming:
                candidate = cumulative_display + delta
                # Earliest genuine boundary: bare [ARGS] in prose is skipped; a real NAME[ARGS] is
                # pulled back to NAME so the name is not flushed.
                signal_pos = _earliest_tool_signal(
                    candidate, tool_xml_signals, _detect_tools, unrestricted = unrestricted_tools
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
                        yield {
                            "type": "tool_args",
                            "tool_call_id": provisional_render_html_id,
                            "tool_name": "render_html",
                            "text": content_accum,
                        }
                        _live_args_streamed_upto = len(content_accum)
                    continue
                cumulative_display = candidate
                cleaned = strip_tool_markup_streaming(
                    cumulative_display,
                    auto_heal_tool_calls = auto_heal_tool_calls,
                    tool_protocol_active = tool_protocol_active,
                    enabled_tool_names = _enabled_names_gate,
                )
                # Hold a trailing bare active-tool-name (split rehearsal) until its [ARGS] arrives;
                # released by later prose or the end-of-stream flush.
                if tool_protocol_active:
                    _hold = _held_rehearsal_tail_len(
                        cleaned, _detect_tools, unrestricted = unrestricted_tools
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
                # Bracket-tag forms arrive mid-buffer, so substring-check too (mirrors GGUF); [ARGS]
                # counts only with an active NAME so prose is not drained into a no-op.
                if sig == "[ARGS]":
                    if (
                        _earliest_tool_signal(
                            stripped,
                            ("[ARGS]",),
                            _detect_tools,
                            unrestricted = unrestricted_tools,
                        )
                        >= 0
                    ):
                        is_match = True
                        break
                elif sig.startswith("[") and sig in stripped:
                    is_match = True
                    break

            # Split rehearsal: hold the bare name until its [ARGS] arrives and matches above.
            is_rehearsal_prefix = False
            if (
                not is_match
                and not is_prefix
                and tool_protocol_active
                and _is_rehearsal_prefix(stripped, _detect_tools, unrestricted = unrestricted_tools)
            ):
                is_prefix = True
                is_rehearsal_prefix = True

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
                    yield {
                        "type": "tool_args",
                        "tool_call_id": provisional_render_html_id,
                        "tool_name": "render_html",
                        "text": content_accum,
                    }
                    _live_args_streamed_upto = len(content_accum)
            elif is_prefix and (is_rehearsal_prefix or len(stripped) < _MAX_BUFFER_CHARS):
                # A rehearsal prefix is self-bounded; the buffer cap must not cut long MCP names short.
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
                # Same trailing-name hold as STREAMING for this first flush out of BUFFERING.
                if tool_protocol_active:
                    _hold = _held_rehearsal_tail_len(
                        cleaned, _detect_tools, unrestricted = unrestricted_tools
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
            # Buffer never resolved: [ARGS] is name-gated so a prose answer with a literal
            # ``foo[ARGS]{...}`` is not parsed.
            stripped = content_buffer.lstrip()
            _bare_eos = strip_llama3_leading_sentinels(stripped)
            if (
                stripped
                and tool_protocol_active
                and _has_genuine_tool_signal(
                    stripped,
                    tool_xml_signals,
                    _detect_tools,
                    unrestricted = unrestricted_tools,
                )
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
                else:
                    # Turn ended as a plain answer (no [ARGS] followed): the held rehearsal tail is real
                    # prose, release it.
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
            # DRAINING: parse tool calls out of full content. Gate the bare rehearsal on the
            # ORIGINAL tool list (``_enabled_names_gate``), the same names detection/strip used to
            # drain here: a spent one-shot (render_html) is off the active list but its re-emitted
            # ``render_html[ARGS]{..}`` must still parse so it routes to the repeat no-op instead of
            # being dropped into a blank continuation.
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
                enabled_tool_names = _enabled_names_gate,
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
        # Collect no-op nudges and flush them after the batch, so a no-op doesn't
        # abort it and drop the parallel calls that follow.
        deferred_noop_msgs: list = []

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
                deferred_noop_msgs.append(completion.model_message())
                logger.info(
                    "Suppressed local safetensors tool call as internal no-op: "
                    f"action={decision.action} tool={decision.tool_name}"
                )
                continue

            if not assistant_appended:
                assistant_msg["tool_calls"] = [decision.as_assistant_tool_call()]
                conversation.append(assistant_msg)
                assistant_appended = True
            else:
                assistant_msg.setdefault("tool_calls", []).append(decision.as_assistant_tool_call())

            # Bypass wins over the confirm gate at the loop level too, so a
            # direct internal caller passing both flags never prompts. In
            # "auto" mode only calls detected as potentially unsafe pause.
            # "off" never prompts (sandbox stays on).
            needs_confirm = (
                bool(confirm_tool_calls) and not bypass_permissions and permission_mode != "off"
            )
            if needs_confirm and permission_mode == "auto":
                from core.inference.tools import is_potentially_unsafe_tool_call
                needs_confirm = is_potentially_unsafe_tool_call(
                    decision.tool_name, decision.arguments
                )
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
                # Execute in a worker thread so live stdout chunks and heartbeats
                # stream while the tool blocks (the SSE route turns heartbeats into
                # keepalives). execute_tool is injectable; pass output_callback
                # only when it accepts it.
                def _invoke_tool(_output_callback, _decision = decision):
                    kwargs = dict(
                        cancel_event = cancel_event,
                        timeout = eff_timeout,
                        session_id = session_id,
                        thread_id = thread_id,
                        rag_scope = rag_scope,
                        disable_sandbox = bypass_permissions,
                    )
                    if _accepts_output_callback(execute_tool):
                        kwargs["output_callback"] = _output_callback
                    return execute_tool(_decision.tool_name, _decision.arguments, **kwargs)

                try:
                    result = yield from stream_tool_execution(
                        _invoke_tool,
                        tool_name = decision.tool_name,
                        tool_call_id = decision.tool_call_id,
                        cancel_event = cancel_event,
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

        append_deferred_nudges(conversation, deferred_noop_msgs)

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
