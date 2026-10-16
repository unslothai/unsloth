# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text heuristics and constants for the tool-calling loop."""

import re

from typing import Optional
from core.inference.tool_call_parser import (
    blocked_markerless_prefix_end,
    held_bare_gemma_tail_len,
    is_reprompt_restatement as _is_reprompt_restatement,
    is_short_intent_without_action as _is_short_intent_without_action,
    promotable_gemma_call_pos,
    REPROMPT_MAX_CHARS as _REPROMPT_MAX_CHARS,
)
from core.tool_healing import _markerless_promotable

# A transport error can arrive before the child is reapable; a request path cannot
# afford the 5s the background MTP reload spends on the same race.
_RESPAWN_REAP_GRACE_S = 1.0


def _finalize_reasoning_only_cumulative(
    cumulative: str,
    reasoning_text: str,
    finish_reason: Optional[str],
    promote_reasoning_only: bool,
    resumed_reasoning: str = "",
) -> str:
    """Close a live thinking block and promote it only after a clean stop.

    Local inference streams cumulative snapshots. Replacing ``<think>...`` with
    bare reasoning at EOF makes the final snapshot shorter, so suffix-based
    route consumers drop the intended fallback. Keep the snapshot append-only.
    A length-truncated thought is not a final answer, so close it without
    promotion and let the client surface the ``length`` terminal state. Raw
    consumers that do not split reasoning from visible content can disable the
    fallback to avoid returning the same reasoning twice.

    Include ``resumed_reasoning`` in the promoted reply; the stream contains only the new tail.
    """
    visible_fallback = (
        resumed_reasoning + reasoning_text
        if promote_reasoning_only and finish_reason != "length"
        else ""
    )
    return cumulative + "</think>" + visible_fallback


def _ends_inside_quote(text: str) -> bool:
    """Heuristic: text ends on a quote opener left unmatched on its line, or a lone quote."""
    stripped = (text or "").rstrip()
    if not stripped or stripped[-1] not in "`\"'":
        return False
    if len(stripped) > 1 and not (stripped[-2].isspace() or stripped[-2] in "*_([{"):
        return False
    last = stripped[-1]
    line = stripped.rsplit("\n", 1)[-1]
    # An apostrophe between letters (Qwen2's, don't) is not a quote mark.
    marks = sum(
        1
        for i, ch in enumerate(line)
        if ch == last
        and not (
            ch == "'" and 0 < i < len(line) - 1 and line[i - 1].isalnum() and line[i + 1].isalnum()
        )
    )
    return marks % 2 == 1


def _quote_cut_event(
    reasoning_text: str,
    answer_text: str,
    finish_reason: Optional[str],
    promote_reasoning_only: bool,
) -> Optional[dict]:
    """Flag a mid-quote `stop` in the shown answer; off for Anthropic, which cannot relay it."""
    if not promote_reasoning_only or finish_reason != "stop":
        return None
    shown = answer_text if answer_text.strip() else reasoning_text
    return {"type": "quote_cut"} if _ends_inside_quote(shown) else None


# Only large streamed tool payloads get an early provisional card; render_html
# is exempt because it needs immediate artifact feedback.
_PROVISIONAL_ARGS_MIN_CHARS = 256
_DEFAULT_STREAM_STALL_TIMEOUT_S = 120.0  # 2 min
# Cap tool calls from a single TEXTUAL-fallback turn (mirrors the safetensors
# loop). Structured delta.tool_calls are grammar-bounded by llama-server; text
# parsed from content is not, so one runaway turn could fan out unbounded.
_MAX_TOOL_CALLS_PER_TURN = 8
# How many times a turn may be resumed after ending inside its own reasoning. Two, because
# the first retry already runs with thinking off: if that still produces nothing visible,
# the window is too small for this model and another attempt only spends more of the user's
# time reaching the same place.
_MAX_LENGTH_CONTINUATIONS = 2
_CONTINUE_AFTER_LENGTH_STATUS = "Continuing after a long thought..."
_CONTINUE_TRUNCATED_ANSWER_STATUS = "Continuing the answer..."
# Below this, a tool result carries no content: what comes back is the notice saying it was
# cut, which costs a call and teaches the model nothing. Sized to leave a short line of real
# output rather than to guarantee a useful one; the point is to catch the collapse to zero.
_MIN_USEFUL_RESULT_TOKENS = 64
# llama.cpp's default per-image embedding cap plus mtmd's delimiters, when the admission figure is out of reach.
_FALLBACK_IMAGE_TOKENS = 4096 + 128


def _mcp_image_tokens(backend) -> int:
    """Per-image KV allowance for the loaded projector, the one admission charges."""
    try:
        from routes.inference import _openai_llama_admission_image_tokens  # noqa: PLC0415
        return int(_openai_llama_admission_image_tokens(backend))
    except Exception:  # noqa: BLE001 -- sizing must never break a tool call
        return _FALLBACK_IMAGE_TOKENS


# How many times one tool may return a byte-identical result before the model is told the
# call is not going to change. Follows OpenClaw's tool-loop detection, which keys on the
# RESULT rather than the arguments and never intervenes while results are still changing:
# the arguments varied on every one of the 18 calls observed here, so an argument-keyed
# guard would have watched the whole thing happen.
_MAX_IDENTICAL_TOOL_RESULTS = 3

# Separates "no kwargs were passed" from "None was passed", which is itself a real
# reasoning-kwargs value meaning send no chat_template_kwargs at all.
_UNSET = object()


# The tail `_fit_result_to_room` appends when a result did not fit. A result carrying it
# is the WINDOW's answer, not the tool's, so two calls with different arguments that both
# end here are not two different answers -- they are the same dead end reached twice.
_WINDOW_NOTICE_MARKER = "chars for the model;"
# The OTHER shape the same cause produces. At a budget of zero there is no room for a
# body at all, so `_truncate` returns `_zero_room_stub` instead of a truncated result
# with a notice appended, and that text carries neither the marker above nor any of the
# result. Missing it is the case this classification exists for: the nudge is skipped and
# the no-progress key falls back to including the arguments, so a model reading a file in
# different slices repeats forever, which is the loop the guard was written to stop.
_ZERO_ROOM_MARKER = "no context room left"


def _is_window_notice(result: str) -> bool:
    return _WINDOW_NOTICE_MARKER in result or _ZERO_ROOM_MARKER in result


def _note_before_envelopes(result: str, tool_name: str, add_note) -> str:
    """Text after the __MCP_IMAGES__ array breaks its parse and ships the base64 as text (#11358)."""
    if not isinstance(result, str):
        return add_note(result)
    from core.inference.tools import _split_frontend_suffix

    body, suffix = _split_frontend_suffix(result, tool_name)
    return add_note(body) + suffix


# How far back the classifier looks for llama.cpp's argument-parsing error. It
# prints one and exits, so it is always at the end; generous enough to survive a
# usage hint printed after it, small enough that a 10 MB unterminated line (which
# _drain_stdout keeps whole) stays cheap to scan.
_FAILURE_SCAN_TAIL_CHARS = 64 * 1024
# How much of a quoted fragment from the child's own output an error message carries.
# Long enough for any real flag or reason, short enough that a wrapper printing
# megabytes cannot make the API error unreadable.
_FAILURE_QUOTE_CHARS = 200
# Obligation phrasing INTENT_SIGNAL leaves alone ("I need to call ..."), paired with
# an action verb. Sentence-anchored: mid-sentence the same words are prose that names
# a tool ("The API I should invoke is foo() because ..."), and suppressing that loses
# a real answer. "should"/"must" sit outside the need|have|ought group because they
# take a bare infinitive. "invoke"/"query" stay out of the verb list: they read as
# technical prose far more often than as a stall.
_FORCED_PLAN_INTENT = re.compile(
    r"(?:^|[.!?]\s+)\s*"
    r"(?:i\s+(?:(?:need|have|ought)\s+to|should|must)|need\s+to|going\s+to|must|should)"
    r"\s+(?:\w+\s+){0,2}?(?:call|use|run|search|fetch|render)\b",
    re.I | re.M,
)
# "the answer is not in the context" announces a *missing* answer, so the negated
# forms are excluded or the plan behind them would ship as the final response.
_FINAL_ANSWER_SIGNAL = re.compile(
    r"\b(?:final\s+answer|answer\s*:|here\s+is|here's|in\s+summary|result\s*:"
    r"|(?:the\s+)?answer\s+is(?!\s+(?:not|unavailable|unknown|unclear|missing)\b))\b",
    re.I,
)
# A plan that pivots ("I should call web_search, but Tokyo is the capital") has an
# answer attached, so the turn must survive. Leaking a plan sentence is cosmetic;
# dropping an answer is not, so the doubtful case keeps the output. The pivot has to
# carry text of its own: "I should call web_search, though." answers nothing.
_ANSWER_PIVOT = re.compile(
    r"\b(?:but|however|although|though|that\s+said|in\s+the\s+meantime|meanwhile)\b"
    r"[\W_]*(?:\w+[\W_]+){1,}\w",
    re.I,
)


def _gguf_active_tool_names(active_tools: list[dict]) -> list[str]:
    names = [
        (tool.get("function") or {}).get("name")
        for tool in (active_tools or [])
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    return [name for name in names if name]


# Rehearsal NAME chars (word + hyphen, matching the parser); the lookbehind excludes the
# Mistral [CALL_ID]...[ARGS] shape.
_GGUF_REHEARSAL_ARGS_RE = re.compile(r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]")


def _gguf_rehearsal_signal_pos(text: str, active_tools: list[dict]) -> int:
    """Index of the first ``NAME[ARGS]`` whose NAME is markerless-promotable, else -1. An
    inactive-name ``foo[ARGS]`` in prose is not a call, nor is a bare execution-class
    ``terminal[ARGS]``; mirrors the safetensors ``_earliest_tool_signal`` gating."""
    active = set(_gguf_active_tool_names(active_tools))
    if not active:
        return -1
    for m in _GGUF_REHEARSAL_ARGS_RE.finditer(text):
        if _markerless_promotable(m.group(1), active):
            return m.start()
    return -1


def _gguf_has_genuine_tool_signal(text: str, signals, active_tools: list[dict]) -> bool:
    """True when ``text`` holds a genuine tool-call boundary for one of ``signals``.

    Unambiguous markers (``<tool_call>``, ``[TOOL_CALLS]``, ``<function=``) count on a
    plain substring hit; an ``[ARGS]`` hit is genuine only when an active tool name
    precedes it, so inactive-name prose is neither drained nor parsed."""
    for sig in signals:
        if sig == "[ARGS]":
            if _gguf_rehearsal_signal_pos(text, active_tools) >= 0:
                return True
            continue
        if sig in text:
            return True
    # Bare Gemma is not in ``signals``, but the parser promotes it wherever it sits, so a
    # mid-prose one is a boundary too.
    return promotable_gemma_call_pos(text, lambda: set(_gguf_active_tool_names(active_tools))) >= 0


_TEXT_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([\w.\-]+)"')
_TEXT_TOOL_GEMMA_RE = re.compile(r"\s*call:([\w.\-]+)")
_TEXT_TOOL_REHEARSAL_RE = re.compile(r"\s*([\w.\-]+)\s*\[ARGS\]")


def _sniff_text_tool_name(text: str, enabled_names: set) -> str:
    """Best-effort tool name from a partially drained TEXT tool call, gated on
    enabled names so prose can never spawn a card. Used only to open the live
    argument pane early; the authoritative parse still happens at stream end.

    The two anchored arms only match a MARKERLESS leading call (a wrapper pushes the shape
    off position 0), so they take the parser's gate: a card for a bare ``call:terminal{``
    would show a call that never runs. The ``"name":`` arm searches the whole prefix and so
    also sees a trusted ``[TOOL_CALLS][{"name":"terminal",..}]``; the markerless bare-JSON
    form cannot reach it, since ``strip_leading_bare_json_call`` refuses to drain it."""
    # A blocked leading call will NOT run, so naming the card after it hands the client a card
    # the real call then reuses by id. Every markerless format is skipped, not just bare JSON.
    text = text[blocked_markerless_prefix_end(text, 0, enabled_names) :]
    m = _TEXT_TOOL_NAME_RE.search(text[:4096])
    if m and m.group(1) in enabled_names:
        return m.group(1)
    m = _TEXT_TOOL_GEMMA_RE.match(text[:256])
    if m and _markerless_promotable(m.group(1), enabled_names):
        return m.group(1)
    m = _TEXT_TOOL_REHEARSAL_RE.match(text[:256])
    if m and _markerless_promotable(m.group(1), enabled_names):
        return m.group(1)
    return ""


def _is_rehearsal_prefix(stripped: str, active_tools: list[dict]) -> bool:
    """True if ``stripped`` is a (possibly partial) prefix of ``NAME[ARGS]`` for a
    markerless-promotable tool: the bare name arriving in its own chunk before
    ``[ARGS]{...}``. An execution-class name is prose here, so it streams rather than being
    held. Mirrors the safetensors loop so the split rehearsal is not leaked."""
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    for name in _gguf_active_tool_names(active_tools):
        # Active by construction, so only the class is left. The shared gate, not the built-in
        # three: an mcp__* name is refused too, and holding its suffix withholds visible text.
        if not _markerless_promotable(name, None):
            continue
        if stripped == name or f"{name}[ARGS]".startswith(stripped):
            return True
    return False


def _held_rehearsal_tail_len(text: str, active_tools: list[dict]) -> int:
    """Length of a trailing bare tool-name token that may be a split rehearsal call
    (``...web_search`` with ``[ARGS]{...}`` still to arrive), so STREAMING can hold it
    instead of leaking the name. Returns 0 for ordinary prose. Mirrors safetensors, including
    the trailing bare-Gemma ``call:NAME{..`` the signal scan misses until its ``{``."""
    i = len(text)
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    tail = text[i:]
    held = len(tail) if tail and _is_rehearsal_prefix(tail, active_tools) else 0
    return max(
        held, held_bare_gemma_tail_len(text, lambda: set(_gguf_active_tool_names(active_tools)))
    )


def _should_suppress_forced_no_tool_output(text: str, previous: str = "") -> bool:
    """Suppress only repeated forced-turn planning text, not final answers.

    ``previous`` is the stall text that triggered the nudge, so a retry that
    moved on can be told from one that just said the same thing again.
    """
    stripped = text.strip()
    if not stripped or len(stripped) >= _REPROMPT_MAX_CHARS:
        return False
    if _FINAL_ANSWER_SIGNAL.search(stripped):
        return False
    plan = _FORCED_PLAN_INTENT.search(stripped)
    if plan is not None:
        # Only the plan itself is safe to drop; anything the turn pivots to after it
        # is the answer the user is waiting for.
        return _ANSWER_PIVOT.search(stripped[plan.end() :]) is None
    if not _is_short_intent_without_action(stripped):
        return False
    # INTENT_SIGNAL also fires on lead-ins to a real answer ("Now I have the results.
    # The capital is Tokyo."), so a bare intent match is a stall only when the retry
    # adds nothing. No ``previous`` keeps the standalone "is this a stall?" contract.
    return not previous or _is_reprompt_restatement(stripped, previous)
