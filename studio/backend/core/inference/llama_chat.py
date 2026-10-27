# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF chat requests: the completion stream, the agentic tool loop and token counting.

LlamaCppBackend inherits LlamaChatMixin, so these methods keep running on the backend's own state:
the llama-server process, its base URL, the loaded model's chat template and capabilities.
Helpers that only these requests use live here too.
"""

import functools
import hashlib
import json
import math
import re
import ssl
import threading
import time
from typing import Any, Callable, Generator, Optional, Union

import httpx

from core.inference.chat_compaction import (
    _archive_and_recall,
    _boundary_metadata,
    _can_reset_epoch,
    _compaction_fit_kwargs,
    _conversation_recall_reserve,
    _fit_with_instruction_pins,
    _keeps_compaction_boundary,
    _memory_tool_withheld,
    _records_boundary,
    _sticky_compaction_details,
    _sticky_compaction_state,
)
from core.inference.context_window import (
    _reply_floor,
    compact_completed_tool_arguments,
    compact_executed_call_arguments,
    compact_refused_tool_arguments,
    estimate_messages_tokens_conservative,
    estimate_messages_tokens_dense,
    messages_without_unpriced_media,
    prompt_budget,
    retrieval_budget,
    tool_result_budget,
    turn_is_servable,
)
from core.inference.llama_tool_schema import llama_grammar_tools
from core.inference.mcp_images import (
    DETACHED_IMAGE_TURN_TEXT as MCP_DETACHED_IMAGE_TURN_TEXT,
    IMAGE_TURN_TEXT as MCP_IMAGE_TURN_TEXT,
    MAX_MODEL_IMAGES as MCP_MAX_MODEL_IMAGES,
    MAX_TOTAL_MODEL_IMAGES as MCP_MAX_TOTAL_MODEL_IMAGES,
    append_image_turn as append_mcp_image_turn,
    is_image_tool as mcp_is_image_tool,
    mentions_images as mcp_images_mentioned_in,
)
from core.inference.passthrough_healing import nudge_enabled as _nudge_enabled
from core.inference.repetition_guard import is_repetition_dominated
from core.inference.stream_errors import stream_error_from_chunk
from core.inference.tool_call_parser import (
    BUDGET_EXHAUSTED_NUDGE,
    MAX_ACT_REPROMPTS as _MAX_REPROMPTS,
    NUDGE_TOOL_CALLS_STATUS as _NUDGE_TOOL_CALLS_STATUS,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    RAG_SEARCH_TOOLS,
    REPROMPT_MAX_CHARS as _REPROMPT_MAX_CHARS,
    StreamingMarkupStripper as _StreamingMarkupStripper,
    TOOL_XML_SIGNALS as _SHARED_TOOL_XML_SIGNALS,
    _GEMMA_BARE_TC_PREFIX_RE,
    _balanced_brace_end,
    blocked_bare_json_chain_may_continue,
    blocked_gemma_chain_may_continue,
    blocked_markerless_prefix_end,
    continue_after_length_message as _continue_after_length_message,
    held_bare_gemma_tail_len,
    is_reprompt_repeat as _is_reprompt_repeat,
    is_reprompt_restatement as _is_reprompt_restatement,
    is_short_intent_without_action as _is_short_intent_without_action,
    leading_bare_gemma_call_is_promotable,
    promotable_gemma_call_pos,
    reasoning_cap_spent_message as _reasoning_cap_spent_message,
    repeated_result_message as _repeated_result_message,
    reprompt_to_act_message as _reprompt_to_act_message,
    starved_result_message as _starved_result_message,
    strip_leading_bare_json_call,
    strip_llama3_leading_sentinels,
    strip_tool_markup as _shared_strip_tool_markup,
    thinking_exhausted_message as _thinking_exhausted_message,
    unfinished_thought_progress as _unfinished_thought_progress,
)
from core.inference.tool_loop_controller import (
    BROWSER_TOOL_NAMES,
    REPEATABLE_TOOL_NAMES,
    ToolLoopController,
    _WORKSPACE_READ_TOOLS,
    _reruns_after_new_work,
    append_deferred_nudges,
    awaiting_approval_status,
    deferred_nudge_text,
    provisional_tool_provenance,
    restore_replayed_tool_results,
    tool_call_limit_nudge,
    with_appended_text,
)
from core.tool_healing import _markerless_promotable
from loggers import get_logger
from state.tool_approvals import (
    DECISION_EXPIRED,
    TOOL_APPROVAL_EXPIRED_MESSAGE,
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    decision_reason,
    new_approval_id,
    note_host_access,
    wait_tool_decision,
)

logger = get_logger(__name__)


def _backend_supports_tools(backend) -> bool:
    """Returns False instead of raising: supports_tools reads load-time state that may not exist yet."""
    try:
        return bool(backend.supports_tools)
    except Exception:  # noqa: BLE001
        return False


class _LlamaStreamCancelled(Exception):
    """Internal signal for an expected client/request cancellation."""

    __slots__ = ()


# vulkan-hpp formats DeviceLostError as "vk::Queue::submit: ErrorDeviceLost".
_GPU_DEVICE_LOST_MARKERS = ("ErrorDeviceLost", "VK_ERROR_DEVICE_LOST")


def _is_gpu_device_lost(exc: BaseException) -> bool:
    """True when llama-server reported a lost GPU device (500 body or SSE error, #11453)."""
    text = str(exc)
    return any(marker in text for marker in _GPU_DEVICE_LOST_MARKERS)


# llama.h: #define LLAMA_DEFAULT_SEED 0xFFFFFFFF
_LLAMA_RANDOM_SEED = 0xFFFFFFFF


def _apply_seeded_llama_request(
    payload: dict,
    seed: Optional[int],
    *,
    reuse_prompt_cache: bool = False,
) -> None:
    """Disable prompt caching for fixed seeds so repeated requests stay reproducible.

    ``reuse_prompt_cache``: a tool-loop continuation of this run; cold, it re-prefills (#10698).
    """
    if seed is None:
        return
    payload["seed"] = seed
    if reuse_prompt_cache:
        return
    # Compared as uint32: the schemas also accept 4294967295, the same "pick at random".
    if (seed & 0xFFFFFFFF) != _LLAMA_RANDOM_SEED:
        payload["cache_prompt"] = False


# Artifact = content no plan has, else a plan with code gets wiped. Numbered lists excluded: they cost errors both ways. Keep runs bounded against backtracking.
_CLOSED_CODE_FENCE = re.compile(
    r"(?<!`)(?P<bf>`{3,})(?!`)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=bf)`*[ \t]*(?:\r?\n|\Z)"
    r"|(?<!~)(?P<tf>~{3,})(?!~)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=tf)~*[ \t]*(?:\r?\n|\Z)",
    re.IGNORECASE,
)


_CLOSED_MARKUP_ARTIFACT = re.compile(
    r"(?:<!doctype\b[\s\S]{0,200}?)?<html\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</html\s*>"
    r"|<svg\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</svg\s*>",
    re.IGNORECASE,
)


_HAS_ANSWER_ARTIFACT = re.compile(
    # Tilde fences too, since models emit tildes when the body holds backticks. CommonMark: 3+ to open, as many to close, on a cleanly ended line.
    r"(?<!`)(?P<bf>`{3,})(?!`)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=bf)`*[ \t]*(?:\r?\n|\Z)"
    r"|(?<!~)(?P<tf>~{3,})(?!~)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=tf)~*[ \t]*(?:\r?\n|\Z)"
    r"|(?:<!doctype\b[\s\S]{0,200}?)?<html\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</html\s*>"
    r"|<svg\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</svg\s*>",
    re.IGNORECASE,
)


_FENCE_RUN_RE = re.compile(r"(?<!`)(?P<backticks>`{3,})(?!`)|(?<!~)(?P<tildes>~{3,})(?!~)")


# One list marker may precede the quote: "- > ```py" is a quote in a list item.
_BLOCKQUOTE_PREFIX = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])?[ \t]*(?:>[ \t]?)+")


# Language token that makes an inline run an opener; never ends in a dot so "here." is prose.
_FENCE_INFO_STRING_RE = re.compile(r"[A-Za-z][\w+#-]*(?:\.[\w+#-]+)*")


# A fence on a list-marker line is block level, so the prose rules below do not apply.
_LIST_MARKER_ONLY = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+$")


_ONE_QUOTE_MARKER = re.compile(r"[ \t]*>[ \t]?")


def _has_unclosed_code_fence(text: str) -> bool:
    """Column-0 fences always count; inline ones only if they cannot read as prose, since models open fences mid-line. Quote pairing is done here, not in the regex."""
    active_char: Optional[str] = None
    active_len = 0
    active_quote = 0
    active_base = 0
    closed_any = False
    for raw_line in text.splitlines():
        # A quote marker four columns in is indented code, not a container. List markers stay alone: deeper, they may continue a list not visible in one line.
        lead = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
        quote = _BLOCKQUOTE_PREFIX.match(raw_line)
        indented_quote = quote is not None and len(lead.expandtabs(4)) > 3
        if indented_quote:
            quote = None
        quote_depth = quote.group(0).count(">") if quote else 0
        line = raw_line[quote.end() :] if quote else raw_line
        runs = list(_FENCE_RUN_RE.finditer(line))
        for index, m in enumerate(runs):
            fence = m.group("backticks") or m.group("tildes")
            trailing = line[m.end() :].strip()
            ch = fence[0]
            prefix = line[: m.start()]
            indent = len(prefix.expandtabs(4))
            blank_prefix = not prefix.strip()
            if active_char is not None:
                # A closer is alone on its line, within 3 columns of the container.
                if (
                    blank_prefix
                    and active_base <= indent <= active_base + 3
                    and ch == active_char
                    and len(fence) >= active_len
                    and quote_depth == active_quote
                    and not trailing
                ):
                    active_char, active_len, active_quote, active_base = None, 0, 0, 0
                    closed_any = True
                continue
            # A later backtick run closes an inline span at any column; other delimiters are info-string text, even tildes inside a tilde info string.
            if ch == "`" and any(
                (later.group("backticks") or later.group("tildes"))[0] == ch
                for later in runs[index + 1 :]
            ):
                continue
            if blank_prefix:
                # A fence's baseline is 0 (its own indent); past three columns it is indented code and opens nothing.
                if indent <= 3:
                    active_char, active_len = ch, len(fence)
                    active_quote, active_base = quote_depth, 0
                continue
            # Models open mid-prose fences: a bare info string marks one, and only before any close. A column-0 fence is a real block and is unaffected.
            in_list = _LIST_MARKER_ONLY.match(prefix) is not None
            if not in_list:
                if indented_quote:
                    continue
                if trailing and not _FENCE_INFO_STRING_RE.fullmatch(trailing):
                    continue
                if closed_any:
                    continue
            active_char, active_len, active_quote = ch, len(fence), quote_depth
            # A list marker is the container, so its width is the baseline; mid-sentence the baseline is 0.
            active_base = indent if in_list else 0
    return active_char is not None


# "First, I'll create an <html></html> skeleton" is a plan, not a page.
_EMPTY_MARKUP_SKELETON = re.compile(
    r"<(html|svg)\b[^>]*>\s*</\1\s*>",
    re.IGNORECASE,
)


_DOCTYPE_PREFIX = re.compile(
    r"^<!doctype\b[\s\S]{0,200}?>",
    re.IGNORECASE,
)


def _is_empty_markup_skeleton(matched: str) -> bool:
    """True if ``matched`` is an empty <html></html> / <svg></svg>, doctype
    prefix and surrounding whitespace allowed."""
    candidate = _DOCTYPE_PREFIX.sub("", matched.strip(), count = 1).strip()
    return _EMPTY_MARKUP_SKELETON.fullmatch(candidate) is not None


def _is_blank_fence(matched: str, depth: int = 0) -> bool:
    """Only whitespace between the delimiter lines counts as blank, so a lone space is not an answer."""
    if matched[:1] not in ("`", "~"):
        return False
    lines = matched.splitlines()
    # Depth is the quote depth of the opener, read from the match start line. A deeper last line is a nested delimiter in the body, so it is content, not blank.
    closing = _BLOCKQUOTE_PREFIX.match(lines[-1]) if lines else None
    if (closing.group(0).count(">") if closing else 0) != depth:
        return False
    for line in lines[1:-1]:
        for _ in range(depth):
            marker = _ONE_QUOTE_MARKER.match(line)
            if marker is None:
                break
            line = line[marker.end() :]
        if line.strip():
            return False
    return True


def _first_real_artifact(text: str):
    """First _HAS_ANSWER_ARTIFACT match with something in it.

    Every match is inspected, so a skeleton followed by a real page counts."""
    for m in _HAS_ANSWER_ARTIFACT.finditer(text):
        if _is_empty_markup_skeleton(m.group(0)):
            continue
        # The container the fence sits in is whatever quotes the line it opens on.
        line_start = text.rfind("\n", 0, m.start()) + 1
        quote = _BLOCKQUOTE_PREFIX.match(text[line_start : m.start()])
        depth = quote.group(0).count(">") if quote else 0
        if _is_blank_fence(m.group(0), depth):
            continue
        return m
    return None


def _strip_markup_outside_fences(text: str) -> str:
    """Blocks that begin inside a code fence are kept, so an example's <html> cannot pair with prose."""
    fences = [m.span() for m in _CLOSED_CODE_FENCE.finditer(text)]

    def _keep_examples(m):
        # Either endpoint inside a fence makes the match that example's content.
        return m.group(0) if any(s <= m.start() < e or s < m.end() <= e for s, e in fences) else ""

    return _CLOSED_MARKUP_ARTIFACT.sub(_keep_examples, text)


# A reasoning opener ends at a tag boundary: "<think-card>" is an element of the
# answer, not a thought.
_THINK_OPEN_RE = re.compile(r"<think[\s>]")


_THINKING_OPEN_RE = re.compile(r"<thinking[\s>]")


_BRACKET_THINK_OPEN_RE = re.compile(r"\[THINK\]")


def _artifact_spans(text: str) -> "list[tuple[int, int]]":
    """Spans of the complete fences and complete markup in ``text``."""
    return [m.span() for m in _CLOSED_CODE_FENCE.finditer(text)] + [
        m.span() for m in _CLOSED_MARKUP_ARTIFACT.finditer(text)
    ]


def _find_outside_artifacts(
    text: str,
    needle: str,
    spans = None,
) -> int:
    """Ignores needle matches inside complete code fences or markup, since an example is not reasoning."""
    at = text.find(needle)
    if at < 0:
        return -1
    if spans is None:
        spans = _artifact_spans(text)
    while at >= 0:
        if not any(start <= at < end for start, end in spans):
            return at
        at = text.find(needle, at + 1)
    return -1


def _text_outside_think(text: str) -> str:
    """Drops only a leading reasoning block; a later think tag is quoted, not reasoning. A prefilled template may send only the closer."""
    stripped = text.lstrip()
    pairs = (
        (_THINKING_OPEN_RE, "<thinking", "</thinking>"),
        (_THINK_OPEN_RE, "<think", "</think>"),
        (_BRACKET_THINK_OPEN_RE, "[THINK]", "[/THINK]"),
    )
    # Settle the leading opener first: its block may name another marker.
    for lead, _opener, closer in pairs:
        if lead.match(stripped):
            # First closer, plainly: guessing which quoted closer is an example let spans swallow the real boundary. Matches _split_think_segments in routes/inference.py.
            close = text.find(closer)
            # With no closer, a thought cut off by the window runs to the end, and none of it was shown.
            return text[close + len(closer) :] if close >= 0 else ""
    # No leading opener: a bare closer comes from a template that prefilled the opener.
    spans = None
    for lead, _opener, closer in pairs:
        if closer in text and spans is None:
            spans = _artifact_spans(text)
        close = _find_outside_artifacts(text, closer, spans)
        # Boundary-aware so <think-card> in the trace is not an opener, which leaves the prefill unstripped.
        if close >= 0 and not lead.search(text[:close]):
            return text[close + len(closer) :]
    return text


def _has_answer_artifact(text: str) -> bool:
    """A closed fence or complete page counts; unfinished fences and empty skeletons do not."""
    # Strip closed artifacts first: tags or backticks inside finished code are content.
    if _has_unclosed_code_fence(_strip_markup_outside_fences(text)):
        return False
    # Nested pages are deliberately not inspected; that needs an HTML parser.
    return _first_real_artifact(text) is not None


# High floor for reasoning-heavy GGUFs and clients that omit max_tokens.
_DEFAULT_MAX_TOKENS_FLOOR = 32768


# A readable local name for the shared `context_window` policy, not a second implementation.
_retrieval_budget = retrieval_budget


def _finalize_reasoning_only_cumulative(
    cumulative: str,
    reasoning_text: str,
    finish_reason: Optional[str],
    promote_reasoning_only: bool,
    resumed_reasoning: str = "",
) -> str:
    """Snapshot must stay append-only or suffix consumers drop the fallback. Promote only on clean stop, never on length; include resumed_reasoning."""
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


# Only large streamed tool args get an early provisional card; render_html is exempt.
_PROVISIONAL_ARGS_MIN_CHARS = 256


# Textual-fallback tool calls are not grammar-bounded, so cap them per turn.
_MAX_TOOL_CALLS_PER_TURN = 8


# The first retry already runs with thinking off; more attempts rarely help.
_MAX_LENGTH_CONTINUATIONS = 2


_CONTINUE_AFTER_LENGTH_STATUS = "Continuing after a long thought..."


_CONTINUE_TRUNCATED_ANSWER_STATUS = "Continuing the answer..."


# Below this a tool result carries only the cut notice; the floor catches a collapse to zero.
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


# Unset (no kwargs passed) differs from None, a real value meaning send no chat_template_kwargs.
_UNSET = object()


# A result with this tail is the window's answer, not the tool's.
_WINDOW_NOTICE_MARKER = "chars for the model;"


# Zero-budget results use _zero_room_stub, which lacks the marker above; missing it lets a model re-read slices forever, so catch it too.
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


# Sentence-anchored: mid-sentence the same words are prose naming a tool and suppressing them loses answers. Keep invoke and query out of the verb list.
_FORCED_PLAN_INTENT = re.compile(
    r"(?:^|[.!?]\s+)\s*"
    r"(?:i\s+(?:(?:need|have|ought)\s+to|should|must)|need\s+to|going\s+to|must|should)"
    r"\s+(?:\w+\s+){0,2}?(?:call|use|run|search|fetch|render)\b",
    re.I | re.M,
)


# Negated forms announce a missing answer, so they are excluded.
_FINAL_ANSWER_SIGNAL = re.compile(
    r"\b(?:final\s+answer|answer\s*:|here\s+is|here's|in\s+summary|result\s*:"
    r"|(?:the\s+)?answer\s+is(?!\s+(?:not|unavailable|unknown|unclear|missing)\b))\b",
    re.I,
)


# A plan that pivots to an answer must survive; the pivot needs its own text. When in doubt keep output: a leaked plan is cosmetic, a dropped answer is not.
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


# Lookbehind excludes the Mistral [CALL_ID]...[ARGS] shape.
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
    """Unambiguous markers count on a plain substring hit; an [ARGS] hit is genuine only with an active tool name before it, so inactive-name prose is not parsed."""
    for sig in signals:
        if sig == "[ARGS]":
            if _gguf_rehearsal_signal_pos(text, active_tools) >= 0:
                return True
            continue
        if sig in text:
            return True
    # The parser promotes bare Gemma anywhere, so a mid-prose one is a boundary too.
    return promotable_gemma_call_pos(text, lambda: set(_gguf_active_tool_names(active_tools))) >= 0


_TEXT_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([\w.\-]+)"')


_TEXT_TOOL_GEMMA_RE = re.compile(r"\s*call:([\w.\-]+)")


_TEXT_TOOL_REHEARSAL_RE = re.compile(r"\s*([\w.\-]+)\s*\[ARGS\]")


def _sniff_text_tool_name(text: str, enabled_names: set) -> str:
    """Only enabled names are returned, so prose cannot open a tool card; the final parse still decides."""
    # A blocked leading call will not run, so skip every markerless format before naming.
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
    """Execution-class names stream rather than being held; otherwise it mirrors the safetensors loop."""
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    for name in _gguf_active_tool_names(active_tools):
        # Use the shared gate, which also refuses mcp__* names.
        if not _markerless_promotable(name, None):
            continue
        if stripped == name or f"{name}[ARGS]".startswith(stripped):
            return True
    return False


def _held_rehearsal_tail_len(text: str, active_tools: list[dict]) -> int:
    """Holds a trailing bare tool name back while its [ARGS] may still arrive, including a bare-Gemma call:NAME{ prefix that the signal scan misses until its brace."""
    i = len(text)
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    tail = text[i:]
    held = len(tail) if tail and _is_rehearsal_prefix(tail, active_tools) else 0
    return max(
        held, held_bare_gemma_tail_len(text, lambda: set(_gguf_active_tool_names(active_tools)))
    )


def _should_suppress_forced_no_tool_output(text: str, previous: str = "") -> bool:
    """Suppresses only a repeat of the stall text that triggered the nudge, never a final answer."""
    stripped = text.strip()
    if not stripped or len(stripped) >= _REPROMPT_MAX_CHARS:
        return False
    if _FINAL_ANSWER_SIGNAL.search(stripped):
        return False
    plan = _FORCED_PLAN_INTENT.search(stripped)
    if plan is not None:
        # Only the plan itself is safe to drop; anything the turn pivots to after it is the answer.
        return _ANSWER_PIVOT.search(stripped[plan.end() :]) is None
    if not _is_short_intent_without_action(stripped):
        return False
    # INTENT_SIGNAL also fires on answer lead-ins; a stall only if the retry adds nothing.
    return not previous or _is_reprompt_restatement(stripped, previous)


def _backfill_usage_from_timings(usage, timings):
    """Fills usage from timings when absent or zero, so the UI's t/s is not diluted by wall-clock time."""
    if not timings:
        return usage
    if usage and usage.get("completion_tokens"):
        return usage
    predicted_n = timings.get("predicted_n")
    prompt_n = timings.get("prompt_n")
    if predicted_n is None and prompt_n is None:
        return usage
    out = dict(usage or {})
    if not out.get("completion_tokens") and predicted_n is not None:
        out["completion_tokens"] = predicted_n
    if not out.get("prompt_tokens") and prompt_n is not None:
        out["prompt_tokens"] = prompt_n
    out["total_tokens"] = int(out.get("prompt_tokens") or 0) + int(
        out.get("completion_tokens") or 0
    )
    return out


def _context_usage_event(usage, timings):
    """UI frame pricing the context one finished tool-loop pass leaves behind, or None.

    ``timings.prompt_n`` excludes the KV-cache hit (``cache_n``) while ``usage.prompt_tokens``
    includes it, so a prompt read from timings adds the hit back.
    """
    usage = usage or {}
    timings = timings or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or timings.get(
        "cache_n"
    )
    prompt = usage.get("prompt_tokens")
    if not prompt and timings.get("prompt_n") is not None:
        prompt = int(timings.get("prompt_n") or 0) + int(timings.get("cache_n") or 0)
    completion = usage.get("completion_tokens") or timings.get("predicted_n")
    if not prompt:
        return None
    prompt, completion = int(prompt), int(completion or 0)
    return {
        "type": "context_usage",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "context_tokens": prompt + completion,
        "cached_tokens": int(cached or 0),
    }


def _llama_chunk_has_generated_output(data: dict) -> bool:
    if data.get("type") == "diffusion_frame":
        return True
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and any(
            value not in (None, "", []) for key, value in delta.items() if key != "role"
        ):
            return True
        if choice.get("text") not in (None, ""):
            return True
    return False


def _perf_callback_wants_timings(callback) -> bool:
    """A callback that only tracks prefill progress opts out of per-token timings."""
    return callback is not None and getattr(callback, "wants_timings", True)


def _report_live_llama_timings(callback, chunk) -> None:
    """Report request-scoped llama.cpp progress without altering the public stream."""
    if callback is None or not isinstance(chunk, dict):
        return
    timings = chunk.get("timings")
    sample = dict(timings) if isinstance(timings, dict) else {}
    # Live samples are not TTFT; terminal metadata retains prompt_ms for the fallback.
    sample.pop("prompt_ms", None)
    progress = chunk.get("prompt_progress")
    if isinstance(progress, dict):
        sample["prompt_progress"] = dict(progress)
        try:
            processed = max(0.0, float(progress.get("processed", 0)))
            cached = max(0.0, float(progress.get("cache", 0)))
            elapsed_ms = float(progress.get("time_ms", 0))
            prompt_n = max(0.0, processed - cached)
            if math.isfinite(elapsed_ms) and elapsed_ms > 0 and math.isfinite(prompt_n):
                sample.update(
                    prompt_n = prompt_n,
                    prompt_per_second = prompt_n / (elapsed_ms / 1000.0),
                )
        except (TypeError, ValueError, OverflowError):
            pass
    if getattr(callback, "needs_phase", True) and _llama_chunk_has_generated_output(chunk):
        sample["running_phase"] = "token_generation"
    if not sample:
        return
    try:
        callback(sample)
    except Exception:
        logger.debug("Ignoring API monitor throughput callback failure", exc_info = True)


class CountAborted(Exception):
    """A token count stood down mid-flight because its answer stopped mattering."""

    __slots__ = ()


@functools.lru_cache(maxsize = 1)
def _local_ssl_context() -> ssl.SSLContext:
    # httpx loads CA certificates even for loopback HTTP. Share that setup, not streaming sockets.
    return ssl.create_default_context()


class LlamaChatMixin:
    """The request-side methods of LlamaCppBackend; they run on the backend's state."""

    def generate_chat_completion(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        replay_preserve_thinking: bool = True,
        continue_final_message: bool = False,
        seed: Optional[int] = None,
        promote_reasoning_only: bool = True,
        perf_callback: Optional[Callable[[dict], None]] = None,
        reasoning_provenance: Optional[dict] = None,
        context_overflow: Optional[str] = None,
        context_policy: Optional[str] = None,
        compaction_headroom_ratio: Optional[float] = None,
        thread_id: Optional[str] = None,
        tools_withheld: bool = False,
        thinking_budget_tokens: Optional[int] = None,
        _allow_respawn_retry: bool = True,
        compact_context: bool = False,
    ) -> Generator[Union[str, dict], None, None]:
        """Streams a chat completion via /v1/chat/completions (llama-server applies template and vision); yields cumulative text like InferenceBackend."""
        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")
        if replay_preserve_thinking:
            preserve_thinking = self._replay_preserve_thinking(
                preserve_thinking, messages, enable_thinking, reasoning_effort
            )

        from core.inference.chat_template_helpers import (
            neutralize_control_markup_in_messages,
            trailing_assistant_resume_kind,
        )

        openai_messages = self._build_openai_messages(messages, image_b64)
        _resume_kind = (
            trailing_assistant_resume_kind(openai_messages) if continue_final_message else None
        )
        continue_final_message = _resume_kind is not None
        resumed_reasoning = (
            openai_messages[-1]["reasoning_content"] if _resume_kind == "reasoning_content" else ""
        )

        payload = {
            # llama-server applies the chat template itself (#7066).
            "messages": neutralize_control_markup_in_messages(
                openai_messages, None, self.markup_profile
            ),
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        retry_messages = messages
        retry_image_b64 = image_b64
        retry_max_tokens = max_tokens
        retry_context_overflow = context_overflow
        retry_preflight_context_length = None
        # Progress events let advancing prefills renew the first-token deadline.
        payload["return_progress"] = True
        if _perf_callback_wants_timings(perf_callback):
            payload["timings_per_token"] = True
        if logit_bias:
            payload["logit_bias"] = logit_bias
        # Per-request enable_thinking / reasoning_effort / preserve_thinking
        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        if _reasoning_kw is not None:
            payload["chat_template_kwargs"] = _reasoning_kw
        if thinking_budget_tokens is not None:
            payload["thinking_budget_tokens"] = thinking_budget_tokens
        if continue_final_message:
            # llama-server applies the template; it rejects both flags set true.
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        # Default cap to the model context when known.
        payload["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        if context_overflow == "truncate_oldest" and self._effective_context_length:
            try:
                _before_fit = openai_messages
                # Same value for boundary read and fit, so they agree on epochs.
                _can_reset = _can_reset_epoch(
                    thread_id,
                    _backend_supports_tools(self),
                    tools_withheld = tools_withheld,
                )
                _sticky, _sticky_is_checkpoint, _sticky_manual = _sticky_compaction_details(
                    thread_id,
                    _before_fit,
                    context_policy = context_policy,
                    can_reset = _can_reset,
                    compaction_headroom_ratio = compaction_headroom_ratio,
                )
                openai_messages, truncation = _fit_with_instruction_pins(
                    openai_messages,
                    context_length = self._effective_context_length,
                    max_tokens = payload["max_tokens"],
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            self._countable_messages(fitted), None, self.markup_profile
                        ),
                        None,
                        None,
                        strict = True,
                        chat_template_kwargs = _reasoning_kw,
                        continue_final_message = continue_final_message,
                        should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                    ),
                    estimate_message = self._estimate_countable_message,
                    reserve_tokens = _conversation_recall_reserve(thread_id),
                    sticky_dropped = _sticky,
                    sticky_is_checkpoint = _sticky_is_checkpoint,
                    sticky_manual = _sticky_manual,
                    force_reset = compact_context,
                    keeps_boundary = _keeps_compaction_boundary(thread_id),
                    can_reset = _can_reset,
                    recall_offered = False,
                    **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                )
                if truncation:
                    # Inline: no tools array here, and strict templates reject a lone tool role.
                    _recalled = _archive_and_recall(
                        openai_messages,
                        _before_fit,
                        branch_messages = _before_fit,
                        thread_id = thread_id,
                        style = "inline",
                        # Checkpoint mode recalls once; rolling recalls every turn that evicted.
                        force_recall = bool(truncation.get("checkpoint_started", True)),
                        # A rescued refusal may evict messages; archive them without recall.
                        recall_done = not truncation["fits"],
                        # This single-shot answer never returns to the prompt.
                        recall_budget_tokens = _retrieval_budget(
                            self._effective_context_length,
                            payload["max_tokens"],
                            truncation.get("prompt_tokens_after") or 0,
                        ),
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                self._countable_messages(fitted),
                                None,
                                self.markup_profile,
                            ),
                            None,
                            None,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                            continue_final_message = continue_final_message,
                        ),
                        cancel_check = lambda: bool(cancel_event and cancel_event.is_set()),
                        background = True,
                    )
                    openai_messages = _recalled["conversation"]
                    truncation = {**truncation, **_recalled["counts"]}
                if truncation and _records_boundary(truncation):
                    truncation = {
                        **truncation,
                        **_boundary_metadata(
                            openai_messages, _before_fit, compaction_headroom_ratio
                        ),
                    }
                payload["messages"] = neutralize_control_markup_in_messages(
                    openai_messages, None, self.markup_profile
                )
                # Reuse the fitted request; re-running preflight would duplicate the event.
                retry_messages = openai_messages
                retry_image_b64 = None
                retry_max_tokens = payload["max_tokens"]
                retry_context_overflow = None
                retry_preflight_context_length = self._effective_context_length
                # Re-cost here, not at the breaks: results and nudges appended after the last re-cost left the pool unaware of this final pass, the largest request.
                if truncation:
                    yield {"type": "context_truncated", **truncation}
            except Exception as exc:
                logger.warning("Could not preflight the rolling context window: %s", exc)
        if stop:
            payload["stop"] = stop
        _apply_seeded_llama_request(payload, seed)
        payload["stream_options"] = {"include_usage": True}

        served_by = getattr(self, "_process", None)
        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""
        in_thinking = False
        _stream_done = False
        _metadata_usage = None
        _metadata_timings = None
        _metadata_finish_reason = None

        def _replay_on_replacement_server():
            context_overflow_ = retry_context_overflow
            max_tokens_ = retry_max_tokens
            if (
                retry_preflight_context_length is not None
                and retry_preflight_context_length != self._effective_context_length
            ):
                # Refit the compacted prompt against the replacement server's window;
                # any event now reports only additional evictions.
                context_overflow_ = context_overflow
                if max_tokens is None:
                    max_tokens_ = None
            yield from self.generate_chat_completion(
                retry_messages,
                image_b64 = retry_image_b64,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_tokens = max_tokens_,
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                stop = stop,
                cancel_event = cancel_event,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                seed = seed,
                promote_reasoning_only = promote_reasoning_only,
                perf_callback = perf_callback,
                reasoning_provenance = reasoning_provenance,
                context_overflow = context_overflow_,
                context_policy = context_policy,
                compaction_headroom_ratio = compaction_headroom_ratio,
                # The retry refits and can evict more turns, so it must get the same thread_id to
                # archive them.
                thread_id = thread_id,
                # The retry refits, so it must be told the same about this request's
                # tools as the first attempt was.
                tools_withheld = tools_withheld,
                thinking_budget_tokens = thinking_budget_tokens,
                compact_context = compact_context,
                _allow_respawn_retry = False,
            )

        try:
            with self._open_stream(url, payload, cancel_event) as (
                response,
                first_token_deadline,
            ):
                buffer = ""
                has_content_tokens = False
                # Track content separately: a literal `</think>` can appear in the answer.
                answer_text = ""
                reasoning_text = ""
                _prov_entry = None
                for raw_chunk in self._iter_text_cancellable(
                    response,
                    cancel_event,
                    first_token_deadline = first_token_deadline,
                ):
                    buffer += raw_chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue
                        if line == "data: [DONE]":
                            if in_thinking:
                                if has_content_tokens:
                                    # Real thinking + content: close the tag
                                    cumulative += "</think>"
                                    _prov_entry = None
                                    yield cumulative
                                else:
                                    # Reasoning-only reply (e.g. Qwen3 always-think): show it as the
                                    # main response.
                                    cumulative = _finalize_reasoning_only_cumulative(
                                        cumulative,
                                        reasoning_text,
                                        _metadata_finish_reason,
                                        promote_reasoning_only,
                                        resumed_reasoning,
                                    )
                                    _prov_entry = None
                                    yield cumulative
                            _stream_done = True
                            break  # exit inner while
                        if not line.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line[6:])
                            _report_live_llama_timings(perf_callback, data)
                            # Diffusion frame from the shim: forward untouched, never into content.
                            if data.get("type") == "diffusion_frame":
                                yield data
                                continue
                            # Capture server timings/usage from final chunks.
                            _chunk_timings = data.get("timings")
                            if _chunk_timings:
                                _metadata_timings = _chunk_timings
                            _chunk_usage = data.get("usage")
                            if _chunk_usage:
                                _metadata_usage = _chunk_usage
                            # Error chunks have no choices; raise so the failure is surfaced.
                            _stream_error = stream_error_from_chunk(data)
                            if _stream_error is not None:
                                raise _stream_error
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                _fr = choices[0].get("finish_reason")
                                if _fr:
                                    _metadata_finish_reason = _fr

                                # Reasoning/thinking tokens: llama-server
                                # sends these as "reasoning_content"; wrap
                                # in <think> tags for the frontend parser.
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    reasoning_text += reasoning
                                    if not in_thinking:
                                        # A leading <think> opened here is genuine reasoning, unlike
                                        # literal tags.
                                        if reasoning_provenance is not None and not cumulative:
                                            reasoning_provenance["wrapped"] = (
                                                reasoning_provenance.get("wrapped", 0) + 1
                                            )
                                            _prov_entry = {"len": 0}
                                            reasoning_provenance.setdefault("wraps", []).append(
                                                _prov_entry
                                            )
                                        cumulative += "<think>"
                                        in_thinking = True
                                    cumulative += reasoning
                                    if _prov_entry is not None:
                                        _prov_entry["len"] += len(reasoning)
                                    yield cumulative

                                token = delta.get("content", "")
                                if token:
                                    has_content_tokens = True
                                    answer_text += token
                                    if in_thinking:
                                        cumulative += "</think>"
                                        in_thinking = False
                                        _prov_entry = None
                                    cumulative += token
                                    yield cumulative
                        except json.JSONDecodeError:
                            logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                    if _stream_done:
                        break  # exit outer for
                _cut = _quote_cut_event(
                    reasoning_text,
                    answer_text,
                    _metadata_finish_reason,
                    promote_reasoning_only,
                )
                if _cut is not None:
                    yield _cut
                if _metadata_usage or _metadata_timings or _metadata_finish_reason:
                    _metadata_usage = _backfill_usage_from_timings(
                        _metadata_usage, _metadata_timings
                    )
                    yield {
                        "type": "metadata",
                        # Never None: consumers call usage.get(...).
                        "usage": _metadata_usage or {},
                        "timings": _metadata_timings,
                        "finish_reason": _metadata_finish_reason,
                    }

        except _LlamaStreamCancelled:
            return
        except httpx.ConnectError as e:
            # Server already down. If this was an MTP crash, recover by
            # reloading without MTP (scheduled in the background) and fail this
            # request. Otherwise the server was likely SIGKILL'd by GPU pressure
            # from a concurrent training run: respawn the same config and retry the
            # generation once (bounded by the private flag, no duplicate output).
            if self._maybe_recover_from_mtp_crash(e):
                raise RuntimeError("Lost connection to llama-server")
            if _allow_respawn_retry and not cumulative and self._respawn_if_dead():
                logger.warning(
                    "llama-server was unreachable; respawned it and retrying the generation"
                )
                yield from _replay_on_replacement_server()
                return
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            if _is_gpu_device_lost(e):
                self._retire_device_lost_server(served_by)
                if _allow_respawn_retry and not cumulative and self._respawn_if_dead():
                    logger.warning(
                        "llama-server lost its GPU device; restarted it and retrying the generation"
                    )
                    yield from _replay_on_replacement_server()
                    return
                raise
            # Died mid-generation: recover MTP, re-raise unchanged for this request.
            self._maybe_recover_from_mtp_crash(e)
            raise

    def generate_chat_completion_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        replayed_image_parts: "tuple" = (),
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        # False when the caller knows the history held reasoning its conversion dropped.
        replay_preserve_thinking: bool = True,
        continue_final_message: bool = False,
        max_tool_iterations: int = 25,
        auto_heal_tool_calls: bool = True,
        nudge_tool_calls: Optional[bool] = None,
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        rag_scope: Optional[dict] = None,
        seed: Optional[int] = None,
        disable_parallel_tool_use: bool = False,
        confirm_tool_calls: bool = False,
        bypass_permissions: bool = False,
        permission_mode: Optional[str] = None,
        promote_reasoning_only: bool = True,
        perf_callback: Optional[Callable[[dict], None]] = None,
        reasoning_provenance: Optional[dict] = None,
        context_overflow: Optional[str] = None,
        context_policy: Optional[str] = None,
        compaction_headroom_ratio: Optional[float] = None,
        tool_choice: Any = None,
        # Append new params only: no bare *, so inserting rebinds positional args. MAY BLOCK for KV room; called at each round top so admission sees actual occupancy.
        on_conversation_grew: Optional[Callable[[list], None]] = None,
        on_decode_slot: Optional[Callable[[str, int], None]] = None,
        thinking_budget_tokens: Optional[int] = None,
        mcp_image = None,
        deduplicate_tool_calls: bool = True,
        instruction_anchor_ids = None,
        sandbox_level: Optional[str] = None,
        compact_context: bool = False,
        website_policy: Optional[dict] = None,
    ) -> Generator[dict, None, None]:
        """
        Agentic loop: let the model call tools, execute them, and continue.

        permission_mode: "ask" confirms every call (with confirm_tool_calls),
        "auto" only pauses calls detected as potentially unsafe, "off" never
        pauses (sandbox stays on), "full" is the same as bypass_permissions.
        Unset/unknown behaves as "ask".

        Yields dicts:
          {"type": "status", "text": "Searching: ..."/"Reading: ..."}   -- tool status updates
          {"type": "content", "text": "token"}            -- streamed content tokens (cumulative)
          {"type": "reasoning", "text": "token"}          -- streamed reasoning tokens (cumulative)
        """
        if replay_preserve_thinking:
            preserve_thinking = self._replay_preserve_thinking(
                preserve_thinking, messages, enable_thinking, reasoning_effort
            )
        from core.inference.tool_stream_exec import (
            accepts_kwarg,
            accepts_output_callback,
            search_images_kwargs,
            stream_tool_execution,
        )
        from core.inference.tools import (
            build_rag_autoinject,
            execute_tool,
            has_text_only_provisional_card,
            is_high_risk_tool_call,
            mcp_image_share,
            mcp_image_targets,
            never_needs_approval,
        )
        from core.inference.mcp_image import note_attached_image

        # full and bypass_permissions are one switch, first wins; off keeps the sandbox but never prompts; unset defaults to auto; unknown falls back to ask.
        from state.tool_policy import (
            account_tool_stream,
            needs_tool_confirmation,
            normalize_sandbox_level,
            normalize_tool_permissions,
            requires_os_isolation,
            runs_without_os_sandbox,
            tool_call_may_prompt,
        )

        permission_mode, bypass_permissions = normalize_tool_permissions(
            permission_mode, bypass_permissions
        )
        sandbox_level = normalize_sandbox_level(sandbox_level)
        stream_tool_execution = account_tool_stream(stream_tool_execution)

        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        if mcp_image is not None:
            messages = note_attached_image(
                messages, mcp_image_targets(_gguf_active_tool_names(tools))
            )
        conversation = restore_replayed_tool_results(messages)
        # Seeded with tool results an explicit skill load relies on, so fitting keeps them.
        _rolling_anchor_ids: set[int] = set(instruction_anchor_ids or ())
        # Recall once per request, or each refit stacks more recalled turns.
        _conversation_recall_done = False
        # Apply the sticky boundary once; refits run on already-fitted history.
        _sticky_boundary_applied = False
        # Estimator misses the tool catalogue and template framing; the offset corrects it.
        _prompt_token_offset: Optional[int] = None
        # The branch this request is on, kept aside before anything is evicted from or
        # injected into `conversation`. The archive is keyed by thread and the stored rows
        # are the whole DAG (Retry keeps the replaced response as a sibling), so recall
        # has to be told which branch is live, and the client's messages are the answer.
        # Restored dicts are new; both branch lists match by identity.
        _request_branch: list[dict] = list(conversation)
        # ...and everything this request adds to it. A long agent run can evict its own
        # earlier tool exchange, which is archived; filtered against the messages the
        # client sent, that document looks like an abandoned branch and is refused, so the
        # model cannot get back a tool result it still needs to answer. Accumulated rather
        # than read off `conversation`, which has already lost what earlier fits evicted.
        _live_branch: list[dict] = list(conversation)
        _live_branch_ids: set[int] = {id(message) for message in conversation}

        def _extend_live_branch(current: list[dict]) -> list[dict]:
            for message in current:
                if id(message) not in _live_branch_ids:
                    _live_branch_ids.add(id(message))
                    _live_branch.append(message)
            return _live_branch

        for message in reversed(conversation):
            if message.get("role") == "user":
                _rolling_anchor_ids.add(id(message))
                break

        _tool_card_by_message: dict[int, str] = {}
        # Feedback folded in after tool_end, reported so the client replays it.
        _model_note_events: list[dict] = []

        def _attach_internal_feedback_to_tool_result(feedback: str) -> bool:
            """Keep controller instructions inside the current tool exchange."""
            if not conversation or conversation[-1].get("role") != "tool":
                return False
            content = conversation[-1].get("content")
            if not isinstance(content, str):
                return False
            # Replaced, not mutated: ``conversation`` copies the list, not the dicts, and
            # /v1/messages passes a caller's own history through, which may end on a tool
            # result. An in-place edit would leave the nudge there and re-send it next turn.
            previous = conversation[-1]
            conversation[-1] = {**previous, "content": f"{content}\n\n{feedback}"}
            card_id = _tool_card_by_message.pop(id(previous), None)
            if card_id:
                _tool_card_by_message[id(conversation[-1])] = card_id
                _model_note_events.append(
                    {
                        "type": "tool_model_note",
                        "tool_call_id": card_id,
                        "suffix": f"\n\n{feedback}",
                    }
                )
            return True

        # Forced first-pass RAG so doc questions do not lose to web_search; skipped only in ask mode, where retrieval would prompt.
        from core.inference.chat_template_helpers import (
            forced_tool_name,
            trailing_assistant_resume_kind,
            trailing_assistant_text,
        )

        initial_forced_name = forced_tool_name(tool_choice)
        if initial_forced_name and initial_forced_name not in _gguf_active_tool_names(tools):
            raise ValueError(
                f"Forced tool '{initial_forced_name}' is not enabled for this request."
            )

        # A resumed turn must keep the partial trailing; autoinject would move the boundary.
        _skip_autoinject = (
            tool_choice == "none"
            or (
                confirm_tool_calls
                and not bypass_permissions
                and permission_mode not in ("auto", "off")
            )
        ) or bool(
            continue_final_message and trailing_assistant_resume_kind(conversation) is not None
        )
        _auto = None if _skip_autoinject else build_rag_autoinject(conversation, rag_scope)
        if _auto:
            for _ev in _auto["events"]:
                yield _ev
            conversation.extend(_auto["messages"])
        _auto_satisfies_forced_choice = bool(_auto) and (
            tool_choice == "required" or forced_tool_name(tool_choice) == "search_knowledge_base"
        )

        _accumulated_completion_tokens = 0
        _accumulated_predicted_ms = 0.0
        _accumulated_predicted_n = 0
        # GGUF buffers reasoning; emit server-side timing before answer text.
        _reasoning_started_at: Optional[float] = None
        _reasoning_summary_emitted = False

        # Gate telling a genuine NAME[ARGS] rehearsal from inactive-name prose; built from the
        # ORIGINAL tools list so a spent one-shot still reads as a tool name. None = no gate.
        from core.inference.tools import mcp_loadable_specs

        # Plus the MCP tools a search can load mid-turn (#10997).
        _loadable_mcp = mcp_loadable_specs(tools)
        _loadable_mcp_names = set(_gguf_active_tool_names(_loadable_mcp))
        _detect_tools = list(tools or []) + _loadable_mcp
        _enabled_names_gate = set(_gguf_active_tool_names(_detect_tools)) if tools else None
        # Detection must see the same names as the strip gate (ORIGINAL list, incl. a spent
        # one-shot), else its repeat is stripped but never drained and the turn ends blank.

        def _reasoning_summary_event(started_at: float) -> dict:
            return {
                "type": "reasoning_summary",
                "duration_ms": round((time.monotonic() - started_at) * 1000.0),
            }

        # Enabled-name gate for the markerless Gemma strip (disabled/example
        # names stay visible). Set per iteration; None = pre-loop name-agnostic.
        _enabled_tool_names = None

        def _gemma_lead_promotable(text: str) -> bool:
            # Reads _enabled_tool_names at call time: it is rebound each tool iteration.
            return leading_bare_gemma_call_is_promotable(text, _enabled_tool_names)

        def _strip_tool_markup(
            text: str,
            *,
            final: bool = False,
            force: bool = False,
        ) -> str:
            if not (auto_heal_tool_calls or force):
                return text
            # Shared parser-side strip covers every family; tool_healing strip omits loop-only forms (python_tag, Mistral name) and would leak them into display.
            return _shared_strip_tool_markup(
                text, final = final, enabled_tool_names = _enabled_names_gate
            )

        # Avoids quadratic rescans. Think blocks stay verbatim: a rehearsed call inside one must not be deleted.
        _streaming_stripper = _StreamingMarkupStripper(_enabled_names_gate)

        # Caller's continue_final_message is restored once a turn progresses; forced only to join one answer.
        _final_answer_stripper = _StreamingMarkupStripper(_enabled_names_gate, seg_final = False)

        def _strip_tool_markup_streaming(text: str, *, force: bool = False) -> str:
            if not (auto_heal_tool_calls or force):
                return text
            return _streaming_stripper.strip(text)

        def _build_metadata_event(usage, timings, finish_reason):
            """Final usage+timings metadata event for the given pass, merging its
            usage/timings with the running cross-iteration accumulators. None when
            there is nothing to report."""
            _fu = _backfill_usage_from_timings(usage, timings) or {}
            _fp = _fu.get("prompt_tokens", 0)
            _tc = _fu.get("completion_tokens", 0) + _accumulated_completion_tokens
            if not (usage or timings or _accumulated_completion_tokens or finish_reason):
                return None
            _mt = dict(timings) if timings else {}
            if _accumulated_predicted_ms or _accumulated_predicted_n:
                _mt["predicted_ms"] = _mt.get("predicted_ms", 0) + _accumulated_predicted_ms
                _mt["predicted_n"] = _mt.get("predicted_n", 0) + _accumulated_predicted_n
                if _mt["predicted_ms"] > 0:
                    _mt["predicted_per_second"] = _mt["predicted_n"] / (
                        _mt["predicted_ms"] / 1000.0
                    )
            _usage = {
                "prompt_tokens": _fp,
                "completion_tokens": _tc,
                "total_tokens": _fp + _tc,
            }
            # Earlier passes' completions are already inside _fp, so the context is _fp + this pass only.
            if _accumulated_completion_tokens:
                _usage["context_tokens"] = _fp + int(_fu.get("completion_tokens") or 0)
            # Preserve KV-cache hit details (cached_tokens) so the tool path
            # reports them like the standard non-tool path does, not always 0.
            if _fu.get("prompt_tokens_details"):
                _usage["prompt_tokens_details"] = _fu["prompt_tokens_details"]
            return {
                "type": "metadata",
                "usage": _usage,
                "timings": _mt,
                "finish_reason": finish_reason,
            }

        def _flush_reasoning_and_buffer():
            """Close a live-streamed <think> block (or emit the buffered reasoning
            as one block if it never streamed), then append the held
            content_buffer to the cumulative display text."""
            nonlocal cumulative_display, in_thinking, _prov_entry, _buffer_in_display
            if in_thinking:
                cumulative_display += "</think>"
                in_thinking = False
                _prov_entry = None
            elif reasoning_accum:
                if (
                    reasoning_provenance is not None
                    and not cumulative_display
                    and not _suppress_visible_output
                ):
                    reasoning_provenance["wrapped"] = reasoning_provenance.get("wrapped", 0) + 1
                    reasoning_provenance.setdefault("wraps", []).append(
                        {"len": len(reasoning_accum)}
                    )
                cumulative_display += "<think>" + reasoning_accum + "</think>"
            cumulative_display += content_buffer
            # Not cleared: callers measure len(content_buffer) right after; recorded in _buffer_in_display so the cancel flush does not re-add it.
            _buffer_in_display = True

        def _cancelled_hold_text() -> str:
            """Keeps display text the guards still hold, so a cancel does not drop a completed reply."""
            if _suppress_visible_output:
                return ""
            held = cumulative_display + ("" if _buffer_in_display else content_buffer)
            if not held:
                return ""
            cleaned = _strip_tool_markup(held, final = True, force = True)
            return cleaned if len(cleaned) > len(_last_emitted) else ""

        def _close_streamed_think() -> bool:
            """Close a live-streamed <think> before a tool call drains, so
            consumers without a reasoning extractor (Anthropic) get a balanced
            block. Returns True when the caller should yield the result."""
            nonlocal cumulative_display, in_thinking, _last_emitted, _prov_entry
            if not in_thinking:
                return False
            cumulative_display += "</think>"
            in_thinking = False
            _prov_entry = None
            if len(cumulative_display) > len(_last_emitted) and not _suppress_visible_output:
                _last_emitted = cumulative_display
                return True
            return False

        def _looks_like_enabled_bare_json(text: str, enabled_tool_names: set) -> bool:
            """True when ``text`` opens with an ENABLED markerless bare-JSON call; an ordinary JSON answer returns False."""
            probe = strip_llama3_leading_sentinels(text.lstrip())
            if not (probe.startswith("{") and ('"name"' in probe or '"function"' in probe)):
                return False
            return strip_leading_bare_json_call(probe, enabled_tool_names) != probe

        # Sanitize at construction: structured tool_calls bypass the name gate; a tool dropped for unsafe markup must leave the controller or stay executable by name.
        from core.inference.chat_template_helpers import (
            neutralize_tool_descriptions as _neutralize_tool_descriptions,
        )

        controller_tools = (
            []
            if tool_choice == "none"
            else _neutralize_tool_descriptions(tools, None, self.markup_profile)
        )
        tool_controller = ToolLoopController(
            tools = controller_tools,
            auto_heal_tool_calls = auto_heal_tool_calls,
            deduplicate_tool_calls = deduplicate_tool_calls,
            session_id = session_id,
            thread_id = thread_id,
        )

        def _tool_succeeded(tool_name: str) -> bool:
            key_prefix = f"{tool_name}:"
            return any(
                record.executed and not record.is_error and record.key.startswith(key_prefix)
                for record in tool_controller.history
            )

        _MAX_BUFFER_CHARS = 32
        # Hold a leading ``{`` well past the 32-char XML cap until it balances (mirrors safetensors).
        _MAX_BARE_JSON_BUFFER = 16384
        _append_budget_exhausted_nudge = True
        # RAG: cap knowledge-base searches per assistant turn. The controller is
        # tool-agnostic, so this gate stays in the loop.
        _kb_search_count = 0

        # ── Re-prompt on plan-without-action ─────────────────
        # Re-prompt once when the model states intent without a tool call; direct answers will not match. Pattern shared with tool_call_parser.INTENT_SIGNAL.
        _reprompt_count = 0
        # Budgeted apart from _reprompt_count so a pre-tool nudge can't spend it.
        _post_tool_reprompts = 0
        # A browser task is many steps, each a chance to announce the next one and stop: there the
        # budget is one nudge per tool turn, as in the external loop, not one per request.
        _browser_turns = any(
            ((tool.get("function") or {}).get("name") in BROWSER_TOOL_NAMES)
            for tool in (tools or [])
        )
        # Text that triggered the last nudge; if the retry restates it, stop.
        _last_reprompt_text = ""
        # Counts real tool turns so re-prompt slots don't extend the budget.
        _tool_iters_done = 0
        _forced_tool_call_pending = False
        # Turns cut off mid-thought: continue in a fresh turn with thinking off, the one setting that guarantees visible output; compaction cannot help.
        _length_continuations = 0
        _effective_enable_thinking = enable_thinking
        # An explicit effort wins over enable_thinking, so clear it too.
        _effective_reasoning_effort = reasoning_effort
        # Restored once a turn gets somewhere: it is turned on to stitch ONE answer
        # back together, not for the rest of the request.
        _caller_continue_final_message = continue_final_message
        # Single-element lists so the nested execution scope can rebind them.
        _last_tool_result_key: list = [None]
        _identical_result_runs: list = [0]
        _starved_call: list = [False]

        # Reserve extra iterations for re-prompts so they don't consume the
        # caller's tool-call budget; only when tool iterations are allowed.
        _extra = _MAX_REPROMPTS + 1 if max_tool_iterations > 0 else 0
        # Owned by this request and dropped with it: the sweep below re-reads every earlier
        # turn on each iteration, and the rewrite is a function of the text alone (#7066).
        from core.inference.chat_template_helpers import sweep_cache as _sweep_cache

        _markup_cache = _sweep_cache()

        def _evict_until_it_fits(
            candidate,
            tools,
            reasoning_kw,
            continue_flag = False,
            protect = None,
        ):
            """Evicts older turns from a continuation one eviction short, using the plain fit only, not the full preflight (which would archive and recall again)."""
            if context_overflow != "truncate_oldest" or not self._effective_context_length:
                return None
            try:
                _fitted, _truncation = _fit_with_instruction_pins(
                    list(candidate),
                    context_length = self._effective_context_length,
                    # Room the continuation needs, not the caller's cap (a small cap overstates room).
                    max_tokens = max(
                        int(max_tokens or 0), _reply_floor(self._effective_context_length)
                    ),
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            self._countable_messages(fitted),
                            _markup_cache,
                            self.markup_profile,
                        ),
                        None,
                        tools,
                        strict = True,
                        chat_template_kwargs = reasoning_kw,
                        continue_final_message = continue_flag,
                    ),
                    estimate_message = self._estimate_countable_message,
                    anchor_ids = (_rolling_anchor_ids | protect) if protect else _rolling_anchor_ids,
                )
            except Exception:
                logger.debug("continuation eviction: fit failed", exc_info = True)
                return None
            if not _truncation or not _truncation.get("dropped_messages"):
                return None
            logger.info(
                "Continuation did not fit; evicted %s older message(s) to try to keep it",
                _truncation.get("dropped_messages"),
            )
            # Reset each iteration: set when a reasoning retry was refused for the caller's allowance.
            return _fitted

        def _loop_continuation_fits(
            candidate,
            tools,
            reasoning_kw,
            continue_flag = False,
        ) -> bool:
            """Whether the retry's prompt still fits with the partial; an uncountable prompt fails open."""
            if not self._effective_context_length:
                return True
            try:
                _tokens = self.count_chat_tokens(
                    neutralize_control_markup_in_messages(
                        self._countable_messages(candidate),
                        _markup_cache,
                        self.markup_profile,
                    ),
                    None,
                    tools,
                    strict = True,
                    chat_template_kwargs = reasoning_kw,
                    # continue_final_message renders differently; price the request actually sent.
                    continue_final_message = continue_flag,
                )
            except Exception:
                logger.debug("in-loop continuation: prompt count failed", exc_info = True)
                return True
            return _tokens + _reply_floor(self._effective_context_length) <= (
                self._effective_context_length
            )

        def _loop_budget_left(spent_this_attempt: int) -> "Optional[int]":
            """What is left of the caller's own max_tokens, since the loop rebuilds it; None when unset."""
            if max_tokens is None:
                return None
            window = self._effective_context_length or 0
            if window and max_tokens >= window:
                return None
            return max(0, max_tokens - _accumulated_completion_tokens - spent_this_attempt)

        # A forced function applies until the model produces it; execution or denial resolves it.
        _forced_choice_resolved = _auto_satisfies_forced_choice
        # Continuations get a credit so they don't consume the caller's tool budget.
        # Holds the remaining output cap for the next iteration.
        _continuation_max_tokens: Optional[int] = None
        _continuation_credits = 0
        _MAX_CONTINUATION_CREDITS = _MAX_LENGTH_CONTINUATIONS * max(1, max_tool_iterations)
        # A fixed seed keeps the prompt cache once this run has sent a request (#10698).
        _prompt_sent = False
        # Resent on the final pass with tool_choice "none": tools head the system prompt.
        _last_sent_tools = None
        iteration = -1
        # Cap is across the whole conversation, seeded with already promoted images.
        loop_mcp_image_parts: list = list(replayed_image_parts)
        while True:
            iteration += 1
            while _model_note_events:
                yield _model_note_events.pop(0)
            # Here rather than at each append: six sites grow the conversation and all of
            # them pass through this one point before the cache must hold the result.
            if on_conversation_grew is not None:
                try:
                    on_conversation_grew(conversation)
                except Exception:  # accounting must never break a run in progress
                    logger.debug("tool loop recost failed", exc_info = True)
            if iteration >= max_tool_iterations + _extra + _continuation_credits:
                break
            if cancel_event is not None and cancel_event.is_set():
                return
            # Whether this turn ran a tool; a no-op-only turn stays False and doesn't consume budget.
            _turn_executed_real_tool = False
            # Set when a reasoning retry was refused because the CALLER's allowance was
            # spent rather than the window. Per iteration: it describes this turn only.
            _reasoning_cap_spent = False

            active_tools = tool_controller.active_tools()
            if not active_tools:
                _append_budget_exhausted_nudge = False
                break
            from core.inference.chat_template_helpers import (
                neutralize_tool_descriptions,
                reconciled_tool_choice,
            )

            # Drop unsafe-markup tools from execution too, not just from the prompt.
            safe_tools = neutralize_tool_descriptions(
                active_tools, _markup_cache, self.markup_profile
            )
            requested_choice = "auto" if tool_choice is None else tool_choice
            if _forced_choice_resolved and requested_choice not in ("auto", "none"):
                requested_choice = "auto"
            requested_choice = reconciled_tool_choice(requested_choice, tools, safe_tools) or "auto"
            forced_name = forced_tool_name(requested_choice)
            if forced_name:
                matching_tools = [
                    tool
                    for tool in safe_tools
                    if (tool.get("function") or {}).get("name") == forced_name
                ]
                if matching_tools:
                    safe_tools = matching_tools
                    requested_choice = "required"
            # Gate the markerless bare-JSON form on enabled names so an ordinary JSON answer isn't misread as a call.
            _enabled_tool_names = {
                (tool.get("function") or {}).get("name")
                for tool in safe_tools
                if (tool.get("function") or {}).get("name")
            } | (set() if forced_name else _loadable_mcp_names)
            # Shared signal tuple so GGUF BUFFERING wakes on every format the parser knows (like safetensors).
            _tool_xml_signals = _SHARED_TOOL_XML_SIGNALS

            # Build payload -- stream: True so we detect tool signals
            # in the first 1-2 chunks without a non-streaming penalty.
            from core.inference.chat_template_helpers import (
                append_assistant_turn,
                neutralize_control_markup,
                neutralize_control_markup_in_messages,
            )

            _reasoning_kw = self._request_reasoning_kwargs(
                _effective_enable_thinking, _effective_reasoning_effort, preserve_thinking
            )
            # On a continuation only the remainder of the caller's cap; read before the payload.
            _iteration_max_tokens = (
                _continuation_max_tokens if _continuation_max_tokens is not None else max_tokens
            )
            _preflight_context_length = None
            _preflight_succeeded = False
            if context_overflow == "truncate_oldest" and self._effective_context_length:
                _preflight_context_length = self._effective_context_length
                try:
                    _before_fit = conversation
                    _iteration_can_reset = _can_reset_epoch(
                        thread_id,
                        _backend_supports_tools(self),
                        tools_withheld = _memory_tool_withheld(thread_id, tools),
                    )
                    # The saved boundary and /compact both apply once, to the loop's first fit.
                    _iteration_force_reset = compact_context and not _sticky_boundary_applied
                    _iteration_sticky, _iteration_sticky_is_checkpoint, _iteration_sticky_manual = (
                        (0, True, False)
                        if _sticky_boundary_applied
                        else _sticky_compaction_details(
                            thread_id,
                            _request_branch,
                            context_policy = context_policy,
                            can_reset = _iteration_can_reset,
                            compaction_headroom_ratio = compaction_headroom_ratio,
                        )
                    )
                    conversation, truncation = _fit_with_instruction_pins(
                        conversation,
                        context_length = self._effective_context_length,
                        max_tokens = _iteration_max_tokens,
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                self._countable_messages(fitted),
                                _markup_cache,
                                self.markup_profile,
                            ),
                            None,
                            safe_tools,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                            continue_final_message = bool(
                                continue_final_message
                                and trailing_assistant_resume_kind(fitted) is not None
                            ),
                            should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                        ),
                        estimate_message = self._estimate_countable_message,
                        anchor_ids = _rolling_anchor_ids,
                        keeps_boundary = _keeps_compaction_boundary(thread_id),
                        can_reset = _iteration_can_reset,
                        recall_offered = "search_conversation" in (_enabled_tool_names or ()),
                        reserve_tokens = _conversation_recall_reserve(thread_id),
                        sticky_dropped = _iteration_sticky,
                        sticky_is_checkpoint = _iteration_sticky_is_checkpoint,
                        sticky_manual = _iteration_sticky_manual,
                        force_reset = _iteration_force_reset,
                        **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                    )
                    # Accounted for in this request now, whatever the fit decided.
                    _sticky_boundary_applied = True
                    if truncation and truncation.get("fits"):
                        # Before the recall injection, so both sides describe the same
                        # messages, priced by the fit with the catalogue and template.
                        _prompt_token_offset = int(
                            truncation.get("prompt_tokens_after") or 0
                        ) - sum(self._estimate_countable_message(m) for m in conversation)
                    if truncation:
                        _recalled = _archive_and_recall(
                            conversation,
                            _before_fit,
                            branch_messages = _extend_live_branch(_before_fit),
                            thread_id = thread_id,
                            # Forge a tool exchange only when search_conversation is advertised.
                            style = (
                                "tool" if "search_conversation" in _enabled_tool_names else "inline"
                            ),
                            # Checkpoint mode recalls once, on the turn that reset; rolling has
                            # no such key and recalls on every turn that evicted anything.
                            force_recall = bool(truncation.get("checkpoint_started", True)),
                            # A rescued refusal may evict messages; archive them without recall.
                            recall_done = _conversation_recall_done or not truncation["fits"],
                            # Leave room for the answer appended on the next iteration.
                            recall_budget_tokens = _retrieval_budget(
                                self._effective_context_length,
                                # As above: the cap this iteration will actually send.
                                _iteration_max_tokens,
                                truncation.get("prompt_tokens_after") or 0,
                                reply_returns = True,
                            ),
                            count_tokens = lambda fitted: self.count_chat_tokens(
                                neutralize_control_markup_in_messages(
                                    self._countable_messages(fitted),
                                    _markup_cache,
                                    self.markup_profile,
                                ),
                                None,
                                safe_tools,
                                strict = True,
                                chat_template_kwargs = _reasoning_kw,
                            ),
                            cancel_check = lambda: bool(cancel_event and cancel_event.is_set()),
                            background = True,
                        )
                        conversation = _recalled["conversation"]
                        truncation = {**truncation, **_recalled["counts"]}
                        if _recalled["recalled"]:
                            _conversation_recall_done = True
                            # Anchor exactly what the injection added, not the last two messages.
                            for _message in _recalled["anchored"]:
                                _rolling_anchor_ids.add(id(_message))
                            for _ev in _recalled["events"]:
                                yield _ev
                    if truncation and _records_boundary(truncation):
                        truncation = {
                            **truncation,
                            **_boundary_metadata(
                                conversation, _request_branch, compaction_headroom_ratio
                            ),
                        }
                    # `fits` False too: it carries the does-not-fit diagnosis.
                    if truncation:
                        yield {"type": "context_truncated", **truncation}
                    _preflight_succeeded = True
                except Exception as exc:
                    logger.warning("Could not preflight the rolling context window: %s", exc)

            payload = {
                # Re-run each iteration: tool results may contain forged turns.
                "messages": neutralize_control_markup_in_messages(
                    conversation, _markup_cache, self.markup_profile
                ),
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if top_k >= 0 else 0,
                "min_p": min_p,
                "repeat_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }

            # Progress events feed the first-token deadline; timings stay opt-in.
            payload["return_progress"] = True
            if on_decode_slot is not None:
                payload["verbose"] = True
                payload["response_fields"] = ["id_slot"]
            if _perf_callback_wants_timings(perf_callback):
                payload["timings_per_token"] = True
            if logit_bias:
                payload["logit_bias"] = logit_bias
            # An empty list would still advertise tool use.
            if safe_tools:
                payload["tools"] = llama_grammar_tools(safe_tools)
                payload["tool_choice"] = requested_choice
                _last_sent_tools = safe_tools
            if _reasoning_kw is not None:
                payload["chat_template_kwargs"] = _reasoning_kw
            if thinking_budget_tokens is not None:
                payload["thinking_budget_tokens"] = thinking_budget_tokens
            # Re-checked per iteration: once a tool result is appended the partial is
            # no longer trailing, so later turns are normal.
            _resume_kind = (
                trailing_assistant_resume_kind(conversation) if continue_final_message else None
            )
            if _resume_kind is not None:
                payload["continue_final_message"] = True
                payload["add_generation_prompt"] = False
            _resumed_reasoning = (
                conversation[-1]["reasoning_content"] if _resume_kind == "reasoning_content" else ""
            )
            payload["max_tokens"] = (
                max_tokens
                if max_tokens is not None
                else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
            )
            # A continuation spends the caller's output budget once, not per attempt.
            if _continuation_max_tokens is not None:
                payload["max_tokens"] = _continuation_max_tokens
                _continuation_max_tokens = None
            if stop:
                payload["stop"] = stop
            _apply_seeded_llama_request(payload, seed, reuse_prompt_cache = _prompt_sent)
            _prompt_sent = True

            _respawn_truncations: list[dict] = []

            def _refit_iteration_after_respawn() -> None:
                nonlocal conversation
                _before_respawn_fit = conversation
                if (
                    _preflight_context_length is None
                    or not self._effective_context_length
                    or (
                        _preflight_succeeded
                        and _preflight_context_length == self._effective_context_length
                    )
                ):
                    return
                if max_tokens is None:
                    payload["max_tokens"] = self._effective_context_length
                try:
                    conversation, truncation = _fit_with_instruction_pins(
                        conversation,
                        context_length = self._effective_context_length,
                        max_tokens = _iteration_max_tokens,
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                self._countable_messages(fitted),
                                _markup_cache,
                                self.markup_profile,
                            ),
                            None,
                            safe_tools,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                            continue_final_message = bool(
                                continue_final_message
                                and trailing_assistant_resume_kind(fitted) is not None
                            ),
                            should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                        ),
                        estimate_message = self._estimate_countable_message,
                        anchor_ids = _rolling_anchor_ids,
                        keeps_boundary = _keeps_compaction_boundary(thread_id),
                        can_reset = _can_reset_epoch(
                            thread_id,
                            _backend_supports_tools(self),
                            tools_withheld = _memory_tool_withheld(thread_id, tools),
                        ),
                        recall_offered = "search_conversation" in (_enabled_tool_names or ()),
                        **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                    )
                    # Record here: the reopened stream raises before forwarding sees the refusal.
                    from core.inference import context_refusal  # noqa: PLC0415

                    context_refusal.record_fit(truncation)
                    if truncation:
                        # Cards with a first tool_args event; later fragments stream so big payloads don't look frozen.
                        truncation.update(
                            _archive_and_recall(
                                conversation,
                                _before_respawn_fit,
                                thread_id = thread_id,
                                style = "inline",
                                recall_done = True,
                                branch_messages = _extend_live_branch(conversation),
                                cancel_check = lambda: bool(cancel_event and cancel_event.is_set()),
                                background = True,
                            )["counts"]
                        )
                    payload["messages"] = neutralize_control_markup_in_messages(
                        conversation, _markup_cache, self.markup_profile
                    )
                    if truncation:
                        if _records_boundary(truncation):
                            truncation.update(
                                _boundary_metadata(
                                    conversation, _request_branch, compaction_headroom_ratio
                                )
                            )
                        # Report every shortened retry, including a rescued refusal.
                        _respawn_truncations.append(truncation)
                except Exception as exc:
                    logger.warning("Could not refit rolling context after respawn: %s", exc)
                    raise

            try:
                # ── Speculative buffer state machine ──────────────────
                # BUFFERING: hold content, check for tool signals
                # STREAMING: no tool, yield tokens; DRAINING: tool found, consume silently
                _S_BUFFERING = 0
                _S_STREAMING = 1
                _S_DRAINING = 2

                detect_state = _S_BUFFERING
                content_buffer = ""  # Raw content held during BUFFERING
                # Whether content_buffer has already been added to cumulative_display.
                _buffer_in_display = False
                content_accum = ""  # All content tokens (for tool parsing)
                reasoning_accum = ""
                _prov_entry = None
                # Time each reasoning pass so final answers can replace tool timing.
                _reasoning_started_at = None
                _reasoning_summary_emitted = False
                _deferred_reasoning_summary = None
                cumulative_display = ""  # Cumulative yielded text (with <think>)
                # A new buffer, not a continuation of the previous iteration's.
                _streaming_stripper.reset()
                in_thinking = False
                has_content_tokens = False
                tool_calls_acc = {}  # Structured delta.tool_calls fragments
                # accumulator key each index maps to now, (index, call_id) after a fork
                tool_calls_open_key: dict[Any, Any] = {}
                tool_calls_synthetic_ids: set[Any] = set()  # slots still on a placeholder id
                has_structured_tc = False
                _iter_usage = None
                _iter_timings = None
                _iter_finish_reason = None
                _stream_done = False
                _last_emitted = ""
                # Provisional tool_start cards already shown, keyed by tool_call_id.
                provisional_started_tool_calls: dict[str, str] = {}
                resolved_provisional_tool_call_ids: set[str] = set()
                _suppress_visible_output = _forced_tool_call_pending
                # Same reply-room remainder as the preflight; whole-cap pricing would compact turns with room.
                arg_streamed_tool_call_ids: set[str] = set()
                # Provisional card uses the parser's first-call id ('call_0') so tool_start reconciles.
                _text_args_call_start = -1
                _text_args_streamed_upto = -1
                _text_args_id = ""
                _text_args_name = ""
                _confirm_gated_iteration = bool(confirm_tool_calls) and not bypass_permissions

                # Compact when the prompt eats the reply room, not only when it doesn't fit. Never refuses.
                if self._effective_context_length and conversation:
                    # Same remainder the preflight used: this check decides whether to
                    # compact, and pricing a continuation as if it had the whole cap
                    # compacts a turn that had room.
                    _reply_target = prompt_budget(
                        self._effective_context_length, _iteration_max_tokens
                    )
                    if (
                        estimate_messages_tokens_dense(self._countable_messages(conversation))
                        > 0.7 * _reply_target
                    ):
                        try:
                            _prompt_now = self.count_chat_tokens(
                                neutralize_control_markup_in_messages(
                                    self._countable_messages(conversation),
                                    _markup_cache,
                                    self.markup_profile,
                                ),
                                None,
                                safe_tools,
                                strict = True,
                                chat_template_kwargs = _reasoning_kw,
                            )
                        except Exception:
                            logger.debug("reply-room fit: prompt count failed", exc_info = True)
                            _prompt_now = None
                        if _prompt_now is not None and _prompt_now > _reply_target:
                            _roomy, _n_roomy = compact_completed_tool_arguments(
                                conversation, protect_last = 1
                            )
                            if _n_roomy:
                                conversation[:] = _roomy
                                payload["messages"] = neutralize_control_markup_in_messages(
                                    conversation, _markup_cache, self.markup_profile
                                )
                                logger.info(
                                    "Compacted %d completed tool call(s) for reply room: "
                                    "prompt was %d tokens against a %d-token budget",
                                    _n_roomy,
                                    _prompt_now,
                                    _reply_target,
                                )

                with self._open_chat_stream_with_respawn_retry(
                    payload,
                    cancel_event,
                    on_respawn = _refit_iteration_after_respawn,
                ) as (
                    response,
                    first_token_deadline,
                ):
                    for truncation in _respawn_truncations:
                        yield {"type": "context_truncated", **truncation}
                    raw_buf = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response,
                        cancel_event,
                        first_token_deadline = first_token_deadline,
                    ):
                        raw_buf += raw_chunk
                        while "\n" in raw_buf:
                            line, raw_buf = raw_buf.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                # Flush thinking state for STREAMING
                                if detect_state == _S_STREAMING and in_thinking:
                                    if has_content_tokens:
                                        cumulative_display += "</think>"
                                        _prov_entry = None
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": _strip_tool_markup(
                                                    cumulative_display,
                                                    final = True,
                                                ),
                                            }
                                    else:
                                        cumulative_display = _finalize_reasoning_only_cumulative(
                                            cumulative_display,
                                            reasoning_accum,
                                            _iter_finish_reason,
                                            promote_reasoning_only,
                                            _resumed_reasoning,
                                        )
                                        _prov_entry = None
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                chunk_data = json.loads(line[6:])
                                if on_decode_slot is not None:
                                    slot = self.decode_slot_from_chunk(chunk_data)
                                    if slot is not None:
                                        on_decode_slot(self.base_url, slot)

                                _report_live_llama_timings(perf_callback, chunk_data)
                                _ct = chunk_data.get("timings")
                                if _ct:
                                    _iter_timings = _ct
                                _cu = chunk_data.get("usage")
                                if _cu:
                                    _iter_usage = _cu

                                # Error chunks have no choices; don't drop them.
                                _stream_error = stream_error_from_chunk(chunk_data)
                                if _stream_error is not None:
                                    raise _stream_error
                                choices = chunk_data.get("choices", [])
                                if not choices:
                                    continue

                                delta = choices[0].get("delta", {})
                                _fr = choices[0].get("finish_reason")
                                if _fr:
                                    _iter_finish_reason = _fr

                                # ── Structured tool_calls ──
                                tc_deltas = delta.get("tool_calls")
                                if tc_deltas:
                                    # Preserve any visible preface before draining
                                    # the structured tool call.
                                    has_structured_tc = True
                                    detect_state = _S_DRAINING
                                    # Close the reasoning prefix before the tool card
                                    # (mirrors the is_match path).
                                    if _close_streamed_think():
                                        yield {"type": "content", "text": cumulative_display}
                                    for tc_d in tc_deltas:
                                        idx = tc_d.get("index", 0)
                                        tc_id = tc_d.get("id")
                                        # a second call at a reused index forks its own slot; bare fragments follow whichever call owns the index now
                                        acc_key = tool_calls_open_key.get(idx, idx)
                                        if isinstance(tc_id, str) and tc_id:
                                            open_id = tool_calls_acc.get(acc_key, {}).get("id")
                                            # a synthetic placeholder is not a rival call, so a first real id still lands on the slot that is open
                                            if (
                                                open_id
                                                and acc_key not in tool_calls_synthetic_ids
                                                and open_id != tc_id
                                            ):
                                                first_id = tool_calls_acc.get(idx, {}).get("id")
                                                acc_key = idx if first_id == tc_id else (idx, tc_id)
                                        tool_calls_open_key[idx] = acc_key
                                        if acc_key not in tool_calls_acc:
                                            tool_calls_acc[acc_key] = {
                                                "id": tc_id or f"call_{idx}",
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": "",
                                                },
                                            }
                                            if not tc_id:
                                                tool_calls_synthetic_ids.add(acc_key)
                                        elif tc_id:
                                            # Display only, accumulator untouched: first tool_args event sends the backlog, later ones fragments.
                                            tool_calls_acc[acc_key]["id"] = tc_id
                                            tool_calls_synthetic_ids.discard(acc_key)
                                        func = tc_d.get("function", {})
                                        if func.get("name"):
                                            tool_calls_acc[acc_key]["function"]["name"] += func[
                                                "name"
                                            ]
                                        if func.get("arguments"):
                                            tool_calls_acc[acc_key]["function"]["arguments"] += (
                                                func["arguments"]
                                            )
                                        current_name = tool_calls_acc[acc_key]["function"].get(
                                            "name", ""
                                        )
                                        fallback_id = f"call_{idx}"
                                        current_id = tool_calls_acc[acc_key].get("id", fallback_id)
                                        is_first_accumulated_call = (
                                            next(iter(tool_calls_acc)) == acc_key
                                        )
                                        already_started = (
                                            current_id in provisional_started_tool_calls
                                        )
                                        # Empty/synthetic ids cannot reconcile with real starts.
                                        has_real_id = bool(current_id) and current_id != fallback_id
                                        # Show one early card per eligible streamed tool call.
                                        _is_completed_one_shot = (
                                            current_name == "render_html"
                                            and _tool_succeeded("render_html")
                                        )
                                        # render_html is one-shot.
                                        _one_shot_already_provisional = (
                                            current_name == "render_html"
                                            and "render_html"
                                            in provisional_started_tool_calls.values()
                                        )
                                        # Later parallel cards reconcile only when parallel use is on; early cards follow the prompt rule, not the raw confirm flag, so always-safe tools stream.
                                        _confirm_gated = (
                                            tool_call_may_prompt(
                                                confirm_tool_calls = bool(confirm_tool_calls),
                                                bypass_permissions = bypass_permissions,
                                                permission_mode = permission_mode,
                                                name = current_name,
                                                sandbox_level = sandbox_level,
                                                session_id = session_id,
                                            )
                                            # A text-preview card still streams while gated;
                                            # hiding it blanks the chat.
                                            and not has_text_only_provisional_card(current_name)
                                        )
                                        # Keep small-argument tools on the normal path.
                                        _args_len = len(
                                            tool_calls_acc[acc_key]["function"].get("arguments", "")
                                        )
                                        _payload_is_large = (
                                            current_name == "render_html"
                                            or _args_len >= _PROVISIONAL_ARGS_MIN_CHARS
                                        )
                                        if (
                                            current_name
                                            and (
                                                is_first_accumulated_call
                                                or not disable_parallel_tool_use
                                            )
                                            and has_real_id
                                            and not already_started
                                            and not _is_completed_one_shot
                                            and not _one_shot_already_provisional
                                            and not _confirm_gated
                                            and _payload_is_large
                                            and any(
                                                (tool.get("function") or {}).get("name")
                                                == current_name
                                                for tool in safe_tools
                                            )
                                        ):
                                            provisional_started_tool_calls[current_id] = (
                                                current_name
                                            )
                                            yield {
                                                "type": "tool_start",
                                                "tool_name": current_name,
                                                "tool_call_id": current_id,
                                                "arguments": {},
                                                "provenance": provisional_tool_provenance(
                                                    current_name
                                                ),
                                            }
                                        # Stream argument text so the UI shows the code being
                                        # written: first event the backlog, later the fragment.
                                        # Display only; accumulator untouched.
                                        if current_id in provisional_started_tool_calls:
                                            if current_id not in arg_streamed_tool_call_ids:
                                                arg_streamed_tool_call_ids.add(current_id)
                                                _args_backlog = tool_calls_acc[acc_key][
                                                    "function"
                                                ].get("arguments", "")
                                                if _args_backlog:
                                                    yield {
                                                        "type": "tool_args",
                                                        "tool_call_id": current_id,
                                                        "tool_name": current_name,
                                                        "text": _args_backlog,
                                                    }
                                            elif func.get("arguments"):
                                                yield {
                                                    "type": "tool_args",
                                                    "tool_call_id": current_id,
                                                    "tool_name": current_name,
                                                    "text": func["arguments"],
                                                }
                                    continue

                                # ── Reasoning tokens ──
                                # Reasoning streams live except while DRAINING: it is orthogonal to tool detection, and the route resets prev_text on tool_start so think stays a monotonic prefix.
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    if _reasoning_started_at is None:
                                        _reasoning_started_at = time.monotonic()
                                    reasoning_accum += reasoning
                                    if detect_state != _S_DRAINING:
                                        if not in_thinking:
                                            # Suppressed iterations emit no content, so don't record
                                            # their wraps.
                                            if (
                                                reasoning_provenance is not None
                                                and not cumulative_display
                                                and not _suppress_visible_output
                                            ):
                                                reasoning_provenance["wrapped"] = (
                                                    reasoning_provenance.get("wrapped", 0) + 1
                                                )
                                                _prov_entry = {"len": 0}
                                                reasoning_provenance.setdefault("wraps", []).append(
                                                    _prov_entry
                                                )
                                            cumulative_display += "<think>"
                                            in_thinking = True
                                        cumulative_display += reasoning
                                        if _prov_entry is not None:
                                            _prov_entry["len"] += len(reasoning)
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }

                                # ── Content tokens ──
                                token = delta.get("content", "")
                                if token:
                                    # First answer token ends reasoning.
                                    if (
                                        _reasoning_started_at is not None
                                        and not _reasoning_summary_emitted
                                    ):
                                        _reasoning_summary_emitted = True
                                        _summary = _reasoning_summary_event(_reasoning_started_at)
                                        if _suppress_visible_output:
                                            _deferred_reasoning_summary = _summary
                                        else:
                                            yield _summary
                                    has_content_tokens = True
                                    content_accum += token

                                    if detect_state == _S_DRAINING:
                                        # Stream the drained TEXT call to a provisional card, gated
                                        # on name sniff and size.
                                        if not has_structured_tc and _text_args_call_start >= 0:
                                            if not _text_args_id:
                                                _call_text = content_accum[_text_args_call_start:]
                                                _sniffed = _sniff_text_tool_name(
                                                    _call_text, _enabled_tool_names
                                                )
                                                # Structured-path rule: gated calls
                                                # stream only from a text-preview card.
                                                if (
                                                    _sniffed
                                                    and not (
                                                        _confirm_gated_iteration
                                                        and (
                                                            permission_mode != "off"
                                                            or tool_call_may_prompt(
                                                                confirm_tool_calls = True,
                                                                bypass_permissions = False,
                                                                permission_mode = "off",
                                                                name = _sniffed,
                                                                sandbox_level = sandbox_level,
                                                                session_id = session_id,
                                                            )
                                                        )
                                                        and not has_text_only_provisional_card(
                                                            _sniffed
                                                        )
                                                    )
                                                    and (
                                                        _sniffed == "render_html"
                                                        or len(_call_text)
                                                        >= _PROVISIONAL_ARGS_MIN_CHARS
                                                    )
                                                ):
                                                    _text_args_id = "call_0"
                                                    _text_args_name = _sniffed
                                                    if (
                                                        _text_args_id
                                                        not in provisional_started_tool_calls
                                                    ):
                                                        provisional_started_tool_calls[
                                                            _text_args_id
                                                        ] = _sniffed
                                                        yield {
                                                            "type": "tool_start",
                                                            "tool_name": _sniffed,
                                                            "tool_call_id": _text_args_id,
                                                            "arguments": {},
                                                            "provenance": provisional_tool_provenance(
                                                                _sniffed
                                                            ),
                                                        }
                                                    yield {
                                                        "type": "tool_args",
                                                        "tool_call_id": _text_args_id,
                                                        "tool_name": _sniffed,
                                                        "text": _call_text,
                                                    }
                                                    _text_args_streamed_upto = len(content_accum)
                                            elif len(content_accum) > _text_args_streamed_upto:
                                                yield {
                                                    "type": "tool_args",
                                                    "tool_call_id": _text_args_id,
                                                    "tool_name": _text_args_name,
                                                    "text": content_accum[
                                                        _text_args_streamed_upto:
                                                    ],
                                                }
                                                _text_args_streamed_upto = len(content_accum)

                                    elif detect_state == _S_STREAMING:
                                        if in_thinking:
                                            cumulative_display += "</think>"
                                            in_thinking = False
                                            _prov_entry = None
                                        cumulative_display += token
                                        cleaned = _strip_tool_markup_streaming(cumulative_display)
                                        # Hold a trailing bare tool name (split rehearsal) until [ARGS] arrives; later prose or stream end releases it.
                                        _hold = _held_rehearsal_tail_len(cleaned, _detect_tools)
                                        _emit = (
                                            cleaned[: len(cleaned) - _hold] if _hold else cleaned
                                        )
                                        if len(_emit) > len(_last_emitted):
                                            _last_emitted = _emit
                                            if not _suppress_visible_output:
                                                yield {
                                                    "type": "content",
                                                    "text": _emit,
                                                }

                                    elif detect_state == _S_BUFFERING:
                                        content_buffer += token
                                        stripped_buf = content_buffer.lstrip()
                                        if not stripped_buf:
                                            continue

                                        # Bracket tags arrive mid-buffer, so substring-check too;
                                        # ``[ARGS]`` counts only as a regex-matched NAME[ARGS].
                                        is_prefix = False
                                        is_match = False
                                        for sig in _tool_xml_signals:
                                            if stripped_buf.startswith(sig):
                                                is_match = True
                                                break
                                            if sig.startswith(stripped_buf):
                                                is_prefix = True
                                                break
                                            if sig == "[ARGS]":
                                                # Only active NAME[ARGS] counts; inactive-name prose is gated out, not drained/parsed
                                                if (
                                                    _gguf_rehearsal_signal_pos(
                                                        stripped_buf, _detect_tools
                                                    )
                                                    >= 0
                                                ):
                                                    is_match = True
                                                    break
                                            elif sig.startswith("[") and sig in stripped_buf:
                                                is_match = True
                                                break

                                        # Split rehearsal: hold the bare name until its [ARGS] arrives and matches above.
                                        is_rehearsal_prefix = False
                                        if (
                                            not is_match
                                            and not is_prefix
                                            and _is_rehearsal_prefix(stripped_buf, _detect_tools)
                                        ):
                                            is_prefix = True
                                            is_rehearsal_prefix = True

                                        # Signal-less call shapes: Llama-3.2 bare JSON and Gemma call:NAME{...}.
                                        _hold_buffer = False
                                        # Whole buffer is the call (no visible prefix) -- drain silently.
                                        _drain_silently = False
                                        if not is_match and not is_prefix:
                                            _bare = strip_llama3_leading_sentinels(stripped_buf)
                                            if _bare.startswith("{"):
                                                if _balanced_brace_end(_bare, 0) is None:
                                                    if len(stripped_buf) < _MAX_BARE_JSON_BUFFER:
                                                        _hold_buffer = True
                                                    elif _looks_like_enabled_bare_json(
                                                        _bare, _enabled_tool_names
                                                    ):
                                                        # Oversized, still-open enabled call: drain, don't leak; a giant ordinary JSON answer still streams.
                                                        _drain_silently = True
                                                elif self._parse_tool_calls_from_text(
                                                    content_buffer,
                                                    allow_incomplete = auto_heal_tool_calls,
                                                    enabled_tool_names = _enabled_tool_names,
                                                ):
                                                    _drain_silently = True
                                                elif blocked_bare_json_chain_may_continue(
                                                    content_buffer, _enabled_tool_names
                                                ):
                                                    if len(stripped_buf) < _MAX_BARE_JSON_BUFFER:
                                                        _hold_buffer = True
                                                    else:
                                                        # Fail closed rather than expose text a later peer
                                                        # could make executable.
                                                        _drain_silently = True
                                            elif (
                                                "call:".startswith(stripped_buf)
                                                or _GEMMA_BARE_TC_PREFIX_RE.match(stripped_buf)
                                                is not None
                                                or _gemma_lead_promotable(stripped_buf)
                                                or blocked_gemma_chain_may_continue(
                                                    stripped_buf, _enabled_tool_names
                                                )
                                            ):
                                                # Whitespace-tolerant like the parser; a name this gate rejects streams as prose.
                                                if _gemma_lead_promotable(stripped_buf):
                                                    _drain_silently = True
                                                elif blocked_gemma_chain_may_continue(
                                                    stripped_buf, _enabled_tool_names
                                                ):
                                                    # A promotable peer behind a blocked call must not
                                                    # stream first.
                                                    if self._parse_tool_calls_from_text(
                                                        content_buffer,
                                                        allow_incomplete = auto_heal_tool_calls,
                                                        enabled_tool_names = _enabled_tool_names,
                                                    ):
                                                        _drain_silently = True
                                                    elif len(stripped_buf) < _MAX_BARE_JSON_BUFFER:
                                                        _hold_buffer = True
                                                    else:
                                                        _drain_silently = True
                                                elif len(stripped_buf) < _MAX_BUFFER_CHARS:
                                                    _hold_buffer = True

                                        if _drain_silently:
                                            # The buffered content IS the call, drained unyielded; a live <think> prefix is separate: close it.
                                            detect_state = _S_DRAINING
                                            # Call text begins at the held buffer (live arg display only; UI extracts the code).
                                            _text_args_call_start = len(content_accum) - len(
                                                content_buffer
                                            )
                                            if _close_streamed_think():
                                                yield {
                                                    "type": "content",
                                                    "text": cumulative_display,
                                                }
                                        elif is_match:
                                            # Flush visible prefix before DRAINING so the route sends it before tool_start; use the final strip, a streaming one would leak markup.
                                            _flush_reasoning_and_buffer()
                                            cleaned = _strip_tool_markup(
                                                cumulative_display,
                                                final = True,
                                                force = True,
                                            )
                                            if len(cleaned) > len(_last_emitted):
                                                _last_emitted = cleaned
                                                if not _suppress_visible_output:
                                                    yield {
                                                        "type": "content",
                                                        "text": cleaned,
                                                    }
                                            detect_state = _S_DRAINING
                                            # Live-arg display starts at the held buffer (visible prefix flushed above; UI extracts the code).
                                            _text_args_call_start = len(content_accum) - len(
                                                content_buffer
                                            )
                                        elif _hold_buffer or (
                                            is_prefix
                                            and (
                                                is_rehearsal_prefix
                                                or len(stripped_buf) < _MAX_BUFFER_CHARS
                                            )
                                        ):
                                            # Rehearsal prefix is self-bounded; the cap must not cut
                                            # long MCP names.
                                            pass  # keep buffering
                                        else:
                                            # Not a tool -- flush buffer
                                            detect_state = _S_STREAMING
                                            # Flush reasoning accumulated during BUFFERING.
                                            _flush_reasoning_and_buffer()
                                            cleaned = _strip_tool_markup(
                                                cumulative_display,
                                            )
                                            # Same trailing-name hold as STREAMING, for this first flush out of BUFFERING.
                                            _hold = _held_rehearsal_tail_len(cleaned, _detect_tools)
                                            _emit = (
                                                cleaned[: len(cleaned) - _hold]
                                                if _hold
                                                else cleaned
                                            )
                                            if len(_emit) > len(_last_emitted):
                                                _last_emitted = _emit
                                                if not _suppress_visible_output:
                                                    yield {
                                                        "type": "content",
                                                        "text": _emit,
                                                    }

                            except json.JSONDecodeError:
                                logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                        if _stream_done:
                            break  # exit outer for

                # ── Resolve BUFFERING at stream end ──
                if detect_state == _S_BUFFERING:
                    stripped_buf = content_buffer.lstrip()
                    # Bare JSON has no XML signal; route to DRAINING or it is flushed raw.
                    _bare_eos = strip_llama3_leading_sentinels(stripped_buf)
                    # Gate on enabled names so an ordinary JSON answer isn't routed to DRAINING and dropped.
                    _is_bare_tc = bool(active_tools) and _looks_like_enabled_bare_json(
                        _bare_eos, _enabled_tool_names
                    )
                    if stripped_buf and _gguf_has_genuine_tool_signal(
                        stripped_buf, _tool_xml_signals, _detect_tools
                    ):
                        detect_state = _S_DRAINING
                    elif _is_bare_tc:
                        detect_state = _S_DRAINING
                    elif content_accum or reasoning_accum:
                        detect_state = _S_STREAMING
                        if content_buffer:
                            # Flush reasoning first.
                            _flush_reasoning_and_buffer()
                            if not _suppress_visible_output:
                                yield {
                                    "type": "content",
                                    "text": _strip_tool_markup(
                                        cumulative_display,
                                        final = True,
                                    ),
                                }
                        elif reasoning_accum and not has_content_tokens:
                            # Reasoning-only reply: show as main response, like the no-tool path.
                            if _reasoning_started_at is not None and not _reasoning_summary_emitted:
                                _reasoning_summary_emitted = True
                                _summary = _reasoning_summary_event(_reasoning_started_at)
                                if _suppress_visible_output:
                                    _deferred_reasoning_summary = _summary
                                else:
                                    yield _summary
                            cumulative_display = _finalize_reasoning_only_cumulative(
                                cumulative_display,
                                reasoning_accum,
                                _iter_finish_reason,
                                promote_reasoning_only,
                                _resumed_reasoning,
                            )
                            _prov_entry = None
                            if not _suppress_visible_output:
                                yield {
                                    "type": "content",
                                    "text": cumulative_display,
                                }
                    else:
                        # A leading `{` is an ordinary JSON answer; other partial markup is dropped.
                        _held = strip_llama3_leading_sentinels(content_buffer.lstrip())
                        if _held.startswith("{") and not _suppress_visible_output:
                            yield {"type": "content", "text": _held}
                        return

                # ── STREAMING path: no tool call ──
                if detect_state == _S_STREAMING:
                    # Safety net: re-parse unconditionally; bare-JSON and Gemma calls have no XML signal.
                    _safety_tc = self._parse_tool_calls_from_text(
                        content_accum,
                        allow_incomplete = auto_heal_tool_calls,
                        enabled_tool_names = _enabled_tool_names,
                    )
                    if not _safety_tc:
                        # ── Re-prompt on plan-without-action ──
                        # Classify only what the user sees.
                        _visible_raw = content_accum.strip()
                        _visible = (
                            _strip_tool_markup(content_accum, final = True).strip()
                            if _visible_raw
                            else ""
                        )
                        _reasoning = reasoning_accum.strip()
                        # Magistral-style bracketed reasoning is stripped whole; fall back to raw.
                        _stripped = _visible or _reasoning or _visible_raw
                        # Thinking rendered as <think> in CONTENT stays in _visible; a fence inside one was never shown.
                        _visible_answer = _text_outside_think(_visible).strip()
                        # On the Anthropic path reasoning stays a thinking block, unseen.
                        _reasoning_shown = (
                            not has_content_tokens
                            and promote_reasoning_only
                            and _iter_finish_reason != "length"
                        )
                        _artifact_text = (
                            _visible_answer
                            if _visible_answer
                            else (_reasoning if _reasoning_shown else "")
                        )
                        # A plan only thought is not announced; _stripped stays whole, replayed and compared for repeats.
                        _intent_text = _visible_answer if _visible_answer else _stripped

                        # ── Continue an answer the window cut in half ──
                        # Answer cut by the window: continue it via continue_final_message so it
                        # resumes mid-token.
                        _partial_answer = content_accum.strip()
                        if (
                            _iter_finish_reason == "length"
                            and _partial_answer
                            and not tool_calls_acc
                            and _length_continuations < _MAX_LENGTH_CONTINUATIONS
                        ):
                            # Refuse to continue repetitive output, or the echo is stitched into the answer.
                            if is_repetition_dominated(_partial_answer):
                                logger.info(
                                    "Truncated answer is repetition-dominated (%d chars); "
                                    "keeping it as-is rather than continuing an echo",
                                    len(_partial_answer),
                                )
                            else:
                                _length_continuations += 1
                                logger.info(
                                    "Answer cut off at %d chars; continuing %d/%d",
                                    len(_partial_answer),
                                    _length_continuations,
                                    _MAX_LENGTH_CONTINUATIONS,
                                )
                                # Candidate first: an unservable retry must not spend conversation or usage.
                                _cand_c = list(conversation)
                                append_assistant_turn(
                                    _cand_c,
                                    # Unstripped, so the replayed prefix matches the streamed text exactly.
                                    {"role": "assistant", "content": content_accum},
                                    continue_final_message = True,
                                )
                                _fu_c = (
                                    _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                                )
                                # A `length` stop spent the whole allowance; charge that when usage is missing.
                                _spent_c = _fu_c.get("completion_tokens", 0) or int(
                                    payload.get("max_tokens") or 0
                                )
                                _cap_left_c = _loop_budget_left(_spent_c)
                                _fits_c = _cap_left_c != 0 and _loop_continuation_fits(
                                    _cand_c, safe_tools, _reasoning_kw, True
                                )
                                if _cap_left_c != 0 and not _fits_c:
                                    # Like the thought continuation: abandoning ends the turn, so preflight never drops older exchanges.
                                    _evicted_c = _evict_until_it_fits(
                                        _cand_c, safe_tools, _reasoning_kw, True
                                    )
                                    if _evicted_c is not None and _loop_continuation_fits(
                                        _evicted_c, safe_tools, _reasoning_kw, True
                                    ):
                                        _cand_c = _evicted_c
                                        _fits_c = True
                                if _cap_left_c == 0 or not _fits_c:
                                    logger.info(
                                        "Not continuing the answer: %s",
                                        "the caller's output cap is spent"
                                        if _cap_left_c == 0
                                        else "the retry prompt would not be served",
                                    )
                                    _length_continuations -= 1
                                else:
                                    conversation[:] = _cand_c
                                    continue_final_message = True
                                    if _cap_left_c is not None:
                                        _continuation_max_tokens = _cap_left_c
                                    _accumulated_completion_tokens += _fu_c.get(
                                        "completion_tokens", 0
                                    )
                                    _it_c = _iter_timings or {}
                                    _accumulated_predicted_ms += _it_c.get("predicted_ms", 0)
                                    _accumulated_predicted_n += _it_c.get("predicted_n", 0)
                                    if _continuation_credits < _MAX_CONTINUATION_CREDITS:
                                        _continuation_credits += 1
                                    yield {"type": "status", "text": ""}
                                    yield {
                                        "type": "status",
                                        "text": _CONTINUE_TRUNCATED_ANSWER_STATUS,
                                    }
                                    continue

                        # ── Continue a turn that ended inside its own thought ──
                        # Ahead of the re-prompt, which only handles SHORT responses.
                        if (
                            _iter_finish_reason == "length"
                            and not content_accum.strip()
                            and reasoning_accum.strip()
                            and _length_continuations < _MAX_LENGTH_CONTINUATIONS
                        ):
                            _length_continuations += 1
                            # Thinking off, or the retry fails the same way.
                            _effective_enable_thinking = False
                            _effective_reasoning_effort = None
                            logger.info(
                                "Turn ended inside its reasoning (%d chars, finish_reason "
                                "length); continuing %d/%d with thinking off",
                                len(reasoning_accum),
                                _length_continuations,
                                _MAX_LENGTH_CONTINUATIONS,
                            )
                            # Check fit and cap: the recovery request must not be rejected.
                            _cand_l = list(conversation)
                            append_assistant_turn(
                                _cand_l,
                                {
                                    "role": "assistant",
                                    # The stream holds only the resumed thought's tail.
                                    "content": _unfinished_thought_progress(
                                        _resumed_reasoning + reasoning_accum
                                    ),
                                },
                                continue_final_message = continue_final_message,
                            )
                            if _resumed_reasoning:
                                _cand_l[-1].pop("reasoning_content", None)
                            _cand_l.append(
                                {
                                    "role": "user",
                                    "content": _continue_after_length_message(),
                                }
                            )
                            _fu_l = _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                            _spent_l = _fu_l.get("completion_tokens", 0) or int(
                                payload.get("max_tokens") or 0
                            )
                            _cap_left_l = _loop_budget_left(_spent_l)
                            # Recompute kwargs with thinking off so admission prices the real request.
                            _off_kw_l = self._request_reasoning_kwargs(
                                False, None, preserve_thinking
                            )
                            # Counted ONCE normally: eviction below runs only on a candidate already found too large.
                            _fits_l = _cap_left_l != 0 and _loop_continuation_fits(
                                _cand_l, safe_tools, _off_kw_l
                            )
                            if _cap_left_l != 0 and not _fits_l:
                                # Try eviction: refusing here ends the turn.
                                _evicted_l = _evict_until_it_fits(_cand_l, safe_tools, _off_kw_l)
                                if _evicted_l is not None and _loop_continuation_fits(
                                    _evicted_l, safe_tools, _off_kw_l
                                ):
                                    _cand_l = _evicted_l
                                    _fits_l = True
                            if _cap_left_l == 0 or not _fits_l:
                                logger.info(
                                    "Not continuing the thought: %s",
                                    "the caller's output cap is spent"
                                    if _cap_left_l == 0
                                    else "the retry prompt would not be served",
                                )
                                # Record whether Max Tokens or the window ran out, so the message
                                # names the right lever.
                                _reasoning_cap_spent = _cap_left_l == 0
                                _length_continuations -= 1
                                _effective_enable_thinking = enable_thinking
                                _effective_reasoning_effort = reasoning_effort
                            else:
                                conversation[:] = _cand_l
                                if _cap_left_l is not None:
                                    _continuation_max_tokens = _cap_left_l
                                _accumulated_completion_tokens += _fu_l.get("completion_tokens", 0)
                                _it_l = _iter_timings or {}
                                _accumulated_predicted_ms += _it_l.get("predicted_ms", 0)
                                _accumulated_predicted_n += _it_l.get("predicted_n", 0)
                                # Blank status first so the route resets its text cursor.
                                if _continuation_credits < _MAX_CONTINUATION_CREDITS:
                                    _continuation_credits += 1
                                yield {"type": "status", "text": ""}
                                yield {"type": "status", "text": _CONTINUE_AFTER_LENGTH_STATUS}
                                continue

                        if (
                            _iter_finish_reason == "length"
                            and not content_accum.strip()
                            and reasoning_accum.strip()
                        ):
                            # Allowance spent: say so rather than return an empty message.
                            logger.info(
                                "Reasoning exhausted the window %d times; giving up with "
                                "an explanation",
                                _length_continuations,
                            )
                            # Re-read the cap: the last attempt may have exhausted Max Tokens.
                            _fu_g = _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                            _spent_g = _fu_g.get("completion_tokens", 0) or int(
                                payload.get("max_tokens") or 0
                            )
                            yield {"type": "status", "text": ""}
                            yield {
                                "type": "content",
                                "text": (
                                    _reasoning_cap_spent_message(max_tokens)
                                    if _reasoning_cap_spent or _loop_budget_left(_spent_g) == 0
                                    else _thinking_exhausted_message(self._effective_context_length)
                                ),
                            }
                            _meta = _build_metadata_event(
                                _iter_usage, _iter_timings, _iter_finish_reason
                            )
                            if _meta is not None:
                                yield _meta
                            return

                        _render_html_already_done_intent = _tool_succeeded(
                            "render_html"
                        ) and re.search(
                            r"(?i)\brender[_\s-]?html\b",
                            _stripped,
                        )
                        # Post-tool stall gets one nudge (retries re-run tools). _auto counts RAG autoinject.
                        _already_acted = bool(_auto) or any(
                            record.executed for record in tool_controller.history
                        )
                        if _already_acted:
                            _reprompt_used, _reprompt_cap = _post_tool_reprompts, 1
                        else:
                            _reprompt_used, _reprompt_cap = _reprompt_count, _MAX_REPROMPTS
                        # None follows the shared process default; explicit values win.
                        if (
                            auto_heal_tool_calls
                            and _nudge_enabled(nudge_tool_calls)
                            and active_tools
                            and not _render_html_already_done_intent
                            and _reprompt_used < _reprompt_cap
                            and not _is_reprompt_repeat(_stripped, _last_reprompt_text)
                            # _stripped is what gets replayed; intent text may be a short tail.
                            and len(_stripped) < _REPROMPT_MAX_CHARS
                            and _is_short_intent_without_action(_intent_text)
                            and not (_artifact_text and _has_answer_artifact(_artifact_text))
                        ):
                            _reprompt_count += 1
                            if _already_acted:
                                _post_tool_reprompts += 1
                            _last_reprompt_text = _stripped
                            logger.info(
                                f"Re-prompt {_reprompt_used + 1}/{_reprompt_cap}: "
                                f"model responded without calling tools "
                                f"({len(_stripped)} chars)"
                            )
                            # Merge into a resumed partial to keep role alternation.
                            append_assistant_turn(
                                conversation,
                                {"role": "assistant", "content": _stripped},
                                continue_final_message = continue_final_message,
                            )
                            available_tool_names = [
                                (tool.get("function") or {}).get("name")
                                for tool in active_tools
                                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                            ]
                            available_tool_names = [name for name in available_tool_names if name]
                            tool_hint = " or ".join(available_tool_names) or "an available tool"
                            _forced_tool_call_pending = True
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": _reprompt_to_act_message(tool_hint),
                                }
                            )
                            # Accumulate tokens and timing from this iteration.
                            _fu_r = _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                            _accumulated_completion_tokens += _fu_r.get("completion_tokens", 0)
                            _it_r = _iter_timings or {}
                            _accumulated_predicted_ms += _it_r.get("predicted_ms", 0)
                            _accumulated_predicted_n += _it_r.get("predicted_n", 0)
                            # Blank first (the route resets its cursor only on empty status).
                            yield {"type": "status", "text": ""}
                            yield {"type": "status", "text": _NUDGE_TOOL_CALLS_STATUS}
                            continue

                        if _forced_tool_call_pending:
                            _forced_tool_call_pending = False
                            if not _should_suppress_forced_no_tool_output(
                                _stripped,
                                _last_reprompt_text,
                            ):
                                if cumulative_display:
                                    forced_visible_text = _strip_tool_markup(
                                        cumulative_display,
                                        final = True,
                                    )
                                elif content_accum:
                                    forced_visible_text = _strip_tool_markup(
                                        content_accum,
                                        final = True,
                                    )
                                else:
                                    forced_visible_text = reasoning_accum
                                if forced_visible_text:
                                    yield {
                                        "type": "content",
                                        "text": forced_visible_text,
                                    }
                                    if _deferred_reasoning_summary is not None:
                                        yield _deferred_reasoning_summary
                        elif not _suppress_visible_output:
                            # Plain answer with no [ARGS] following: the held rehearsal tail is real prose, so release it.
                            _final_clean = _strip_tool_markup_streaming(cumulative_display)
                            if len(_final_clean) > len(_last_emitted):
                                yield {"type": "content", "text": _final_clean}

                        # Content was already streamed.  Yield metadata.
                        yield {"type": "status", "text": ""}
                        _cut = _quote_cut_event(
                            reasoning_accum,
                            _visible,
                            _iter_finish_reason,
                            promote_reasoning_only,
                        )
                        if _cut is not None:
                            yield _cut
                        _meta = _build_metadata_event(
                            _iter_usage, _iter_timings, _iter_finish_reason
                        )
                        if _meta is not None:
                            yield _meta
                        return

                    # Safety net caught tool XML -- treat as tool call.
                    tool_calls = _safety_tc
                    content_text = _strip_tool_markup(
                        content_accum,
                        final = True,
                        force = True,
                    )
                    logger.info(
                        f"Safety net: parsed {len(tool_calls)} tool call(s) from streamed content"
                    )
                else:
                    # ── DRAINING path: assemble tool_calls ──
                    tool_calls = None
                    content_text = content_accum
                    if has_structured_tc:
                        # Drop incomplete fragments. First-seen order so forked calls run last.
                        tool_calls = [
                            call
                            for call in tool_calls_acc.values()
                            if (call.get("function", {}).get("name", "").strip())
                        ] or None
                    if not tool_calls:
                        # Unconditional: bare-JSON / Gemma calls carry no XML signal.
                        tool_calls = self._parse_tool_calls_from_text(
                            content_accum,
                            allow_incomplete = auto_heal_tool_calls,
                            enabled_tool_names = _enabled_tool_names,
                        )
                    if tool_calls and not has_structured_tc:
                        content_text = _strip_tool_markup(
                            content_text,
                            final = True,
                            force = True,
                        )
                        # Also drop a leading bare-JSON call so it isn't replayed as text.
                        content_text = strip_leading_bare_json_call(
                            content_text, _enabled_tool_names
                        )
                    if tool_calls:
                        logger.info(
                            f"Parsed {len(tool_calls)} tool call(s) from "
                            f"{'structured delta' if has_structured_tc else 'content text'}"
                        )
                    if not tool_calls:
                        # False positive: close provisional cards or they spin forever.
                        for _pid, _pname in provisional_started_tool_calls.items():
                            if _pid not in resolved_provisional_tool_call_ids:
                                resolved_provisional_tool_call_ids.add(_pid)
                                yield {
                                    "type": "tool_end",
                                    "tool_name": _pname,
                                    "tool_call_id": _pid,
                                    "result": "",
                                    "provenance": provisional_tool_provenance(_pname),
                                }
                        # Merge metrics from prior tool iterations so they aren't dropped.
                        yield {"type": "status", "text": ""}
                        if content_accum:
                            # Strip leaked tool-call XML before yielding.
                            content_accum = _strip_tool_markup(content_accum, final = True)
                        # With Auto-Heal on, drop a truncated ENABLED-tool JSON fragment; off keeps it.
                        if content_accum and active_tools and auto_heal_tool_calls:
                            content_accum = strip_leading_bare_json_call(
                                content_accum, _enabled_tool_names
                            )
                        if content_accum:
                            yield {"type": "content", "text": content_accum}
                        _meta = _build_metadata_event(
                            _iter_usage, _iter_timings, _iter_finish_reason
                        )
                        if _meta is not None:
                            yield _meta
                        return

                # ── Execute tool calls ──
                # Progress made: reset stall counters and restore thinking.
                _length_continuations = 0
                _effective_enable_thinking = enable_thinking
                _effective_reasoning_effort = reasoning_effort
                # Captured before the reset: later records must merge into a resumed partial.
                _merge_into_partial = continue_final_message
                continue_final_message = _caller_continue_final_message
                _accumulated_completion_tokens += (
                    _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                ).get("completion_tokens", 0)
                _it = _iter_timings or {}
                _accumulated_predicted_ms += _it.get("predicted_ms", 0)
                _accumulated_predicted_n += _it.get("predicted_n", 0)
                # Refreshes the context meter from the usage llama-server already reported (no
                # recount). Sent once the first tool is running, so it never delays a tool.
                _pending_context_usage = _context_usage_event(_iter_usage, _iter_timings)

                # Dedupe and cap TEXTUAL fallback calls (see _MAX_TOOL_CALLS_PER_TURN).
                _over_cap: list = []
                if tool_calls and not has_structured_tc and len(tool_calls) > 1:
                    _seen_keys: set = set()
                    _last_workspace_key = None
                    # One rerun per piece of new work: `test, edit A, test, edit B, test` keeps
                    # every test, while `read, edit, read, edit` stops replaying and cannot
                    # fill the cap.
                    _novel_kept = 0
                    _novel_at_last_keep: dict = {}
                    _deduped: list = []
                    if not deduplicate_tool_calls:
                        _deduped = tool_calls[:_MAX_TOOL_CALLS_PER_TURN]
                        _over_cap = tool_calls[_MAX_TOOL_CALLS_PER_TURN:]
                    for _tc in tool_calls if deduplicate_tool_calls else ():
                        _fn = _tc.get("function", {}) or {}
                        _key = (_fn.get("name", ""), str(_fn.get("arguments", "")))
                        if _reruns_after_new_work(_fn.get("name")):
                            if _key == _last_workspace_key:
                                continue
                            if _key in _seen_keys:
                                if _novel_kept <= _novel_at_last_keep.get(_key, 0):
                                    continue
                            elif _fn.get("name") not in _WORKSPACE_READ_TOOLS:
                                _novel_kept += 1
                            _novel_at_last_keep[_key] = _novel_kept
                            _last_workspace_key = _key
                        # A browser call acts on the page as it is then: two scrolls are two scrolls.
                        # Two identical generate_image calls are two images.
                        elif _key in _seen_keys and _fn.get("name") not in REPEATABLE_TOOL_NAMES:
                            continue
                        _seen_keys.add(_key)
                        if len(_deduped) < _MAX_TOOL_CALLS_PER_TURN:
                            _deduped.append(_tc)
                        else:
                            _over_cap.append(_tc)
                    if len(_deduped) + len(_over_cap) != len(tool_calls):
                        logger.info(
                            "GGUF textual fallback: collapsed %d repeated tool call(s) "
                            "in one turn to %d",
                            len(tool_calls),
                            len(_deduped) + len(_over_cap),
                        )
                    if _over_cap:
                        logger.info(
                            "GGUF textual fallback: skipped %d tool call(s) over the "
                            "per-turn limit of %d",
                            len(_over_cap),
                            _MAX_TOOL_CALLS_PER_TURN,
                        )
                    tool_calls = _deduped

                # Truncate before building assistant_msg so extra calls never execute.
                if disable_parallel_tool_use and tool_calls and len(tool_calls) > 1:
                    tool_calls = tool_calls[:1]
                    _over_cap = []

                assistant_msg: dict = {"role": "assistant", "content": content_text}
                if reasoning_accum.strip():
                    assistant_msg["reasoning_content"] = reasoning_accum
                assistant_appended = False
                # Collect no-op feedback and flush it after the batch, so a
                # suppressed call never drops a later parallel call.
                deferred_noop_msgs: list = []
                # Which tools the no-ops concerned, so the flush below can tell a same-tool trailing result.
                deferred_noop_tools: set = set()
                # Per result, so a parallel batch is not squeezed into one result's worth of images.
                batch_mcp_images: list = []
                # Where this batch's results start, for the image turn\'s wording.
                batch_conversation_start = len(conversation)

                # The text-path provisional card uses the parser's default id ("call_0");
                # a Mistral-style call carries its own id and would open a duplicate. Reuse
                # the card's id for the first matching call (same tool name) to reconcile.
                _text_provisional_id = _text_args_id if not has_structured_tc else ""

                for _call_index, tc in enumerate(tool_calls or []):
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    if (
                        _text_provisional_id
                        and _text_provisional_id in provisional_started_tool_calls
                        and _text_provisional_id not in resolved_provisional_tool_call_ids
                        and tc.get("id") not in provisional_started_tool_calls
                        and provisional_started_tool_calls[_text_provisional_id] == tool_name
                    ):
                        tc = {**tc, "id": _text_provisional_id}
                    provisional_match = tc.get("id") in provisional_started_tool_calls
                    decision = tool_controller.prepare_call(
                        tc,
                        forced = _forced_tool_call_pending,
                        provisional = provisional_match,
                        allowed_tool_names = {forced_name} if forced_name else None,
                    )
                    image_share = None
                    if decision.should_execute and mcp_image is not None:
                        image_share = mcp_image_share(
                            decision.tool_name, decision.arguments, mcp_image
                        )
                        if image_share is not None:
                            decision = tool_controller.reprepare_call(
                                decision,
                                forced = _forced_tool_call_pending,
                                provisional = provisional_match,
                                allowed_tool_names = {forced_name} if forced_name else None,
                            )
                    # The frontend groups a parallel batch into one assistant message by this.
                    decision.provenance["round_id"] = iteration

                    if not decision.should_execute:
                        if provisional_match:
                            # Close the provisional card when the call becomes an internal no-op.
                            resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                            yield {
                                "type": "tool_end",
                                "tool_name": decision.tool_name,
                                "tool_call_id": decision.tool_call_id,
                                "result": "",
                                "provenance": decision.provenance,
                            }
                        completion = tool_controller.record_noop(decision)
                        deferred_noop_msgs.append(completion.model_message())
                        deferred_noop_tools.add(decision.tool_name)
                        if _forced_tool_call_pending:
                            _forced_tool_call_pending = False
                        logger.info(
                            "Suppressed local GGUF tool call as internal no-op: "
                            f"action={decision.action} tool={decision.tool_name}"
                        )
                        continue

                    if not assistant_appended:
                        assistant_msg["tool_calls"] = [decision.as_assistant_tool_call()]
                        # Merges into a resumed partial, so a continued turn that calls a tool stays one assistant message.
                        append_assistant_turn(
                            conversation,
                            assistant_msg,
                            continue_final_message = _merge_into_partial,
                        )
                        assistant_appended = True
                    else:
                        assistant_msg.setdefault("tool_calls", []).append(
                            decision.as_assistant_tool_call()
                        )

                    # Bypass wins here too, so a direct internal caller with both
                    # flags never prompts. "auto" pauses only high-risk calls; "off"
                    # pauses only a high-risk python/terminal call without OS isolation.
                    needs_confirm = needs_tool_confirmation(
                        confirm_tool_calls = bool(confirm_tool_calls),
                        bypass_permissions = bypass_permissions,
                        permission_mode = permission_mode,
                        name = decision.tool_name,
                        arguments = decision.arguments,
                        is_high_risk = is_high_risk_tool_call,
                        never_needs = never_needs_approval,
                        sandbox_level = sandbox_level,
                        session_id = session_id,
                    )
                    # Sending the user's image always asks, whatever the permission mode.
                    needs_confirm = needs_confirm or image_share is not None
                    strict_isolation = requires_os_isolation(
                        confirm_tool_calls = bool(confirm_tool_calls),
                        bypass_permissions = bypass_permissions,
                        permission_mode = permission_mode,
                        name = decision.tool_name,
                        arguments = decision.arguments,
                        prompted = needs_confirm,
                        is_high_risk = is_high_risk_tool_call,
                        sandbox_level = sandbox_level,
                    )
                    approval_id = new_approval_id() if needs_confirm else ""
                    decision_slot = (
                        begin_tool_decision(session_id, approval_id) if needs_confirm else None
                    )
                    if needs_confirm:
                        note_host_access(approval_id, decision.tool_name, decision.arguments)
                    start_event = decision.tool_start_event()
                    start_event["approval_id"] = approval_id
                    start_event["awaiting_confirmation"] = needs_confirm
                    if image_share is not None:
                        start_event["image_disclosure"] = image_share["disclosure"]

                    try:
                        # Gated calls are not running yet; a counting badge while waiting on a human reads as a hang.
                        yield {
                            "type": "status",
                            "text": (
                                awaiting_approval_status(decision.tool_name)
                                if needs_confirm
                                else decision.status_text
                            ),
                        }
                        yield start_event
                        if needs_confirm and _pending_context_usage is not None:
                            # Nothing runs while a human decides, so the meter need not wait for it.
                            yield _pending_context_usage
                            _pending_context_usage = None

                        _decision = (
                            wait_tool_decision(
                                decision_slot,
                                approval_id,
                                cancel_event = cancel_event,
                            )
                            if decision_slot is not None
                            else None
                        )
                        # Read before decision_slot is dropped in the deny branch.
                        _decision_reason = decision_reason(decision_slot)
                        if _decision is not None and _decision != "deny":
                            # tool_approved first: the route forwards it before retaking the model slot.
                            yield decision.tool_approved_event(approval_id)
                            yield {"type": "status", "text": decision.status_text}
                        if _decision == "deny":
                            decision_slot = None
                            _forced_choice_resolved = True
                            resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                            # Unanswered approval is not a user decision; this text is the call's only account once buttons go.
                            _denied_text = (
                                TOOL_APPROVAL_EXPIRED_MESSAGE
                                if _decision_reason == DECISION_EXPIRED
                                else TOOL_REJECTED_MESSAGE
                            )
                            yield {
                                "type": "tool_end",
                                "tool_name": decision.tool_name,
                                "tool_call_id": decision.tool_call_id,
                                "result": _denied_text,
                                "provenance": decision.provenance,
                            }
                            denied_message = {
                                "role": "tool",
                                "name": decision.tool_name,
                                "content": _denied_text,
                            }
                            if decision.tool_call_id:
                                denied_message["tool_call_id"] = decision.tool_call_id
                                _tool_card_by_message[id(denied_message)] = decision.tool_call_id
                            conversation.append(denied_message)
                            # The denial renders the refused arguments, so compact them or the
                            # context overflows.
                            if decision.tool_call_id:
                                conversation[:] = compact_refused_tool_arguments(
                                    conversation, decision.tool_call_id
                                )
                                # Rebuilt, not mutated: rebind the handle or later calls land on a stale object.
                                for _dmsg in reversed(conversation):
                                    if _dmsg.get("role") != "assistant":
                                        continue
                                    if any(
                                        (_tc or {}).get("id") == decision.tool_call_id
                                        for _tc in (_dmsg.get("tool_calls") or [])
                                    ):
                                        assistant_msg = _dmsg
                                        break
                            if _forced_tool_call_pending:
                                _forced_tool_call_pending = False
                            continue
                        decision_slot = None
                    finally:
                        if decision_slot is not None:
                            abort_tool_decision(decision_slot, approval_id)

                    # Conservative estimate only decides whether to run the exact count, never whether to refuse; base64 is about 1 char per token.
                    _unservable_text: Optional[str] = None
                    # Set when the gate runs an oversized call rather than refusing, since compaction will reclaim room.
                    _compact_after_execution = False
                    # Exact count of the prompt compaction WILL produce, so the result budget below is priced against it.
                    _compacted_turn_tokens: Optional[int] = None
                    if self._effective_context_length:
                        # This iteration's cap, like every other sizing decision in it.
                        _room_target = prompt_budget(
                            self._effective_context_length, _iteration_max_tokens
                        )
                        # Cheap conservative estimate decides whether to do an exact count (base64
                        # ~1 char/token).
                        if (
                            estimate_messages_tokens_conservative(
                                self._countable_messages(conversation)
                            )
                            > 0.7 * _room_target
                        ):
                            # Templates render tool_calls only once a tool reply follows, so add an
                            # empty stand-in.
                            _probe_message: dict = {
                                "role": "tool",
                                "name": decision.tool_name,
                                "content": "",
                            }
                            if decision.tool_call_id:
                                _probe_message["tool_call_id"] = decision.tool_call_id
                            _compacted_calls = 0
                            for _attempt in range(2):
                                try:
                                    _turn_tokens = self.count_chat_tokens(
                                        neutralize_control_markup_in_messages(
                                            self._countable_messages(
                                                [*conversation, _probe_message]
                                            ),
                                            _markup_cache,
                                            self.markup_profile,
                                        ),
                                        None,
                                        safe_tools,
                                        strict = True,
                                        chat_template_kwargs = _reasoning_kw,
                                    )
                                except Exception:
                                    # Count failed: run it and let llama-server decide.
                                    logger.debug(
                                        "pre-execution fit: prompt count failed",
                                        exc_info = True,
                                    )
                                    break
                                if turn_is_servable(
                                    self._effective_context_length, max_tokens, _turn_tokens
                                ):
                                    break
                                if _attempt == 0:
                                    # Compact history first, then re-price against the real tokenizer instead of trusting the saving.
                                    _compacted, _n = compact_completed_tool_arguments(
                                        conversation, protect_last = 1
                                    )
                                    if _n:
                                        conversation[:] = _compacted
                                        _compacted_calls = _n
                                        logger.info(
                                            "Compacted arguments of %d completed tool call(s) "
                                            "to fit the context window",
                                            _n,
                                        )
                                        continue
                                # The refusal renders the call arguments, so compact them too or the next generation overflows the window; nothing ran, so nothing on disk reflects them.
                                if decision.tool_call_id:
                                    _as_if_run = compact_executed_call_arguments(
                                        [*conversation, _probe_message],
                                        decision.tool_call_id,
                                    )
                                    try:
                                        _after_tokens = self.count_chat_tokens(
                                            neutralize_control_markup_in_messages(
                                                self._countable_messages(_as_if_run),
                                                _markup_cache,
                                                self.markup_profile,
                                            ),
                                            None,
                                            safe_tools,
                                            strict = True,
                                            chat_template_kwargs = _reasoning_kw,
                                        )
                                    except Exception:
                                        _after_tokens = None
                                    if _after_tokens is not None and turn_is_servable(
                                        self._effective_context_length,
                                        max_tokens,
                                        _after_tokens,
                                    ):
                                        _compact_after_execution = True
                                        _compacted_turn_tokens = _after_tokens
                                        logger.info(
                                            "Running %s despite a %d-token turn: compacting "
                                            "its own arguments afterwards brings the next "
                                            "prompt to %d, so the file gets written",
                                            decision.tool_name,
                                            _turn_tokens,
                                            _after_tokens,
                                        )
                                        break
                                from core.inference import context_refusal  # noqa: PLC0415

                                _unservable_text = context_refusal.describe_unservable_tool_call(
                                    decision.tool_name,
                                    _turn_tokens,
                                    self._effective_context_length,
                                    compacted_calls = _compacted_calls,
                                )
                                logger.warning(
                                    "Refused %s before execution: prompt would be %d tokens "
                                    "against a %d-token window",
                                    decision.tool_name,
                                    _turn_tokens,
                                    self._effective_context_length,
                                )
                                break

                    if _unservable_text is not None:
                        # Same as the user-denied path: no new frontend case; the model reads a result, not a stalled stream.
                        resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                        yield {
                            "type": "tool_end",
                            "tool_name": decision.tool_name,
                            "tool_call_id": decision.tool_call_id,
                            "result": _unservable_text,
                            "provenance": decision.provenance,
                        }
                        _refused_message = {
                            "role": "tool",
                            "name": decision.tool_name,
                            "content": _unservable_text,
                        }
                        if decision.tool_call_id:
                            _refused_message["tool_call_id"] = decision.tool_call_id
                            _tool_card_by_message[id(_refused_message)] = decision.tool_call_id
                        conversation.append(_refused_message)
                        # The refusal renders the call's arguments, so compact them too.
                        _before_refused_compaction = len(json.dumps(conversation, default = str))
                        conversation[:] = compact_refused_tool_arguments(
                            conversation, decision.tool_call_id
                        )
                        _after_refused_compaction = len(json.dumps(conversation, default = str))
                        if _after_refused_compaction < _before_refused_compaction:
                            logger.info(
                                "Compacted refused %s arguments: %d -> %d chars",
                                decision.tool_name,
                                _before_refused_compaction,
                                _after_refused_compaction,
                            )
                        if _forced_tool_call_pending:
                            _forced_tool_call_pending = False
                        # Worker thread so output streams while the tool blocks; result is byte-identical to a direct call; _invoke_tool fills _last_result_budget for the log.
                        _forced_choice_resolved = True
                        # Compaction above rebuilt this assistant turn, so the handle the batch appends to is stale.
                        for _rmsg in reversed(conversation):
                            if _rmsg.get("role") != "assistant":
                                continue
                            if any(
                                (_tc or {}).get("id") == decision.tool_call_id
                                for _tc in (_rmsg.get("tool_calls") or [])
                            ):
                                assistant_msg = _rmsg
                                break
                        continue

                    _effective_timeout = None if tool_call_timeout >= 9999 else tool_call_timeout
                    # RAG: cap paraphrased KB re-searches that slip past the dup guard.
                    if (
                        decision.tool_name in RAG_SEARCH_TOOLS
                        and _kb_search_count >= RAG_MAX_SEARCHES_PER_TURN
                    ):
                        result = RAG_SEARCH_CAP_NUDGE
                    else:
                        # Execute in a worker thread so live stdout chunks and heartbeats
                        # stream while the tool blocks (the SSE route turns heartbeats into
                        # keepalives). Result is byte-identical to a direct call.
                        # Filled in by `_invoke_tool` below so the result log can report
                        # the budget that was actually handed over, from a scope that
                        # outlives the closure.
                        _last_result_budget: list = ["<not passed>"]

                        # Only a call the user answered: the executor lets it reach the host paths it names.
                        _host_access_approved = _decision not in (None, "deny")

                        def _invoke_tool(
                            _output_callback,
                            _decision = decision,
                            _approved = _host_access_approved,
                            _strict = strict_isolation,
                        ):
                            # Bound top_k to the room left: the result lands in the protected tool exchange, so an oversized top_k ends the turn in an unrecoverable context error.
                            kwargs = dict(
                                cancel_event = cancel_event,
                                timeout = _effective_timeout,
                                session_id = session_id,
                                thread_id = thread_id,
                                rag_scope = rag_scope,
                                disable_sandbox = bypass_permissions,
                            )
                            if website_policy:
                                kwargs["website_policy"] = website_policy
                            # Run unasked only because the OS sandbox was on: refuse if it is not any more.
                            if _strict and accepts_kwarg(execute_tool, "tool_execution_mode"):
                                kwargs["tool_execution_mode"] = "required"
                            elif runs_without_os_sandbox(
                                _decision.tool_name, sandbox_level
                            ) and accepts_kwarg(execute_tool, "tool_execution_mode"):
                                kwargs["tool_execution_mode"] = "software"
                            # Same branch the forced recall is filtered against, so a
                            # model-initiated search cannot reach a sibling response the
                            # forced recall correctly refused.
                            if accepts_kwarg(execute_tool, "conversation_branch"):
                                kwargs["conversation_branch"] = _extend_live_branch(conversation)
                            if _approved and accepts_kwarg(execute_tool, "host_access_approved"):
                                kwargs["host_access_approved"] = True
                            # Exact count with the catalogue, recomputed on every search: a cached count misses the loop's new exchanges; the offset is only the fallback.
                            if self._effective_context_length:
                                # Exact count with the catalogue, recomputed every call; the offset
                                # is only a fallback.
                                _exact_prompt_tokens: Optional[int] = None
                                # When compaction will run, price the result against the compacted prompt, or a large call gets a near-zero budget and its output is thrown away.
                                _size_probe: dict = {
                                    "role": "tool",
                                    "name": decision.tool_name,
                                    "content": "",
                                }
                                if decision.tool_call_id:
                                    _size_probe["tool_call_id"] = decision.tool_call_id
                                try:
                                    _exact_prompt_tokens = self.count_chat_tokens(
                                        neutralize_control_markup_in_messages(
                                            self._countable_messages([*conversation, _size_probe]),
                                            _markup_cache,
                                            self.markup_profile,
                                        ),
                                        None,
                                        safe_tools,
                                        strict = True,
                                        chat_template_kwargs = _reasoning_kw,
                                    )
                                except Exception:
                                    logger.debug(
                                        "recall budget: exact prompt count failed",
                                        exc_info = True,
                                    )
                                # When compaction will run, price against the compacted prompt.
                                _spent = (
                                    _compacted_turn_tokens
                                    if _compact_after_execution and _compacted_turn_tokens
                                    else _exact_prompt_tokens
                                    if _exact_prompt_tokens is not None
                                    else estimate_messages_tokens_dense(
                                        self._countable_messages(conversation)
                                    )
                                    + (
                                        _prompt_token_offset
                                        if _prompt_token_offset is not None
                                        else estimate_messages_tokens_dense(safe_tools or [])
                                    )
                                )
                                # What ANY result may add before the next prompt is over
                                # budget, for every tool and not just the retrieval ones.
                                # The old cap was a share of the WINDOW and never fell as
                                # the thread filled, so the last result before an overflow
                                # claimed as much as the first, and nothing downstream
                                # recovers: the fit protects the newest turn.
                                if accepts_kwarg(execute_tool, "result_budget_tokens"):
                                    # Split across the rest of the batch and their arguments.
                                    _pending = list(tool_calls or [])[_call_index + 1 :]
                                    _pending_msgs = [
                                        {
                                            "role": "assistant",
                                            "content": json.dumps(_call, default = str),
                                        }
                                        for _call in _pending
                                    ]
                                    # reserve the notice before sharing space among retained results.
                                    if _over_cap:
                                        _pending_msgs.append(
                                            tool_call_limit_nudge(
                                                _over_cap, _MAX_TOOL_CALLS_PER_TURN
                                            )
                                        )
                                    # A near-zero budget yields only a truncation notice the model reads as a failure and retries; compact first, since compaction reclaims that room.
                                    _pending_args = 0
                                    _pending_exact = True
                                    if _pending_msgs:
                                        try:
                                            _pending_args = self.count_chat_tokens(
                                                _pending_msgs, None, None, strict = True
                                            )
                                        except Exception:
                                            logger.debug(
                                                "result budget: pending call count failed",
                                                exc_info = True,
                                            )
                                            _pending_exact = False
                                            _pending_args = 2 * estimate_messages_tokens_dense(
                                                _pending_msgs
                                            )
                                    # This iteration's cap, not the caller's whole one. A
                                    # recovery turn with 100 of 1000 tokens left had the
                                    # result priced as if 1000 were still to come, which
                                    # reserves room away and can starve a read the request
                                    # had space for.
                                    _result_max_tokens = _iteration_max_tokens
                                    # Auto-compaction off: nothing fits the next prompt to
                                    # `prompt_budget`, whose quarter-window reply reserve
                                    # emptied every result from 75% of the window on (#13784).
                                    # Price to the reply floor the admission gate uses instead,
                                    # only on an exact count of a thread with no unpriced media:
                                    # the quarter reserve is what covers either error.
                                    if (
                                        context_overflow != "truncate_oldest"
                                        and (
                                            _exact_prompt_tokens is not None
                                            or (_compact_after_execution and _compacted_turn_tokens)
                                        )
                                        and _pending_exact
                                        and messages_without_unpriced_media(conversation)
                                        is conversation
                                    ):
                                        _ctx = self._effective_context_length
                                        _requested = (
                                            _iteration_max_tokens
                                            if _iteration_max_tokens and _iteration_max_tokens > 0
                                            else _ctx
                                        )
                                        _reserve = min(_requested, _reply_floor(_ctx))
                                        # Images join the thread after the whole batch, unpriced:
                                        # the ones already collected, and this call's own.
                                        _images = sum(map(len, batch_mcp_images))
                                        if (
                                            mcp_is_image_tool(decision.tool_name)
                                            and image_share is None
                                            and self.is_vision
                                        ):
                                            _images += MCP_MAX_MODEL_IMAGES
                                        if _images:
                                            _reserve += min(
                                                MCP_MAX_TOTAL_MODEL_IMAGES, _images
                                            ) * _mcp_image_tokens(self)
                                        _result_max_tokens = min(
                                            _reserve, _requested, max(1, _ctx // 4)
                                        )
                                    _result_budget = tool_result_budget(
                                        self._effective_context_length,
                                        _result_max_tokens,
                                        _spent + _pending_args,
                                    ) // (len(_pending) + 1)
                                    # A near-zero budget yields only a truncation notice; compact first.
                                    if (
                                        _result_budget < _MIN_USEFUL_RESULT_TOKENS
                                        and self._effective_context_length
                                    ):
                                        _roomier, _n_roomier = compact_completed_tool_arguments(
                                            conversation, protect_last = 1
                                        )
                                        if _n_roomier:
                                            conversation[:] = _roomier
                                            try:
                                                # Same stand-in reply as initial sizing; else reply-gated templates drop this call's args from count.
                                                _spent_after = self.count_chat_tokens(
                                                    neutralize_control_markup_in_messages(
                                                        self._countable_messages(
                                                            [*conversation, _size_probe]
                                                        ),
                                                        _markup_cache,
                                                        self.markup_profile,
                                                    ),
                                                    None,
                                                    safe_tools,
                                                    strict = True,
                                                    chat_template_kwargs = _reasoning_kw,
                                                )
                                            except Exception:
                                                logger.debug(
                                                    "result budget rescue: count failed",
                                                    exc_info = True,
                                                )
                                                _spent_after = None
                                            if _spent_after is not None:
                                                _rescued = tool_result_budget(
                                                    self._effective_context_length,
                                                    # The same cap the first pricing used,
                                                    # or the rescue is measured against a
                                                    # different request from the one it is
                                                    # rescuing.
                                                    _result_max_tokens,
                                                    _spent_after + _pending_args,
                                                ) // (len(_pending) + 1)
                                                logger.info(
                                                    "Result budget for %s was %d; compacted "
                                                    "%d completed call(s) and it is now %d",
                                                    decision.tool_name,
                                                    _result_budget,
                                                    _n_roomier,
                                                    _rescued,
                                                )
                                                _result_budget = max(_result_budget, _rescued)
                                    kwargs["result_budget_tokens"] = _result_budget
                                    _last_result_budget[0] = kwargs["result_budget_tokens"]
                                    # Known before the tool runs; the repeat guard below would only reach it three calls later.
                                    _starved_call[0] = _result_budget < _MIN_USEFUL_RESULT_TOKENS
                                if accepts_kwarg(execute_tool, "conversation_budget_tokens"):
                                    if accepts_kwarg(execute_tool, "conversation_token_counter"):
                                        # Hand over the tokenizer: char estimates undercount dense ASCII.
                                        kwargs["conversation_token_counter"] = lambda text: (
                                            self.count_chat_tokens(
                                                [{"role": "tool", "content": text}],
                                                None,
                                                None,
                                                strict = False,
                                            )
                                        )
                                    # Result and reply are protected on the next fit, so the result cannot spend the whole shared budget.
                                    kwargs["conversation_budget_tokens"] = _retrieval_budget(
                                        self._effective_context_length,
                                        # Iteration cap: a near-spent continuation must not reserve the whole allowance and starve recall.
                                        _iteration_max_tokens,
                                        _spent,
                                        reply_returns = True,
                                    )
                            if accepts_output_callback(execute_tool):
                                kwargs["output_callback"] = _output_callback
                            kwargs.update(search_images_kwargs(execute_tool, _decision.tool_name))
                            if image_share is not None:
                                kwargs["mcp_image"] = image_share["image"]
                            return execute_tool(
                                _decision.tool_name,
                                _decision.arguments,
                                **kwargs,
                            )

                        # Without this the handler below re-raises and a bad
                        # argument kills the whole answer. A raising tool is the
                        # tool's failure; the other two loops hand it back.
                        _started_events = (
                            (_pending_context_usage,) if _pending_context_usage is not None else ()
                        )
                        _pending_context_usage = None
                        try:
                            result = yield from stream_tool_execution(
                                _invoke_tool,
                                tool_name = decision.tool_name,
                                tool_call_id = decision.tool_call_id,
                                cancel_event = cancel_event,
                                started_events = _started_events,
                            )
                        except _LlamaStreamCancelled:
                            # Nudge only on evidence of truncation, since short results like OK can be complete; a false nudge makes the model discard a write or repeat it.
                            raise
                        except Exception as _tool_exc:
                            if cancel_event is not None and cancel_event.is_set():
                                raise
                            logger.exception(
                                "Tool %s raised: %s",
                                decision.tool_name,
                                _tool_exc,
                            )
                            result = f"Error: tool raised an exception: {_tool_exc}"
                        if decision.tool_name in RAG_SEARCH_TOOLS:
                            _kb_search_count += 1
                    # Logs actual cost beside allowance: only the pair tells an undelivered budget from an ignored one.
                    logger.info(
                        "tool result: name=%s budget_tokens=%s chars=%d",
                        decision.tool_name,
                        _last_result_budget[0],
                        len(result) if isinstance(result, str) else -1,
                    )
                    # Result was priced at zero before the call ran; tell the model now, not after three calls prove it.
                    _result_was_starved = False
                    # Nudge only on evidence of truncation: short results like 'OK' can be complete.
                    if (
                        _starved_call[0]
                        and isinstance(result, str)
                        and (not result.strip() or _is_window_notice(result))
                    ):
                        _starved_call[0] = False
                        _result_was_starved = True
                        logger.info(
                            "%s ran with a result budget below %d; telling the model the "
                            "window cannot carry the result",
                            decision.tool_name,
                            _MIN_USEFUL_RESULT_TOKENS,
                        )
                        result = _note_before_envelopes(
                            result,
                            decision.tool_name,
                            lambda body: _starved_result_message(decision.tool_name, body),
                        )
                    else:
                        # Cleared either way: it describes THIS call's budget, else the notice would pass to the next call.
                        _starved_call[0] = False

                    # ── No-progress guard: the same answer, over and over ──
                    # No-progress guard keyed on the RESULT, so it never fires while results change.
                    if isinstance(result, str):
                        # Arguments are in the key unless the result is the window's own truncation notice.
                        _result_key = (
                            decision.tool_name,
                            hashlib.sha1(result.encode("utf-8", "replace")).hexdigest(),
                            None
                            if _result_was_starved or _is_window_notice(result)
                            else hashlib.sha1(
                                json.dumps(decision.arguments, sort_keys = True, default = str).encode(
                                    "utf-8", "replace"
                                )
                            ).hexdigest(),
                        )
                        if _result_key == _last_tool_result_key[0]:
                            _identical_result_runs[0] += 1
                        else:
                            _last_tool_result_key[0] = _result_key
                            _identical_result_runs[0] = 1
                        if _identical_result_runs[0] >= _MAX_IDENTICAL_TOOL_RESULTS:
                            logger.info(
                                "%s returned the same result %d times; saying so instead of "
                                "letting the turn spend itself repeating it",
                                decision.tool_name,
                                _identical_result_runs[0],
                            )
                            # Told, not silently stopped. The model can still finish the
                            # task another way, and hard-stopping a turn that is otherwise
                            # healthy would trade one dead end for a worse one.
                            result = _note_before_envelopes(
                                result,
                                decision.tool_name,
                                lambda body: _repeated_result_message(
                                    decision.tool_name, _identical_result_runs[0], body
                                ),
                            )
                            _identical_result_runs[0] = 0
                            _last_tool_result_key[0] = None
                    completion = tool_controller.record_result(decision, result)
                    resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                    # Reset so the post-tool nudge isn't swallowed as a repeat.
                    _last_reprompt_text = ""
                    # A tool ran this turn, so it counts against the caller's budget. Loading
                    # MCP tools does not, or a budget of one could never call what it found.
                    if decision.tool_name != "mcp_tool_search":
                        _turn_executed_real_tool = True
                    _forced_choice_resolved = True
                    _tool_end = completion.tool_end_event()
                    yield _tool_end
                    conversation.append(completion.tool_message())
                    if _tool_end.get("tool_call_id"):
                        _tool_card_by_message[id(conversation[-1])] = _tool_end["tool_call_id"]
                    # Parsed only for a result that carries them and a model that will
                    # be shown them; the marker test first, so a text result never asks.
                    _completion_images = (
                        completion.mcp_images()
                        if mcp_images_mentioned_in(completion.result or "") and self.is_vision
                        else []
                    )
                    if _completion_images:
                        batch_mcp_images.append(_completion_images)
                    if _compact_after_execution and decision.tool_call_id:
                        # Compaction rebuilds the messages, so rebind to the live message carrying this call; scan backwards because ids restart at call_0 every turn.
                        _before_len = len(json.dumps(conversation, default = str))
                        conversation[:] = compact_executed_call_arguments(
                            conversation, decision.tool_call_id
                        )
                        logger.info(
                            "Compacted %s arguments after running it: %d -> %d chars",
                            decision.tool_name,
                            _before_len,
                            len(json.dumps(conversation, default = str)),
                        )
                        # The loop also ends on the outer iteration range that no-op and continuation turns consume; missing it causes a retry and a no-tools user turn.
                        for _msg in reversed(conversation):
                            if _msg.get("role") != "assistant":
                                continue
                            if any(
                                (_tc or {}).get("id") == decision.tool_call_id
                                for _tc in (_msg.get("tool_calls") or [])
                            ):
                                assistant_msg = _msg
                                break

                    if _forced_tool_call_pending:
                        _forced_tool_call_pending = False

                # Loop-ending turn offers no tools, so this notice must not ask for a retry; it rides the result.
                _over_cap_final = bool(_over_cap) and (
                    tool_controller.force_final_answer
                    or not tool_controller.active_tools()
                    or (_turn_executed_real_tool and _tool_iters_done + 1 >= max_tool_iterations)
                    # Fold feedback only into a result from the SAME tool, since templates label the block with the tool name; a retry notice for skipped calls is never folded.
                    or iteration + 1 >= max_tool_iterations + _extra + _continuation_credits
                )
                _final_over_cap = _over_cap if _over_cap_final else []
                if _over_cap_final:
                    _over_cap = []
                if _over_cap:
                    deferred_noop_msgs.append(
                        tool_call_limit_nudge(
                            _over_cap,
                            _MAX_TOOL_CALLS_PER_TURN,
                            unavailable_tools = {
                                _limit_decision.tool_name
                                for _call in _over_cap
                                if (_limit_decision := tool_controller.prepare_call(_call)).action
                                in ("disabled", "render_html_repeat")
                            },
                        )
                    )
                # A mixed execute/no-op batch already has a real tool result, so keeping the
                # feedback with that result beats appending a newer user turn, which makes
                # templates hide this turn's structured reasoning. Only when the result is
                # the SAME tool the feedback is about: templates label the whole block with
                # the result's own tool name (gemma-4.jinja resolves tool_call_id -> name and
                # wraps the body), so folding a note about tool A into tool B's result reads
                # as B's own output. Then the user turn is the lesser loss. A retry notice for
                # skipped calls is never folded: read as the tail of a result, the calls are not re-issued.
                _fold_target_matches = (
                    not _over_cap
                    and len(deferred_noop_tools) == 1
                    and bool(conversation)
                    and conversation[-1].get("role") == "tool"
                    and conversation[-1].get("name") in deferred_noop_tools
                )
                if deferred_noop_msgs and assistant_appended and _fold_target_matches:
                    if not _attach_internal_feedback_to_tool_result(
                        deferred_nudge_text(deferred_noop_msgs)
                    ):
                        append_deferred_nudges(conversation, deferred_noop_msgs)
                elif deferred_noop_msgs and assistant_appended:
                    append_deferred_nudges(conversation, deferred_noop_msgs)
                else:
                    # Blank trace is not a turn: reuse the field's emptiness test so whitespace-only splits add nothing.
                    if not assistant_appended and (
                        content_text or assistant_msg.get("reasoning_content") or _resumed_reasoning
                    ):
                        if assistant_msg.get("reasoning_content") or _resumed_reasoning:
                            # Whole thought: the nudge below hides the original's field.
                            assistant_msg["content"] = neutralize_control_markup(
                                _resumed_reasoning + reasoning_accum,
                                self.markup_profile,
                            )
                            _continued_partial = (
                                trailing_assistant_text(conversation)
                                if continue_final_message
                                else None
                            )
                            if _continued_partial and not _continued_partial.endswith("\n"):
                                assistant_msg["content"] = f"\n{assistant_msg['content']}"
                            if content_text:
                                assistant_msg["content"] += f"\n{content_text}"
                            assistant_msg.pop("reasoning_content", None)
                        append_assistant_turn(
                            conversation,
                            assistant_msg,
                            continue_final_message = _merge_into_partial,
                        )
                        if _resumed_reasoning:
                            conversation[-1].pop("reasoning_content", None)
                        assistant_appended = True
                    append_deferred_nudges(conversation, deferred_noop_msgs)
                if _final_over_cap:
                    _limit_text = tool_call_limit_nudge(
                        _final_over_cap, _MAX_TOOL_CALLS_PER_TURN, final = True
                    )["content"]
                    # Same rule as _fold_target_matches.
                    _limit_names = {
                        (_tc.get("function") or {}).get("name") for _tc in _final_over_cap
                    }
                    _limit_foldable = (
                        bool(conversation)
                        and conversation[-1].get("role") == "tool"
                        and _limit_names == {conversation[-1].get("name")}
                    )
                    if not (
                        _limit_foldable and _attach_internal_feedback_to_tool_result(_limit_text)
                    ):
                        if deferred_noop_msgs and conversation[-1].get("role") == "user":
                            conversation[-1] = {
                                **conversation[-1],
                                "content": f"{conversation[-1]['content']}\n\n{_limit_text}",
                            }
                        else:
                            conversation.append({"role": "user", "content": _limit_text})

                if batch_mcp_images and self.is_vision:
                    # One block after the batch: 'the tool call above' fits one result; with several it says so instead.
                    _batch_results = sum(
                        1
                        for m in conversation[batch_conversation_start:]
                        if isinstance(m, dict) and m.get("role") == "tool"
                    )
                    append_mcp_image_turn(
                        conversation,
                        batch_mcp_images,
                        per_result = True,
                        owned = loop_mcp_image_parts,
                        lead = MCP_DETACHED_IMAGE_TURN_TEXT
                        if _batch_results != 1
                        else MCP_IMAGE_TURN_TEXT,
                    )

                # Close provisional cards not resolved by execution/no-op handling.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "",
                            "provenance": provisional_tool_provenance(_pname),
                        }

                # No call in the batch ran (denied, duplicate, capped), so nothing was waiting on it.
                if _pending_context_usage is not None:
                    yield _pending_context_usage
                    _pending_context_usage = None

                # Clear tool status badge before next generation/final pass.
                yield {"type": "status", "text": ""}
                if tool_controller.force_final_answer or not tool_controller.active_tools():
                    _append_budget_exhausted_nudge = False
                    break
                # Only real tool turns use the cap, so reserved re-prompt slots never add rounds (GGUF parity).
                if _turn_executed_real_tool:
                    _tool_iters_done += 1
                    if _browser_turns:
                        _post_tool_reprompts = 0
                    if _tool_iters_done >= max_tool_iterations:
                        break
                continue

            except _LlamaStreamCancelled:
                _held = _cancelled_hold_text()
                if _held:
                    yield {"type": "content", "text": _held}
                return
            except httpx.ConnectError:
                # Mark unresolved provisional cards as failed before raising.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "Error: lost connection to llama-server before the tool call completed.",
                            "provenance": provisional_tool_provenance(_pname),
                        }
                raise RuntimeError("Lost connection to llama-server")
            except Exception as e:
                if cancel_event is not None and cancel_event.is_set():
                    return
                # Same cleanup for other mid-iteration failures.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "Error: the tool call was interrupted before it completed.",
                            "provenance": provisional_tool_provenance(_pname),
                        }
                raise

        # ── Tool iteration cap reached -- synthesize final answer ──
        # Cap reached with no final text: nudge the final streaming pass to answer rather than request tools.
        if max_tool_iterations > 0 and _append_budget_exhausted_nudge:
            if not _attach_internal_feedback_to_tool_result(BUDGET_EXHAUSTED_NUDGE):
                # Ride the trailing user turn (over-cap notice or promoted image): two in a row break Gemma.
                if conversation[-1].get("role") == "user":
                    conversation[-1] = {
                        **conversation[-1],
                        "content": with_appended_text(
                            conversation[-1]["content"], f"\n\n{BUDGET_EXHAUSTED_NUDGE}"
                        ),
                    }
                else:
                    conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})

        while _model_note_events:
            yield _model_note_events.pop(0)
        # Clear status.
        yield {"type": "status", "text": ""}

        # Final streaming pass with the full conversation context.
        _final_tools = _last_sent_tools
        from core.inference.chat_template_helpers import neutralize_control_markup_in_messages

        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        _final_max_tokens = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        _final_preflight_context_length = None
        _final_preflight_succeeded = False
        if context_overflow == "truncate_oldest" and self._effective_context_length:
            _final_preflight_context_length = self._effective_context_length
            try:
                _before_final_fit = conversation
                # Withheld, not `tools`: this is the synthesised final answer and
                # `stream_payload` below sends tool_choice "none", so the request's
                # catalogue answers a question it does not pose and would let a NEW epoch
                # start on the one pass that can never call `search_conversation`.
                # Replaying an epoch already in force is unaffected.
                _final_can_reset = _can_reset_epoch(
                    thread_id, _backend_supports_tools(self), tools_withheld = True
                )
                _final_sticky, _final_sticky_is_checkpoint = (
                    (0, True)
                    if _sticky_boundary_applied
                    else _sticky_compaction_state(
                        thread_id,
                        _request_branch,
                        context_policy = context_policy,
                        can_reset = _final_can_reset,
                        compaction_headroom_ratio = compaction_headroom_ratio,
                    )
                )
                conversation, truncation = _fit_with_instruction_pins(
                    conversation,
                    context_length = self._effective_context_length,
                    max_tokens = _final_max_tokens,
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            self._countable_messages(fitted), None, self.markup_profile
                        ),
                        None,
                        _final_tools,
                        strict = True,
                        chat_template_kwargs = _reasoning_kw,
                        should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                    ),
                    estimate_message = self._estimate_countable_message,
                    anchor_ids = _rolling_anchor_ids,
                    keeps_boundary = _keeps_compaction_boundary(thread_id),
                    can_reset = _final_can_reset,
                    reserve_tokens = _conversation_recall_reserve(thread_id),
                    sticky_dropped = _final_sticky,
                    sticky_is_checkpoint = _final_sticky_is_checkpoint,
                    **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                )
                _sticky_boundary_applied = True
                if truncation:
                    _recalled = _archive_and_recall(
                        conversation,
                        _before_final_fit,
                        branch_messages = _extend_live_branch(_before_final_fit),
                        thread_id = thread_id,
                        # Inline: under tool_choice "none" a forged tool role breaks strict templates.
                        style = "inline",
                        # Checkpoint mode recalls once, on the reset turn; rolling mode recalls on every turn that evicted.
                        force_recall = bool(truncation.get("checkpoint_started", True)),
                        # A rescued refusal may evict messages; archive them without recall.
                        recall_done = _conversation_recall_done or not truncation["fits"],
                        # The synthesized final answer never returns to the prompt.
                        recall_budget_tokens = _retrieval_budget(
                            self._effective_context_length,
                            max_tokens,
                            truncation.get("prompt_tokens_after") or 0,
                        ),
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                self._countable_messages(fitted),
                                None,
                                self.markup_profile,
                            ),
                            None,
                            _final_tools,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                        ),
                        cancel_check = lambda: bool(cancel_event and cancel_event.is_set()),
                        background = True,
                    )
                    conversation = _recalled["conversation"]
                    truncation = {**truncation, **_recalled["counts"]}
                    if _recalled["recalled"]:
                        _conversation_recall_done = True
                        for _message in _recalled["anchored"]:
                            _rolling_anchor_ids.add(id(_message))
                        for _ev in _recalled["events"]:
                            yield _ev
                if truncation and _records_boundary(truncation):
                    truncation = {
                        **truncation,
                        **_boundary_metadata(
                            conversation, _request_branch, compaction_headroom_ratio
                        ),
                    }
                # Re-cost here, not at the breaks: results and nudges appended after the last re-cost left the pool unaware of this final pass, the largest request.
                if truncation:
                    yield {"type": "context_truncated", **truncation}
                _final_preflight_succeeded = True
            except Exception as exc:
                logger.warning("Could not preflight the rolling context window: %s", exc)

        # Continuations for the final pass: it runs after the loop breaks, so the tool loop continuation never reaches a turn cut off mid-answer.
        if on_conversation_grew is not None:
            try:
                on_conversation_grew(conversation)
            except Exception:  # accounting must never break a run in progress
                logger.debug("tool loop final recost failed", exc_info = True)

        stream_payload = {
            "messages": neutralize_control_markup_in_messages(
                conversation, None, self.markup_profile
            ),
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        if logit_bias:
            stream_payload["logit_bias"] = logit_bias
        if _final_tools:
            stream_payload["tools"] = llama_grammar_tools(_final_tools)
            stream_payload["tool_choice"] = "none"
        if _reasoning_kw is not None:
            stream_payload["chat_template_kwargs"] = _reasoning_kw
        if thinking_budget_tokens is not None:
            stream_payload["thinking_budget_tokens"] = thinking_budget_tokens
        stream_payload["max_tokens"] = _final_max_tokens
        if stop:
            stream_payload["stop"] = stop
        _apply_seeded_llama_request(stream_payload, seed, reuse_prompt_cache = _prompt_sent)
        stream_payload["stream_options"] = {"include_usage": True}

        # Progress events feed the first-token deadline; timings stay opt-in.
        stream_payload["return_progress"] = True
        if _perf_callback_wants_timings(perf_callback):
            stream_payload["timings_per_token"] = True

        _final_respawn_truncations: list[dict] = []
        # Messages appended to the payload but absent from `conversation`.
        _refit_tail: list = []
        _refit_tail_merged = False

        def _record_refit_tail(committed: list, appended: list, merged: bool) -> None:
            """Track appended messages that a respawn refit must replay."""
            nonlocal _refit_tail, _refit_tail_merged
            _live = {id(_message) for _message in committed}
            _new = [_message for _message in appended if id(_message) in _live]
            if len(_new) != len(appended):
                # An eviction split the tail, so fall back to `conversation` alone.
                _refit_tail = []
                _refit_tail_merged = False
                return
            if not merged:
                _refit_tail = _refit_tail + _new
            elif _refit_tail:
                _refit_tail = _refit_tail[:-1] + _new
            else:
                _refit_tail = _new
                _refit_tail_merged = True

        def _refit_final_after_respawn() -> None:
            nonlocal conversation
            _before_respawn_fit = conversation
            if (
                _final_preflight_context_length is None
                or not self._effective_context_length
                or (
                    _final_preflight_succeeded
                    and _final_preflight_context_length == self._effective_context_length
                )
            ):
                return
            if max_tokens is None:
                stream_payload["max_tokens"] = self._effective_context_length
            try:
                conversation, truncation = _fit_with_instruction_pins(
                    conversation,
                    context_length = self._effective_context_length,
                    max_tokens = max_tokens,
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            self._countable_messages(fitted), None, self.markup_profile
                        ),
                        None,
                        _final_tools,
                        strict = True,
                        chat_template_kwargs = _reasoning_kw,
                        should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                    ),
                    estimate_message = self._estimate_countable_message,
                    anchor_ids = _rolling_anchor_ids,
                    keeps_boundary = _keeps_compaction_boundary(thread_id),
                    can_reset = _can_reset_epoch(
                        thread_id,
                        _backend_supports_tools(self),
                        # The final pass again, so again tool_choice is "none".
                        tools_withheld = True,
                    ),
                    **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                )
                if truncation:
                    # Archive only; see the iteration respawn refit above.
                    truncation.update(
                        _archive_and_recall(
                            conversation,
                            _before_respawn_fit,
                            thread_id = thread_id,
                            style = "inline",
                            recall_done = True,
                            branch_messages = _extend_live_branch(conversation),
                            cancel_check = lambda: bool(cancel_event and cancel_event.is_set()),
                            background = True,
                        )["counts"]
                    )
                # Recorded, not only forwarded, for the same reason as the iteration refit above.
                from core.inference import context_refusal  # noqa: PLC0415

                context_refusal.record_fit(truncation)
                # Copy because neutralization may return `conversation` itself.
                _refit_messages = list(
                    neutralize_control_markup_in_messages(conversation, None, self.markup_profile)
                )
                if _refit_tail:
                    # A merged tail replaces the trailing assistant prefill.
                    if _refit_tail_merged and trailing_assistant_text(_refit_messages) is not None:
                        _refit_messages[-1:] = _refit_tail
                    else:
                        _refit_messages.extend(_refit_tail)
                stream_payload["messages"] = _refit_messages
                if _refit_tail:
                    # Recheck the restored tail against the replacement window.
                    _refit_continue = bool(stream_payload.get("continue_final_message"))
                    if not _continuation_would_be_served(
                        stream_payload["messages"], _refit_continue
                    ):
                        # Protect the current user turn and its recovery tail.
                        _refit_protect = {id(_message) for _message in _refit_tail}
                        for _message in reversed(
                            _refit_messages[: len(_refit_messages) - len(_refit_tail)]
                        ):
                            _refit_protect.add(id(_message))
                            if _message.get("role") == "user":
                                break
                        _refit_evicted = _evict_until_it_fits(
                            stream_payload["messages"],
                            _final_tools,
                            stream_payload.get("chat_template_kwargs"),
                            _refit_continue,
                            _refit_protect,
                        )
                        if _refit_evicted is not None:
                            # Nothing else is safe to evict.
                            stream_payload["messages"] = _refit_evicted
                if truncation:
                    if _records_boundary(truncation):
                        truncation.update(
                            _boundary_metadata(
                                conversation, _request_branch, compaction_headroom_ratio
                            )
                        )
                    _final_respawn_truncations.append(truncation)
            except Exception as exc:
                logger.warning("Could not refit rolling context after respawn: %s", exc)
                raise

        cumulative = ""
        _last_emitted = ""
        in_thinking = False
        has_content_tokens = False
        reasoning_text = ""
        # Track content separately to preserve literal `</think>` in the answer.
        answer_text = ""
        _prov_entry = None
        _final_reasoning_started_at: Optional[float] = None
        _final_reasoning_summary_emitted = False
        _metadata_usage = None
        _metadata_timings = None
        _metadata_finish_reason = None
        _stream_done = False

        # A caller cap yields only its remainder, never a fresh one, so continuations cannot exceed the limit; None when unset or the whole window.
        _final_length_continuations = 0
        _continue_final = False
        _final_replayed_chars = 0

        def _remaining_output_budget() -> "Optional[int]":
            """What is left of a cap the CALLER set, or None when they set none.

            `finish_reason: "length"` does not say which wall was hit. A caller asking for
            at most 100 completion tokens gets it at their own cap, and continuing twice
            more returned roughly 300 -- over the limit the API promised, for latency and
            tokens nobody asked for. Only the context wall deserves a continuation, and a
            caller who set a cap gets the remainder of it rather than a fresh one.

            None means no explicit cap: `max_tokens` unset, or set to the whole window,
            which is what the backend substitutes for "Max" and is indistinguishable from
            unset. There the length stop IS the context wall, which is the case this
            continuation exists for.
            """
            if max_tokens is None:
                return None
            window = self._effective_context_length or 0
            if window and max_tokens >= window:
                return None
            # Whether the continuation fits once the partial is appended, so a reopen is not rejected before any token; an uncountable prompt fails open.
            _this_attempt = (
                _backfill_usage_from_timings(_metadata_usage, _metadata_timings) or {}
            ).get("completion_tokens", 0) or int(stream_payload.get("max_tokens") or 0)
            return max(0, max_tokens - _accumulated_completion_tokens - _this_attempt)

        def _continuation_would_be_served(
            candidate_messages,
            continue_flag: bool,
            reasoning_kw = _UNSET,
        ) -> bool:
            """Whether the continuation fits once the partial is appended; an uncountable prompt
            fails open."""
            if not self._effective_context_length:
                return True
            try:
                _tokens = self.count_chat_tokens(
                    self._countable_messages(candidate_messages),
                    None,
                    _final_tools,
                    strict = True,
                    chat_template_kwargs = (
                        stream_payload.get("chat_template_kwargs")
                        if reasoning_kw is _UNSET
                        else reasoning_kw
                    ),
                    continue_final_message = continue_flag,
                )
            except Exception:
                logger.debug("final continuation: prompt count failed", exc_info = True)
                return True
            # Not turn_is_servable: that reserves room for a tool-result notice.
            _floor = _reply_floor(self._effective_context_length)
            if _tokens + _floor <= self._effective_context_length:
                return True
            logger.info(
                "Final continuation would not be served (%d tokens of a %d-token "
                "window); keeping the partial answer instead",
                _tokens,
                self._effective_context_length,
            )
            return False

        # Per-attempt mark: emitted state is cumulative across continuations.
        _attempt_started_at = ""
        while True:
            try:
                with self._open_chat_stream_with_respawn_retry(
                    stream_payload,
                    cancel_event,
                    on_respawn = _refit_final_after_respawn,
                ) as (
                    response,
                    first_token_deadline,
                ):
                    for truncation in _final_respawn_truncations:
                        yield {"type": "context_truncated", **truncation}
                    buffer = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response,
                        cancel_event,
                        first_token_deadline = first_token_deadline,
                    ):
                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                if in_thinking:
                                    if (
                                        _final_reasoning_started_at is not None
                                        and not _final_reasoning_summary_emitted
                                    ):
                                        _final_reasoning_summary_emitted = True
                                        yield _reasoning_summary_event(_final_reasoning_started_at)
                                    if has_content_tokens:
                                        cumulative += "</think>"
                                        _prov_entry = None
                                        yield {
                                            "type": "content",
                                            "text": _strip_tool_markup(cumulative, final = True),
                                        }
                                    else:
                                        cumulative = _finalize_reasoning_only_cumulative(
                                            cumulative,
                                            reasoning_text,
                                            _metadata_finish_reason,
                                            promote_reasoning_only,
                                        )
                                        # Clear it, or the retry closes the block twice and shows a stray tag.
                                        in_thinking = False
                                        _prov_entry = None
                                        yield {"type": "content", "text": cumulative}
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                chunk_data = json.loads(line[6:])

                                _report_live_llama_timings(perf_callback, chunk_data)
                                # Capture server timings/usage from final chunks.
                                _chunk_timings = chunk_data.get("timings")
                                if _chunk_timings:
                                    _metadata_timings = _chunk_timings
                                _chunk_usage = chunk_data.get("usage")
                                if _chunk_usage:
                                    _metadata_usage = _chunk_usage
                                # See the note on the first stream loop.
                                _stream_error = stream_error_from_chunk(chunk_data)
                                if _stream_error is not None:
                                    raise _stream_error
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    _fr = choices[0].get("finish_reason")
                                    if _fr:
                                        _metadata_finish_reason = _fr

                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        if _final_reasoning_started_at is None:
                                            _final_reasoning_started_at = time.monotonic()
                                        reasoning_text += reasoning
                                        if not in_thinking:
                                            if reasoning_provenance is not None and not cumulative:
                                                reasoning_provenance["wrapped"] = (
                                                    reasoning_provenance.get("wrapped", 0) + 1
                                                )
                                                _prov_entry = {"len": 0}
                                                reasoning_provenance.setdefault("wraps", []).append(
                                                    _prov_entry
                                                )
                                            cumulative += "<think>"
                                            in_thinking = True
                                        cumulative += reasoning
                                        if _prov_entry is not None:
                                            _prov_entry["len"] += len(reasoning)
                                        yield {"type": "content", "text": cumulative}

                                    token = delta.get("content", "")
                                    if token:
                                        if (
                                            _final_reasoning_started_at is not None
                                            and not _final_reasoning_summary_emitted
                                        ):
                                            _final_reasoning_summary_emitted = True
                                            yield _reasoning_summary_event(
                                                _final_reasoning_started_at
                                            )
                                        has_content_tokens = True
                                        answer_text += token
                                        if in_thinking:
                                            cumulative += "</think>"
                                            in_thinking = False
                                            _prov_entry = None
                                        cumulative += token
                                        cleaned = (
                                            _final_answer_stripper.strip(cumulative)
                                            if auto_heal_tool_calls
                                            else cumulative
                                        )
                                        # Emit only when cleaned text grows (monotonic).
                                        if len(cleaned) > len(_last_emitted):
                                            _last_emitted = cleaned
                                            yield {"type": "content", "text": cleaned}
                            except json.JSONDecodeError:
                                logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                        if _stream_done:
                            break  # exit outer for
                    # Commit the candidate only once another stream runs, so usage is not double counted; replay only new text, or the fragment is repeated.
                    _shown_this_attempt = _last_emitted[len(_attempt_started_at) :]
                    if (
                        _metadata_finish_reason == "length"
                        and has_content_tokens
                        and _shown_this_attempt.strip()
                        and _final_length_continuations < _MAX_LENGTH_CONTINUATIONS
                        and not is_repetition_dominated(_last_emitted)
                    ):
                        from core.inference.chat_template_helpers import (  # noqa: PLC0415
                            append_assistant_turn as _append_assistant_turn,
                        )

                        _final_length_continuations += 1
                        logger.info(
                            "Final answer cut off at %d chars; continuing %d/%d",
                            len(_last_emitted),
                            _final_length_continuations,
                            _MAX_LENGTH_CONTINUATIONS,
                        )
                        # Neutralize the replayed text before it is counted, so delimiters are not read back as chat structure and the count matches the payload.
                        _candidate_messages = list(stream_payload["messages"])
                        _merged_f = trailing_assistant_text(_candidate_messages) is not None
                        _append_assistant_turn(
                            _candidate_messages,
                            {
                                "role": "assistant",
                                "content": _last_emitted[_final_replayed_chars:],
                            },
                            continue_final_message = True,
                        )
                        # Neutralize the model's replayed text so delimiters aren't read as chat structure.
                        _candidate_messages = neutralize_control_markup_in_messages(
                            _candidate_messages, None, self.markup_profile
                        )
                        _continuation_tail = _candidate_messages[-1:]
                        _next_cap = _remaining_output_budget()
                        _served = _next_cap != 0 and _continuation_would_be_served(
                            _candidate_messages, True
                        )
                        if _next_cap != 0 and not _served:
                            # As the in-loop path: last decision of the turn, so one eviction short still gets that eviction.
                            _evicted_f = _evict_until_it_fits(
                                _candidate_messages,
                                _final_tools,
                                stream_payload.get("chat_template_kwargs"),
                                True,
                            )
                            if _evicted_f is not None and _continuation_would_be_served(
                                _evicted_f, True
                            ):
                                _candidate_messages = _evicted_f
                                _served = True
                        if _served:
                            stream_payload["messages"] = _candidate_messages
                            _record_refit_tail(_candidate_messages, _continuation_tail, _merged_f)
                            _final_replayed_chars = len(_last_emitted)
                            stream_payload["continue_final_message"] = True
                            stream_payload["add_generation_prompt"] = False
                            if _next_cap is not None:
                                stream_payload["max_tokens"] = _next_cap
                            # Folded in only now, else the reported usage counts just the last fragment.
                            _fu_f = (
                                _backfill_usage_from_timings(_metadata_usage, _metadata_timings)
                                or {}
                            )
                            _accumulated_completion_tokens += _fu_f.get("completion_tokens", 0)
                            _it_f = _metadata_timings or {}
                            _accumulated_predicted_ms += _it_f.get("predicted_ms", 0)
                            _accumulated_predicted_n += _it_f.get("predicted_n", 0)
                            _stream_done = False
                            _metadata_finish_reason = None
                            # Reset per attempt so the next one is not charged twice.
                            _metadata_usage = None
                            _metadata_timings = None
                            # Next attempt starts from what is on screen now, so a continuation with nothing new is judged alone.
                            _attempt_started_at = _last_emitted
                            _continue_final = True
                            # No blank status: it resets the route cursor and the partial would be sent twice.
                            yield {
                                "type": "status",
                                "text": _CONTINUE_TRUNCATED_ANSWER_STATUS,
                            }
                        else:
                            _meta = _build_metadata_event(
                                _metadata_usage, _metadata_timings, _metadata_finish_reason
                            )
                            if _meta is not None:
                                yield _meta
                    elif (
                        _metadata_finish_reason == "length"
                        and not _shown_this_attempt.strip()
                        and reasoning_text.strip()
                    ):
                        # Reasoning-only stop after the loop broke; would render empty otherwise.
                        if _final_length_continuations < _MAX_LENGTH_CONTINUATIONS:
                            from core.inference.chat_template_helpers import (  # noqa: PLC0415
                                append_assistant_turn as _append_assistant_turn,
                            )

                            _final_length_continuations += 1
                            logger.info(
                                "Final pass ended inside its reasoning (%d chars); "
                                "continuing %d/%d with thinking off",
                                len(reasoning_text),
                                _final_length_continuations,
                                _MAX_LENGTH_CONTINUATIONS,
                            )
                            # Build the candidate first; commit only if it will run, else payload and usage are double-counted.
                            _candidate_r = list(stream_payload["messages"])
                            _merged_r = (
                                bool(stream_payload.get("continue_final_message"))
                                and trailing_assistant_text(_candidate_r) is not None
                            )
                            _append_assistant_turn(
                                _candidate_r,
                                {
                                    "role": "assistant",
                                    "content": _unfinished_thought_progress(reasoning_text),
                                },
                                continue_final_message = bool(
                                    stream_payload.get("continue_final_message")
                                ),
                            )
                            _candidate_r.append(
                                {"role": "user", "content": _continue_after_length_message()}
                            )
                            # Neutralize: the progress note quotes the model's reasoning.
                            _candidate_r = neutralize_control_markup_in_messages(
                                _candidate_r, None, self.markup_profile
                            )
                            # Admit with thinking-off kwargs, matching the request actually sent.
                            _off_kw = self._request_reasoning_kwargs(False, None, preserve_thinking)
                            _next_cap_r = _remaining_output_budget()
                            # Protect the current user turn and recovery tail.
                            _recovery_tail = _candidate_r[-2:]
                            _recovery_protect = {id(_message) for _message in _recovery_tail}
                            for _message in reversed(_candidate_r[:-2]):
                                _recovery_protect.add(id(_message))
                                if _message.get("role") == "user":
                                    break
                            _served_r = _next_cap_r != 0 and _continuation_would_be_served(
                                _candidate_r, False, _off_kw
                            )
                            if _next_cap_r != 0 and not _served_r:
                                _evicted_r = _evict_until_it_fits(
                                    _candidate_r, _final_tools, _off_kw, False, _recovery_protect
                                )
                                if _evicted_r is not None and _continuation_would_be_served(
                                    _evicted_r, False, _off_kw
                                ):
                                    _candidate_r = _evicted_r
                                    _served_r = True
                            if _served_r:
                                stream_payload["messages"] = _candidate_r
                                # Preserve both recovery messages across a respawn.
                                _record_refit_tail(_candidate_r, _recovery_tail, _merged_r)
                                # User-ended recovery invalidates continuation flags.
                                stream_payload.pop("continue_final_message", None)
                                stream_payload.pop("add_generation_prompt", None)
                                if _next_cap_r is not None:
                                    stream_payload["max_tokens"] = _next_cap_r
                                if _off_kw is not None:
                                    stream_payload["chat_template_kwargs"] = _off_kw
                                else:
                                    stream_payload.pop("chat_template_kwargs", None)
                                _fu_r = (
                                    _backfill_usage_from_timings(_metadata_usage, _metadata_timings)
                                    or {}
                                )
                                _accumulated_completion_tokens += _fu_r.get("completion_tokens", 0)
                                _it_r = _metadata_timings or {}
                                _accumulated_predicted_ms += _it_r.get("predicted_ms", 0)
                                _accumulated_predicted_n += _it_r.get("predicted_n", 0)
                                _stream_done = False
                                _metadata_finish_reason = None
                                # Reset per attempt so the next one is not charged twice.
                                _metadata_usage = None
                                _metadata_timings = None
                                # Next attempt starts from what is on screen now, so a continuation with nothing new is judged alone.
                                _attempt_started_at = _last_emitted
                                _continue_final = True
                                # No blank status here: it would replay what was already streamed.
                                yield {
                                    "type": "status",
                                    "text": _CONTINUE_AFTER_LENGTH_STATUS,
                                }
                            else:
                                yield {"type": "status", "text": ""}
                                # Content events are cumulative: don't overwrite an answer already shown.
                                if not _last_emitted.strip():
                                    yield {
                                        "type": "content",
                                        # Allowance spent with nothing shown: say so, since an empty message is not actionable; re-read the cap, as the last attempt may have hit Max Tokens.
                                        "text": (
                                            _reasoning_cap_spent_message(max_tokens)
                                            if _next_cap_r == 0
                                            else _thinking_exhausted_message(
                                                self._effective_context_length
                                            )
                                        ),
                                    }
                                _meta = _build_metadata_event(
                                    _metadata_usage,
                                    _metadata_timings,
                                    _metadata_finish_reason,
                                )
                                if _meta is not None:
                                    yield _meta
                        else:
                            # Allowance spent with nothing shown: say so. Re-read the cap here.
                            _cap_left_g = _remaining_output_budget()
                            yield {"type": "status", "text": ""}
                            # Content is cumulative: this must not overwrite an answer an earlier attempt already showed.
                            if not _last_emitted.strip():
                                yield {
                                    "type": "content",
                                    "text": (
                                        _reasoning_cap_spent_message(max_tokens)
                                        if _cap_left_g == 0
                                        else _thinking_exhausted_message(
                                            self._effective_context_length
                                        )
                                    ),
                                }
                            _meta = _build_metadata_event(
                                _metadata_usage, _metadata_timings, _metadata_finish_reason
                            )
                            if _meta is not None:
                                yield _meta
                    else:
                        _cut = _quote_cut_event(
                            reasoning_text,
                            answer_text,
                            _metadata_finish_reason,
                            promote_reasoning_only,
                        )
                        if _cut is not None:
                            yield _cut
                        _meta = _build_metadata_event(
                            _metadata_usage, _metadata_timings, _metadata_finish_reason
                        )
                        if _meta is not None:
                            yield _meta

                if _continue_final:
                    _continue_final = False
                    continue
                break

            except _LlamaStreamCancelled:
                # Strict callers raise, others get 0; should_abort is polled between the two calls; prefer_native tries /v1/chat/completions/input_tokens first.
                return
            except httpx.ConnectError:
                raise RuntimeError("Lost connection to llama-server")
            except Exception as e:
                if cancel_event is not None and cancel_event.is_set():
                    return
                raise

    def count_chat_tokens(
        self,
        messages,
        system = None,
        tools = None,
        strict: bool = False,
        chat_template_kwargs = None,
        continue_final_message: bool = False,
        should_abort = None,
        prefer_native: bool = False,
        template_options = None,
    ) -> int:
        """Strict callers raise: the text-only fallback is an approximation, not a count; others get 0."""
        if not self.is_loaded:
            if strict:
                raise RuntimeError("llama-server is not loaded")
            return 0

        def _block_text(content) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if isinstance(block.get("text"), str):
                            parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
                return "".join(parts)
            return ""

        # Normalize system into a leading message / plain text.
        system_text = ""
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = _block_text(system)

        # Count the neutralized prompt: raw text budgets a prompt generation never sends (#7066).
        from core.inference.chat_template_helpers import (
            neutralize_control_markup,
            neutralize_control_markup_in_messages,
            neutralize_tool_descriptions,
        )

        _profile = self.markup_profile
        messages = neutralize_control_markup_in_messages(messages, None, _profile)
        # Use the model's own profile so the count matches what generation sends.
        system_text = (
            neutralize_control_markup(system_text)
            if _profile is None
            else _profile.rewrite_control(system_text)
        )
        tools = neutralize_tool_descriptions(tools, None, _profile)

        try:
            with httpx.Client(
                timeout = 10,
                headers = self._auth_headers,
                trust_env = False,
                verify = _local_ssl_context(),
            ) as client:

                def _tokenize(text: str) -> int:
                    r = client.post(
                        f"{self.base_url}/tokenize",
                        json = {"content": text, "add_special": True},
                    )
                    if r.status_code != 200:
                        if strict:
                            raise RuntimeError("llama-server tokenizer failed")
                        return 0
                    tokens = r.json().get("tokens", [])
                    if not isinstance(tokens, list):
                        if strict:
                            raise RuntimeError("llama-server tokenizer returned invalid tokens")
                        return 0
                    return len(tokens)

                # 1. Try /apply-template to render the real chat prompt.
                from core.inference.mcp_images import prepare_image_turn_boundaries

                template_messages = prepare_image_turn_boundaries(
                    list(messages) if messages else [],
                    getattr(self, "_chat_template_override", None)
                    or getattr(self, "_chat_template", None),
                )
                if system_text:
                    template_messages = [
                        {"role": "system", "content": system_text}
                    ] + template_messages

                def _render_count(render_messages) -> tuple[Optional[int], bool]:
                    """(count or None, refused). Only a 500 is a template refusal; 503 = loading or no slot."""
                    try:
                        # /apply-template renders tool schemas; omitting them undercounts.
                        template_body = {"messages": render_messages}
                        if tools:
                            template_body["tools"] = llama_grammar_tools(tools)
                        # These can affect the prompt rendered for a client-owned tool call.
                        for key in ("tool_choice", "parallel_tool_calls", "response_format"):
                            if template_options and key in template_options:
                                template_body[key] = template_options[key]
                        # Layered over the load-time --chat-template-kwargs: only keys sent here move.
                        if chat_template_kwargs:
                            template_body["chat_template_kwargs"] = chat_template_kwargs
                        if continue_final_message:
                            template_body["continue_final_message"] = True
                            template_body["add_generation_prompt"] = False
                        if prefer_native:
                            if should_abort is not None and should_abort():
                                raise CountAborted()
                            try:
                                native = client.post(
                                    f"{self.base_url}/v1/chat/completions/input_tokens",
                                    json = template_body,
                                )
                                if native.status_code == 200:
                                    count = native.json().get("input_tokens")
                                    if type(count) is int and count > 0:
                                        return count, False
                            except Exception:
                                pass
                            if should_abort is not None and should_abort():
                                raise CountAborted()
                        resp = client.post(
                            f"{self.base_url}/apply-template",
                            json = template_body,
                        )
                        if resp.status_code != 200:
                            return None, resp.status_code == 500
                        prompt = resp.json().get("prompt", "")
                        if isinstance(prompt, str):
                            if should_abort is not None and should_abort():
                                raise CountAborted()
                            return _tokenize(prompt), False
                    except CountAborted:
                        # Not a template failure. Must precede the generic except.
                        raise
                    except Exception:
                        pass
                    return None, False

                # Qwen3.5+ templates raise on an empty render; retry behind an empty user turn.
                renders_empty = not template_messages or (
                    len(template_messages) == 1
                    and isinstance(template_messages[0], dict)
                    and template_messages[0].get("role") == "assistant"
                )
                with_user_turn = [{"role": "user", "content": ""}] + template_messages
                if renders_empty and self._empty_chat_render_refused:
                    count, _refused = _render_count(with_user_turn)
                else:
                    count, _refused = _render_count(template_messages)
                    if count is None and _refused and renders_empty:
                        if should_abort is not None and should_abort():
                            raise CountAborted()
                        count, _ = _render_count(with_user_turn)
                        if count is not None:
                            self._empty_chat_render_refused = True
                if count is not None:
                    return count

                # The fallback drops role markers, special tokens and tool schemas (~30% of a
                # six-turn two-tool prompt), so strict callers error rather than undercount.
                if strict:
                    raise RuntimeError("llama-server could not render the chat template")

                # Fallback: plain text plus serialized tools.
                parts = []
                if system_text:
                    parts.append(system_text)
                for msg in messages or []:
                    if isinstance(msg, dict):
                        parts.append(_block_text(msg.get("content", "")))
                if tools:
                    try:
                        parts.append(json.dumps(tools, ensure_ascii = False))
                    except Exception:
                        pass
                return _tokenize("\n".join(p for p in parts if p))
        except Exception:
            if strict:
                raise
            return 0
