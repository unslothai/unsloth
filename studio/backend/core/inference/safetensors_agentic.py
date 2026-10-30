# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Safetensors/transformers agentic tool loop.

Wraps a single-turn cumulative-text generator with the same tool-calling, thinking-block, status and
metadata event protocol the GGUF path uses, so the front-end SSE shape is identical across backends.

Unlike the GGUF path (``llama_cpp.py``), which uses llama-server's structured ``delta.tool_calls``,
native transformers has no such channel, so this loop parses tool calls from the cumulative text and
dispatches via ``core.inference.tools``.
"""

import bisect
import inspect
import json
import re
import threading
from typing import Callable, Generator, Optional

from loggers import get_logger

from core.inference.llama_cpp import _has_answer_artifact
from core.inference.tool_call_parser import (
    _GEMMA_BARE_TC_PREFIX_RE,
    _balanced_brace_end,
    blocked_bare_json_chain_may_continue,
    blocked_gemma_chain_may_continue,
    held_bare_gemma_tail_len,
    leading_bare_gemma_call_is_promotable,
    promotable_gemma_call_pos,
    _strip_mistral_reasoning,
    strip_segment as _parser_strip_segment,
    BUDGET_EXHAUSTED_NUDGE,
    MAX_ACT_REPROMPTS,
    NUDGE_TOOL_CALLS_STATUS,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    RAG_SEARCH_TOOLS,
    StreamingMarkupStripper,
    TOOL_XML_SIGNALS,
    is_reprompt_repeat,
    is_short_intent_without_action,
    parse_tool_calls_from_text,
    reprompt_to_act_message,
    strip_leading_bare_json_call,
    strip_llama3_leading_sentinels,
    strip_tool_markup,
)

from core.tool_healing import (
    _markerless_promotable,
    _THINK_CLOSE_RE,
    _think_spans_outside_tool_markup,
    strip_outside_think,
)
from core.inference.mcp_images import (
    DETACHED_IMAGE_TURN_TEXT as MCP_DETACHED_IMAGE_TURN_TEXT,
    IMAGE_TURN_TEXT as MCP_IMAGE_TURN_TEXT,
    append_placeholder_turn,
    png_payloads_per_result,
    trim_image_turns,
)
from core.inference.tool_loop_controller import (
    _WORKSPACE_READ_TOOLS,
    BROWSER_TOOL_NAMES,
    REPEATABLE_TOOL_NAMES,
    TRUNCATED_TOOL_TURN_MESSAGE,
    ToolLoopController,
    _reruns_after_new_work,
    append_deferred_nudges,
    awaiting_approval_status,
    coerce_tool_arguments,
    restore_replayed_tool_results,
    with_appended_text,
    status_for_tool,
    tool_call_limit_nudge,
    tool_event_provenance,
)
from core.inference.chat_template_helpers import (
    append_assistant_turn,
    trailing_assistant_resume_kind,
    forced_tool_name,
    forced_turn_tools,
)
from core.inference.passthrough_healing import nudge_enabled
from state.tool_approvals import (
    DECISION_EXPIRED,
    TOOL_APPROVAL_EXPIRED_MESSAGE,
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    decision_reason,
    note_host_access,
    wait_tool_decision,
)


logger = get_logger(__name__)


# Buffer cap while disambiguating a possible tool-call prefix.
_MAX_BUFFER_CHARS = 32

# Memory bound for holding a leading bare-JSON object whose top-level "{" never balances.
_MAX_BARE_JSON_BUFFER = 16384


# No grammar constraint here: dedupe calls and cap the count against runaway turns.
_MAX_TOOL_CALLS_PER_TURN = 8

# Enough settled text to catch a protocol literal split across two cumulative snapshots.
# ``_rehearsal_name_start`` still walks back through the candidate when the split is ``[ARGS]``.
_TOOL_SIGNAL_OVERLAP = max(map(len, TOOL_XML_SIGNALS)) - 1


def _active_tool_names(active_tools: list[dict]) -> list[str]:
    names = [
        (tool.get("function") or {}).get("name")
        for tool in active_tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    return [name for name in names if name]


# Unrestricted mode has no tool list, so any identifier may open a rehearsal; [ and ARGS letters optional so split chunks are still held.
_UNRESTRICTED_REHEARSAL_RE = re.compile(r"[\w-]+(?:\[(?:A(?:R(?:G(?:S)?)?)?)?)?")


def _is_rehearsal_prefix(
    stripped: str,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> bool:
    """Names that are not markerless-promotable stream as prose rather than being held for a call."""
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    if unrestricted:
        if _UNRESTRICTED_REHEARSAL_RE.fullmatch(stripped) is None:
            return False
        name, bracket, _ = stripped.partition("[")
        # Until '[' lands the name is open: terminal may yet become terminal_logs.
        return not bracket or _markerless_promotable(name, None)
    for name in _active_tool_names(active_tools):
        # Shared gate, not the built-in three: refuses mcp__* names too, and holding a suffix hides text.
        if not _markerless_promotable(name, None):
            continue
        if stripped == name or f"{name}[ARGS]".startswith(stripped):
            return True
    return False


def _held_rehearsal_tail_len(
    text: str,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> int:
    """Length of a trailing bare tool-name token held while streaming so the name does not leak; also call:NAME{ prefixes, seen only once the brace arrives."""
    # Restricted mode: same NAME[ARGS] bound as llama_cpp._held_rehearsal_tail_len.
    limit = (
        len(text)
        if unrestricted
        else max(
            (len(name) for name in _active_tool_names(active_tools) if isinstance(name, str)),
            default = 0,
        )
        + len("[ARGS]")
    )
    i = len(text)
    stop = max(0, i - limit - 1)
    while i > stop and not text[i - 1].isspace():
        i -= 1
    tail = text[i:]
    held = (
        len(tail)
        if tail
        and len(tail) <= limit
        and _is_rehearsal_prefix(tail, active_tools, unrestricted = unrestricted)
        else 0
    )
    return max(
        held,
        held_bare_gemma_tail_len(
            text, lambda: None if unrestricted else _active_tool_names(active_tools)
        ),
    )


def _rehearsal_name_start(
    candidate: str,
    signal_pos: int,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> int:
    """Draining on a name the parser will not promote would hold the turn for a call that never comes."""
    if not candidate.startswith("[ARGS]", signal_pos):
        return signal_pos
    j = signal_pos
    while j > 0 and (candidate[j - 1].isalnum() or candidate[j - 1] in "_-"):
        j -= 1
    if j < signal_pos and _markerless_promotable(
        candidate[j:signal_pos], None if unrestricted else _active_tool_names(active_tools)
    ):
        return j
    return signal_pos


def _earliest_tool_signal(
    candidate: str,
    signals,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
    start: int = 0,
    streaming: bool = False,
) -> int:
    """Markers inside <think> or [THINK] are not boundaries; draining there drops the visible answer."""
    think_spans = None
    floor = 0
    while True:
        best = -1
        for sig in signals:
            if sig != "[ARGS]":
                p = candidate.find(sig, start)
                if p >= 0 and (best < 0 or p < best):
                    best = p
                continue
            from_idx = start
            while True:
                p = candidate.find("[ARGS]", from_idx)
                if p < 0:
                    break
                name_start = _rehearsal_name_start(
                    candidate, p, active_tools, unrestricted = unrestricted
                )
                if name_start < p:
                    if name_start >= floor and (best < 0 or name_start < best):
                        best = name_start
                    break
                # Bare or prose [ARGS] is skipped so a later real call in the same chunk is still found.
                from_idx = p + len("[ARGS]")
        # Bare Gemma is promoted anywhere by the parser, so a mid-prose one is a boundary too; the catalogue is lazy because this runs per streamed delta.
        gemma = promotable_gemma_call_pos(
            candidate,
            None if unrestricted else (lambda: _active_tool_names(active_tools)),
            start,
            floor = floor,
            streaming = streaming,
        )
        if gemma >= floor and (best < 0 or gemma < best):
            best = gemma
        if best < 0:
            return -1
        if "<think" not in candidate and "[THINK" not in candidate:
            return best
        if think_spans is None:
            think_spans = _think_spans_outside_tool_markup(candidate)
        span_end = next((end for begin, end in think_spans if begin <= best < end), None)
        if span_end is None:
            return best
        if span_end <= floor:
            # Already scanning past this span and it still wins: nothing else is a boundary.
            return -1
        start = floor = span_end


def _has_genuine_tool_signal(
    candidate: str,
    signals,
    active_tools: list[dict],
    *,
    unrestricted: bool = False,
) -> bool:
    """An [ARGS] hit counts only after an active tool name, so buffering does not drain ordinary prose."""
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
    """Keeps reasoning blocks verbatim (else cumulative text shrinks then regrows), leaves inactive-name examples visible, and does not trim whitespace."""
    if not (auto_heal_tool_calls or tool_protocol_active):
        return text

    # Drop a leading Magistral [THINK] block (not the think channel); an unclosed one is held, then dropped at EOF if its closer never arrives.
    text = _strip_mistral_reasoning(text)

    def _seg(segment: str, is_last: bool) -> str:
        # Scan order lives in the parser's strip_segment so all strip paths cannot drift; end-of-turn arms run only on the last segment.
        return _parser_strip_segment(
            segment, seg_final = is_last, enabled_tool_names = enabled_tool_names
        )

    # Keep think blocks verbatim: shrinking cumulative text breaks append-by-length consumers.
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


def _append_raw_turn(conversation: list, assistant_msg: dict, *, continue_final_message: bool):
    """``append_assistant_turn`` for raw-text turns: a resumed thought is replayed whole after the
    re-emitted ``<think>``; text without that opener drops it."""
    if not (
        continue_final_message
        and trailing_assistant_resume_kind(conversation) == "reasoning_content"
        and isinstance(assistant_msg.get("content"), str)
    ):
        append_assistant_turn(
            conversation, assistant_msg, continue_final_message = continue_final_message
        )
        return
    thought = conversation[-1]["reasoning_content"]
    merged = {**conversation[-1], **assistant_msg}
    merged.pop("reasoning_content", None)
    if merged["content"].startswith("<think>"):
        merged["content"] = f"<think>{thought}{merged['content'][len('<think>'):]}"
    conversation[-1] = merged


def _reprompt_intent_text(
    text: str,
    *,
    reasoning_prefilled: bool = False,
    visible_only: bool = False,
) -> str:
    """Classifies visible text (think-block promises are private planning); reasoning falls back only on a reasoning-only stall, visible_only returns empty."""
    prefilled_reasoning = ""
    if reasoning_prefilled:
        close = _THINK_CLOSE_RE.search(text)
        if close is None:
            return "" if visible_only else text.strip()
        prefilled_reasoning = text[: close.end()].strip()
        text = text[close.end() :].strip()
        if not text:
            return "" if visible_only else prefilled_reasoning

    spans = _think_spans_outside_tool_markup(text)
    if not spans:
        return text.strip()

    visible: list[str] = []
    reasoning: list[str] = []
    cursor = 0
    for start, end in spans:
        visible.append(text[cursor:start])
        reasoning.append(text[start:end])
        cursor = end
    visible.append(text[cursor:])

    visible_text = "".join(visible).strip()
    reasoning_text = "".join(reasoning).strip()
    if visible_text or visible_only:
        return visible_text
    return "\n".join(part for part in (prefilled_reasoning, reasoning_text) if part).strip()


def _looks_like_enabled_bare_json(text: str, enabled_tool_names: Optional[set]) -> bool:
    """True when ``text`` opens with an ENABLED markerless bare-JSON call; an ordinary JSON answer returns False."""
    probe = strip_llama3_leading_sentinels(text.lstrip())
    if not (probe.startswith("{") and ('"name"' in probe or '"function"' in probe)):
        return False
    return strip_leading_bare_json_call(probe, enabled_tool_names) != probe


_FUNCTION_SIGNAL_RE = re.compile(r"<function=([\w-]+)>")
_TOOL_CALL_NAME_RE = re.compile(r'"name"\s*:\s*"([\w-]+)"')
# Mistral name/v11 and rehearsal forms, matching the parser so bracket-tag render_html fires early.
_MISTRAL_RENDER_NAME_RE = re.compile(
    r"\[TOOL_CALLS\]\s*([\w-]+)(?:\[CALL_ID\][\w-]+)?(?:\[ARGS\])?\s*(?=\{)"
)
_REHEARSAL_RENDER_NAME_RE = re.compile(r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]\s*(?=\{)")


def _first_detected_tool_name(content: str) -> Optional[str]:
    """The earliest marker wins; markers inside <think> or [THINK] blocks are skipped."""
    think_spans = _think_spans_outside_tool_markup(content)
    _think_starts = [s for s, _e in think_spans]

    def _in_think(pos: int) -> bool:
        if not think_spans:
            return False
        i = bisect.bisect_right(_think_starts, pos) - 1
        return i >= 0 and think_spans[i][0] <= pos < think_spans[i][1]

    def _first_outside(start: int, finder) -> int:
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
            # A bare "name" search can latch onto an argument key; use the parser.
            arr_calls = parse_tool_calls_from_text(content[mt:])
            if arr_calls:
                candidates.append((mt, (arr_calls[0].get("function") or {}).get("name") or ""))
    for rm in _REHEARSAL_RENDER_NAME_RE.finditer(content):
        if not _in_think(rm.start(1)):
            candidates.append((rm.start(1), rm.group(1)))
            break

    if not candidates:
        return None
    _pos, name = min(candidates, key = lambda c: c[0])
    return name or None


def _detect_render_html_tool_start(content: str) -> bool:
    """Return True when the FIRST tool call in ``content`` is clearly render_html."""
    return _first_detected_tool_name(content) == "render_html"


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


def _accepts_kwarg(func: Callable[..., str], name: str) -> bool:
    """Forward a kwarg only when the callable declares it or takes **kwargs; tests inject fakes."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _accepts_output_callback(func: Callable[..., str]) -> bool:
    return _accepts_kwarg(func, "output_callback")


def _search_images_kwargs(func: Callable[..., str], tool_name: str) -> dict[str, bool]:
    from core.inference.tool_stream_exec import search_images_kwargs
    return search_images_kwargs(func, tool_name)


def _call_single_turn(
    single_turn,
    conversation: list,
    active_tools: list[dict],
    tool_protocol_active: bool = True,
):
    """Call a single-turn generator with the tool schemas and protocol flag it supports."""
    try:
        return single_turn(
            conversation, active_tools = active_tools, tool_protocol_active = tool_protocol_active
        )
    except TypeError as exc:
        # A bare signature reports the FIRST unexpected kwarg, so accept either name.
        if "tool_protocol_active" not in str(exc) and "active_tools" not in str(exc):
            raise
    try:
        return single_turn(conversation, active_tools = active_tools)
    except TypeError as exc:
        if "active_tools" not in str(exc):
            raise
        return single_turn(conversation)


def _dense_message_tokens(messages: list[dict]) -> int:
    """`estimate_messages_tokens_dense`, imported where it is used like the rest here."""
    from core.inference.context_window import estimate_messages_tokens_dense
    return estimate_messages_tokens_dense(messages)


def _spent_prompt_tokens(
    conversation: list[dict],
    tools: Optional[list[dict]],
    generation_stats_holder: Optional[dict],
    prompt_dense_tokens: int,
) -> int:
    """Estimates only the tail since the last reported count (no tokenizer here); dense because 4 chars/token undercounts CJK; never zero, or recall switches off."""
    stats = (generation_stats_holder or {}).get("stats")
    usage = stats.get("usage") if isinstance(stats, dict) else None
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens > 0:
        added = _dense_message_tokens(conversation) - prompt_dense_tokens
        return prompt_tokens + max(0, added)
    return _dense_message_tokens(conversation) + _dense_message_tokens(tools or [])


def run_safetensors_tool_loop(
    *,
    single_turn: Callable[[list], Generator[str, None, None]],
    messages: list[dict],
    tools: list[dict],
    execute_tool: Callable[..., str],
    cancel_event: Optional[threading.Event] = None,
    auto_heal_tool_calls: bool = True,
    deduplicate_tool_calls: bool = True,
    nudge_tool_calls: Optional[bool] = None,
    max_tool_iterations: int = 25,
    tool_call_timeout: int = 300,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    rag_scope: Optional[dict] = None,
    website_policy: Optional[dict] = None,
    confirm_tool_calls: bool = False,
    mcp_image = None,
    bypass_permissions: bool = False,
    permission_mode: Optional[str] = None,
    sandbox_level: Optional[str] = None,
    reasoning_prefilled: bool = False,
    continue_final_message: bool = False,
    markup = None,
    renderable_tools = None,
    context_length: Optional[int] = None,
    max_tokens: Optional[int] = None,
    generation_stats_holder: Optional[dict] = None,
    images_sink: Optional[list] = None,
    caller_image_indexes: "tuple[int, ...]" = (),
    context_fitter: Optional[Callable[[list, list, list], dict]] = None,
    tool_choice = None,
    disable_parallel_tool_use: bool = False,
) -> Generator[dict, None, None]:
    """Yields status, content, tool_start and tool_end event dicts; content is cumulative text per turn."""
    if mcp_image is not None:
        from core.inference.mcp_image import note_attached_image
        from core.inference.tools import mcp_image_targets
        messages = note_attached_image(messages, mcp_image_targets(_active_tool_names(tools)))
    conversation = restore_replayed_tool_results(messages)
    # Where the caller's own attachment sits in the seeded sink. The cap is about what
    # the loop RE-SENDS, so that entry is never the one it drops -- but it is not the
    # whole seed: a resumed chat seeds replayed pictures too, and exempting those would
    # let the prompt carry a second full allowance. The route reserves the attachment's
    # slot by trimming replay to limit - 1 before interleaving it.
    caller_images = tuple(caller_image_indexes)
    # The branch this request is on. A GGUF-compacted thread keeps its archive across a
    # switch to safetensors, so search_conversation is advertised here too and needs the
    # same filtering: the stored rows are the whole DAG, and Retry leaves the replaced
    # response in them.
    # Restored dicts are new; the branch matches by identity.
    _live_branch = list(conversation)
    # ...and the replies and tool results the loop adds to it: a fit can evict this request's
    # own earlier tool exchange into the archive, where the client's messages alone would
    # refuse it. Not the loop's user turns: those are its own notices, and recall searches
    # for the branch's last user turn, which has to stay the request.
    _live_branch_ids = {id(message) for message in _live_branch}

    def _extend_live_branch(current: list) -> list:
        for message in current:
            if id(message) not in _live_branch_ids and message.get("role") != "user":
                _live_branch_ids.add(id(message))
                _live_branch.append(message)
        return _live_branch

    # Mirrors GGUF: full == bypass, unset -> auto, unknown -> ask, off keeps sandbox but never prompts; confirm_tool_calls=True alone is resolved to ask upstream.
    from core.inference.tool_stream_exec import stream_tool_execution
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
    from state.tool_approvals import LiveToolPermissions, bind_permission_upgrade

    live_permissions = LiveToolPermissions(session_id, cancel_event)
    sandbox_level = normalize_sandbox_level(sandbox_level)
    stream_tool_execution = account_tool_stream(stream_tool_execution)
    from core.inference.skill_mentions import load_mentioned_skills

    yield from load_mentioned_skills(
        conversation,
        tools if max_tool_iterations > 0 else [],
        permission_mode = permission_mode,
        bypass_permissions = bypass_permissions,
        confirm_tool_calls = confirm_tool_calls,
        session_id = session_id,
        cancel_event = cancel_event,
        permission_upgrade = live_permissions,
        context_length = context_length,
        continue_final_message = continue_final_message,
        # The downstream tokenizer owns fitting; only system context is guaranteed.
        dedup_tool_context = False,
    )

    # Forced first-pass RAG (GGUF parity) so doc questions do not lose to web_search; skipped only when retrieval would prompt (ask mode).
    from core.inference.tools import build_rag_autoinject

    # Resumed turn keeps partial trailing (autoinject would move the boundary); off/auto never prompt, so keep first-pass RAG despite a stale confirm flag.
    _skip_autoinject = (
        confirm_tool_calls and not bypass_permissions and permission_mode not in ("auto", "off")
    ) or bool(continue_final_message and trailing_assistant_resume_kind(conversation) is not None)
    _auto = None if _skip_autoinject else build_rag_autoinject(conversation, rag_scope)
    if _auto:
        for _ev in _auto["events"]:
            yield _ev
        conversation.extend(_auto["messages"])
    # Autoinjected KB search counts as an executed tool for the plan-without-action gate.
    rag_autoinjected = bool(_auto)
    # As on llama-server: a forced tool is the only one offered until a call runs or is denied.
    forced_resolved = rag_autoinjected and forced_tool_name(tool_choice) == "search_knowledge_base"

    unrestricted_tools = not tools
    # Gate telling a genuine NAME[ARGS] rehearsal from inactive-name prose; built from the
    # ORIGINAL tools list so a spent one-shot still reads as a tool name. None = unrestricted.
    from core.inference.tools import mcp_loadable_specs

    # Plus the MCP tools a search can load mid-turn (#10997).
    _loadable_mcp = [] if unrestricted_tools else mcp_loadable_specs(tools)
    _detect_tools = [] if unrestricted_tools else list(tools or []) + _loadable_mcp
    _enabled_names_gate = None if unrestricted_tools else set(_active_tool_names(_detect_tools))
    # Detection must see the same names as the strip gate (ORIGINAL list, incl. a spent
    # one-shot), else its repeat is stripped but never drained and the turn ends blank.
    # Sanitized at construction: prepare_call authorizes against the controller, so a tool
    # dropped from the prompt for unsafe markup must leave the controller too. The gates above
    # keep the ORIGINAL names: those decide what LOOKS like a call, not what may run (#7066).
    from core.inference.chat_template_helpers import neutralize_tool_descriptions

    # Use the renderer's markup profile and the all-templates-safe catalog, so prepare_call never authorizes a tool the render left out of the prompt.
    _authorized = (
        renderable_tools
        if renderable_tools is not None
        else neutralize_tool_descriptions(tools, None, markup)
    )
    tool_controller = ToolLoopController(
        tools = (None if unrestricted_tools else _authorized),
        auto_heal_tool_calls = auto_heal_tool_calls,
        deduplicate_tool_calls = deduplicate_tool_calls,
        session_id = session_id,
        thread_id = thread_id,
    )
    # RAG: cap knowledge-base searches per assistant turn (controller-agnostic).
    kb_search_count = 0
    final_attempt_done = False
    next_call_id = 0
    reprompt_count = 0
    # A browser task is many steps, each a chance to announce the next one and stop: there one
    # nudge per tool turn, as in the GGUF and external loops; other requests never nudge after a tool.
    browser_turns = any(name in BROWSER_TOOL_NAMES for name in _active_tool_names(tools or []))
    post_tool_reprompts = 0
    # Text that triggered the last nudge; if the retry restates it, stop (GGUF parity).
    last_reprompt_text = ""
    # A denied tool confirmation must not be answered with a plan-without-action
    # re-prompt (which would raise the confirmation gate again).
    tool_denied = False
    # Only turns that executed a tool consume max_tool_iterations (GGUF parity).
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
    _extra_iters = (
        MAX_ACT_REPROMPTS + (max_tool_iterations if browser_turns else 0)
        if max_tool_iterations > 0
        else 0
    )
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
            elif not forced_resolved:
                active_tools = forced_turn_tools(tool_choice, active_tools)

        tool_protocol_active = not final_attempt_done and (unrestricted_tools or bool(active_tools))
        tool_xml_signals = TOOL_XML_SIGNALS if tool_protocol_active else ()
        # Gate the markerless bare-JSON form on enabled names so an ordinary JSON answer isn't misread as a call.
        _enabled_tool_names = (
            None
            if unrestricted_tools
            else set(_active_tool_names(active_tools))
            | (set(_active_tool_names(_loadable_mcp)) if active_tools else set())
        )

        # Refit each iteration, since tool results grow the prompt; returns messages and events.
        if context_fitter is not None:
            fit_result = context_fitter(
                conversation, active_tools, _extend_live_branch(conversation)
            )
            conversation = list(fit_result.get("messages") or conversation)
            yield from fit_result.get("events") or ()

        # Cumulative snapshots: keep scans incremental with overlap for split literals.
        _streaming_stripper = StreamingMarkupStripper(_enabled_names_gate)
        _tool_signal_scanned_upto = 0

        def _strip_streaming_display(text: str) -> str:
            if not (auto_heal_tool_calls or tool_protocol_active):
                return text
            # Safetensors-only Magistral leading-reasoning removal first, then the shared strip.
            return _streaming_stripper.strip(_strip_mistral_reasoning(text))

        def _cancelled_buffer_text() -> str:
            """Held text a cancel would drop: the streaming tail not yet in last_emitted, and the buffer only while not already folded (BUFFERING misses bare-JSON/Gemma)."""
            held = cumulative_display + ("" if buffer_in_display else content_buffer)
            if not held:
                return ""
            cleaned = strip_tool_markup(held, final = True, enabled_tool_names = _enabled_tool_names)
            return cleaned if len(cleaned) > len(last_emitted) else ""

        detect_state = _state_buffering
        content_buffer = ""
        # Whether content_buffer has already been added to cumulative_display.
        buffer_in_display = False
        content_accum = ""
        cumulative_display = ""
        last_emitted = ""
        provisional_render_html_started = False
        provisional_resolved = False
        provisional_render_html_id = f"call_{next_call_id}"
        # Live-args offset for the provisional render_html card, streaming drained call text as tool_args.
        _live_args_streamed_upto = -1
        # Suppress the early render_html card when confirmation can prompt, since a provisional card would show it running before approval (auto mode keeps it).
        _provisional_confirm_gated = tool_call_may_prompt(
            confirm_tool_calls = bool(confirm_tool_calls),
            bypass_permissions = bypass_permissions,
            permission_mode = permission_mode,
            name = "render_html",
        )

        def _should_start_provisional_render_html(content: str) -> bool:
            # Re-resolved per chunk: single_turn may mutate active_tools and the first name can change.
            if (
                _tool_succeeded("render_html")
                or _provisional_confirm_gated
                or provisional_render_html_started
            ):
                return False
            if not any(
                ((tool.get("function") or {}).get("name") == "render_html") for tool in active_tools
            ):
                return False
            return _first_detected_tool_name(content) == "render_html"

        # Conversation as the prompt renders it, so later appends are charged alone against the count.
        prompt_dense_tokens = _dense_message_tokens(conversation)
        gen = _call_single_turn(single_turn, conversation, active_tools, tool_protocol_active)
        prev_cumulative = ""

        _gen_iter = iter(gen)
        while True:
            try:
                cumulative = next(_gen_iter)
            except StopIteration:
                break
            except Exception:
                # Pipeline raised mid-stream: close a provisional render_html card as errored so it never spins.
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
                emit = _cancelled_buffer_text()
                if emit:
                    yield {"type": "content", "text": emit}
                return

            if not isinstance(cumulative, str):
                continue  # defensive: pipeline yields only strings

            # Length deltas need grow-only snapshots: a rebuilt text diffed by length splices two renderings; MLX withholds while matching stop sequences.
            delta = cumulative[len(prev_cumulative) :]
            prev_cumulative = cumulative
            if not delta:
                continue
            content_accum += delta

            if detect_state == _state_draining:
                if _should_start_provisional_render_html(content_accum):
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
                elif (
                    provisional_render_html_started
                    and not provisional_resolved
                    and _live_args_streamed_upto >= 0
                    and len(content_accum) > _live_args_streamed_upto
                ):
                    # Display-only partial call for the live canvas; content_accum still feeds the stream-end parser.
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
                # Earliest real boundary: skip prose [ARGS], but pull a NAME[ARGS] back to NAME so it is kept.
                signal_pos = _earliest_tool_signal(
                    candidate,
                    tool_xml_signals,
                    _detect_tools,
                    unrestricted = unrestricted_tools,
                    start = max(0, _tool_signal_scanned_upto - _TOOL_SIGNAL_OVERLAP),
                    streaming = True,
                )
                if signal_pos >= 0:
                    before_tool = candidate[:signal_pos]
                    cleaned_before = _strip_streaming_display(before_tool)
                    if len(cleaned_before) > len(last_emitted):
                        last_emitted = cleaned_before
                        yield {"type": "content", "text": cleaned_before}
                    cumulative_display = candidate
                    detect_state = _state_draining
                    if _should_start_provisional_render_html(content_accum):
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
                _tool_signal_scanned_upto = len(candidate)
                cumulative_display = candidate
                cleaned = _strip_streaming_display(cumulative_display)
                # Hold a trailing bare tool name until its [ARGS] arrives; later prose or stream end releases it.
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
                # Substring-match mid-buffer bracket tags, as GGUF does; bare [ARGS] counts only with an active NAME.
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

            # Llama-3.2 bare JSON has no XML signal: hold a leading brace until it closes, then drain if it parses as a call, else stream it as content.
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
                        # Oversized open enabled-tool call: drain (memory bound) rather than leak the raw prefix; ordinary large JSON answers still stream.
                        detect_state = _state_draining
                        continue
                elif parse_tool_calls_from_text(
                    content_buffer,
                    id_offset = next_call_id,
                    allow_incomplete = auto_heal_tool_calls,
                    enabled_tool_names = _enabled_tool_names,
                ):
                    detect_state = _state_draining
                    continue
                elif blocked_bare_json_chain_may_continue(content_buffer, _enabled_tool_names):
                    if len(stripped) < _MAX_BARE_JSON_BUFFER:
                        continue
                    # Chain outgrew the bounded buffer: fail closed rather than stream
                    # content a later peer could make executable.
                    detect_state = _state_draining
                    continue

            # Gemma call:NAME{...} has no signal entry, so buffer it or it streams raw until end-of-turn; (?<!\w) keeps recall: out.
            _gemma_lead = leading_bare_gemma_call_is_promotable(stripped, _enabled_tool_names)
            _gemma_chain = blocked_gemma_chain_may_continue(stripped, _enabled_tool_names)
            if (
                not is_match
                and not is_prefix
                and tool_protocol_active
                and (
                    "call:".startswith(stripped)
                    or _GEMMA_BARE_TC_PREFIX_RE.match(stripped) is not None
                    or _gemma_lead
                    or _gemma_chain
                )
            ):
                if _gemma_lead:
                    detect_state = _state_draining
                    continue
                if _gemma_chain:
                    # A promotable peer behind a blocked call must not stream before the
                    # end-of-turn parser gets it.
                    if parse_tool_calls_from_text(
                        stripped,
                        id_offset = next_call_id,
                        allow_incomplete = auto_heal_tool_calls,
                        enabled_tool_names = _enabled_tool_names,
                    ):
                        detect_state = _state_draining
                        continue
                    if len(stripped) < _MAX_BARE_JSON_BUFFER:
                        continue
                    detect_state = _state_draining
                    continue
                # Names can exceed 32 chars (MCP), so buffer a call: prefix up to the bare-JSON bound rather than a short fixed cap that would flush real calls raw.
                if _GEMMA_BARE_TC_PREFIX_RE.match(stripped) is not None:
                    if len(stripped) < _MAX_BARE_JSON_BUFFER:
                        continue
                    detect_state = _state_draining
                    continue
                if len(stripped) < _MAX_BUFFER_CHARS:
                    continue  # bare "call:" prefix still forming

            if is_match:
                # Tool signal: flush any visible prefix before DRAINING so the route sends it before tool_start.
                cumulative_display += content_buffer
                buffer_in_display = True
                cleaned = _strip_streaming_display(cumulative_display)
                if len(cleaned) > len(last_emitted):
                    last_emitted = cleaned
                    yield {"type": "content", "text": cleaned}
                detect_state = _state_draining
                if _should_start_provisional_render_html(content_accum):
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
                buffer_in_display = True
                cleaned = _strip_streaming_display(cumulative_display)
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

        if cancel_event is not None and cancel_event.is_set():
            emit = _cancelled_buffer_text()
            if emit:
                yield {"type": "content", "text": emit}
            return

        if detect_state == _state_buffering:
            # [ARGS] is name-gated so prose with a literal foo[ARGS]{...} is not parsed.
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
                # A held bare-JSON enabled-tool fragment has no XML signal, so drain it; plain JSON still streams.
                detect_state = _state_draining
            else:
                # Drain so re-prompt + safety-net parser still fire on short emissions.
                if content_buffer:
                    cumulative_display += content_buffer
                    buffer_in_display = True
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
                # Re-prompt once on plan-without-action before tools run; explicit request flags beat the default.
                intent_text = _reprompt_intent_text(
                    content_accum,
                    reasoning_prefilled = reasoning_prefilled,
                )
                already_acted = any(record.executed for record in tool_controller.history)
                if (
                    auto_heal_tool_calls
                    and nudge_enabled(nudge_tool_calls)
                    and active_tools
                    and (
                        post_tool_reprompts < 1
                        if already_acted
                        else reprompt_count < MAX_ACT_REPROMPTS
                    )
                    # Retrieval injected up front is no reason to stop a browser task mid-way.
                    and (not rag_autoinjected or (browser_turns and already_acted))
                    and not tool_denied
                    and (browser_turns or not already_acted)
                    and not is_reprompt_repeat(intent_text, last_reprompt_text)
                    and is_short_intent_without_action(intent_text)
                    # Markup stripped first: a call fenced inside <tool_call> is not an answer.
                    and not _has_answer_artifact(
                        strip_tool_markup(
                            _reprompt_intent_text(
                                content_accum,
                                reasoning_prefilled = reasoning_prefilled,
                                visible_only = True,
                            ),
                            final = True,
                            enabled_tool_names = _enabled_tool_names,
                        )
                    )
                ):
                    if already_acted:
                        post_tool_reprompts += 1
                    else:
                        reprompt_count += 1
                    last_reprompt_text = intent_text
                    logger.info(
                        "Safetensors re-prompt %d/%d: model responded without "
                        "calling tools (%d chars)",
                        post_tool_reprompts if already_acted else reprompt_count,
                        1 if already_acted else MAX_ACT_REPROMPTS,
                        len(intent_text),
                    )
                    # Merge into a resumed partial: a second assistant turn breaks alternation.
                    _append_raw_turn(
                        conversation,
                        {"role": "assistant", "content": intent_text},
                        continue_final_message = continue_final_message,
                    )
                    tool_hint = " or ".join(_active_tool_names(active_tools)) or "an available tool"
                    conversation.append(
                        {
                            "role": "user",
                            "content": reprompt_to_act_message(tool_hint),
                        }
                    )
                    # Blank first: it clears the badge and resets the route's per-turn
                    # text cursor. The badge then shows the pause is a re-prompt, not a stall.
                    yield {"type": "status", "text": ""}
                    yield {"type": "status", "text": NUDGE_TOOL_CALLS_STATUS}
                    continue

                # Restore raw text when a literal tool marker in prose never parsed as a call; route-level cleanup still applies the Auto-Heal policy afterward.
                if content_accum and any(sig in content_accum for sig in tool_xml_signals):
                    yield {"type": "content", "text": content_accum}
                else:
                    # No [ARGS] followed, so the held rehearsal tail is real prose and is released.
                    final_clean = _strip_streaming_display(cumulative_display)
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
            # Gate on the ORIGINAL tools so a spent one-shot repeat routes to the no-op.
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
                    # Unparsed bare-JSON call: auto-heal drops it, while off keeps it visible under the strict contract.
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
            # Strip a leading bare-JSON call from kept content, since _strip_tool_markup_final only knows XML.
            content_text = strip_leading_bare_json_call(content_text, _enabled_tool_names)

        # Transformers/MLX report the finished turn here; never run a healed, token-cut call.
        turn_stats = (generation_stats_holder or {}).get("stats") or {}
        if tool_calls and (
            turn_stats.get("truncated")
            or turn_stats.get("finish_reason") in ("length", "content_filter")
        ):
            if provisional_render_html_started and not provisional_resolved:
                provisional_resolved = True
                yield {
                    "type": "tool_end",
                    "tool_name": "render_html",
                    "tool_call_id": provisional_render_html_id,
                    "result": TRUNCATED_TOOL_TURN_MESSAGE,
                    "provenance": _tool_event_provenance(provisional = True),
                }
            # Empty status resets the route cursor; send only text not already streamed.
            display = _strip_streaming_display(cumulative_display)
            notice = display[len(last_emitted) :] + ("\n\n" if display else "")
            yield {"type": "status", "text": ""}
            yield {
                "type": "content",
                "text": notice + TRUNCATED_TOOL_TURN_MESSAGE,
            }
            return

        if final_attempt_done:
            if content_text:
                yield {"type": "content", "text": content_text}
            yield {"type": "status", "text": ""}
            return

        # Collapse exact-duplicate calls and cap the count (runaway-turn guard).
        over_cap: list = []
        if tool_calls:
            seen_keys: set = set()
            last_workspace_key = None
            # One rerun per piece of new work: `test, edit A, test, edit B, test` keeps every
            # test, while `read, edit, read, edit` stops replaying and cannot fill the cap.
            novel_kept = 0
            novel_at_last_keep: dict = {}
            deduped: list = []
            if not deduplicate_tool_calls:
                deduped = tool_calls[:_MAX_TOOL_CALLS_PER_TURN]
                over_cap = tool_calls[_MAX_TOOL_CALLS_PER_TURN:]
            for _tc in tool_calls if deduplicate_tool_calls else ():
                _fn = _tc.get("function", {}) or {}
                _key = (_fn.get("name", ""), str(_fn.get("arguments", "")))
                if _reruns_after_new_work(_fn.get("name")):
                    if _key == last_workspace_key:
                        continue
                    if _key in seen_keys:
                        if novel_kept <= novel_at_last_keep.get(_key, 0):
                            continue
                    elif _fn.get("name") not in _WORKSPACE_READ_TOOLS:
                        novel_kept += 1
                    novel_at_last_keep[_key] = novel_kept
                    last_workspace_key = _key
                # A browser call acts on the page as it is then: two scrolls are two scrolls.
                # Two identical generate_image calls are two images.
                elif _key in seen_keys and _fn.get("name") not in REPEATABLE_TOOL_NAMES:
                    continue
                seen_keys.add(_key)
                if len(deduped) < _MAX_TOOL_CALLS_PER_TURN:
                    deduped.append(_tc)
                else:
                    over_cap.append(_tc)
            if len(deduped) + len(over_cap) != len(tool_calls):
                logger.info(
                    "Safetensors: collapsed %d repeated tool call(s) in one turn to %d",
                    len(tool_calls),
                    len(deduped) + len(over_cap),
                )
            if over_cap:
                logger.info(
                    "Safetensors: skipped %d tool call(s) over the per-turn limit of %d",
                    len(over_cap),
                    _MAX_TOOL_CALLS_PER_TURN,
                )
            tool_calls = deduped
            # Truncated before the assistant turn is built, so extra calls never run.
            if disable_parallel_tool_use and len(tool_calls) > 1:
                tool_calls, over_cap = tool_calls[:1], []

        assistant_msg: dict = {"role": "assistant", "content": content_text}
        assistant_appended = False
        # Defer no-op nudges so one no-op does not abort the rest of the batch.
        deferred_noop_msgs: list = []
        # Per result; see append_image_turn's per_result note.
        batch_mcp_images: list = []
        # Where this batch's results start, for the image turn\'s wording.
        batch_conversation_start = len(conversation)

        for _call_index, tc in enumerate(tool_calls or []):
            func = tc.get("function", {}) or {}
            tool_name = func.get("name", "") or ""
            provisional_match = (
                provisional_render_html_started
                and tool_name == "render_html"
                and tc.get("id", "") == provisional_render_html_id
            )
            decision = tool_controller.prepare_call(tc, provisional = provisional_match)
            # Frontend groups a batch's tool cards by round_id; without it replay splits the batch.
            decision.provenance["round_id"] = iteration
            image_share = None
            if decision.should_execute and mcp_image is not None:
                from core.inference.tools import mcp_image_share
                image_share = mcp_image_share(decision.tool_name, decision.arguments, mcp_image)
                if image_share is not None:
                    decision = tool_controller.reprepare_call(
                        decision, provisional = provisional_match
                    )

            if not decision.should_execute:
                if content_text and not assistant_appended:
                    _append_raw_turn(
                        conversation,
                        assistant_msg,
                        continue_final_message = continue_final_message,
                    )
                    assistant_appended = True
                if provisional_match and not provisional_resolved:
                    # Close a shown provisional render_html card so it never dangles if the call becomes a no-op.
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
                # Merges into a resumed partial, so a continued turn that calls a tool stays one assistant message.
                _append_raw_turn(
                    conversation,
                    assistant_msg,
                    continue_final_message = continue_final_message,
                )
                assistant_appended = True
            else:
                assistant_msg.setdefault("tool_calls", []).append(decision.as_assistant_tool_call())

            # Bypass wins here too, so a direct internal caller with both flags
            # never prompts. "auto" pauses only high-risk calls; "off" pauses only a
            # high-risk python/terminal call without OS isolation.
            permission_mode, bypass_permissions = live_permissions.apply(
                permission_mode, bypass_permissions
            )
            needs_confirm = needs_tool_confirmation(
                confirm_tool_calls = bool(confirm_tool_calls),
                bypass_permissions = bypass_permissions,
                permission_mode = permission_mode,
                name = decision.tool_name,
                arguments = decision.arguments,
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
                sandbox_level = sandbox_level,
            )
            approval_id = new_approval_id() if needs_confirm else ""
            decision_slot = begin_tool_decision(session_id, approval_id) if needs_confirm else None
            bind_permission_upgrade(decision_slot, live_permissions, eligible = image_share is None)
            if needs_confirm:
                note_host_access(approval_id, decision.tool_name, decision.arguments)
            start_event = decision.tool_start_event()
            start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirm
            if image_share is not None:
                start_event["image_disclosure"] = image_share["disclosure"]

            try:
                # A gated call has not started: say waiting, not "Running" (GGUF parity).
                yield {
                    "type": "status",
                    "text": (
                        awaiting_approval_status(decision.tool_name)
                        if needs_confirm
                        else decision.status_text
                    ),
                }
                yield start_event

                _decision = (
                    wait_tool_decision(
                        decision_slot,
                        approval_id,
                        cancel_event = cancel_event,
                    )
                    if decision_slot is not None
                    else None
                )
                # Read before decision_slot is dropped in the deny branch below.
                _decision_reason = decision_reason(decision_slot)
                permission_mode, bypass_permissions = live_permissions.apply(
                    permission_mode, bypass_permissions
                )
                if bypass_permissions:
                    strict_isolation = False
                if _decision is not None and _decision != "deny":
                    yield decision.tool_approved_event(approval_id)
                    yield {"type": "status", "text": decision.status_text}
                if _decision == "deny":
                    decision_slot = None
                    if provisional_match:
                        provisional_resolved = True
                    # An unanswered approval is not the user's decision; this string is the only record.
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
                    tool_denied = True
                    denied_message = {
                        "role": "tool",
                        "name": decision.tool_name,
                        "content": _denied_text,
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
                decision.tool_name in RAG_SEARCH_TOOLS
                and kb_search_count >= RAG_MAX_SEARCHES_PER_TURN
            ):
                result = RAG_SEARCH_CAP_NUDGE
            else:
                # Worker thread so stdout/heartbeats stream; host paths only for an answered call.
                _host_access_approved = _decision not in (None, "deny")

                def _invoke_tool(
                    _output_callback,
                    _decision = decision,
                    _approved = _host_access_approved,
                    _strict = strict_isolation,
                ):
                    kwargs = dict(
                        cancel_event = cancel_event,
                        timeout = eff_timeout,
                        session_id = session_id,
                        thread_id = thread_id,
                        rag_scope = rag_scope,
                        disable_sandbox = bypass_permissions,
                    )
                    if website_policy:
                        kwargs["website_policy"] = website_policy
                    # Run unasked only because the OS sandbox was on: refuse if it is not any more.
                    if _strict and _accepts_kwarg(execute_tool, "tool_execution_mode"):
                        kwargs["tool_execution_mode"] = "required"
                    elif runs_without_os_sandbox(
                        _decision.tool_name, sandbox_level
                    ) and _accepts_kwarg(execute_tool, "tool_execution_mode"):
                        kwargs["tool_execution_mode"] = "software"
                    if _accepts_kwarg(execute_tool, "conversation_branch"):
                        kwargs["conversation_branch"] = _extend_live_branch(conversation)
                    if _approved and _accepts_kwarg(execute_tool, "host_access_approved"):
                        kwargs["host_access_approved"] = True
                    # Without a budget the tool clamp is skipped and top_k=8 adds ~4K tokens.
                    if context_length and _accepts_kwarg(
                        execute_tool, "conversation_budget_tokens"
                    ):
                        from core.inference.context_window import retrieval_budget

                        # From the last reported token count, not characters (which hand out room the prompt lacks); result and reply share the budget (GGUF parity).
                        spent = _spent_prompt_tokens(
                            conversation,
                            tools,
                            generation_stats_holder,
                            prompt_dense_tokens,
                        )
                        kwargs["conversation_budget_tokens"] = retrieval_budget(
                            int(context_length),
                            max_tokens,
                            spent,
                            reply_returns = True,
                        )
                    # No rolling fit downstream, so cap every tool result, not just retrieval.
                    if context_length and _accepts_kwarg(execute_tool, "result_budget_tokens"):
                        from core.inference.context_window import (
                            estimate_messages_tokens_conservative as _spent_tokens,
                            estimate_messages_tokens_dense as _dense_tokens,
                            tool_result_budget,
                        )

                        # Tools layer cannot see a native model's window; pass it explicitly.
                        if _accepts_kwarg(execute_tool, "context_tokens"):
                            kwargs["context_tokens"] = int(context_length)
                        # Results priced at 2 chars/token: base64/JSON/hashes run denser than English.
                        results = [
                            message for message in conversation if message.get("role") == "tool"
                        ]
                        # Tool results excluded here and priced separately, else charged twice.
                        rest = [
                            message for message in conversation if message.get("role") != "tool"
                        ]
                        # Split across pending calls so the first does not take the batch's room (GGUF parity).
                        pending = list(tool_calls or [])[_call_index + 1 :]
                        pending_args = [
                            {"role": "assistant", "content": json.dumps(call, default = str)}
                            for call in pending
                        ]
                        # the notice restores skipped arguments after the retained results.
                        if over_cap:
                            pending_args.append(
                                tool_call_limit_nudge(over_cap, _MAX_TOOL_CALLS_PER_TURN)
                            )
                        kwargs["result_budget_tokens"] = tool_result_budget(
                            int(context_length),
                            max_tokens,
                            # Conservative for the whole thread: user turns can hold dense blobs too.
                            _spent_tokens(rest)
                            + _spent_tokens(tools or [])
                            # All ASCII of a result counts at two per token, since hexdumps and stack traces carry spaces.
                            + _spent_tokens(results, dense_ascii = True)
                            # Doubled: pending args can hold base64, minified JSON or code, which cannot be priced exactly.
                            + 2 * _dense_tokens(pending_args),
                        ) // (len(pending) + 1)
                    if _accepts_output_callback(execute_tool):
                        kwargs["output_callback"] = _output_callback
                    kwargs.update(_search_images_kwargs(execute_tool, _decision.tool_name))
                    if image_share is not None:
                        kwargs["mcp_image"] = image_share["image"]
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
                if decision.tool_name in RAG_SEARCH_TOOLS:
                    kb_search_count += 1

            completion = tool_controller.record_result(decision, result)
            if provisional_match:
                provisional_resolved = True
            # Loading MCP tools is not a tool round, or a budget of one could never call what it found.
            if decision.tool_name != "mcp_tool_search":
                _turn_executed_real_tool = True
            # Any executed call, discovery included, releases a forced choice (as on llama-server).
            forced_resolved = True
            yield completion.tool_end_event()
            conversation.append(completion.tool_message())
            # Parsed only when there is a sink for them.
            _completion_images = completion.mcp_images() if images_sink is not None else []
            if _completion_images:
                batch_mcp_images.append(_completion_images)

        # The turn that ends the loop offers no tools again, so do not ask for a retry there.
        over_cap_final = bool(over_cap) and (
            tool_controller.force_final_answer
            or (not unrestricted_tools and not tool_controller.active_tools())
            or (_turn_executed_real_tool and _executed_tool_iters + 1 >= max_tool_iterations)
        )
        if over_cap:
            deferred_noop_msgs.append(
                tool_call_limit_nudge(
                    over_cap,
                    _MAX_TOOL_CALLS_PER_TURN,
                    final = over_cap_final,
                    unavailable_tools = {
                        _limit_decision.tool_name
                        for call in over_cap
                        if (_limit_decision := tool_controller.prepare_call(call)).action
                        in ("disabled", "render_html_repeat")
                    },
                )
            )
        append_deferred_nudges(conversation, deferred_noop_msgs)
        if batch_mcp_images and images_sink is not None:
            # Sink only for image-reading models: pixels ride beside the prompt, which carries markers only.
            encoded = png_payloads_per_result(batch_mcp_images)
            if encoded:
                images_sink.extend(encoded)
                # Merged into the deferred nudge: two user turns in a row break strict VLM templates.
                _batch_results = sum(
                    1
                    for m in conversation[batch_conversation_start:]
                    if isinstance(m, dict) and m.get("role") == "tool"
                )
                append_placeholder_turn(
                    conversation,
                    len(encoded),
                    sum(len(r) for r in batch_mcp_images),
                    lead = MCP_DETACHED_IMAGE_TURN_TEXT
                    if _batch_results != 1
                    else MCP_IMAGE_TURN_TEXT,
                )
                # Rebased: this trim deletes entries before the attachment, and a reused old index would protect the wrong payload and delete the attachment.
                caller_images = trim_image_turns(conversation, images_sink, keep = caller_images)

        yield {"type": "status", "text": ""}
        forced_resolved = forced_resolved or tool_denied

        if tool_controller.force_final_answer:
            final_attempt_done = True
            continue
        if not unrestricted_tools and not tool_controller.active_tools():
            final_attempt_done = True
            continue
        # Only real tool turns count against the cap, so a no-op correction turn is free (GGUF parity).
        if _turn_executed_real_tool:
            _executed_tool_iters += 1
            post_tool_reprompts = 0
            if browser_turns:
                # A new step may stall in the same words as the last one (external loop parity).
                last_reprompt_text = ""
        if _executed_tool_iters >= max_tool_iterations and not final_attempt_done:
            final_attempt_done = True
            # Ride a trailing user turn, as the GGUF loop does.
            if over_cap or conversation[-1].get("role") == "user":
                conversation[-1] = {
                    **conversation[-1],
                    "content": with_appended_text(
                        conversation[-1]["content"], f"\n\n{BUDGET_EXHAUSTED_NUDGE}"
                    ),
                }
            else:
                conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})

    yield {"type": "status", "text": ""}
