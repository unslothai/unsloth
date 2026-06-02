# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Safetensors/transformers agentic tool loop.

Wraps a single-turn cumulative-text generator (the existing
``InferenceOrchestrator.generate_chat_response`` pipeline that streams
from a worker subprocess) with the tool-calling, thinking-block,
status, and metadata event protocol used by the GGUF path. Keeps the
front-end SSE shape identical across backends so the chat UI does not
care which engine actually ran the model.

The GGUF path lives in ``llama_cpp.py`` and talks to llama-server's
structured ``delta.tool_calls`` directly. Native transformers has no
such structured channel, so this loop parses tool calls from the
cumulative text and dispatches them via ``core.inference.tools``.
"""

import json
import re
import threading
from typing import Callable, Generator, Optional
from urllib.parse import urlparse

from loggers import get_logger

from core.inference.tool_call_parser import (
    BUDGET_EXHAUSTED_NUDGE,
    DUPLICATE_CALL_NUDGE,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    RENDER_HTML_REPEAT_NUDGE,
    TOOL_ERROR_NUDGE,
    TOOL_ERROR_PREFIXES,
    TOOL_XML_SIGNALS,
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)


logger = get_logger(__name__)


# Buffer cap while waiting to disambiguate a possible tool-call prefix.
_MAX_BUFFER_CHARS = 32


def _status_for_tool(tool_name: str, arguments: dict) -> str:
    """Return a human-readable status line matching the GGUF path."""
    if tool_name == "web_search":
        url = (arguments.get("url") or "").strip()
        if url:
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https") and parsed.hostname:
                host = parsed.hostname
                if host.startswith("www."):
                    host = host[4:]
                return f"Reading: {host}"
            return "Reading page..."
        query = arguments.get("query", "")
        return f"Searching: {query}"
    if tool_name == "python":
        preview = (arguments.get("code") or "").strip().split("\n")[0][:60]
        return f"Running Python: {preview}" if preview else "Running Python..."
    if tool_name == "terminal":
        preview = (arguments.get("command") or "")[:60]
        return f"Running: {preview}" if preview else "Running command..."
    return f"Calling: {tool_name}"


_CANONICAL_HEAL_ARG = {
    "python": "code",
    "terminal": "command",
    "render_html": "code",
}


_FUNCTION_SIGNAL_RE = re.compile(r"<function=([\w-]+)>")
_TOOL_CALL_NAME_RE = re.compile(r'"name"\s*:\s*"([\w-]+)"')


def _detect_render_html_tool_start(content: str) -> bool:
    """Return True when the first drained tool call is clearly render_html."""
    function_match = _FUNCTION_SIGNAL_RE.search(content)
    tool_call_index = content.find("<tool_call>")
    if not function_match and tool_call_index < 0:
        return False

    if function_match and (
        tool_call_index < 0 or function_match.start() < tool_call_index
    ):
        return function_match.group(1) == "render_html"

    if tool_call_index >= 0:
        name_match = _TOOL_CALL_NAME_RE.search(content[tool_call_index:])
        return bool(name_match and name_match.group(1) == "render_html")

    return False


def _coerce_arguments(raw_args, *, heal: bool, tool_name: str = "") -> dict:
    """Normalise tool ``arguments`` to a dict.

    Some templates emit a JSON string, others a bare query string. With
    ``heal=True`` we accept a bare string as ``{<canonical_key>: ...}``
    so a Hermes-style call without proper JSON still runs the tool. The
    canonical key is picked per tool: ``code`` for python, ``command``
    for terminal, ``query`` for everything else (e.g. web_search).
    """
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        if heal:
            key = _CANONICAL_HEAL_ARG.get(tool_name, "query")
            return {key: raw_args}
        return {"raw": raw_args}
    return {}


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
) -> Generator[dict, None, None]:
    """Drive an agentic tool loop on top of a cumulative-text generator.

    ``single_turn(messages)`` must yield cumulative assistant text
    (each yield is a snapshot including all previously emitted tokens).
    The loop:

    * Buffers the leading characters of every turn so it can decide
      whether the model is about to emit a tool call. Plain content
      starts streaming as soon as the buffer rules it out.
    * On detecting ``<tool_call>`` or ``<function=`` in the cumulative
      text, drains the rest of the turn silently and parses tool calls
      out of the full content.
    * Executes each tool via ``execute_tool``, appends the assistant
      tool-call message and the tool result to the conversation, and
      re-enters ``single_turn`` for the next iteration.
    * After ``max_tool_iterations`` turns without a final answer, asks
      the model once more to produce a final answer with no tools.

    Yields event dicts matching the GGUF path:

    * ``{"type": "status", "text": ...}`` -- empty string clears the badge.
    * ``{"type": "content", "text": ...}`` -- cumulative cleaned text for
      the current assistant turn (the consumer should diff against its
      own ``prev_text`` cursor).
    * ``{"type": "tool_start", "tool_name", "tool_call_id", "arguments"}``
    * ``{"type": "tool_end", "tool_name", "tool_call_id", "result"}``
    """
    conversation = list(messages)

    # Forced first-pass RAG (mirrors the GGUF loop): splice attached-doc passages
    # + citations in before the model answers, gated on a cosine floor, so doc
    # questions don't lose to web_search.
    from core.inference.tools import build_rag_autoinject

    _auto = build_rag_autoinject(conversation, rag_scope)
    if _auto:
        for _ev in _auto["events"]:
            yield _ev
        conversation.extend(_auto["messages"])

    tool_call_history: list[tuple[str, bool]] = []
    render_html_succeeded = False
    kb_search_count = 0  # executed search_knowledge_base calls this turn
    final_attempt_done = False
    allowed_tool_names = {
        (tool.get("function") or {}).get("name")
        for tool in (tools or [])
        if (tool.get("function") or {}).get("name")
    }
    next_call_id = 0

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

        detect_state = _state_buffering
        content_buffer = ""
        content_accum = ""
        cumulative_display = ""
        last_emitted = ""
        provisional_render_html_started = False
        provisional_render_html_id = f"call_{next_call_id}"

        gen = single_turn(conversation)
        prev_cumulative = ""

        for cumulative in gen:
            if cancel_event is not None and cancel_event.is_set():
                return

            if not isinstance(cumulative, str):
                continue  # defensive: pipeline only yields strings

            delta = cumulative[len(prev_cumulative) :]
            prev_cumulative = cumulative
            if not delta:
                continue
            content_accum += delta

            if detect_state == _state_draining:
                if (
                    not render_html_succeeded
                    and not provisional_render_html_started
                    and _detect_render_html_tool_start(content_accum)
                ):
                    provisional_render_html_started = True
                    yield {
                        "type": "tool_start",
                        "tool_name": "render_html",
                        "tool_call_id": provisional_render_html_id,
                        "arguments": {},
                    }
                continue

            if detect_state == _state_streaming:
                candidate = cumulative_display + delta
                signal_pos = -1
                for sig in TOOL_XML_SIGNALS:
                    p = candidate.find(sig)
                    if p >= 0 and (signal_pos < 0 or p < signal_pos):
                        signal_pos = p
                if signal_pos >= 0:
                    before_tool = candidate[:signal_pos]
                    cleaned_before = strip_tool_markup(before_tool)
                    if len(cleaned_before) > len(last_emitted):
                        last_emitted = cleaned_before
                        yield {"type": "content", "text": cleaned_before}
                    cumulative_display = candidate
                    detect_state = _state_draining
                    if (
                        not render_html_succeeded
                        and not provisional_render_html_started
                        and _detect_render_html_tool_start(content_accum)
                    ):
                        provisional_render_html_started = True
                        yield {
                            "type": "tool_start",
                            "tool_name": "render_html",
                            "tool_call_id": provisional_render_html_id,
                            "arguments": {},
                        }
                    continue
                cumulative_display = candidate
                cleaned = strip_tool_markup(cumulative_display)
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
            for sig in TOOL_XML_SIGNALS:
                if stripped.startswith(sig):
                    is_match = True
                    break
                if sig.startswith(stripped):
                    is_prefix = True
                    break

            if is_match:
                detect_state = _state_draining
                if (
                    not render_html_succeeded
                    and not provisional_render_html_started
                    and _detect_render_html_tool_start(content_accum)
                ):
                    provisional_render_html_started = True
                    yield {
                        "type": "tool_start",
                        "tool_name": "render_html",
                        "tool_call_id": provisional_render_html_id,
                        "arguments": {},
                    }
            elif is_prefix and len(stripped) < _MAX_BUFFER_CHARS:
                continue
            else:
                detect_state = _state_streaming
                cumulative_display += content_buffer
                cleaned = strip_tool_markup(cumulative_display)
                if len(cleaned) > len(last_emitted):
                    last_emitted = cleaned
                    yield {"type": "content", "text": cleaned}

        # Stream finished -- resolve what we collected.
        if cancel_event is not None and cancel_event.is_set():
            return

        if detect_state == _state_buffering:
            # Buffer never resolved -- tool XML or plain content.
            stripped = content_buffer.lstrip()
            if stripped and has_tool_signal(stripped):
                detect_state = _state_draining
            else:
                if content_buffer:
                    cumulative_display += content_buffer
                    yield {
                        "type": "content",
                        "text": strip_tool_markup(cumulative_display, final = True),
                    }
                yield {"type": "status", "text": ""}
                return

        if detect_state == _state_streaming:
            # No tool detected mid-stream -- check for late tool XML.
            safety_tc = None
            if has_tool_signal(content_accum):
                safety_tc = parse_tool_calls_from_text(
                    content_accum,
                    id_offset = next_call_id,
                )
            if not safety_tc:
                # Final answer: streaming already emitted content.
                # Skip a final=True re-strip so literal "<tool_call>"
                # in prose survives when no real tool call parsed.
                yield {"type": "status", "text": ""}
                return
            tool_calls = safety_tc
            content_text = strip_tool_markup(content_accum, final = True)
            logger.info(
                "Safetensors safety net: parsed %d tool call(s) from streamed content",
                len(tool_calls),
            )
        else:
            # DRAINING: parse tool calls out of full content.
            tool_calls = parse_tool_calls_from_text(
                content_accum,
                id_offset = next_call_id,
            )
            if not tool_calls and auto_heal_tool_calls:
                # Parser found nothing -- surface raw content so any
                # literal "<tool_call>" prose is preserved.
                if content_accum:
                    yield {"type": "content", "text": content_accum}
                if provisional_render_html_started:
                    yield {
                        "type": "tool_end",
                        "tool_name": "render_html",
                        "tool_call_id": provisional_render_html_id,
                        "result": "Error: render_html tool call could not be parsed.",
                    }
                yield {"type": "status", "text": ""}
                return
            content_text = strip_tool_markup(content_accum, final = True)

        if final_attempt_done:
            # Final-answer turn re-called a tool -- stop the loop.
            if content_text:
                yield {"type": "content", "text": content_text}
            yield {"type": "status", "text": ""}
            return

        assistant_msg: dict = {"role": "assistant", "content": content_text}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            next_call_id += len(tool_calls)
        conversation.append(assistant_msg)

        for tc in tool_calls or []:
            func = tc.get("function", {}) or {}
            tool_name = func.get("name", "") or ""
            arguments = _coerce_arguments(
                func.get("arguments", {}),
                heal = auto_heal_tool_calls,
                tool_name = tool_name,
            )

            repeat_render_html = tool_name == "render_html" and render_html_succeeded
            if not repeat_render_html:
                yield {"type": "status", "text": _status_for_tool(tool_name, arguments)}
                yield {
                    "type": "tool_start",
                    "tool_name": tool_name,
                    "tool_call_id": tc.get("id", ""),
                    "arguments": arguments,
                }

            tc_key = tool_name + str(arguments)
            if repeat_render_html:
                result = RENDER_HTML_REPEAT_NUDGE
            elif allowed_tool_names and tool_name not in allowed_tool_names:
                result = (
                    f"Error: tool '{tool_name}' is not enabled for this "
                    "request. Use one of the enabled tools or provide a "
                    "final answer."
                )
            elif (
                tool_name == "search_knowledge_base"
                and kb_search_count >= RAG_MAX_SEARCHES_PER_TURN
            ):
                result = RAG_SEARCH_CAP_NUDGE
            else:
                already_ran_ok = any(
                    k == tc_key and not err for k, err in tool_call_history
                )
                if already_ran_ok:
                    result = DUPLICATE_CALL_NUDGE
                else:
                    eff_timeout = (
                        None if tool_call_timeout >= 9999 else tool_call_timeout
                    )
                    try:
                        result = execute_tool(
                            tool_name,
                            arguments,
                            cancel_event = cancel_event,
                            timeout = eff_timeout,
                            session_id = session_id,
                            rag_scope = rag_scope,
                        )
                    except Exception as exc:
                        logger.exception("Tool %s raised: %s", tool_name, exc)
                        result = f"Error: tool raised an exception: {exc}"
                    if tool_name == "search_knowledge_base":
                        kb_search_count += 1

            if not repeat_render_html:
                yield {
                    "type": "tool_end",
                    "tool_name": tool_name,
                    "tool_call_id": tc.get("id", ""),
                    "result": result,
                }

            is_error = isinstance(result, str) and result.lstrip().startswith(
                TOOL_ERROR_PREFIXES
            )
            if tool_name == "render_html" and not is_error:
                render_html_succeeded = True
            tool_call_history.append((tc_key, is_error))

            # Strip frontend image sentinel from the model's view.
            # Cut at the first occurrence so leading and consecutive
            # sentinels are both removed.
            result_for_model = result
            if isinstance(result_for_model, str) and "__IMAGES__:" in result_for_model:
                result_for_model = result_for_model.split("__IMAGES__:", 1)[0].rstrip()
            # Strip the RAG citation source-map (kept for the UI via tool_end).
            if (
                isinstance(result_for_model, str)
                and "__RAG_SOURCES__:" in result_for_model
            ):
                result_for_model = result_for_model.split("__RAG_SOURCES__:", 1)[
                    0
                ].rstrip()
            if is_error:
                result_for_model = result_for_model + TOOL_ERROR_NUDGE

            tool_msg: dict = {
                "role": "tool",
                "name": tool_name,
                "content": result_for_model,
            }
            tool_call_id = tc.get("id")
            if tool_call_id:
                tool_msg["tool_call_id"] = tool_call_id
            conversation.append(tool_msg)

        # Clear the status badge before the next turn.
        yield {"type": "status", "text": ""}

        if iteration + 1 >= max_tool_iterations and not final_attempt_done:
            # Budget exhausted; nudge a final plain answer.
            final_attempt_done = True
            conversation.append(
                {
                    "role": "user",
                    "content": BUDGET_EXHAUSTED_NUDGE,
                }
            )

    yield {"type": "status", "text": ""}
