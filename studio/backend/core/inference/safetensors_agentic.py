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
import threading
from typing import Callable, Generator, Optional
from urllib.parse import urlparse

from loggers import get_logger

from core.inference.tool_call_parser import (
    TOOL_XML_SIGNALS,
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)


logger = get_logger(__name__)


# Maximum prefix length we will buffer while waiting to decide whether
# the model is about to emit ``<tool_call>`` or ``<function=``. Set just
# above the longest signal prefix to give a small safety margin.
_MAX_BUFFER_CHARS = 32

# Tool messages always reach the user via SSE events; the assistant
# turn that emitted them is replaced with a stripped version so the
# chat history does not show raw XML.
_ERROR_PREFIXES = (
    "Error",
    "Search failed",
    "Execution error",
    "Blocked:",
    "Exit code",
    "Failed to fetch",
    "Failed to resolve",
    "No query provided",
)


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


def _coerce_arguments(raw_args, *, heal: bool) -> dict:
    """Normalise tool ``arguments`` to a dict.

    Some templates emit a JSON string, others a bare query string. With
    ``heal=True`` we accept a bare string as ``{"query": ...}`` so a
    Hermes-style call without proper JSON still runs the tool.
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
        return {"query": raw_args} if heal else {"raw": raw_args}
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
    tool_call_history: list[tuple[str, bool]] = []
    final_attempt_done = False

    for iteration in range(max_tool_iterations + 1):
        if cancel_event is not None and cancel_event.is_set():
            return

        _state_buffering = 0
        _state_streaming = 1
        _state_draining = 2

        detect_state = _state_buffering
        content_buffer = ""
        content_accum = ""
        cumulative_display = ""
        last_emitted = ""

        gen = single_turn(conversation)
        prev_cumulative = ""

        for cumulative in gen:
            if cancel_event is not None and cancel_event.is_set():
                return

            if not isinstance(cumulative, str):
                # The worker pipeline only yields strings; defensive
                # skip in case a future change starts yielding dicts.
                continue

            delta = cumulative[len(prev_cumulative) :]
            prev_cumulative = cumulative
            if not delta:
                continue
            content_accum += delta

            if detect_state == _state_draining:
                continue

            if detect_state == _state_streaming:
                cumulative_display += delta
                cleaned = strip_tool_markup(cumulative_display)
                if len(cleaned) > len(last_emitted):
                    last_emitted = cleaned
                    yield {"type": "content", "text": cleaned}
                continue

            # BUFFERING: hold leading content until we know it is not a
            # tool call.
            content_buffer += delta
            stripped = content_buffer.lstrip()
            if not stripped:
                continue

            is_match = False
            is_prefix = False
            for sig in TOOL_XML_SIGNALS if auto_heal_tool_calls else ():
                if stripped.startswith(sig):
                    is_match = True
                    break
                if sig.startswith(stripped):
                    is_prefix = True
                    break

            if is_match:
                detect_state = _state_draining
            elif is_prefix and len(stripped) < _MAX_BUFFER_CHARS:
                continue
            else:
                detect_state = _state_streaming
                cumulative_display += content_buffer
                cleaned = strip_tool_markup(cumulative_display)
                if len(cleaned) > len(last_emitted):
                    last_emitted = cleaned
                    yield {"type": "content", "text": cleaned}

        # Stream finished. Decide what to do with what we collected.
        if cancel_event is not None and cancel_event.is_set():
            return

        if detect_state == _state_buffering:
            # Buffer never resolved. Treat any leaked tool XML as a
            # tool call, otherwise emit the buffer as plain content.
            stripped = content_buffer.lstrip()
            if (
                stripped
                and auto_heal_tool_calls
                and has_tool_signal(stripped)
            ):
                detect_state = _state_draining
            else:
                if content_buffer:
                    cumulative_display += content_buffer
                    yield {
                        "type": "content",
                        "text": strip_tool_markup(
                            cumulative_display, final=True
                        ),
                    }
                yield {"type": "status", "text": ""}
                return

        if detect_state == _state_streaming:
            # No tool detected this iteration. Either we are done or
            # we caught a tool-call XML late in the stream.
            safety_tc = None
            if auto_heal_tool_calls and has_tool_signal(content_accum):
                safety_tc = parse_tool_calls_from_text(content_accum)
            if not safety_tc:
                # Final answer arrived. Flush and exit.
                if cumulative_display:
                    cleaned = strip_tool_markup(cumulative_display, final=True)
                    if cleaned and cleaned != last_emitted:
                        yield {"type": "content", "text": cleaned}
                yield {"type": "status", "text": ""}
                return
            tool_calls = safety_tc
            content_text = strip_tool_markup(content_accum, final=True)
            logger.info(
                "Safetensors safety net: parsed %d tool call(s) from streamed content",
                len(tool_calls),
            )
        else:
            # DRAINING: parse the tool calls out of the full content.
            tool_calls = parse_tool_calls_from_text(content_accum)
            if not tool_calls and auto_heal_tool_calls:
                # Drained but parser found nothing. Treat as plain text.
                cleaned = strip_tool_markup(content_accum, final=True)
                if cleaned:
                    yield {"type": "content", "text": cleaned}
                yield {"type": "status", "text": ""}
                return
            content_text = strip_tool_markup(content_accum, final=True)

        if final_attempt_done:
            # We already asked the model for a final answer and it tried
            # to call another tool. Stop here so we do not loop forever.
            if content_text:
                yield {"type": "content", "text": content_text}
            yield {"type": "status", "text": ""}
            return

        assistant_msg: dict = {"role": "assistant", "content": content_text}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        conversation.append(assistant_msg)

        for tc in tool_calls or []:
            func = tc.get("function", {}) or {}
            tool_name = func.get("name", "") or ""
            arguments = _coerce_arguments(
                func.get("arguments", {}),
                heal=auto_heal_tool_calls,
            )

            yield {"type": "status", "text": _status_for_tool(tool_name, arguments)}
            yield {
                "type": "tool_start",
                "tool_name": tool_name,
                "tool_call_id": tc.get("id", ""),
                "arguments": arguments,
            }

            tc_key = tool_name + str(arguments)
            prev = tool_call_history[-1] if tool_call_history else None
            if prev and prev[0] == tc_key and not prev[1]:
                result = (
                    "You already made this exact call. Do not repeat the "
                    "same tool call. Try a different approach: fetch a URL "
                    "from previous results, use Python to process data you "
                    "already have, or provide your final answer now."
                )
            else:
                eff_timeout = (
                    None if tool_call_timeout >= 9999 else tool_call_timeout
                )
                try:
                    result = execute_tool(
                        tool_name,
                        arguments,
                        cancel_event=cancel_event,
                        timeout=eff_timeout,
                        session_id=session_id,
                    )
                except Exception as exc:
                    logger.exception("Tool %s raised: %s", tool_name, exc)
                    result = f"Error: tool raised an exception: {exc}"

            yield {
                "type": "tool_end",
                "tool_name": tool_name,
                "tool_call_id": tc.get("id", ""),
                "result": result,
            }

            is_error = isinstance(result, str) and result.lstrip().startswith(
                _ERROR_PREFIXES
            )
            tool_call_history.append((tc_key, is_error))

            # Strip frontend image sentinel before feeding the result
            # back to the model so it does not see UI plumbing.
            result_for_model = result
            if isinstance(result_for_model, str) and "\n__IMAGES__:" in result_for_model:
                result_for_model = result_for_model.rsplit("\n__IMAGES__:", 1)[0]
            if is_error:
                result_for_model = (
                    result_for_model + "\n\nThe tool call encountered an issue. "
                    "Please try a different approach or rephrase your request."
                )

            tool_msg: dict = {
                "role": "tool",
                "name": tool_name,
                "content": result_for_model,
            }
            tool_call_id = tc.get("id")
            if tool_call_id:
                tool_msg["tool_call_id"] = tool_call_id
            conversation.append(tool_msg)

        # Clear the status badge before the next generation turn.
        yield {"type": "status", "text": ""}

        if iteration + 1 >= max_tool_iterations and not final_attempt_done:
            # Budget exhausted; nudge the model for a final plain
            # answer on the next iteration.
            final_attempt_done = True
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "You have used all available tool calls. Based on "
                        "everything you have found so far, provide your "
                        "final answer now. Do not call any more tools."
                    ),
                }
            )

    yield {"type": "status", "text": ""}
