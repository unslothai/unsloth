# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-call healing for the client-tool passthrough.

With server-side tools disabled (``unsloth run --disable-tools``, every
``unsloth start`` coding agent), requests carrying the client's own ``tools``
bypass Studio's tool loop and are relayed to/from llama-server verbatim. Small
GGUF models often emit their tool calls as TEXT (``<tool_call>{...}</tool_call>``,
Gemma ``<|tool_call>...``, ``<function=...>`` XML) instead of structured
``tool_calls`` -- on the passthrough that text reaches the agent as prose and
the turn dies. This module promotes such text back into structured calls on the
RESPONSE side only: the upstream request body is never touched, no extra
generation is issued, so llama-server slot/KV-cache reuse is byte-identical.

Healing only ever fires when the request declared client tools, and only
promotes calls whose function name exactly matches a declared tool. Responses
without a tool signal, requests without tools, and Studio's own enable-tools
loop are untouched. Per-request opt-out: ``auto_heal_tool_calls: false``.
Process kill-switch: ``UNSLOTH_DISABLE_TOOL_CALL_HEALING=1``.
"""

import json
import os
from typing import Any, Optional

from core.inference.tool_call_parser import TOOL_XML_SIGNALS, has_tool_signal
from core.inference.tool_loop_controller import coerce_tool_arguments
from core.tool_healing import parse_tool_calls_from_text, strip_tool_call_markup

# Read once at import (same convention as the other UNSLOTH_* switches).
_HEALING_DISABLED = os.environ.get("UNSLOTH_DISABLE_TOOL_CALL_HEALING", "0") == "1"

_MAX_SIGNAL_LEN = max(len(s) for s in TOOL_XML_SIGNALS)
# A suspected-but-unclosed tool block larger than this is declared a false
# alarm and flushed, bounding memory on a model rambling XML-lookalike text.
_MAX_HOLD_CHARS = 64 * 1024


def heal_gate(auto_heal: Optional[bool], tools: Optional[list]) -> Optional[set]:
    """Return the declared client-tool name set when healing applies, else None.

    ``tools`` is the OpenAI-shaped list forwarded to llama-server
    (``[{"type": "function", "function": {"name": ...}}, ...]``). The name set
    doubles as the promotion allowlist so healed calls can never invent a tool
    the client did not declare.
    """
    if _HEALING_DISABLED or auto_heal is False:
        return None
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names or None


def _promote(calls: list, allowed_tools: set, id_offset: int = 0) -> list:
    """Filter parsed calls to declared tools and normalize their arguments.

    ``function.arguments`` leaves ``parse_tool_calls_from_text`` as whatever the
    model wrote; OpenAI clients require a JSON-object string, so coerce through
    the same canonical-key healing the enable-tools loop uses.
    """
    promoted = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name not in allowed_tools:
            continue
        coerced = coerce_tool_arguments(function.get("arguments"), heal = True, tool_name = name)
        promoted.append(
            {
                "id": f"call_{id_offset + len(promoted)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(coerced.arguments, ensure_ascii = False),
                },
            }
        )
    return promoted


def heal_openai_message(msg: dict, allowed_tools: set) -> bool:
    """Promote text-form tool calls in a non-streaming OpenAI message. In place.

    No-op (returns False) unless the message has NO structured ``tool_calls``
    (grammar mode already worked when it does) and its content carries a tool
    signal that parses into at least one declared tool.
    """
    if not isinstance(msg, dict) or msg.get("tool_calls"):
        return False
    content = msg.get("content")
    if not isinstance(content, str) or not has_tool_signal(content):
        return False
    calls = _promote(
        parse_tool_calls_from_text(content, allow_incomplete = False), allowed_tools
    )
    if not calls:
        return False
    msg["tool_calls"] = calls
    # OpenAI requires content = null on a pure tool-call turn.
    msg["content"] = strip_tool_call_markup(content, final = True) or None
    return True


def _earliest_signal(buffer: str) -> int:
    best = -1
    for signal in TOOL_XML_SIGNALS:
        index = buffer.find(signal)
        if index >= 0 and (best < 0 or index < best):
            best = index
    return best


def _partial_signal_suffix(buffer: str) -> int:
    """Length of the longest buffer suffix that is a proper prefix of a signal."""
    for length in range(min(len(buffer), _MAX_SIGNAL_LEN - 1), 0, -1):
        tail = buffer[-length:]
        if any(signal.startswith(tail) for signal in TOOL_XML_SIGNALS):
            return length
    return 0


class StreamToolCallHealer:
    """Buffer-and-repair state machine for streamed passthrough content.

    ``feed(text)`` / ``finalize()`` yield ``("text", str)`` events for content
    to relay and ``("tool_call", dict)`` events carrying an OpenAI-shaped call
    (string ``function.arguments``). Normal prose is forwarded immediately; only
    a trailing partial-signal window (< max signal length) or a suspected tool
    block is ever withheld, so streaming latency stays bounded. A false alarm
    (the buffer can no longer become a parseable declared call) flushes the held
    text verbatim.
    """

    def __init__(self, allowed_tools: set) -> None:
        self._allowed = set(allowed_tools)
        self._buffer = ""
        self._holding = False
        self._id_offset = 0
        # Structured delta.tool_calls seen upstream: grammar mode already
        # worked, so healing goes dormant and text relays verbatim.
        self.dormant = False

    @property
    def healed(self) -> bool:
        return self._id_offset > 0

    def structured_tool_call_seen(self) -> list:
        """Go dormant; flush anything held so no text is swallowed."""
        self.dormant = True
        held, self._buffer, self._holding = self._buffer, "", False
        return [("text", held)] if held else []

    def feed(self, text: str) -> list:
        if self.dormant:
            return [("text", text)] if text else []
        self._buffer += text
        return self._drain()

    def _drain(self) -> list:
        events: list = []
        while True:
            if not self._holding:
                start = _earliest_signal(self._buffer)
                if start >= 0:
                    if start:
                        events.append(("text", self._buffer[:start]))
                    self._buffer = self._buffer[start:]
                    self._holding = True
                else:
                    keep = _partial_signal_suffix(self._buffer)
                    emit = self._buffer[: len(self._buffer) - keep]
                    if emit:
                        events.append(("text", emit))
                    self._buffer = self._buffer[len(self._buffer) - keep :]
                    return events
            # HOLD: promote once at least one complete block parses.
            parsed = parse_tool_calls_from_text(
                self._buffer, id_offset = self._id_offset, allow_incomplete = False
            )
            if not parsed:
                if len(self._buffer) > _MAX_HOLD_CHARS:
                    events.append(("text", self._buffer))
                    self._buffer = ""
                    self._holding = False
                    continue
                return events
            promoted = _promote(parsed, self._allowed, id_offset = self._id_offset)
            if not promoted:
                # Complete blocks, but none names a declared tool: false alarm.
                # Flush the raw text so the client still sees what the model said.
                events.append(("text", self._buffer))
                self._buffer = ""
                self._holding = False
                continue
            self._id_offset += len(promoted)
            events.extend(("tool_call", call) for call in promoted)
            # Drop the closed markup, keep surrounding text/partial blocks, and
            # loop back to rescan the remainder (e.g. a second call following).
            self._buffer = strip_tool_call_markup(self._buffer, final = False)
            self._holding = False

    def finalize(self) -> list:
        """End of stream: last-chance heal of the residue, else flush it."""
        if not self._buffer:
            return []
        residue, self._buffer = self._buffer, ""
        holding, self._holding = self._holding, False
        if self.dormant or not holding:
            return [("text", residue)]
        promoted = _promote(
            parse_tool_calls_from_text(
                residue, id_offset = self._id_offset, allow_incomplete = True
            ),
            self._allowed,
            id_offset = self._id_offset,
        )
        if not promoted:
            return [("text", residue)]
        self._id_offset += len(promoted)
        events = [("tool_call", call) for call in promoted]
        remainder = strip_tool_call_markup(residue, final = True)
        if remainder:
            events.append(("text", remainder))
        return events


def _last_assistant_text(data: Any) -> str:
    """First-choice assistant content of a non-streaming chat response, or ''."""
    try:
        message = data["choices"][0]["message"]
        content = message.get("content")
        return content if isinstance(content, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""


def nudge_should_retry(data: Any, allowed_tools: Optional[set]) -> bool:
    """True when the first response tried to call a tool but nothing healed.

    Trigger only on: healing enabled (allowed_tools set), zero structured
    calls, a tool signal present in the text, and zero promotable calls -- the
    exact failure a single re-ask can fix. Clean prose never retries.
    """
    if not allowed_tools:
        return False
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return False
    if message.get("tool_calls"):
        return False
    text = message.get("content")
    if not isinstance(text, str) or not has_tool_signal(text):
        return False
    return not _promote(
        parse_tool_calls_from_text(text, allow_incomplete = True), allowed_tools
    )


def nudge_messages(data: Any, allowed_tools: set) -> list:
    """The two-message suffix appended for the single nudge retry.

    The retry body is the original body plus this suffix, so the prompt prefix
    is byte-identical and llama-server's slot/prefix cache is reused (same
    shape as the enable-tools loop's reprompt).
    """
    tool_hint = " or ".join(f"`{name}`" for name in sorted(allowed_tools)) or "an available tool"
    return [
        {"role": "assistant", "content": _last_assistant_text(data)},
        {
            "role": "user",
            "content": (
                "You have access to the declared tools. If a tool is needed to "
                f"complete the action you described, call {tool_hint} now using the "
                "native tool-call format with valid JSON arguments, not prose. If no "
                "tool is needed, provide the final answer directly."
            ),
        },
    ]
