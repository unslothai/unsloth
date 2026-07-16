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
promotes calls whose function name exactly matches a declared tool. Promotion
removes EXACTLY the promoted calls' markup spans (the parser reports them):
undeclared calls, unparseable blocks, and suppressed alternate formats keep
every byte and relay as text, so healing can never silently delete model
output. Responses without a tool signal, requests without tools, and Studio's
own enable-tools loop are untouched. Per-request opt-out:
``auto_heal_tool_calls: false``. Process kill-switch:
``UNSLOTH_DISABLE_TOOL_CALL_HEALING=1``.
"""

import json
import os
from collections.abc import Mapping
from typing import Any, Optional

from core.inference.tool_loop_controller import coerce_tool_arguments
from core.tool_healing import parse_tool_calls_from_text

# Only the formats this healer's parser can promote -- narrower than the loops'
# broader TOOL_XML_SIGNALS. A loop-only marker (Llama <|python_tag|>, bare
# [ARGS]) would buffer a streamed call as prose without promoting it, so keep a
# healer-aligned list. Mistral's [TOOL_CALLS] IS promotable, so it stays in.
_HEAL_SIGNALS = (
    "<tool_call>",
    "<|tool_call>",
    "<function=",
    "[TOOL_CALLS]",
    # TML Inkling native call marker (leaks as text when the server-side
    # parser misses a narration-then-call turn).
    "<|content_invoke_tool_json|>",
)


def _has_heal_signal(text: str) -> bool:
    return any(s in text for s in _HEAL_SIGNALS)


# Read once at import (same convention as the other UNSLOTH_* switches).
_HEALING_DISABLED = os.environ.get("UNSLOTH_DISABLE_TOOL_CALL_HEALING", "0") == "1"
# Nudging is OPT-IN: per-request nudge_tool_calls=true, or flip the process
# default with UNSLOTH_TOOL_CALL_NUDGE=1 (e.g. an `unsloth run` operator).
_NUDGE_DEFAULT = os.environ.get("UNSLOTH_TOOL_CALL_NUDGE", "0") == "1"


def nudge_enabled(request_flag: Optional[bool]) -> bool:
    return _NUDGE_DEFAULT if request_flag is None else bool(request_flag)


_MAX_SIGNAL_LEN = max(len(s) for s in _HEAL_SIGNALS)
# A suspected-but-unclosed tool block larger than this is declared a false
# alarm and flushed, bounding memory on a model rambling XML-lookalike text.
_MAX_HOLD_CHARS = 64 * 1024


def heal_gate(
    auto_heal: Optional[bool],
    tools: Optional[list],
    tool_choice: Any = None,
) -> Optional[set]:
    """Return the declared client-tool name set when healing applies, else None.

    ``tools`` is the OpenAI-shaped list forwarded to llama-server
    (``[{"type": "function", "function": {"name": ...}}, ...]``). The name set
    doubles as the promotion allowlist so healed calls can never invent a tool
    the client did not declare.

    ``tool_choice`` (OpenAI shape) constrains the allowlist so healing never
    contradicts the request: ``"none"`` forbids tool calls outright (text-form
    markup stays text), and a forced ``{"type": "function", "function":
    {"name": N}}`` narrows promotion to that one function. ``"auto"`` /
    ``"required"`` / absent keep the full declared set.
    """
    if _HEALING_DISABLED or auto_heal is False:
        return None
    if tool_choice == "none":
        return None
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        forced = function.get("name") if isinstance(function, dict) else None
        if isinstance(forced, str):
            names &= {forced}
    return names or None


def _tool_schemas_by_name(tools: Optional[list]) -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str):
            schemas[name] = function.get("parameters")
    return schemas


def _string_arg_key_from_schema(schema: Any) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return None
    required_names = [name for name in required if isinstance(name, str)]
    if len(required_names) != 1:
        return None
    key = required_names[0]

    if key not in properties:
        return None
    prop_schema = properties.get(key)
    if isinstance(prop_schema, dict):
        prop_type = prop_schema.get("type")
        if isinstance(prop_type, list):
            if "string" not in prop_type:
                return None
        elif prop_type is not None and prop_type != "string":
            return None
    return key


def _coerce_promoted_arguments(
    raw_args: Any, tool_name: str, tool_schemas: Optional[dict]
) -> Optional[dict]:
    if isinstance(raw_args, Mapping):
        return dict(raw_args)
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        if tool_schemas is not None:
            key = _string_arg_key_from_schema(tool_schemas.get(tool_name))
            return {key: raw_args} if key else None
    coerced = coerce_tool_arguments(raw_args, heal = True, tool_name = tool_name)
    return coerced.arguments


def _promote(
    calls: list,
    allowed_tools: set,
    id_offset: int = 0,
    tool_schemas: Optional[dict] = None,
) -> list:
    """Filter parsed calls to declared tools and normalize their arguments.

    Bare string arguments on the client-tool passthrough use the declared
    schema's single required string property. If the schema is ambiguous, the
    call stays text instead of inventing a generic key.
    """
    promoted = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name not in allowed_tools:
            continue
        arguments = _coerce_promoted_arguments(function.get("arguments"), name, tool_schemas)
        if arguments is None:
            continue
        promoted.append(
            {
                "id": f"call_{id_offset + len(promoted)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii = False),
                },
            }
        )
    return promoted


def _remove_spans(text: str, spans: list) -> str:
    """Text with the given non-overlapping, sorted (start, end) ranges removed."""
    pieces = []
    pos = 0
    for start, end in spans:
        pieces.append(text[pos:start])
        pos = end
    pieces.append(text[pos:])
    return "".join(pieces)


def heal_openai_message_events(
    msg: dict,
    allowed_tools: set,
    tools: Optional[list] = None,
) -> Optional[list]:
    if not isinstance(msg, dict) or msg.get("tool_calls"):
        return None
    content = msg.get("content")
    if not isinstance(content, str) or not _has_heal_signal(content):
        return None
    parsed, spans = parse_tool_calls_from_text(content, allow_incomplete = True, with_spans = True)
    tool_schemas = _tool_schemas_by_name(tools) if tools is not None else None
    events: list = []
    pos = 0
    call_count = 0
    for call, (start, end) in zip(parsed, spans):
        promoted = _promote([call], allowed_tools, id_offset = call_count, tool_schemas = tool_schemas)
        if promoted:
            if content[pos:start]:
                events.append(("text", content[pos:start]))
            events.append(("tool_call", promoted[0]))
            call_count += 1
        else:
            events.append(("text", content[pos:end]))
        pos = end
    if not call_count:
        return None
    if content[pos:]:
        events.append(("text", content[pos:]))
    return events


def heal_openai_message(
    msg: dict,
    allowed_tools: set,
    tools: Optional[list] = None,
) -> bool:
    """Promote text-form tool calls in a non-streaming OpenAI message. In place.

    No-op (returns False) unless the message has NO structured ``tool_calls``
    (grammar mode already worked when it does) and its content carries a tool
    signal that parses into at least one declared call. Only the promoted
    calls' markup spans are removed from the content; undeclared calls and
    anything the parser did not consume stay in the text byte-intact.
    """
    events = heal_openai_message_events(msg, allowed_tools, tools)
    if not events:
        return False
    calls = [value for kind, value in events if kind == "tool_call"]
    content = "".join(value for kind, value in events if kind == "text").strip()
    msg["tool_calls"] = calls
    # OpenAI requires content = null on a pure tool-call turn.
    msg["content"] = content or None
    return True


def _earliest_signal(buffer: str) -> int:
    best = -1
    for signal in _HEAL_SIGNALS:
        index = buffer.find(signal)
        if index >= 0 and (best < 0 or index < best):
            best = index
    return best


def _closed_signal_span(buffer: str) -> Optional[tuple[int, int]]:
    spans = []
    for open_tag, close_tag in (
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call>", "<tool_call|>"),
        ("<function=", "</function>"),
    ):
        start = buffer.find(open_tag)
        if start < 0:
            continue
        end = buffer.find(close_tag, start)
        if end >= 0:
            spans.append((start, end + len(close_tag)))
    return min(spans, key = lambda span: span[0]) if spans else None


def _partial_signal_suffix(buffer: str) -> int:
    """Length of the longest buffer suffix that is a proper prefix of a signal."""
    for length in range(min(len(buffer), _MAX_SIGNAL_LEN - 1), 0, -1):
        tail = buffer[-length:]
        if any(signal.startswith(tail) for signal in _HEAL_SIGNALS):
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

    def __init__(
        self,
        allowed_tools: set,
        tools: Optional[list] = None,
    ) -> None:
        self._allowed = set(allowed_tools)

        self._tool_schemas = _tool_schemas_by_name(tools) if tools is not None else None
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
            # HOLD: drain the first contiguous run per pass so events keep document
            # order (a later declared call must not overtake an earlier undeclared one
            # flushing as text). A run is one markup call OR a whole Mistral [TOOL_CALLS]
            # array of contiguous spans, so later calls in it are not stranded as text.
            parsed, spans = parse_tool_calls_from_text(
                self._buffer,
                id_offset = self._id_offset,
                allow_incomplete = False,
                with_spans = True,
            )
            if not parsed:
                closed_span = _closed_signal_span(self._buffer)
                if closed_span:
                    _start, end = closed_span
                    events.append(("text", self._buffer[:end]))
                    self._buffer = self._buffer[end:]
                    self._holding = False
                    continue
                if len(self._buffer) > _MAX_HOLD_CHARS:
                    events.append(("text", self._buffer))
                    self._buffer = ""
                    self._holding = False
                    continue
                return events
            pos = 0
            run_end = spans[0][1]
            for order, (call, (start, end)) in enumerate(zip(parsed, spans)):
                # Stop at the first gap or incomplete trailing block: leave it for the
                # next pass to re-hold and stream incrementally, not flush as text early.
                if order and start != run_end:
                    break
                promoted = _promote(
                    [call],
                    self._allowed,
                    id_offset = self._id_offset,
                    tool_schemas = self._tool_schemas,
                )
                if promoted:
                    # Flush any leading text, then drop the promoted markup span.
                    if self._buffer[pos:start]:
                        events.append(("text", self._buffer[pos:start]))
                    events.append(("tool_call", promoted[0]))
                    self._id_offset += 1
                else:
                    # Undeclared/unusable name: markup is DATA, flush it (and prior text) verbatim.
                    events.append(("text", self._buffer[pos:end]))
                pos = end
                run_end = end
            # Everything past the drained run (later blocks) stays and is rescanned.
            self._buffer = self._buffer[run_end:]
            self._holding = False

    def finalize(self) -> list:
        """End of stream: last-chance heal of the residue, else flush it.

        Events keep document order; only the promoted calls' markup spans are
        dropped, every other residue byte flushes as text.
        """
        if not self._buffer:
            return []
        residue, self._buffer = self._buffer, ""
        holding, self._holding = self._holding, False
        if self.dormant or not holding:
            return [("text", residue)]
        parsed, spans = parse_tool_calls_from_text(
            residue,
            id_offset = self._id_offset,
            allow_incomplete = True,
            with_spans = True,
        )
        events: list = []
        pos = 0
        any_promoted = False
        for call, (start, end) in zip(parsed, spans):
            promoted = _promote(
                [call],
                self._allowed,
                id_offset = self._id_offset,
                tool_schemas = self._tool_schemas,
            )
            if promoted:
                if residue[pos:start]:
                    events.append(("text", residue[pos:start]))
                events.append(("tool_call", promoted[0]))
                self._id_offset += 1
                any_promoted = True
            else:
                events.append(("text", residue[pos:end]))
            pos = end
        if not any_promoted:
            return [("text", residue)]
        tail = residue[pos:].strip()
        if tail:
            events.append(("text", tail))
        return events


def _first_choice_message(data: Any) -> Optional[dict]:
    """First-choice message dict of a non-streaming chat response, else None.

    Upstream error bodies can carry ``"message": null`` (or no choices at all),
    so never assume the shape: a non-dict message means "nothing to heal".
    """
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    return message if isinstance(message, dict) else None


def _last_assistant_text(data: Any) -> str:
    """First-choice assistant content of a non-streaming chat response, or ''."""
    message = _first_choice_message(data)
    content = message.get("content") if message else None
    return content if isinstance(content, str) else ""


def _heal_would_promote(
    text: str,
    allowed_tools: set,
    tools: Optional[list] = None,
) -> bool:
    """Whether ``heal_openai_message`` would promote at least one call."""
    parsed = parse_tool_calls_from_text(text, allow_incomplete = True)
    tool_schemas = _tool_schemas_by_name(tools) if tools is not None else None
    return bool(_promote(parsed, allowed_tools, tool_schemas = tool_schemas))


def response_has_promotable_calls(
    data: Any,
    allowed_tools: set,
    tools: Optional[list] = None,
) -> bool:
    """True when a non-streaming chat response carries a usable tool call
    (structured naming a DECLARED tool, or text-form that healing would
    promote). Used to decide whether a nudge retry actually improved on the
    original response; a hallucinated undeclared call is not an improvement."""
    message = _first_choice_message(data)
    if not message:
        return False
    tool_calls = message.get("tool_calls")
    if tool_calls:
        # ALL structured calls must be declared: the caller forwards the whole
        # list (and a parallel cap could keep only the FIRST one), so a mixed
        # response with a single hallucinated name could still hand the client
        # an undeclared tool.
        return all(
            isinstance(tc, dict)
            and isinstance(tc.get("function"), dict)
            and tc["function"].get("name") in allowed_tools
            for tc in tool_calls
        )
    text = message.get("content")
    if not isinstance(text, str):
        return False
    return _heal_would_promote(text, allowed_tools, tools)


def nudge_should_retry(
    data: Any,
    allowed_tools: Optional[set],
    tools: Optional[list] = None,
) -> bool:
    """True when the first response tried to call a tool but nothing healed.

    Trigger only on: healing enabled (allowed_tools set), zero structured
    calls, a tool signal present in the text, and zero promotable calls -- the
    exact failure a single re-ask can fix. Clean prose never retries.
    """
    if not allowed_tools:
        return False
    message = _first_choice_message(data)
    if not message or message.get("tool_calls"):
        return False
    text = message.get("content")
    if not isinstance(text, str) or not _has_heal_signal(text):
        return False
    return not _heal_would_promote(text, allowed_tools, tools)


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
