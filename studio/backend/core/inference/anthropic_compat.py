# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Anthropic Messages API ↔ OpenAI format translation utilities.

Pure functions plus stateful stream emitters; no FastAPI, no I/O.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional, Union


def openai_finish_to_anthropic_stop(finish_reason, had_tool_calls = False) -> str:
    """Map an OpenAI finish_reason to an Anthropic stop_reason.
    'length' -> 'max_tokens' (truncation wins even mid tool call, so a cut-off
    tool call isn't mislabeled tool_use); tool_calls / had_tool_calls -> 'tool_use';
    'stop_sequence' -> 'stop_sequence'; 'stop'/None/unknown -> 'end_turn'."""
    # Truncation takes precedence: a tool call cut off at max_tokens has possibly incomplete arguments, so report
    # max_tokens rather than telling the client to run the tool.
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls" or had_tool_calls:
        return "tool_use"
    if finish_reason == "stop_sequence":
        return "stop_sequence"
    # "stop", None, and any unknown value collapse to end_turn.
    return "end_turn"


def anthropic_tool_use_id(upstream_id = None) -> str:
    """Return an Anthropic-style tool_use id (prefix 'toolu_'). Reuses an
    upstream id only if it already starts with 'toolu_'; otherwise mints a fresh
    'toolu_<24 hex>'."""
    if upstream_id and isinstance(upstream_id, str) and upstream_id.startswith("toolu_"):
        return upstream_id
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _anthropic_image_block_to_openai_part(block: dict) -> Optional[dict]:
    """Translate one Anthropic ``image`` block to an OpenAI ``image_url`` part.

    Accepts both source shapes:
      - ``{"type": "base64", "media_type": "image/jpeg", "data": "..."}``
      - ``{"type": "url", "url": "https://..."}``

    Returns ``None`` when the source is malformed so the caller can skip it.
    """
    source = block.get("source") or {}
    stype = source.get("type")
    if stype == "base64":
        data = source.get("data")
        if not data:
            return None
        media_type = source.get("media_type") or "image/jpeg"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    if stype == "url":
        url = source.get("url")
        if not url:
            return None
        return {"type": "image_url", "image_url": {"url": url}}
    return None


def anthropic_messages_to_openai(
    messages: list[dict],
    system: Optional[Union[str, list]] = None,
    preserve_thinking: bool = False,
) -> list[dict]:
    """Convert Anthropic messages + system to OpenAI-format message dicts.

    User messages with ``image`` blocks are emitted as OpenAI multimodal
    content arrays (``[{type: "text", ...}, {type: "image_url", ...}]``) so
    they flow through llama-server's native vision pathway.

    ``preserve_thinking`` keeps replayed assistant ``thinking`` blocks as
    ``reasoning_content`` on the converted message, so templates that render
    historical reasoning (Qwen3.6-style ``preserve_thinking``) actually receive
    it; otherwise thinking is dropped from the prompt. ``redacted_thinking``
    carries only ciphertext and is always dropped.
    """
    result: list[dict] = []

    if system:
        if isinstance(system, str):
            result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            if parts:
                result.append({"role": "system", "content": "\n".join(parts)})

    for msg in messages:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if role == "assistant":
            # text + tool_use (no images in Anthropic's model), plus replayed thinking when preservation is requested
            # Assistant content: text + tool_use (no images in Anthropic's model), plus replayed thinking when
            # preservation is requested.
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            thinking_parts: list[str] = []
            for block in content:
                b = block if isinstance(block, dict) else block.model_dump()
                btype = b.get("type", "")
                if btype == "text":
                    text_parts.append(b["text"])
                elif btype == "thinking" and preserve_thinking:
                    t = b.get("thinking") or ""
                    if t:
                        thinking_parts.append(t)
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {
                                "name": b["name"],
                                "arguments": json.dumps(b["input"]),
                            },
                        }
                    )
            msg_dict: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg_dict["content"] = "\n".join(text_parts)
            if thinking_parts:
                msg_dict["reasoning_content"] = "\n\n".join(thinking_parts)
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            result.append(msg_dict)
            continue

        if role == "user":
            # Ordered parts preserve text/image interleaving; tool_result -> own "tool" messages.
            user_parts: list[dict] = []
            has_image = False
            tool_results: list[dict] = []
            for block in content:
                b = block if isinstance(block, dict) else block.model_dump()
                btype = b.get("type", "")
                if btype == "text":
                    user_parts.append({"type": "text", "text": b["text"]})
                elif btype == "image":
                    part = _anthropic_image_block_to_openai_part(b)
                    if part is not None:
                        user_parts.append(part)
                        has_image = True
                elif btype == "tool_result":
                    tc = b.get("content", "")
                    if isinstance(tc, list):
                        tc = " ".join(
                            p["text"] for p in tc if isinstance(p, dict) and p.get("type") == "text"
                        )
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": str(tc),
                        }
                    )

            for tr in tool_results:
                result.append(tr)

            if has_image:
                result.append({"role": "user", "content": user_parts})
            else:
                # No images: collapse text parts to a plain string.
                text = "\n".join(p["text"] for p in user_parts)
                if text:
                    result.append({"role": "user", "content": text})

    return result


def fold_tool_results_into_user(messages: list[dict]) -> list[dict]:
    """Rewrite ``role="tool"`` as user turns: Gemma 2 / 3 have no tool role and
    check alternation by index parity, so one makes llama-server 400 the whole
    request. Shape mirrors minja's ``polyfill_tool_responses``.
    """
    out: list[dict] = []
    call_names: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                fn = tc.get("function")
                if isinstance(tc_id, str) and tc_id and isinstance(fn, dict):
                    name = fn.get("name")
                    if isinstance(name, str) and name:
                        call_names[tc_id] = name

        if msg.get("role") != "tool":
            out.append(msg)
            continue

        response: dict[str, Any] = {}
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name") or call_names.get(tool_call_id or "")
        if name:
            response["tool"] = name
        response["content"] = msg.get("content", "")
        if tool_call_id:
            response["tool_call_id"] = tool_call_id
        out.append(
            {
                "role": "user",
                "content": json.dumps({"tool_response": response}, indent = 2),
            }
        )
    return out


_ANTHROPIC_SCHEMA_CLIENT_TOOL_PARAMETERS = {
    "bash": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "restart": {"type": "boolean"},
        },
        "anyOf": [
            {"required": ["command"]},
            {"properties": {"restart": {"const": True}}, "required": ["restart"]},
        ],
    },
    "text_editor": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "str_replace", "create", "insert"],
            },
            "path": {"type": "string"},
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "file_text": {"type": "string"},
            "insert_line": {"type": "integer"},
            "insert_text": {"type": "string"},
        },
        "required": ["command", "path"],
    },
    "computer": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "text": {"type": "string"},
            "duration": {"type": "number"},
            "scroll_direction": {"type": "string"},
            "scroll_amount": {"type": "integer"},
            "start_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "key": {"type": "string"},
        },
        "required": ["action"],
        "additionalProperties": True,
    },
    "memory": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
            },
            "path": {"type": "string"},
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"},
            "insert_text": {"type": "string"},
            "old_path": {"type": "string"},
            "new_path": {"type": "string"},
        },
        "required": ["command"],
    },
}

_ANTHROPIC_SCHEMA_CLIENT_TOOL_DESCRIPTIONS = {
    "bash": "Run a command in the caller-owned persistent bash session, or restart it.",
    "text_editor": "View, create, or edit files in the caller-owned filesystem.",
    "computer": "Interact with the caller-owned computer using an action and its parameters.",
    "memory": "Store and retrieve files in the caller-owned persistent memory directory.",
}


def anthropic_schema_client_tool_kind(tool) -> Optional[str]:
    """Return the kind of a schema-less Anthropic client tool, if recognized."""
    td = tool if isinstance(tool, dict) else tool.model_dump()
    if td.get("input_schema") is not None:
        return None
    type_ = td.get("type")
    if not isinstance(type_, str):
        return None
    kind, separator, version = type_.rpartition("_")
    if (
        separator
        and kind in _ANTHROPIC_SCHEMA_CLIENT_TOOL_PARAMETERS
        and len(version) == 8
        and version.isdigit()
    ):
        return kind
    return None


def _anthropic_schema_client_tool_parameters(td: dict, kind: str) -> dict:
    parameters = _ANTHROPIC_SCHEMA_CLIENT_TOOL_PARAMETERS[kind]
    if kind != "text_editor":
        return parameters

    version = td["type"].rpartition("_")[2]
    commands = list(parameters["properties"]["command"]["enum"])
    if version < "20250429":
        commands.append("undo_edit")
    return {
        **parameters,
        "properties": {
            **parameters["properties"],
            "command": {**parameters["properties"]["command"], "enum": commands},
        },
    }


def anthropic_tools_to_openai(tools: list) -> list[dict]:
    """Convert Anthropic client tools to OpenAI function-tool format."""
    result = []
    for t in tools:
        td = t if isinstance(t, dict) else t.model_dump()
        name = td.get("name")
        input_schema = td.get("input_schema")
        schema_client_kind = anthropic_schema_client_tool_kind(td)
        if schema_client_kind is not None:
            input_schema = _anthropic_schema_client_tool_parameters(td, schema_client_kind)
        if not name or input_schema is None:
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": td.get("description")
                    or _ANTHROPIC_SCHEMA_CLIENT_TOOL_DESCRIPTIONS.get(schema_client_kind, ""),
                    "parameters": input_schema,
                },
            }
        )
    return result


def anthropic_tool_choice_to_openai(tc: Any) -> Any:
    """Translate Anthropic `tool_choice` into OpenAI `tool_choice`.

    Anthropic formats (all dict shapes with a ``type`` discriminator):

    - ``{"type": "auto"}``                       → ``"auto"``
    - ``{"type": "any"}``                        → ``"required"``
    - ``{"type": "none"}``                       → ``"none"``
    - ``{"type": "tool", "name": "get_weather"}``
          → ``{"type": "function", "function": {"name": "get_weather"}}``

    Returns ``None`` for ``None`` or any unrecognized shape (caller falls
    back to its own default, typically ``"auto"``).
    """
    if tc is None:
        return None
    if not isinstance(tc, dict):
        return None
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        name = tc.get("name")
        if not name:
            return None
        return {"type": "function", "function": {"name": name}}
    return None


def build_anthropic_sse_event(event_type: str, data: dict) -> str:
    """Format a single Anthropic SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _message_delta_usage(usage: Optional[dict]) -> dict:
    """Usage block for a message_delta event (cumulative token counts). Cache
    fields are always 0 — no prompt caching backend. ``usage`` may be None when a
    metadata event carried usage=None (e.g. only finish_reason set)."""
    usage = usage or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": usage.get("completion_tokens", 0),
    }


def _partial_tag_suffix_len(text: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that ends ``text``.

    A streamed delta can cut a ``<think>`` tag anywhere; the caller holds that
    suffix back until the next delta settles whether it was markup or prose.
    """
    for k in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class AnthropicStreamEmitter:
    """Converts generate_chat_completion_with_tools() events into Anthropic
    Messages SSE strings."""

    def __init__(
        self,
        parse_think: bool = True,
        think_provenance: Optional[dict] = None,
    ) -> None:
        # Off when the route knows reasoning markup cannot be genuine (thinking disabled or a non-reasoning model):
        # literal <think> in prose then streams as ordinary text instead of being consumed as a trace.
        self._parse_think = parse_think
        self.block_index: int = 0
        self._block_index_used: bool = False
        self._text_block_open: bool = False
        self._thinking_block_open: bool = False
        self._open_tool_call_id: Optional[str] = None
        # The mapped Anthropic ``toolu_*`` id published in content_block_start, reused for the paired tool_result so
        # consumers can correlate them.
        self._open_tool_use_id: Optional[str] = None
        self._open_tool_args_sent: bool = False
        self._prev_text: str = ""
        # the generator folds reasoning_content into the cumulative text as <think>...</think> markup
        # <think> routing: the generator folds reasoning_content into the cumulative text as <think>...</think> markup
        # (the UI chat parses it), but Anthropic clients expect typed thinking blocks. Split the markup back out: text
        # inside the tags streams as thinking_delta in a "thinking" content block, everything else as ordinary text.
        # _tag_buf holds back a trailing partial tag until the next delta decides it.
        self._route_mode: str = "text"
        self._tag_buf: str = ""
        # Leading whitespace of a thinking span, held until real reasoning arrives so a whitespace-only trace never
        # opens a block. See _emit_thinking_delta.
        self._thinking_ws_hold: str = ""
        # Genuine reasoning only ever arrives as a single LEADING <think> block per synthesis turn (the generator folds
        # reasoning_content in as a prefix). Once that block closed, or once real answer text has streamed, any later
        # <think> is the model quoting the tag and must stay literal.
        self._think_consumed: bool = False
        self._turn_has_text: bool = False
        # "wrapped" counts the leading <think> tags the generator opened from reasoning_content
        # Live provenance from the generator: "wrapped" counts the leading <think> tags IT opened from
        # reasoning_content. When provided, a leading tag is only parsed as reasoning if a generator wrap is available
        # -- a model answering with literal <think> markup (and no genuine trace) keeps it as text. None falls back to
        # the leading-tag heuristic (test doubles / callers without provenance).
        self._think_provenance = think_provenance
        self._wraps_consumed: int = 0
        # the block spans exactly the wrap's N reasoning chars
        # Active wrap entry ({"len": N} from the generator) while a provenance -backed thinking block streams: the block
        # spans exactly N reasoning chars, so a literal "</think>" INSIDE the trace never ends it early.
        self._active_wrap: Optional[dict] = None
        self._wrap_chars: int = 0
        self._close_skip: int = 0
        self._usage: dict = {}

    def start(
        self,
        message_id: str,
        model: str,
        input_tokens: int = 0,
    ) -> list[str]:
        """Emit message_start; content blocks open lazily on first output."""
        events = []
        events.append(
            build_anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                    },
                },
            )
        )
        return events

    def feed(self, event: dict) -> list[str]:
        """Process one generator event, return SSE strings."""
        etype = event.get("type", "")
        if etype == "content":
            return self._handle_content(event)
        elif etype == "tool_start":
            return self._handle_tool_start(event)
        elif etype == "tool_end":
            return self._handle_tool_end(event)
        elif etype == "metadata":
            self._usage = event.get("usage", {})
            return []
        return []

    def finish(
        self,
        stop_reason: str = "end_turn",
        stop_sequence = None,
    ) -> list[str]:
        """Close any open block and emit message_delta + message_stop."""
        events = []
        if self._tag_buf:
            held, self._tag_buf = self._tag_buf, ""
            if self._route_mode == "thinking":
                events.extend(self._emit_thinking_delta(held))
            else:
                events.extend(self._emit_text_delta(held))
        if (
            self._text_block_open
            or self._thinking_block_open
            or self._open_tool_call_id is not None
        ):
            events.append(self._close_block())
            self._open_tool_call_id = None
            self._open_tool_use_id = None
            self._open_tool_args_sent = False
        events.append(
            build_anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": stop_sequence,
                    },
                    "usage": _message_delta_usage(self._usage),
                },
            )
        )
        events.append(
            build_anthropic_sse_event(
                "message_stop",
                {
                    "type": "message_stop",
                },
            )
        )
        return events

    def _handle_content(self, event: dict) -> list[str]:
        cumulative = event.get("text", "")
        new_text = cumulative[len(self._prev_text) :]
        self._prev_text = cumulative
        if not new_text:
            return []
        return self._route_text(new_text)

    def _route_text(self, new_text: str) -> list[str]:
        """Split ``<think>`` markup out of the delta into typed blocks."""
        if not self._parse_think:
            return self._emit_text_delta(new_text)
        events: list[str] = []
        data = self._tag_buf + new_text
        self._tag_buf = ""
        while data:
            if self._route_mode == "text":
                if self._think_consumed or self._turn_has_text:
                    events.extend(self._emit_text_delta(data))
                    break
                open_tag = "<think>"
                i = data.find(open_tag)
                if i == -1:
                    keep = _partial_tag_suffix_len(data, open_tag)
                    emit = data[: len(data) - keep]
                    self._tag_buf = data[len(data) - keep :] if keep else ""
                    if emit:
                        events.extend(self._emit_text_delta(emit))
                    break
                if i:
                    events.extend(self._emit_text_delta(data[:i]))
                    # consumed: the run before the tag has already been delivered
                    # Consumed: whatever happens to the tag below, the run before it has already been delivered.
                    # Re-including it in the literal-text branch below sent it to the client twice.
                    data = data[i:]
                    i = 0
                    if self._turn_has_text:
                        continue
                if (
                    self._think_provenance is not None
                    and self._think_provenance.get("wrapped", 0) <= self._wraps_consumed
                ):
                    events.extend(self._emit_text_delta(data))
                    break
                wraps = (
                    self._think_provenance.get("wraps")
                    if self._think_provenance is not None
                    else None
                )
                self._active_wrap = (
                    wraps[self._wraps_consumed]
                    if wraps and self._wraps_consumed < len(wraps)
                    else None
                )
                self._wrap_chars = 0
                self._wraps_consumed += 1
                data = data[i + len(open_tag) :]
                self._route_mode = "thinking"
            else:
                if self._close_skip:
                    skip = min(self._close_skip, len(data))
                    self._close_skip -= skip
                    data = data[skip:]
                    if self._close_skip == 0:
                        self._route_mode = "text"
                        self._think_consumed = True
                        self._active_wrap = None
                    continue
                if self._active_wrap is not None:
                    # Provenance-backed span: consume exactly the generator's reasoning length, then skip its closing
                    # tag. A literal "</think>" inside the trace stays part of the thinking.
                    remaining = int(self._active_wrap.get("len", 0)) - self._wrap_chars
                    if remaining > 0:
                        take = min(remaining, len(data))
                        events.extend(self._emit_thinking_delta(data[:take]))
                        self._wrap_chars += take
                        data = data[take:]
                        continue
                    self._close_skip = len("</think>")
                    continue
                close_tag = "</think>"
                i = data.find(close_tag)
                if i == -1:
                    keep = _partial_tag_suffix_len(data, close_tag)
                    emit = data[: len(data) - keep]
                    self._tag_buf = data[len(data) - keep :] if keep else ""
                    if emit:
                        events.extend(self._emit_thinking_delta(emit))
                    break
                if i:
                    events.extend(self._emit_thinking_delta(data[:i]))
                data = data[i + len(close_tag) :]
                self._route_mode = "text"
                self._think_consumed = True
        return events

    def _emit_text_delta(self, text: str) -> list[str]:
        if text.strip():
            self._turn_has_text = True
        events: list[str] = []
        if self._thinking_block_open:
            events.append(self._close_block())
        if not self._text_block_open:
            events.extend(self._open_text_block())
        events.append(
            build_anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        )
        return events

    def _emit_thinking_delta(self, text: str) -> list[str]:
        if not self._thinking_block_open:
            # a whitespace-only trace is not a thought
            # A trace that is only whitespace is not a thought: Qwen3-style templates render "<think>\n\n</think>" on
            # every reply when thinking is off, and llama-server parses that into reasoning_content, so an empty
            # thinking block would be attached to ordinary answers. The non-streaming reducer already drops those, so
            # hold the leading whitespace run and only open the block once real reasoning arrives; the held run is then
            # emitted with it so the trace stays verbatim.
            held = self._thinking_ws_hold + text
            if not held.strip():
                self._thinking_ws_hold = held
                return []
            self._thinking_ws_hold = ""
            text = held
        events: list[str] = []
        if self._text_block_open:
            events.append(self._close_block())
        if not self._thinking_block_open:
            events.extend(self._open_thinking_block())
        events.append(
            build_anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "thinking_delta", "thinking": text},
                },
            )
        )
        return events

    def _handle_tool_start(self, event: dict) -> list[str]:
        tool_call_id = event.get("tool_call_id", "")
        args = event.get("arguments", {})
        if tool_call_id and self._open_tool_call_id == tool_call_id:
            return self._tool_arguments_delta(args)

        events = []
        if self._tag_buf:
            held, self._tag_buf = self._tag_buf, ""
            if self._route_mode == "thinking":
                events.extend(self._emit_thinking_delta(held))
            else:
                events.extend(self._emit_text_delta(held))
        if self._text_block_open or self._thinking_block_open:
            events.append(self._close_block())
        # Defensive: close a stale open tool_use block before starting another.
        elif self._open_tool_call_id is not None:
            events.append(self._close_block())
            self._open_tool_call_id = None
            self._open_tool_use_id = None
            self._open_tool_args_sent = False

        self._alloc_block_index()
        self._open_tool_call_id = tool_call_id
        self._open_tool_use_id = anthropic_tool_use_id(tool_call_id)
        self._open_tool_args_sent = False
        events.append(
            build_anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": self._open_tool_use_id,
                        "name": event.get("tool_name", ""),
                        "input": {},
                    },
                },
            )
        )
        events.extend(self._tool_arguments_delta(args))
        return events

    def _tool_arguments_delta(self, args: dict) -> list[str]:
        if not args:
            return []
        if self._open_tool_args_sent:
            return []
        self._open_tool_args_sent = True
        return [
            build_anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(args),
                    },
                },
            )
        ]

    def _handle_tool_end(self, event: dict) -> list[str]:
        events = []
        if self._open_tool_call_id is not None or self._text_block_open:
            events.append(self._close_block())
        # Reuse the id published in content_block_start; fall back to mapping the raw id only if no tool_start preceded
        # this end.
        tool_use_id = self._open_tool_use_id or anthropic_tool_use_id(event.get("tool_call_id", ""))
        self._open_tool_call_id = None
        self._open_tool_use_id = None
        self._open_tool_args_sent = False
        events.append(
            build_anthropic_sse_event(
                "tool_result",
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": event.get("result", ""),
                },
            )
        )
        # the next content delta opens a fresh block lazily
        # Reset text tracking for the next synthesis turn; the next content delta opens a fresh text (or thinking) block
        # lazily, and the new turn may legitimately open with its own leading <think> block.
        self._prev_text = ""
        self._tag_buf = ""
        self._thinking_ws_hold = ""
        self._route_mode = "text"
        self._think_consumed = False
        self._turn_has_text = False
        self._active_wrap = None
        self._wrap_chars = 0
        self._close_skip = 0
        return events

    def _alloc_block_index(self) -> None:
        if self._block_index_used:
            self.block_index += 1
        self._block_index_used = True

    def _open_text_block(self) -> list[str]:
        self._alloc_block_index()
        self._text_block_open = True
        return [
            build_anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        ]

    def _open_thinking_block(self) -> list[str]:
        self._alloc_block_index()
        self._thinking_block_open = True
        return [
            build_anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    # strict stream decoders reject a thinking block without a signature
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            )
        ]

    def _close_block(self) -> str:
        self._text_block_open = False
        self._thinking_block_open = False
        return build_anthropic_sse_event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": self.block_index,
            },
        )


class AnthropicPassthroughEmitter:
    """Converts llama-server's OpenAI-format streaming chunks into Anthropic SSE.

    Used for the client-side tool-use pass-through path: the client (e.g.
    Claude Code) sends its own tool definitions in ``tools`` and executes
    them itself. We forward them to llama-server and translate the streaming
    response back to Anthropic format without executing anything.
    """

    def __init__(self, reasoning_as_thinking: bool = True) -> None:
        # When thinking is effectively off, llama-server's format parser can still shunt a literal <think> example the
        # model was asked to produce into reasoning_content; reconstruct it as visible text instead of a typed thinking
        # block.
        self._reasoning_as_thinking = reasoning_as_thinking
        self._reasoning_text_open = False
        self.block_index: int = -1
        self._current_block_type: Optional[str] = None
        self._tool_call_states: dict = {}  # delta index -> {block_index, id, name}
        self._usage: dict = {}
        self._stop_reason: str = "end_turn"
        self._stop_sequence: Optional[str] = None
        # Optional text-form tool-call healing (client-tool passthrough only).
        self._healer = None
        self._healed_tool_use = False
        self._healed_call_count = 0
        self._heal_disable_parallel = False

    def enable_healing(
        self,
        allowed_tools: set,
        tools: Optional[list] = None,
        *,
        disable_parallel_tool_use: bool = False,
    ) -> None:
        """Promote text-form tool calls in streamed content to tool_use blocks.

        Only calls naming a tool in ``allowed_tools`` (the client's declared
        tools) are promoted; everything else streams as text exactly as before.
        Never enabled for Unsloth's own tool loop.
        """
        from core.inference.passthrough_healing import StreamToolCallHealer

        self._healer = StreamToolCallHealer(allowed_tools, tools)
        self._heal_disable_parallel = disable_parallel_tool_use

    def start(
        self,
        message_id: str,
        model: str,
        input_tokens: int = 0,
    ) -> list[str]:
        return [
            build_anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                    },
                },
            )
        ]

    def feed_chunk(self, chunk: dict) -> list[str]:
        """Process one OpenAI streaming chat.completion.chunk."""
        events: list[str] = []

        # usage-only chunks carry token totals
        usage = chunk.get("usage")
        if usage:
            self._usage = usage

        choices = chunk.get("choices") or []
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")

        # llama-server splits <think> into reasoning_content whenever it can parse the model's reasoning format (it does
        # so for tool-calling turns, i.e.
        # ── Reasoning ── llama-server splits <think> into reasoning_content whenever it can parse the model's reasoning
        # format (it does so for tool-calling turns, which is every Claude Code turn). Reading only `content` drops the
        # entire thinking trace, so the model appears not to think at all.
        # ── Reasoning ──
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not self._reasoning_as_thinking:
                prefix = "" if self._reasoning_text_open else "<think>"
                self._reasoning_text_open = True
                events.extend(self._emit_text_delta(prefix + reasoning))
            else:
                if self._current_block_type != "thinking":
                    if self._current_block_type is not None:
                        events.append(self._close_current_block())
                    events.extend(self._open_thinking_block())
                events.append(
                    build_anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self.block_index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning},
                        },
                    )
                )
        # checked unconditionally, not elif: one chunk can carry the final reasoning fragment AND same-chunk content
        # Reconstructed literal block ends where the answer resumes -- checked unconditionally (not elif): one chunk can
        # carry the final reasoning fragment AND same-chunk content/tool output, and the closing tag must land between
        # them.
        if self._reasoning_text_open and (
            delta.get("content") or delta.get("tool_calls") or finish_reason
        ):
            self._reasoning_text_open = False
            events.extend(self._emit_text_delta("</think>"))

        # grammar mode worked: flush anything the healer held (it preceded the call in the model's output) and relay
        # verbatim from here
        # ── Structured tool calls take precedence over healing ── Grammar mode worked: flush anything the healer held
        # (it preceded the call in the model's output) and relay verbatim from here on.
        # ── Structured tool calls take precedence over healing ──
        if delta.get("tool_calls") and self._healer is not None and not self._healer.dormant:
            for kind, value in self._healer.structured_tool_call_seen():
                if kind == "text" and value:
                    events.extend(self._emit_text_delta(value))

        # ── Text content ──
        content = delta.get("content")
        if content and self._healer is not None and not self._healer.dormant:
            # Route text through the healer: held/promoted portions become synthetic tool_use blocks, the rest streams
            # as text unchanged.
            for kind, value in self._healer.feed(content):
                if kind == "text":
                    events.extend(self._emit_text_delta(value))
                else:
                    events.extend(self._emit_healed_tool_use(value))
        elif content:
            events.extend(self._emit_text_delta(content))

        # ── Tool calls (streaming deltas) ──
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            tc_idx = tc.get("index", 0)
            fn = tc.get("function") or {}
            if (
                self._heal_disable_parallel
                and tc_idx not in self._tool_call_states
                and (self._healed_call_count + len(self._tool_call_states)) >= 1
            ):
                # disable_parallel_tool_use: a healed call already consumed the single allowed slot. The caller's
                # chunk-level cap only sees native indexes, so drop this native call (and its later argument deltas,
                # which never allocate a state either).
                continue
            if tc_idx not in self._tool_call_states:
                if self._current_block_type is not None:
                    events.append(self._close_current_block())
                tc_id = anthropic_tool_use_id(tc.get("id", ""))
                tc_name = fn.get("name", "")
                self.block_index += 1
                self._current_block_type = "tool_use"
                self._tool_call_states[tc_idx] = {
                    "block_index": self.block_index,
                    "id": tc_id,
                    "name": tc_name,
                }
                events.append(
                    build_anthropic_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": self.block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input": {},
                            },
                        },
                    )
                )

            args_delta = fn.get("arguments", "")
            if args_delta:
                events.append(
                    build_anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._tool_call_states[tc_idx]["block_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_delta,
                            },
                        },
                    )
                )

        # ── Finish reason ──
        if finish_reason:
            self._stop_reason = openai_finish_to_anthropic_stop(finish_reason)

        return events

    def finish(self) -> list[str]:
        events: list[str] = []
        if self._reasoning_text_open:
            self._reasoning_text_open = False
            events.extend(self._emit_text_delta("</think>"))
        if self._healer is not None:
            # Last-chance heal of any held residue (e.g. an unclosed tool block).
            for kind, value in self._healer.finalize():
                if kind == "text" and value:
                    events.extend(self._emit_text_delta(value))
                elif kind == "tool_call":
                    events.extend(self._emit_healed_tool_use(value))
        if self._healed_tool_use and self._stop_reason != "max_tokens":
            # A promoted call must stop for tool use; a truncation still wins (its arguments may be incomplete).
            self._stop_reason = "tool_use"
        if self._current_block_type is not None:
            events.append(self._close_current_block())
        events.append(
            build_anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self._stop_reason,
                        "stop_sequence": self._stop_sequence,
                    },
                    "usage": _message_delta_usage(self._usage),
                },
            )
        )
        events.append(
            build_anthropic_sse_event(
                "message_stop",
                {"type": "message_stop"},
            )
        )
        return events

    def _emit_text_delta(self, content: str) -> list[str]:
        events: list[str] = []
        if self._current_block_type != "text":
            if self._current_block_type is not None:
                events.append(self._close_current_block())
            events.extend(self._open_text_block())
        events.append(
            build_anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": content},
                },
            )
        )
        return events

    def _emit_healed_tool_use(self, call: dict) -> list[str]:
        # A healed call arrives complete, so its tool_use block opens, carries one input_json_delta, and closes
        # immediately; an open text block is closed first (only the safe prefix ever streamed into it).
        if (
            self._heal_disable_parallel
            and (self._healed_call_count + len(self._tool_call_states)) >= 1
        ):
            return []
        events: list[str] = []
        if self._current_block_type is not None:
            events.append(self._close_current_block())
        function = call.get("function") or {}
        tool_id = anthropic_tool_use_id("")
        self.block_index += 1
        self._current_block_type = "tool_use"
        events.append(
            build_anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": function.get("name", ""),
                        "input": {},
                    },
                },
            )
        )
        arguments = function.get("arguments") or ""
        if arguments:
            events.append(
                build_anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        },
                    },
                )
            )
        events.append(self._close_current_block())
        self._healed_tool_use = True
        self._healed_call_count += 1
        return events

    def _open_text_block(self) -> list[str]:
        self.block_index += 1
        self._current_block_type = "text"
        return [
            build_anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        ]

    def _open_thinking_block(self) -> list[str]:
        self.block_index += 1
        self._current_block_type = "thinking"
        return [
            build_anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            )
        ]

    def _close_current_block(self) -> str:
        idx = self.block_index
        self._current_block_type = None
        return build_anthropic_sse_event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": idx,
            },
        )
