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
    # Truncation takes precedence: a tool call cut off at max_tokens has possibly
    # incomplete arguments, so report max_tokens rather than telling the client to
    # run the tool.
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
    messages: list[dict], system: Optional[Union[str, list]] = None
) -> list[dict]:
    """Convert Anthropic messages + system to OpenAI-format message dicts.

    User messages with ``image`` blocks are emitted as OpenAI multimodal
    content arrays (``[{type: "text", ...}, {type: "image_url", ...}]``) so
    they flow through llama-server's native vision pathway.
    """
    result: list[dict] = []

    # System prompt
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
            # Assistant content: text + tool_use only (no images in Anthropic's model).
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                b = block if isinstance(block, dict) else block.model_dump()
                btype = b.get("type", "")
                if btype == "text":
                    text_parts.append(b["text"])
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

            if has_image:
                result.append({"role": "user", "content": user_parts})
            else:
                # No images: collapse text parts to a plain string.
                text = "\n".join(p["text"] for p in user_parts)
                if text:
                    result.append({"role": "user", "content": text})
            for tr in tool_results:
                result.append(tr)

    return result


def anthropic_tools_to_openai(tools: list) -> list[dict]:
    """Convert Anthropic client tools to OpenAI function-tool format."""
    result = []
    for t in tools:
        td = t if isinstance(t, dict) else t.model_dump()
        name = td.get("name")
        input_schema = td.get("input_schema")
        if not name or input_schema is None:
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": td.get("description", ""),
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


class AnthropicStreamEmitter:
    """Converts generate_chat_completion_with_tools() events into Anthropic
    Messages SSE strings."""

    def __init__(self) -> None:
        self.block_index: int = 0
        self._text_block_open: bool = False
        self._open_tool_call_id: Optional[str] = None
        # The mapped Anthropic ``toolu_*`` id published in content_block_start,
        # reused for the paired tool_result so consumers can correlate them.
        self._open_tool_use_id: Optional[str] = None
        self._open_tool_args_sent: bool = False
        self._prev_text: str = ""
        # Net <think> minus </think> in the text emitted to the client. Tracked
        # from emitted deltas (not _prev_text, which a final bare shrink clobbers)
        # so an unclosed reasoning-only block can be balanced before close.
        self._open_think_tags: int = 0
        self._usage: dict = {}

    def start(
        self,
        message_id: str,
        model: str,
        input_tokens: int = 0,
    ) -> list[str]:
        """Emit message_start and open the first text content block."""
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
        events.extend(self._open_text_block())
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
        # status events — no Anthropic equivalent
        return []

    def finish(
        self,
        stop_reason: str = "end_turn",
        stop_sequence = None,
    ) -> list[str]:
        """Close any open block and emit message_delta + message_stop."""
        events = []
        if self._text_block_open or self._open_tool_call_id is not None:
            events.extend(self._close_open_think())
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

    def _close_open_think(self) -> list[str]:
        """Emit a ``</think>`` delta when the streamed text left a ``<think>``
        open. This emitter diffs cumulative snapshots and drops the generator's
        final bare shrink, so a reasoning-only reply would otherwise end on an
        unclosed tag. Mirrors the chat route's reasoning extractor, which closes
        the block on finish; balances the block before it is closed."""
        if not self._text_block_open or self._open_think_tags <= 0:
            return []
        self._open_think_tags = 0
        return [
            build_anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": "</think>"},
                },
            )
        ]

    def _handle_content(self, event: dict) -> list[str]:
        cumulative = event.get("text", "")
        new_text = cumulative[len(self._prev_text) :]
        self._prev_text = cumulative
        if not new_text:
            return []
        self._open_think_tags += new_text.count("<think>") - new_text.count("</think>")
        if not self._text_block_open:
            events = self._open_text_block()
        else:
            events = []
        events.append(
            build_anthropic_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": new_text},
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
        if self._text_block_open:
            events.extend(self._close_open_think())
            events.append(self._close_block())
        # Defensive: close a stale open tool_use block before starting another.
        elif self._open_tool_call_id is not None:
            events.append(self._close_block())
            self._open_tool_call_id = None
            self._open_tool_use_id = None
            self._open_tool_args_sent = False

        # Open a tool_use block.
        self.block_index += 1
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
        # Close the tool_use block.
        if self._open_tool_call_id is not None or self._text_block_open:
            events.append(self._close_block())
        # Reuse the id published in content_block_start; fall back to mapping
        # the raw id only if no tool_start preceded this end.
        tool_use_id = self._open_tool_use_id or anthropic_tool_use_id(event.get("tool_call_id", ""))
        self._open_tool_call_id = None
        self._open_tool_use_id = None
        self._open_tool_args_sent = False
        # Emit custom tool_result event (non-standard, ignored by SDKs)
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
        # Open a new text block for the model's next response
        self.block_index += 1
        events.extend(self._open_text_block())
        # Reset text tracking for the next synthesis turn
        self._prev_text = ""
        self._open_think_tags = 0
        return events

    def _open_text_block(self) -> list[str]:
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

    def _close_block(self) -> str:
        self._text_block_open = False
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

    def __init__(self) -> None:
        self.block_index: int = -1
        self._current_block_type: Optional[str] = None  # "text" | "tool_use" | None
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

        # ── Structured tool calls take precedence over healing ──
        # Grammar mode worked: flush anything the healer held (it preceded the
        # call in the model's output) and relay verbatim from here on.
        if delta.get("tool_calls") and self._healer is not None and not self._healer.dormant:
            for kind, value in self._healer.structured_tool_call_seen():
                if kind == "text" and value:
                    events.extend(self._emit_text_delta(value))

        # ── Text content ──
        content = delta.get("content")
        if content and self._healer is not None and not self._healer.dormant:
            # Route text through the healer: held/promoted portions become
            # synthetic tool_use blocks, the rest streams as text unchanged.
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
                # disable_parallel_tool_use: a healed call already consumed the
                # single allowed slot. The caller's chunk-level cap only sees
                # native indexes, so drop this native call (and its later
                # argument deltas, which never allocate a state either).
                continue
            if tc_idx not in self._tool_call_states:
                # New tool call — close prior block, open tool_use block
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
        if self._healer is not None:
            # Last-chance heal of any held residue (e.g. an unclosed tool block).
            for kind, value in self._healer.finalize():
                if kind == "text" and value:
                    events.extend(self._emit_text_delta(value))
                elif kind == "tool_call":
                    events.extend(self._emit_healed_tool_use(value))
        if self._healed_tool_use and self._stop_reason != "max_tokens":
            # A promoted call must stop for tool use; a truncation still wins
            # (its arguments may be incomplete).
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
        # A healed call arrives complete, so its tool_use block opens, carries
        # one input_json_delta, and closes immediately; an open text block is
        # closed first (only the safe prefix ever streamed into it).
        if (
            self._heal_disable_parallel
            and (self._healed_call_count + len(self._tool_call_states)) >= 1
        ):
            # Healed and native calls share the single allowed slot.
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
