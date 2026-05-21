# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Anthropic Messages API ↔ OpenAI format translation utilities.

Pure functions and a stateful stream emitter — no FastAPI, no I/O.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union


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
) -> list[dict]:
    """Convert Anthropic messages + system to OpenAI-format message dicts.

    User messages that carry ``image`` blocks are emitted as OpenAI
    multimodal content arrays (``[{type: "text", ...}, {type: "image_url", ...}]``)
    so they flow through llama-server's native vision pathway.
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
            # Assistant content carries text + tool_use; images aren't
            # part of Anthropic's assistant content model.
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
            # Build an ordered part list so text/image interleaving is
            # preserved (e.g. [text, image, text, image]). tool_result
            # blocks become their own OpenAI "tool" role messages.
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
                            p["text"]
                            for p in tc
                            if isinstance(p, dict) and p.get("type") == "text"
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
                # No images — collapse text parts to a plain string so
                # existing text-only callers keep their simple shape.
                text = "\n".join(p["text"] for p in user_parts)
                if text:
                    result.append({"role": "user", "content": text})
            for tr in tool_results:
                result.append(tr)

    return result


def anthropic_tools_to_openai(tools: list) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-tool format."""
    result = []
    for t in tools:
        td = t if isinstance(t, dict) else t.model_dump()
        result.append(
            {
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td.get("description", ""),
                    "parameters": td.get("input_schema", {}),
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

    Returns ``None`` for ``None`` or any unrecognized shape (caller may
    then fall back to its own default, typically ``"auto"``).
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


class AnthropicStreamEmitter:
    """Converts generator events from generate_chat_completion_with_tools()
    into Anthropic Messages SSE strings."""

    def __init__(self) -> None:
        self.block_index: int = 0
        self._text_block_open: bool = False
        self._prev_text: str = ""
        self._usage: dict = {}

    def start(self, message_id: str, model: str) -> list[str]:
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
                        "usage": {"input_tokens": 0, "output_tokens": 0},
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

    def finish(self, stop_reason: str = "end_turn") -> list[str]:
        """Close any open block and emit message_delta + message_stop."""
        events = []
        if self._text_block_open:
            events.append(self._close_block())
        events.append(
            build_anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {
                        "output_tokens": self._usage.get("completion_tokens", 0),
                    },
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
        events = []
        # Close current text block if open
        if self._text_block_open:
            events.append(self._close_block())
        # Open a tool_use block
        self.block_index += 1
        events.append(
            build_anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": event.get("tool_call_id", ""),
                        "name": event.get("tool_name", ""),
                        "input": {},
                    },
                },
            )
        )
        # Emit the arguments as input_json_delta
        args = event.get("arguments", {})
        if args:
            events.append(
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
            )
        return events

    def _handle_tool_end(self, event: dict) -> list[str]:
        events = []
        # Close the tool_use block
        events.append(self._close_block())
        # Emit custom tool_result event (non-standard, ignored by SDKs)
        events.append(
            build_anthropic_sse_event(
                "tool_result",
                {
                    "type": "tool_result",
                    "tool_use_id": event.get("tool_call_id", ""),
                    "content": event.get("result", ""),
                },
            )
        )
        # Open a new text block for the model's next response
        self.block_index += 1
        events.extend(self._open_text_block())
        # Reset text tracking for the next synthesis turn
        self._prev_text = ""
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

    Used for the client-side tool-use pass-through path: the client (e.g. Claude
    Code) sends its own tool definitions in the ``tools`` field and expects to
    execute them itself. We forward them to llama-server and translate the
    streaming response back to Anthropic format without executing anything.
    """

    def __init__(self) -> None:
        self.block_index: int = -1
        self._current_block_type: Optional[str] = None  # "text" | "tool_use" | None
        self._tool_call_states: dict = {}  # delta index -> {block_index, id, name}
        self._usage: dict = {}
        self._stop_reason: str = "end_turn"

    def start(self, message_id: str, model: str) -> list[str]:
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
                        "usage": {"input_tokens": 0, "output_tokens": 0},
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

        # ── Text content ──
        content = delta.get("content")
        if content:
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

        # ── Tool calls (streaming deltas) ──
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            tc_idx = tc.get("index", 0)
            fn = tc.get("function") or {}
            if tc_idx not in self._tool_call_states:
                # New tool call — close prior block, open tool_use block
                if self._current_block_type is not None:
                    events.append(self._close_current_block())
                tc_id = tc.get("id", "")
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
            if finish_reason == "tool_calls":
                self._stop_reason = "tool_use"
            elif finish_reason == "length":
                self._stop_reason = "max_tokens"
            else:
                self._stop_reason = "end_turn"

        return events

    def finish(self) -> list[str]:
        events: list[str] = []
        if self._current_block_type is not None:
            events.append(self._close_current_block())
        events.append(
            build_anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self._stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": self._usage.get("completion_tokens", 0),
                    },
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
