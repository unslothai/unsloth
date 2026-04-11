# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Anthropic Messages API ↔ OpenAI format translation utilities.

Pure functions and a stateful stream emitter — no FastAPI, no I/O.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union


def anthropic_messages_to_openai(
    messages: list[dict],
    system: Optional[Union[str, list]] = None,
) -> list[dict]:
    """Convert Anthropic messages + system to OpenAI-format message dicts."""
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

        # Content is a list of blocks
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        for block in content:
            b = block if isinstance(block, dict) else block.model_dump()
            btype = b.get("type", "")

            if btype == "text":
                text_parts.append(b["text"])
            elif btype == "tool_use":
                tool_calls.append({
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    },
                })
            elif btype == "tool_result":
                tc = b.get("content", "")
                if isinstance(tc, list):
                    tc = " ".join(
                        p["text"] for p in tc
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": b["tool_use_id"],
                    "content": str(tc),
                })

        if role == "assistant":
            msg_dict: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg_dict["content"] = "\n".join(text_parts)
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            result.append(msg_dict)
        elif role == "user":
            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})
            for tr in tool_results:
                result.append(tr)

    return result


def anthropic_tools_to_openai(tools: list) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-tool format."""
    result = []
    for t in tools:
        td = t if isinstance(t, dict) else t.model_dump()
        result.append({
            "type": "function",
            "function": {
                "name": td["name"],
                "description": td.get("description", ""),
                "parameters": td.get("input_schema", {}),
            },
        })
    return result


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
        events.append(build_anthropic_sse_event("message_start", {
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
        }))
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
        events.append(build_anthropic_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "output_tokens": self._usage.get("completion_tokens", 0),
            },
        }))
        events.append(build_anthropic_sse_event("message_stop", {
            "type": "message_stop",
        }))
        return events

    def _handle_content(self, event: dict) -> list[str]:
        cumulative = event.get("text", "")
        new_text = cumulative[len(self._prev_text):]
        self._prev_text = cumulative
        if not new_text:
            return []
        if not self._text_block_open:
            events = self._open_text_block()
        else:
            events = []
        events.append(build_anthropic_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.block_index,
            "delta": {"type": "text_delta", "text": new_text},
        }))
        return events

    def _handle_tool_start(self, event: dict) -> list[str]:
        events = []
        # Close current text block if open
        if self._text_block_open:
            events.append(self._close_block())
        # Open a tool_use block
        self.block_index += 1
        events.append(build_anthropic_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {
                "type": "tool_use",
                "id": event.get("tool_call_id", ""),
                "name": event.get("tool_name", ""),
                "input": {},
            },
        }))
        # Emit the arguments as input_json_delta
        args = event.get("arguments", {})
        if args:
            events.append(build_anthropic_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(args),
                },
            }))
        return events

    def _handle_tool_end(self, event: dict) -> list[str]:
        events = []
        # Close the tool_use block
        events.append(self._close_block())
        # Open a new text block for the model's next response
        self.block_index += 1
        events.extend(self._open_text_block())
        # Reset text tracking for the next synthesis turn
        self._prev_text = ""
        return events

    def _open_text_block(self) -> list[str]:
        self._text_block_open = True
        return [build_anthropic_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {"type": "text", "text": ""},
        })]

    def _close_block(self) -> str:
        self._text_block_open = False
        return build_anthropic_sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": self.block_index,
        })
