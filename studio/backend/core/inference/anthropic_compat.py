# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Anthropic Messages <-> OpenAI format translation; pure functions and stream emitters, no I/O."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional, Union

from core.inference.tool_loop_controller import is_tool_error


def openai_finish_to_anthropic_stop(finish_reason, had_tool_calls = False) -> str:
    """Maps finish_reason to stop_reason: length to max_tokens (truncation wins over tool use), tool_calls to tool_use, stop_sequence kept, else end_turn"""
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls" or had_tool_calls:
        return "tool_use"
    if finish_reason == "stop_sequence":
        return "stop_sequence"
    # "stop", None, and any unknown value collapse to end_turn.
    return "end_turn"


def anthropic_tool_use_id(upstream_id = None, prefix = "toolu_") -> str:
    """Upstream tool id if it starts with prefix, else a fresh prefix plus 24 hex digits."""
    if upstream_id and isinstance(upstream_id, str) and upstream_id.startswith(prefix):
        return upstream_id
    return f"{prefix}{uuid.uuid4().hex[:24]}"


_WEB_SEARCH_HIT = re.compile(r"^Title: (.*)\nURL: (\S+)", re.MULTILINE)
# Claude Code's billing line varies per request and would break prefix caching.
_BILLING_LINE = re.compile(r"\Ax-anthropic-billing-header:[^\n]*(?:\n|\Z)")


def web_search_tool_result_content(
    result: str, arguments: Optional[dict] = None
) -> Union[list, dict]:
    """``content`` of a web_search_tool_result: the search hits, the page a ``url`` call fetched,
    or the API's error object when the tool failed."""
    if is_tool_error(result):
        return {"type": "web_search_tool_result_error", "error_code": "unavailable"}
    fetched = str((arguments or {}).get("url") or "").strip()
    if fetched:
        hits = [(fetched, fetched)] if result.strip() else []
    else:
        hits = _WEB_SEARCH_HIT.findall(result)
    return [
        {
            "type": "web_search_result",
            "title": title.strip(),
            "url": url,
            "encrypted_content": "",
            "page_age": None,
        }
        for title, url in hits
    ]


def _web_search_result_text(content: Any) -> str:
    """Render a replayed web_search_tool_result ``content`` in web_search's own output format."""
    if isinstance(content, dict):
        return f"Search failed: {content.get('error_code', 'unavailable')}"
    hits = [
        f"Title: {hit.get('title', '')}\nURL: {hit['url']}"
        for hit in content or []
        if isinstance(hit, dict) and hit.get("url")
    ]
    return "\n\n---\n\n".join(hits) or "No results found."


TOOL_RESULT_IMAGE_OMITTED = "[image omitted: this model cannot view images]"
DOCUMENT_OMITTED = "[document omitted: only text documents can be read]"
DOCUMENT_IMAGE_OMITTED = "[image omitted: images inside documents are not read]"


def _anthropic_block_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            texts.append(b["text"])
        elif b.get("type") == "image":
            texts.append(DOCUMENT_IMAGE_OMITTED)
    return texts


def anthropic_reference_block_text(block: dict) -> str:
    btype = block.get("type")
    source = block.get("source")
    if btype == "search_result":
        labels = {"Title": block.get("title"), "Source": source}
        body = _anthropic_block_texts(block.get("content"))
    elif btype == "document":
        labels = {"Title": block.get("title"), "Context": block.get("context")}
        stype = source.get("type") if isinstance(source, dict) else None
        if stype == "text" and isinstance(source.get("data"), str):
            body = [source["data"]]
        elif stype == "content":
            body = _anthropic_block_texts(source.get("content"))
        else:
            body = [DOCUMENT_OMITTED]
    else:
        return ""
    header = [
        f"{label}: {value}" for label, value in labels.items() if isinstance(value, str) and value
    ]
    return "\n".join([*header, *body])


def _anthropic_image_block_to_openai_part(block: dict) -> Optional[dict]:
    """Accepts base64 or url image sources; returns None for a malformed source so the caller skips it."""
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    stype = source.get("type")
    if stype == "base64":
        data = source.get("data")
        if not isinstance(data, str) or not data:
            return None
        media_type = source.get("media_type") or "image/jpeg"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    if stype == "url":
        url = source.get("url")
        if not isinstance(url, str) or not url:
            return None
        return {"type": "image_url", "image_url": {"url": url}}
    return None


def _tool_error_content(content: Union[str, list]) -> Union[str, list]:
    if isinstance(content, list):
        return [{"type": "text", "text": "Error:"}, *content]
    if not content:
        return "Error: tool returned no content"
    return content if content.startswith("Error:") else f"Error: {content}"


def anthropic_messages_to_openai(
    messages: list[dict],
    system: Optional[Union[str, list]] = None,
    preserve_thinking: bool = False,
    tool_result_images: bool = True,
) -> list[dict]:
    """Convert Anthropic messages + system to OpenAI-format message dicts.

    User messages with ``image`` blocks are emitted as OpenAI multimodal
    content arrays (``[{type: "text", ...}, {type: "image_url", ...}]``) so
    they flow through llama-server's native vision pathway.

    ``tool_result_images=False`` turns tool-result images into a text note for a
    text-only model; clients resend history, so rejecting would fail every later turn.

    ``preserve_thinking`` keeps replayed assistant ``thinking`` blocks as
    ``reasoning_content`` on the converted message, so templates that render
    historical reasoning (Qwen3.6-style ``preserve_thinking``) actually receive
    it; otherwise only the current turn's thinking (after the last real query) is kept. ``redacted_thinking``
    carries only ciphertext and is always dropped.
    """
    result: list[dict] = []

    if system:
        if isinstance(system, str):
            if system := _BILLING_LINE.sub("", system, count = 1):
                result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            parts = [
                kept
                for part in parts
                if (kept := _BILLING_LINE.sub("", part, count = 1)) or kept == part
            ]
            if parts:
                result.append({"role": "system", "content": "\n".join(parts)})

    # The current turn's thinking stays even with preservation off: templates render reasoning after
    # the last real query (a user message that is not only tool results), as it was generated.
    last_query = -1
    for index, msg in enumerate(messages):
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content
        if role == "user" and (
            isinstance(content, str)
            or any(
                (block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
                != "tool_result"
                for block in content
            )
        ):
            last_query = index

    for index, msg in enumerate(messages):
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content
        keep_thinking = preserve_thinking or index > last_query

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if role == "assistant":
            # Assistant turns map to text and tool_use; web search pairs split into call, tool result, then text
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            thinking_parts: list[str] = []
            server_calls: dict[str, dict] = {}
            flushed = False

            def _flush_assistant() -> None:
                msg_dict: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    msg_dict["content"] = "\n".join(text_parts)
                if thinking_parts:
                    msg_dict["reasoning_content"] = "\n\n".join(thinking_parts)
                if tool_calls:
                    msg_dict["tool_calls"] = list(tool_calls)
                result.append(msg_dict)
                text_parts.clear()
                tool_calls.clear()
                thinking_parts.clear()

            for block in content:
                b = block if isinstance(block, dict) else block.model_dump()
                btype = b.get("type", "")
                if btype == "text":
                    text_parts.append(b["text"])
                elif btype == "thinking" and keep_thinking:
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
                elif btype == "server_tool_use":
                    server_calls[b.get("id")] = b
                elif btype == "web_search_tool_result" and b.get("tool_use_id") in server_calls:
                    call = server_calls.pop(b["tool_use_id"])
                    tool_calls.append(
                        {
                            "id": b["tool_use_id"],
                            "type": "function",
                            "function": {
                                "name": call.get("name") or "web_search",
                                "arguments": json.dumps(call.get("input") or {}),
                            },
                        }
                    )
                    _flush_assistant()
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": _web_search_result_text(b.get("content")),
                        }
                    )
                    flushed = True
            if text_parts or tool_calls or thinking_parts or not flushed:
                _flush_assistant()
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
                elif reference := anthropic_reference_block_text(b):
                    user_parts.append({"type": "text", "text": reference})
                elif btype == "image":
                    part = _anthropic_image_block_to_openai_part(b)
                    if part is not None:
                        user_parts.append(part)
                        has_image = True
                elif btype == "tool_result":
                    tc = b.get("content", "")
                    if isinstance(tc, list):
                        parts = []
                        for item in tc:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") == "text":
                                parts.append({"type": "text", "text": item["text"]})
                            elif reference := anthropic_reference_block_text(item):
                                parts.append({"type": "text", "text": reference})
                            elif item.get("type") == "image":
                                if not tool_result_images:
                                    parts.append(
                                        {"type": "text", "text": TOOL_RESULT_IMAGE_OMITTED}
                                    )
                                    continue
                                part = _anthropic_image_block_to_openai_part(item)
                                if part is not None:
                                    parts.append(part)
                        tc = (
                            parts
                            if any(p["type"] == "image_url" for p in parts)
                            else "\n".join(p["text"] for p in parts)
                        )
                    if b.get("is_error"):
                        tc = _tool_error_content(tc)
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": tc if isinstance(tc, list) else str(tc),
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
    """Gemma 2/3 have no tool role and check alternation by index parity, so llama-server would 400."""
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
        content = response["content"]
        images = []
        if isinstance(content, list):
            images = [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
        if images:
            # Images must stay real parts; other tool output stays inside the tool_response wrapper so it never reads as user text and still matches the archive
            response["content"] = [p for p in content if p not in images]
            folded_content = [
                {"type": "text", "text": json.dumps({"tool_response": response}, indent = 2)},
                *images,
            ]
        else:
            folded_content = json.dumps({"tool_response": response}, indent = 2)
        out.append({"role": "user", "content": folded_content})
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
    """Maps tool_choice to OpenAI: any to required, tool name to function dict; None or unrecognized returns None, so the caller defaults, usually auto"""
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
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def cached_prompt_tokens(usage: Optional[dict]) -> Optional[int]:
    """Prompt tokens the server reused from its cache (llama-server ``prompt_tokens_details``), or None."""
    details = (usage or {}).get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if isinstance(cached, bool) or not isinstance(cached, (int, float)):
        return None
    return max(0, int(cached))


def anthropic_input_split(usage: Optional[dict]) -> tuple[int, int]:
    """(input_tokens, cache_read_input_tokens): Anthropic counts cache reads apart from input, so the two
    still sum to the prompt. Nothing is ever reported as cache creation."""
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = min(prompt, cached_prompt_tokens(usage) or 0)
    return prompt - cached, cached


def _message_delta_usage(usage: Optional[dict]) -> dict:
    """Usage block for a message_delta event (cumulative token counts). ``usage`` may be None when a
    metadata event carried usage=None (e.g. only finish_reason set)."""
    usage = usage or {}
    input_tokens, cache_read = anthropic_input_split(usage)
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cache_read,
        "output_tokens": usage.get("completion_tokens", 0),
    }


def _partial_tag_suffix_len(text: str, tag: str) -> int:
    """A streamed delta can split a ``<think>`` tag anywhere; the caller holds back this suffix."""
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
        # Off when thinking is disabled or the model is non-reasoning, so literal <think> streams as text
        self._parse_think = parse_think
        self.block_index: int = 0
        self._block_index_used: bool = False
        self._text_block_open: bool = False
        self._thinking_block_open: bool = False
        self._open_tool_call_id: Optional[str] = None
        # Reuse the toolu_* id from content_block_start for the paired tool_result so they correlate
        self._open_tool_use_id: Optional[str] = None
        self._open_tool_args_sent: bool = False
        self._open_tool_args: dict = {}
        self._prev_text: str = ""
        # <think> routing: the generator folds reasoning_content into the cumulative text as <think>...</think> markup
        # (the UI chat parses it), but Anthropic clients expect typed thinking blocks. Split the markup back out: text
        # inside the tags streams as thinking_delta in a "thinking" content block, everything else as ordinary text.
        # _tag_buf holds back a trailing partial tag until the next delta decides it.
        self._route_mode: str = "text"
        self._tag_buf: str = ""
        # Hold leading thinking whitespace until real reasoning arrives, so blank traces open no block
        self._thinking_ws_hold: str = ""
        # Genuine reasoning only arrives as one LEADING <think> block; any later one is literal.
        self._think_consumed: bool = False
        self._turn_has_text: bool = False
        # wrapped counts leading think tags the generator opened; a tag parses as reasoning only with a wrap, so literal markup stays text
        self._think_provenance = think_provenance
        self._wraps_consumed: int = 0
        # The block spans exactly N reasoning chars, so a literal "</think>" inside never ends it.
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
        elif etype == "status" and not event.get("text"):
            # A new cumulative segment can start without a tool_end (a rejected call).
            self._prev_text = ""
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
                    # Already delivered; re-including it below sent it twice.
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
                    # Take exactly the provenance reasoning length, so a literal </think> inside it stays thinking
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
            # Qwen3 templates emit an empty think block when thinking is off; hold whitespace until real reasoning, else an empty block lands on every answer
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

        web_search = event.get("tool_name") == "web_search"
        self._alloc_block_index()
        self._open_tool_call_id = tool_call_id
        self._open_tool_use_id = anthropic_tool_use_id(
            tool_call_id, "srvtoolu_" if web_search else "toolu_"
        )
        self._open_tool_args_sent = False
        self._open_tool_args = {}
        events.append(
            build_anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {
                        "type": "server_tool_use" if web_search else "tool_use",
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
        self._open_tool_args = args
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
        # Reuse the content_block_start id; map the raw id only when no tool_start preceded this end
        tool_use_id = self._open_tool_use_id or anthropic_tool_use_id(event.get("tool_call_id", ""))
        tool_args, self._open_tool_args = self._open_tool_args, {}
        self._open_tool_call_id = None
        self._open_tool_use_id = None
        self._open_tool_args_sent = False
        if event.get("tool_name") == "web_search":
            self._alloc_block_index()
            events.append(
                build_anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {
                            "type": "web_search_tool_result",
                            "tool_use_id": tool_use_id,
                            "content": web_search_tool_result_content(
                                event.get("result", ""), tool_args
                            ),
                        },
                    },
                )
            )
            events.append(self._close_block())
        else:
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
        # Reset per synthesis turn: a new turn may legitimately open with its own leading <think> block
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
        # With thinking off, llama-server can still shunt a literal think tag into reasoning_content; rebuild it as visible text, not a thinking block
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
        """Only tools in ``allowed_tools`` are promoted; never enabled for Unsloth's own tool loop."""
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

        # llama-server splits think into reasoning_content on tool-calling turns; reading only content drops the trace so the model seems not to think
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
        # Not elif: one chunk can carry the last reasoning AND content, and the closing tag goes between.
        if self._reasoning_text_open and (
            delta.get("content") or delta.get("tool_calls") or finish_reason
        ):
            self._reasoning_text_open = False
            events.extend(self._emit_text_delta("</think>"))

        # Structured tool calls win over healing: flush held text that preceded the call, then relay verbatim
        if delta.get("tool_calls") and self._healer is not None and not self._healer.dormant:
            for kind, value in self._healer.structured_tool_call_seen():
                if kind == "text" and value:
                    events.extend(self._emit_text_delta(value))

        content = delta.get("content")
        if content and self._healer is not None and not self._healer.dormant:
            # Healer output: held or promoted spans become synthetic tool_use blocks; the rest streams as text
            for kind, value in self._healer.feed(content):
                if kind == "text":
                    events.extend(self._emit_text_delta(value))
                else:
                    events.extend(self._emit_healed_tool_use(value))
        elif content:
            events.extend(self._emit_text_delta(content))

        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            tc_idx = tc.get("index", 0)
            fn = tc.get("function") or {}
            if (
                self._heal_disable_parallel
                and tc_idx not in self._tool_call_states
                and (self._healed_call_count + len(self._tool_call_states)) >= 1
            ):
                # disable_parallel_tool_use: a healed call already used the single slot, so drop this native call and its later argument deltas
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
            for kind, value in self._healer.finalize(promote = self._stop_reason != "max_tokens"):
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
        # A healed call arrives whole, so emit one input_json_delta and close; close open text first
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
