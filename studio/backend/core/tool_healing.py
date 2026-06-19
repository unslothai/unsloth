# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lightweight tool-call XML parsing and stripping helpers.

External inference servers import this module without pulling in the inference
orchestrator, structlog, httpx, or the rest of the studio backend.
"""

import json
import re

# Pre-compiled patterns for tool XML stripping. The hyphen in the name
# char-class lets dashed MCP tool/parameter names (mcp__srv__list-issues,
# issue-number) parse alongside the built-ins.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*$", re.DOTALL),
]

# Pre-compiled patterns for tool-call XML parsing.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_GEMMA_START_RE = re.compile(r"<\|tool_call>call:([\w-]+)\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w-]+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")
_GEMMA_QUOTE = '<|"|>'


def _balanced_brace_end(content: str, brace_start: int) -> int:
    depth = 0
    i = brace_start
    in_string = False
    in_gemma_string = False
    while i < len(content):
        if content.startswith(_GEMMA_QUOTE, i):
            in_gemma_string = not in_gemma_string
            i += len(_GEMMA_QUOTE)
            continue
        ch = content[i]
        if in_gemma_string:
            i += 1
            continue
        if in_string:
            if ch == "\\" and i + 1 < len(content):
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _normalise_gemma_quoted_strings(src: str) -> str:
    parts: list[str] = []
    i = 0
    while i < len(src):
        if not src.startswith(_GEMMA_QUOTE, i):
            parts.append(src[i])
            i += 1
            continue
        end = src.find(_GEMMA_QUOTE, i + len(_GEMMA_QUOTE))
        if end < 0:
            parts.append(src[i:])
            break
        raw_value = src[i + len(_GEMMA_QUOTE) : end]
        parts.append(json.dumps(raw_value))
        i = end + len(_GEMMA_QUOTE)
    return "".join(parts)


def _quote_gemma_object_keys(src: str) -> str:
    parts: list[str] = []
    i = 0
    in_string = False
    while i < len(src):
        ch = src[i]
        if in_string:
            parts.append(ch)
            if ch == "\\" and i + 1 < len(src):
                parts.append(src[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            parts.append(ch)
            i += 1
            continue
        if ch not in "{,":
            parts.append(ch)
            i += 1
            continue

        parts.append(ch)
        i += 1
        key_start = i
        while i < len(src) and src[i].isspace():
            i += 1
        key_name_start = i
        while i < len(src) and (src[i].isalnum() or src[i] in "_-"):
            i += 1
        key_name = src[key_name_start:i]
        colon_pos = i
        while colon_pos < len(src) and src[colon_pos].isspace():
            colon_pos += 1
        if key_name and colon_pos < len(src) and src[colon_pos] == ":":
            parts.append(src[key_start:key_name_start])
            parts.append(json.dumps(key_name))
            parts.append(src[i:colon_pos])
            parts.append(":")
            i = colon_pos + 1
        else:
            parts.append(src[key_start:i])
    return "".join(parts)


def _gemma_arguments_to_json(args_src: str) -> dict:
    args_src = args_src.strip()
    if not args_src:
        return {}
    src = _normalise_gemma_quoted_strings(args_src)
    src = "{" + src + "}"
    src = _quote_gemma_object_keys(src)
    return json.loads(src)


def parse_tool_calls_from_text(content: str) -> list[dict]:
    """
    Parse tool calls from XML markup in content text.

    Handles formats like:
      <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
      <|tool_call>call:web_search{query:"..."}<tool_call|>
      <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
    Closing tags (</tool_call>, </function>, </parameter>) are all
    optional since models frequently omit them.
    """
    tool_calls = []

    # Pattern 1: JSON inside <tool_call> tags. Balanced-brace extraction that
    # skips braces inside JSON strings.
    for m in _TC_JSON_START_RE.finditer(content):
        brace_start = m.end() - 1  # position of the opening {
        i = _balanced_brace_end(content, brace_start)
        if i >= 0:
            json_str = content[brace_start : i + 1]
            try:
                obj = json.loads(json_str)
                tc = {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": obj.get("name", ""),
                        "arguments": obj.get("arguments", {}),
                    },
                }
                if isinstance(tc["function"]["arguments"], dict):
                    tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
                tool_calls.append(tc)
            except (json.JSONDecodeError, ValueError):
                pass

    # Pattern 1b: Gemma 4 native <|tool_call>call:name{key:value}<tool_call|>.
    for m in _TC_GEMMA_START_RE.finditer(content):
        brace_start = m.end() - 1
        i = _balanced_brace_end(content, brace_start)
        if i < 0:
            continue
        try:
            tool_calls.append(
                {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": m.group(1),
                        "arguments": json.dumps(_gemma_arguments_to_json(content[m.end() : i])),
                    },
                }
            )
        except (json.JSONDecodeError, ValueError):
            pass

    # Pattern 2: XML-style <function=name><parameter=key>value</parameter></function>
    # All closing tags optional; models frequently omit them.
    if not tool_calls:
        # Step 1: Find <function=name> positions and extract bodies. Use only
        # </tool_call> or the next <function= as hard boundaries (</function>
        # can appear in code values); trim a trailing </function> afterwards.
        func_starts = list(_TC_FUNC_START_RE.finditer(content))
        for idx, fm in enumerate(func_starts):
            func_name = fm.group(1)
            body_start = fm.end()
            # Boundaries: next <function= tag or </tool_call>
            next_func = func_starts[idx + 1].start() if idx + 1 < len(func_starts) else len(content)
            end_tag = _TC_END_TAG_RE.search(content[body_start:])
            if end_tag:
                body_end = body_start + end_tag.start()
            else:
                body_end = len(content)
            body_end = min(body_end, next_func)
            body = content[body_start:body_end]
            body = _TC_FUNC_CLOSE_RE.sub("", body)  # trim closing </function>

            # Step 2: Extract parameters from body. For single-parameter
            # functions, use body end as the only boundary to avoid matching
            # </parameter> inside code strings.
            arguments = {}
            param_starts = list(_TC_PARAM_START_RE.finditer(body))
            if len(param_starts) == 1:
                # Value is everything after the tag to end of body, less a
                # trailing </parameter>.
                pm = param_starts[0]
                val = body[pm.end() :]
                val = _TC_PARAM_CLOSE_RE.sub("", val)
                arguments[pm.group(1)] = val.strip()
            else:
                for pidx, pm in enumerate(param_starts):
                    param_name = pm.group(1)
                    val_start = pm.end()
                    # Value ends at next <parameter= or end of body
                    next_param = (
                        param_starts[pidx + 1].start()
                        if pidx + 1 < len(param_starts)
                        else len(body)
                    )
                    val = body[val_start:next_param]
                    val = _TC_PARAM_CLOSE_RE.sub("", val)  # trim trailing </parameter>
                    arguments[param_name] = val.strip()

            tc = {
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments),
                },
            }
            tool_calls.append(tc)
    return tool_calls


def strip_tool_call_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call XML markup from text.

    When ``final`` is False, only fully closed tool-call blocks are removed.
    When ``final`` is True, trailing incomplete tool-call blocks are removed
    too, and the result is stripped of surrounding whitespace.
    """
    patterns = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in patterns:
        text = pat.sub("", text)
    return text.strip() if final else text
