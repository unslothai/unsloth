# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-call XML parsing and stripping helpers.

Extracted verbatim from studio/backend/core/inference/llama_cpp.py so that
external inference servers (llama-server wrappers, llama-swap, custom
shims) can reuse the same logic without importing the inference
orchestrator, structlog, httpx, or the rest of the studio backend.

The regexes and function bodies are byte-for-byte identical to the
original inline implementation in llama_cpp.py. Any change made here must
preserve that equivalence; tests/python/test_tool_healing_extraction_is_exact.py
verifies it with AST comparison.
"""

import json
import re

# Pre-compiled patterns for tool XML stripping. Hyphen in the
# function/parameter name char-class tracks OpenAI's allowed set so
# MCP tool names with dashes (mcp__srv__list-issues) and parameter
# names with dashes (`issue-number`) parse alongside the built-ins.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*$", re.DOTALL),
]

# Pre-compiled patterns for tool-call XML parsing.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w-]+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")


def parse_tool_calls_from_text(content: str) -> list[dict]:
    """
    Parse tool calls from XML markup in content text.

    Handles formats like:
      <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
      <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
    Closing tags (</tool_call>, </function>, </parameter>) are all optional
    since models frequently omit them.
    """
    tool_calls = []

    # Pattern 1: JSON inside <tool_call> tags.
    # Use balanced-brace extraction that skips braces inside JSON strings.
    for m in _TC_JSON_START_RE.finditer(content):
        brace_start = m.end() - 1  # position of the opening {
        depth, i = 0, brace_start
        in_string = False
        while i < len(content):
            ch = content[i]
            if in_string:
                if ch == "\\" and i + 1 < len(content):
                    i += 2  # skip escaped character
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
                    break
            i += 1
        if depth == 0:
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
                    tc["function"]["arguments"] = json.dumps(
                        tc["function"]["arguments"]
                    )
                tool_calls.append(tc)
            except (json.JSONDecodeError, ValueError):
                pass

    # Pattern 2: XML-style <function=name><parameter=key>value</parameter></function>
    # All closing tags optional -- models frequently omit </parameter>,
    # </function>, and/or </tool_call>.
    if not tool_calls:
        # Step 1: Find all <function=name> positions and extract their bodies.
        # Body boundary: use only </tool_call> or next <function= as hard
        # boundaries.  We avoid using </function> as a boundary because
        # code parameter values can contain that literal string.
        # After extracting, we trim a trailing </function> if present.
        func_starts = list(_TC_FUNC_START_RE.finditer(content))
        for idx, fm in enumerate(func_starts):
            func_name = fm.group(1)
            body_start = fm.end()
            # Hard boundaries: next <function= tag or </tool_call>
            next_func = (
                func_starts[idx + 1].start()
                if idx + 1 < len(func_starts)
                else len(content)
            )
            end_tag = _TC_END_TAG_RE.search(content[body_start:])
            if end_tag:
                body_end = body_start + end_tag.start()
            else:
                body_end = len(content)
            body_end = min(body_end, next_func)
            body = content[body_start:body_end]
            # Trim trailing </function> if present (it's the real closing tag)
            body = _TC_FUNC_CLOSE_RE.sub("", body)

            # Step 2: Extract parameters from body.
            # For single-parameter functions (the common case: code, command,
            # query), use body end as the only boundary to avoid false matches
            # on </parameter> inside code strings.
            arguments = {}
            param_starts = list(_TC_PARAM_START_RE.finditer(body))
            if len(param_starts) == 1:
                # Single parameter: value is everything from after the tag
                # to end of body, trimming any trailing </parameter>.
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
                    # Trim trailing </parameter> if present
                    val = _TC_PARAM_CLOSE_RE.sub("", val)
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
