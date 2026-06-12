# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-call XML parsing and stripping helpers.

Extracted verbatim from studio/backend/core/inference/llama_cpp.py so external
inference servers can reuse the logic without importing the inference
orchestrator, structlog, httpx, or the rest of the studio backend.

Regexes and bodies are byte-for-byte identical to the original; any change must
preserve that. test_tool_healing_extraction_is_exact.py verifies via AST.
"""

import json
import re

# Pre-compiled patterns for tool XML stripping. The hyphen in the name
# char-class lets dashed MCP tool/parameter names (mcp__srv__list-issues,
# issue-number) parse alongside the built-ins.
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
    Closing tags (</tool_call>, </function>, </parameter>) are all
    optional since models frequently omit them.
    """
    tool_calls = []

    # Pattern 1: JSON inside <tool_call> tags. Balanced-brace extraction that
    # skips braces inside JSON strings.
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
                    tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
                tool_calls.append(tc)
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
