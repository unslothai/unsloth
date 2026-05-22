# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Bracket-tag, rehearsal, and thinking-block-strip logic adapted from forge
# (https://github.com/antoinezambelli/forge), Copyright (c) 2025-2026
# Antoine Zambelli, used under the MIT License.

"""Tool-call parsing and stripping helpers for external reusers.

Kept in lockstep with ``core/inference/tool_call_parser.py`` so external
inference servers (llama-server wrappers, llama-swap, custom shims) can
reuse the same logic without importing the studio backend. Any change
here must also land there.

Handles four serializations (see ``parse_tool_calls_from_text``):

* ``<tool_call>{json}</tool_call>``
* ``<function=name><parameter=k>v</parameter></function>``
* ``[TOOL_CALLS]name{json}`` (Mistral / Devstral fallback)
* ``name[ARGS]{json}`` (reasoning-model rehearsal)
"""

import json
import re

# One level of nested JSON objects in the strip regexes. Deeper nesting
# may leak partial markup, but the call is still parsed correctly.
_BRACKETED_JSON_ONE_LEVEL = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

# Pre-compiled patterns for tool markup stripping.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=\w+>.*?</function>", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\]\w+\s*" + _BRACKETED_JSON_ONE_LEVEL, re.DOTALL),
    re.compile(r"\b\w+\[ARGS\]\s*" + _BRACKETED_JSON_ONE_LEVEL, re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=\w+>.*$", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\]\w+\s*\{.*$", re.DOTALL),
    re.compile(r"\b\w+\[ARGS\]\s*\{.*$", re.DOTALL),
]

# Pre-compiled patterns for tool-call XML parsing.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=(\w+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=(\w+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")

# Thinking blocks stripped before any tool-call pattern is matched.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>|\[THINK\].*?\[/THINK\]", re.DOTALL)

# Mistral ``[TOOL_CALLS]name{json}`` prefix.
_MISTRAL_BRACKET_RE = re.compile(r"\[TOOL_CALLS\](\w+)\s*(?=\{)")

# Rehearsal ``name[ARGS]{json}`` prefix.
_REHEARSAL_RE = re.compile(r"\b(\w+)\[ARGS\]\s*(?=\{)")


def _balanced_json_span(text: str, start: int) -> int | None:
    """Return the end index of a balanced JSON object opening at ``start``,
    or ``None`` if the braces don't balance. Honors escapes and strings.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for j in range(start, len(text)):
        ch = text[j]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_string:
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
    return None


def parse_tool_calls_from_text(content: str) -> list[dict]:
    """Parse tool calls from a mix of XML, bracket-tag, and rehearsal
    markup. Each strategy runs only when the prior one found nothing:

    1. ``<tool_call>{json}</tool_call>``
    2. ``<function=name><parameter=k>v</parameter></function>``
    3. ``[TOOL_CALLS]name{json}``
    4. ``name[ARGS]{json}``

    Closing tags are optional. ``<think>`` and ``[THINK]`` blocks are
    stripped first so post-reasoning calls are recognised.
    """
    # Strip thinking blocks so post-reasoning calls are recognised.
    content = _THINK_TAG_RE.sub("", content)

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

    # Pattern 3: Mistral bracket-tag [TOOL_CALLS]name{json}.
    if not tool_calls:
        for m in _MISTRAL_BRACKET_RE.finditer(content):
            tool_name = m.group(1)
            args_start = m.end()
            args_end = _balanced_json_span(content, args_start)
            if args_end is None:
                continue
            try:
                args = json.loads(content[args_start : args_end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(args, dict):
                continue
            tool_calls.append(
                {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }
            )

    # Pattern 4: Rehearsal name[ARGS]{json}.
    if not tool_calls:
        for m in _REHEARSAL_RE.finditer(content):
            tool_name = m.group(1)
            args_start = m.end()
            args_end = _balanced_json_span(content, args_start)
            if args_end is None:
                continue
            try:
                args = json.loads(content[args_start : args_end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(args, dict):
                continue
            tool_calls.append(
                {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }
            )

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
