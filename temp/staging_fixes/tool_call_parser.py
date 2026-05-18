# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Backend-neutral tool-call XML parser.

Extracts OpenAI-format ``tool_calls`` from model text emitted in either
``<tool_call>{json}</tool_call>`` or ``<function=name><parameter=k>v...``
shape. Closing tags are tolerated when missing because models frequently
omit them.

Used by both the GGUF (llama-server) path and the safetensors path. The
shared helpers keep parsing behaviour identical across backends so the
frontend renders tool calls the same way regardless of where the model
runs.
"""

import json
import re


# Tool XML strip patterns. ``_TOOL_CLOSED_PATS`` removes only closed
# pairs. ``_TOOL_ALL_PATS`` also removes a trailing unclosed run so a
# truncated stream tail does not leak markup into the UI.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=\w+>.*?</function>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=\w+>.*$", re.DOTALL),
]


# Prefixes streamed content can start with when the model is about to
# emit a tool call. The streaming buffer uses these to decide whether
# to hold or yield in-progress text.
TOOL_XML_SIGNALS = ("<tool_call>", "<function=")


# Pre-compiled patterns reused by ``parse_tool_calls_from_text``.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=(\w+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=(\w+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")


def strip_tool_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call XML from streamed text.

    ``final=False`` only removes closed pairs (used during streaming so
    in-progress XML stays buffered). ``final=True`` also removes a
    trailing unclosed run and trims the result.
    """
    pats = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in pats:
        text = pat.sub("", text)
    return text.strip() if final else text


def parse_tool_calls_from_text(content: str) -> list[dict]:
    """Parse OpenAI-format ``tool_calls`` from model text.

    Returns a list of ``{"id", "type", "function": {"name", "arguments"}}``
    dicts. ``arguments`` is always a JSON string so callers can hand it
    straight back into an OpenAI-style response.

    Handles two shapes:

    - JSON inside ``<tool_call>`` tags:
      ``<tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>``
    - XML-style function blocks:
      ``<function=name><parameter=k>v</parameter></function>``

    Closing tags (``</tool_call>``, ``</function>``, ``</parameter>``)
    are all optional since models frequently omit them.
    """
    tool_calls: list[dict] = []

    # Pattern 1: JSON inside <tool_call> tags. Use balanced-brace
    # extraction that skips braces inside JSON strings so embedded
    # ``"{"`` characters don't confuse the depth counter.
    for m in _TC_JSON_START_RE.finditer(content):
        brace_start = m.end() - 1  # position of the opening {
        depth, i = 0, brace_start
        in_string = False
        while i < len(content):
            ch = content[i]
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

    # Pattern 2: XML-style <function=name><parameter=key>value...
    # All closing tags optional. Avoid </function> as a body boundary
    # because code parameter values can contain that literal string.
    if not tool_calls:
        func_starts = list(_TC_FUNC_START_RE.finditer(content))
        for idx, fm in enumerate(func_starts):
            func_name = fm.group(1)
            body_start = fm.end()
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
            body = _TC_FUNC_CLOSE_RE.sub("", body)

            arguments: dict = {}
            param_starts = list(_TC_PARAM_START_RE.finditer(body))
            if len(param_starts) == 1:
                # Single parameter: take everything from after the tag
                # to the end of the body so embedded </parameter> inside
                # code strings does not truncate the value.
                pm = param_starts[0]
                val = body[pm.end() :]
                val = _TC_PARAM_CLOSE_RE.sub("", val)
                arguments[pm.group(1)] = val.strip()
            else:
                for pidx, pm in enumerate(param_starts):
                    param_name = pm.group(1)
                    val_start = pm.end()
                    next_param = (
                        param_starts[pidx + 1].start()
                        if pidx + 1 < len(param_starts)
                        else len(body)
                    )
                    val = body[val_start:next_param]
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


def has_tool_signal(text: str) -> bool:
    """Return True if ``text`` contains any tool-call XML signal."""
    return any(s in text for s in TOOL_XML_SIGNALS)
