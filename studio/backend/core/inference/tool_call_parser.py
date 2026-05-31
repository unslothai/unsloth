# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Backend-neutral tool-call XML parser shared by GGUF and safetensors.
Tolerates missing closing tags in either ``<tool_call>{json}</tool_call>``
or ``<function=name><parameter=k>v...`` shape.
"""

import json
import re


# _TOOL_CLOSED_PATS: closed pairs only. _TOOL_ALL_PATS: also trailing
# unclosed runs so truncated tails don't leak markup.
# Function-name char set tracks OpenAI's ^[a-zA-Z0-9_-]{1,64}$ so MCP
# tool names that contain a hyphen (e.g. mcp__srv__list-issues) parse
# the same as the built-in web_search/python/terminal names.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*$", re.DOTALL),
]


# Prefixes the streaming buffer watches for to gate in-progress text.
TOOL_XML_SIGNALS = ("<tool_call>", "<function=")


# Nudges + error prefixes shared by the GGUF and safetensors loops.
TOOL_ERROR_PREFIXES = (
    "Error",
    "Search failed",
    "Execution error",
    "Blocked:",
    "Exit code",
    "Failed to fetch",
    "Failed to resolve",
    "No query provided",
)

DUPLICATE_CALL_NUDGE = (
    "You already made this exact call. Do not repeat the same tool "
    "call. Try a different approach: fetch a URL from previous "
    "results, use Python to process data you already have, or "
    "provide your final answer now."
)

TOOL_ERROR_NUDGE = (
    "\n\nThe tool call encountered an issue. Please try a different "
    "approach or rephrase your request."
)

BUDGET_EXHAUSTED_NUDGE = (
    "You have used all available tool calls. Based on everything you "
    "have found so far, provide your final answer now. Do not call "
    "any more tools."
)


# Pre-compiled patterns reused by ``parse_tool_calls_from_text``.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
# Parameter names can carry hyphens too (e.g. MCP tool schemas with
# `issue-number`, `repo-name`); using `\w+` here dropped those keys.
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w-]+)>\s*")
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


def parse_tool_calls_from_text(content: str, *, id_offset: int = 0) -> list[dict]:
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

    # Pattern 1: <tool_call>{json}. Balanced-brace scan that skips
    # braces inside JSON strings.
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
                    "id": f"call_{id_offset + len(tool_calls)}",
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

    # Pattern 2: <function=name><parameter=k>v... -- closing tags
    # optional; don't use </function> as body boundary because code
    # values can contain that literal.
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
                # Single param: take everything to body end so
                # embedded </parameter> in code strings is preserved.
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
                "id": f"call_{id_offset + len(tool_calls)}",
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
