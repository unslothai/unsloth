# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Bracket-tag, rehearsal, and thinking-block-strip logic adapted from forge
# (https://github.com/antoinezambelli/forge), Copyright (c) 2025-2026
# Antoine Zambelli, used under the MIT License.

"""
Backend-neutral tool-call parser shared by GGUF and safetensors.

Handles four serializations a local model may emit when bypassing
native function calling:

* ``<tool_call>{json}</tool_call>``
* ``<function=name><parameter=k>v</parameter></function>``
* ``[TOOL_CALLS]name{json}`` (Mistral / Devstral fallback)
* ``name[ARGS]{json}`` (reasoning-model rehearsal)

Closing tags are optional. ``<think>...</think>`` and
``[THINK]...[/THINK]`` blocks are stripped before parsing so calls
emitted after a reasoning preamble are still found.
"""

import json
import re


# One level of nested JSON objects in the strip regexes. Deeper nesting
# may leave partial markup, but the call is still parsed correctly.
_BRACKETED_JSON_ONE_LEVEL = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

# _TOOL_CLOSED_PATS: closed pairs only. _TOOL_ALL_PATS: also trailing
# unclosed runs so truncated tails don't leak markup.
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


# Prefixes the streaming buffer watches for to gate in-progress text.
TOOL_XML_SIGNALS = ("<tool_call>", "<function=", "[TOOL_CALLS]", "[ARGS]")


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
_TC_FUNC_START_RE = re.compile(r"<function=(\w+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=(\w+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")

# Thinking blocks stripped before any tool-call pattern is matched.
_THINK_TAG_RE = re.compile(
    r"<think>.*?</think>|\[THINK\].*?\[/THINK\]", re.DOTALL
)

# Mistral ``[TOOL_CALLS]name{json}`` prefix. Args extracted via
# brace-balance scan to handle nested JSON.
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


def _make_tool_call(name: str, args_obj, id_offset: int, n_existing: int) -> dict:
    """Build the OpenAI tool-call dict shape Studio's loops expect."""
    args_str = args_obj if isinstance(args_obj, str) else json.dumps(args_obj)
    return {
        "id": f"call_{id_offset + n_existing}",
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


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

    Returns ``{"id", "type", "function": {"name", "arguments"}}`` dicts
    with ``arguments`` as a JSON string. Strategies, in order; each
    runs only when prior strategies found nothing:

    1. ``<tool_call>{json}</tool_call>``
    2. ``<function=name><parameter=k>v</parameter></function>``
    3. ``[TOOL_CALLS]name{json}`` (Mistral / Devstral fallback)
    4. ``name[ARGS]{json}`` (reasoning-model rehearsal)

    Closing tags are optional. ``<think>`` and ``[THINK]`` blocks are
    stripped first so post-reasoning calls are still recognised.
    """
    # Strip thinking blocks so post-reasoning calls are recognised.
    content = _THINK_TAG_RE.sub("", content)

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

    # Pattern 3: Mistral bracket-tag [TOOL_CALLS]name{json}. Lower
    # priority than XML forms since bracket-tag is more prose-likely.
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
                _make_tool_call(tool_name, args, id_offset, len(tool_calls))
            )

    # Pattern 4: Rehearsal name[ARGS]{json}. Last-resort fallback.
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
                _make_tool_call(tool_name, args, id_offset, len(tool_calls))
            )

    return tool_calls


def has_tool_signal(text: str) -> bool:
    """Return True if ``text`` contains any tool-call XML signal."""
    return any(s in text for s in TOOL_XML_SIGNALS)
