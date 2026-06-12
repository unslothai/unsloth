# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Backend-neutral tool-call XML parser shared by GGUF and safetensors.
Tolerates missing closing tags in either ``<tool_call>{json}</tool_call>``
or ``<function=name><parameter=k>v...`` shape.
"""

import json
import re


# _TOOL_CLOSED_PATS: closed pairs only. _TOOL_ALL_PATS: also trailing unclosed
# runs so truncated tails don't leak markup. The [\w-] name set matches OpenAI's
# so hyphenated MCP tool names (mcp__srv__list-issues) parse like built-ins.
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

RENDER_HTML_REPEAT_NUDGE = (
    "Error: render_html was already called for this response. Do not call "
    "render_html again in this response unless the user asks for changes. "
    "Provide the final answer now."
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

# The exact-args dup guard misses paraphrased re-searches, so also cap executed
# KB searches per turn, then nudge.
RAG_MAX_SEARCHES_PER_TURN = 3
RAG_SEARCH_CAP_NUDGE = (
    "You have already searched the knowledge base several times this turn. "
    "Do not search again. Answer the question using the passages already "
    "retrieved above; if they do not contain the answer, say so plainly."
)


# Pre-compiled patterns reused by ``parse_tool_calls_from_text``.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
# [\w-] so hyphenated MCP param names (issue-number) aren't dropped.
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w-]+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")
_PARAM_CLOSE_TAG = "</parameter>"
_FUNC_CLOSE_TAG = "</function>"


def _inside_open_parameter(content: str, pos: int) -> bool:
    """Return True when ``pos`` falls inside an unclosed parameter value."""
    last_param_start = -1
    for match in _TC_PARAM_START_RE.finditer(content, 0, pos):
        last_param_start = match.start()
    if last_param_start < 0:
        return False
    last_param_close = content.rfind(_PARAM_CLOSE_TAG, 0, pos)
    last_func_close = content.rfind(_FUNC_CLOSE_TAG, 0, pos)
    return last_param_start > max(last_param_close, last_func_close)


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


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Parse OpenAI-format ``tool_calls`` from model text.

    Returns a list of ``{"id", "type", "function": {"name", "arguments"}}``
    dicts. ``arguments`` is always a JSON string so callers can hand it
    straight back into an OpenAI-style response.

    Handles two shapes:

    - JSON inside ``<tool_call>`` tags:
      ``<tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>``
    - XML-style function blocks:
      ``<function=name><parameter=k>v</parameter></function>``

    ``allow_incomplete=True`` keeps the historical healing behavior for
    missing closing tags. ``allow_incomplete=False`` accepts only
    well-formed wrappers so disabled Auto-Heal can still parse valid
    local tool protocol without repairing truncated output.
    """
    tool_calls: list[dict] = []

    # Pattern 1: <tool_call>{json}. Balanced-brace scan, skipping braces in
    # JSON strings.
    for m in _TC_JSON_START_RE.finditer(content):
        brace_start = m.end() - 1  # opening {
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
        if depth != 0:
            continue
        if not allow_incomplete:
            tail_after_json = content[i + 1 :].lstrip()
            if _TC_END_TAG_RE.match(tail_after_json) is None:
                continue
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
                tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
            tool_calls.append(tc)
        except (json.JSONDecodeError, ValueError):
            pass

    # Pattern 2: <function=name><parameter=k>v... -- closing tags optional;
    # </function> isn't a body boundary since code values can contain it.
    if not tool_calls:
        func_starts = [
            fm
            for fm in _TC_FUNC_START_RE.finditer(content)
            if not _inside_open_parameter(content, fm.start())
        ]
        for idx, fm in enumerate(func_starts):
            func_name = fm.group(1)
            body_start = fm.end()
            next_func = func_starts[idx + 1].start() if idx + 1 < len(func_starts) else len(content)
            end_tag = _TC_END_TAG_RE.search(content[body_start:])
            if end_tag:
                body_end = body_start + end_tag.start()
            else:
                body_end = len(content)
            body_end = min(body_end, next_func)
            body = content[body_start:body_end]
            if not allow_incomplete:
                # Bound the body at the closing </function> tag rather than
                # the end of the response, so a complete call followed by
                # trailing prose is still accepted (matching the JSON-style
                # <tool_call> path, which already tolerates trailing text).
                # rfind picks the last </function>, so a literal </function>
                # inside a code parameter value stays in the body.
                close_idx = body.rfind(_FUNC_CLOSE_TAG)
                if close_idx < 0:
                    continue
                body = body[:close_idx]
            else:
                body = _TC_FUNC_CLOSE_RE.sub("", body)

            arguments: dict = {}
            param_starts = list(_TC_PARAM_START_RE.finditer(body))
            if len(param_starts) == 1:
                # Single param: take everything to body end so an embedded
                # </parameter> in code strings is preserved.
                pm = param_starts[0]
                val = body[pm.end() :]
                if not allow_incomplete:
                    stripped_val = val.rstrip()
                    if not stripped_val.endswith(_PARAM_CLOSE_TAG):
                        continue
                    val = stripped_val[: -len(_PARAM_CLOSE_TAG)]
                else:
                    val = _TC_PARAM_CLOSE_RE.sub("", val)
                arguments[pm.group(1)] = val.strip()
            else:
                valid_params = True
                for pidx, pm in enumerate(param_starts):
                    param_name = pm.group(1)
                    val_start = pm.end()
                    next_param = (
                        param_starts[pidx + 1].start()
                        if pidx + 1 < len(param_starts)
                        else len(body)
                    )
                    val = body[val_start:next_param]
                    if not allow_incomplete:
                        stripped_val = val.rstrip()
                        if not stripped_val.endswith(_PARAM_CLOSE_TAG):
                            valid_params = False
                            break
                        val = stripped_val[: -len(_PARAM_CLOSE_TAG)]
                    else:
                        val = _TC_PARAM_CLOSE_RE.sub("", val)
                    arguments[param_name] = val.strip()
                if not valid_params:
                    continue

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
