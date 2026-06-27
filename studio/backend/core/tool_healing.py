# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lightweight tool-call XML parsing and stripping helpers.

External inference servers import this module without pulling in the inference
orchestrator, structlog, httpx, or the rest of the studio backend.
"""

import json
import re

# Tool XML stripping patterns. Hyphen in the name class matches dashed MCP names
# (mcp__srv__list-issues). The Gemma marker is anchored to (?:<tool_call|>|\Z)
# (like _TOOL_XML_RE) so an unclosed run strips linearly, not quadratically;
# strip_tool_call_markup uses the quote-aware _strip_gemma_native_spans instead,
# leaving this regex as the streaming fallback.
_TC_JSON_CLOSED_PAT = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_TC_GEMMA_CLOSED_PAT = re.compile(r"<\|tool_call>.*?(?:<tool_call\|>|\Z)", re.DOTALL)
_TC_FUNC_CLOSED_PAT = re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL)
_TOOL_CLOSED_PATS = [
    _TC_JSON_CLOSED_PAT,
    _TC_GEMMA_CLOSED_PAT,
    re.compile(r"<tool_call\|>"),
    _TC_FUNC_CLOSED_PAT,
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*$", re.DOTALL),
]
# A lazy closed-pair pattern rescans to EOF from every opener when its close
# token is absent (O(n^2); O(n^3) re-run per streamed token). Skip the doomed
# sweep when the token is missing -- identical output, no backtracking.
_PAT_REQUIRED_TOKEN = {
    _TC_JSON_CLOSED_PAT: "</tool_call>",
    _TC_FUNC_CLOSED_PAT: "</function>",
}


def strip_tool_patterns(text: str, patterns) -> str:
    """Apply strip ``patterns`` in order, skipping a closed-pair pass whose close
    token is absent (avoids its quadratic no-match rescan)."""
    for pat in patterns:
        token = _PAT_REQUIRED_TOKEN.get(pat)
        if token is not None and token not in text:
            continue
        text = pat.sub("", text)
    return text

# Pre-compiled patterns for tool-call XML parsing.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_GEMMA_START_RE = re.compile(r"<\|tool_call>call:([\w-]+)\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_GEMMA_END_TAG_RE = re.compile(r"<tool_call\|>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w-]+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")
_GEMMA_QUOTE = '<|"|>'
_PARAM_CLOSE_TAG = "</parameter>"
_FUNC_CLOSE_TAG = "</function>"
# A bare Gemma value ends at `}` or a comma starting the next `key:` pair. The
# key must be identifier-shaped, so a comma before digits-then-colon stays in
# the value (`location:New York, NY`; `meet at 10:00, 11:00`).
_GEMMA_NEXT_KEY_RE = re.compile(r"\s*[A-Za-z_][\w-]*\s*:")


def _balanced_brace_end(
    content: str,
    brace_start: int,
    *,
    gemma_quotes: bool = False,
) -> int:
    depth = 0
    i = brace_start
    in_string = False
    in_gemma_string = False
    while i < len(content):
        if gemma_quotes and not in_string and content.startswith(_GEMMA_QUOTE, i):
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


def _balanced_bracket_end(src: str, start: int) -> int:
    """Index of the ``]`` matching the ``[`` at ``start``, or -1. Tracks nested
    ``[]``/``{}`` and double-quoted strings."""
    depth = 0
    i = start
    in_string = False
    while i < len(src):
        ch = src[i]
        if in_string:
            if ch == "\\" and i + 1 < len(src):
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top_level_commas(src: str) -> list:
    """Split on commas that are not inside a nested ``[]``/``{}`` or a string."""
    parts: list[str] = []
    depth = 0
    in_string = False
    start = 0
    i = 0
    while i < len(src):
        ch = src[i]
        if in_string:
            if ch == "\\" and i + 1 < len(src):
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(src[start:i])
            start = i + 1
        i += 1
    parts.append(src[start:])
    return parts


def _quote_gemma_array_elements(body: str) -> str:
    """Normalise a Gemma array value so json.loads succeeds: quote bare string
    elements, recurse into object/nested-array elements, leave quoted strings,
    numbers, and JSON literals as-is. Gemma emits these unquoted
    (``labels:[bug,ui]``, ``items:[{path:a}]``) and the call would otherwise drop."""
    out: list[str] = []
    for element in _split_top_level_commas(body):
        stripped = element.strip()
        if not stripped or stripped[0] == '"':
            out.append(element)
            continue
        if stripped[0] == "{":
            out.append(_quote_gemma_object_keys(stripped))
            continue
        if stripped[0] == "[":
            inner_end = _balanced_bracket_end(stripped, 0)
            if inner_end == len(stripped) - 1:
                out.append("[" + _quote_gemma_array_elements(stripped[1:inner_end]) + "]")
            else:
                out.append(element)
            continue
        try:
            json.loads(stripped)
            out.append(element)
        except (json.JSONDecodeError, ValueError):
            out.append(json.dumps(stripped))
    return ",".join(out)


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
            # Quote bare string values ({unit:celsius}); JSON scalars/objects/
            # arrays/quoted strings stay as-is.
            ws = i
            while i < len(src) and src[i].isspace():
                i += 1
            parts.append(src[ws:i])
            if i < len(src) and src[i] == "[":
                arr_end = _balanced_bracket_end(src, i)
                if arr_end < 0:
                    parts.append(src[i:])
                    i = len(src)
                else:
                    parts.append("[" + _quote_gemma_array_elements(src[i + 1 : arr_end]) + "]")
                    i = arr_end + 1
            elif i < len(src) and src[i] not in '"{':
                v_start = i
                # Bare value: up to `}` or a comma that starts the next key:pair.
                while i < len(src):
                    if src[i] == "}":
                        break
                    if src[i] == "," and _GEMMA_NEXT_KEY_RE.match(src, i + 1):
                        break
                    i += 1
                raw = src[v_start:i]
                try:
                    json.loads(raw.strip())
                    parts.append(raw)
                except (json.JSONDecodeError, ValueError):
                    parts.append(json.dumps(raw.strip()) if raw.strip() else raw)
        else:
            parts.append(src[key_start:i])
    return "".join(parts)


def _gemma_arguments_to_json(args_src: str) -> dict:
    """Parse Gemma 4's native call:name{key:value} argument object."""
    args_src = args_src.strip()
    if not args_src:
        return {}
    src = _normalise_gemma_quoted_strings(args_src)
    src = "{" + src + "}"
    src = _quote_gemma_object_keys(src)
    return json.loads(src)


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


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Parse OpenAI-format tool calls from model text.

    Handles formats like:
      <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
      <|tool_call>call:web_search{query:"..."}<tool_call|>
      <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
    """
    tool_calls: list[dict] = []
    # Collect JSON- and Gemma-format candidates with byte spans, then accept in
    # document order (tools run in returned order). A marker inside an open
    # <function=...><parameter=> value is that parameter's data, so skip it.
    candidates = []  # (start, brace_end, kind, match)
    # Unclosed starts (braces never balance): everything after is the call's
    # argument data through EOF, so the XML fallback must treat it as excluded.
    unclosed_starts = []
    for m in _TC_JSON_START_RE.finditer(content):
        if _inside_open_parameter(content, m.start()):
            continue
        end = _balanced_brace_end(content, m.end() - 1)
        if end >= 0:
            candidates.append((m.start(), end, "json", m))
        else:
            unclosed_starts.append(m.start())
    for m in _TC_GEMMA_START_RE.finditer(content):
        if _inside_open_parameter(content, m.start()):
            continue
        end = _balanced_brace_end(content, m.end() - 1, gemma_quotes = True)
        if end >= 0:
            candidates.append((m.start(), end, "gemma", m))
        else:
            unclosed_starts.append(m.start())
    candidates.sort(key = lambda c: c[0])

    spans = [(s, e) for s, e, _kind, _m in candidates]
    for idx, (start, end, kind, m) in enumerate(candidates):
        # Skip a candidate nested in another candidate's span: it is the outer
        # call's argument data. Checked against every span (not just parsed
        # ones), so a marker in an outer call that fails to parse is never run.
        if any(s <= start and end <= e for j, (s, e) in enumerate(spans) if j != idx):
            continue
        if not allow_incomplete:
            tail = content[end + 1 :].lstrip()
            close_re = _TC_END_TAG_RE if kind == "json" else _TC_GEMMA_END_TAG_RE
            if close_re.match(tail) is None:
                continue
        try:
            if kind == "json":
                obj = json.loads(content[m.end() - 1 : end + 1])
                name = obj.get("name", "")
                arguments = obj.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
            else:
                name = m.group(1)
                arguments = json.dumps(_gemma_arguments_to_json(content[m.end() : end]))
        except (json.JSONDecodeError, ValueError):
            continue
        tool_calls.append(
            {
                "id": f"call_{id_offset + len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    if not tool_calls:
        # Exclude <function=> markers inside any collected JSON/Gemma candidate
        # span (its argument data, even if it failed to parse) and inside any
        # unclosed start through EOF; otherwise nested XML in a malformed call
        # escapes into an executable tool call.
        exclusion_spans = spans + [(s, len(content)) for s in unclosed_starts]
        func_starts = [
            fm
            for fm in _TC_FUNC_START_RE.finditer(content)
            if not _inside_open_parameter(content, fm.start())
            and not any(s <= fm.start() <= e for s, e in exclusion_spans)
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
                close_idx = body.rfind(_FUNC_CLOSE_TAG)
                if close_idx < 0:
                    continue
                body = body[:close_idx]
            else:
                body = _TC_FUNC_CLOSE_RE.sub("", body)

            arguments: dict = {}
            param_starts = list(_TC_PARAM_START_RE.finditer(body))
            if len(param_starts) == 1:
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


def _strip_gemma_native_spans(text: str, *, final: bool) -> str:
    """Remove complete Gemma-native ``<|tool_call>call:NAME{...}<tool_call|>``
    spans, brace/quote-balanced so a literal ``<tool_call|>`` inside a
    ``<|"|>``-quoted argument cannot truncate the span and leak its suffix. An
    incomplete span is dropped to EOF when ``final``, else kept (still streaming).
    """
    out: list[str] = []
    cursor = 0
    for match in _TC_GEMMA_START_RE.finditer(text):
        start = match.start()
        if start < cursor:
            continue
        brace_end = _balanced_brace_end(text, match.end() - 1, gemma_quotes = True)
        if brace_end < 0:
            # Unbalanced: no complete span from here on. Drop the rest if final,
            # else keep it, and stop -- later starts are inside this unclosed run,
            # so rescanning them would re-walk to EOF each time (quadratic).
            if final:
                out.append(text[cursor:start])
                cursor = len(text)
            break
        # Match the close marker via re pos (no remainder copy): streaming
        # re-scans a growing buffer per token, so slicing here is quadratic.
        close_idx = brace_end + 1
        while close_idx < len(text) and text[close_idx].isspace():
            close_idx += 1
        close = _TC_GEMMA_END_TAG_RE.match(text, close_idx)
        if close is None:
            if final:
                out.append(text[cursor:start])
                cursor = len(text)
            continue
        out.append(text[cursor:start])
        cursor = close.end()
    out.append(text[cursor:])
    return "".join(out)


def strip_tool_call_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call XML markup from text.

    When ``final`` is False, only fully closed tool-call blocks are removed.
    When ``final`` is True, trailing incomplete tool-call blocks are removed
    too, and the result is stripped of surrounding whitespace.
    """
    # Well-formed Gemma spans first (quote-aware); the regex patterns below then
    # mop up any malformed Gemma span the helper could not match.
    text = _strip_gemma_native_spans(text, final = final)
    patterns = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    text = strip_tool_patterns(text, patterns)
    return text.strip() if final else text
