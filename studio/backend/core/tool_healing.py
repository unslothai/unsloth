# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lightweight tool-call XML parsing and stripping helpers.

External inference servers import this module without pulling in the inference
orchestrator, structlog, httpx, or the rest of the studio backend.
"""

import json
import re

# Tool XML stripping patterns. Hyphen in the name class matches dashed MCP names
# (mcp__srv__list-issues). Both lists strip every closed pair first (JSON, Gemma,
# function), so a closed call is removed as a unit before any to-EOF sweep can
# reach markup nested inside it; the final list then adds greedy .*$ sweeps that
# drop an unclosed opener's remainder to EOF. The non-final list keeps incomplete
# blocks (no EOF sweeps), matching the parser's allow_incomplete = False path.
_TC_JSON_CLOSED_PAT = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_TC_GEMMA_CLOSED_PAT = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL)
_TC_FUNC_CLOSED_PAT = re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL)
_TC_GEMMA_END_PAT = re.compile(r"<tool_call\|>")
_TOOL_CLOSED_PATS = [
    _TC_JSON_CLOSED_PAT,
    _TC_GEMMA_CLOSED_PAT,
    _TC_FUNC_CLOSED_PAT,
    _TC_GEMMA_END_PAT,
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*$", re.DOTALL),
]
# Closed JSON/function blocks, stripped before the quote-aware Gemma helper so a
# Gemma opener sitting in their argument data (e.g. a "<|tool_call>call:t{"
# string) cannot make the helper truncate the whole block, and text after it, to
# EOF. Both are guarded below, so the pre-pass is skipped when absent.
_TOOL_CLOSED_BLOCK_PATS = [_TC_JSON_CLOSED_PAT, _TC_FUNC_CLOSED_PAT]
# A lazy closed-pair pattern rescans to EOF from every opener when its close
# token is absent (O(n^2); O(n^3) re-run per streamed token). Skip the doomed
# sweep when the token is missing -- identical output, no backtracking. The
# to-EOF sweeps are greedy, so they short-circuit on the first opener.
_PAT_REQUIRED_TOKEN = {
    _TC_JSON_CLOSED_PAT: "</tool_call>",
    _TC_GEMMA_CLOSED_PAT: "<tool_call|>",
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


def _marker_coverage(content: str, markers) -> list[tuple[int, int]]:
    """Coverage region ``[start, end]`` for each marker, used to skip markers that
    are another call's data. Each marker's own close is paired to it with a
    per-format stack (a close after the braces pops the nearest still-open marker
    of that format), so an inner call's close is not mistaken for the outer's:

    - braces never balance -> covers ``[start, EOF)`` (the whole ambiguous tail);
    - braces balance and a matching close is found -> covers ``[start, close_end]``
      so a marker smuggled between the braces and that close is treated as data;
    - braces balance but no close is paired -> covers only the brace region, so a
      later sibling after an omitted close marker is still recovered as its own call.
    """
    n = len(content)
    brace_regions = [(s, be) for (s, be, _k, _m) in markers if be >= 0]
    events = []  # (position, order) with order 0 = braces-done, 1 = close marker
    for idx, (_start, brace_end, _kind, _m) in enumerate(markers):
        if brace_end >= 0:
            events.append((brace_end, 0, _kind, idx))
    for kind, close_re in (("json", _TC_END_TAG_RE), ("gemma", _TC_GEMMA_END_TAG_RE)):
        for cm in close_re.finditer(content):
            # A close token inside another call's balanced braces is that call's
            # quoted argument data, not a structural close. Ignore it, else it
            # could pop an earlier close-less marker and extend its coverage over a
            # later valid sibling (dropping that sibling).
            if any(s < cm.start() < be for s, be in brace_regions):
                continue
            events.append((cm.start(), 1, kind, cm.end()))
    events.sort(key = lambda e: (e[0], e[1]))
    waiting = {"json": [], "gemma": []}
    close_end_for: dict[int, int] = {}
    for _pos, order, kind, payload in events:
        if order == 0:
            waiting[kind].append(payload)  # marker index, now awaiting its close
        elif waiting[kind]:
            close_end_for[waiting[kind].pop()] = payload  # innermost open marker closes here
    coverage = []
    for idx, (start, brace_end, _kind, _m) in enumerate(markers):
        if brace_end < 0:
            coverage.append((start, n))
        elif idx in close_end_for:
            coverage.append((start, close_end_for[idx]))
        else:
            coverage.append((start, brace_end))
    return coverage


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
    with_spans: bool = False,
):
    """Parse OpenAI-format tool calls from model text.

    Handles formats like:
      <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
      <|tool_call>call:web_search{query:"..."}<tool_call|>
      <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>

    With ``with_spans=True`` returns ``(tool_calls, spans)`` where ``spans[i]``
    is the half-open ``(start, end)`` byte range of ``tool_calls[i]``'s markup
    in ``content`` (including its close tag when present), so a caller can
    remove exactly the parsed markup and keep every other byte intact.
    """
    tool_calls: list[dict] = []
    call_spans: list[tuple] = []
    # Collect JSON/Gemma markers, then decide nesting by each marker's coverage
    # region (see _marker_coverage): a closed outer covers up to its own close
    # marker, so a JSON/Gemma marker smuggled between the outer braces and that
    # close is treated as data, not executed; an outer that balances but has no
    # close of its own covers only its brace region, so a later sibling after an
    # omitted close is still recovered. A marker inside an open
    # <function=...><parameter=> value is that parameter's data, so skip it.
    markers = []  # (start, brace_end, kind, match); brace_end < 0 = unclosed
    for start_re, gemma, kind in (
        (_TC_JSON_START_RE, False, "json"),
        (_TC_GEMMA_START_RE, True, "gemma"),
    ):
        for m in start_re.finditer(content):
            if _inside_open_parameter(content, m.start()):
                continue
            brace_end = _balanced_brace_end(content, m.end() - 1, gemma_quotes = gemma)
            markers.append((m.start(), brace_end, kind, m))
    markers.sort(key = lambda c: c[0])

    coverage = _marker_coverage(content, markers)
    parsed_items = []  # (start, span_end, name, arguments) in document order
    for idx, (start, brace_end, kind, m) in enumerate(markers):
        # Skip a marker whose start falls in another marker's coverage: it is that
        # outer call's data (its braces, or the gap up to its close), not a call.
        # The end is exclusive: a marker starting exactly where another's close
        # ends is the next sibling (adjacent calls), not nested.
        if any(s <= start < e for j, (s, e) in enumerate(coverage) if j != idx):
            continue
        if brace_end < 0:
            continue  # unclosed: not parseable; the fallback still excludes its XML
        if not allow_incomplete:
            tail = content[brace_end + 1 :].lstrip()
            close_re = _TC_END_TAG_RE if kind == "json" else _TC_GEMMA_END_TAG_RE
            if close_re.match(tail) is None:
                continue
        try:
            if kind == "json":
                obj = json.loads(content[m.end() - 1 : brace_end + 1])
                name = obj.get("name", "")
                arguments = obj.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
            else:
                name = m.group(1)
                arguments = json.dumps(_gemma_arguments_to_json(content[m.end() : brace_end]))
        except (json.JSONDecodeError, ValueError):
            continue
        # Span for with_spans callers: through the close tag when present so the
        # healed strip removes the whole wrapped call, else just the braces.
        span_end = brace_end + 1
        close_re = _TC_END_TAG_RE if kind == "json" else _TC_GEMMA_END_TAG_RE
        ws = len(content[span_end:]) - len(content[span_end:].lstrip())
        close_m = close_re.match(content, span_end + ws)
        if close_m:
            span_end = close_m.end()
        parsed_items.append((start, span_end, name, arguments))

    # Function-XML calls parse alongside marker calls (mixed formats promote in
    # document order -- the #6801 contract). Exclude <function=> markers inside
    # any marker's coverage -- its braces, or the gap up to its close -- even if
    # that call failed to parse, so nested XML cannot escape as a real call. A
    # <function=> after a balanced close-less marker is a sibling (recovered),
    # not swallowed to EOF.
    func_starts = [
        fm
        for fm in _TC_FUNC_START_RE.finditer(content)
        if not _inside_open_parameter(content, fm.start())
        and not any(s <= fm.start() < e for s, e in coverage)
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
        close_idx = body.rfind(_FUNC_CLOSE_TAG)
        if close_idx >= 0:
            span_end = body_start + close_idx + len(_FUNC_CLOSE_TAG)
            body = body[:close_idx]
        elif not allow_incomplete:
            continue
        else:
            body = _TC_FUNC_CLOSE_RE.sub("", body)
            span_end = body_end

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
                    param_starts[pidx + 1].start() if pidx + 1 < len(param_starts) else len(body)
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

        span_start = fm.start()
        wrap_open = re.search(r"<tool_call>\s*$", content[:span_start])
        wrap_close = re.match(r"\s*</tool_call>", content[span_end:])
        if wrap_open and wrap_close:
            span_start = wrap_open.start()
            span_end += wrap_close.end()
        parsed_items.append((span_start, span_end, func_name, json.dumps(arguments)))

    parsed_items.sort(key = lambda item: item[0])
    for start, span_end, name, arguments in parsed_items:
        tool_calls.append(
            {
                "id": f"call_{id_offset + len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        call_spans.append((start, span_end))
    if with_spans:
        return tool_calls, call_spans
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
        # Search for the close marker after the braces (not just immediately
        # after): junk between } and <tool_call|> is malformed-call markup, so
        # strip through the close and keep any text after it. None anywhere after
        # means nothing closes from here on, so stop (keeps this linear).
        close = _TC_GEMMA_END_TAG_RE.search(text, brace_end + 1)
        if close is None:
            if final:
                out.append(text[cursor:start])
                cursor = len(text)
            break
        out.append(text[cursor:start])
        cursor = close.end()
    out.append(text[cursor:])
    return "".join(out)


def strip_tool_markup_final(text: str) -> str:
    """Final display strip, shared by ``strip_tool_call_markup`` and the streaming
    wrappers so all three order the passes the same way. Closed JSON/function
    blocks go first, so a Gemma opener in their argument data cannot make the
    quote-aware helper truncate the block (and text after it) to EOF; then
    well-formed Gemma spans (quote-aware); then the regex sweeps mop up malformed
    spans and drop any unclosed remainder to EOF. Surrounding whitespace is kept.
    """
    text = strip_tool_patterns(text, _TOOL_CLOSED_BLOCK_PATS)
    text = _strip_gemma_native_spans(text, final = True)
    return strip_tool_patterns(text, _TOOL_ALL_PATS)


def strip_tool_call_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call XML markup from text.

    When ``final`` is False, only fully closed tool-call blocks are removed.
    When ``final`` is True, trailing incomplete tool-call blocks are removed
    too, and the result is stripped of surrounding whitespace.
    """
    if final:
        return strip_tool_markup_final(text).strip()
    # Non-final: closed JSON/function blocks first (same reason as the final path),
    # then keep any still-incomplete Gemma span (quote-aware), then the closed
    # patterns mop up the rest without touching incomplete blocks.
    text = strip_tool_patterns(text, _TOOL_CLOSED_BLOCK_PATS)
    text = _strip_gemma_native_spans(text, final = False)
    return strip_tool_patterns(text, _TOOL_CLOSED_PATS)
