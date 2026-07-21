# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Bracket-tag, rehearsal, and thinking-block-strip logic adapted from forge
# (https://github.com/antoinezambelli/forge), Copyright (c) 2025-2026
# Antoine Zambelli, used under the MIT License.

"""Lightweight tool-call parsing and stripping helpers.

External inference servers import this module without pulling in the inference
orchestrator, structlog, httpx, or the rest of the studio backend. Kept in
lockstep with ``core/inference/tool_call_parser.py`` so those servers
(llama-server wrappers, llama-swap, custom shims) reuse the same logic. Any
change here must also land there.

Handles these serializations (see ``parse_tool_calls_from_text``):

* ``<tool_call>{json}</tool_call>``
* ``<|tool_call>call:name{...}<tool_call|>`` (Gemma)
* ``<function=name><parameter=k>v</parameter></function>``
* ``[TOOL_CALLS]name{json}`` (Mistral / Devstral fallback)
* ``name[ARGS]{json}`` (reasoning-model rehearsal)
"""

# PEP 604 annotations must stay import-safe on Python 3.9 (requires-python >=3.9).
from __future__ import annotations

import bisect
import json
import re

# One nesting level in the strip regexes; deeper may leak markup (still parsed).
_BRACKETED_JSON_ONE_LEVEL = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

# Rehearsal ``name[ARGS]{..}`` strips; group 1 = name for tool-list gating. Closed =
# complete body, tail = truncated; ``(?<!\[CALL_ID\])`` keeps the v11 call-id from reading as a name.
_REHEARSAL_CLOSED_STRIP_RE = re.compile(
    r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]\s*" + _BRACKETED_JSON_ONE_LEVEL, re.DOTALL
)
_REHEARSAL_TAIL_STRIP_RE = re.compile(
    r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]\s*(?:\{.*)?$", re.DOTALL
)

# Tool-XML strip patterns; hyphen in the name class covers dashed MCP names.
# Closed-pair patterns are named so _PAT_REQUIRED_TOKEN can skip a doomed lazy rescan when
# the close token is absent: an unguarded ``<tag>.*?</tag>`` rescans to EOF from every opener
# (quadratic on a stream of unclosed openers). Also reused by the quote-aware Gemma pre-pass.
_TC_JSON_CLOSED_PAT = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_TC_GEMMA_CLOSED_PAT = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL)
_TC_FUNC_CLOSED_PAT = re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL)
_TOOL_CLOSED_PATS = [
    _TC_JSON_CLOSED_PAT,
    _TC_GEMMA_CLOSED_PAT,
    re.compile(r"<tool_call\|>"),
    _TC_FUNC_CLOSED_PAT,
    # Mirror the parser regexes: tolerate whitespace and v11 [CALL_ID]/[ARGS] metadata.
    re.compile(
        r"\[TOOL_CALLS\]\s*[\w-]+(?:\[CALL_ID\][\w-]+)?(?:\[ARGS\])?\s*"
        + _BRACKETED_JSON_ONE_LEVEL,
        re.DOTALL,
    ),
    _REHEARSAL_CLOSED_STRIP_RE,
    # Drop the bare v11 [/TOOL_CALLS] closer the balanced scan leaves behind.
    re.compile(r"\[/TOOL_CALLS\]"),
]
# Bare open markers strip a partial call mid-stream; the rehearsal tail needs `{` or EOF
# so prose ``foo[ARGS]`` survives. The XML open-tail forms reach EOF and are reused by
# _tool_call_markup_spans (a think tag in an unclosed call's args stays argument data).
_TOOL_OPEN_XML_TAIL_PATS = [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[\w-]+>.*$", re.DOTALL),
]
_TOOL_ALL_PATS = (
    _TOOL_CLOSED_PATS
    + _TOOL_OPEN_XML_TAIL_PATS
    + [
        re.compile(r"\[TOOL_CALLS\].*$", re.DOTALL),
        _REHEARSAL_TAIL_STRIP_RE,
    ]
)

# Rehearsal strips (name in group 1); name-gated via ``enabled_tool_names``, strip-all when None.
_REHEARSAL_STRIP_PATS = frozenset(
    {_REHEARSAL_CLOSED_STRIP_RE, _REHEARSAL_TAIL_STRIP_RE}
)

# Stripped before the quote-aware Gemma helper so a Gemma opener quoted in argument
# data cannot make the helper truncate the block and its tail.
_TOOL_CLOSED_BLOCK_PATS = [_TC_JSON_CLOSED_PAT, _TC_FUNC_CLOSED_PAT]
# A lazy closed-pair pattern whose close token is absent would rescan to EOF from every
# opener; skip that doomed (quadratic) pass. Shared by both strip helpers.
_PAT_REQUIRED_TOKEN = {
    _TC_JSON_CLOSED_PAT: "</tool_call>",
    _TC_GEMMA_CLOSED_PAT: "<tool_call|>",
    _TC_FUNC_CLOSED_PAT: "</function>",
}


def strip_tool_patterns(text: str, patterns) -> str:
    """Apply ``patterns`` in order, skipping closed-pair passes with no close token."""
    for pat in patterns:
        token = _PAT_REQUIRED_TOKEN.get(pat)
        if token is not None and token not in text:
            continue
        text = pat.sub("", text)
    return text


def apply_tool_strip_patterns(
    text: str,
    patterns,
    enabled_tool_names = None,
) -> str:
    """Apply strip ``patterns`` to ``text``. A bare rehearsal ``name[ARGS]{..}`` pattern
    strips only when ``name`` is an enabled tool (or when ``enabled_tool_names`` is
    ``None``); every other pattern is removed unconditionally. A closed-pair pattern whose
    close token is absent is skipped so an unclosed-marker stream stays linear."""
    for pat in patterns:
        token = _PAT_REQUIRED_TOKEN.get(pat)
        if token is not None and token not in text:
            continue
        if enabled_tool_names is not None and pat in _REHEARSAL_STRIP_PATS:
            text = pat.sub(
                lambda m: "" if m.group(1) in enabled_tool_names else m.group(0), text
            )
        else:
            text = pat.sub("", text)
    return text


# Pre-compiled patterns for tool-call XML parsing.
# <|content_invoke_tool_json|> is TML Inkling's native call marker; its JSON uses
# an ``args`` key and the block closes with <|end_message|>.
_TC_JSON_START_RE = re.compile(r"(?:<tool_call>|<\|content_invoke_tool_json\|>)\s*\{")
_TC_GEMMA_START_RE = re.compile(r"<\|tool_call>\s*call\s*:\s*([\w.\-]+)\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>|<\|end_message\|>")
_TC_GEMMA_END_TAG_RE = re.compile(r"<tool_call\|>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
# Horizontal-whitespace trailing class keeps the wrapping newline; _trim_param_value trims it.
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w-]+)>[^\S\n]*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")
_GEMMA_QUOTE = '<|"|>'
_PARAM_CLOSE_TAG = "</parameter>"
_FUNC_CLOSE_TAG = "</function>"
# A bare (unquoted) Gemma value ends at `}` or at a comma that begins the next
# `key:` pair. A comma NOT followed by a key token is part of the value (e.g.
# `location:New York, NY`), so it must not terminate the value. The key token
# must be identifier-shaped (start with a letter or underscore); a comma
# followed by digits-then-colon is value text such as a timestamp or ratio
# (`meet at 10:00, 11:00 tomorrow`), not a new key.
_GEMMA_NEXT_KEY_RE = re.compile(r"\s*[A-Za-z_][\w.\-]*\s*:")

# A candidate starting inside a think block is a rehearsal (block kept so literal tags in
# real args survive); ``$`` accepts an unclosed block mid-stream.
_THINK_TAG_RE = re.compile(
    r"<think>.*?(?:</think>|$)|\[THINK\].*?(?:\[/THINK\]|$)", re.DOTALL
)
# Bare open/close markers for prefilled-reasoning turns (template opens <think> in the prompt).
_THINK_OPEN_RE = re.compile(r"<think>|\[THINK\]")
_THINK_CLOSE_RE = re.compile(r"</think>|\[/THINK\]")

# Mistral canonical array: [TOOL_CALLS] + JSON list of {"name","arguments"} objects.
_MISTRAL_ARRAY_RE = re.compile(r"\[TOOL_CALLS\]\s*(?=\[)")

# Mistral name form + v11 [ARGS]/[CALL_ID] shapes; [CALL_ID] is metadata, not the name,
# and hyphens keep dashed MCP names whole.
_MISTRAL_BRACKET_RE = re.compile(
    r"\[TOOL_CALLS\]\s*([\w-]+)(?:\[CALL_ID\][\w-]+)?(?:\[ARGS\])?\s*(?=\{)"
)

# Rehearsal ``name[ARGS]{json}`` (no [TOOL_CALLS]); the lookbehind keeps the v11 call-id
# from being taken as the function name.
_REHEARSAL_RE = re.compile(r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]\s*(?=\{)")

# Above this size skip the balanced scan; the linear regex catch-all bounds pathological output.
_MAX_BRACKET_SCAN_CHARS = 1_000_000


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


def _decode_array_items(text: str, body_start: int, body_end: int):
    """Return ``(objs, ends)`` for each top-level element of the JSON array between
    ``body_start`` (at or before its ``[``) and ``body_end`` (exclusive): the decoded
    object and its absolute exclusive end offset.

    Decoding element-by-element with ``raw_decode`` tolerates the comma-less object
    separators the repo's own Mistral/Ollama multi-call templates emit
    (``[{...}{...}]``; see ollama_template_mappers.py). A single ``json.loads`` of the
    whole body rejects that form and would drop every call. The ends also tile the
    region across the calls' spans so a with_spans consumer strips each exactly once."""
    decoder = json.JSONDecoder()
    objs: list = []
    ends: list[int] = []
    i = text.find("[", body_start)
    if i < 0:
        return objs, ends
    i += 1
    while i < body_end:
        while i < body_end and text[i] in " \t\r\n,":
            i += 1
        if i >= body_end or text[i] == "]":
            break
        try:
            obj, rel = decoder.raw_decode(text[i:body_end])
        except (json.JSONDecodeError, ValueError):
            break
        i += rel
        objs.append(obj)
        ends.append(i)
    return objs, ends


def _iter_bracket_spans(
    text: str,
    start: int = 0,
    enabled_tool_names = None,
):
    """Yield ``(span_start, span_end, kind, match)`` for each balanced bracket-tag
    call from ``start`` on, in document order; ``span_end`` exclusive. ``kind`` is
    ``"array"`` ([TOOL_CALLS] [..]), ``"name"`` ([TOOL_CALLS]name{..}, incl. v11
    [CALL_ID]/[ARGS]) or ``"rehearsal"`` (name[ARGS]{..}).

    ``enabled_tool_names`` (set, or None = unrestricted) gates only the ambiguous
    bare rehearsal form: name[ARGS]{..} is a call ONLY when ``name`` is enabled, so a
    prose ``foo[ARGS]{..}`` (foo disabled) is neither parsed nor stripped. Explicit
    [TOOL_CALLS] markers stay unconditional, keeping parse/strip/detection symmetric.

    Balance-only (no JSON validation) so strip and parse share one scan. The cursor
    jumps past each consumed span, so a marker inside consumed JSON is never
    re-matched and each regex re-searches only once its match falls behind: linear."""
    n = len(text)
    specs = (
        ("array", _MISTRAL_ARRAY_RE),
        ("name", _MISTRAL_BRACKET_RE),
        ("rehearsal", _REHEARSAL_RE),
    )
    nexts = {kind: rx.search(text, start) for kind, rx in specs}
    cursor = start
    while cursor < n:
        for kind, rx in specs:
            m = nexts[kind]
            if m is not None and m.start() < cursor:
                nexts[kind] = rx.search(text, cursor)
        live = [(kind, m) for kind, m in nexts.items() if m is not None]
        if not live:
            return
        kind, m = min(live, key = lambda km: km[1].start())
        if kind == "array":
            end = _balanced_bracket_end(text, m.end())
            end = None if end < 0 else end
        else:
            end = _balanced_json_span(text, m.end())
        if end is None:
            # Truncated body: skip and keep scanning; the caller's catch-all strips the tail.
            cursor = m.end()
            continue
        if (
            kind == "rehearsal"
            and enabled_tool_names is not None
            and m.group(1) not in enabled_tool_names
        ):
            # Inactive-name rehearsal is prose: advance past its body without yielding.
            cursor = end + 1
            continue
        yield (m.start(), end + 1, kind, m)
        cursor = end + 1


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
    """Normalise the elements of a Gemma array value so json.loads succeeds.

    Gemma may emit ``labels:[bug,ui]`` without per-element quotes, or arrays of
    objects (``items:[{path:a}]``) whose keys/values also lack quotes; left
    as-is json.loads fails and the whole call is dropped. Bare string elements
    are quoted, object and nested-array elements are normalised recursively, and
    quoted strings (already normalised from ``<|"|>``), numbers, and JSON
    literals are preserved."""
    out: list[str] = []
    for element in _split_top_level_commas(body):
        stripped = element.strip()
        if not stripped or stripped[0] == '"':
            out.append(element)
            continue
        if stripped[0] == "{":
            # Object element: quote its keys/bare values like a top-level object.
            out.append(_quote_gemma_object_keys(stripped))
            continue
        if stripped[0] == "[":
            # Nested array: normalise its elements too.
            inner_end = _balanced_bracket_end(stripped, 0)
            if inner_end == len(stripped) - 1:
                out.append(
                    "[" + _quote_gemma_array_elements(stripped[1:inner_end]) + "]"
                )
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
        while i < len(src) and (src[i].isalnum() or src[i] in "_-."):
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
            # Gemma may emit bare string values ({unit:celsius}); quote them so
            # json.loads succeeds. JSON scalars/objects/arrays/quoted stay as-is.
            ws = i
            while i < len(src) and src[i].isspace():
                i += 1
            parts.append(src[ws:i])
            if i < len(src) and src[i] == "[":
                # Array value: quote bare string elements (e.g. labels:[bug,ui])
                # so json.loads succeeds instead of dropping the call.
                arr_end = _balanced_bracket_end(src, i)
                if arr_end < 0:
                    parts.append(src[i:])
                    i = len(src)
                else:
                    parts.append(
                        "[" + _quote_gemma_array_elements(src[i + 1 : arr_end]) + "]"
                    )
                    i = arr_end + 1
            elif i < len(src) and src[i] not in '"{':
                v_start = i
                # Consume the bare value up to `}` or a comma that starts the
                # next key:value pair; a comma inside the value (e.g.
                # `New York, NY`) does not terminate it.
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
                    # Quote bare value; empty ({k:}) becomes "" so json.loads sees {"k":""} not invalid {"k":}.
                    parts.append(json.dumps(raw.strip()))
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
    # The parameter's OWN close tag decides: if it closes after ``pos`` the position is
    # argument data (even across literal function closes); an unclosed one falls back to func close.
    own_close = content.find(_PARAM_CLOSE_TAG, last_param_start)
    if own_close >= 0:
        return own_close > pos
    func_close = content.find(_FUNC_CLOSE_TAG, last_param_start)
    return func_close < 0 or pos < func_close


def _func_close_index(content: str, body_start: int, body: str) -> int:
    """Index in ``body`` of the first ``</function>`` that is not argument
    data (not inside an open parameter value); -1 when every close is data.
    Taking the LAST close swallowed prose between the real close and a
    literal ``</function>`` mentioned later in the answer."""
    idx = body.find(_FUNC_CLOSE_TAG)
    while idx >= 0:
        if not _inside_open_parameter(content, body_start + idx):
            return idx
        idx = body.find(_FUNC_CLOSE_TAG, idx + 1)
    return -1


def _trim_param_value(val: str) -> str:
    """Trim the single wrapping newline the chat template adds around an XML
    parameter value, preserving indentation inside VALUE (``str.strip()`` destroyed
    code/diff argument indentation)."""
    if val.startswith("\n"):
        val = val[1:]
    if val.endswith("\n"):
        val = val[:-1]
    return val


def _marker_coverage(content: str, markers) -> list[tuple[int, int]]:
    """Coverage ``[start, end]`` per marker, used to skip markers that are another
    call's data. Closes pair to markers via a per-format stack so an inner close
    is not mistaken for the outer's. Unbalanced braces cover to EOF; balanced with
    a paired close cover through it (markers before the close are data); balanced
    without one cover only the braces, so a later sibling is still recovered."""
    n = len(content)
    brace_regions = [(s, be) for (s, be, _k, _m) in markers if be >= 0]
    events = []  # (position, order) with order 0 = braces-done, 1 = close marker
    for idx, (_start, brace_end, _kind, _m) in enumerate(markers):
        if brace_end >= 0:
            events.append((brace_end, 0, _kind, idx))
    for kind, close_re in (("json", _TC_END_TAG_RE), ("gemma", _TC_GEMMA_END_TAG_RE)):
        for cm in close_re.finditer(content):
            # A close inside another call's balanced braces is quoted data; it
            # must not pop an earlier close-less marker and swallow a sibling.
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
            close_end_for[waiting[kind].pop()] = (
                payload  # innermost open marker closes here
            )
    coverage = []
    for idx, (start, brace_end, _kind, _m) in enumerate(markers):
        if brace_end < 0:
            coverage.append((start, n))
        elif idx in close_end_for:
            coverage.append((start, close_end_for[idx]))
        else:
            coverage.append((start, brace_end))
    return coverage


def _build_markers(content: str):
    """JSON/Gemma tool markers as ``(start, brace_end, kind, match)`` in document
    order; ``brace_end < 0`` marks an unbalanced (to-EOF) open."""
    markers = []
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
    return markers


def marker_coverage(content: str) -> list[tuple[int, int]]:
    """Coverage spans of JSON/Gemma tool markers so other parsers can treat markup
    inside a marker's coverage (even a marker that failed to parse) as that call's
    data rather than a sibling call."""
    return _marker_coverage(content, _build_markers(content))


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
    enabled_tool_names = None,
    with_spans: bool = False,
):
    """Parse OpenAI-format tool calls from model text.

    Handles formats like:
      <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
      <|tool_call>call:web_search{query:"..."}<tool_call|>
      <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
      [TOOL_CALLS]web_search{"query":"..."}        (Mistral / Devstral fallback)
      web_search[ARGS]{"query":"..."}             (reasoning-model rehearsal)

    A call rehearsed inside a ``<think>`` / ``[THINK]`` block is skipped, not
    executed; the block is kept so a literal tag in a real argument is preserved.

    With ``with_spans=True`` returns ``(tool_calls, spans)`` where ``spans[i]``
    is the half-open ``(start, end)`` byte range of ``tool_calls[i]``'s markup
    in ``content`` (including its close tag when present), so a caller can
    remove exactly the parsed markup and keep every other byte intact.
    """
    # Candidates starting inside a think block are rehearsals, skipped; blocks are kept, and a
    # think marker opening inside a call is argument data (excluded from spans).
    _think_spans = _think_spans_outside_tool_markup(content)
    _think_starts = [s for s, _e in _think_spans]

    def _in_think(pos: int) -> bool:
        # Spans are ordered and non-overlapping; bisect gives O(log M) per candidate.
        i = bisect.bisect_right(_think_starts, pos) - 1
        return i >= 0 and _think_spans[i][0] <= pos < _think_spans[i][1]

    tool_calls: list[dict] = []
    call_spans: list[tuple] = []
    # Collect JSON/Gemma markers; _marker_coverage decides nesting so a marker inside
    # another call's coverage (even one that failed to parse) is data, not executed. A
    # marker opening inside a think block is a rehearsal and is skipped.
    parsed_items = []  # (start, span_end, name, arguments) in document order
    markers = [mk for mk in _build_markers(content) if not _in_think(mk[0])]
    coverage = _marker_coverage(content, markers)
    for idx, (start, brace_end, kind, m) in enumerate(markers):
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
                # Accept ``parameters`` alias for ``arguments`` (Llama-3.2 drift inside
                # Hermes) and ``args`` (TML Inkling native calls).
                arguments = obj.get("arguments")
                if arguments is None:
                    arguments = obj.get("parameters")
                if arguments is None:
                    arguments = obj.get("args", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                # Inkling echoes the bare tool name (and a role opener) before the
                # marker: <|message_model|>NAME<|content_invoke_tool_json|>{...}.
                # Fold that echo into the markup span so promotion removes it too.
                if name and content.startswith("<|content_invoke_tool_json|>", start):
                    pre = content[:start]
                    if pre.endswith(name):
                        start -= len(name)
                        if content[:start].endswith("<|message_model|>"):
                            start -= len("<|message_model|>")
            else:
                name = m.group(1)
                arguments = json.dumps(
                    _gemma_arguments_to_json(content[m.end() : brace_end])
                )
        except (json.JSONDecodeError, ValueError):
            continue
        span_end = brace_end + 1
        close_re = _TC_END_TAG_RE if kind == "json" else _TC_GEMMA_END_TAG_RE
        ws = len(content[span_end:]) - len(content[span_end:].lstrip())
        close_m = close_re.match(content, span_end + ws)
        if close_m:
            span_end = close_m.end()
        parsed_items.append((start, span_end, name, arguments))

    func_starts = [
        fm
        for fm in _TC_FUNC_START_RE.finditer(content)
        if not _inside_open_parameter(content, fm.start())
        and not _in_think(fm.start())
        and not any(s <= fm.start() < e for s, e in coverage)
    ]
    for idx, fm in enumerate(func_starts):
        func_name = fm.group(1)
        body_start = fm.end()
        next_func = (
            func_starts[idx + 1].start() if idx + 1 < len(func_starts) else len(content)
        )
        end_tag = _TC_END_TAG_RE.search(content[body_start:])
        if end_tag:
            body_end = body_start + end_tag.start()
        else:
            body_end = len(content)
        body_end = min(body_end, next_func)
        body = content[body_start:body_end]
        close_idx = _func_close_index(content, body_start, body)
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
            arguments[pm.group(1)] = _trim_param_value(val)
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
                arguments[param_name] = _trim_param_value(val)
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

    # Patterns 3+4: Mistral [TOOL_CALLS] and bare rehearsal via one balanced scan in document
    # order, so a Mistral call and a rehearsal in one message both parse.
    if not tool_calls:
        for start, end, kind, m in _iter_bracket_spans(
            content, enabled_tool_names = enabled_tool_names
        ):
            if _in_think(start):
                continue
            # Extend the region over an immediately-following v11 closer so with_spans consumers strip it too.
            closer = re.match(r"\s*\[/TOOL_CALLS\]", content[end:])
            region_end = end + closer.end() if closer else end
            if kind == "array":
                # Decode elements individually (comma-tolerant): one json.loads of the whole
                # body rejects the comma-less multi-call arrays Mistral/Ollama templates emit.
                payload, item_ends = _decode_array_items(content, m.end(), end)
                if not payload:
                    continue
                # Tile the region so every byte belongs to exactly one span; a with_spans consumer
                # keeps skipped bytes visible and strips promoted markup exactly once.
                tile_start = start
                last_span_idx = -1
                for item_idx, item in enumerate(payload):
                    if not isinstance(item, dict) or "name" not in item:
                        continue
                    args = item.get("arguments", {})
                    if isinstance(args, str):
                        # ``arguments`` may itself be a JSON string (OpenAI spec).
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    if not isinstance(args, (dict, str)):
                        # ``"arguments": null`` (or any non-object scalar) becomes {} like the
                        # <tool_call> path, not the string "null" auto-heal would mangle to
                        # a bogus {"query":"null"}.
                        args = {}
                    tool_calls.append(
                        {
                            "id": f"call_{id_offset + len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                # A bare scalar string stays raw (like the <tool_call> path);
                                # json.dumps would double-encode it so the arg healer wraps
                                # "weather" with its literal quotes.
                                "arguments": args
                                if isinstance(args, str)
                                else json.dumps(args),
                            },
                        }
                    )
                    item_end = (
                        item_ends[item_idx] if item_idx < len(item_ends) else region_end
                    )
                    last_span_idx = len(call_spans)
                    call_spans.append((tile_start, item_end))
                    tile_start = item_end
                if last_span_idx >= 0:
                    tile_start, _tile_end = call_spans[last_span_idx]
                    call_spans[last_span_idx] = (tile_start, region_end)
            else:
                try:
                    payload = json.loads(content[m.end() : end])
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(payload, dict):
                    continue
                tool_calls.append(
                    {
                        "id": f"call_{id_offset + len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": m.group(1),
                            "arguments": json.dumps(payload),
                        },
                    }
                )
                call_spans.append((start, region_end))

    if with_spans:
        return tool_calls, call_spans
    return tool_calls


def _strip_bracket_tag_calls(text: str, enabled_tool_names = None) -> str:
    """Strip complete [TOOL_CALLS] arrays / name / bare name[ARGS]{..} calls with one
    balanced forward scan, so nested JSON args are removed whole (a fixed-depth regex
    left two-level args behind). Truncated tails go to the caller's catch-all. Linear.
    ``enabled_tool_names`` gates the rehearsal form (inactive-name prose kept; None
    strips every span)."""
    if len(text) > _MAX_BRACKET_SCAN_CHARS:
        return text
    out: list[str] = []
    cursor = 0
    for start, end, _kind, _m in _iter_bracket_spans(
        text, enabled_tool_names = enabled_tool_names
    ):
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _tool_call_markup_spans(text: str) -> list[tuple[int, int]]:
    """Spans of tool-call markup, so a literal <think>/[THINK] inside a call's args is
    stripped WITH the call, not kept as a reasoning block. Covers closed XML/bracket
    calls and an unclosed XML call (run via allow_incomplete); without the open-ended
    span the unclosed call's markup would leak after execution."""
    # Skip a lazy closed-pair pattern whose close token is absent: its finditer would rescan
    # to EOF from every opener (quadratic on a stream of unclosed openers).
    spans = [
        m.span()
        for pat in _TOOL_CLOSED_PATS
        if (_PAT_REQUIRED_TOKEN.get(pat) is None or _PAT_REQUIRED_TOKEN[pat] in text)
        for m in pat.finditer(text)
    ]
    spans.extend((start, end) for start, end, _kind, _m in _iter_bracket_spans(text))
    # An unclosed opener is a real incomplete call only outside closed/bracket spans.
    for pat in _TOOL_OPEN_XML_TAIL_PATS:
        for m in pat.finditer(text):
            if not any(s <= m.start() < e for s, e in spans):
                spans.append(m.span())
    return spans


def _think_spans_outside_tool_markup(text: str) -> list[tuple[int, int]]:
    """<think>/[THINK] block spans, minus any whose opening marker sits INSIDE a
    tool-call span (that tag is argument data, not reasoning). Keeping it would drop a
    real call after it as rehearsed and leak the call's markup. START tested only, so
    a greedy unclosed <think> past the call is still that call's argument data."""
    think_spans = [m.span() for m in _THINK_TAG_RE.finditer(text)]
    call_spans = _tool_call_markup_spans(text)
    # Prefilled reasoning: the template opens <think> in the prompt, so add a leading span
    # (0..close) to skip calls rehearsed there; guarded so a stray close in a normal answer is safe.
    close = _THINK_CLOSE_RE.search(text)
    if close is not None:
        opener = _THINK_OPEN_RE.search(text)
        if (
            (opener is None or close.start() < opener.start())
            and not any(cs <= close.start() < ce for cs, ce in call_spans)
            and any(cs >= close.end() for cs, ce in call_spans)
        ):
            think_spans = [(0, close.end())] + think_spans
    if not think_spans:
        return think_spans
    if not call_spans:
        return think_spans
    return [
        (s, e)
        for (s, e) in think_spans
        if not any(cs <= s < ce for cs, ce in call_spans)
    ]


def strip_outside_think(text: str, strip_segment) -> str:
    """Apply ``strip_segment(segment, is_last)`` to visible text around <think>/[THINK]
    blocks, preserving the blocks verbatim (tool-looking text inside is rehearsal).
    ``is_last`` is True only after the final block, so trailing-tail patterns apply
    only there. Shared by every strip path so they stay consistent."""
    # A think marker opening inside a complete call is argument text; excluding it lets the
    # stripper see the whole call. START-tested, so an unclosed match stays argument data.
    think_spans = _think_spans_outside_tool_markup(text)
    if not think_spans:
        return strip_segment(text, True)
    pieces: list[str] = []
    prev = 0
    for s, e in think_spans:
        pieces.append(strip_segment(text[prev:s], False))
        pieces.append(text[s:e])
        prev = e
    pieces.append(strip_segment(text[prev:], True))
    return "".join(pieces)


def _strip_gemma_native_spans(text: str, *, final: bool) -> str:
    """Remove complete Gemma-native spans, brace/quote-balanced so a literal
    ``<tool_call|>`` in a quoted argument cannot truncate the span. An incomplete
    span is dropped to EOF when ``final``, else kept (still streaming)."""
    out: list[str] = []
    cursor = 0
    for match in _TC_GEMMA_START_RE.finditer(text):
        start = match.start()
        if start < cursor:
            continue
        brace_end = _balanced_brace_end(text, match.end() - 1, gemma_quotes = True)
        if brace_end < 0:
            # Unbalanced: nothing completes from here on. Drop the rest if final,
            # else keep it; stop either way (rescanning would be quadratic).
            if final:
                out.append(text[cursor:start])
                cursor = len(text)
            break
        # Junk between } and <tool_call|> is malformed-call markup: strip through
        # the close, keep text after it. No close anywhere means stop (linear).
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


def _gemma_span_ranges(text: str) -> list:
    """``(start, end)`` of each complete Gemma-native span; same walk as
    ``_strip_gemma_native_spans`` without stripping."""
    ranges: list[tuple] = []
    cursor = 0
    for match in _TC_GEMMA_START_RE.finditer(text):
        start = match.start()
        if start < cursor:
            continue
        brace_end = _balanced_brace_end(text, match.end() - 1, gemma_quotes = True)
        if brace_end < 0:
            break
        close = _TC_GEMMA_END_TAG_RE.search(text, brace_end + 1)
        if close is None:
            break
        ranges.append((start, close.end()))
        cursor = close.end()
    return ranges


def _strip_closed_blocks_outside_gemma(text: str) -> str:
    """Closed JSON/function pre-pass that skips matches starting inside a complete
    Gemma span: deleting across the span boundary would mangle the Gemma close and
    truncate the tail. A skipped match resumes at the covering span's end, so a
    real function-XML call after the span is still stripped."""
    ranges = _gemma_span_ranges(text)
    if not ranges:
        return strip_tool_patterns(text, _TOOL_CLOSED_BLOCK_PATS)
    for pat in _TOOL_CLOSED_BLOCK_PATS:
        token = _PAT_REQUIRED_TOKEN.get(pat)
        if token is not None and token not in text:
            continue
        out: list[str] = []
        pos = 0
        while True:
            m = pat.search(text, pos)
            if m is None:
                out.append(text[pos:])
                break
            covering = next((r for r in ranges if r[0] <= m.start() < r[1]), None)
            if covering is not None:
                out.append(text[pos : covering[1]])
                pos = covering[1]
                continue
            out.append(text[pos : m.start()])
            pos = m.end()
        new_text = "".join(out)
        if new_text != text:
            text = new_text
            ranges = _gemma_span_ranges(text)
    return text


def _strip_markup_segment(
    text: str,
    *,
    final: bool,
    enabled_tool_names = None,
) -> str:
    # Bracket-tag calls (Mistral/rehearsal) first via balanced scan (any nesting depth,
    # rehearsal name-gated); then the quote-aware Gemma-native passes so a literal
    # <tool_call|> in an argument cannot truncate a block; finally the regex XML/tail sweeps.
    text = _strip_bracket_tag_calls(text, enabled_tool_names = enabled_tool_names)
    text = _strip_closed_blocks_outside_gemma(text)
    text = _strip_gemma_native_spans(text, final = final)
    patterns = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    return apply_tool_strip_patterns(
        text, patterns, enabled_tool_names = enabled_tool_names
    )


def strip_tool_call_markup(
    text: str,
    *,
    final: bool = False,
    enabled_tool_names = None,
) -> str:
    """Strip tool-call XML markup from text.

    When ``final`` is False, only fully closed tool-call blocks are removed.
    When ``final`` is True, trailing incomplete tool-call blocks are removed
    too, and the result is stripped of surrounding whitespace.

    ``<think>`` / ``[THINK]`` reasoning is preserved verbatim (see
    ``strip_outside_think``); the trailing-tail patterns apply only after the
    last block. ``enabled_tool_names`` keeps an inactive-name ``foo[ARGS]{..}``
    example visible (it is prose, not a call) so display cleanup matches detection.
    """
    result = strip_outside_think(
        text,
        lambda seg, is_last: _strip_markup_segment(
            seg, final = final and is_last, enabled_tool_names = enabled_tool_names
        ),
    )
    return result.strip() if final else result
