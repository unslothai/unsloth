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

import bisect
import json
import re

# One level of nested JSON objects in the bracket-tag strip regexes. Deeper
# nesting may leak partial markup, but the call is still parsed correctly.
_BRACKETED_JSON_ONE_LEVEL = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

# Bare ``name[ARGS]{..}`` rehearsal strip patterns; group 1 captures the name so the
# strip can be gated on the enabled tool list (see ``apply_tool_strip_patterns``). The
# closed form matches a complete one-level body; the tail form matches a trailing /
# truncated rehearsal to end-of-text. The ``(?<!\[CALL_ID\])`` guard keeps the v11
# ``[CALL_ID]<id>[ARGS]`` id token from being taken as a rehearsal name.
_REHEARSAL_CLOSED_STRIP_RE = re.compile(
    r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]\s*" + _BRACKETED_JSON_ONE_LEVEL, re.DOTALL
)
_REHEARSAL_TAIL_STRIP_RE = re.compile(
    r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]\s*(?:\{.*)?$", re.DOTALL
)

# Pre-compiled tool-XML strip patterns. The hyphen in the name char-class lets
# dashed MCP names (mcp__srv__list-issues, issue-number) parse alongside the
# built-ins, including the Mistral [TOOL_CALLS] and rehearsal [ARGS] forms.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
    re.compile(r"<tool_call\|>"),
    re.compile(r"<function=[\w-]+>.*?</function>", re.DOTALL),
    # Aligned with the parser regexes (_MISTRAL_BRACKET_RE / _REHEARSAL_RE):
    # tolerate whitespace after [TOOL_CALLS] and the v11 [CALL_ID]/[ARGS] metadata,
    # and keep the (?<!\[CALL_ID\]) guard so the call-id token is never taken as a
    # rehearsal name.
    re.compile(
        r"\[TOOL_CALLS\]\s*[\w-]+(?:\[CALL_ID\][\w-]+)?(?:\[ARGS\])?\s*"
        + _BRACKETED_JSON_ONE_LEVEL,
        re.DOTALL,
    ),
    _REHEARSAL_CLOSED_STRIP_RE,
    # Orphan closing marker. The Mistral v11 wrapper closes a call with
    # ``[TOOL_CALLS]...[/TOOL_CALLS]``; the balanced scan strips the call body but
    # leaves the bare ``[/TOOL_CALLS]`` behind, so remove it here too (it is never
    # legitimate visible content).
    re.compile(r"\[/TOOL_CALLS\]"),
]
# Trailing-unclosed patterns. The bracket-tag open forms match the bare marker
# (no `{`), so a partial `[TOOL_CALLS]web_search` / `python[ARGS]` streamed before
# its brace is stripped instead of leaking, like `<tool_call>.*$` strips a bare open.
# The rehearsal tail requires a following `{` or end-of-text so prose that merely
# mentions ``foo[ARGS] to the template`` is not truncated as a phantom call.
# Open-ended XML openers (no closing tag): an INCOMPLETE ``<tool_call>`` /
# ``<function=`` call that the parser still executes via ``allow_incomplete``. Each
# runs to end-of-text. Reused by ``_tool_call_markup_spans`` so a literal
# ``<think>`` / ``[THINK]`` inside an UNCLOSED call's arguments is recognised as
# argument data (not a reasoning block) and the open call's markup is stripped
# instead of leaking after the call is parsed.
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

# The two bare ``name[ARGS]{..}`` rehearsal strip patterns (name in group 1). When a
# caller passes ``enabled_tool_names`` these are applied name-gated -- an inactive-name
# example in prose is kept -- so display cleanup matches the parse / detection gate.
# Applied ungated (``enabled_tool_names is None``) they strip every rehearsal span.
_REHEARSAL_STRIP_PATS = frozenset({_REHEARSAL_CLOSED_STRIP_RE, _REHEARSAL_TAIL_STRIP_RE})


def apply_tool_strip_patterns(text: str, patterns, enabled_tool_names=None) -> str:
    """Apply strip ``patterns`` to ``text``. A bare rehearsal ``name[ARGS]{..}`` pattern
    strips only when ``name`` is an enabled tool (or when ``enabled_tool_names`` is
    ``None``); every other pattern is removed unconditionally."""
    for pat in patterns:
        if enabled_tool_names is not None and pat in _REHEARSAL_STRIP_PATS:
            text = pat.sub(
                lambda m: "" if m.group(1) in enabled_tool_names else m.group(0), text
            )
        else:
            text = pat.sub("", text)
    return text

# Pre-compiled patterns for tool-call XML parsing.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_GEMMA_START_RE = re.compile(r"<\|tool_call>call:([\w-]+)\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=([\w-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_GEMMA_END_TAG_RE = re.compile(r"<tool_call\|>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
# Trailing class is horizontal whitespace only so the wrapping newline + the
# value's first-line indentation survive; _trim_param_value trims one newline.
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
_GEMMA_NEXT_KEY_RE = re.compile(r"\s*[A-Za-z_][\w-]*\s*:")

# Spans of thinking blocks: a tool-call candidate STARTING inside one is a
# rehearsal and is skipped (the block is kept, not stripped, so a literal tag in a
# real argument survives). The trailing ``$`` accepts an unclosed block during
# streaming, else a call rehearsed inside an open <think> would fall outside every
# span and could be executed as a real call.
_THINK_TAG_RE = re.compile(r"<think>.*?(?:</think>|$)|\[THINK\].*?(?:\[/THINK\]|$)", re.DOTALL)

# Mistral ``[TOOL_CALLS] [{...}, ...]`` canonical array form: ``[TOOL_CALLS]``
# followed by a JSON list of ``{"name","arguments"}`` objects.
_MISTRAL_ARRAY_RE = re.compile(r"\[TOOL_CALLS\]\s*(?=\[)")

# Mistral name form: ``[TOOL_CALLS]name{json}`` and the v11 shapes
# ``[TOOL_CALLS]name[ARGS]{json}`` / ``[TOOL_CALLS]name[CALL_ID]<id>[ARGS]{json}``.
# The opaque ``[CALL_ID]`` token is metadata, never the function name; the name is
# the token right after ``[TOOL_CALLS]``. Hyphens are kept so dashed MCP names
# (mcp__srv__list-issues) are captured whole.
_MISTRAL_BRACKET_RE = re.compile(
    r"\[TOOL_CALLS\]\s*([\w-]+)(?:\[CALL_ID\][\w-]+)?(?:\[ARGS\])?\s*(?=\{)"
)

# Rehearsal ``name[ARGS]{json}`` prefix (no ``[TOOL_CALLS]``). The lookbehind
# rejects the call-id token in ``[CALL_ID]<id>[ARGS]`` so the id is never taken as
# the function name (that shape is handled by ``_MISTRAL_BRACKET_RE`` instead).
_REHEARSAL_RE = re.compile(r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]\s*(?=\{)")

# Defensive cap: above this many chars, skip the balanced bracket scan and let the
# linear regex catch-all handle stripping (the scan is linear, but this bounds the
# worst case on pathological untrusted output).
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


def _iter_bracket_spans(text: str, start: int = 0, enabled_tool_names=None):
    """Yield ``(span_start, span_end, kind, match)`` for each balanced bracket-tag
    tool call from ``start`` on, in document order. ``kind`` is ``"array"``
    (``[TOOL_CALLS] [..]``), ``"name"`` (``[TOOL_CALLS]name{..}`` incl. the v11
    ``[CALL_ID]`` / ``[ARGS]`` shapes) or ``"rehearsal"`` (``name[ARGS]{..}``).
    ``span_end`` is exclusive.

    ``enabled_tool_names`` (a set, or ``None`` for unrestricted / unknown) gates the
    ambiguous bare ``"rehearsal"`` form only: ``name[ARGS]{..}`` is a genuine call
    ONLY when ``name`` is an enabled tool, so a literal ``foo[ARGS]{"x":1}`` example in
    prose (``foo`` not enabled) is neither parsed nor stripped -- it is not tool markup.
    The explicit ``[TOOL_CALLS]`` markers stay unconditional (they are unambiguous, so
    an inactive name there is still a disabled call, not prose). This keeps the parse,
    strip and streaming-detection paths symmetric on the active-tool gate.

    Balance-based only (does not validate JSON) so the strip path (drop the span)
    and the parse path (``json.loads`` the body) share one scan. Forward scan: the
    cursor jumps past each consumed span, so a marker inside an already-consumed
    JSON string is never re-matched, and each regex is re-searched only once its
    cached match falls behind the cursor -- which keeps the whole scan linear
    instead of re-scanning the tail per match."""
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
            # Unbalanced / truncated body: skip just this marker and keep scanning
            # (a later call may still be complete); the caller's catch-all strips
            # the truncated tail.
            cursor = m.end()
            continue
        if (
            kind == "rehearsal"
            and enabled_tool_names is not None
            and m.group(1) not in enabled_tool_names
        ):
            # Inactive-name ``foo[ARGS]{..}`` is literal prose, not a call. Advance past
            # its balanced body (so a marker inside it is never re-matched) without
            # yielding, so neither parse nor strip treats it as tool markup.
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
                    parts.append("[" + _quote_gemma_array_elements(src[i + 1 : arr_end]) + "]")
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


def _trim_param_value(val: str) -> str:
    """Trim a single wrapping newline the chat template adds around an XML
    parameter value (``<parameter=k>\nVALUE\n</parameter>``) while preserving any
    significant leading indentation / trailing whitespace inside VALUE. Using
    ``str.strip()`` here destroyed the indentation of code/diff arguments; SGLang's
    qwen3_coder detector trims only the wrapping newline."""
    if val.startswith("\n"):
        val = val[1:]
    if val.endswith("\n"):
        val = val[:-1]
    return val


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
    enabled_tool_names=None,
) -> list[dict]:
    """Parse OpenAI-format tool calls from model text.

    Handles formats like:
      <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
      <|tool_call>call:web_search{query:"..."}<tool_call|>
      <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
      [TOOL_CALLS]web_search{"query":"..."}        (Mistral / Devstral fallback)
      web_search[ARGS]{"query":"..."}             (reasoning-model rehearsal)

    A call rehearsed inside a ``<think>`` / ``[THINK]`` block is skipped, not
    executed; the block is kept so a literal tag in a real argument is preserved.
    """
    # A tool-call marker inside a <think>/[THINK] reasoning block is a rehearsal,
    # not a real call, so skip any candidate that STARTS inside one. We no longer
    # delete the blocks from ``content``: a real tool argument may legitimately
    # contain a ``<think>`` / ``[THINK]`` literal, and stripping corrupted it. A
    # think marker that itself opens INSIDE a tool call is argument data, not a
    # reasoning block, so it is excluded (else a real call after it is dropped).
    _think_spans = _think_spans_outside_tool_markup(content)
    _think_starts = [s for s, _e in _think_spans]

    def _in_think(pos: int) -> bool:
        # Think spans are non-overlapping and in document order, so the only span
        # that can contain ``pos`` is the one with the greatest start <= pos.
        # bisect keeps this O(log M) per candidate instead of a linear scan.
        i = bisect.bisect_right(_think_starts, pos) - 1
        return i >= 0 and _think_spans[i][0] <= pos < _think_spans[i][1]

    tool_calls: list[dict] = []
    # Collect JSON- and Gemma-format candidates with their byte spans, then
    # accept them in document order. Both order and spans matter:
    #   * tools execute in returned order, so a call appearing earlier in the
    #     text must be emitted first even across the two formats;
    #   * a tool-call marker INSIDE another call's argument string is data, not a
    #     call, so a candidate starting within an already accepted span is
    #     skipped (covers a JSON marker nested in a Gemma arg and a Gemma marker
    #     nested in a JSON arg alike, regardless of which format is outer).
    candidates = []  # (start, brace_end, kind, match)
    for m in _TC_JSON_START_RE.finditer(content):
        # A marker that begins inside an open <function=...><parameter=...> value
        # is that parameter's data, not its own call; skip it (same guard the
        # XML-style parser below applies to nested <function= markers).
        if _inside_open_parameter(content, m.start()):
            continue
        if _in_think(m.start()):
            continue
        end = _balanced_brace_end(content, m.end() - 1)
        if end >= 0:
            candidates.append((m.start(), end, "json", m))
    for m in _TC_GEMMA_START_RE.finditer(content):
        if _inside_open_parameter(content, m.start()):
            continue
        if _in_think(m.start()):
            continue
        end = _balanced_brace_end(content, m.end() - 1, gemma_quotes = True)
        if end >= 0:
            candidates.append((m.start(), end, "gemma", m))
    candidates.sort(key = lambda c: c[0])

    spans = [(s, e) for s, e, _kind, _m in candidates]
    for idx, (start, end, kind, m) in enumerate(candidates):
        # Skip a candidate nested inside another candidate's brace span: it is
        # the enclosing call's argument data, not its own call. Checked against
        # every candidate span (not only the ones that parsed successfully), so a
        # marker inside an outer call that later fails to normalize is still
        # never promoted to its own executable tool call.
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
        func_starts = [
            fm
            for fm in _TC_FUNC_START_RE.finditer(content)
            if not _inside_open_parameter(content, fm.start()) and not _in_think(fm.start())
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

            tc = {
                "id": f"call_{id_offset + len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments),
                },
            }
            tool_calls.append(tc)

    # Patterns 3 + 4: Mistral bracket-tag and rehearsal forms, walked in document
    # order via one balanced scan: ``[TOOL_CALLS] [..]`` (canonical array),
    # ``[TOOL_CALLS]name{..}`` (incl. the v11 ``[CALL_ID]`` / ``[ARGS]`` shapes) and
    # bare ``name[ARGS]{..}``. Gated behind the XML / JSON / Gemma forms (those are
    # canonical and win), but unified so a Mistral call AND a rehearsal call in one
    # message both parse instead of the second being dropped yet still stripped.
    if not tool_calls:
        for start, end, kind, m in _iter_bracket_spans(
            content, enabled_tool_names = enabled_tool_names
        ):
            if _in_think(start):
                continue
            try:
                payload = json.loads(content[m.end() : end])
            except (json.JSONDecodeError, ValueError):
                continue
            if kind == "array":
                if not isinstance(payload, list):
                    continue
                for item in payload:
                    if not isinstance(item, dict) or "name" not in item:
                        continue
                    args = item.get("arguments", {})
                    if isinstance(args, str):
                        # ``arguments`` may itself be a JSON string (OpenAI spec).
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    tool_calls.append(
                        {
                            "id": f"call_{id_offset + len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": json.dumps(args),
                            },
                        }
                    )
            else:
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

    return tool_calls


def _strip_bracket_tag_calls(text: str, enabled_tool_names=None) -> str:
    """Remove complete ``[TOOL_CALLS] [..]`` arrays and ``[TOOL_CALLS]name{json}`` /
    ``name[ARGS]{json}`` calls with one balanced forward scan, so arbitrarily nested
    JSON args are stripped whole (a fixed-depth regex left two-level args un-stripped
    or ate the trailing prose after them). A truncated tail (bare marker / unbalanced
    JSON) is left for the caller's catch-all patterns. Linear in ``len(text)``.

    ``enabled_tool_names`` gates the bare ``name[ARGS]{..}`` rehearsal form: an
    inactive-name example in prose is kept verbatim (matches the parse / detection
    gate), while ``None`` strips every rehearsal span (unrestricted / unknown)."""
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
    """Spans of tool-call markup. A literal ``<think>`` / ``[THINK]`` inside a call's
    arguments lives within one of these spans and must be stripped WITH the call, not
    preserved as a reasoning block.

    Covers complete (closed) XML/bracket calls AND an INCOMPLETE (unclosed) XML call:
    the parser executes the unclosed call via ``allow_incomplete``, so a literal
    ``<think>`` in its arguments is argument data too -- without the open-ended span
    the call's markup leaks after execution (the segment stripper never sees the open
    tag and the (missing) close together)."""
    spans = [m.span() for pat in _TOOL_CLOSED_PATS for m in pat.finditer(text)]
    spans.extend((start, end) for start, end, _kind, _m in _iter_bracket_spans(text))
    # An unclosed XML opener is only a real incomplete call when it is not already
    # inside a closed/bracket span (a complete call's opener matches the same
    # position but must not extend the span to EOF).
    for pat in _TOOL_OPEN_XML_TAIL_PATS:
        for m in pat.finditer(text):
            if not any(s <= m.start() < e for s, e in spans):
                spans.append(m.span())
    return spans


def _think_spans_outside_tool_markup(text: str) -> list[tuple[int, int]]:
    """``<think>`` / ``[THINK]`` block spans, minus any whose opening marker sits
    INSIDE a tool-call markup span.

    A think tag that opens within a call's arguments is literal argument data, not a
    reasoning block. Keeping it would (a) make the rehearsal-skip in
    ``parse_tool_calls_from_text`` treat a real call appearing after it as rehearsed
    and drop it, and (b) make ``strip_outside_think`` split the call's open/close
    across the preserved block and leak the raw markup. We test the START only: a
    greedy unclosed ``<think>`` match runs to end-of-text and so extends past the
    call, yet it is still that call's argument data."""
    think_spans = [m.span() for m in _THINK_TAG_RE.finditer(text)]
    if not think_spans:
        return think_spans
    call_spans = _tool_call_markup_spans(text)
    if not call_spans:
        return think_spans
    return [(s, e) for (s, e) in think_spans if not any(cs <= s < ce for cs, ce in call_spans)]


def strip_outside_think(text: str, strip_segment) -> str:
    """Apply ``strip_segment(segment, is_last)`` to the visible text around
    ``<think>`` / ``[THINK]`` blocks, preserving the blocks verbatim.

    Tool-looking text inside a reasoning block is the model rehearsing (the parser
    skips it too), so stripping it would corrupt visible reasoning. ``is_last`` is
    True only for the segment after the final block, so a caller can apply its
    trailing-tail patterns only there. Shared by every strip path (this module's
    ``strip_tool_call_markup``, the route display strip, and the GGUF streaming
    strip) so they stay consistent."""
    # A ``<think>`` / ``[THINK]`` literal that OPENS inside a complete tool call is
    # argument text, not reasoning. Excluding it lets the segment stripper see the
    # whole call (the split would otherwise hide the open/close pair and leak the raw
    # call after execution). Tested on the marker start so an unclosed greedy match
    # that runs past the call's closer is still treated as argument data.
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


def _strip_markup_segment(text: str, *, final: bool, enabled_tool_names=None) -> str:
    # Balanced-brace strip for bracket-tag calls first (handles any JSON nesting
    # depth); the regex patterns then cover the XML forms and truncated tails. The
    # rehearsal patterns are name-gated so an inactive-name example stays visible.
    text = _strip_bracket_tag_calls(text, enabled_tool_names = enabled_tool_names)
    patterns = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    return apply_tool_strip_patterns(text, patterns, enabled_tool_names = enabled_tool_names)


def strip_tool_call_markup(text: str, *, final: bool = False, enabled_tool_names=None) -> str:
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
