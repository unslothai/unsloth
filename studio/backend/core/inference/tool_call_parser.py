# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Backend-neutral tool-call parser shared by GGUF, safetensors, and MLX, so the
safetensors + MLX agentic loop sees the same call shape llama-server gives GGUF:

  - ``<tool_call>{json}</tool_call>``           (Qwen / Hermes)
  - ``<function=name><parameter=k>v</parameter></function>``  (Qwen3.5 xml)
  - ``<|python_tag|>NAME.call(k="v", ...)``     (Llama-3 built-in tools)
  - ``<|python_tag|>{"name":..., "parameters":...}``  (Llama-3 custom)
  - ``{"name":..., "parameters":...}``          (Llama-3.2 bare JSON)
  - ``[TOOL_CALLS] [{...}, ...]``               (Mistral v0.3 / Nemo / Small)
  - ``[TOOL_CALLS]name{json}``                  (Mistral v11+ / Magistral)
  - ``[TOOL_CALLS]name[ARGS]{json}``            (Ministral / Mistral Large 3)
  - ``<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>``  (Gemma 4)

Missing closing tags / brackets are tolerated: models often truncate mid-stream.
"""

# Keeps PEP 604 `X | None` lazy for python 3.9 (imported standalone by external servers).
from __future__ import annotations

import json
import re
from typing import Any, Optional

# Shared parser handles Qwen/Hermes, Qwen3.5 XML, Gemma 4; this module adds Llama-3, Mistral, bare JSON.
from core import tool_healing as _tool_healing


# Flip the streaming buffer STREAMING->DRAINING so partial markup never leaks.
TOOL_XML_SIGNALS = (
    "<tool_call>",
    "<function=",
    '<function name="',
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<|tool_call>",
)


# Closed pairs only (mid-stream); _TOOL_ALL_PATS eats unclosed tails at end-of-turn.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    # Match to the real ``</function>`` (lookahead, not greedy ``.*``) so a literal
    # ``</function>`` in a value doesn't truncate and each call stays separate.
    re.compile(
        r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>'
        r'(?:(?!<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>).)*'
        r"</function>",
        re.DOTALL,
    ),
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>.*$', re.DOTALL),
    # Bare-word markers drop a trailing truncated call only when the next chars look like
    # a call start, so prose mentioning the marker is kept; a marker at end-of-text drops.
    re.compile(r"<\|tool_call>(?=\s*call\s*:|\s*$).*$", re.DOTALL),
    re.compile(
        r"\[TOOL_CALLS\](?=\s*(?:[\[{]|[A-Za-z_][\w.\-]*[\[{])|\s*$).*$",
        re.DOTALL,
    ),
    re.compile(
        r"<\|python_tag\|>(?=\s*(?:\{|[A-Za-z_][\w.]*\()|\s*$).*$",
        re.DOTALL,
    ),
]


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


# Qwen / Hermes ``<tool_call>{json}``.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
# Qwen3.5 ``<function=name>`` plus attribute form ``<function name="name">`` (MiniCPM-5,
# MiniMax-M2); name in group(1) or group(2).
_TC_FUNC_START_RE = re.compile(r'<function(?:=([\w\.\-]+)|\s+name="([\w\.\-]+)")>\s*')
# Body ends at ``</tool_call>`` or ``</function>`` so trailing prose stays out of args.
_TC_END_TAG_RE = re.compile(r"</(?:tool_call|function)>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
# Horizontal whitespace only so the wrapping newline + indent survive (``_trim_param_value``
# trims one newline), preserving code indent.
_TC_PARAM_START_RE = re.compile(
    r'<(?:parameter|param)(?:=([\w\.\-]+)|\s+name="([\w\.\-]+)")>[^\S\n]*'
)
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</(?:parameter|param)>\s*$")

# Llama-3 ``<|python_tag|>NAME.call(...)``.
_LLAMA3_PYTHON_TAG = "<|python_tag|>"
_LLAMA3_PY_CALL_RE = re.compile(
    r"<\|python_tag\|>\s*([\w\.\-]+)\s*\.\s*call\s*\(",
)
# Anchored at the char after ``<|python_tag|>`` plus the ``; NAME.call(`` chain sep, so
# a ``.call(`` inside JSON args is ignored.
_LLAMA3_PY_CALL_HEAD_RE = re.compile(r"\s*([\w\.\-]+)\s*\.\s*call\s*\(")
_LLAMA3_CALL_CHAIN_RE = re.compile(r"\s*;\s*([\w\.\-]+)\s*\.\s*call\s*\(")
# ``.call(k=v)`` kwarg tokens, hand-scanned below (not finditer) to stay linear on a
# truncated body (ReDoS).
_LLAMA3_KEY_RE = re.compile(r"\w+")
_LLAMA3_WS_RE = re.compile(r"\s*")
# ints, decimals, sci notation; trailing ``(?![\w.])`` stops ``1.2.3`` truncating to ``1.2``.
_LLAMA3_NUM_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])")
_LLAMA3_LIT_RE = re.compile(r"true|false|null")

# Mistral ``[TOOL_CALLS]`` trigger. v11+ chains ``name{json}`` (Magistral) or
# ``name[ARGS]{json}`` (Ministral / Large 3).
_MISTRAL_TRIGGER = "[TOOL_CALLS]"
_MISTRAL_ARGS_MARKER = "[ARGS]"
# Mistral Small 3.2 emits ``name[CALL_ID]<id>[ARGS]{json}`` (absent on Ministral / Magistral).
_MISTRAL_CALL_ID_MARKER = "[CALL_ID]"
# Magistral wraps reasoning in ``[THINK]...[/THINK]``; a ``[TOOL_CALLS]`` inside is not a real call.
_MISTRAL_THINK_OPEN = "[THINK]"
_MISTRAL_THINK_CLOSE = "[/THINK]"
_MISTRAL_V11_NAME_RE = re.compile(r"\s*([\w\.\-]+)\s*")

# Gemma 4: ``<|tool_call>call:NAME{...}<tool_call|>``, ``<|"|>`` wraps strings.
_GEMMA_TC_RE = re.compile(r"<\|tool_call>\s*call\s*:\s*([\w\.\-]+)\s*\{")
_GEMMA_STR_BEGIN = '<|"|>'
_GEMMA_STR_END = '<|"|>'
_GEMMA_TC_END = "<tool_call|>"


def _balanced_bracket_end(text: str, start: int) -> int | None:
    """Index of the ``]`` matching ``[`` at ``text[start]`` (ignores brackets in JSON strings)."""
    if start >= len(text) or text[start] != "[":
        return None
    depth = 0
    in_string = False
    esc = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def _skip_mistral_call_id(text: str, pos: int) -> int:
    """Skip an optional ``[CALL_ID]<id>`` (Mistral Small 3.2); return the next token pos."""
    n = len(text)
    i = pos
    while i < n and text[i] in " \t\n\r":
        i += 1
    if not text.startswith(_MISTRAL_CALL_ID_MARKER, i):
        return pos
    i += len(_MISTRAL_CALL_ID_MARKER)
    while i < n and text[i] in " \t\n\r":
        i += 1
    # The id is a short opaque token; stop at whitespace or the next marker.
    while i < n and text[i] not in " \t\n\r[{":
        i += 1
    while i < n and text[i] in " \t\n\r":
        i += 1
    return i


def _strip_mistral_reasoning(content: str) -> str:
    """Drop a leading Magistral ``[THINK]`` block so rehearsed calls inside reasoning are not promoted; unclosed drops to EOF."""
    i = 0
    n = len(content)
    while i < n and content[i] in " \t\n\r":
        i += 1
    if not content.startswith(_MISTRAL_THINK_OPEN, i):
        return content
    close = content.find(_MISTRAL_THINK_CLOSE, i + len(_MISTRAL_THINK_OPEN))
    if close == -1:
        return content[:i]
    return content[:i] + content[close + len(_MISTRAL_THINK_CLOSE) :]


def _strip_mistral_closed_calls(text: str) -> str:
    """Strip cleanly-closed ``[TOOL_CALLS]`` blocks via balanced scanning (a non-greedy regex would truncate nested JSON); unclosed runs wait for ``final=True``."""
    n = len(text)
    out = []
    cursor = 0
    while cursor < n:
        idx = text.find(_MISTRAL_TRIGGER, cursor)
        if idx == -1:
            out.append(text[cursor:])
            break
        out.append(text[cursor:idx])
        body_start = idx + len(_MISTRAL_TRIGGER)
        i = body_start
        while i < n and text[i] in " \t\n\r":
            i += 1
        # Array shape: ``[TOOL_CALLS] [...]``.
        if i < n and text[i] == "[":
            end = _balanced_bracket_end(text, i)
            if end is None:
                # Truncated; let caller buffer / final-strip.
                out.append(text[idx:])
                break
            cursor = end + 1
            if text.startswith("</s>", cursor):
                cursor += len("</s>")
            continue
        # Single-object shape ``[TOOL_CALLS] { json }``: the parser accepts it, so strip it too.
        if i < n and text[i] == "{":
            end = _balanced_brace_end(text, i)
            if end is None:
                out.append(text[idx:])
                break
            cursor = end + 1
            if text.startswith("</s>", cursor):
                cursor += len("</s>")
            continue
        # Named shape: ``[TOOL_CALLS] name [ARGS]? { json }``.
        name_match = _MISTRAL_V11_NAME_RE.match(text, i)
        if not name_match:
            out.append(text[idx:body_start])
            cursor = body_start
            continue
        i = name_match.end()
        while i < n and text[i] in " \t\n\r":
            i += 1
        i = _skip_mistral_call_id(text, i)
        if text.startswith(_MISTRAL_ARGS_MARKER, i):
            i += len(_MISTRAL_ARGS_MARKER)
            while i < n and text[i] in " \t\n\r":
                i += 1
        if i >= n or text[i] != "{":
            out.append(text[idx:i])
            cursor = i
            continue
        end = _balanced_brace_end(text, i)
        if end is None:
            out.append(text[idx:])
            break
        cursor = end + 1
        # Consume the optional EOS marker so ``...{json}</s>`` doesn't leave ``</s>`` as content.
        if text.startswith("</s>", cursor):
            cursor += len("</s>")
    return "".join(out)


_FUNC_CLOSE_TAG_RE = re.compile(r"</function>")


def _strip_function_xml_calls(text: str, *, final: bool) -> str:
    """Strip ``<function=...>`` calls by mirroring the parser: an opener inside an open ``<parameter>`` is data and each call closes at its first ``</function>`` that is not parameter data; ``final`` drops a trailing unclosed call."""
    starts = [
        m for m in _TC_FUNC_START_RE.finditer(text) if not _inside_open_parameter(text, m.start())
    ]
    if not starts:
        return text
    out: list[str] = []
    pos = 0
    for idx, m in enumerate(starts):
        if m.start() < pos:
            continue  # opener already inside a previously consumed call span
        out.append(text[pos : m.start()])
        next_start = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        close = None
        for cm in _FUNC_CLOSE_TAG_RE.finditer(text, m.end(), next_start):
            if not _inside_open_parameter(text, cm.start()):
                close = cm  # first close that is not parameter data = the real close
                break
        if close is not None:
            pos = close.end()
        elif final:
            pos = len(text)  # trailing unclosed call -- drop to EOF
        else:
            out.append(text[m.start() :])  # keep the unclosed call buffered mid-stream
            pos = len(text)
            break
    out.append(text[pos:])
    return "".join(out)


def strip_tool_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call markup; ``final=True`` also drops trailing unclosed runs and trims."""
    if final:
        # End-of-turn only: drop a leading Magistral ``[THINK]...[/THINK]`` block (bracket form,
        # not the ``<think>`` reasoning channel) so raw reasoning doesn't leak into display/history.
        text = _strip_mistral_reasoning(text)
    text = _strip_mistral_closed_calls(text)
    # Scan-strip the function-XML form first (parser-accurate: a literal ``<function=...>`` in
    # a value is data, not a call); the regex arms below cover the other formats.
    text = _strip_function_xml_calls(text, final = final)
    pats = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in pats:
        text = pat.sub("", text)
    return text.strip() if final else text


def has_tool_signal(text: str) -> bool:
    return any(s in text for s in TOOL_XML_SIGNALS)


def _mistral_region_end(text: str, idx: int) -> int | None:
    """Exclusive end of the balanced ``[TOOL_CALLS]`` call at ``idx``, or ``None`` when truncated (array, object, and named forms)."""
    n = len(text)
    i = idx + len(_MISTRAL_TRIGGER)
    while i < n and text[i] in " \t\n\r":
        i += 1
    if i < n and text[i] == "[":
        end = _balanced_bracket_end(text, i)
        return None if end is None else end + 1
    if i < n and text[i] == "{":
        end = _balanced_brace_end(text, i)
        return None if end is None else end + 1
    name_match = _MISTRAL_V11_NAME_RE.match(text, i)
    if not name_match:
        return None
    i = name_match.end()
    while i < n and text[i] in " \t\n\r":
        i += 1
    i = _skip_mistral_call_id(text, i)
    if text.startswith(_MISTRAL_ARGS_MARKER, i):
        i += len(_MISTRAL_ARGS_MARKER)
        while i < n and text[i] in " \t\n\r":
            i += 1
    if i >= n or text[i] != "{":
        return None
    end = _balanced_brace_end(text, i)
    return None if end is None else end + 1


def _xml_signal_inside_leading_mistral(content: str) -> bool:
    """True when a parseable Mistral call is the first tool emission in document order: it owns the turn, so later XML (quoted in its arguments or in trailing prose) is not promoted over it. A signal BEFORE the trigger keeps normal order."""
    trig = content.find(_MISTRAL_TRIGGER)
    if trig < 0:
        return False
    first_xml = _first_foreign_tool_signal(content)
    if first_xml is not None and first_xml < trig:
        return False
    # Only plain prose precedes the trigger (preamble-tolerant); prose merely mentioning
    # the marker has no parseable region and keeps the normal order.
    return _mistral_region_end(content, trig) is not None


_ATTR_FUNC_OPEN_RE = re.compile(r'<function\s+name="')


def _first_foreign_tool_signal(content: str) -> int | None:
    """Offset of the first signal a non-envelope parser would fire on (XML forms plus the Llama-3 ``<|python_tag|>`` marker)."""
    first = None
    for sig in ("<tool_call>", "<|tool_call>", "<function=", "<|python_tag|>"):
        p = content.find(sig)
        if p >= 0 and (first is None or p < first):
            first = p
    attr = _ATTR_FUNC_OPEN_RE.search(content)
    if attr is not None and (first is None or attr.start() < first):
        first = attr.start()
    return first


def _xml_signal_inside_leading_bare_json(content: str) -> bool:
    """True when the first foreign signal sits inside a LEADING bare-JSON call's balanced body: quoted argument data, so the bare-JSON parser takes the outer call first."""
    i = 0
    n = len(content)
    while i < n and content[i] in " \t\n\r":
        i += 1
    if i >= n or content[i] != "{":
        return False
    end = _balanced_brace_end(content, i)
    if end is None:
        return False
    if _top_level_bare_json_name(content[i : end + 1]) is None:
        # Not a call object, but a nameless object that parses as real JSON is an envelope
        # too (markup in its strings is data); non-JSON braced prose keeps the old behaviour.
        try:
            json.loads(content[i : end + 1])
        except ValueError:
            return False
    first_xml = _first_foreign_tool_signal(content)
    # The Mistral trigger is foreign to a JSON envelope too, so fold it into first_xml.
    trig = content.find(_MISTRAL_TRIGGER)
    if trig >= 0 and (first_xml is None or trig < first_xml):
        first_xml = trig
    # Inside the balanced body the signal is quoted argument data, so the leading call owns
    # the turn; a non-call object takes the decline path (dropped, only the tail parsed).
    return first_xml is not None and i < first_xml


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
    enabled_tool_names: Optional[set] = None,
) -> list[dict]:
    """Return OpenAI-format tool calls, first-match wins. ``allow_incomplete`` heals truncated calls (``False`` = strict closed-only); ``enabled_tool_names`` gates the markerless bare-JSON form."""
    # Drop Magistral reasoning before any dispatch so a rehearsed call inside
    # [THINK]...[/THINK] is not promoted; keeps the parse path aligned with the display strip.
    content = _strip_mistral_reasoning(content)

    # A leading bare-JSON value is decided FIRST so markup quoted in its arguments stays
    # data. Must precede the Mistral guard, whose preamble tolerance would else claim a
    # trigger quoted inside the leading object.
    if _xml_signal_inside_leading_bare_json(content):
        calls = _parse_llama3_bare_json(
            content, id_offset = id_offset, enabled_tool_names = enabled_tool_names
        )
        if calls:
            return calls
        # Disabled/example name: the leading object is ordinary content. Drop it and parse
        # only the tail -- a real call after it still parses, nothing inside it is promoted.
        i = 0
        while i < len(content) and content[i] in " \t\n\r":
            i += 1
        end = _balanced_brace_end(content, i)  # guard guarantees a balanced object
        return parse_tool_calls_from_text(
            content[end + 1 :],
            id_offset = id_offset,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )

    # A [TOOL_CALLS] call that is the first tool emission owns the turn: XML quoted in its
    # arguments or in trailing prose is not promoted, and a plain-prose preface keeps it.
    if _xml_signal_inside_leading_mistral(content):
        calls = _parse_mistral_tool_calls(
            content, id_offset = id_offset, allow_incomplete = allow_incomplete
        )
        if calls:
            return calls

    # A leading MiniCPM/MiniMax ``<function name="...">`` call owns the turn: tool_healing
    # does not know the wrapper, so gate it here. A signal before the opener keeps normal order.
    attr = _ATTR_FUNC_OPEN_RE.search(content)
    if attr is not None:
        first_other = None
        for sig in (
            "<tool_call>",
            "<|tool_call>",
            "<function=",
            "<|python_tag|>",
            _MISTRAL_TRIGGER,
        ):
            p = content.find(sig)
            if p >= 0 and (first_other is None or p < first_other):
                first_other = p
        if first_other is None or attr.start() < first_other:
            calls = _parse_function_xml(
                content, id_offset = id_offset, allow_incomplete = allow_incomplete
            )
            if calls:
                return calls

    # A leading Llama-3 ``<|python_tag|>`` call owns the turn like the others: markup quoted
    # in a ``.call(...)`` argument is not promoted. tool_healing does not know the tag, so
    # gate it here. A foreign signal before the tag keeps normal order.
    py_tag = content.find(_LLAMA3_PYTHON_TAG)
    if py_tag >= 0:
        first_other = None
        for sig in ("<tool_call>", "<|tool_call>", "<function=", _MISTRAL_TRIGGER):
            p = content.find(sig)
            if p >= 0 and (first_other is None or p < first_other):
                first_other = p
        attr = _ATTR_FUNC_OPEN_RE.search(content)
        if attr is not None and (first_other is None or attr.start() < first_other):
            first_other = attr.start()
        if first_other is None or py_tag < first_other:
            calls = _parse_llama3_python_tag(
                content, id_offset = id_offset, allow_incomplete = allow_incomplete
            )
            if calls:
                return calls

    # Qwen/Hermes, Qwen3.5 XML, and Gemma 4 use the shared tool_healing parser (the
    # strict/Auto-Heal + nested-marker + ``<|"|>`` handling GGUF relies on).
    calls = _tool_healing.parse_tool_calls_from_text(
        content,
        id_offset = id_offset,
        allow_incomplete = allow_incomplete,
    )
    if calls:
        return calls

    # Formats tool_healing does not cover: ``<function name="...">`` (MiniCPM-5 / MiniMax-M2),
    # Llama-3 and Mistral. Run only after tool_healing found nothing, so a strict-rejected
    # call is never re-healed here.
    for parser in (
        _parse_function_xml,  # <function name="..."> attribute form
        _parse_llama3_python_tag,  # Llama-3 <|python_tag|>
        _parse_mistral_tool_calls,  # Mistral [TOOL_CALLS]
    ):
        calls = parser(content, id_offset = id_offset, allow_incomplete = allow_incomplete)
        if calls:
            return calls

    # Llama-3.2 bare ``{"name":..., "parameters":...}``. Strict (starts with ``{``
    # and parses to the right shape) so plain prose stays untouched.
    return _parse_llama3_bare_json(
        content, id_offset = id_offset, enabled_tool_names = enabled_tool_names
    )


def _parse_tool_call_json(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    out: list[dict] = []
    for m in _TC_JSON_START_RE.finditer(content):
        brace_start = m.end() - 1
        end = _balanced_brace_end(content, brace_start)
        if end is None:
            continue
        # Strict mode: a balanced body that never closed its ``<tool_call>`` is truncated
        # (trailing prose after the close is still tolerated).
        if not allow_incomplete and not content[end + 1 :].lstrip().startswith("</tool_call>"):
            continue
        try:
            obj = json.loads(content[brace_start : end + 1])
        except (json.JSONDecodeError, ValueError):
            continue
        name = obj.get("name", "")
        # Accept both ``arguments`` (Hermes/Qwen) and ``parameters`` (Llama-3 drift).
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters", {})
        if isinstance(args, dict):
            args_str = json.dumps(args)
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = json.dumps({"value": args})
        if not name:
            continue
        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    return out


def _trim_param_value(val: str) -> str:
    """Trim only the template's wrapping newline around an XML parameter value; ``str.strip()`` destroyed code/diff indentation."""
    if val.startswith("\n"):
        val = val[1:]
    if val.endswith("\n"):
        val = val[:-1]
    return val


def _inside_open_parameter(text: str, pos: int) -> bool:
    """True if ``pos`` is inside an unclosed ``<parameter>`` block, i.e. the opener at ``pos`` is literal argument data, not a nested call."""
    last_param_open = -1
    for m in _TC_PARAM_START_RE.finditer(text, 0, pos):
        last_param_open = m.start()
    if last_param_open < 0:
        return False
    # The parameter's OWN close tag decides: if it closes after ``pos`` the position is
    # argument data (even across literal ``</function>``); an unclosed one falls back to func close.
    own_closes = [
        c
        for c in (
            text.find("</parameter>", last_param_open),
            text.find("</param>", last_param_open),
        )
        if c >= 0
    ]
    if own_closes:
        return min(own_closes) > pos
    func_closes = [
        c
        for c in (
            text.find("</function>", last_param_open),
            text.find("</tool_call>", last_param_open),
        )
        if c >= 0
    ]
    return not func_closes or pos < min(func_closes)


def _parse_function_xml(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    out: list[dict] = []
    # Skip ``<function ...>`` openers that are literals inside an open parameter value,
    # else the nested marker becomes a second call and truncates the real argument.
    func_starts = [
        fm
        for fm in _TC_FUNC_START_RE.finditer(content)
        if not _inside_open_parameter(content, fm.start())
    ]
    for idx, fm in enumerate(func_starts):
        # group(1) is ``<function=name>``, group(2) is ``<function name="...">``.
        func_name = fm.group(1) or fm.group(2)
        body_start = fm.end()
        next_func = func_starts[idx + 1].start() if idx + 1 < len(func_starts) else len(content)
        # The call ends at the FIRST </function> / </tool_call> not inside an open parameter:
        # a literal close in an argument is skipped as data, prose after the real close is not
        # folded in (mirrors _strip_function_xml_calls).
        close_match = None
        for cm in _TC_END_TAG_RE.finditer(content, body_start, next_func):
            if not _inside_open_parameter(content, cm.start()):
                close_match = cm
                break
        has_close = close_match is not None
        if has_close:
            body_end = close_match.start()
        else:
            body_end = min(len(content), next_func)
        # Strict mode: a call that never reached its close is truncated; do not heal it.
        if not allow_incomplete and not has_close:
            continue
        body = _TC_FUNC_CLOSE_RE.sub("", content[body_start:body_end])

        args: dict = {}
        param_unclosed = False
        # Same nested-literal guard: a ``<parameter>`` opener inside an open value is literal text.
        param_starts = [
            pm
            for pm in _TC_PARAM_START_RE.finditer(body)
            if not _inside_open_parameter(body, pm.start())
        ]
        if len(param_starts) == 1:
            pm = param_starts[0]
            raw_val = body[pm.end() :]
            if not _TC_PARAM_CLOSE_RE.search(raw_val):
                param_unclosed = True
            val = _TC_PARAM_CLOSE_RE.sub("", raw_val)
            args[pm.group(1) or pm.group(2)] = _trim_param_value(val)
        else:
            for pidx, pm in enumerate(param_starts):
                val_start = pm.end()
                next_param = (
                    param_starts[pidx + 1].start() if pidx + 1 < len(param_starts) else len(body)
                )
                raw_val = body[val_start:next_param]
                if not _TC_PARAM_CLOSE_RE.search(raw_val):
                    param_unclosed = True
                val = _TC_PARAM_CLOSE_RE.sub("", raw_val)
                args[pm.group(1) or pm.group(2)] = _trim_param_value(val)

        # Strict mode: every parameter must close; a dangling one means the call was cut off.
        # A closed call with no parameters is a valid zero-argument call, so keep it.
        if not allow_incomplete and param_unclosed:
            continue

        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": func_name, "arguments": json.dumps(args)},
            }
        )
    return out


def _llama3_kv_value(body: str, p: int, n: int) -> tuple[Any, int | None]:
    """One ``.call`` value at ``body[p:]``; returns ``(value, len)`` or ``(None, None)``."""
    if p >= n:
        return None, None
    if body[p] == '"':
        # ``"((?:\\.|[^"\\])*)"`` by hand so an unterminated quote is O(n), not O(n^2).
        j = p + 1
        while j < n:
            c = body[j]
            if c == "\\":
                # ``\\.`` needs a following non-newline char; else the body can't match.
                if j + 1 >= n or body[j + 1] == "\n":
                    return None, None
                j += 2
                continue
            if c == '"':
                raw = body[p + 1 : j]
                # json.loads keeps \n/\uXXXX escapes and literal UTF-8 (emoji/CJK) intact.
                try:
                    return json.loads('"' + raw + '"'), j + 1 - p
                except (json.JSONDecodeError, ValueError):
                    return raw, j + 1 - p
            j += 1
        return None, None  # unterminated
    nm = _LLAMA3_NUM_RE.match(body, p)
    if nm:
        v = nm.group(0)
        # Sci notation and decimals decode as float; a bare integer stays int.
        return (float(v) if any(c in v for c in ".eE") else int(v)), nm.end() - p
    lm = _LLAMA3_LIT_RE.match(body, p)
    if lm:
        return {"true": True, "false": False, "null": None}[lm.group(0)], lm.end() - p
    return None, None


def _parse_llama3_kv_args(body: str) -> dict[str, Any]:
    """Left-to-right ``k=v`` kwargs from a ``.call(...)`` body (linear scan; later keys win)."""
    args: dict[str, Any] = {}
    n = len(body)
    i = 0
    while i < n:
        km = _LLAMA3_KEY_RE.match(body, i)
        if km is None:
            i += 1
            continue
        p = _LLAMA3_WS_RE.match(body, km.end()).end()
        if p >= n or body[p] != "=":
            i = km.end()
            continue
        p = _LLAMA3_WS_RE.match(body, p + 1).end()
        val, length = _llama3_kv_value(body, p, n)
        if length is None:
            i = km.end()
            continue
        args[km.group(0)] = val
        i = p + length
    return args


def _parse_llama3_python_tag(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Parse Llama-3 ``<|python_tag|>`` emissions: ``NAME.call(...)``, bare JSON, ``; `` multi-call, ``parameters``/``arguments`` keys."""
    out: list[dict] = []
    if _LLAMA3_PYTHON_TAG not in content:
        return out

    # 1. ``NAME.call(...)`` built-in form, anchored to ``<|python_tag|>`` (optionally
    #    ``; ``-chained) so a ``.call(...)`` inside a JSON string argument isn't mistaken for one.
    pos = content.find(_LLAMA3_PYTHON_TAG)
    truncated = False
    while pos >= 0 and not truncated:
        head = _LLAMA3_PY_CALL_HEAD_RE.match(content, pos + len(_LLAMA3_PYTHON_TAG))
        if head is None:
            # Tag is the custom JSON form (``{...}``) or noise -- leave it to step 2.
            break
        name = head.group(1)
        open_idx = head.end()
        i = open_idx
        while True:
            i = open_idx
            depth = 1
            in_string = False
            esc = False
            while i < len(content) and depth > 0:
                ch = content[i]
                if in_string:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            break
                i += 1
            # Truncated ``.call(...)`` (no closing paren): reject in strict mode.
            if not allow_incomplete and depth > 0:
                truncated = True
                break
            body = content[open_idx:i]
            out.append(
                {
                    "id": f"call_{id_offset + len(out)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(_parse_llama3_kv_args(body)),
                    },
                }
            )
            # ``)`` then optional ``; NAME.call(`` chains the next built-in call.
            chain = _LLAMA3_CALL_CHAIN_RE.match(content, i + 1)
            if chain is None:
                break
            name = chain.group(1)
            open_idx = chain.end()
        # Past the consumed region: a second ``<|python_tag|>`` may carry more calls.
        pos = content.find(_LLAMA3_PYTHON_TAG, i + 1)

    # 2. ``<|python_tag|>{"name":.., "parameters":..}``; raw_decode peels ``; ``-separated objects.
    if not out:
        decoder = json.JSONDecoder()
        idx = content.find(_LLAMA3_PYTHON_TAG)
        while idx >= 0:
            search_from = idx + len(_LLAMA3_PYTHON_TAG)
            cursor = search_from
            while cursor < len(content):
                brace = content.find("{", cursor)
                if brace < 0:
                    break
                # Stop at the next ``<|python_tag|>``.
                next_tag = content.find(_LLAMA3_PYTHON_TAG, search_from, brace)
                if next_tag >= 0:
                    break
                try:
                    obj, end_offset = decoder.raw_decode(content[brace:])
                except (json.JSONDecodeError, ValueError):
                    cursor = brace + 1
                    continue
                if not isinstance(obj, dict):
                    cursor = brace + end_offset
                    continue
                name = obj.get("name") or obj.get("function") or ""
                args = obj.get("parameters") if "parameters" in obj else obj.get("arguments", {})
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                elif isinstance(args, str):
                    args_str = args
                else:
                    args_str = json.dumps({"value": args})
                if name:
                    out.append(
                        {
                            "id": f"call_{id_offset + len(out)}",
                            "type": "function",
                            "function": {"name": name, "arguments": args_str},
                        }
                    )
                cursor = brace + end_offset
            idx = content.find(_LLAMA3_PYTHON_TAG, cursor)
    return out


# Llama-3 special-token sentinels (chainable, any order) plus the header role label.
_LLAMA3_BARE_JSON_SENTINELS = (
    "<|begin_of_text|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
)
_LLAMA3_HEADER_ROLES = ("assistant", "user", "system", "tool", "ipython")


def strip_llama3_leading_sentinels(content: str) -> str:
    """Strip leading Llama-3 sentinels leaked from a prior turn; shared by the parser and the streaming guards."""
    stripped = content.lstrip()
    while True:
        stripped = stripped.lstrip()
        matched = False
        for sentinel in _LLAMA3_BARE_JSON_SENTINELS:
            if stripped.startswith(sentinel):
                stripped = stripped[len(sentinel) :]
                if sentinel == "<|start_header_id|>":
                    for role in _LLAMA3_HEADER_ROLES:
                        if stripped.startswith(role):
                            stripped = stripped[len(role) :]
                            break
                matched = True
                break
        if not matched:
            return stripped


def _parse_llama3_bare_json(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
    enabled_tool_names: Optional[set] = None,
) -> list[dict]:
    """Llama-3.2 bare ``{"name":.., "parameters":..}`` (strict). ``enabled_tool_names`` keeps ordinary JSON answers from being misread; ``None`` is name-agnostic."""
    out: list[dict] = []
    stripped = strip_llama3_leading_sentinels(content)
    if not stripped.startswith("{"):
        return out

    decoder = json.JSONDecoder()
    cursor = 0
    n = len(stripped)
    while cursor < n:
        # Skip whitespace and the Llama-3 ``;`` inter-call separator.
        while cursor < n and stripped[cursor] in " \t\n\r;":
            cursor += 1
        if cursor >= n or stripped[cursor] != "{":
            break
        try:
            obj, end_offset = decoder.raw_decode(stripped[cursor:])
        except (json.JSONDecodeError, ValueError):
            break
        if not isinstance(obj, dict):
            break
        name = obj.get("name") or obj.get("function") or ""
        if not isinstance(name, str) or not name:
            break
        # Markerless JSON is ambiguous: only a call when the name is an enabled tool.
        if enabled_tool_names is not None and name not in enabled_tool_names:
            break
        # ``parameters`` must be a dict (Llama-3 spec); ``arguments`` may be a dict or a
        # JSON-string of one (OpenAI).
        if "parameters" in obj:
            args = obj.get("parameters")
            if not isinstance(args, dict):
                break
            args_str = json.dumps(args)
        elif "arguments" in obj:
            args = obj.get("arguments")
            if isinstance(args, dict):
                args_str = json.dumps(args)
            elif isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    break
                if not isinstance(parsed, dict):
                    break
                args_str = args
            else:
                break
        else:
            break
        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
        cursor += end_offset
    return out


def _parse_mistral_tool_calls(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Parse Mistral ``[TOOL_CALLS]`` emissions: pre-v11 array/object and v11+ named forms."""
    out: list[dict] = []
    content = _strip_mistral_reasoning(content)
    idx = content.find(_MISTRAL_TRIGGER)
    if idx < 0:
        return out

    # Disambiguate the first occurrence: array / single object (pre-v11) or bare-name (v11+).
    j = idx + len(_MISTRAL_TRIGGER)
    k = j
    while k < len(content) and content[k] in " \t\n\r":
        k += 1
    if k >= len(content):
        return out

    if content[k] == "[":
        return _parse_mistral_array(content, k, id_offset, allow_incomplete = allow_incomplete)

    if content[k] == "{":
        # Pre-v11 single ``{"name":...}``; fall through to v11+ if it carries no ``name``.
        end = _balanced_brace_end(content, k)
        if end is not None:
            try:
                obj = json.loads(content[k : end + 1])
                if isinstance(obj, dict) and obj.get("name"):
                    _consume_mistral_call(content[k : end + 1], out, id_offset)
                    return out
            except (json.JSONDecodeError, ValueError):
                pass

    # v11+: walk every ``[TOOL_CALLS]``, parsing ``name{json}`` or ``name[ARGS]{json}``.
    pos = idx
    while pos >= 0:
        cur = pos + len(_MISTRAL_TRIGGER)
        nm = _MISTRAL_V11_NAME_RE.match(content, cur)
        if not nm:
            pos = content.find(_MISTRAL_TRIGGER, cur)
            continue
        name = nm.group(1)
        after_name = nm.end()
        after_name = _skip_mistral_call_id(content, after_name)
        if content.startswith(_MISTRAL_ARGS_MARKER, after_name):
            after_name += len(_MISTRAL_ARGS_MARKER)
        while after_name < len(content) and content[after_name] in " \t\n\r":
            after_name += 1
        if after_name >= len(content) or content[after_name] != "{":
            pos = content.find(_MISTRAL_TRIGGER, cur)
            continue
        end = _balanced_brace_end(content, after_name)
        if end is None:
            break
        try:
            args = json.loads(content[after_name : end + 1])
        except (json.JSONDecodeError, ValueError):
            pos = content.find(_MISTRAL_TRIGGER, end + 1)
            continue
        if not isinstance(args, dict):
            pos = content.find(_MISTRAL_TRIGGER, end + 1)
            continue
        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
        )
        pos = content.find(_MISTRAL_TRIGGER, end + 1)
    return out


def _parse_mistral_array(
    content: str,
    start: int,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Pre-v11 ``[TOOL_CALLS] [{...}, ...]`` array form."""
    out: list[dict] = []
    j = start
    depth = 0
    in_string = False
    esc = False
    while j < len(content):
        ch = content[j]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    # An unclosed array (no matching ]) is truncated; reject in strict mode.
    if not allow_incomplete and depth != 0:
        return out
    body = content[start : j + 1] if depth == 0 else content[start:]

    try:
        arr = json.loads(body)
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict):
                    _consume_mistral_call(json.dumps(obj), out, id_offset)
        return out
    except (json.JSONDecodeError, ValueError):
        if not allow_incomplete:
            return out

    # Healing path for unclosed arrays: walk top-level objects, advancing past each
    # balanced ``{...}`` (re-scanning from every ``{`` would be quadratic ReDoS).
    pos = 0
    blen = len(body)
    while pos < blen:
        brace = body.find("{", pos)
        if brace < 0:
            break
        end = _balanced_brace_end(body, brace)
        if end is None:
            break  # truncated mid-object: nothing after it can balance
        _consume_mistral_call(body[brace : end + 1], out, id_offset)
        pos = end + 1
    return out


def _consume_mistral_call(obj_text: str, out: list[dict], id_offset: int) -> None:
    try:
        obj = json.loads(obj_text)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(obj, dict):
        return
    name = obj.get("name") or ""
    # Mistral uses ``arguments``; accept the ``parameters`` alias too.
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters", {})
    if isinstance(args, dict):
        args_str = json.dumps(args)
    elif isinstance(args, str):
        args_str = args
    else:
        args_str = json.dumps({"value": args})
    if name:
        out.append(
            {
                "id": obj.get("id") or f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )


def _parse_gemma_tool_calls(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Gemma 4: ``<|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>``."""
    out: list[dict] = []
    for m in _GEMMA_TC_RE.finditer(content):
        name = m.group(1)
        body_start = m.end() - 1
        end_marker = content.find(_GEMMA_TC_END, body_start)
        # No closing <tool_call|> tag: truncated call, reject in strict mode.
        if not allow_incomplete and end_marker < 0:
            continue
        scan_end = end_marker if end_marker >= 0 else len(content)
        end = _gemma_balanced_brace_end(content, body_start, scan_end)
        if end is None:
            continue
        body = content[body_start + 1 : end]
        try:
            args = _gemma_parse_mapping_body(body)
        except Exception:
            args = {}
        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return out


def _balanced_brace_end(text: str, brace_pos: int) -> int | None:
    """Index of the ``}`` matching ``{`` at ``brace_pos`` (ignores braces in JSON strings)."""
    if brace_pos >= len(text) or text[brace_pos] != "{":
        return None
    depth = 0
    in_string = False
    esc = False
    i = brace_pos
    while i < len(text):
        ch = text[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


_BARE_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _top_level_bare_json_name(probe: str) -> Optional[str]:
    """Top-level ``"name"`` (or ``"function"`` alias) of a bare-JSON object, else None; nested objects are skipped and truncated tails return None."""
    if not probe.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    function_value = None  # the ``"function"`` alias, used only if no ``"name"`` key
    i = 1
    n = len(probe)
    while i < n:
        while i < n and probe[i] in " \t\r\n,":
            i += 1
        if i >= n or probe[i] == "}":
            # End of object, no top-level ``"name"``: fall back to the ``"function"`` alias.
            return function_value
        if probe[i] != '"':
            return None
        try:
            key, consumed = decoder.raw_decode(probe[i:])
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(key, str):
            return None
        i += consumed
        while i < n and probe[i] in " \t\r\n":
            i += 1
        if i >= n or probe[i] != ":":
            return None
        i += 1
        while i < n and probe[i] in " \t\r\n":
            i += 1
        if key == "name":
            if i < n and probe[i] == '"':
                try:
                    value, _consumed = decoder.raw_decode(probe[i:])
                except (json.JSONDecodeError, ValueError):
                    return None
                return value if isinstance(value, str) else None
            return None
        if key == "function" and function_value is None and i < n and probe[i] == '"':
            # ``"function"`` is an alias; record it but keep scanning (``"name"`` wins).
            try:
                value, consumed = decoder.raw_decode(probe[i:])
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(value, str):
                function_value = value
            i += consumed
            continue
        # Skip a non-name top-level value; a truncated one returns None (keep the text).
        if i < n and probe[i] == "{":
            end = _balanced_brace_end(probe, i)
            if end is None:
                return None
            i = end + 1
        elif i < n and probe[i] == "[":
            end = _balanced_bracket_end(probe, i)
            if end is None:
                return None
            i = end + 1
        else:
            try:
                _value, consumed = decoder.raw_decode(probe[i:])
            except (json.JSONDecodeError, ValueError):
                return None
            i += consumed
    # No top-level ``"name"`` key: fall back to the ``"function"`` alias if seen.
    return function_value


def strip_leading_bare_json_call(text: str, enabled_tool_names: Optional[set] = None) -> str:
    """Remove leading Llama-3.2 bare-JSON calls (including a ``;``-chained run)
    that ``strip_tool_markup`` misses; non-call text is unchanged and
    ``enabled_tool_names`` gates like the parser. Consuming the whole chain
    matters because the loops keep this text as next-turn assistant history: a
    leftover executed call would be replayed alongside the structured
    ``tool_calls``."""
    remainder = text
    stripped_any = False
    while True:
        probe = strip_llama3_leading_sentinels(remainder.lstrip())
        # Skip the Llama-3 ``;`` inter-call separator between chained calls.
        if stripped_any:
            probe = probe.lstrip(" \t\n\r;")
        if not (probe.startswith("{") and ('"name"' in probe or '"function"' in probe)):
            return probe.lstrip() if stripped_any else text
        if enabled_tool_names is not None:
            # Only suppress when the leading object's TOP-LEVEL name is an enabled tool
            # (a nested ``"name"`` is data); an unknown name is kept.
            name = _top_level_bare_json_name(probe)
            if name not in enabled_tool_names:
                return probe.lstrip() if stripped_any else text
        end = _balanced_brace_end(probe, 0)
        if end is None:
            return ""  # truncated bare-JSON call -- nothing recoverable
        # A closed object must have the CALL SHAPE the parser accepts; an ordinary JSON
        # answer it rejects is content, so keep it visible.
        try:
            obj = json.loads(probe[: end + 1])
        except (json.JSONDecodeError, ValueError):
            return probe.lstrip() if stripped_any else text
        if not _bare_json_call_shaped(obj):
            return probe.lstrip() if stripped_any else text
        remainder = probe[end + 1 :]
        stripped_any = True


def _bare_json_call_shaped(obj) -> bool:
    """The shape gate ``_parse_llama3_bare_json`` applies to a decoded object."""
    if not isinstance(obj, dict):
        return False
    # The parser requires a TOP-LEVEL name; a nested one is data, not the call name.
    name = obj.get("name") or obj.get("function") or ""
    if not isinstance(name, str) or not name:
        return False
    if "parameters" in obj:
        return isinstance(obj.get("parameters"), dict)
    args = obj.get("arguments")
    if isinstance(args, dict):
        return True
    if isinstance(args, str):
        try:
            return isinstance(json.loads(args), dict)
        except (json.JSONDecodeError, ValueError):
            return False
    return False


def _gemma_balanced_brace_end(text: str, brace_pos: int, hard_stop: int) -> int | None:
    """Like ``_balanced_brace_end`` but skips ``<|"|>`` strings and matches {}/[] symmetrically."""
    if brace_pos >= len(text) or text[brace_pos] != "{":
        return None
    depth = 0
    i = brace_pos
    while i < hard_stop:
        if text.startswith(_GEMMA_STR_BEGIN, i):
            close = text.find(_GEMMA_STR_END, i + len(_GEMMA_STR_BEGIN))
            if close < 0:
                return None
            i = close + len(_GEMMA_STR_END)
            continue
        ch = text[i]
        if ch == "{" or ch == "[":
            depth += 1
        elif ch == "}" or ch == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _gemma_parse_value(text: str, i: int):
    """Parse one Gemma arg value at ``i``; returns ``(value, next_index)``."""
    if text.startswith(_GEMMA_STR_BEGIN, i):
        close = text.find(_GEMMA_STR_END, i + len(_GEMMA_STR_BEGIN))
        if close < 0:
            return text[i + len(_GEMMA_STR_BEGIN) :], len(text)
        return text[i + len(_GEMMA_STR_BEGIN) : close], close + len(_GEMMA_STR_END)
    if text[i] == "{":
        end = _gemma_balanced_brace_end(text, i, len(text))
        if end is None:
            return {}, len(text)
        return _gemma_parse_mapping_body(text[i + 1 : end]), end + 1
    if text[i] == "[":
        j, depth = i, 0
        while j < len(text):
            if text.startswith(_GEMMA_STR_BEGIN, j):
                k = text.find(_GEMMA_STR_END, j + len(_GEMMA_STR_BEGIN))
                if k < 0:
                    j = len(text)
                    break
                j = k + len(_GEMMA_STR_END)
                continue
            ch = text[j]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[i + 1 : j]
        items: list[Any] = []
        k = 0
        while k < len(body):
            if body[k] in " \t\n\r,":
                k += 1
                continue
            v, k = _gemma_parse_value(body, k)
            items.append(v)
        return items, j + 1
    # Primitive: number / true/false/null / bare identifier.
    end = i
    while end < len(text) and text[end] not in ",}]" and not text.startswith(_GEMMA_STR_BEGIN, end):
        end += 1
    if end == i:
        # Stray delimiter, nothing consumed: advance past it so callers can't spin forever.
        return "", i + 1
    raw = text[i:end].strip()
    if raw == "true":
        return True, end
    if raw == "false":
        return False, end
    if raw == "null":
        return None, end
    try:
        return int(raw), end
    except ValueError:
        pass
    try:
        return float(raw), end
    except ValueError:
        pass
    return raw, end


def _gemma_parse_mapping_body(body: str) -> dict[str, Any]:
    """Parse a Gemma argument mapping (content between `{` and `}`)."""
    out: dict[str, Any] = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        if body.startswith(_GEMMA_STR_BEGIN, i):
            close = body.find(_GEMMA_STR_END, i + len(_GEMMA_STR_BEGIN))
            if close < 0:
                break
            key = body[i + len(_GEMMA_STR_BEGIN) : close]
            i = close + len(_GEMMA_STR_END)
        else:
            kstart = i
            while i < n and body[i] != ":":
                i += 1
            key = body[kstart:i].strip()
        while i < n and body[i] in " \t\n\r":
            i += 1
        if i < n and body[i] == ":":
            i += 1
        while i < n and body[i] in " \t\n\r":
            i += 1
        if i >= n:
            out[key] = None
            break
        v, i = _gemma_parse_value(body, i)
        out[key] = v
    return out
