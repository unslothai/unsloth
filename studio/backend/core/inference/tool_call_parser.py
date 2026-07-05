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
  - ``<｜tool▁calls▁begin｜>...function<｜tool▁sep｜>NAME\\n``\\`\\`\\`json\\n{...}\\n\\`\\`\\`...``  (DeepSeek R1)
  - ``<｜tool▁calls▁begin｜>...<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}<｜tool▁call▁end｜>...``  (DeepSeek V3 / V3.1)
  - ``<tool_call>NAME\\n<arg_key>k</arg_key>\\n<arg_value>v</arg_value>...</tool_call>``  (GLM 4.5 / 4.6 / 4.7)
  - ``<|tool_calls_section_begin|>...<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{json}<|tool_call_end|>...``  (Kimi K2)

Missing closing tags / brackets are tolerated: models often truncate mid-stream.
"""

# Dependency-light (external servers import it standalone); `from __future__ import
# annotations` keeps PEP 604 `X | None` annotations lazy so python 3.9 import works.
from __future__ import annotations

import json
import re
from typing import Any, Optional

# Qwen/Hermes, Qwen3.5 XML and Gemma 4 go through the shared `core.tool_healing`
# (strict/Auto-Heal contract); this module adds Llama-3, Mistral and bare JSON.
from core import tool_healing as _tool_healing


# Flip the streaming buffer STREAMING->DRAINING so partial markup never leaks.
TOOL_XML_SIGNALS = (
    "<tool_call>",
    "<function=",
    '<function name="',
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<|tool_call>",
    # DeepSeek R1 / V3 / V3.1 -- 5 opener variants llama.cpp keeps.
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁call▁begin｜>",
    "<｜tool_calls_begin｜>",
    "<｜tool▁calls｜>",
    "<｜tool calls begin｜>",
    "<｜tool\\_calls\\_begin｜>",
    # Kimi K2 / Moonshot.
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
)


# DeepSeek envelope opener variants llama.cpp accepts, including the short
# ``<｜tool▁calls｜>`` form and the space / escaped-underscore spellings. Shared
# by ``_DEEPSEEK_BEGIN_RE`` (parsing) and the strip patterns below so a signal we
# parse can never be left un-stripped (a short-opener envelope used to leak).
_DEEPSEEK_OPEN_ALT = (
    r"tool▁calls▁begin|tool_calls_begin|tool calls begin|tool\\_calls\\_begin|tool▁calls"
)
_DEEPSEEK_OPEN_RE_SRC = r"<｜(?:" + _DEEPSEEK_OPEN_ALT + r")｜>"

# Closed pairs only (mid-stream); _TOOL_ALL_PATS also eats unclosed tails at
# end-of-turn. ``[\w-]+`` on ``<function=...>`` tracks OpenAI's
# ``^[a-zA-Z0-9_-]{1,64}$`` so hyphenated MCP names parse like built-ins.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    # Extend to the call's REAL ``</function>`` (last one before the next
    # ``<function=`` or end) so a literal ``</function>`` inside a value doesn't
    # truncate the strip; the lookahead keeps each call separate (greedy ``.*``
    # would merge them).
    re.compile(
        r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>'
        r'(?:(?!<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>).)*'
        r"</function>",
        re.DOTALL,
    ),
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\]\s*\[.*?\](?:\s*</s>)?", re.DOTALL),
    # Mistral v11+ ``[TOOL_CALLS]name{json}`` (may chain), close at ``}``.
    re.compile(r"\[TOOL_CALLS\]\s*[\w\.\-]+\s*(?:\[ARGS\])?\s*\{.*?\}", re.DOTALL),
    # DeepSeek R1 / V3 / V3.1: full envelope (any opener variant) ... end.
    re.compile(_DEEPSEEK_OPEN_RE_SRC + r".*?<｜tool▁calls▁end｜>", re.DOTALL),
    # Kimi K2: ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>``.
    re.compile(r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", re.DOTALL),
    # Kimi K2 bare (section-less) closed call ``<|tool_call_begin|>...<|tool_call_end|>``.
    # The parser accepts this shape; without a CLOSED arm here the unclosed
    # ``<|tool_call_begin|>.*$`` catch-all below eats trailing prose to EOS.
    re.compile(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>.*$', re.DOTALL),
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\].*$", re.DOTALL),
    re.compile(r"<\|python_tag\|>.*$", re.DOTALL),
    # DeepSeek envelopes truncated mid-stream (any opener variant).
    re.compile(_DEEPSEEK_OPEN_RE_SRC + r".*$", re.DOTALL),
    re.compile(r"<｜tool▁call▁begin｜>.*$", re.DOTALL),
    # Kimi K2 envelope truncated.
    re.compile(r"<\|tool_calls_section_begin\|>.*$", re.DOTALL),
    re.compile(r"<\|tool_call_begin\|>.*$", re.DOTALL),
    # Gemma 4 wrapper-less ``call:NAME{...}`` (closed, nested, and truncated) is
    # handled entirely by ``_strip_gemma_wrapperless_calls`` above -- the single
    # authority so the enabled-name gate is honoured (a disabled/example name in
    # prose is kept, not deleted). It runs before this list in every caller.
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
# Qwen3.5 ``<function=name>`` and the attribute form ``<function name="name">``
# (MiniCPM-5, MiniMax-M2); name class ``[\w.\-]+`` lands in group(1) or group(2).
_TC_FUNC_START_RE = re.compile(r'<function(?:=([\w\.\-]+)|\s+name="([\w\.\-]+)")>\s*')
# Body ends at ``</tool_call>`` (Hermes) or ``</function>`` (Qwen3.5 / MiniCPM-5)
# so it stops at the close even when prose follows (else prose leaked into args).
_TC_END_TAG_RE = re.compile(r"</(?:tool_call|function)>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
# Horizontal whitespace only (``[^\S\n]*``, not ``\s*``) so the wrapping newline +
# first-line indentation survive; ``_trim_param_value`` trims one newline, preserving
# code indentation (SGLang qwen3_coder).
_TC_PARAM_START_RE = re.compile(
    r'<(?:parameter|param)(?:=([\w\.\-]+)|\s+name="([\w\.\-]+)")>[^\S\n]*'
)
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</(?:parameter|param)>\s*$")

# Llama-3 ``<|python_tag|>NAME.call(...)``.
_LLAMA3_PYTHON_TAG = "<|python_tag|>"
_LLAMA3_PY_CALL_RE = re.compile(
    r"<\|python_tag\|>\s*([\w\.\-]+)\s*\.\s*call\s*\(",
)
# Anchored at a fixed offset (char after ``<|python_tag|>``) plus the ``; NAME.call(``
# chain separator; fixed-offset (not a free scan) ignores ``.call(`` inside JSON args.
_LLAMA3_PY_CALL_HEAD_RE = re.compile(r"\s*([\w\.\-]+)\s*\.\s*call\s*\(")
_LLAMA3_CALL_CHAIN_RE = re.compile(r"\s*;\s*([\w\.\-]+)\s*\.\s*call\s*\(")
# Llama-3 ``.call(k=v)`` kwarg tokens, hand-scanned below (not finditer) to stay
# linear on a truncated body; finditer retries every offset of a long run (ReDoS).
_LLAMA3_KEY_RE = re.compile(r"\w+")
_LLAMA3_WS_RE = re.compile(r"\s*")
# ints, decimals (1.5, 1., .5) and sci notation; trailing ``(?![\w.])`` stops a token
# like ``1.2.3`` being truncated to ``1.2`` (which would mis-parse the remainder).
_LLAMA3_NUM_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])")
_LLAMA3_LIT_RE = re.compile(r"true|false|null")

# Mistral ``[TOOL_CALLS]`` trigger. v11+ chains them, each followed by a bare name
# plus ``{json}`` (Magistral) or ``[ARGS]{json}`` (Ministral / Large 3).
_MISTRAL_TRIGGER = "[TOOL_CALLS]"
_MISTRAL_ARGS_MARKER = "[ARGS]"
# Mistral Small 3.2 emits ``name[CALL_ID]<id>[ARGS]{json}`` (absent on Ministral /
# Magistral); llama.cpp distinguishes the two on ``[CALL_ID]`` (common/chat.cpp).
_MISTRAL_CALL_ID_MARKER = "[CALL_ID]"
# Magistral wraps reasoning in ``[THINK]...[/THINK]``; a ``[TOOL_CALLS]`` inside
# that block is chain-of-thought, not a real call.
_MISTRAL_THINK_OPEN = "[THINK]"
_MISTRAL_THINK_CLOSE = "[/THINK]"
_MISTRAL_V11_NAME_RE = re.compile(r"\s*([\w\.\-]+)\s*")

# DeepSeek R1 / V3 / V3.1 markers (full-width pipe U+FF5C, lower-
# one-eighth-block U+2581). llama.cpp accepts five variants of the
# outer block-open; we mirror its tolerance.
_DEEPSEEK_BEGIN_RE = re.compile(_DEEPSEEK_OPEN_RE_SRC)
_DEEPSEEK_END = "<｜tool▁calls▁end｜>"
_DEEPSEEK_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_DEEPSEEK_SEP = "<｜tool▁sep｜>"
_DEEPSEEK_CALL_END = "<｜tool▁call▁end｜>"
# R1 wraps args in a Markdown ```json ... ``` fence and prefixes the call with
# the literal ``function`` token; V3 / V3.1 do not. Scanned with ``str.find``
# (a greedy ``([^\n]+)\n```json`` regex is O(N^2) on a fence-less truncated body).
_DEEPSEEK_R1_FUNC_MARKER = "function" + _DEEPSEEK_SEP
_DEEPSEEK_R1_FENCE = "\n```json\n"
_DEEPSEEK_R1_CLOSE_RE = re.compile(r"```[\s\r\n]*" + re.escape(_DEEPSEEK_CALL_END))
# V3 / V3.1 (bare-JSON, no fence) is handled by ``str.find`` on the sep
# marker; see ``_parse_deepseek_tool_calls`` -- a ``([^\n<]+?)`` regex
# is O(N^2) on adversarial truncated bodies.

# GLM 4.5 / 4.6 / 4.7: ``<tool_call>NAME[\n]<arg_key>K</arg_key>...``.
# GLM 4.7 strips the ``\n`` after the name, so the lookahead also allows
# ``<arg_key>`` directly and ``</tool_call>`` for a zero-argument call
# (``<tool_call>get_current_date</tool_call>``). First-char ``[^\n<{]``
# excludes Qwen.
# Name class ``[\w.\-]+`` mirrors the other parsers (``_TC_FUNC_START_RE``) so
# literal prose like ``<tool_call>not a call</tool_call>`` is NOT parsed as a tool
# named "not a call"; a broad ``[^\n<]*`` captured arbitrary spaced text. ``{`` is
# still excluded (Qwen's ``<tool_call>{json}`` is handled by the JSON parser).
_GLM_TC_OPEN_RE = re.compile(r"<tool_call>\s*([\w.\-]+)\s*(?=\n|<arg_key>|</tool_call>)")
_GLM_TC_CLOSE = "</tool_call>"
_GLM_ARG_KEY_OPEN = "<arg_key>"
_GLM_ARG_KEY_CLOSE = "</arg_key>"
_GLM_ARG_VAL_OPEN = "<arg_value>"
_GLM_ARG_VAL_CLOSE = "</arg_value>"
# Template emits string args raw and non-strings via tojson; without
# tool schema we can only safely decode unambiguous JSON literals.
# Bare ``42`` / ``true`` / ``null`` remain ambiguous with strings.
_GLM_JSON_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# Kimi K2 / Moonshot (ASCII pipes). Id between begin and arg_begin is
# ``functions.NAME:IDX``; strip ``functions.`` and ``:N`` for the name.
_KIMI_SECTION_BEGIN = "<|tool_calls_section_begin|>"
_KIMI_SECTION_END = "<|tool_calls_section_end|>"
_KIMI_CALL_BEGIN = "<|tool_call_begin|>"
_KIMI_ARG_BEGIN = "<|tool_call_argument_begin|>"
_KIMI_CALL_END = "<|tool_call_end|>"
_KIMI_ID_RE = re.compile(r"^(?:functions\.)?([\w\.\-]+)(?::(\d+))?$")

# Gemma 4: ``<|tool_call>call:NAME{...}<tool_call|>``, ``<|"|>`` wraps strings.
_GEMMA_TC_RE = re.compile(r"<\|tool_call>\s*call\s*:\s*([\w\.\-]+)\s*\{")
_GEMMA_STR_BEGIN = '<|"|>'
_GEMMA_STR_END = '<|"|>'
_GEMMA_TC_END = "<tool_call|>"

# skip_special_tokens strips the ``<|tool_call>`` / ``<tool_call|>`` wrapper and
# the ``<|"|>`` string markers, so a streamed Gemma-4 tool call arrives as a
# bare ``call:NAME{k:v, ...}`` with unquoted values. Match that here; the
# ``(?<!\w)`` guard avoids catching words like ``recall:``.
_GEMMA_BARE_TC_RE = re.compile(r"(?<!\w)call\s*:\s*([\w\.\-]+)\s*\{")
# A (possibly partial) leading wrapper-less Gemma prefix -- ``call``, ``call :``,
# ``call : partial_name`` -- before the ``{`` arrives. Whitespace-tolerant like
# ``_GEMMA_BARE_TC_RE`` so the streaming buffer holds ``call : web_search`` (not just
# the exact ``call:`` spelling) instead of leaking it as visible text.
_GEMMA_BARE_TC_PREFIX_RE = re.compile(r"(?<!\w)call\s*(?::\s*[\w\.\-]*)?$")
# Keys must start with a letter/underscore so a comma followed by a time or
# ratio inside a value (``query:meet at 10:00, 11:00``) is not misread as a new
# ``11:``-style key, matching the wrapped path's _GEMMA_NEXT_KEY_RE.
_GEMMA_KEY_RE = re.compile(r"\s*([A-Za-z_][\w.\-]*)\s*:")


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
    """Drop a leading Magistral ``[THINK]...[/THINK]`` so a ``[TOOL_CALLS]`` inside
    reasoning is not taken as a real call; an unclosed ``[THINK]`` drops from it on."""
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
    """Strip cleanly-closed ``[TOOL_CALLS]`` blocks (array, ``name{json}``,
    ``name[ARGS]{json}``) via balanced scanning -- a non-greedy ``\\{.*?\\}`` would
    truncate at the first ``}`` and lose nested JSON. Unclosed runs are left for
    ``final=True`` cleanup."""
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
        # Single-object shape ``[TOOL_CALLS] { json }`` (no name/array): the parser
        # accepts it, so the display strip must remove it too (else it leaks).
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
        # Consume the optional EOS marker too, mirroring the array shape, so a
        # ``[TOOL_CALLS]name{json}</s>`` tail doesn't leave ``</s>`` as content.
        if text.startswith("</s>", cursor):
            cursor += len("</s>")
    return "".join(out)


def _strip_gemma_wrapperless_calls(text: str, enabled_tool_names: Optional[set] = None) -> str:
    """Strip closed wrapper-less Gemma ``call:NAME{...}`` calls with balanced brace
    scanning, so a nested object/array argument (``call:f{loc:{city:NYC},n:3}``) is
    removed whole instead of leaving a trailing ``}`` -- the ``[^{}]*`` cleanup regex
    only matches brace-free bodies. Unclosed runs are left for the regex tails.

    ``enabled_tool_names`` (when given) gates the strip on the name the same way
    ``_parse_gemma_tool_calls`` gates parsing: a disabled/example ``call:foo{...}`` in
    prose is kept visible instead of being deleted from the answer. ``None`` strips
    every closed call (unrestricted mode / direct callers)."""
    if _whole_content_is_json_value(text):
        return text
    n = len(text)
    out = []
    cursor = 0
    while cursor < n:
        m = _GEMMA_BARE_TC_RE.search(text, cursor)
        if not m:
            out.append(text[cursor:])
            break
        disabled = enabled_tool_names is not None and m.group(1) not in enabled_tool_names
        brace = m.end() - 1  # _GEMMA_BARE_TC_RE consumes through the opening ``{``
        # Same boundary scanner as _parse_gemma_tool_calls: the strip span must
        # cover exactly what the parser consumed, or a quoted ``}`` in a code
        # argument leaves the call's tail visible after the strip.
        end = _gemma_body_brace_end(text, brace)
        closed = end is not None
        next_index = (end + 1) if closed else len(text)
        if not closed:
            # Unclosed (truncated / still-streaming) call. Drop a real (enabled) call
            # to EOS so the raw markup does not leak; keep a disabled/example name as
            # prose. Nothing parseable follows an unclosed call, so stop either way.
            out.append(text[cursor:] if disabled else text[cursor : m.start()])
            break
        if disabled:
            # Disabled/example name -- prose, not a call: keep it whole and continue.
            out.append(text[cursor:next_index])
        else:
            out.append(text[cursor : m.start()])
        cursor = next_index  # already past the matching ``}``
    return "".join(out)


_FUNC_CLOSE_TAG_RE = re.compile(r"</function>")


def _strip_function_xml_calls(text: str, *, final: bool) -> str:
    """Strip ``<function=NAME>...</function>`` (and ``<function name="NAME">``) calls
    by mirroring the parser, not a regex: a ``<function=...>`` inside an open
    ``<parameter>`` value is literal data, not a new call, and each call closes at its
    REAL (last) ``</function>``. ``final`` also drops a trailing unclosed call to EOF;
    otherwise it stays buffered (still streaming)."""
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
            close = cm  # the call's real close is the LAST </function> before the next call
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


def _glm_value_close(
    text: str,
    vs: int,
    *,
    strict: bool = False,
) -> int:
    """Index of the ``</arg_value>`` that really ends the GLM value beginning at
    ``vs``: the first one whose next non-space token is ``<arg_key>``, ``</tool_call>``
    or end-of-text. A literal ``</arg_value>`` inside the value (e.g.
    ``print("</arg_value>")``) is followed by ordinary text and skipped; a literal
    ``</tool_call>`` inside the value does not end it.

    The next-token test alone cannot tell a real close from the full pair
    ``</arg_value></tool_call>`` embedded mid-value (code documenting the GLM
    format: ``print("</arg_value></tool_call>")``), so a candidate is also
    required to sit at balanced quote state: an embedded pair lives inside a
    string literal, whose open quote is still unclosed at that point. Quote
    openers are contextual, mirroring the Gemma scanners: a single quote
    opens only after punctuation context (an apostrophe in ``what's the
    weather`` is prose), a double quote also at the start of a word. When no
    candidate balances, the first token-valid candidate wins as before --
    except in ``strict`` mode (Auto-Heal off), which refuses the in-quote
    fallback: a truncated value whose only close candidates sit inside a
    string literal must reject the call instead of executing truncated
    arguments. Returns -1 if unclosed."""
    n = len(text)
    search = vs
    first_candidate = -1
    quote = ""
    prev = ":"
    prev_raw = ":"
    qpos = vs  # quote-state cursor; advanced incrementally to each candidate
    while True:
        ve = text.find(_GLM_ARG_VAL_CLOSE, search)
        if ve < 0:
            return -1 if strict else first_candidate
        j = ve + len(_GLM_ARG_VAL_CLOSE)
        while j < n and text[j] in " \t\r\n":
            j += 1
        if j >= n or text.startswith(_GLM_ARG_KEY_OPEN, j) or text.startswith(_GLM_TC_CLOSE, j):
            while qpos < ve:
                ch = text[qpos]
                if quote:
                    if ch == "\\" and qpos + 1 < ve:
                        qpos += 2
                        continue
                    if ch == quote:
                        quote = ""
                elif ch in "\"'" and (prev in ":{[(,=" or (ch == '"' and prev_raw.isspace())):
                    quote = ch
                if not ch.isspace():
                    prev = ch
                prev_raw = ch
                qpos += 1
            if not quote:
                return ve
            if first_candidate < 0:
                first_candidate = ve
        search = ve + len(_GLM_ARG_VAL_CLOSE)


def _strip_glm_calls(text: str, *, final: bool) -> str:
    """Strip GLM 4.x ``<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>
    ...</tool_call>`` calls by scanning to each call's REAL ``</tool_call>``.

    The real close is the ``</tool_call>`` that follows the last consumed
    ``<arg_value>`` and precedes the next ``<arg_key>`` (mirrors
    ``_parse_glm_tool_calls``), so a literal ``</tool_call>`` inside an argument value
    -- e.g. ``print("</tool_call>")`` -- is treated as data instead of the non-greedy
    ``<tool_call>.*?</tool_call>`` regex stopping there and leaking the call's tail.
    Qwen / Hermes ``<tool_call>{json}`` has no NAME token after the opener, so
    ``_GLM_TC_OPEN_RE`` does not match it and it is left to the regex arms. With
    ``final`` a truncated call (no real close) is dropped to EOS; otherwise the
    unclosed call is left buffered for a later pass."""
    out: list[str] = []
    cursor = 0
    n = len(text)
    while True:
        m = _GLM_TC_OPEN_RE.search(text, cursor)
        if not m:
            break
        apos = m.end()
        close = -1
        while True:
            ks = text.find(_GLM_ARG_KEY_OPEN, apos)
            tc = text.find(_GLM_TC_CLOSE, apos)
            if tc >= 0 and (ks < 0 or tc < ks):
                close = tc
                break
            if ks < 0:
                break  # no close and no more keys -- truncated body
            ke = text.find(_GLM_ARG_KEY_CLOSE, ks + len(_GLM_ARG_KEY_OPEN))
            if ke < 0:
                break
            vstart = ke + len(_GLM_ARG_KEY_CLOSE)
            while vstart < n and text[vstart] in " \t\r\n":
                vstart += 1
            if not text.startswith(_GLM_ARG_VAL_OPEN, vstart):
                apos = ke + len(_GLM_ARG_KEY_CLOSE)
                continue
            vs = vstart + len(_GLM_ARG_VAL_OPEN)
            # Real </arg_value> = first whose next non-space token is <arg_key> /
            # </tool_call> / end (mirror _parse_glm_tool_calls), so a literal
            # </arg_value> or </tool_call> inside the value is data, not an early close.
            ve = _glm_value_close(text, vs)
            if ve < 0:
                break  # unclosed <arg_value> -- truncated
            apos = ve + len(_GLM_ARG_VAL_CLOSE)
        if close >= 0:
            out.append(text[cursor : m.start()])
            cursor = close + len(_GLM_TC_CLOSE)
            continue
        # Truncated GLM call (no real close yet).
        if final:
            out.append(text[cursor : m.start()])
            cursor = n
        # Non-final: leave the unclosed call (and any tail) buffered as-is.
        break
    out.append(text[cursor:])
    return "".join(out)


def strip_tool_markup(
    text: str,
    *,
    final: bool = False,
    enabled_tool_names: Optional[set] = None,
) -> str:
    """Strip tool-call markup. ``final=False`` keeps in-progress markup buffered;
    ``final=True`` also drops trailing unclosed runs and trims. ``enabled_tool_names``
    gates the markerless Gemma ``call:NAME{...}`` strip so a disabled/example name in
    prose is kept (mirrors the parser gate); ``None`` strips every closed call."""
    if final:
        # End-of-turn only: drop a leading Magistral ``[THINK]...[/THINK]`` block.
        # Its bracket form is not the ``<think>`` the reasoning channel renders, so
        # without this the raw reasoning leaks into the safetensors display/history
        # (GGUF/llama.cpp routes it natively). Streaming (final=False) is untouched.
        text = _strip_mistral_reasoning(text)
    text = _strip_mistral_closed_calls(text)
    if final:
        text = _strip_gemma_wrapperless_calls(text, enabled_tool_names)
    # Scan-strip the function-XML form (parser-accurate: a literal ``<function=...>``
    # inside a parameter value is data, not a call), then the remaining regex arms
    # cover the other formats. The function regex arms stay in the pattern lists for
    # the streaming callers but no-op here once the scan ran.
    text = _strip_function_xml_calls(text, final = final)
    # GLM 4.x <tool_call>NAME<arg_key>..<arg_value>..</tool_call>: scan to the call's
    # real close so a literal </tool_call> inside an arg value is data, not a leak
    # (the regex <tool_call>.*?</tool_call> arm would stop at that literal). Qwen
    # <tool_call>{json} is left to the regex arms.
    text = _strip_glm_calls(text, final = final)
    pats = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in pats:
        text = pat.sub("", text)
    return text.strip() if final else text


def has_tool_signal(text: str) -> bool:
    return any(s in text for s in TOOL_XML_SIGNALS)


# A Qwen/Hermes ``<tool_call>`` or ``<function=...>`` envelope whose arguments
# contain literal DeepSeek/Kimi markers (e.g. a user asking about that syntax)
# must be parsed as the OUTER call, not the embedded marker. Detect that such an
# envelope opens before the first DeepSeek/Kimi marker so the marker pre-pass is
# skipped for it.
_EMBEDDED_MARKER_RE = re.compile(
    _DEEPSEEK_OPEN_RE_SRC + "|" + re.escape(_KIMI_SECTION_BEGIN) + "|" + re.escape(_KIMI_CALL_BEGIN)
)
# Both ``<function=NAME>`` and the attribute form ``<function name="NAME">`` are
# supported outer envelopes (see ``_TC_FUNC_START_RE``), so the guard must cover
# both -- otherwise an attribute-form call whose argument embeds a DeepSeek/Kimi
# marker is hijacked by the marker pre-pass and the wrong tool runs.
_OUTER_ENVELOPE_OPEN_RE = re.compile(r'<tool_call>|<function(?:=|\s+name=")|<\|tool_call>')
# The CLOSED forms of the outer envelopes above, each spanning to its REAL final
# close so a literal ``</tool_call>``/``</function>`` inside an argument value is
# treated as data, not the envelope boundary. ``_TOOL_CLOSED_PATS[0]`` is the LAZY
# ``<tool_call>.*?</tool_call>`` strip form (stops at the first, possibly in-string,
# close), so use a real-close ``<tool_call>`` pattern here -- the negative lookahead
# keeps back-to-back calls separate, mirroring the ``<function>`` arm.
_OUTER_ENVELOPE_CLOSED_PATS = (
    re.compile(r"<tool_call>(?:(?!<tool_call>).)*</tool_call>", re.DOTALL),
    _TOOL_CLOSED_PATS[1],
    # Wrapped Gemma: a DeepSeek/Kimi example quoted inside its <|"|> string
    # arguments is data for the outer call, exactly like the XML envelopes.
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
)


def _marker_inside_leading_envelope(content: str) -> bool:
    first_marker = _EMBEDDED_MARKER_RE.search(content)
    if first_marker is None:
        return False
    # A leading bare-JSON call object or Mistral [TOOL_CALLS] call is an outer
    # envelope too: a DeepSeek/Kimi marker inside its argument strings is data
    # (a query documenting the marker), and running the pre-pass on it would
    # promote the embedded no-arg literal and drop the real outer call.
    i = 0
    n = len(content)
    while i < n and content[i] in " \t\n\r":
        i += 1
    if content.startswith("{", i):
        end = _balanced_brace_end(content, i)
        if (
            end is not None
            and _top_level_bare_json_name(content[i : end + 1]) is not None
            and i < first_marker.start()
        ):
            # Inside the closed call the marker is argument data; AFTER it the
            # closed leading call still owns the turn in document order (the
            # marker is a trailing example, or data in a later ``;``-chained
            # call's strings) -- same rule as the closed XML envelopes below.
            return True
    elif content.startswith(_MISTRAL_TRIGGER, i):
        end = _mistral_region_end(content, i)
        if end is not None and i < first_marker.start():
            return True
    # Remove CLOSED outer envelopes first. Their patterns extend to the REAL final
    # close, so a literal ``</function>``/``</tool_call>`` inside an argument value does
    # not end them early. If that removes every marker, the marker sat inside a closed
    # outer call (e.g. a user asking about the syntax), so skip the DeepSeek/Kimi
    # pre-pass and let the outer XML parser own it.
    # A closed non-DeepSeek/Kimi call that PRECEDES the first marker owns the
    # turn in document order: the marker is a trailing example or argument
    # data, so the pre-pass must not steal it (same rule as the other leading
    # guards). Markers inside the closed call are covered a fortiori.
    for _pat in _OUTER_ENVELOPE_CLOSED_PATS:
        m = _pat.search(content)
        if m is not None and m.start() < first_marker.start():
            return True
    residue = content
    for _pat in _OUTER_ENVELOPE_CLOSED_PATS:
        residue = _pat.sub("", residue)
    marker = _EMBEDDED_MARKER_RE.search(residue)
    if marker is None:
        return True
    # A marker still stands. Any outer opener left in the residue is necessarily
    # UNCLOSED (the closed ones were removed); if one begins before the marker it is a
    # truncated outer call whose arguments hold the marker and Auto-Heal will repair
    # it, so skip the pre-pass. Otherwise the marker is a real standalone DeepSeek/Kimi
    # call (possibly after a closed syntax example), so run the pre-pass.
    opener = _OUTER_ENVELOPE_OPEN_RE.search(residue)
    return opener is not None and opener.start() < marker.start()


def _mistral_region_end(text: str, idx: int) -> int | None:
    """Exclusive end of the balanced ``[TOOL_CALLS]`` call starting at ``idx``,
    or ``None`` when truncated/unrecognised (same shapes as the strip scan:
    array, single-object, and named ``name [CALL_ID]? [ARGS]? {json}``)."""
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
    """True when the first foreign tool signal sits inside the balanced body of
    an earlier Mistral ``[TOOL_CALLS]`` call -- i.e. it is that call's argument
    data (a query quoting tool markup), not a real call. The shared XML parser
    (or the python_tag parser) would otherwise promote the literal and execute
    the wrong tool, so the Mistral parser must take the outer call first. A
    signal BEFORE the trigger keeps the normal order (the outer XML call wins
    and a ``[TOOL_CALLS]`` literal inside its arguments stays data)."""
    trig = content.find(_MISTRAL_TRIGGER)
    if trig < 0:
        return False
    lead = 0
    n = len(content)
    while lead < n and content[lead] in " \t\n\r":
        lead += 1
    # A LEADING parseable Mistral call owns the turn outright (first-match in
    # document order): literal XML in trailing prose after the call must not
    # be promoted over it by the earlier shared XML pass.
    if trig == lead and _mistral_region_end(content, trig) is not None:
        return True
    first_xml = _first_foreign_tool_signal(content)
    if first_xml is None or first_xml < trig:
        return False
    end = _mistral_region_end(content, trig)
    return end is not None and first_xml < end


def _first_foreign_tool_signal(content: str) -> int | None:
    """Offset of the first tool signal a non-envelope parser would fire on:
    the XML forms plus the Llama-3 ``<|python_tag|>`` marker
    (``_parse_llama3_python_tag`` also runs before the Mistral parser, so a
    python_tag literal inside a leading envelope's arguments needs the same
    protection as XML literals)."""
    first = None
    for sig in ("<tool_call>", "<|tool_call>", "<function=", "<|python_tag|>"):
        p = content.find(sig)
        if p >= 0 and (first is None or p < first):
            first = p
    attr = re.search(r'<function\s+name="', content)
    if attr is not None and (first is None or attr.start() < first):
        first = attr.start()
    # DeepSeek/Kimi markers are foreign to a JSON envelope too: their pre-pass
    # runs before the bare-JSON parser, so a marker literal inside a leading
    # object must route through the same guard (and, for a disabled object,
    # the same drop-and-parse-the-tail recursion, so a real DeepSeek/Kimi call
    # AFTER the object is still reached).
    marker = _EMBEDDED_MARKER_RE.search(content)
    if marker is not None and (first is None or marker.start() < first):
        first = marker.start()
    return first


def _xml_signal_inside_leading_bare_json(content: str) -> bool:
    """True when the first foreign tool signal sits inside the balanced body of
    a LEADING bare-JSON call object -- i.e. it is a string argument quoting
    tool markup (a ``code`` value citing ``<function=...>``), not a real call.
    The shared XML parser would otherwise promote the literal and execute the
    wrong tool, so the bare-JSON parser must take the outer call first
    (sibling of ``_xml_signal_inside_leading_mistral``)."""
    i = 0
    n = len(content)
    while i < n and content[i] in " \t\n\r":
        i += 1
    if i >= n or content[i] not in "{[":
        return False
    if content[i] == "[":
        # A leading ARRAY is never a call: it qualifies only as a structured
        # JSON answer, whose quoted literals are data.
        end = _balanced_bracket_end(content, i)
        if end is None:
            return False
        try:
            json.loads(content[i : end + 1])
        except ValueError:
            return False
        first_xml = _first_foreign_tool_signal(content)
        trig = content.find(_MISTRAL_TRIGGER)
        if trig >= 0 and (first_xml is None or trig < first_xml):
            first_xml = trig
        return first_xml is not None and i < first_xml < end
    end = _balanced_brace_end(content, i)
    if end is None:
        return False
    if _top_level_bare_json_name(content[i : end + 1]) is None:
        # Not a call object -- but a NAMELESS leading object that parses as
        # real JSON (a structured answer) is an envelope for this purpose too:
        # markup quoted inside its strings is data, and the guard's
        # decline-then-parse-the-tail path already handles it (the bare-JSON
        # parser rejects the nameless object, the object is dropped, the tail
        # is parsed). Non-JSON braced prose keeps the old behaviour.
        try:
            json.loads(content[i : end + 1])
        except ValueError:
            return False
    first_xml = _first_foreign_tool_signal(content)
    # The Mistral trigger is foreign to a JSON envelope too: the Mistral parser
    # runs before the bare-JSON one, so a "[TOOL_CALLS]..." literal inside the
    # leading object's strings would otherwise be promoted over the outer call.
    trig = content.find(_MISTRAL_TRIGGER)
    if trig >= 0 and (first_xml is None or trig < first_xml):
        first_xml = trig
    return first_xml is not None and i < first_xml < end


def _signal_inside_leading_wrapperless_gemma(
    content: str, enabled_tool_names: Optional[set]
) -> bool:
    """True when the first foreign tool signal sits inside the balanced body
    of a LEADING wrapper-less Gemma call -- a quoted literal in its argument
    (a query citing another tool syntax) is data, and the earlier passes would
    promote it and drop the outer call (sibling of the Mistral/bare-JSON
    leading guards). Markerless form, so the name must be an enabled tool
    (``None`` keeps the name-agnostic behaviour)."""
    first = _first_foreign_tool_signal(content)
    # The Mistral trigger is foreign to a Gemma call too: the Mistral parser
    # runs before the Gemma fallback, so a ``[TOOL_CALLS]...`` literal quoted
    # inside a leading Gemma body would otherwise be promoted over the outer
    # call (same local inclusion as the bare-JSON guard).
    trig = content.find(_MISTRAL_TRIGGER)
    if trig >= 0 and (first is None or trig < first):
        first = trig
    if first is None:
        return False
    # The call need not open the response: a visible preamble before
    # ``call:NAME{...}`` is the normal shape. What matters is that an ENABLED
    # balanced call begins before the first foreign signal and contains it;
    # a signal that precedes every such call keeps the normal order.
    cursor = 0
    while True:
        m = _GEMMA_BARE_TC_RE.search(content, cursor)
        if m is None or m.start() > first:
            return False
        if enabled_tool_names is not None and m.group(1) not in enabled_tool_names:
            cursor = m.end()
            continue
        end = _gemma_body_brace_end(content, m.end() - 1)
        if end is None:
            return False
        return m.end() - 1 < first <= end


def _disabled_gemma_call_end_containing_signal(
    content: str, enabled_tool_names: Optional[set]
) -> int | None:
    """End offset (exclusive) of the earliest DISABLED wrapper-less Gemma call
    whose balanced body contains the first foreign signal, else None.

    A disabled/example name is prose by design, so a tool literal quoted in
    its argument is data: the caller drops the span for parsing and recurses
    on the tail (sibling of the nameless-JSON guard). An ENABLED call before
    the signal defers to the enabled-call guard instead."""
    if enabled_tool_names is None:
        return None
    first = _first_foreign_tool_signal(content)
    # Mirror the enabled-call guard: the Mistral trigger counts as a foreign
    # signal here too, so a disabled example quoting ``[TOOL_CALLS]...`` drops
    # the span instead of letting the Mistral parser promote the literal.
    trig = content.find(_MISTRAL_TRIGGER)
    if trig >= 0 and (first is None or trig < first):
        first = trig
    if first is None:
        return None
    cursor = 0
    while True:
        m = _GEMMA_BARE_TC_RE.search(content, cursor)
        if m is None or m.start() > first:
            return None
        if m.group(1) in enabled_tool_names:
            return None
        end = _gemma_body_brace_end(content, m.end() - 1)
        if end is None:
            cursor = m.end()
            continue
        if m.end() - 1 < first <= end:
            return end + 1
        cursor = end + 1


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
    enabled_tool_names: Optional[set] = None,
) -> list[dict]:
    """Return OpenAI-format tool calls, first-match wins so calls are never double-counted.

    ``allow_incomplete=True`` (default) heals truncated calls (missing close tag /
    unclosed parameter); ``False`` accepts only well-formed closed calls (trailing
    prose tolerated), matching llama-server's strict path when Auto-Heal is off.

    ``enabled_tool_names`` gates only the markerless Llama-3.2 bare-JSON form (the
    marker-based forms carry an explicit signal, so a disabled-tool name there is a
    real call attempt). ``None`` keeps the name-agnostic behaviour."""
    # Magistral reasoning is dropped BEFORE any parser dispatch: a rehearsed
    # call inside [THINK]...[/THINK] (in any format, e.g. a <function=...>
    # snippet while the real [TOOL_CALLS] follows the think block) must never
    # be promoted by an earlier pass. The display strip already drops the think
    # block; the parse path has to agree with it.
    content = _strip_mistral_reasoning(content)

    # A [TOOL_CALLS] call whose JSON arguments quote tool XML must win over the
    # literal: when the first XML signal sits inside a leading balanced Mistral
    # body it is argument data, so the Mistral parser runs first (mirrors the
    # leading-envelope guard in the DeepSeek/Kimi pre-pass on the follow-up
    # branch).
    if _xml_signal_inside_leading_mistral(content):
        calls = _parse_mistral_tool_calls(
            content, id_offset = id_offset, allow_incomplete = allow_incomplete
        )
        if calls:
            return calls

    # Same principle for a leading bare-JSON call: a string argument quoting
    # tool markup must stay data, so the bare-JSON parser takes the outer call
    # before the shared XML pass can promote the literal.
    if _xml_signal_inside_leading_bare_json(content):
        calls = _parse_llama3_bare_json(
            content, id_offset = id_offset, enabled_tool_names = enabled_tool_names
        )
        if calls:
            return calls
        # Disabled/example name: the leading object is ordinary content, so
        # the literals quoted inside it are data too. Drop the object and
        # parse only the tail -- a real call AFTER the object still parses,
        # while nothing inside it can be promoted by the passes below.
        i = 0
        while i < len(content) and content[i] in " \t\n\r":
            i += 1
        # The guard guarantees a balanced leading value (object or array).
        end = (_balanced_brace_end if content[i] == "{" else _balanced_bracket_end)(content, i)
        return parse_tool_calls_from_text(
            content[end + 1 :],
            id_offset = id_offset,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )

    # And for a leading wrapper-less Gemma call (markerless, so gated on an
    # enabled name): a quoted foreign literal inside its body is data, and
    # tool_healing would otherwise promote it before the Gemma fallback runs.
    if _signal_inside_leading_wrapperless_gemma(content, enabled_tool_names):
        calls = _parse_gemma_tool_calls(
            content,
            id_offset = id_offset,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )
        if calls:
            return calls

    # A DISABLED wrapper-less Gemma call is prose by design, so a tool
    # literal quoted inside it is data too: drop the span and parse the tail
    # (a real call after the example still parses).
    _prose_end = _disabled_gemma_call_end_containing_signal(content, enabled_tool_names)
    if _prose_end is not None:
        return parse_tool_calls_from_text(
            content[_prose_end:],
            id_offset = id_offset,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )

    # DeepSeek / Kimi use unique (often full-width) markers that do not collide
    # with the shared formats, so try them first -- unless a Qwen/Hermes
    # <tool_call> / <function=...> envelope opens before the first such marker, in
    # which case the marker is literal data inside the outer call's arguments and
    # the shared parser below must take the outer call.
    if not _marker_inside_leading_envelope(content):
        # Dispatch by earliest envelope opener: a raw parseable DeepSeek
        # example quoted inside a Kimi call's argument string (or vice versa)
        # must not hijack the turn just because of a fixed parser order.
        _ds = _DEEPSEEK_BEGIN_RE.search(content)
        _ds_pos = _ds.start() if _ds else len(content)
        _km_section = content.find(_KIMI_SECTION_BEGIN)
        _km_bare = content.find(_KIMI_CALL_BEGIN)
        _km_pos = min(p for p in (_km_section, _km_bare, len(content)) if p >= 0)
        pre_pass = [
            (_ds_pos, _parse_deepseek_tool_calls),
            (_km_pos, _parse_kimi_tool_calls),
        ]
        pre_pass.sort(key = lambda pair: pair[0])
        for _pos, parser in pre_pass:
            calls = parser(content, id_offset = id_offset, allow_incomplete = allow_incomplete)
            if calls:
                return calls

    # Qwen/Hermes, Qwen3.5 XML, and Gemma 4 go through the shared tool_healing
    # parser (strict/Auto-Heal contract + nested-marker, trailing-prose, and
    # ``<|"|>`` quoted-string handling the GGUF path relies on).
    calls = _tool_healing.parse_tool_calls_from_text(
        content,
        id_offset = id_offset,
        allow_incomplete = allow_incomplete,
    )
    if calls:
        return calls

    # Formats tool_healing does not cover: GLM shares the ``<tool_call>`` opener but
    # uses a bare name (no ``{``) so tool_healing skips it; _parse_function_xml adds
    # the ``<function name="...">`` form; the rest add Llama-3 and Mistral. These run
    # only after tool_healing finds nothing, so a strict-rejected call is never
    # re-healed here.
    for parser in (
        _parse_glm_tool_calls,  # GLM 4.x <tool_call>name
        _parse_function_xml,  # <function name="..."> attribute form
        _parse_llama3_python_tag,  # Llama-3 <|python_tag|>
        _parse_mistral_tool_calls,  # Mistral [TOOL_CALLS]
    ):
        calls = parser(content, id_offset = id_offset, allow_incomplete = allow_incomplete)
        if calls:
            return calls

    # Llama-3.2 bare ``{"name":..., "parameters":...}``. Strict (starts with ``{``
    # and parses to the right shape) so plain prose stays untouched. Runs before
    # the markerless Gemma scan: this form only ever matches a LEADING call
    # object, and document order says that call owns the turn -- an enabled
    # ``call:NAME{...}`` quoted inside its string arguments is data, while a
    # leading Gemma call is untouched (that content never starts with ``{``).
    calls = _parse_llama3_bare_json(
        content, id_offset = id_offset, enabled_tool_names = enabled_tool_names
    )
    if calls:
        return calls

    # Gemma wrapper-less ``call:NAME{...}`` (special tokens stripped). Markerless like
    # the Llama-3.2 bare-JSON form above, so it takes the same ``enabled_tool_names``
    # gate: a disabled/example name in prose must not be stolen as a call.
    return _parse_gemma_tool_calls(
        content,
        id_offset = id_offset,
        allow_incomplete = allow_incomplete,
        enabled_tool_names = enabled_tool_names,
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
        # Strict mode: a balanced JSON body that never closed its ``<tool_call>``
        # is a truncated call, not a finished one. Trailing prose after the close
        # is still tolerated (matches the GGUF strict path).
        if not allow_incomplete and not content[end + 1 :].lstrip().startswith("</tool_call>"):
            continue
        try:
            obj = json.loads(content[brace_start : end + 1])
        except (json.JSONDecodeError, ValueError):
            continue
        name = obj.get("name", "")
        # Accept both ``arguments`` (Hermes/Qwen) and ``parameters``
        # (Llama-3 template drift) so a fine-tune that swaps the key
        # keeps its payload instead of silently parsing to ``{}``.
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
    """Trim one wrapping newline the template adds around an XML parameter value
    (``<parameter=k>\nVALUE\n</parameter>``), preserving inner indentation.
    ``str.strip()`` destroyed code/diff indentation; SGLang's qwen3_coder trims only
    the wrapping newline."""
    if val.startswith("\n"):
        val = val[1:]
    if val.endswith("\n"):
        val = val[:-1]
    return val


def _inside_open_parameter(text: str, pos: int) -> bool:
    """True if ``pos`` sits inside an unclosed ``<parameter>``/``<param>`` block --
    i.e. a ``<function>`` / ``<parameter>`` opener at ``pos`` is a literal inside an
    argument value (e.g. code that prints tool-call XML), not a real nested call.
    Compares the last parameter opener before ``pos`` against the last
    parameter/function close before it."""
    last_param_open = -1
    for m in _TC_PARAM_START_RE.finditer(text, 0, pos):
        last_param_open = m.start()
    if last_param_open < 0:
        return False
    last_close = max(
        text.rfind("</parameter>", 0, pos),
        text.rfind("</param>", 0, pos),
        text.rfind("</function>", 0, pos),
    )
    return last_param_open > last_close


def _parse_function_xml(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    out: list[dict] = []
    # Skip ``<function ...>`` openers that are literals inside an open parameter
    # value (a code/search argument echoing tool-call XML), else the nested marker
    # is promoted to a second call and truncates the real argument.
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
        # Use the LAST </function> / </tool_call> within this call's window: a
        # literal close tag inside a code/search argument (e.g. print("</function>"))
        # appears before the real close, so first-match would truncate the argument.
        close_match = None
        for cm in _TC_END_TAG_RE.finditer(content, body_start, next_func):
            close_match = cm
        has_close = close_match is not None
        if has_close:
            body_end = close_match.start()
        else:
            body_end = min(len(content), next_func)
        # Strict mode: a function call that never reached its ``</function>`` /
        # ``</tool_call>`` close is truncated, so do not heal it into a call.
        if not allow_incomplete and not has_close:
            continue
        body = _TC_FUNC_CLOSE_RE.sub("", content[body_start:body_end])

        args: dict = {}
        param_unclosed = False
        # Same nested-literal guard for parameters: a ``<parameter>`` opener inside
        # an already-open parameter value is literal text, not a real second param.
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

        # Strict mode: every parameter must close with ``</parameter>`` /
        # ``</param>``; a dangling parameter means the call was cut off. A closed
        # call with no parameters is a valid zero-argument call (the function close
        # was already required above), so do not reject it.
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
    """One ``.call`` value (string/number/true/false/null) at ``body[p:]``.
    Returns ``(value, consumed_len)`` or ``(None, None)`` if none matches."""
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
        # Scientific notation (1e-3, -2E+4, 0.5e2) and decimals decode as float; a
        # bare integer stays int. ``"." in v`` alone missed the exponent forms and
        # truncated them to the mantissa (1e-3 -> 1).
        return (float(v) if any(c in v for c in ".eE") else int(v)), nm.end() - p
    lm = _LLAMA3_LIT_RE.match(body, p)
    if lm:
        return {"true": True, "false": False, "null": None}[lm.group(0)], lm.end() - p
    return None, None


def _parse_llama3_kv_args(body: str) -> dict[str, Any]:
    """``k=v, ...`` kwargs from a ``.call(...)`` body, left to right (later keys win).
    Linear hand-scan replacing the quadratic ``_LLAMA3_KV_RE.finditer`` walk."""
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
    """Parse the Llama-3 emissions: ``<|python_tag|>NAME.call(...)`` (built-in),
    ``<|python_tag|>{"name":..., "parameters":...}`` (custom), multi-call via
    ``; ``, ``parameters`` or ``arguments`` key."""
    out: list[dict] = []
    if _LLAMA3_PYTHON_TAG not in content:
        return out

    # 1. ``NAME.call(...)`` built-in form, anchored to ``<|python_tag|>`` and
    #    optionally ``; ``-chained within one emission. Anchoring each call to the
    #    tag boundary (not a free scan over the whole text) keeps a literal
    #    ``<|python_tag|>x.call(...)`` embedded in a JSON string argument of the
    #    custom form from being mistaken for a real built-in call.
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
            # Truncated ``.call(...)`` with no closing paren (depth > 0 at EOF):
            # reject in strict mode (Auto-Heal off) instead of executing a partial.
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

    # 2. ``<|python_tag|>{"name":..., "parameters":...}``. ``raw_decode``
    #    peels multiple ``; ``-separated objects from one emission.
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
                # Skip rather than fabricate ``{"value": args}`` when the
                # model emits a non-dict / non-string ``arguments`` value.
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                elif isinstance(args, str):
                    args_str = args
                else:
                    cursor = brace + end_offset
                    continue
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


# Llama-3 special-token sentinels (chainable, any order) plus the role label the
# template inserts between ``<|start_header_id|>`` and ``<|end_header_id|>``.
_LLAMA3_BARE_JSON_SENTINELS = (
    "<|begin_of_text|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
)
_LLAMA3_HEADER_ROLES = ("assistant", "user", "system", "tool", "ipython")


def strip_llama3_leading_sentinels(content: str) -> str:
    """Strip leading Llama-3 special-token sentinels (and the role label after
    ``<|start_header_id|>``) that can leak from a prior turn before a bare-JSON tool
    call. Shared by the parser and the streaming buffering guards so a
    sentinel-prefixed ``{"name":...}`` is recognised the same everywhere."""
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
    """Llama-3.2 ``custom_tools`` bare ``{"name":.., "parameters":{..}}`` (no ``<|python_tag|>``),
    strict so prose/echoes don't fire. ``enabled_tool_names`` gates on the parsed name so an
    ordinary JSON answer isn't misread as a call to a disabled tool; ``None`` is name-agnostic."""
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
        # Markerless JSON is ambiguous: only treat it as a call when the name is an
        # enabled tool, else it is an ordinary JSON answer (do not steal it).
        if enabled_tool_names is not None and name not in enabled_tool_names:
            break
        # ``parameters`` must be a dict (Llama-3 spec); ``arguments`` may be a dict
        # or JSON-string of one (OpenAI). Looser would fire on prose like
        # ``{"name":"x","parameters":"sentence"}``.
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
    """Parse all Mistral emissions: pre-v11 ``[TOOL_CALLS][...]`` / ``[TOOL_CALLS]{...}``
    and v11+ ``[TOOL_CALLS]name{json}`` / ``[TOOL_CALLS]name[ARGS]{json}``."""
    out: list[dict] = []
    content = _strip_mistral_reasoning(content)
    idx = content.find(_MISTRAL_TRIGGER)
    if idx < 0:
        return out

    # Disambiguate the first occurrence: array (pre-v11), single object
    # (pre-v11), or bare-name (v11+).
    j = idx + len(_MISTRAL_TRIGGER)
    k = j
    while k < len(content) and content[k] in " \t\n\r":
        k += 1
    if k >= len(content):
        return out

    if content[k] == "[":
        return _parse_mistral_array(content, k, id_offset, allow_incomplete = allow_incomplete)

    if content[k] == "{":
        # Pre-v11 single ``{"name":...}``; fall through if it doesn't
        # carry a ``name`` so v11+ handling still gets a chance.
        end = _balanced_brace_end(content, k)
        if end is not None:
            try:
                obj = json.loads(content[k : end + 1])
                if isinstance(obj, dict) and obj.get("name"):
                    _consume_mistral_call(content[k : end + 1], out, id_offset)
                    return out
            except (json.JSONDecodeError, ValueError):
                pass

    # v11+: walk every ``[TOOL_CALLS]``, parsing ``name{json}`` or
    # ``name[ARGS]{json}`` after each trigger.
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
    # An unclosed array (no matching ]) is a truncated call. In strict mode
    # (Auto-Heal off) reject it instead of recovering objects by hand below.
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
    # balanced ``{...}`` instead of re-scanning from every ``{`` (quadratic ReDoS).
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
    # Mistral uses ``arguments``; accept the ``parameters`` alias too (sibling paths
    # and SGLang's base detector alias it) so an array object keyed on it keeps args.
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


def _whole_content_is_json_value(text: str) -> bool:
    """True when the entire content is one valid JSON value (a structured
    answer, e.g. a response_format turn). Markerless scans must treat text
    inside it as data: an answer documenting an enabled tool's syntax must
    not execute that tool or have the example stripped from display."""
    t = text.strip()
    if t[:1] not in "{[":
        return False
    try:
        json.loads(t)
    except ValueError:
        return False
    return True


def _parse_gemma_tool_calls(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
    enabled_tool_names: Optional[set] = None,
) -> list[dict]:
    """Gemma 4: ``<|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>``.

    Also handles the ``skip_special_tokens`` stream where the ``<|tool_call>``
    wrapper and ``<|"|>`` string markers were stripped, leaving a bare
    ``call:NAME{k:v, ...}`` with unquoted values.

    ``enabled_tool_names`` (when given) gates on the parsed name, exactly like the
    markerless Llama-3.2 bare-JSON form: once the ``<|tool_call>`` wrapper is gone the
    ``call:NAME{...}`` shape is indistinguishable from prose that documents the Gemma
    tool syntax, so an example/disabled name (``call:foo{x:1}``) must not be stolen as
    a call -- that suppresses the real answer. ``None`` keeps the name-agnostic
    behaviour (unrestricted mode / direct callers)."""
    out: list[dict] = []
    # The wrapped ``<|tool_call>call:...<tool_call|>`` form -- including its
    # strict (Auto-Heal off) and nested-marker handling -- is owned by the
    # shared tool_healing parser, which runs before this fallback. Only the
    # wrapper-less ``call:NAME{...}`` stream (special tokens stripped) is left
    # for us, so defer content carrying an actual wrapped opener. Marker
    # LITERALS alone are not enough: a wrapper-less call whose argument merely
    # mentions ``<|tool_call>`` or ``<|"|>`` (a query about Gemma's own
    # syntax) has nothing tool_healing can parse, and deferring it would lose
    # the call entirely.
    if _GEMMA_TC_RE.search(content):
        return out
    # A whole-content JSON value is a structured ANSWER: a quoted example of
    # an enabled tool's syntax inside it must not become a real call.
    if _whole_content_is_json_value(content):
        return out
    # Manual cursor (not ``finditer``): after consuming a ``call:NAME{...}`` we must
    # resume scanning AFTER its balanced body, otherwise a nested ``call:OTHER{...}``
    # mentioned inside this call's own string argument (e.g. a query that quotes the
    # Gemma tool syntax) is re-matched and executed as a spurious extra tool call.
    cursor = 0
    while True:
        m = _GEMMA_BARE_TC_RE.search(content, cursor)
        if m is None:
            break
        name = m.group(1)
        body_start = m.end() - 1
        end = _gemma_body_brace_end(content, body_start)
        if end is None:
            # Unclosed/unbalanced call: nothing parseable follows (mirrors the
            # strip contract), and scanning on would promote a call quoted
            # inside the truncated call's own argument text.
            break
        # Resume past this call's balanced body so its arguments are never
        # rescanned for calls.
        cursor = end + 1
        # Markerless: only a call when the name is an enabled tool, else it is prose
        # (an example / disabled name) and must stay in the visible answer.
        if enabled_tool_names is not None and name not in enabled_tool_names:
            continue
        body = content[body_start + 1 : end]
        try:
            args = _gemma_parse_stripped_body(body)
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


def _gemma_body_brace_end(text: str, brace_pos: int) -> int | None:
    """Index of the ``}`` closing the wrapper-less Gemma body at ``brace_pos``.

    Unlike ``_balanced_brace_end`` (JSON), values here are raw after
    ``skip_special_tokens``, so single-quoted strings hide braces exactly like
    double-quoted ones (``code:print('}')``). Mirror the quote rules of
    ``_gemma_parse_stripped_body`` so the boundary always agrees with the body
    parser and a quoted ``}`` can never truncate the executed arguments.

    A single quote opens a string only at value-start context (right after
    ``:``, ``{``, ``[``, ``(``, ``,`` or ``=``): an apostrophe inside an
    unquoted value (``query:what's the weather``) is prose, and treating it
    as an opener would swallow the real closing brace and lose the whole
    call. A double quote also opens at the start of a word (after
    whitespace), so a quoted phrase mid-value (``query:find "a, b"``) hides
    its delimiters instead of splitting the value."""
    if brace_pos >= len(text) or text[brace_pos] != "{":
        return None
    depth = 0
    quote = ""
    prev = ""
    prev_raw = ""
    i = brace_pos
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'" and (prev in ":{[(,=" or (ch == '"' and prev_raw.isspace())):
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        if not ch.isspace():
            prev = ch
        prev_raw = ch
        i += 1
    return None


_BARE_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _top_level_bare_json_name(probe: str) -> Optional[str]:
    """TOP-LEVEL ``"name"`` (or ``"function"`` alias, name wins) of a bare-JSON object, else None.

    Skips nested objects/arrays so a nested ``"name"`` isn't mistaken for the call name; a
    truncated tail returns None so the caller keeps the text."""
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
            # End of the object with no top-level ``"name"``: fall back to a recorded
            # ``"function"`` alias if one was seen.
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
            # ``"function"`` is an accepted alias for the call name. Record a string
            # value but keep scanning: a top-level ``"name"`` still takes precedence.
            try:
                value, consumed = decoder.raw_decode(probe[i:])
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(value, str):
                function_value = value
            i += consumed
            continue
        # Skip a non-name top-level value; a truncated one means we cannot prove a
        # top-level name exists, so return None (keep the text).
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
    """Remove a leading (optionally sentinel-prefixed) Llama-3.2 bare-JSON call that
    ``strip_tool_markup`` (XML/bracket only) misses. Non-call text is returned unchanged;
    a truncated call collapses to ``""``. ``enabled_tool_names`` gates the strip like the
    parser: a name not in it stays as an ordinary JSON answer (``None`` keeps prior behaviour)."""
    probe = strip_llama3_leading_sentinels(text.lstrip())
    if not (probe.startswith("{") and ('"name"' in probe or '"function"' in probe)):
        return text
    if enabled_tool_names is not None:
        # Only suppress when the leading object is (or may be) a real call, i.e. its
        # TOP-LEVEL name is an enabled tool. A nested ``"name"`` (e.g.
        # {"result":{"name":"web_search",...}}) is ordinary data, not the call name,
        # so it must not gate the strip. An unknown / un-extractable name is kept.
        name = _top_level_bare_json_name(probe)
        if name not in enabled_tool_names:
            return text
    end = _balanced_brace_end(probe, 0)
    if end is None:
        return ""  # truncated bare-JSON call -- nothing recoverable
    return probe[end + 1 :].lstrip()


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


def _gemma_parse_value(
    text: str,
    i: int,
    *,
    in_mapping: bool = False,
):
    """Parse one Gemma arg value at ``i`` in a single O(n) forward pass; returns
    ``(value, next_index, closed)``. ``closed`` is False when a string/object/array
    runs off the end without its terminator, so the caller can fall back to raw.
    ``in_mapping`` applies the top-level rule that a comma only ends the value
    when a ``key:`` follows (array elements split on every top-level comma)."""
    if text.startswith(_GEMMA_STR_BEGIN, i):
        close = text.find(_GEMMA_STR_END, i + len(_GEMMA_STR_BEGIN))
        if close < 0:
            return text[i + len(_GEMMA_STR_BEGIN) :], len(text), False
        return text[i + len(_GEMMA_STR_BEGIN) : close], close + len(_GEMMA_STR_END), True
    if text[i] == "{":
        return _gemma_parse_mapping(text, i)
    if text[i] == "[":
        return _gemma_parse_array(text, i)
    if text[i] in "\"'":
        # Raw-quoted string (stripped stream): delimiters inside it are data,
        # so ``{city:"New, York"}`` is one value, not a split pair. Returned
        # unquoted, matching the top-level scalar coercion.
        quote = text[i]
        j = i + 1
        n = len(text)
        while j < n:
            if text[j] == "\\" and j + 1 < n:
                j += 2
                continue
            if text[j] == quote:
                return text[i + 1 : j], j + 1, True
            j += 1
        return text[i + 1 :], n, False
    # Primitive: number / true/false/null / bare identifier / unquoted code.
    # Same delimiter rules as the top-level value scan in
    # _gemma_parse_stripped_body: ``()``/``{}``/``[]`` depth and contextual
    # quote openers hide commas and closers inside the value (``f(1,2)``,
    # ``say "a, b"``), so nested arguments are not split into corrupted keys.
    end = i
    n = len(text)
    depth = 0
    quote = ""
    prev = ":"
    prev_raw = ":"
    while end < n and not text.startswith(_GEMMA_STR_BEGIN, end):
        ch = text[end]
        if quote:
            if ch == "\\" and end + 1 < n:
                end += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'" and (prev in ":{[(,=" or (ch == '"' and prev_raw.isspace())):
            quote = ch
        elif ch in "{[(":
            depth += 1
        elif ch in "}])":
            if depth == 0:
                break
            depth -= 1
        elif ch == "," and depth == 0:
            if not in_mapping or _GEMMA_KEY_RE.match(text, end + 1):
                break
        if not ch.isspace():
            prev = ch
        prev_raw = ch
        end += 1
    if end == i:
        # Stray ``}`` / ``]`` / ``,`` where a value was expected: consume one char
        # so every caller (e.g. _gemma_parse_array) always advances and can never
        # loop forever on malformed input.
        return "", i + 1, True
    raw = text[i:end].strip()
    if raw == "true":
        return True, end, True
    if raw == "false":
        return False, end, True
    if raw == "null":
        return None, end, True
    try:
        return int(raw), end, True
    except ValueError:
        pass
    try:
        return float(raw), end, True
    except ValueError:
        pass
    return raw, end, True


def _gemma_parse_array(text: str, start: int):
    """Parse a Gemma ``[...]`` array at ``text[start] == '['`` in one forward
    pass; returns ``(list, next_index, closed)``."""
    items: list[Any] = []
    i, n = start + 1, len(text)
    while i < n:
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i < n and text[i] == "]":
            return items, i + 1, True
        if i >= n:
            break
        v, i, _closed = _gemma_parse_value(text, i)
        items.append(v)
    return items, i, False


def _gemma_coerce_scalar(raw: str) -> Any:
    """Coerce an unquoted Gemma value to bool/int/float/None, else keep str.
    Quotes are stripped first so the loop's duplicate-call collapse treats quoted
    and unquoted variants of the same value as identical."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _gemma_strip_quoted_leaves(value: Any) -> Any:
    """Recursively unquote quoted string leaves of a nested stripped-stream
    value. The nested mapping/array parser keeps raw quote characters that the
    top level coerces away, so ``{loc:{city:"New York"}}`` must not hand the
    tool ``'"New York"'`` while a top-level ``city:"New York"`` hands it
    ``'New York'``."""
    if isinstance(value, str):
        v = value.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            return v[1:-1]
        return value
    if isinstance(value, dict):
        return {k: _gemma_strip_quoted_leaves(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_gemma_strip_quoted_leaves(v) for v in value]
    return value


def _gemma_parse_stripped_body(body: str) -> dict[str, Any]:
    """Parse a quote-less Gemma arg body ``key:value, key2:value2`` (the
    ``skip_special_tokens`` stream with ``<|"|>`` markers removed). Each value runs
    to the next top-level ``, key:`` boundary, tracking ``{}``/``[]``/``()`` depth so
    commas/braces inside a ``code`` / ``command`` value aren't truncated."""
    out: dict[str, Any] = {}
    i, n = 0, len(body)
    while i < n:
        m = _GEMMA_KEY_RE.match(body, i)
        if not m:
            break
        key = m.group(1)
        i = m.end()
        vstart = i
        depth = 0
        quote = ""
        # Quote openers mirror _gemma_body_brace_end: a single quote only at
        # value-start context (an apostrophe in ``query:what's up, n:3`` is
        # prose), a double quote also at the start of a word so a quoted
        # phrase mid-value (``query:find "a, b", n:3``) hides its delimiters.
        prev = ":"
        prev_raw = ":"
        while i < n:
            ch = body[i]
            if quote:
                # Inside a quoted value: honor escapes and only a matching quote closes
                # it, so a ``, key:`` shape inside the string (e.g. a search query like
                # ``"weather, location: Boston"``) is not mistaken for a value boundary.
                if ch == "\\" and i + 1 < n:
                    i += 2
                    continue
                if ch == quote:
                    quote = ""
            elif ch in "\"'" and (prev in ":{[(,=" or (ch == '"' and prev_raw.isspace())):
                quote = ch
            elif ch in "{[(":
                depth += 1
            elif ch in "}])":
                if depth > 0:
                    depth -= 1
            elif ch == "," and depth == 0 and _GEMMA_KEY_RE.match(body, i + 1):
                break
            if not ch.isspace():
                prev = ch
            prev_raw = ch
            i += 1
        raw_val = body[vstart:i].strip()
        if raw_val[:1] in "{[":
            # Nested object/array in the wrapper-less stream: parse in one pass,
            # accepting it only when the single-pass parser both closed the value
            # and consumed all of it, so a truncated / malformed value falls back
            # to the raw string.
            parsed, end, closed = _gemma_parse_value(raw_val, 0)
            out[key] = (
                _gemma_strip_quoted_leaves(parsed)
                if (closed and end == len(raw_val))
                else _gemma_coerce_scalar(raw_val)
            )
        else:
            out[key] = _gemma_coerce_scalar(raw_val)
        if i < n and body[i] == ",":
            i += 1
    return out


def _gemma_parse_mapping(text: str, start: int):
    """Parse a Gemma ``{key:value, ...}`` mapping at ``text[start] == '{'`` in one
    forward pass; returns ``(dict, next_index, closed)`` (``closed`` True iff the
    matching ``}`` was reached)."""
    out: dict[str, Any] = {}
    i, n = start + 1, len(text)
    while i < n:
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i < n and text[i] == "}":
            return out, i + 1, True
        if i >= n:
            break
        if text.startswith(_GEMMA_STR_BEGIN, i):
            close = text.find(_GEMMA_STR_END, i + len(_GEMMA_STR_BEGIN))
            if close < 0:
                break
            key = text[i + len(_GEMMA_STR_BEGIN) : close]
            i = close + len(_GEMMA_STR_END)
        else:
            kstart = i
            while i < n and text[i] not in ":}":
                i += 1
            key = text[kstart:i].strip()
        while i < n and text[i] in " \t\n\r":
            i += 1
        if i < n and text[i] == ":":
            i += 1
        while i < n and text[i] in " \t\n\r":
            i += 1
        if i >= n:
            out[key] = None
            break
        if text[i] == "}":
            out[key] = None
            return out, i + 1, True
        v, i, _closed = _gemma_parse_value(text, i, in_mapping = True)
        out[key] = v
    return out, i, False


# ── DeepSeek R1 / V3 / V3.1 ─────────────────────────────────────────


def _find_outside_json_strings(text: str, needle: str, start: int) -> int:
    """Index of ``needle`` at or after ``start`` that lies OUTSIDE any JSON string
    (double-quoted, backslash-escaped), or -1. A sentinel/marker that legitimately
    appears inside a tool-call argument string (e.g. a query mentioning the DeepSeek
    envelope-end token) must not be mistaken for the real structural terminator."""
    i = start
    n = len(text)
    in_string = False
    esc = False
    while i < n:
        ch = text[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if text.startswith(needle, i):
            return i
        i += 1
    return -1


def _parse_deepseek_tool_calls(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """DeepSeek R1 / V3 / V3.1.

    R1:    ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\\n``\\`\\`\\`json\\n{...}\\n\\`\\`\\`<｜tool▁call▁end｜>...``
    V3.x:  ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}<｜tool▁call▁end｜>...``

    Mirrors llama.cpp's pre-autoparser ``common_chat_parse_deepseek_r1`` /
    ``_v3_1`` handling; tolerates the 5 opener variants llama.cpp keeps.
    """
    out: list[dict] = []
    begin = _DEEPSEEK_BEGIN_RE.search(content)
    if not begin:
        return out
    scan_start = begin.end()
    # Find the envelope-end OUTSIDE JSON strings: a call argument (search query,
    # code) may legitimately contain the literal <｜tool▁calls▁end｜> token, and a
    # raw find would truncate ``body`` before the balanced JSON closes, dropping the
    # whole valid call.
    end_pos = _find_outside_json_strings(content, _DEEPSEEK_END, scan_start)
    # Strict mode (Auto-Heal off): an envelope with no closing <｜tool▁calls▁end｜>
    # is truncated mid-stream; reject instead of healing out to EOF.
    if not allow_incomplete and end_pos < 0:
        return out
    scan_end = end_pos if end_pos >= 0 else len(content)
    body = content[scan_start:scan_end]

    # R1 path first: ``function<｜tool▁sep｜>NAME\n```json\n{...}\n```<｜tool▁call▁end｜>``.
    pos = 0
    while pos < len(body):
        fpos = body.find(_DEEPSEEK_R1_FUNC_MARKER, pos)
        if fpos < 0:
            break
        name_start = fpos + len(_DEEPSEEK_R1_FUNC_MARKER)
        nl = body.find("\n", name_start)
        if nl < 0:
            break
        if not body.startswith(_DEEPSEEK_R1_FENCE, nl):
            pos = name_start
            continue
        name = body[name_start:nl].strip()
        json_start = nl + len(_DEEPSEEK_R1_FENCE)
        # Walk a balanced ``{`` even if the trailing fence is truncated.
        if json_start >= len(body) or body[json_start] != "{":
            pos = json_start
            continue
        brace_end = _balanced_brace_end(body, json_start)
        if brace_end is None:
            break
        try:
            args = json.loads(body[json_start : brace_end + 1])
        except (json.JSONDecodeError, ValueError):
            pos = brace_end + 1
            continue
        if not isinstance(args, dict):
            pos = brace_end + 1
            continue
        # Closing ``` fence + ``<｜tool▁call▁end｜>`` must IMMEDIATELY follow the JSON
        # (after optional whitespace). An unbounded ``search`` would land on a LATER
        # call's terminator, and ``pos = close_m.end()`` below would then advance past
        # that valid call -- so a multi-call turn whose first call omits its fence
        # dropped the rest. When the immediate close is absent we heal this call (name
        # known, JSON balanced) and advance by just the JSON so the next call is still
        # scanned; strict mode rejects it instead. Both keep later well-formed calls,
        # matching the V3/V3.1 and Kimi parsers.
        after = brace_end + 1
        while after < len(body) and body[after] in " \t\r\n":
            after += 1
        close_m = _DEEPSEEK_R1_CLOSE_RE.match(body, after)
        if not allow_incomplete and close_m is None:
            pos = brace_end + 1
            continue
        if name:
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
        pos = close_m.end() if close_m else brace_end + 1
    if out:
        return out

    # V3 / V3.1: name then bare JSON. Use ``str.find`` for the sep marker and walk
    # back for the name (a ``[^\n<]+`` regex search is O(N^2) on truncated bodies).
    pos = 0
    while pos < len(body):
        sep_pos = body.find(_DEEPSEEK_SEP, pos)
        if sep_pos < 0:
            break
        # Walk left from sep_pos to the name start; stop at ``\n`` (turn boundary),
        # ``<`` (tag start), or ``>`` (end of an optional ``<｜tool▁call▁begin｜>``).
        name_start = sep_pos
        while name_start > pos and body[name_start - 1] not in "\n<>":
            name_start -= 1
        name = body[name_start:sep_pos].strip()
        json_start = sep_pos + len(_DEEPSEEK_SEP)
        while json_start < len(body) and body[json_start] in " \t\n\r":
            json_start += 1
        if json_start >= len(body) or body[json_start] != "{":
            pos = sep_pos + len(_DEEPSEEK_SEP)
            continue
        brace_end = _balanced_brace_end(body, json_start)
        if brace_end is None:
            break
        # Strict mode (Auto-Heal off): a real V3 call closes with the per-call
        # terminator <｜tool▁call▁end｜>. Without it the call is truncated or merged
        # with the next, so reject it instead of executing on a bare balanced
        # object (the envelope-level <｜tool▁calls▁end｜> alone is not enough).
        if not allow_incomplete:
            after = brace_end + 1
            while after < len(body) and body[after] in " \t\r\n":
                after += 1
            if not body.startswith(_DEEPSEEK_CALL_END, after):
                # Strict mode: this call is missing its <｜tool▁call▁end｜> terminator
                # (truncated or merged). Skip it but keep scanning so a LATER
                # well-formed call is still returned, matching the Kimi strict parser's
                # recovery instead of dropping the rest of the envelope.
                pos = brace_end + 1
                continue
        try:
            args = json.loads(body[json_start : brace_end + 1])
        except (json.JSONDecodeError, ValueError):
            pos = brace_end + 1
            continue
        if not isinstance(args, dict):
            pos = brace_end + 1
            continue
        if name:
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
        # Advance just past this call's JSON; the loop re-locates the next
        # <｜tool▁sep｜>. Searching forward for the optional <｜tool▁call▁end｜>
        # could land on a LATER call's end marker and skip the call between.
        pos = brace_end + 1
    return out


# ── GLM 4.5 / 4.6 / 4.7 ─────────────────────────────────────────────


def _parse_glm_tool_calls(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """GLM 4.5 / 4.6 / 4.7.

    ``<tool_call>NAME[\\n]<arg_key>K</arg_key>[\\n]<arg_value>V</arg_value>
    ...</tool_call>``. Multi-call is back-to-back blocks, no envelope.
    Mirrors llama.cpp's GLM 4.x tool-call handling (``common_chat_params_init_glm_4_5``
    plus its generalized XML-style parser, llama.cpp PRs #15904 / #16932).
    """
    out: list[dict] = []
    pos = 0
    while pos < len(content):
        m = _GLM_TC_OPEN_RE.search(content, pos)
        if not m:
            break
        name = m.group(1).strip()
        apos = m.end()  # absolute position in ``content``; advances past each pair

        args: dict[str, Any] = {}
        valid = True
        close = -1
        # Walk arg_key/arg_value pairs directly against ``content`` (not a body
        # pre-bounded by the first </tool_call>): an arg value may legitimately
        # contain a literal </tool_call> -- e.g. ``print("</tool_call>")`` -- and each
        # <arg_value> is reliably delimited by its own </arg_value>, so the call's
        # real close is the </tool_call> that precedes the next <arg_key> (or ends
        # the block when no key follows). ``str.find`` keeps this linear; a lazy-group
        # ``finditer`` is O(N^2) when an unclosed body holds many bare ``<arg_key>``.
        while True:
            ks = content.find(_GLM_ARG_KEY_OPEN, apos)
            tc = content.find(_GLM_TC_CLOSE, apos)
            # </tool_call> before the next <arg_key> (or no more keys) ends the call;
            # a literal </tool_call> inside a value sits before ``apos`` already.
            if tc >= 0 and (ks < 0 or tc < ks):
                close = tc
                break
            if ks < 0:
                break  # no close and no more keys -- truncated body
            ke = content.find(_GLM_ARG_KEY_CLOSE, ks + len(_GLM_ARG_KEY_OPEN))
            if ke < 0:
                break
            vstart = ke + len(_GLM_ARG_KEY_CLOSE)
            while vstart < len(content) and content[vstart] in " \t\r\n":
                vstart += 1
            if not content.startswith(_GLM_ARG_VAL_OPEN, vstart):
                # A key with no <arg_value> tag: strict mode rejects the call
                # (same contract as an unclosed value below) instead of
                # executing it with the argument silently dropped; Auto-Heal
                # keeps the lenient skip.
                if not allow_incomplete:
                    valid = False
                apos = ke + len(_GLM_ARG_KEY_CLOSE)
                continue
            vs = vstart + len(_GLM_ARG_VAL_OPEN)
            # The value's real </arg_value> is the first whose next non-space token is
            # <arg_key> / </tool_call> / end: a value may legitimately contain a literal
            # </arg_value> (print("</arg_value>")) or </tool_call>, and a first-match
            # find would truncate it and execute the tool with corrupted arguments.
            ve = _glm_value_close(content, vs, strict = not allow_incomplete)
            key = content[ks + len(_GLM_ARG_KEY_OPEN) : ke].strip()
            if ve < 0:
                # Unclosed <arg_value>: strict mode rejects the whole call instead
                # of executing it with the argument silently dropped; with Auto-Heal
                # keep the partial value (a truncated query must not become a no-arg
                # call). Either way stop -- nothing valid follows an unclosed value.
                if not allow_incomplete:
                    valid = False
                else:
                    args[key] = content[vs:].rstrip()
                break
            raw_val = content[vs:ve]
            apos = ve + len(_GLM_ARG_VAL_CLOSE)
            # Template emits non-strings via ``tojson``, strings verbatim. Probe for
            # an unambiguous JSON literal; else keep the value RAW so whitespace in
            # string args (code, diffs) survives -- matches vLLM glm4_moe. A value
            # that starts with ``"`` is a verbatim string whose quotes are
            # meaningful (e.g. a search query); leaving ``"`` out of the probe keeps
            # those quotes instead of decoding them away.
            probe = raw_val.strip()
            if (
                probe[:1] in "{["
                or probe in ("true", "false", "null")
                or _GLM_JSON_NUMERIC_RE.fullmatch(probe)
            ):
                try:
                    args[key] = json.loads(probe)
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
            args[key] = raw_val

        # Strict mode (Auto-Heal off): a block with no </tool_call> close is
        # truncated; reject it instead of healing the body out to EOF.
        if not allow_incomplete and close < 0:
            valid = False

        if name and valid:
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
        pos = close + len(_GLM_TC_CLOSE) if close >= 0 else len(content)
    return out


# ── Kimi K2 / Moonshot ──────────────────────────────────────────────


def _parse_kimi_tool_calls(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Kimi K2.

    ``<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX
    <|tool_call_argument_begin|>{json}<|tool_call_end|>...
    <|tool_calls_section_end|>``. Full id is preserved on ``tool_calls
    [i].id`` for round-trip through the chat template. Outer loop walks
    every section in the stream (vLLM / SGLang parity); mirrors llama.cpp's
    Kimi K2 handling via its generalized XML-style parser (llama.cpp PR #16932).
    """
    out: list[dict] = []
    outer_pos = 0
    while True:
        section_start = content.find(_KIMI_SECTION_BEGIN, outer_pos)
        if section_start < 0:
            break
        scan_start = section_start + len(_KIMI_SECTION_BEGIN)
        # Find the section-end OUTSIDE JSON strings: a later call's argument (query,
        # code) may legitimately contain the literal <|tool_calls_section_end|> token,
        # and a raw find would truncate the body there, dropping the later valid call.
        section_end = _find_outside_json_strings(content, _KIMI_SECTION_END, scan_start)
        scan_end = section_end if section_end >= 0 else len(content)
        body = content[scan_start:scan_end]
        # Truncated tail: parse what we have, then exit. In strict mode a section
        # with no <|tool_calls_section_end|> is truncated; reject it instead.
        if section_end < 0:
            if allow_incomplete:
                out.extend(
                    _parse_kimi_section_body(
                        body, id_offset = id_offset + len(out), allow_incomplete = True
                    )
                )
            return out
        outer_pos = section_end + len(_KIMI_SECTION_END)
        out.extend(
            _parse_kimi_section_body(
                body, id_offset = id_offset + len(out), allow_incomplete = allow_incomplete
            )
        )

    # llama.cpp treats the ``<|tool_calls_section_begin|>`` wrapper as optional --
    # Kimi K2 can emit a bare ``<|tool_call_begin|>`` call. If the section loop
    # matched nothing but a bare call is present, parse all content as one section.
    if not out and _KIMI_CALL_BEGIN in content:
        out.extend(
            _parse_kimi_section_body(
                content, id_offset = id_offset, allow_incomplete = allow_incomplete
            )
        )
    return out


def _parse_kimi_section_body(
    body: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Parse one Kimi K2 section body (between begin / end markers)."""
    out: list[dict] = []
    pos = 0
    while pos < len(body):
        call_start = body.find(_KIMI_CALL_BEGIN, pos)
        if call_start < 0:
            break
        id_start = call_start + len(_KIMI_CALL_BEGIN)
        arg_begin = body.find(_KIMI_ARG_BEGIN, id_start)
        if arg_begin < 0:
            break
        full_id = body[id_start:arg_begin].strip()
        m = _KIMI_ID_RE.match(full_id)
        if m:
            # ``_KIMI_ID_RE`` already drops the ``functions.`` prefix and ``:idx``
            # suffix; group(1) is the whole name. Do NOT split on ``.`` -- a dotted
            # MCP name (functions.mcp.server-list:0) must stay ``mcp.server-list``.
            name = m.group(1)
        else:
            base = full_id.split(":")[0]
            name = base[len("functions.") :] if base.startswith("functions.") else base
        # Drop bare-counter ids (``3``, ``42``) -- matches vLLM; SGLang
        # infers name from tool schema, which we don't have here.
        if name.isdigit():
            json_start = arg_begin + len(_KIMI_ARG_BEGIN)
            brace_end = (
                _balanced_brace_end(body, json_start)
                if (json_start < len(body) and body[json_start] == "{")
                else None
            )
            if brace_end is None:
                pos = arg_begin + len(_KIMI_ARG_BEGIN)
            else:
                pos = brace_end + 1
            continue
        json_start = arg_begin + len(_KIMI_ARG_BEGIN)
        # Balanced brace lets truncated trailing ``<|tool_call_end|>``
        # still surface a call.
        while json_start < len(body) and body[json_start] in " \t\n\r":
            json_start += 1
        if json_start >= len(body) or body[json_start] != "{":
            pos = arg_begin + len(_KIMI_ARG_BEGIN)
            continue
        brace_end = _balanced_brace_end(body, json_start)
        if brace_end is None:
            # Malformed / truncated JSON: skip this call but keep parsing later
            # ones instead of dropping the rest of the section (vLLM recovers them).
            nxt = body.find(_KIMI_CALL_BEGIN, json_start)
            if nxt < 0:
                break
            pos = nxt
            continue
        try:
            args = json.loads(body[json_start : brace_end + 1])
        except (json.JSONDecodeError, ValueError):
            pos = brace_end + 1
            continue
        if not isinstance(args, dict):
            pos = brace_end + 1
            continue
        if not allow_incomplete:
            # Strict mode: this call must close with <|tool_call_end|> before the
            # next <|tool_call_begin|>; otherwise it is truncated, so reject it.
            end_marker = body.find(_KIMI_CALL_END, brace_end + 1)
            next_call = body.find(_KIMI_CALL_BEGIN, brace_end + 1)
            if end_marker < 0 or (next_call >= 0 and end_marker > next_call):
                pos = brace_end + 1
                continue
        if name:
            out.append(
                {
                    "id": full_id or f"call_{id_offset + len(out)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            )
        # Advance past this call's JSON; the loop re-locates the next
        # <|tool_call_begin|>. Searching forward for <|tool_call_end|> could
        # skip a following call when this one's end marker is missing.
        pos = brace_end + 1
    return out
