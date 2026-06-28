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

import json
import re
from typing import Any

# Qwen/Hermes, Qwen3.5 XML, and Gemma 4 are parsed by the shared
# ``core.tool_healing`` helper (also imported by external servers), which carries
# the strict/Auto-Heal (``allow_incomplete``) contract. This module adds the rest:
# Llama-3 ``<|python_tag|>``, Mistral ``[TOOL_CALLS]``, Llama-3.2 bare JSON.
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
    re.compile(r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>.*?</function>', re.DOTALL),
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\]\s*\[.*?\](?:\s*</s>)?", re.DOTALL),
    # Mistral v11+ ``[TOOL_CALLS]name{json}`` (may chain), close at ``}``.
    re.compile(r"\[TOOL_CALLS\]\s*[\w\.\-]+\s*(?:\[ARGS\])?\s*\{.*?\}", re.DOTALL),
    # DeepSeek R1 / V3 / V3.1: full envelope (any opener variant) ... end.
    re.compile(_DEEPSEEK_OPEN_RE_SRC + r".*?<｜tool▁calls▁end｜>", re.DOTALL),
    # Kimi K2: ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>``.
    re.compile(r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", re.DOTALL),
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
    # Gemma 4 wrapper-less ``call:NAME{...}`` (skip_special_tokens stripped
    # the ``<|tool_call>`` markers). Bounded to a single brace level so it
    # cleans the common query/command leak without eating unrelated prose.
    re.compile(r"(?<!\w)call:[\w\.\-]+\{[^{}]*\}", re.DOTALL),
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
# Qwen3.5 ``<function=name>...`` AND the attribute form
# ``<function name="name">...`` (MiniCPM-5, MiniMax-M2). Name class ``[\w\.\-]+``
# (hyphenated MCP / dotted names) lands in group(1) or group(2).
_TC_FUNC_START_RE = re.compile(r'<function(?:=([\w\.\-]+)|\s+name="([\w\.\-]+)")>\s*')
# Body ends at ``</tool_call>`` (Hermes) or ``</function>`` (Qwen3.5 / MiniCPM-5)
# so it stops at the close even when prose follows; without ``</function>``,
# trailing prose leaked into the last parameter value.
_TC_END_TAG_RE = re.compile(r"</(?:tool_call|function)>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
# Trailing class is horizontal whitespace only (``[^\S\n]*``, not ``\s*``) so the
# wrapping newline + value's first-line indentation survive; ``_trim_param_value``
# then trims one wrapping newline, preserving code indentation (SGLang qwen3_coder).
_TC_PARAM_START_RE = re.compile(
    r'<(?:parameter|param)(?:=([\w\.\-]+)|\s+name="([\w\.\-]+)")>[^\S\n]*'
)
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</(?:parameter|param)>\s*$")

# Llama-3 ``<|python_tag|>NAME.call(...)``.
_LLAMA3_PYTHON_TAG = "<|python_tag|>"
_LLAMA3_PY_CALL_RE = re.compile(
    r"<\|python_tag\|>\s*([\w\.\-]+)\s*\.\s*call\s*\(",
)
# Llama-3 ``.call(k=v, ...)`` kwarg tokens. Scanned by hand below (not finditer)
# to stay linear on a truncated/unterminated body; finditer retries every offset
# of a long word run / unterminated quote (quadratic ReDoS).
_LLAMA3_KEY_RE = re.compile(r"\w+")
_LLAMA3_WS_RE = re.compile(r"\s*")
_LLAMA3_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
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
_GLM_TC_OPEN_RE = re.compile(r"<tool_call>\s*([^\n<{][^\n<]*?)\s*(?=\n|<arg_key>|</tool_call>)")
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
_GEMMA_KEY_RE = re.compile(r"\s*([\w\.\-]+)\s*:")


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
    """Skip an optional ``[CALL_ID]<id>`` segment (Mistral Small 3.2); returns the
    next token pos (``[ARGS]`` / ``{``), or ``pos`` unchanged when absent."""
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
    """Drop a leading Magistral ``[THINK]...[/THINK]`` block so a ``[TOOL_CALLS]``
    inside the chain-of-thought is not taken as a real call (llama.cpp parses
    reasoning separately; see test-chat.cpp). Only the leading block is removed; an
    unclosed ``[THINK]`` (still streaming) drops everything from it onward."""
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
    return "".join(out)


def strip_tool_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call markup. ``final=False`` keeps in-progress markup buffered;
    ``final=True`` also drops trailing unclosed runs and trims."""
    text = _strip_mistral_closed_calls(text)
    pats = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in pats:
        text = pat.sub("", text)
    return text.strip() if final else text


def has_tool_signal(text: str) -> bool:
    return any(s in text for s in TOOL_XML_SIGNALS)


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Return OpenAI-format tool calls, trying each format and returning at the
    first match so we never double-count.

    ``allow_incomplete=True`` (default) heals truncated calls (missing close tag /
    unclosed parameter); ``False`` accepts only well-formed closed calls (trailing
    prose tolerated), matching llama-server's strict path when Auto-Heal is off."""
    # DeepSeek / Kimi use unique (often full-width) markers that do not collide
    # with the shared formats, so try them first.
    for parser in (
        _parse_deepseek_tool_calls,
        _parse_kimi_tool_calls,
    ):
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
        _parse_gemma_tool_calls,  # Gemma wrapper-less call:NAME{...} (tokens stripped)
    ):
        calls = parser(content, id_offset = id_offset, allow_incomplete = allow_incomplete)
        if calls:
            return calls

    # Llama-3.2 bare ``{"name":..., "parameters":...}``. Strict (starts with ``{``
    # and parses to the right shape) so plain prose stays untouched.
    return _parse_llama3_bare_json(content, id_offset = id_offset)


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


def _parse_function_xml(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    out: list[dict] = []
    func_starts = list(_TC_FUNC_START_RE.finditer(content))
    for idx, fm in enumerate(func_starts):
        # group(1) is ``<function=name>``, group(2) is ``<function name="...">``.
        func_name = fm.group(1) or fm.group(2)
        body_start = fm.end()
        next_func = func_starts[idx + 1].start() if idx + 1 < len(func_starts) else len(content)
        end_tag = _TC_END_TAG_RE.search(content[body_start:])
        has_close = end_tag is not None and (body_start + end_tag.start()) < next_func
        if has_close:
            body_end = body_start + end_tag.start()
        else:
            body_end = min(len(content), next_func)
        # Strict mode: a function call that never reached its ``</function>`` /
        # ``</tool_call>`` close is truncated, so do not heal it into a call.
        if not allow_incomplete and not has_close:
            continue
        body = _TC_FUNC_CLOSE_RE.sub("", content[body_start:body_end])

        args: dict = {}
        param_unclosed = False
        param_starts = list(_TC_PARAM_START_RE.finditer(body))
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
        # ``</param>``; a dangling parameter means the call was cut off.
        if not allow_incomplete and (param_unclosed or not param_starts):
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
        return (float(v) if "." in v else int(v)), nm.end() - p
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

    # 1. ``NAME.call(...)`` built-in form.
    for m in _LLAMA3_PY_CALL_RE.finditer(content):
        name = m.group(1)
        i = m.end()
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
            continue
        body = content[m.end() : i]
        args = _parse_llama3_kv_args(body)
        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )

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


def _parse_llama3_bare_json(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Llama-3.2 ``custom_tools``: bare ``{"name":..., "parameters":{...}}`` without
    ``<|python_tag|>``. Strict (starts with ``{`` after sentinel strip; ``name``
    non-empty; ``parameters``/``arguments`` a dict) so prose and echoes don't fire."""
    out: list[dict] = []
    stripped = content.lstrip()
    # Sentinels can chain in any order, so loop until none match.
    _sentinels = (
        "<|begin_of_text|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",
    )
    # Consume the role label Meta's template inserts between ``<|start_header_id|>``
    # and ``<|end_header_id|>`` so a round-trip like
    # ``<|start_header_id|>assistant<|end_header_id|>\n\n{json}`` reaches the body.
    _header_roles = ("assistant", "user", "system", "tool", "ipython")
    while True:
        stripped = stripped.lstrip()
        matched = False
        for sentinel in _sentinels:
            if stripped.startswith(sentinel):
                stripped = stripped[len(sentinel) :]
                if sentinel == "<|start_header_id|>":
                    for role in _header_roles:
                        if stripped.startswith(role):
                            stripped = stripped[len(role) :]
                            break
                matched = True
                break
        if not matched:
            break
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


def _parse_gemma_tool_calls(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list[dict]:
    """Gemma 4: ``<|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>``.

    Also handles the ``skip_special_tokens`` stream where the ``<|tool_call>``
    wrapper and ``<|"|>`` string markers were stripped, leaving a bare
    ``call:NAME{k:v, ...}`` with unquoted values.
    """
    out: list[dict] = []
    # The wrapped ``<|tool_call>call:...<tool_call|>`` form -- including its
    # strict (Auto-Heal off) and nested-marker handling -- is owned by the
    # shared tool_healing parser, which runs before this fallback. Only the
    # wrapper-less ``call:NAME{...}`` stream (special tokens stripped) is left
    # for us, so defer anything still carrying the wrapper or ``<|"|>`` markers.
    if "<|tool_call>" in content or _GEMMA_STR_BEGIN in content:
        return out
    for m in _GEMMA_BARE_TC_RE.finditer(content):
        name = m.group(1)
        body_start = m.end() - 1
        end = _balanced_brace_end(content, body_start)
        if end is None:
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


def _gemma_parse_value(text: str, i: int):
    """Parse one Gemma arg value at ``i``; returns ``(value, next_index, closed)``.

    Single forward pass: ``{}``/``[]`` are parsed in place (no separate
    balanced-brace pre-scan + re-walk), so nested values stay O(n) instead of
    re-scanning each subtree per level. ``closed`` is False when a string /
    object / array runs off the end of ``text`` without its terminator, so the
    caller can fall back to the raw scalar."""
    if text.startswith(_GEMMA_STR_BEGIN, i):
        close = text.find(_GEMMA_STR_END, i + len(_GEMMA_STR_BEGIN))
        if close < 0:
            return text[i + len(_GEMMA_STR_BEGIN) :], len(text), False
        return text[i + len(_GEMMA_STR_BEGIN) : close], close + len(_GEMMA_STR_END), True
    if text[i] == "{":
        return _gemma_parse_mapping(text, i)
    if text[i] == "[":
        return _gemma_parse_array(text, i)
    # Primitive: number / true/false/null / bare identifier.
    end = i
    while end < len(text) and text[end] not in ",}]" and not text.startswith(_GEMMA_STR_BEGIN, end):
        end += 1
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
    Surrounding quotes are stripped first so the loop's duplicate-call collapse
    treats quoted and unquoted variants of the same value as identical."""
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
        while i < n:
            ch = body[i]
            if ch in "{[(":
                depth += 1
            elif ch in "}])":
                if depth > 0:
                    depth -= 1
            elif ch == "," and depth == 0 and _GEMMA_KEY_RE.match(body, i + 1):
                break
            i += 1
        raw_val = body[vstart:i].strip()
        if raw_val[:1] in "{[":
            # Nested object/array in the wrapper-less stream: parse in one pass,
            # accepting it only when the single-pass parser both closed the value
            # and consumed all of it, so a truncated / malformed value falls back
            # to the raw string.
            parsed, end, closed = _gemma_parse_value(raw_val, 0)
            out[key] = parsed if (closed and end == len(raw_val)) else _gemma_coerce_scalar(raw_val)
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
        v, i, _closed = _gemma_parse_value(text, i)
        out[key] = v
    return out, i, False


# ── DeepSeek R1 / V3 / V3.1 ─────────────────────────────────────────


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
    end_pos = content.find(_DEEPSEEK_END, scan_start)
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
        # Move past the closing fence + ``<｜tool▁call▁end｜>``.
        close_m = _DEEPSEEK_R1_CLOSE_RE.search(body, brace_end + 1)
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
        body_start = m.end()
        close = content.find(_GLM_TC_CLOSE, body_start)
        # Strict mode (Auto-Heal off): a block with no </tool_call> close is
        # truncated; reject it instead of healing the body out to EOF.
        if not allow_incomplete and close < 0:
            break
        body_end = close if close >= 0 else len(content)
        body = content[body_start:body_end]

        args: dict[str, Any] = {}
        # Walk arg_key/arg_value pairs with ``str.find``; a lazy-group ``finditer``
        # is O(N^2) when an unclosed body holds many bare ``<arg_key>`` tokens.
        apos = 0
        while True:
            ks = body.find(_GLM_ARG_KEY_OPEN, apos)
            if ks < 0:
                break
            ke = body.find(_GLM_ARG_KEY_CLOSE, ks + len(_GLM_ARG_KEY_OPEN))
            if ke < 0:
                break
            vstart = ke + len(_GLM_ARG_KEY_CLOSE)
            while vstart < len(body) and body[vstart] in " \t\r\n":
                vstart += 1
            if not body.startswith(_GLM_ARG_VAL_OPEN, vstart):
                apos = ke + len(_GLM_ARG_KEY_CLOSE)
                continue
            vs = vstart + len(_GLM_ARG_VAL_OPEN)
            ve = body.find(_GLM_ARG_VAL_CLOSE, vs)
            if ve < 0:
                break
            key = body[ks + len(_GLM_ARG_KEY_OPEN) : ke].strip()
            raw_val = body[vs:ve]
            apos = ve + len(_GLM_ARG_VAL_CLOSE)
            # Template emits non-strings via ``tojson``, strings verbatim. Probe for
            # an unambiguous JSON literal; else keep the value RAW so whitespace in
            # string args (code, diffs) survives -- matches vLLM glm4_moe.
            probe = raw_val.strip()
            if (
                probe[:1] in '{["'
                or probe in ("true", "false", "null")
                or _GLM_JSON_NUMERIC_RE.fullmatch(probe)
            ):
                try:
                    args[key] = json.loads(probe)
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
            args[key] = raw_val

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
        section_end = content.find(_KIMI_SECTION_END, scan_start)
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
            name = m.group(1).split(".")[-1]
        else:
            name = full_id.split(":")[0].split(".")[-1]
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
