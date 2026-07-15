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

# Lazy annotations keep the standalone python 3.9 import working.
from __future__ import annotations

import json
import re
from typing import Any, Optional

# Qwen/Hermes, Qwen3.5 XML and Gemma 4 live in core.tool_healing; this module adds the rest.
from core import tool_healing as _tool_healing


# Flip the streaming buffer STREAMING->DRAINING so partial markup never leaks.
TOOL_XML_SIGNALS = (
    "<tool_call>",
    "<function=",
    '<function name="',
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<|tool_call>",
    # Bare reasoning-rehearsal marker (``name[ARGS]{...}``, no leading [TOOL_CALLS]);
    # keeps a rehearsed call held in the stream so it is promoted, not leaked as prose.
    "[ARGS]",
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
    # TML Inkling native call marker.
    "<|content_invoke_tool_json|>",
)


# DeepSeek opener variants; shared by parse and strip so a parsed signal is always stripped.
_DEEPSEEK_OPEN_ALT = (
    r"tool▁calls▁begin|tool_calls_begin|tool calls begin|tool\\_calls\\_begin|tool▁calls"
)
_DEEPSEEK_OPEN_RE_SRC = r"<｜(?:" + _DEEPSEEK_OPEN_ALT + r")｜>"

# Closed pairs only (mid-stream); _TOOL_ALL_PATS also eats unclosed tails at
# end-of-turn. ``[\w-]+`` on ``<function=...>`` tracks OpenAI's
# ``^[a-zA-Z0-9_-]{1,64}$`` so hyphenated MCP names parse like built-ins.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    # Span to the real ``</function>`` so a literal one inside a value can't truncate the strip.
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
    # Kimi K2 section-less closed call; else the catch-all below eats trailing prose to EOS.
    re.compile(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>.*$', re.DOTALL),
    # Bare-word markers drop a trailing truncated call only when a call-shaped start
    # follows; a prose mention (``See [TOOL_CALLS] docs...``) keeps its tail. Bare marker at EOF drops.
    re.compile(r"<\|tool_call>(?=\s*call\s*:|\s*$).*$", re.DOTALL),
    re.compile(
        r"\[TOOL_CALLS\](?=\s*(?:[\[{]|[A-Za-z_][\w.\-]*(?:[\[{]|\s*$))|\s*$).*$",
        re.DOTALL,
    ),
    re.compile(
        r"<\|python_tag\|>(?=\s*(?:\{|[A-Za-z_][\w.]*\()|\s*$).*$",
        re.DOTALL,
    ),
    # DeepSeek envelopes truncated mid-stream (any opener); same call-shaped lookahead as above.
    re.compile(
        _DEEPSEEK_OPEN_RE_SRC + r"(?=\s*(?:<｜tool▁call▁begin｜>|function)|\s*$).*$",
        re.DOTALL,
    ),
    re.compile(r"<｜tool▁call▁begin｜>(?=\s*function|\s*$).*$", re.DOTALL),
    # Kimi K2 envelope truncated.
    re.compile(
        r"<\|tool_calls_section_begin\|>(?=\s*<\|tool_call_begin\|>|\s*$).*$",
        re.DOTALL,
    ),
    re.compile(
        r"<\|tool_call_begin\|>(?=\s*[A-Za-z_][\w.\-]*:\d|\s*$).*$",
        re.DOTALL,
    ),
    # Gemma wrapper-less ``call:NAME{...}`` is handled by ``_strip_gemma_wrapperless_calls`` (enabled-name gate).
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

# The exact-args dup guard misses paraphrased re-searches, so also cap KB searches per turn.
RAG_MAX_SEARCHES_PER_TURN = 3
RAG_SEARCH_CAP_NUDGE = (
    "You have already searched the knowledge base several times this turn. "
    "Do not search again. Answer the question using the passages already "
    "retrieved above; if they do not contain the answer, say so plainly."
)


# ── Plan-without-action re-prompt (shared by the GGUF and safetensors loops) ──
# Forward-looking intent: the model says what it *will* do, not a final answer.
INTENT_SIGNAL = re.compile(
    r"(?i)("
    # Direct intent ("I'll", "Let me"); lookahead drops negated forms
    # ("I will not") so a refusal does not re-prompt.
    r"\b(i['\u2019](ll|m going to|m gonna)|i am (going to|gonna)|i will|i shall|let me|allow me)\b(?!\s+(?:not|never)\b)"
    r"|"
    # Step/plan framing: "First ...", "Step 1:", "Here's my plan"
    r"\b(?:first\b|step \d+:?|here['\u2019]?s (?:my |the |a )?(?:plan|approach))"
    r"|"
    r"\b(?:now i|next i)\b"
    r")"
)
# Matches GGUF's established default (llama_cpp.py has re-prompted up to 3
# times since #5620); safetensors and MLX inherit the same cap from here.
MAX_ACT_REPROMPTS = 3
REPROMPT_MAX_CHARS = 2000


def is_short_intent_without_action(text: str) -> bool:
    stripped = text.strip()
    return 0 < len(stripped) < REPROMPT_MAX_CHARS and INTENT_SIGNAL.search(stripped) is not None


def reprompt_to_act_message(tool_hint: str) -> str:
    """The user message appended when re-prompting a plan-without-action turn."""
    return (
        "You have access to enabled tools. If a tool is needed to satisfy "
        "the user's request or complete the action you described, call "
        f"{tool_hint} now. If no tool is needed, provide the final answer "
        "and follow the user's requested format."
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

# DeepSeek markers (full-width pipe U+FF5C, block U+2581); five outer-open variants like llama.cpp.
_DEEPSEEK_BEGIN_RE = re.compile(_DEEPSEEK_OPEN_RE_SRC)
_DEEPSEEK_END = "<｜tool▁calls▁end｜>"
_DEEPSEEK_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_DEEPSEEK_SEP = "<｜tool▁sep｜>"
_DEEPSEEK_CALL_END = "<｜tool▁call▁end｜>"
# R1 wraps args in a ```json fence with a ``function`` prefix; V3/V3.1 do not.
# Scanned with ``str.find`` -- the regex forms are O(N^2) on truncated bodies.
_DEEPSEEK_R1_FUNC_MARKER = "function" + _DEEPSEEK_SEP
_DEEPSEEK_R1_FENCE = "\n```json\n"
_DEEPSEEK_R1_CLOSE_RE = re.compile(r"```[\s\r\n]*" + re.escape(_DEEPSEEK_CALL_END))

# GLM 4.5-4.7: ``<tool_call>NAME[\n]<arg_key>K</arg_key>...``; the lookahead also allows a
# direct ``<arg_key>``/``</tool_call>`` (4.7 drops the newline, zero-arg calls close at once).
# Name class ``[\w.\-]+`` keeps prose like ``<tool_call>not a call</tool_call>`` unparsed;
# ``{`` stays with the Qwen JSON parser.
_GLM_TC_OPEN_RE = re.compile(r"<tool_call>\s*([\w.\-]+)\s*(?=\n|<arg_key>|</tool_call>)")
_GLM_TC_CLOSE = "</tool_call>"
_GLM_ARG_KEY_OPEN = "<arg_key>"
_GLM_ARG_KEY_CLOSE = "</arg_key>"
_GLM_ARG_VAL_OPEN = "<arg_value>"
_GLM_ARG_VAL_CLOSE = "</arg_value>"
# Strings arrive raw, non-strings via tojson; only unambiguous JSON literals decode
# (bare ``42``/``true``/``null`` stay strings).
_GLM_JSON_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# Kimi K2 / Moonshot (ASCII pipes). Id ``functions.NAME:IDX`` -- strip ``functions.``/``:N`` for the name.
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

# skip_special_tokens strips the wrapper and ``<|"|>`` markers, so streamed Gemma calls
# arrive as bare ``call:NAME{k:v, ...}``; ``(?<!\w)`` avoids ``recall:``.
_GEMMA_BARE_TC_RE = re.compile(r"(?<!\w)call\s*:\s*([\w\.\-]+)\s*\{")
# Partial leading prefix (``call``, ``call :``, ``call : name``) so the streaming buffer
# holds it instead of leaking visible text.
_GEMMA_BARE_TC_PREFIX_RE = re.compile(r"(?<!\w)call\s*(?::\s*[\w\.\-]*)?$")
# Keys start with a letter/underscore so ``10:00, 11:00`` in a value isn't misread as a new key.
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
    scanning (nested arguments are removed whole). ``enabled_tool_names`` gates the
    strip like the parser gate: a disabled/example name stays visible; ``None``
    strips every closed call."""
    if _whole_content_is_json_value(text):
        return text
    n = len(text)
    out = []
    # Mirror the parse scan: a leading JSON answer's span is data, kept visible.
    cursor = _leading_json_value_end(text) or 0
    if cursor:
        out.append(text[:cursor])
    while cursor < n:
        m = _GEMMA_BARE_TC_RE.search(text, cursor)
        if not m:
            out.append(text[cursor:])
            break
        disabled = enabled_tool_names is not None and m.group(1) not in enabled_tool_names
        brace = m.end() - 1  # _GEMMA_BARE_TC_RE consumes through the opening ``{``
        # Same boundary scanner as the parser: strip exactly what it consumed.
        end = _gemma_body_brace_end(text, brace)
        closed = end is not None
        next_index = (end + 1) if closed else len(text)
        if not closed:
            # Unclosed call: drop an enabled call to EOS; keep a disabled/example name as prose.
            out.append(text[cursor:] if disabled else text[cursor : m.start()])
            break
        if disabled:
            # Disabled/example name is prose: keep it whole.
            out.append(text[cursor:next_index])
        else:
            out.append(text[cursor : m.start()])
        cursor = next_index  # already past the matching ``}``
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


def _glm_value_close(
    text: str,
    vs: int,
    *,
    strict: bool = False,
) -> int:
    """Index of the ``</arg_value>`` that really ends the GLM value at ``vs``: the
    first one whose next non-space token is ``<arg_key>``, ``</tool_call>`` or
    end-of-text AND that sits at balanced quote state (an embedded literal pair
    like ``print("</arg_value></tool_call>")`` lives inside a still-open string).
    Quote openers are contextual (single quote only after punctuation, so
    apostrophes are prose; double quote also at word start), mirroring the Gemma
    scanners. If no candidate balances, the first token-valid one wins -- except
    in ``strict`` mode (Auto-Heal off), which refuses the in-quote fallback rather
    than execute truncated arguments. Returns -1 if unclosed."""
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
    """Strip GLM 4.x calls by scanning to each call's REAL ``</tool_call>`` (the one
    after the last consumed ``<arg_value>``, mirroring ``_parse_glm_tool_calls``), so
    a literal ``</tool_call>`` inside a value is data. Qwen ``<tool_call>{json}`` has
    no NAME token and is left to the regex arms. ``final`` drops a truncated call to
    EOS; otherwise it stays buffered."""
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
    ``final=True`` also drops trailing unclosed runs and trims.

    ``enabled_tool_names`` gates the name-conditioned forms so a disabled/example name in
    prose is kept (mirrors the parser gate): the bare reasoning-rehearsal ``name[ARGS]{...}``
    and the markerless Gemma ``call:NAME{...}`` strip. ``None`` strips every closed call.
    """
    if final:
        # Drop a leading Magistral ``[THINK]...[/THINK]`` at end-of-turn; its bracket
        # form is not the ``<think>`` the reasoning channel renders.
        text = _strip_mistral_reasoning(text)

    def _strip_segment(segment: str, is_last: bool) -> str:
        seg_final = final and is_last
        seg = _strip_mistral_closed_calls(segment)
        # Bare reasoning-rehearsal ``name[ARGS]{json}`` and the Mistral name form promote through
        # the shared balanced scan, so strip them the same way (any nesting depth removed whole).
        # The rehearsal arm is name-gated: an inactive ``foo[ARGS]{..}`` is prose and is kept.
        seg = _tool_healing._strip_bracket_tag_calls(seg, enabled_tool_names = enabled_tool_names)
        if seg_final:
            # Markerless Gemma ``call:NAME{...}`` (name-gated, mirrors the parse gate); end-of-turn only.
            seg = _strip_gemma_wrapperless_calls(seg, enabled_tool_names)
        # Scan-strip the function-XML form (parser-accurate: a literal ``<function=...>`` in a
        # value is data, not a call); the regex arms below cover the other formats.
        seg = _strip_function_xml_calls(seg, final = seg_final)
        # GLM 4.x: scan to the call's real </tool_call> so a literal one inside a value is data,
        # not a leak. Qwen <tool_call>{json} is left to the regex arms.
        seg = _strip_glm_calls(seg, final = seg_final)
        pats = _TOOL_ALL_PATS if seg_final else _TOOL_CLOSED_PATS
        for pat in pats:
            seg = pat.sub("", seg)
        if seg_final:
            # Drop a trailing partial bare rehearsal (``name[ARGS]`` with a truncated or absent
            # body) the balanced scan cannot close; gated so prose ``foo[ARGS] ...`` survives.
            seg = _tool_healing.apply_tool_strip_patterns(
                seg,
                [_tool_healing._REHEARSAL_TAIL_STRIP_RE],
                enabled_tool_names = enabled_tool_names,
            )
        return seg

    # ``<think>`` / ``[THINK]`` reasoning is preserved verbatim (a rehearsed call inside it is
    # not executed, so it must not be stripped from display either); a literal think marker
    # inside a real call's arguments is that call's data and is stripped with the call.
    result = _tool_healing.strip_outside_think(text, _strip_segment)
    return result.strip() if final else result


def has_tool_signal(text: str) -> bool:
    return any(s in text for s in TOOL_XML_SIGNALS)


# A Qwen/Hermes ``<tool_call>``/``<function=...>`` envelope whose arguments carry literal
# DeepSeek/Kimi markers must parse as the OUTER call. Detect it opening before the first
# marker so the pre-pass skips it.
_EMBEDDED_MARKER_RE = re.compile(
    _DEEPSEEK_OPEN_RE_SRC + "|" + re.escape(_KIMI_SECTION_BEGIN) + "|" + re.escape(_KIMI_CALL_BEGIN)
)
# Covers ``<function=NAME>`` and the attribute form. ``<|python_tag|>`` is Llama-3's
# envelope too (built-in ``NAME.call(`` and custom ``{json}``), so a quoted DeepSeek/Kimi
# example is data; the call-shaped lookahead mirrors the ``_TOOL_ALL_PATS`` python_tag arm
# so a bare prose ``<|python_tag|>`` mention isn't treated as one.
_OUTER_ENVELOPE_OPEN_RE = re.compile(
    r'<tool_call>|<function(?:=|\s+name=")|<\|tool_call>'
    r"|<\|python_tag\|>(?=\s*(?:\{|[A-Za-z_][\w.]*\())"
)
# CLOSED outer envelopes, each spanning to its REAL final close so a literal
# ``</tool_call>``/``</function>`` inside a value is data. Wrapped Gemma counts too.
_OUTER_ENVELOPE_CLOSED_PATS = (
    re.compile(r"<tool_call>(?:(?!<tool_call>).)*</tool_call>", re.DOTALL),
    _TOOL_CLOSED_PATS[1],
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
)


def _marker_inside_leading_envelope(content: str, enabled_tool_names: Optional[set] = None) -> bool:
    first_marker = _EMBEDDED_MARKER_RE.search(content)
    if first_marker is None:
        return False
    # A leading bare-JSON or Mistral [TOOL_CALLS] call is an outer envelope too:
    # a DS/Kimi marker in its argument strings is data.
    i = 0
    n = len(content)
    while i < n and content[i] in " \t\n\r":
        i += 1
    if content.startswith("{", i):
        end = _balanced_brace_end(content, i)
        if end is not None and i < first_marker.start():
            name = _top_level_bare_json_name(content[i : end + 1])
            if name is not None and (enabled_tool_names is None or name in enabled_tool_names):
                # The closed leading call owns the turn: a marker inside it is argument
                # data, one after it a trailing example (same rule as the XML envelopes below).
                return True
            if name is not None and first_marker.start() <= end:
                # A disabled-name leading object is prose (can't own the turn), but a marker
                # inside its own strings stays data. A marker AFTER it falls through to the pre-pass.
                return True
    elif content.startswith(_MISTRAL_TRIGGER, i):
        end = _mistral_region_end(content, i)
        if end is not None and i < first_marker.start():
            return True
    # A closed outer call PRECEDING the first marker owns the turn; the pre-pass must
    # not steal a trailing example or argument data.
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
    # A marker still stands; any opener left in the residue is UNCLOSED. One before the
    # marker is a truncated outer call holding the marker as data: skip the pre-pass.
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
    """True when a parseable Mistral call is the first tool emission in document order: it owns the turn, so later XML (quoted in its arguments or in trailing prose) is not promoted over it. A signal BEFORE the trigger keeps normal order."""
    trig = content.find(_MISTRAL_TRIGGER)
    if trig < 0:
        return False
    first_xml = _first_foreign_tool_signal(content)
    if first_xml is not None and first_xml < trig:
        return False
    # Only plain prose precedes the trigger: a visible preface must not hand
    # the turn to a later XML literal (preamble-tolerant, like the
    # wrapperless-Gemma guard). Prose that merely mentions the marker has no
    # parseable region and keeps the normal order.
    return _mistral_region_end(content, trig) is not None


def _parse_bare_rehearsals(
    content: str,
    *,
    id_offset: int = 0,
    enabled_tool_names: Optional[set] = None,
) -> list[dict]:
    """Promote bare reasoning-rehearsal ``name[ARGS]{json}`` calls that a leading [TOOL_CALLS]
    owns-the-turn parse would miss. Only the ``rehearsal`` kind is taken (a Mistral
    ``[TOOL_CALLS]name[ARGS]{..}`` yields ``name`` and is not double-counted), and a rehearsal
    inside a ``<think>`` / ``[THINK]`` block is reasoning, so it is skipped."""
    out: list[dict] = []
    think_spans = _tool_healing._think_spans_outside_tool_markup(content)
    for start, end, kind, m in _tool_healing._iter_bracket_spans(
        content, enabled_tool_names = enabled_tool_names
    ):
        if kind != "rehearsal":
            continue
        if any(s <= start < e for s, e in think_spans):
            continue
        try:
            payload = json.loads(content[m.end() : end])
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": m.group(1), "arguments": json.dumps(payload)},
            }
        )
    return out


_ATTR_FUNC_OPEN_RE = re.compile(r'<function\s+name="')


def _first_foreign_tool_signal(content: str) -> int | None:
    """Offset of the first tool signal a non-envelope parser would fire on
    (XML forms plus ``<|python_tag|>``, which also runs before the Mistral parser)."""
    first = None
    for sig in ("<tool_call>", "<|tool_call>", "<function=", "<|python_tag|>"):
        p = content.find(sig)
        if p >= 0 and (first is None or p < first):
            first = p
    attr = _ATTR_FUNC_OPEN_RE.search(content)
    if attr is not None and (first is None or attr.start() < first):
        first = attr.start()
    # DeepSeek/Kimi markers are foreign to a JSON envelope too: a marker inside a leading
    # object routes through the same guard (and, if disabled, the drop-and-parse-the-tail
    # recursion, so a real call after the object is still reached).
    marker = _EMBEDDED_MARKER_RE.search(content)
    if marker is not None and (first is None or marker.start() < first):
        first = marker.start()
    return first


def _xml_signal_inside_leading_bare_json(content: str) -> bool:
    """True when the first foreign tool signal is a quoted literal inside a
    LEADING bare-JSON call object or JSON answer -- data, not a real call
    (sibling of ``_xml_signal_inside_leading_mistral``)."""
    i = 0
    n = len(content)
    while i < n and content[i] in " \t\n\r":
        i += 1
    if i >= n or content[i] not in "{[":
        return False
    if content[i] == "[":
        # A leading array is only ever a structured answer; its literals are data.
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
        # A NAMELESS object that parses as real JSON is a structured answer / envelope too:
        # quoted markup is data, and the decline path drops it and parses the tail.
        # Non-JSON braced prose keeps the old behaviour.
        try:
            json.loads(content[i : end + 1])
        except ValueError:
            return False
    first_xml = _first_foreign_tool_signal(content)
    # The Mistral trigger is foreign to a JSON envelope too (its parser runs first).
    trig = content.find(_MISTRAL_TRIGGER)
    if trig >= 0 and (first_xml is None or trig < first_xml):
        first_xml = trig
    # Inside the balanced body the signal is quoted data; after the closed object the
    # leading call still owns the turn (mirrors the leading-Mistral rule).
    return first_xml is not None and i < first_xml


def _signal_inside_leading_wrapperless_gemma(
    content: str, enabled_tool_names: Optional[set]
) -> bool:
    """True when the first foreign tool signal is a quoted literal inside (or
    after) a LEADING enabled wrapper-less Gemma call (sibling of the
    Mistral/bare-JSON leading guards). Markerless form, so gated on an enabled
    name (``None`` keeps the name-agnostic behaviour)."""
    first = _first_foreign_tool_signal(content)
    # The Mistral trigger is foreign to a Gemma call too (its parser runs first).
    trig = content.find(_MISTRAL_TRIGGER)
    if trig >= 0 and (first is None or trig < first):
        first = trig
    if first is None:
        return False
    # A preamble before ``call:NAME{...}`` is normal; what matters is an ENABLED balanced
    # call beginning before the first foreign signal.
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
        if m.end() - 1 < first <= end:
            return True
        # An enabled call that CLOSES before the signal still owns the turn (inside-or-after
        # rule, as for closed bare-JSON/Mistral envelopes), gated on an enabled name.
        return enabled_tool_names is not None and end < first


def _disabled_gemma_call_end_containing_signal(
    content: str, enabled_tool_names: Optional[set]
) -> int | None:
    """End offset (exclusive) of the earliest DISABLED wrapper-less Gemma call
    whose balanced body contains the first foreign signal, else None. A disabled
    name is prose, so the quoted literal is data: the caller drops the span and
    recurses on the tail. An ENABLED call defers to the enabled-call guard."""
    if enabled_tool_names is None:
        return None
    first = _first_foreign_tool_signal(content)
    # Mirror the enabled-call guard: the Mistral trigger is foreign here too.
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
    # Drop Magistral [THINK]...[/THINK] BEFORE dispatch: a rehearsed call inside it must
    # never be promoted, and the parse path must agree with the display strip.
    content = _strip_mistral_reasoning(content)

    # A leading bare-JSON value is decided FIRST: a string argument quoting tool markup
    # (XML or a Mistral trigger) must stay data, so the bare-JSON parser takes the outer
    # call before any other pass. Precedes the Mistral guard, whose preamble tolerance
    # would otherwise claim a trigger quoted inside the leading object.
    if _xml_signal_inside_leading_bare_json(content):
        calls = _parse_llama3_bare_json(
            content, id_offset = id_offset, enabled_tool_names = enabled_tool_names
        )
        if calls:
            return calls
        # Disabled/example name: the leading object is content. Drop it and parse the tail.
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

    # A leading enabled wrapper-less Gemma call is decided BEFORE the Mistral guard: its
    # body reads as prose to the preamble tolerance below, so a quoted [TOOL_CALLS] would
    # otherwise steal the turn.
    if _signal_inside_leading_wrapperless_gemma(content, enabled_tool_names):
        calls = _parse_gemma_tool_calls(
            content,
            id_offset = id_offset,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )
        if calls:
            return calls

    # A DISABLED wrapper-less Gemma call is prose: drop the span and parse the tail BEFORE
    # the Mistral guard, whose preamble tolerance would otherwise parse a quoted trigger.
    _prose_end = _disabled_gemma_call_end_containing_signal(content, enabled_tool_names)
    if _prose_end is not None:
        return parse_tool_calls_from_text(
            content[_prose_end:],
            id_offset = id_offset,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )

    # A [TOOL_CALLS] call that is the first tool emission owns the turn: XML quoted in its
    # arguments or trailing prose is not promoted over it, nor does a prose preface forfeit it.
    if _xml_signal_inside_leading_mistral(content):
        calls = _parse_mistral_tool_calls(
            content, id_offset = id_offset, allow_incomplete = allow_incomplete
        )
        if calls:
            # A bare rehearsal ``name[ARGS]{..}`` after the Mistral call is a peer tool call,
            # not foreign XML the owns-the-turn guard protects against: promote it too so a
            # Mistral call and a rehearsal in one message both parse.
            calls.extend(
                _parse_bare_rehearsals(
                    content,
                    id_offset = id_offset + len(calls),
                    enabled_tool_names = enabled_tool_names,
                )
            )
            return calls

    # DeepSeek/Kimi markers are unique, so try them first -- unless an outer envelope
    # opens before the first marker (then the marker is argument data).
    if not _marker_inside_leading_envelope(content, enabled_tool_names):
        # Dispatch by earliest opener so a quoted DS example inside a Kimi call (or vice
        # versa) can't hijack the turn via fixed parser order.
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

    # A leading MiniCPM/MiniMax attribute-form call owns the turn: tool_healing doesn't know
    # the <function name="..."> wrapper, so a <tool_call> quoted in its parameter would beat
    # the outer call. Any earlier signal keeps normal order.
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

    # Qwen/Hermes, Qwen3.5 XML, Gemma 4, plus Mistral [TOOL_CALLS] / bare rehearsal
    # ``name[ARGS]{json}`` use the shared tool_healing parser (strict/Auto-Heal contract +
    # nested-marker, trailing-prose, and ``<|"|>`` quoted-string handling the GGUF path
    # relies on). ``enabled_tool_names`` gates the ambiguous bare-rehearsal form so an
    # inactive ``foo[ARGS]{..}`` stays prose.
    calls = _tool_healing.parse_tool_calls_from_text(
        content,
        id_offset = id_offset,
        allow_incomplete = allow_incomplete,
        enabled_tool_names = enabled_tool_names,
    )
    if calls:
        return calls

    # Formats tool_healing does not cover; these run only after it finds
    # nothing, so a strict-rejected call is never re-healed here. Blank any
    # JSON/Gemma marker coverage first: markup inside a marker's span (even one
    # that failed to parse) is that call's data, not a sibling, so a nested
    # ``<function=...>`` / ``<|python_tag|>`` / ``[TOOL_CALLS]`` must not be promoted.
    fallback_content = content
    coverage = _tool_healing.marker_coverage(content)
    if coverage:
        chars = list(content)
        for cov_start, cov_end in coverage:
            for i in range(cov_start, min(cov_end, len(chars))):
                chars[i] = " "
        fallback_content = "".join(chars)
    for parser in (
        _parse_glm_tool_calls,  # GLM 4.x <tool_call>name
        _parse_function_xml,  # <function name="..."> attribute form
        _parse_llama3_python_tag,  # Llama-3 <|python_tag|>
        _parse_mistral_tool_calls,  # Mistral [TOOL_CALLS]
    ):
        calls = parser(fallback_content, id_offset = id_offset, allow_incomplete = allow_incomplete)
        if calls:
            return calls

    # Llama-3.2 bare ``{"name":..., "parameters":...}`` (strict shape). Only a LEADING call
    # object matches and owns the turn, so an enabled ``call:NAME{...}`` in its arguments
    # stays data (Gemma never starts ``{``).
    calls = _parse_llama3_bare_json(
        content, id_offset = id_offset, enabled_tool_names = enabled_tool_names
    )
    if calls:
        return calls

    # Gemma wrapper-less ``call:NAME{...}``: markerless, so the same enabled-name gate applies.
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
        # Accept ``arguments`` (Hermes/Qwen) and ``parameters`` (Llama-3 drift).
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
    # The parameter's OWN close tag decides: while it closes after ``pos`` the position is
    # argument data, even across several literal function closes. Only an unclosed
    # parameter (heal mode) falls back to the first function close.
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
    # else the nested marker is promoted to a second call and truncates the real argument.
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
        # The call ends at the FIRST </function> / </tool_call> not inside an open
        # parameter: a literal close in a code/search argument is skipped as data, and
        # prose after the real close isn't folded into the last argument (mirrors
        # _strip_function_xml_calls and tool_healing._func_close_index).
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
        # Strict mode: an unclosed function call is truncated -- do not heal it.
        if not allow_incomplete and not has_close:
            continue
        body = _TC_FUNC_CLOSE_RE.sub("", content[body_start:body_end])

        args: dict = {}
        param_unclosed = False
        # A ``<parameter>`` opener inside an open parameter value is literal text.
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

        # Strict mode: a dangling parameter means the call was cut off; a closed
        # zero-parameter call stays valid.
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
        # Scientific notation (1e-3, -2E+4, 0.5e2) and decimals decode as float; a bare
        # integer stays int. ``"." in v`` alone missed the exponent forms (1e-3 -> 1).
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

    # 1. ``NAME.call(...)`` built-in form, anchored to ``<|python_tag|>`` and optionally
    #    ``; ``-chained within one emission. Anchoring to the tag boundary (not a free scan)
    #    keeps a literal ``<|python_tag|>x.call(...)`` quoted in a custom-form JSON argument
    #    from being mistaken for a real built-in call.
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
            # Truncated ``.call(...)`` with no closing paren: reject in strict mode
            # instead of executing a partial.
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

    # 2. ``<|python_tag|>{"name":..., "parameters":...}``. ``raw_decode`` peels multiple
    #    ``; ``-separated objects from one emission.
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
                # Skip rather than fabricate ``{"value": args}`` for a non-dict/non-string value.
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
        # Markerless JSON is ambiguous: treat it as a call only when the name is an enabled
        # tool, else it is an ordinary JSON answer.
        if enabled_tool_names is not None and name not in enabled_tool_names:
            break
        # ``parameters`` must be a dict (Llama-3 spec); ``arguments`` may be a dict or
        # JSON-string of one (OpenAI). Looser would fire on ``{"name":"x","parameters":"sentence"}``.
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

    # Disambiguate the first occurrence: array / single object (pre-v11), or bare-name (v11+).
    j = idx + len(_MISTRAL_TRIGGER)
    k = j
    while k < len(content) and content[k] in " \t\n\r":
        k += 1
    if k >= len(content):
        return out

    if content[k] == "[":
        return _parse_mistral_array(content, k, id_offset, allow_incomplete = allow_incomplete)

    if content[k] == "{":
        # Pre-v11 single ``{"name":...}``; fall through without a ``name`` so v11+ still runs.
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
    # An unclosed array (no matching ]) is a truncated call. In strict mode reject it
    # instead of recovering objects by hand below.
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

    # Healing path for unclosed arrays: walk top-level objects, advancing past each balanced
    # ``{...}`` instead of re-scanning from every ``{`` (quadratic ReDoS).
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
    # Mistral uses ``arguments``; accept the ``parameters`` alias too (sibling paths and
    # SGLang's base detector alias it) so an array object keyed on it keeps args.
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


def _leading_json_value_end(text: str) -> int | None:
    """End index (exclusive) of a balanced LEADING JSON value that parses as
    JSON: a structured answer possibly followed by prose. Markerless scans treat
    its contents as data (extends ``_whole_content_is_json_value``); leading-keyed,
    so a JSON blob mid-prose is not an answer span."""
    i = 0
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] not in "{[":
        return None
    end = (_balanced_brace_end if text[i] == "{" else _balanced_bracket_end)(text, i)
    if end is None:
        return None
    try:
        json.loads(text[i : end + 1])
    except ValueError:
        return None
    return end + 1


def _parse_gemma_tool_calls(
    content: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
    enabled_tool_names: Optional[set] = None,
) -> list[dict]:
    """Gemma 4: ``<|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>``, plus the
    ``skip_special_tokens`` stream where the wrapper and string markers were
    stripped (bare ``call:NAME{k:v, ...}``).

    ``enabled_tool_names`` gates on the parsed name: the wrapper-less shape is
    indistinguishable from prose documenting the syntax, so a disabled/example
    name must not be stolen as a call. ``None`` keeps the name-agnostic behaviour."""
    out: list[dict] = []
    # The WRAPPED form (strict + nested-marker handling) is tool_healing's, which runs
    # first: defer content with a wrapped opener. A marker literal alone is not enough --
    # a wrapper-less call mentioning ``<|tool_call>`` would be lost if deferred.
    if _GEMMA_TC_RE.search(content):
        return out
    # A whole-content JSON value is a structured answer: quoted examples must not become calls.
    if _whole_content_is_json_value(content):
        return out
    # Manual cursor: resume AFTER each consumed balanced body so a nested ``call:OTHER{...}``
    # in an argument is never re-matched. A leading JSON answer's span is data -- scan after it.
    cursor = _leading_json_value_end(content) or 0
    while True:
        m = _GEMMA_BARE_TC_RE.search(content, cursor)
        if m is None:
            break
        name = m.group(1)
        body_start = m.end() - 1
        end = _gemma_body_brace_end(content, body_start)
        if end is None:
            # Unclosed call: nothing parseable follows (mirrors the strip contract);
            # scanning on would promote quoted argument text.
            break
        cursor = end + 1
        # Markerless: a disabled/example name is prose, not a call.
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

    Values are raw after ``skip_special_tokens``, so quoted strings (single or
    double) hide braces; the quote rules mirror ``_gemma_parse_stripped_body`` so
    the boundary always agrees with the body parser. Contextual openers: a single
    quote opens only at value-start context (after ``:{[(,=`` -- apostrophes in
    ``what's the weather`` are prose), a double quote also at word start (so
    ``query:find "a, b"`` hides its delimiters)."""
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
            # End of the object with no top-level ``"name"``: fall back to a recorded ``"function"`` alias.
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
            # ``"function"`` aliases the call name. Record it but keep scanning: a top-level
            # ``"name"`` still wins.
            try:
                value, consumed = decoder.raw_decode(probe[i:])
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(value, str):
                function_value = value
            i += consumed
            continue
        # Skip a non-name top-level value; a truncated one can't prove a top-level name
        # exists, so return None (keep the text).
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
            # Only suppress when the leading object's TOP-LEVEL name is an enabled tool. A
            # nested ``"name"`` (e.g. {"result":{"name":"web_search",...}}) is data, not the
            # call name, so it must not gate the strip. An un-extractable name is kept.
            name = _top_level_bare_json_name(probe)
            if name not in enabled_tool_names:
                return probe.lstrip() if stripped_any else text
        end = _balanced_brace_end(probe, 0)
        if end is None:
            return ""  # truncated bare-JSON call -- nothing recoverable
        # A closed object must have the CALL SHAPE the parser accepts (dict ``parameters``,
        # or dict / JSON-string ``arguments``). An ordinary JSON answer like
        # {"name":"web_search","result":"no call"} is content, so the strip keeps it visible.
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
    # The parser requires a TOP-LEVEL name; a nested one (e.g. in a "result" value of an
    # ordinary JSON answer) is data, and stripping it name-agnostically would delete content.
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
        # Raw-quoted string: delimiters inside are data (``{city:"New, York"}`` is one
        # value); returned unquoted like the top-level scalar coercion.
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
    # Primitive / unquoted code: same delimiter rules as the top-level scan (bracket depth
    # + contextual quote openers hide commas and closers).
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
        # Stray delimiter where a value was expected: consume one char so callers always
        # advance (no infinite loop on malformed input).
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
    """Coerce an unquoted Gemma value to bool/int/float/None, else keep str
    (quotes stripped first so quoted/unquoted variants compare identical)."""
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
    """Recursively unquote quoted string leaves of a nested stripped-stream value,
    so nested ``city:"New York"`` matches the top-level coercion (no stray quotes)."""
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
        # Contextual quote openers mirror _gemma_body_brace_end.
        prev = ":"
        prev_raw = ":"
        while i < n:
            ch = body[i]
            if quote:
                # A ``, key:`` shape inside the quoted string is not a boundary.
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
            # Nested object/array: accept only a fully consumed, closed parse; a
            # truncated/malformed value falls back to the raw string.
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
    """Index of ``needle`` at/after ``start`` OUTSIDE any JSON string, or -1: a
    marker inside an argument string must not be taken as the structural terminator."""
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
    # Envelope end OUTSIDE JSON strings: an argument may contain the literal end token,
    # and a raw find would truncate the call.
    end_pos = _find_outside_json_strings(content, _DEEPSEEK_END, scan_start)
    # Strict mode: an unclosed envelope is truncated; reject, don't heal to EOF.
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
        # The closing fence + <｜tool▁call▁end｜> must IMMEDIATELY follow the JSON, else an
        # unbounded search lands on a LATER call's terminator. Absent close: heal past the
        # JSON (strict rejects); later well-formed calls are still kept.
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
        # Walk left from sep_pos to the name start; stop at ``\n`` (turn boundary), ``<``
        # (tag start), or ``>`` (end of an optional ``<｜tool▁call▁begin｜>``).
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
        # Strict mode: a real V3 call closes with the per-call <｜tool▁call▁end｜>; without
        # it the call is truncated/merged, so skip it but keep scanning for a later
        # well-formed call (matches Kimi strict).
        if not allow_incomplete:
            after = brace_end + 1
            while after < len(body) and body[after] in " \t\r\n":
                after += 1
            if not body.startswith(_DEEPSEEK_CALL_END, after):
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
        # Advance just past the JSON; seeking the optional <｜tool▁call▁end｜> could land on
        # a LATER call's end marker and skip the call between.
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
        # Walk arg pairs directly against ``content``: a value may contain a literal
        # </tool_call>, so the real close is the </tool_call> before the next <arg_key>.
        # ``str.find`` keeps this linear.
        while True:
            ks = content.find(_GLM_ARG_KEY_OPEN, apos)
            tc = content.find(_GLM_TC_CLOSE, apos)
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
                # Key without <arg_value>: strict rejects the call; Auto-Heal skips it.
                if not allow_incomplete:
                    valid = False
                apos = ke + len(_GLM_ARG_KEY_CLOSE)
                continue
            vs = vstart + len(_GLM_ARG_VAL_OPEN)
            # A first-match find on </arg_value> would truncate values containing literal
            # close tags and execute corrupted arguments.
            ve = _glm_value_close(content, vs, strict = not allow_incomplete)
            key = content[ks + len(_GLM_ARG_KEY_OPEN) : ke].strip()
            if ve < 0:
                # Unclosed <arg_value>: strict rejects the whole call; Auto-Heal keeps the
                # partial value (a truncated query is not a no-arg call).
                if not allow_incomplete:
                    valid = False
                    break
                # Bound the healed value at the next structural tag, not EOF, so a value
                # missing only its </arg_value> can't swallow the markup after it.
                nk = content.find(_GLM_ARG_KEY_OPEN, vs)
                tc = content.find(_GLM_TC_CLOSE, vs)
                bounds = [b for b in (nk, tc) if b >= 0]
                if not bounds:
                    args[key] = content[vs:].rstrip()
                    break
                bound = min(bounds)
                args[key] = content[vs:bound].rstrip()
                apos = bound
                continue
            raw_val = content[vs:ve]
            apos = ve + len(_GLM_ARG_VAL_CLOSE)
            # Decode only unambiguous JSON literals; else keep the value RAW so whitespace
            # in string args survives (matches vLLM glm4_moe). ``"`` is left out of the
            # probe: a verbatim string's quotes are meaningful.
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

        # Strict mode: a block with no </tool_call> is truncated; reject it.
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
        # Section end OUTSIDE JSON strings: an argument may contain the literal end token,
        # and a raw find would drop the later valid call.
        section_end = _find_outside_json_strings(content, _KIMI_SECTION_END, scan_start)
        scan_end = section_end if section_end >= 0 else len(content)
        body = content[scan_start:scan_end]
        # Truncated tail: parse what we have, then exit. In strict mode a section with no
        # <|tool_calls_section_end|> is truncated; reject it instead.
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

    # The section wrapper is optional (llama.cpp): a bare <|tool_call_begin|> call parses
    # as one section when the loop matched nothing.
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
            # group(1) is the whole name; do NOT split on ``.`` -- a dotted MCP name stays intact.
            name = m.group(1)
        else:
            base = full_id.split(":")[0]
            name = base[len("functions.") :] if base.startswith("functions.") else base
        # Drop bare-counter ids (``3``, ``42``) -- matches vLLM; SGLang infers the name
        # from the tool schema, which we don't have here.
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
        # Balanced brace lets a truncated trailing end marker still surface a call.
        while json_start < len(body) and body[json_start] in " \t\n\r":
            json_start += 1
        if json_start >= len(body) or body[json_start] != "{":
            pos = arg_begin + len(_KIMI_ARG_BEGIN)
            continue
        brace_end = _balanced_brace_end(body, json_start)
        if brace_end is None:
            # Malformed / truncated JSON: skip this call but keep parsing later ones
            # instead of dropping the rest of the section (vLLM recovers them).
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
            # Strict mode: this call must close with <|tool_call_end|> before the next
            # <|tool_call_begin|>; otherwise it is truncated, so reject it.
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
        # Advance past the JSON; seeking <|tool_call_end|> could skip a following call
        # when this one's end marker is missing.
        pos = brace_end + 1
    return out
