# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Backend-neutral tool-call parser shared by GGUF, safetensors, and MLX.

Covers the emission formats so the safetensors + MLX agentic loop sees
the same call shape llama-server normalises for GGUF:

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

Closing tags / brackets are tolerated when missing because models
frequently truncate them mid-stream.
"""

import json
import re
from typing import Any


# Markers that flip the streaming buffer from STREAMING to DRAINING so
# partial markup never leaks before the parser sees it.
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


# Closed pairs only (mid-stream); _TOOL_ALL_PATS also eats unclosed
# tails for end-of-turn cleanup. ``[\w-]+`` on ``<function=...>`` tracks
# OpenAI's ``^[a-zA-Z0-9_-]{1,64}$`` so MCP tool names with hyphens
# (mcp__srv__list-issues) parse the same as the built-ins.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>.*?</function>', re.DOTALL),
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\]\s*\[.*?\](?:\s*</s>)?", re.DOTALL),
    # Mistral v11+ ``[TOOL_CALLS]name{json}`` (may chain), close at ``}``.
    re.compile(r"\[TOOL_CALLS\]\s*[\w\.\-]+\s*(?:\[ARGS\])?\s*\{.*?\}", re.DOTALL),
    # DeepSeek R1 / V3 / V3.1: full envelope ``<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>``.
    re.compile(r"<｜tool[▁_]calls[▁_]begin｜>.*?<｜tool▁calls▁end｜>", re.DOTALL),
    # Kimi K2: ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>``.
    re.compile(
        r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", re.DOTALL
    ),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>.*$', re.DOTALL),
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\].*$", re.DOTALL),
    re.compile(r"<\|python_tag\|>.*$", re.DOTALL),
    # DeepSeek envelopes truncated mid-stream.
    re.compile(r"<｜tool[▁_]calls[▁_]begin｜>.*$", re.DOTALL),
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

TOOL_ERROR_NUDGE = (
    "\n\nThe tool call encountered an issue. Please try a different "
    "approach or rephrase your request."
)

BUDGET_EXHAUSTED_NUDGE = (
    "You have used all available tool calls. Based on everything you "
    "have found so far, provide your final answer now. Do not call "
    "any more tools."
)


# Qwen / Hermes ``<tool_call>{json}``.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
# Qwen3.5 / Hermes ``<function=name><parameter=k>v`` AND the attribute
# form ``<function name="name"><param name="k">v`` used by MiniCPM-5,
# MiniMax-M2, etc. Name char class is ``[\w\.\-]+`` so MCP tool names
# with hyphens (mcp__srv__list-issues) and dotted module names parse
# the same as the built-ins. Name lands in group(1) or group(2).
_TC_FUNC_START_RE = re.compile(r'<function(?:=([\w\.\-]+)|\s+name="([\w\.\-]+)")>\s*')
# Body terminates at either ``</tool_call>`` (Hermes wrapper) OR
# ``</function>`` (Qwen3.5 / MiniCPM-5 standalone) so the parser stops
# at the close tag even when prose follows. Without ``</function>``,
# trailing prose leaked into the last parameter value.
_TC_END_TAG_RE = re.compile(r"</(?:tool_call|function)>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(
    r'<(?:parameter|param)(?:=([\w\.\-]+)|\s+name="([\w\.\-]+)")>\s*'
)
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</(?:parameter|param)>\s*$")

# Llama-3 ``<|python_tag|>NAME.call(...)``.
_LLAMA3_PYTHON_TAG = "<|python_tag|>"
_LLAMA3_PY_CALL_RE = re.compile(
    r"<\|python_tag\|>\s*([\w\.\-]+)\s*\.\s*call\s*\(",
)
_LLAMA3_KV_RE = re.compile(
    r"""(\w+)\s*=\s*(?:"((?:\\.|[^"\\])*)"|(-?\d+(?:\.\d+)?)|(true|false|null))""",
    re.VERBOSE,
)

# Mistral ``[TOOL_CALLS]`` trigger. v11+ chains them, each followed by
# a bare name plus ``{json}`` (Magistral) or ``[ARGS]{json}`` (Ministral
# / Large 3).
_MISTRAL_TRIGGER = "[TOOL_CALLS]"
_MISTRAL_ARGS_MARKER = "[ARGS]"
# Mistral Small 3.2 emits ``name[CALL_ID]<id>[ARGS]{json}``; the call-id
# segment is absent on Ministral / Magistral. llama.cpp distinguishes the
# two on the presence of ``[CALL_ID]`` (common/chat.cpp).
_MISTRAL_CALL_ID_MARKER = "[CALL_ID]"
# Magistral wraps reasoning in ``[THINK] ... [/THINK]`` before the answer.
# A ``[TOOL_CALLS]`` inside that block is chain-of-thought, not a real call.
_MISTRAL_THINK_OPEN = "[THINK]"
_MISTRAL_THINK_CLOSE = "[/THINK]"
_MISTRAL_V11_NAME_RE = re.compile(r"\s*([\w\.\-]+)\s*")

# DeepSeek R1 / V3 / V3.1 markers (full-width pipe U+FF5C, lower-
# one-eighth-block U+2581). llama.cpp accepts five variants of the
# outer block-open; we mirror its tolerance.
_DEEPSEEK_BEGIN_RE = re.compile(
    r"<｜(?:tool▁calls▁begin|tool_calls_begin|tool calls begin|tool\\_calls\\_begin|tool▁calls)｜>"
)
_DEEPSEEK_END = "<｜tool▁calls▁end｜>"
_DEEPSEEK_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_DEEPSEEK_SEP = "<｜tool▁sep｜>"
_DEEPSEEK_CALL_END = "<｜tool▁call▁end｜>"
# R1 specifically wraps the args in a Markdown ```json ... ``` fence and
# prefixes the call with the literal ``function`` token; V3 / V3.1 do
# not. Detect R1 by the presence of ``function<｜tool▁sep｜>`` followed
# by ``\n```json``.
_DEEPSEEK_R1_FUNC_RE = re.compile(
    r"(?:"
    + re.escape(_DEEPSEEK_CALL_BEGIN)
    + r")?function"
    + re.escape(_DEEPSEEK_SEP)
    + r"([^\n]+)\n```json\n",
)
_DEEPSEEK_R1_CLOSE_RE = re.compile(r"```[\s\r\n]*" + re.escape(_DEEPSEEK_CALL_END))
# V3 / V3.1 (bare-JSON, no fence) is handled by ``str.find`` on the sep
# marker; see ``_parse_deepseek_tool_calls`` -- a ``([^\n<]+?)`` regex
# is O(N^2) on adversarial truncated bodies.

# GLM 4.5 / 4.6 / 4.7: ``<tool_call>NAME[\n]<arg_key>K</arg_key>...``.
# GLM 4.7 strips the ``\n`` after the name, so the lookahead also allows
# ``<arg_key>`` directly and ``</tool_call>`` for a zero-argument call
# (``<tool_call>get_current_date</tool_call>``). First-char ``[^\n<{]``
# excludes Qwen.
_GLM_TC_OPEN_RE = re.compile(
    r"<tool_call>\s*([^\n<{][^\n<]*?)\s*(?=\n|<arg_key>|</tool_call>)"
)
_GLM_TC_CLOSE = "</tool_call>"
_GLM_ARG_PAIR_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)
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
    """Index of `]` matching `[` at ``text[start]``; ignores brackets
    in JSON strings. None if unmatched."""
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
    """Skip an optional ``[CALL_ID]<id>`` segment (Mistral Small 3.2) at
    ``pos``. Returns the position of the next meaningful token (``[ARGS]``
    or ``{``), or ``pos`` unchanged when no ``[CALL_ID]`` is present."""
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
    """Drop a leading Magistral ``[THINK] ... [/THINK]`` reasoning block so a
    ``[TOOL_CALLS]`` emitted *inside* the chain-of-thought is not mistaken for
    a real call (llama.cpp parses reasoning separately; see test-chat.cpp).

    Only a leading block is removed -- the reasoning prefix is always first,
    so a literal ``[THINK]`` inside a later tool argument is left untouched.
    An unclosed leading ``[THINK]`` (still streaming) means nothing has been
    committed yet, so everything from it onward is dropped."""
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
    or ``name[ARGS]{json}``) via balanced brace/bracket scanning.

    A non-greedy ``\\{.*?\\}`` would truncate at the first ``}`` and lose
    nested JSON. Unclosed runs are left for ``final=True`` cleanup.
    """
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
    """Strip tool-call markup. ``final=False`` keeps in-progress
    markup buffered; ``final=True`` also drops trailing unclosed runs
    and trims."""
    text = _strip_mistral_closed_calls(text)
    pats = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in pats:
        text = pat.sub("", text)
    return text.strip() if final else text


def has_tool_signal(text: str) -> bool:
    return any(s in text for s in TOOL_XML_SIGNALS)


def parse_tool_calls_from_text(content: str, *, id_offset: int = 0) -> list[dict]:
    """Return OpenAI-format tool calls. First match wins.

    Order: DeepSeek / Kimi (full-width or unique markers, no collision)
    -&gt; Qwen JSON -&gt; GLM (shares ``<tool_call>`` opener with Qwen but
    Qwen needs ``\\s*{`` after the tag, GLM has a bare name) -&gt;
    Qwen3.5 XML -&gt; Llama-3 python_tag -&gt; Mistral -&gt; Gemma -&gt;
    Llama-3.2 bare JSON (strict ``{`` start).
    """
    for parser in (
        _parse_deepseek_tool_calls,
        _parse_kimi_tool_calls,
        _parse_tool_call_json,
        _parse_glm_tool_calls,
        _parse_function_xml,
        _parse_llama3_python_tag,
        _parse_mistral_tool_calls,
        _parse_gemma_tool_calls,
    ):
        calls = parser(content, id_offset = id_offset)
        if calls:
            return calls
    return _parse_llama3_bare_json(content, id_offset = id_offset)


def _parse_tool_call_json(content: str, *, id_offset: int) -> list[dict]:
    out: list[dict] = []
    for m in _TC_JSON_START_RE.finditer(content):
        brace_start = m.end() - 1
        end = _balanced_brace_end(content, brace_start)
        if end is None:
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


def _parse_function_xml(content: str, *, id_offset: int) -> list[dict]:
    out: list[dict] = []
    func_starts = list(_TC_FUNC_START_RE.finditer(content))
    for idx, fm in enumerate(func_starts):
        # group(1) is ``<function=name>``, group(2) is ``<function name="...">``.
        func_name = fm.group(1) or fm.group(2)
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
        body = _TC_FUNC_CLOSE_RE.sub("", content[body_start:body_end])

        args: dict = {}
        param_starts = list(_TC_PARAM_START_RE.finditer(body))
        if len(param_starts) == 1:
            pm = param_starts[0]
            val = _TC_PARAM_CLOSE_RE.sub("", body[pm.end() :])
            args[pm.group(1) or pm.group(2)] = val.strip()
        else:
            for pidx, pm in enumerate(param_starts):
                val_start = pm.end()
                next_param = (
                    param_starts[pidx + 1].start()
                    if pidx + 1 < len(param_starts)
                    else len(body)
                )
                val = _TC_PARAM_CLOSE_RE.sub("", body[val_start:next_param])
                args[pm.group(1) or pm.group(2)] = val.strip()

        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": func_name, "arguments": json.dumps(args)},
            }
        )
    return out


def _parse_llama3_python_tag(content: str, *, id_offset: int) -> list[dict]:
    """Parse the four Llama-3 emissions: ``<|python_tag|>NAME.call(...)``
    (built-in), ``<|python_tag|>{"name":..., "parameters":...}`` (custom),
    multi-call via ``; `` separators, ``parameters`` or ``arguments`` key.
    """
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
        body = content[m.end() : i]
        args: dict[str, Any] = {}
        for kv in _LLAMA3_KV_RE.finditer(body):
            k = kv.group(1)
            if kv.group(2) is not None:
                # ``json.loads`` on a quoted string handles \n/\t/\uXXXX
                # escapes correctly AND keeps literal UTF-8 bytes (emoji
                # / CJK) intact -- the older ``bytes.decode('unicode_escape')``
                # path mangled non-ASCII.
                try:
                    args[k] = json.loads('"' + kv.group(2) + '"')
                except (json.JSONDecodeError, ValueError):
                    args[k] = kv.group(2)
            elif kv.group(3) is not None:
                v = kv.group(3)
                args[k] = float(v) if "." in v else int(v)
            elif kv.group(4) is not None:
                args[k] = {"true": True, "false": False, "null": None}[kv.group(4)]
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
                args = (
                    obj.get("parameters")
                    if "parameters" in obj
                    else obj.get("arguments", {})
                )
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


def _parse_llama3_bare_json(content: str, *, id_offset: int) -> list[dict]:
    """Llama-3.2 ``custom_tools``: bare ``{"name":..., "parameters":{...}}``
    without ``<|python_tag|>``. Strict (must start with ``{`` after sentinel
    strip; ``name`` non-empty; ``parameters`` or ``arguments`` is a dict) so
    plain prose and tool-message echoes don't trigger."""
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
    # Role labels Meta's Llama-3 chat template inserts between
    # ``<|start_header_id|>`` and ``<|end_header_id|>`` -- consume so a
    # round-trip like
    # ``<|start_header_id|>assistant<|end_header_id|>\n\n{json}``
    # reaches the JSON body.
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
        # ``parameters`` must be a dict (Llama-3 spec).
        # ``arguments`` may be a dict or a JSON-string of a dict (OpenAI shape).
        # Anything looser would fire on prose like ``{"name":"x","parameters":"sentence"}``.
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


def _parse_mistral_tool_calls(content: str, *, id_offset: int) -> list[dict]:
    """Parse all Mistral emissions: pre-v11 ``[TOOL_CALLS][...]`` /
    ``[TOOL_CALLS]{...}`` and v11+ ``[TOOL_CALLS]name{json}`` /
    ``[TOOL_CALLS]name[ARGS]{json}`` (parallel-friendly)."""
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
        return _parse_mistral_array(content, k, id_offset)

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


def _parse_mistral_array(content: str, start: int, id_offset: int) -> list[dict]:
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
    body = content[start : j + 1] if depth == 0 else content[start:]

    try:
        arr = json.loads(body)
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict):
                    _consume_mistral_call(json.dumps(obj), out, id_offset)
        return out
    except (json.JSONDecodeError, ValueError):
        pass

    # Healing path for unclosed arrays: walk objects by hand.
    for m in re.finditer(r"\{", body):
        end = _balanced_brace_end(body, m.start())
        if end is None:
            continue
        _consume_mistral_call(body[m.start() : end + 1], out, id_offset)
    return out


def _consume_mistral_call(obj_text: str, out: list[dict], id_offset: int) -> None:
    try:
        obj = json.loads(obj_text)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(obj, dict):
        return
    name = obj.get("name") or ""
    args = obj.get("arguments") or {}
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


def _parse_gemma_tool_calls(content: str, *, id_offset: int) -> list[dict]:
    """Gemma 4: ``<|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>``.

    Also handles the ``skip_special_tokens`` stream where the ``<|tool_call>``
    wrapper and ``<|"|>`` string markers were stripped, leaving a bare
    ``call:NAME{k:v, ...}`` with unquoted values.
    """
    out: list[dict] = []
    for m in _GEMMA_TC_RE.finditer(content):
        name = m.group(1)
        body_start = m.end() - 1
        end_marker = content.find(_GEMMA_TC_END, body_start)
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
    if out:
        return out

    # Wrapper-less form (special tokens stripped from the stream). Skip when
    # the wrapped markers are present so we never double-parse the same call.
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
    """Index of `}` matching `{` at ``brace_pos``; ignores braces inside
    JSON strings. None if unmatched."""
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


def _gemma_balanced_brace_end(text: str, brace_pos: int, hard_stop: int) -> int | None:
    """Like ``_balanced_brace_end`` but skips ``<|"|>`` strings and
    matches `{`/`[` symmetrically."""
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
    while (
        end < len(text)
        and text[end] not in ",}]"
        and not text.startswith(_GEMMA_STR_BEGIN, end)
    ):
        end += 1
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


def _gemma_coerce_scalar(raw: str) -> Any:
    """Coerce an unquoted Gemma value to bool/int/float/None, else keep str.

    Surrounding matching quotes are stripped first (a stuck model may emit the
    same value sometimes quoted, sometimes not; normalising lets the agentic
    loop's duplicate-call collapse recognise them as identical).
    """
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
    """Parse a quote-less Gemma arg body ``key:value, key2:value2``.

    Used for the ``skip_special_tokens`` stream where the ``<|"|>`` string
    markers were removed. Each value runs until the next top-level ``, key:``
    boundary (or the end), tracking ``{}``/``[]``/``()`` depth, so commas and
    braces inside a ``code`` or ``command`` value are preserved instead of
    truncating at the first comma.
    """
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
        out[key] = _gemma_coerce_scalar(body[vstart:i])
        if i < n and body[i] == ",":
            i += 1
    return out


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


# ── DeepSeek R1 / V3 / V3.1 ─────────────────────────────────────────


def _parse_deepseek_tool_calls(content: str, *, id_offset: int) -> list[dict]:
    """DeepSeek R1 / V3 / V3.1.

    R1:    ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\\n``\\`\\`\\`json\\n{...}\\n\\`\\`\\`<｜tool▁call▁end｜>...``
    V3.x:  ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}<｜tool▁call▁end｜>...``

    Ports llama.cpp ``common_chat_parse_deepseek_r1`` / ``_v3_1`` at
    commit ``51fa458a92d6`` (pre-autoparser refactor); tolerates the 5
    opener variants llama.cpp keeps.
    """
    out: list[dict] = []
    begin = _DEEPSEEK_BEGIN_RE.search(content)
    if not begin:
        return out
    scan_start = begin.end()
    end_pos = content.find(_DEEPSEEK_END, scan_start)
    scan_end = end_pos if end_pos >= 0 else len(content)
    body = content[scan_start:scan_end]

    # R1 path first: ``function<｜tool▁sep｜>NAME\n```json\n{...}\n```<｜tool▁call▁end｜>``.
    pos = 0
    while pos < len(body):
        m = _DEEPSEEK_R1_FUNC_RE.search(body, pos)
        if not m:
            break
        name = m.group(1).strip()
        json_start = m.end()
        # Walk a balanced ``{`` even if the trailing fence is truncated.
        if json_start >= len(body) or body[json_start] != "{":
            pos = m.end()
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

    # V3 / V3.1 path: name then bare JSON. We use ``str.find`` for the
    # sep marker and walk back for the name (regex search with a
    # ``[^\n<]+`` quantifier is O(N^2) on adversarial truncated bodies).
    pos = 0
    while pos < len(body):
        sep_pos = body.find(_DEEPSEEK_SEP, pos)
        if sep_pos < 0:
            break
        # Walk left from sep_pos to find the name start. Stops at
        # ``\n`` (previous turn boundary), ``<`` (start of an arbitrary
        # tag), or ``>`` (end of an optional ``<｜tool▁call▁begin｜>``
        # marker).
        name_start = sep_pos
        while name_start > pos and body[name_start - 1] not in "\n<>":
            name_start -= 1
        name = body[name_start:sep_pos].strip()
        json_start = sep_pos + len(_DEEPSEEK_SEP)
        # Skip any whitespace before the JSON.
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
        # Skip past optional ``<｜tool▁call▁end｜>``.
        next_end = body.find(_DEEPSEEK_CALL_END, brace_end + 1)
        pos = next_end + len(_DEEPSEEK_CALL_END) if next_end >= 0 else brace_end + 1
    return out


# ── GLM 4.5 / 4.6 / 4.7 ─────────────────────────────────────────────


def _parse_glm_tool_calls(content: str, *, id_offset: int) -> list[dict]:
    """GLM 4.5 / 4.6 / 4.7.

    ``<tool_call>NAME[\\n]<arg_key>K</arg_key>[\\n]<arg_value>V</arg_value>
    ...</tool_call>``. Multi-call is back-to-back blocks, no envelope.
    Ports llama.cpp ``common_chat_parse_glm_4_5`` at ``51fa458a92d6``.
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
        body_end = close if close >= 0 else len(content)
        body = content[body_start:body_end]

        args: dict[str, Any] = {}
        for pair in _GLM_ARG_PAIR_RE.finditer(body):
            key = pair.group(1).strip()
            raw_val = pair.group(2)
            # The template emits non-strings via ``tojson`` and strings
            # verbatim. Probe the stripped value for an unambiguous JSON
            # literal; otherwise keep the value RAW so significant leading /
            # trailing whitespace in string args (code, diffs) survives --
            # matches vLLM glm4_moe, which never strips string values.
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


def _parse_kimi_tool_calls(content: str, *, id_offset: int) -> list[dict]:
    """Kimi K2.

    ``<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX
    <|tool_call_argument_begin|>{json}<|tool_call_end|>...
    <|tool_calls_section_end|>``. Full id is preserved on ``tool_calls
    [i].id`` for round-trip through the chat template. Outer loop walks
    every section in the stream (vLLM / SGLang parity); ports llama.cpp
    ``common_chat_parse_kimi_k2`` at ``51fa458a92d6``.
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
        # Truncated tail: parse what we have, then exit.
        if section_end < 0:
            out.extend(_parse_kimi_section_body(body, id_offset = id_offset + len(out)))
            return out
        outer_pos = section_end + len(_KIMI_SECTION_END)
        out.extend(_parse_kimi_section_body(body, id_offset = id_offset + len(out)))

    # llama.cpp treats the ``<|tool_calls_section_begin|>`` wrapper as
    # optional -- Kimi K2 can emit a bare ``<|tool_call_begin|>`` call (e.g.
    # straight after reasoning, without closing the section). If the section
    # loop matched nothing but a bare call marker is present, parse the whole
    # content as one section body so the call is not dropped.
    if not out and _KIMI_CALL_BEGIN in content:
        out.extend(_parse_kimi_section_body(content, id_offset = id_offset))
    return out


def _parse_kimi_section_body(body: str, *, id_offset: int) -> list[dict]:
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
                call_end = body.find(_KIMI_CALL_END, brace_end + 1)
                pos = call_end + len(_KIMI_CALL_END) if call_end >= 0 else brace_end + 1
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
            # Malformed / truncated JSON for this call: skip it but keep
            # parsing later calls instead of dropping the rest of the
            # section (vLLM recovers subsequent calls).
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
        call_end = body.find(_KIMI_CALL_END, brace_end + 1)
        pos = call_end + len(_KIMI_CALL_END) if call_end >= 0 else brace_end + 1
    return out
