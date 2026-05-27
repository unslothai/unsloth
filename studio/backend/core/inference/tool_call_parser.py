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


# ── Streaming-buffer signal markers ─────────────────────────────────


# Prefixes the safetensors / MLX streaming buffer watches for to gate
# in-progress text. When ANY of these appear in the cumulative text,
# the state machine switches from STREAMING to DRAINING so we don't
# leak partial markup to the user before we can parse it.
TOOL_XML_SIGNALS = (
    "<tool_call>",
    "<function=",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<|tool_call>",
    # DeepSeek R1 / V3 / V3.1 (full-width pipes + lower-one-eighth-block).
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁call▁begin｜>",
    # Alternative DeepSeek openers llama.cpp also recognises -- some
    # checkpoints emit ASCII underscores, others a short form, and
    # llama.cpp keeps two more legacy variants (literal-space and
    # backslash-escaped) that we mirror for streaming-gate parity with
    # ``_DEEPSEEK_BEGIN_RE`` below.
    "<｜tool_calls_begin｜>",
    "<｜tool▁calls｜>",
    "<｜tool calls begin｜>",
    "<｜tool\\_calls\\_begin｜>",
    # Kimi K2 / Moonshot section start.
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
)


# ── Strip patterns for ``strip_tool_markup`` ────────────────────────


# _TOOL_CLOSED_PATS: closed pairs only (used during streaming so
# in-progress XML stays buffered). _TOOL_ALL_PATS: also matches trailing
# unclosed runs so truncated tails don't leak markup at end-of-turn.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=\w+>.*?</function>", re.DOTALL),
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
    re.compile(r"<function=\w+>.*$", re.DOTALL),
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\].*$", re.DOTALL),
    re.compile(r"<\|python_tag\|>.*$", re.DOTALL),
    # DeepSeek envelopes truncated mid-stream.
    re.compile(r"<｜tool[▁_]calls[▁_]begin｜>.*$", re.DOTALL),
    re.compile(r"<｜tool▁call▁begin｜>.*$", re.DOTALL),
    # Kimi K2 envelope truncated.
    re.compile(r"<\|tool_calls_section_begin\|>.*$", re.DOTALL),
    re.compile(r"<\|tool_call_begin\|>.*$", re.DOTALL),
]


# ── Nudges + error-result prefixes ──────────────────────────────────


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


# ── Format-specific regexes ─────────────────────────────────────────


# Qwen / Hermes <tool_call>{json}
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
# Qwen3.5 / Hermes XML form <function=name><parameter=k>v
_TC_FUNC_START_RE = re.compile(r"<function=([\w\.\-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w\.\-]+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")

# Llama-3 <|python_tag|>NAME.call(...)
_LLAMA3_PYTHON_TAG = "<|python_tag|>"
_LLAMA3_PY_CALL_RE = re.compile(
    r"<\|python_tag\|>\s*([\w\.\-]+)\s*\.\s*call\s*\(",
)
_LLAMA3_KV_RE = re.compile(
    r"""(\w+)\s*=\s*(?:"((?:\\.|[^"\\])*)"|(-?\d+(?:\.\d+)?)|(true|false|null))""",
    re.VERBOSE,
)

# Mistral [TOOL_CALLS] trigger. v11+ chains multiple triggers, each
# followed by a bare name then either ``{json}`` (Magistral) or
# ``[ARGS]{json}`` (Ministral / Mistral Large 3).
_MISTRAL_TRIGGER = "[TOOL_CALLS]"
_MISTRAL_ARGS_MARKER = "[ARGS]"
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
_DEEPSEEK_V3_FUNC_RE = re.compile(
    r"(?:"
    + re.escape(_DEEPSEEK_CALL_BEGIN)
    + r")?([^\n<]+?)"
    + re.escape(_DEEPSEEK_SEP),
)

# GLM 4.5 / 4.6 / 4.7 markers. Body is ``NAME\n<arg_key>K</arg_key>\n
# <arg_value>V</arg_value>...`` per chat_template.jinja; strings are
# raw, non-strings are JSON-encoded.
_GLM_TC_OPEN_RE = re.compile(r"<tool_call>\s*([^\n<{][^\n<]*)\n")
_GLM_TC_CLOSE = "</tool_call>"
_GLM_ARG_PAIR_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)
# Without tool-schema access at the parse site we use a structural
# heuristic: only deserialize an ``<arg_value>`` body when it
# unambiguously looks like a JSON literal. The template's rule is
# ``{{ v | tojson(ensure_ascii=False) if v is not string else v }}``,
# so a *string* arg arrives raw (``42``) and a *non-string* arg arrives
# JSON-encoded (``"42"`` for an actual string, ``42`` for an int, etc.).
# Bare ``42`` / ``true`` / ``null`` are therefore the model's choice of
# representation for string args -- coercing them to int / bool / None
# would silently mangle a ``search(query="42")`` call. Only the shapes
# below can be safely decoded.
_GLM_JSON_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# Kimi K2 / Moonshot markers (ASCII pipes, NOT full-width). The name
# arrives as ``functions.NAME:IDX`` between ``<|tool_call_begin|>`` and
# ``<|tool_call_argument_begin|>``. Strip the prefix and suffix to
# recover the bare name.
_KIMI_SECTION_BEGIN = "<|tool_calls_section_begin|>"
_KIMI_SECTION_END = "<|tool_calls_section_end|>"
_KIMI_CALL_BEGIN = "<|tool_call_begin|>"
_KIMI_ARG_BEGIN = "<|tool_call_argument_begin|>"
_KIMI_CALL_END = "<|tool_call_end|>"
_KIMI_ID_RE = re.compile(r"^(?:functions\.)?([\w\.\-]+)(?::(\d+))?$")

# Gemma 4 <|tool_call>call:NAME{...}<tool_call|>. ``<|"|>`` wraps strings.
_GEMMA_TC_RE = re.compile(r"<\|tool_call>\s*call\s*:\s*([\w\.\-]+)\s*\{")
_GEMMA_STR_BEGIN = '<|"|>'
_GEMMA_STR_END = '<|"|>'
_GEMMA_TC_END = "<tool_call|>"


# ── Public API ──────────────────────────────────────────────────────


def _balanced_bracket_end(text: str, start: int) -> int | None:
    """Index of the matching ``]`` for the ``[`` at ``text[start]``.

    Skips brackets inside JSON string literals. Returns ``None`` if no
    matching close is found.
    """
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


def _strip_mistral_closed_calls(text: str) -> str:
    """Strip ``[TOOL_CALLS]`` blocks with balanced brace/bracket scanning.

    Handles three Mistral emission shapes:

      - ``[TOOL_CALLS] [ {...}, {...} ]``         (v0.3 / Nemo / Small)
      - ``[TOOL_CALLS] name { json }``            (v11+ / Magistral)
      - ``[TOOL_CALLS] name [ARGS] { json }``     (Ministral / Large 3)

    The regex ``\\{.*?\\}`` truncates at the first ``}``, losing nested
    JSON, so this walks balanced braces/brackets instead. Only matches
    runs that close cleanly; unclosed trailing markup is left in place
    for ``final=True`` cleanup via ``_TOOL_ALL_PATS``.
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
        # Skip whitespace + optional name + optional ``[ARGS]``.
        i = body_start
        while i < n and text[i] in " \t\n\r":
            i += 1
        # Array shape: [TOOL_CALLS] [ ... ]
        if i < n and text[i] == "[":
            end = _balanced_bracket_end(text, i)
            if end is None:
                # Truncated mid-array; leave the trigger and everything
                # after it in place so caller can buffer / final-strip.
                out.append(text[idx:])
                break
            cursor = end + 1
            # Optional trailing </s> (Mistral EOS).
            if text.startswith("</s>", cursor):
                cursor += len("</s>")
            continue
        # Named shape: [TOOL_CALLS] name [ARGS]? { json }
        name_match = _MISTRAL_V11_NAME_RE.match(text, i)
        if not name_match:
            out.append(text[idx:body_start])
            cursor = body_start
            continue
        i = name_match.end()
        while i < n and text[i] in " \t\n\r":
            i += 1
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
    """Strip tool-call markup from streamed text.

    ``final=False`` only removes closed pairs so in-progress markup
    stays buffered. ``final=True`` also removes trailing unclosed runs
    and trims the result.
    """
    # Mistral patterns need balanced brace/bracket scanning -- a
    # non-greedy regex would truncate at the first ``}`` inside a
    # nested JSON object and leak the rest into user-visible text.
    text = _strip_mistral_closed_calls(text)
    pats = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
    for pat in pats:
        text = pat.sub("", text)
    return text.strip() if final else text


def has_tool_signal(text: str) -> bool:
    """True if ``text`` contains any known tool-call signal."""
    return any(s in text for s in TOOL_XML_SIGNALS)


def parse_tool_calls_from_text(content: str, *, id_offset: int = 0) -> list[dict]:
    """Parse OpenAI-format ``tool_calls`` from model text.

    Returns ``[{"id", "type", "function": {"name", "arguments"}}]``
    where ``arguments`` is always a JSON string. Tries each known
    emission format in turn; returns as soon as one yields calls so
    we never double-count.
    """
    # DeepSeek R1 / V3 / V3.1 ``<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>``.
    # Run early -- the full-width markers cannot collide with any
    # other family, and R1's code-fence body would fail Qwen's
    # JSON-start regex anyway.
    calls = _parse_deepseek_tool_calls(content, id_offset = id_offset)
    if calls:
        return calls

    # Kimi K2 ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>``.
    # Markers cannot collide with any other family.
    calls = _parse_kimi_tool_calls(content, id_offset = id_offset)
    if calls:
        return calls

    # Qwen / Hermes <tool_call>{json}
    calls = _parse_tool_call_json(content, id_offset = id_offset)
    if calls:
        return calls

    # GLM 4.5 / 4.6 / 4.7 ``<tool_call>NAME\n<arg_key>K</arg_key>
    # <arg_value>V</arg_value>...</tool_call>``. Marker collides with
    # Qwen's ``<tool_call>``, but Qwen requires ``\s*{`` after the tag
    # while GLM emits a bare name then ``\n``, so Qwen returns no calls
    # before we get here. Running GLM AFTER Qwen also keeps Qwen
    # behaviour unchanged on real Qwen emissions.
    calls = _parse_glm_tool_calls(content, id_offset = id_offset)
    if calls:
        return calls

    # Qwen3.5 / Hermes <function=name><parameter=k>v
    calls = _parse_function_xml(content, id_offset = id_offset)
    if calls:
        return calls

    # Llama-3 <|python_tag|>...
    calls = _parse_llama3_python_tag(content, id_offset = id_offset)
    if calls:
        return calls

    # Mistral [TOOL_CALLS]...
    calls = _parse_mistral_tool_calls(content, id_offset = id_offset)
    if calls:
        return calls

    # Gemma 4 <|tool_call>...<tool_call|>
    calls = _parse_gemma_tool_calls(content, id_offset = id_offset)
    if calls:
        return calls

    # Llama-3.2 bare JSON ``{"name":..., "parameters":...}`` (no tag).
    # Strict: only fires when stripped content STARTS with ``{`` and
    # parses as ``{name: str, parameters|arguments: dict}``. Keeps
    # plain assistant prose unaffected.
    return _parse_llama3_bare_json(content, id_offset = id_offset)


# ── Per-format parsers ──────────────────────────────────────────────


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
        args = obj.get("arguments", {})
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
        body = _TC_FUNC_CLOSE_RE.sub("", content[body_start:body_end])

        args: dict = {}
        param_starts = list(_TC_PARAM_START_RE.finditer(body))
        if len(param_starts) == 1:
            pm = param_starts[0]
            val = _TC_PARAM_CLOSE_RE.sub("", body[pm.end() :])
            args[pm.group(1)] = val.strip()
        else:
            for pidx, pm in enumerate(param_starts):
                val_start = pm.end()
                next_param = (
                    param_starts[pidx + 1].start()
                    if pidx + 1 < len(param_starts)
                    else len(body)
                )
                val = _TC_PARAM_CLOSE_RE.sub("", body[val_start:next_param])
                args[pm.group(1)] = val.strip()

        out.append(
            {
                "id": f"call_{id_offset + len(out)}",
                "type": "function",
                "function": {"name": func_name, "arguments": json.dumps(args)},
            }
        )
    return out


def _parse_llama3_python_tag(content: str, *, id_offset: int) -> list[dict]:
    """Llama-3 emission shapes:
      <|python_tag|>NAME.call(arg="v", ...)               (built-in tools)
      <|python_tag|>{"name":"NAME", "parameters":{...}}   (custom tools)
      <|python_tag|>{"name":...}; {"name":...}            (multi-call, ``; `` sep)
    Accepts both ``parameters`` and ``arguments`` keys per Llama 3.1/3.2.
    """
    out: list[dict] = []
    if _LLAMA3_PYTHON_TAG not in content:
        return out

    # 1. NAME.call(...) built-in form.
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
                # json.loads on a wrapped JSON string handles
                # \n / \t / \uXXXX escapes correctly while preserving
                # literal UTF-8 bytes (emoji, CJK, etc.) that the older
                # ``bytes.decode("unicode_escape")`` path mangled.
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

    # 2. <|python_tag|>{"name":..., "parameters":...} JSON form. Use a
    #    streaming JSON decoder (raw_decode) so we can peel multiple
    #    objects out of the same emission (separated by ``; `` per
    #    Llama 3 template).
    if not out:
        decoder = json.JSONDecoder()
        idx = content.find(_LLAMA3_PYTHON_TAG)
        while idx >= 0:
            search_from = idx + len(_LLAMA3_PYTHON_TAG)
            # Scan all `{` from this trigger; raw_decode jumps the
            # cursor past each parsed object, but if a `{` falls
            # inside an already-decoded object we skip it.
            cursor = search_from
            while cursor < len(content):
                brace = content.find("{", cursor)
                if brace < 0:
                    break
                # Stop if we've hit the next <|python_tag|>.
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


def _parse_llama3_bare_json(content: str, *, id_offset: int) -> list[dict]:
    """Llama-3.2 ``custom_tools`` shape -- bare JSON ``{"name":...,
    "parameters":{...}}`` emitted directly, no ``<|python_tag|>``.

    Strict to avoid firing on tool-message echoes:

    * Content must start with ``{`` once whitespace and any leading
      ``<|begin_of_text|>`` / ``<|eot_id|>`` etc. sentinels are stripped.
    * Object must have ``name`` (non-empty str) plus a dict in
      ``parameters`` or ``arguments``.
    * Loops via ``raw_decode`` to peel multiple ``;``-separated calls.
    """
    out: list[dict] = []
    stripped = content.lstrip()
    # Strip leading Llama-3 sentinel tokens that sometimes precede the
    # JSON (``<|eot_id|>`` from the prior turn, ``<|start_header_id|>``,
    # ``<|begin_of_text|>``). Loop until no sentinel matches: the
    # tokens can appear in any order and chain, so a single pass would
    # leave later sentinels behind once an earlier one consumed its
    # prefix.
    _sentinels = (
        "<|begin_of_text|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",
    )
    while True:
        stripped = stripped.lstrip()
        matched = False
        for sentinel in _sentinels:
            if stripped.startswith(sentinel):
                stripped = stripped[len(sentinel) :]
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
        # Skip whitespace and Llama 3 inter-call separator ``;``.
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
        if "parameters" in obj:
            args = obj.get("parameters")
        elif "arguments" in obj:
            args = obj.get("arguments")
        else:
            break
        if isinstance(args, dict):
            args_str = json.dumps(args)
        elif isinstance(args, str):
            args_str = args
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
    """Mistral emissions covered:
    Pre-v11 array:  ``[TOOL_CALLS] [{"name":..., "arguments":...}, ...]``
    Pre-v11 single: ``[TOOL_CALLS]{"name":..., "arguments":...}``
    v11+ single:    ``[TOOL_CALLS]name{json_args}``
    v11+ parallel:  ``[TOOL_CALLS]a{...}[TOOL_CALLS]b{...}``
    v11+ w/ [ARGS]: ``[TOOL_CALLS]name[ARGS]{json_args}`` (Ministral / Large 3)
    """
    out: list[dict] = []
    idx = content.find(_MISTRAL_TRIGGER)
    if idx < 0:
        return out

    # Decide whether the FIRST occurrence is array / single-object
    # (pre-v11) or v11+ bare-name. Skip whitespace, peek at next char.
    j = idx + len(_MISTRAL_TRIGGER)
    k = j
    while k < len(content) and content[k] in " \t\n\r":
        k += 1
    if k >= len(content):
        return out

    if content[k] == "[":
        return _parse_mistral_array(content, k, id_offset)

    if content[k] == "{":
        # Could be pre-v11 single object ``{"name": ...}`` or a JSON
        # blob immediately following the trigger (rare). Try parsing
        # as an object that exposes ``name``; if not, fall through to
        # v11+ handling so we don't drop emission silently.
        end = _balanced_brace_end(content, k)
        if end is not None:
            try:
                obj = json.loads(content[k : end + 1])
                if isinstance(obj, dict) and obj.get("name"):
                    _consume_mistral_call(content[k : end + 1], out, id_offset)
                    return out
            except (json.JSONDecodeError, ValueError):
                pass

    # v11+ path: walk every ``[TOOL_CALLS]`` and parse ``name{json}``
    # or ``name[ARGS]{json}`` after each trigger.
    pos = idx
    while pos >= 0:
        cur = pos + len(_MISTRAL_TRIGGER)
        nm = _MISTRAL_V11_NAME_RE.match(content, cur)
        if not nm:
            pos = content.find(_MISTRAL_TRIGGER, cur)
            continue
        name = nm.group(1)
        after_name = nm.end()
        # Optional ``[ARGS]`` marker.
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
    """Parse pre-v11 ``[TOOL_CALLS] [{...}, ...]`` JSON array form."""
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

    # Healing path: walk objects manually for unclosed array.
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
    """Gemma 4: <|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>."""
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
    return out


# ── Brace-balancing helpers ─────────────────────────────────────────


def _balanced_brace_end(text: str, brace_pos: int) -> int | None:
    """Index of `}` matching `{` at ``brace_pos`` -- ignores `{` `}`
    inside JSON strings. Returns None if unmatched."""
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
    """Same as ``_balanced_brace_end`` but respects Gemma ``<|"|>``
    string runs and matches `{`/`[` symmetrically."""
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
    """Parse one Gemma argument value starting at ``i``. Returns
    ``(value, next_index)``."""
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
    # Primitive: number, true/false/null, or bare identifier (rare).
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


def _gemma_parse_mapping_body(body: str) -> dict[str, Any]:
    """Parse content between `{` and `}` for a Gemma argument mapping."""
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
    """DeepSeek emissions:
      R1:    ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\\n``\\`\\`\\`json\\n{...}\\n\\`\\`\\`<｜tool▁call▁end｜>...``
      V3.x:  ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}<｜tool▁call▁end｜>...``

    Mirrors llama.cpp's common_chat_parse_deepseek_r1 / _v3_1 (at
    pre-autoparser-refactor commit ``51fa458a92d6``, where the logic
    lived in ``common/chat-parser.cpp`` lines 801-879; the upstream has
    since been split into ``common/chat.cpp`` + ``common/chat-peg-
    parser.cpp`` by llama.cpp PR #18675) and vLLM's
    ``vllm/tool_parsers/deepseekv31_tool_parser.py``. Tolerates the
    five outer-marker variants llama.cpp keeps for real checkpoints.
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

    # V3 / V3.1 path: name then bare JSON.
    pos = 0
    while pos < len(body):
        m = _DEEPSEEK_V3_FUNC_RE.search(body, pos)
        if not m:
            break
        name = m.group(1).strip()
        json_start = m.end()
        # Skip any whitespace before the JSON.
        while json_start < len(body) and body[json_start] in " \t\n\r":
            json_start += 1
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
        # Skip past optional ``<｜tool▁call▁end｜>``.
        next_end = body.find(_DEEPSEEK_CALL_END, brace_end + 1)
        pos = next_end + len(_DEEPSEEK_CALL_END) if next_end >= 0 else brace_end + 1
    return out


# ── GLM 4.5 / 4.6 / 4.7 ─────────────────────────────────────────────


def _parse_glm_tool_calls(content: str, *, id_offset: int) -> list[dict]:
    """GLM 4.x emission:
      ``<tool_call>NAME\\n<arg_key>K1</arg_key>\\n<arg_value>V1</arg_value>...
      <arg_key>Kn</arg_key>\\n<arg_value>Vn</arg_value>\\n</tool_call>``

    Strings come through raw; non-string args are JSON-encoded per the
    template's ``{{ v | tojson(ensure_ascii=False) if v is not string
    else v }}`` rule. Multi-call is back-to-back ``<tool_call>...
    </tool_call>`` blocks with no outer envelope. Mirrors llama.cpp's
    ``common_chat_parse_glm_4_5`` (pre-autoparser-refactor commit
    ``51fa458a92d6``, ``common/chat-parser.cpp`` lines 1040-1052; see
    note in ``_parse_deepseek_tool_calls`` re: post-refactor paths)
    and vLLM's ``vllm/tool_parsers/glm4_moe_tool_parser.py``.
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
            raw_val = pair.group(2).strip()
            # Only try to decode when the body unambiguously looks like a
            # JSON literal. Prose, code, and arbitrary string values stay
            # raw (the template emits string args verbatim). Numeric /
            # boolean / null shapes are still ambiguous with strings that
            # happen to look like primitives -- this is an inherent
            # limitation of the template without per-arg schema access.
            if (
                raw_val[:1] in '{["'
                or raw_val in ("true", "false", "null")
                or _GLM_JSON_NUMERIC_RE.fullmatch(raw_val)
            ):
                try:
                    args[key] = json.loads(raw_val)
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
    """Kimi K2 emission:
      ``<|tool_calls_section_begin|>
        <|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{json}<|tool_call_end|>
        ...
        <|tool_calls_section_end|>``

    Name arrives as ``functions.NAME:IDX``. Strip the ``functions.``
    prefix and ``:N`` suffix to recover the bare name. The full id
    string is preserved as ``tool_calls[i].id`` so the conversation
    replay round-trips the exact form the model emitted (vLLM and
    SGLang both do this). Mirrors llama.cpp's
    ``common_chat_parse_kimi_k2`` (pre-autoparser-refactor commit
    ``51fa458a92d6`` of ``common/chat-parser.cpp``) and vLLM's
    ``vllm/tool_parsers/kimi_k2_tool_parser.py``.
    """
    out: list[dict] = []
    section_start = content.find(_KIMI_SECTION_BEGIN)
    if section_start < 0:
        return out
    scan_start = section_start + len(_KIMI_SECTION_BEGIN)
    section_end = content.find(_KIMI_SECTION_END, scan_start)
    scan_end = section_end if section_end >= 0 else len(content)
    body = content[scan_start:scan_end]

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
        # Drop bare-counter ids (e.g. ``3``) -- SGLang infers the function
        # name from the tool schema in this case, but we don't have the
        # schema at the parse site. Surfacing a tool literally named ``"3"``
        # would only be rejected by the dispatcher, so we match vLLM and
        # skip the call. Per the Kimi K2 tool-call guidance real models
        # emit ``functions.NAME:IDX``, so this path is the exception.
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
        # Walk a balanced brace so streaming truncation that drops the
        # trailing ``<|tool_call_end|>`` still surfaces a call.
        # Skip whitespace before the ``{``.
        while json_start < len(body) and body[json_start] in " \t\n\r":
            json_start += 1
        if json_start >= len(body) or body[json_start] != "{":
            pos = arg_begin + len(_KIMI_ARG_BEGIN)
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
