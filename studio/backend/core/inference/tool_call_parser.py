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
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=\w+>.*$", re.DOTALL),
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\].*$", re.DOTALL),
    re.compile(r"<\|python_tag\|>.*$", re.DOTALL),
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

# Gemma 4 <|tool_call>call:NAME{...}<tool_call|>. ``<|"|>`` wraps strings.
_GEMMA_TC_RE = re.compile(r"<\|tool_call>\s*call\s*:\s*([\w\.\-]+)\s*\{")
_GEMMA_STR_BEGIN = '<|"|>'
_GEMMA_STR_END = '<|"|>'
_GEMMA_TC_END = "<tool_call|>"


# ── Public API ──────────────────────────────────────────────────────


def strip_tool_markup(text: str, *, final: bool = False) -> str:
    """Strip tool-call markup from streamed text.

    ``final=False`` only removes closed pairs so in-progress markup
    stays buffered. ``final=True`` also removes trailing unclosed runs
    and trims the result.
    """
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
    # Qwen / Hermes <tool_call>{json}
    calls = _parse_tool_call_json(content, id_offset = id_offset)
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
                try:
                    args[k] = bytes(kv.group(2), "utf-8").decode("unicode_escape")
                except (UnicodeDecodeError, ValueError):
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
    # JSON (``<|eot_id|>`` from the prior turn, ``<|start_header_id|>``).
    for sentinel in (
        "<|begin_of_text|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",
    ):
        stripped = stripped.lstrip()
        if stripped.startswith(sentinel):
            stripped = stripped[len(sentinel) :]
    stripped = stripped.lstrip()
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
