# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Backend-neutral tool-call parser shared by GGUF, safetensors, and MLX.

Covers the emission formats so the safetensors + MLX agentic loop sees
the same call shape llama-server normalises for GGUF:

  - ``<tool_call>{json}</tool_call>``           (Qwen / Hermes)
  - ``<function=name><parameter=k>v</parameter></function>``  (Qwen3-Coder XML; nested in <tool_call>)
  - ``<|python_tag|>NAME.call(k="v", ...)``     (Llama-3 built-in tools)
  - ``<|python_tag|>{"name":..., "parameters":...}``  (Llama-3 custom)
  - ``{"name":..., "parameters":...}``          (Llama-3.2 bare JSON)
  - ``[TOOL_CALLS] [{...}, ...]``               (Mistral V3 tokenizer: v0.3, Nemo, Small, Ministral-8B-2410, Large-2411)
  - ``[TOOL_CALLS]name{json}``                  (Mistral V11+ Tekken: Magistral)
  - ``[TOOL_CALLS]name[ARGS]{json}``            (Mistral V13 Tekken: Devstral, Magistral-Small-2509)
  - ``<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>``  (Gemma 4; forward-looking, no shipping Gemma model emits this yet)

Closing tags / brackets are tolerated when missing because models
frequently truncate them mid-stream.
"""

import json
import re
from typing import Any


# Flip the stream buffer to DRAINING so partial markup never leaks.
TOOL_XML_SIGNALS = (
    "<tool_call>",
    "<function=",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<|tool_call>",
)


# Closed pairs only; _TOOL_ALL_PATS also eats unclosed tails at end-of-turn.
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=\w+>.*?</function>", re.DOTALL),
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=\w+>.*$", re.DOTALL),
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"\[TOOL_CALLS\].*$", re.DOTALL),
    re.compile(r"<\|python_tag\|>.*$", re.DOTALL),
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
# Qwen3-Coder XML ``<function=name><parameter=k>v``, nested inside <tool_call>.
_TC_FUNC_START_RE = re.compile(r"<function=([\w\.\-]+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=([\w\.\-]+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")

# Llama-3 ``<|python_tag|>NAME.call(...)``.
_LLAMA3_PYTHON_TAG = "<|python_tag|>"
_LLAMA3_PY_CALL_RE = re.compile(
    r"<\|python_tag\|>\s*([\w\.\-]+)\s*\.\s*call\s*\(",
)
_LLAMA3_KV_RE = re.compile(
    r"""(\w+)\s*=\s*(?:"((?:\\.|[^"\\])*)"|(-?\d+(?:\.\d+)?)|(true|false|null))""",
    re.VERBOSE,
)

# Mistral ``[TOOL_CALLS]``: V11+ Tekken chains ``name{json}`` / ``name[ARGS]{json}``
# (Magistral, Devstral); V3 tokenizer models emit the ``[...]`` array form.
_MISTRAL_TRIGGER = "[TOOL_CALLS]"
_MISTRAL_ARGS_MARKER = "[ARGS]"
_MISTRAL_V11_NAME_RE = re.compile(r"\s*([\w\.\-]+)\s*")

# Gemma 4 (forward-looking; Gemma 3 does not emit this shape yet):
# ``<|tool_call>call:NAME{...}<tool_call|>``, ``<|"|>`` wraps strings.
_GEMMA_TC_RE = re.compile(r"<\|tool_call>\s*call\s*:\s*([\w\.\-]+)\s*\{")
_GEMMA_STR_BEGIN = '<|"|>'
_GEMMA_STR_END = '<|"|>'
_GEMMA_TC_END = "<tool_call|>"


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
    """Return OpenAI-format tool calls. Tries each format and returns
    as soon as one matches so we never double-count."""
    for parser in (
        _parse_tool_call_json,  # Qwen / Hermes
        _parse_function_xml,  # Qwen3-Coder XML
        _parse_llama3_python_tag,  # Llama-3
        _parse_mistral_tool_calls,  # Mistral
        _parse_gemma_tool_calls,  # Gemma 4 (forward-looking)
    ):
        calls = parser(content, id_offset = id_offset)
        if calls:
            return calls

    # Llama-3.2 bare ``{"name":..., "parameters":...}``. Strict: only
    # fires on content that starts with ``{`` and parses as the right
    # shape, so plain prose stays untouched.
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
    """Gemma 4 (forward-looking; no shipping Gemma 3 model emits this):
    ``<|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>``.
    Capability gate suppresses the tools pill on real Gemma 3 templates."""
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
