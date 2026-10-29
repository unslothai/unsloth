# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read a tool-call argument object a model sampled with near-JSON syntax.

Only syntax is repaired, never content: single-quoted strings, unquoted keys, Python literals
(True / False / None), trailing commas, raw newlines and tabs inside strings, and arguments that were
JSON-encoded as a string once or twice. Anything ambiguous (an unescaped quote inside a string, an
unknown bare word, an unterminated string, unbalanced brackets) returns None, so a truncated or garbled
call is never turned into a different one. One linear pass; callers use it only after strict JSON failed.
"""

from __future__ import annotations

import json
from typing import Any, Optional

_LITERALS = {
    "true": "true",
    "True": "true",
    "false": "false",
    "False": "false",
    "null": "null",
    "None": "null",
}
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
_MAX_STRING_LAYERS = 2


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


# NaN / Infinity are not JSON and would replay as arguments no provider parses.
_DECODER = json.JSONDecoder(parse_constant = _reject_constant)


def _strictify(text: str) -> Optional[str]:
    """The strict-JSON spelling of ``text``, or None when the repair would have to guess."""
    out: list[str] = []
    n = len(text)
    i = 0
    pending_comma = False
    pending_space: list[str] = []
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            (pending_space if pending_comma else out).append(c)
            i += 1
            continue
        if pending_comma:
            pending_comma = False
            if c not in "}]":
                out.append(",")
            out.extend(pending_space)
            pending_space = []
        if c == ",":
            pending_comma = True
            i += 1
            continue
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            buf = ['"']
            while True:
                if j >= n:
                    return None  # unterminated string: truncated, never guessed
                ch = text[j]
                if ch == quote:
                    break
                if ch == "\\":
                    if j + 1 >= n:
                        return None
                    nxt = text[j + 1]
                    if quote == "'" and nxt == "'":
                        buf.append("'")
                    else:
                        buf.append(text[j : j + 2])
                    j += 2
                    continue
                if ch == '"':
                    buf.append('\\"')  # only reachable inside a single-quoted string
                else:
                    buf.append(_CONTROL_ESCAPES.get(ch, ch))
                j += 1
            buf.append('"')
            out.append("".join(buf))
            i = j + 1
            continue
        if c.isdigit() or c == "-":
            j = i + 1
            while j < n and text[j] in "0123456789eE+-.":
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in "_-"):
                j += 1
            word = text[i:j]
            k = j
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k < n and text[k] == ":":
                out.append(json.dumps(word))  # unquoted key
            elif word in _LITERALS:
                out.append(_LITERALS[word])
            else:
                return None  # a bare word that is not a literal: not ours to guess
            i = j
            continue
        out.append(c)
        i += 1
    if pending_comma:
        out.append(",")
    return "".join(out)


def relaxed_json_object(text: Any) -> Optional[dict]:
    """The argument object ``text`` spells, or None. Strict JSON first; a JSON string holding an object
    (arguments encoded once or twice) is unwrapped; then the syntax repair above."""
    if not isinstance(text, str):
        return None
    value: Any = text
    for _ in range(_MAX_STRING_LAYERS + 1):
        if not isinstance(value, str):
            break
        stripped = value.strip()
        try:
            value = _DECODER.decode(stripped)
            continue
        except (ValueError, RecursionError):
            pass
        if not stripped.startswith("{"):
            return None
        repaired = _strictify(stripped)
        if repaired is None:
            return None
        try:
            value = _DECODER.decode(repaired)
        except (ValueError, RecursionError):
            return None
    return value if isinstance(value, dict) else None
