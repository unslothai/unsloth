# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shipped English UI strings, read by dotted key out of studio/frontend's en.ts.

Browser drivers find controls by their accessible name, and contract tests check that a label
lives in the catalog. Both used to retype the English, so a wording change that is correct by
construction (#11924 dropped the leading "Show" from eight settings labels) turned every Composer
leg and a Repo-tests contract red at once. Reading the catalog keeps them pinned to the key the
component renders, not to its current wording.

A small tokenizer rather than a TypeScript parse: the catalog is a nested object literal of
string values, and the alternative is a Node dependency for suites that are otherwise pure
Python. It tracks nesting, so `settings.chat.showResponseModel` is not confused with another
section's `showResponseModel`, and it accepts a value wrapped onto the line after its key.
"""

from __future__ import annotations

import functools
import json
import re
import string
from pathlib import Path

EN_LOCALE_TS = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "frontend"
    / "src"
    / "i18n"
    / "locales"
    / "en.ts"
)

_QUOTED = r"""(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)"""

_TOKEN = re.compile(
    rf"""
      (?P<comment>//[^\n]*|/\*.*?\*/)
    | (?P<key>(?:[A-Za-z_$][\w$]*|{_QUOTED}))\s*:
    | (?P<string>{_QUOTED})
    | (?P<open>\{{)
    | (?P<close>\}})
    """,
    re.VERBOSE | re.DOTALL,
)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


class _Interpolated(str):
    """A template literal with `${...}` in it: not a label a test can match as written."""


def _decode(literal: str) -> str:
    """The value of a JavaScript string literal, in any of its three quote styles.

    Single-quoted values are how the catalog writes English that itself contains double quotes
    (`'Are you sure you want to delete "{name}"?'`); reading only double-quoted ones handed back
    the inner `{name}` as the label.
    """
    body = literal[1:-1]
    out = []
    interpolated = False
    index = 0
    while index < len(body):
        char = body[index]
        if char == "$" and body[index + 1 : index + 2] == "{" and literal.startswith("`"):
            # Reached outside an escape, so this `${` is a live placeholder, not `\${`.
            interpolated = True
        if char == "\\" and index + 1 < len(body):
            nxt = body[index + 1]
            if nxt == "\n":
                # A line continuation contributes nothing to the value.
                index += 2
                continue
            braced = re.match(r"u\{([0-9A-Fa-f]{1,6})\}", body[index + 1 :])
            if braced:
                out.append(chr(int(braced.group(1), 16)))
                index += 1 + braced.end()
                continue
            if (
                nxt == "x"
                and len(body[index + 2 : index + 4]) == 2
                and all(c in string.hexdigits for c in body[index + 2 : index + 4])
            ):
                out.append(chr(int(body[index + 2 : index + 4], 16)))
                index += 4
                continue
            if (
                nxt == "u"
                and len(body[index + 2 : index + 6]) == 4
                and all(c in string.hexdigits for c in body[index + 2 : index + 6])
            ):
                out.append(chr(int(body[index + 2 : index + 6], 16)))
                index += 6
                continue
            out.append(_ESCAPES.get(nxt, nxt))
            index += 2
            continue
        out.append(char)
        index += 1
    value = "".join(out)
    if interpolated:
        return _Interpolated(value)
    return value


def _flatten(source: str) -> dict[str, str]:
    strings: dict[str, str] = {}
    path: list[str | None] = []
    pending: str | None = None
    pos = 0
    while True:
        match = _TOKEN.search(source, pos)
        if match is None:
            break
        pos = match.end()
        kind = match.lastgroup
        if kind == "comment":
            continue
        if kind == "key":
            # Tried before a plain string, so a quoted key followed by its colon is read as a key.
            raw = match.group("key")
            pending = raw if raw[0] not in "\"'`" else _decode(raw)
        elif kind == "open":
            path.append(pending)
            pending = None
        elif kind == "close":
            if path:
                path.pop()
            pending = None
        elif kind == "string":
            if pending is not None:
                dotted = ".".join(p for p in [*path, pending] if p is not None)
                strings[dotted] = _decode(match.group("string"))
            pending = None
    return strings


@functools.lru_cache(maxsize = None)
def _catalog(path: str) -> dict[str, str]:
    return _flatten(Path(path).read_text(encoding = "utf-8"))


def en_string(key: str, catalog: Path = EN_LOCALE_TS) -> str:
    """The English for `key` (for example `composerSettings.showContext`), or fail naming the key.

    A missing key raises rather than returning "": an empty string is a substring of everything,
    and a locator built from it matches the wrong control instead of failing.
    """
    strings = _catalog(str(catalog))
    if key not in strings:
        raise KeyError(f"the en catalog no longer defines {key!r} ({catalog})")
    value = strings[key]
    if isinstance(value, _Interpolated):
        raise ValueError(
            f"{key!r} is a template with ${{...}} in it, not a fixed label ({catalog})"
        )
    return str(value)


def aria_label_selector(label: str) -> str:
    """A CSS selector for `[aria-label="<label>"]`, with the label quoted as a CSS string.

    Catalog text is data: English that holds a `"`, a `\\` or a control character such as CR or FF
    would otherwise end the string early or read as an escape, and the selector would be invalid or
    match something else.
    """
    quoted = "".join(
        "\\" + char
        if char in '\\"'
        # CSS reads CR, LF and FF as line breaks, which end a string; write controls as code points.
        else f"\\{ord(char):x} "
        if ord(char) < 0x20 or char == "\x7f"
        else char
        for char in label
    )
    return f'[aria-label="{quoted}"]'
