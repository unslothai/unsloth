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

_TOKEN = re.compile(
    r"""
      (?P<comment>//[^\n]*|/\*.*?\*/)
    | (?P<key>[A-Za-z_$][\w$]*|"(?:[^"\\\n]|\\.)*")\s*:
    | (?P<string>"(?:[^"\\\n]|\\.)*")
    | (?P<open>\{)
    | (?P<close>\})
    """,
    re.VERBOSE | re.DOTALL,
)


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
            pending = json.loads(raw) if raw.startswith('"') else raw
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
                strings[dotted] = json.loads(match.group("string"))
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
    return strings[key]
