# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reads JSX attributes with bracket and string awareness; naive '<' or '>' searches cut tags short."""

from __future__ import annotations

import re

# Single quotes are usually apostrophes in JSX prose, not delimiters.
_QUOTES = frozenset('"`')
_COMMENT_SPAN = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def without_comments(source: str) -> str:
    """Blank comments in place so offsets stay valid; comment text must not be read as code or classes."""
    out = list(source)
    index = 0
    while index < len(source):
        char = source[index]
        # Literal first: a // inside a class like bg-[url(...)] is not a comment.
        if char in _QUOTES:
            index = skip_literal(source, index)
            continue
        if source.startswith("//", index):
            end = source.find("\n", index)
            end = len(source) if end == -1 else end
        elif source.startswith("/*", index):
            end = source.find("*/", index)
            assert end != -1, "unterminated block comment"
            end += 2
        else:
            index += 1
            continue
        for blank in range(index, end):
            if out[blank] != "\n":
                out[blank] = " "
        index = end
    return "".join(out)


def skip_literal(source: str, at: int) -> int:
    """The index just past the string or template literal opening at `at`."""
    quote = source[at]
    index = at + 1
    while index < len(source):
        if source[index] == "\\":
            index += 2
            continue
        if source[index] == quote:
            return index + 1
        index += 1
    raise AssertionError(f"unterminated {quote} literal")


def _tag_end(source: str, start: int, at: int) -> int | None:
    """End of the tag opening at start if at is inside it, else None; brackets and strings are tracked."""
    depth = 0
    index = start + 1
    reached = False
    while index < len(source):
        if index == at:
            reached = depth == 0
        char = source[index]
        if char in _QUOTES:
            index = skip_literal(source, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif char == ">" and depth == 0:
            return index if reached else None
        index += 1
    return None


def opening_tag(source: str, at: int) -> tuple[int, int]:
    """Bounds of the JSX tag holding index at; not the nearest '<', which can sit inside an attribute."""
    start = at
    while True:
        try:
            start = source.rindex("<", 0, start)
        except ValueError:
            raise AssertionError("no JSX opening tag encloses this attribute") from None
        end = _tag_end(source, start, at)
        if end is not None:
            return start, end
