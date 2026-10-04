# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reading JSX out of a source file without pretending it is a regular language.

Written for tests/studio/test_update_release_notes.py. They live here because a second file
needed them and hand-rolled a naive version instead: `src.rfind("<")` to find an element's
opening tag, which starts inside `disabled={count < limit}` and drops every class before it,
and `src.find(">")`, which stops inside `onClick={() => go()}`. A layering test built on that
reads a truncated tag and passes over the very regression it exists to catch.

tests/conftest.py puts this directory on sys.path for everything under tests/, which is what
lets both callers share one implementation rather than each growing a private copy.

The behaviour is pinned by test_update_release_notes.py::test_the_class_anchors_do_not_depend_on_any_order,
which covers a comparison before the test id, an arrow function after it, attributes on
either side of className, comments holding apostrophes and unmatched braces, and a `//`
inside a URL literal that is not a comment.
"""

from __future__ import annotations

import re

# A JSX attribute value is double-quoted or a template literal. A single quote is an
# apostrophe in prose far more often than a delimiter here, and treating it as one opens a
# literal that never closes.
_QUOTES = frozenset('"`')
_COMMENT_SPAN = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def without_comments(source: str) -> str:
    """`source` with every comment blanked, each index left where it was.

    Blanked rather than removed so that the offsets the scanners hand around
    stay valid. Both forms, and before any scan, for two reasons. Prose is not
    code: an apostrophe in `// notes don't shrink` would open a string literal
    that never closes, and a `}` written in a block comment would unbalance the
    tag. Prose is not classes either: the comment beside these very rules says
    "shrink-0 keeps the compact card at its natural height", so in block form
    it would satisfy the assertion that the class is there after the class
    itself had been deleted.
    """
    out = list(source)
    index = 0
    while index < len(source):
        char = source[index]
        # A literal first: `bg-[url(https://example.com/a.svg)]` is a class, and
        # blanking from its `//` would eat the rest of the line and its quote.
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
    """The end of the tag opening at `start`, if `at` is one of its attributes.

    `None` when it is not, which is how a `<` that opens no tag is rejected.
    Brackets and string literals are tracked, so the `>` of an inline arrow
    (`onClick={() => go()}`) does not end the tag early and a comparison inside
    an attribute expression (`disabled={count < limit}`) runs out of depth.
    """
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
    """The bounds of the JSX opening tag whose attributes include index `at`.

    Not simply the nearest `<` before it: an attribute expression may hold one
    of its own, as `disabled={count < limit}` does, and starting the scan there
    runs into an unmatched brace. Candidates are tried from the nearest
    outwards and one is accepted only if the tag it opens actually reaches `at`
    with the tag still open and at depth zero.

    Both ends are returned so that attributes can be searched over the whole
    tag rather than the part before some other attribute, which is an order
    dependency of exactly the kind this file is being fixed for.
    """
    start = at
    while True:
        try:
            start = source.rindex("<", 0, start)
        except ValueError:
            raise AssertionError("no JSX opening tag encloses this attribute") from None
        end = _tag_end(source, start, at)
        if end is not None:
            return start, end
