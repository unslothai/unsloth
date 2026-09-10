# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read conditions out of frontend source without pinning the text that spells them.

Several tests in this directory protect a frontend contract by reading a `.ts` or `.tsx`
file and asserting on what it says. Done as an exact substring that is a trap: prettier
rewraps a guard the moment it gains a condition, a binding gets renamed, and a test that
was protecting real behaviour goes red over a refactor that changed none of it. Three tests
broke that way on `main` inside one week.

So compare CONDITIONS. Blank out the parts of the source that only look like code, pull out
the guard bodies, split them into operands, and ask whether the operands the contract needs
are present. A rename, a rewrap, an added clause and a redundant paren all survive that;
dropping a condition does not, which is the whole point.

None of this is a JavaScript parser, and it is not trying to be. It handles the constructs
that appear in the files these tests actually read, and fails loudly rather than silently
when it meets something it cannot represent.
"""

import re


def blank_literals_and_comments(source: str) -> str:
    """`source` with comments and the INSIDE of string literals blanked, length preserved.

    Every earlier version of this scan handled one of these and was defeated by the other.
    Stripping comments with a regex eats a `//` that lives inside a URL string; tracking
    quotes without stripping comments lets a commented-out guard count as live; and doing
    both separately still counts a guard-shaped STRING as a live guard, which masks a real
    condition being dropped. One pass settles all three: walk the source once, and replace
    comment bodies and literal contents with spaces so offsets and delimiters still line up.

    Regex literals are left alone. Telling `/` as division from `/` as a regex needs real
    parsing, and no guard this module is pointed at contains one; a guard that did would be
    read with its slashes intact, which is visible rather than silent.
    """
    out, i, n = [], 0, len(source)
    while i < n:
        char, nxt = source[i], source[i + 1 : i + 2]
        if char == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if char == "/" and nxt == "*":
            while i < n and not (source[i] == "*" and source[i + 1 : i + 2] == "/"):
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
            continue
        if char in "\"'`":
            out.append(char)
            i += 1
            while i < n:
                if source[i] == "\\":
                    out.append("  ")
                    i += 2
                    continue
                if source[i] == char:
                    out.append(char)
                    i += 1
                    break
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


def parenthesised_bodies(source: str, keyword: str):
    """Each `keyword (...)` body in `source`, whitespace collapsed, parentheses balanced."""
    source = blank_literals_and_comments(source)
    for match in re.finditer(rf"\b{re.escape(keyword)}\s*\(", source):
        depth, i = 0, match.end() - 1
        while i < len(source):
            char = source[i]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    yield re.sub(r"\s+", " ", source[match.end() : i]).strip()
                    break
            i += 1


def depth_zero_split(text: str, operator: str) -> list[str]:
    """`text` cut at every `operator` sitting outside parentheses. No unwrapping."""
    parts, depth, current, i = [], 0, "", 0
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and text[i : i + len(operator)] == operator:
            parts.append(current)
            current = ""
            i += len(operator)
            continue
        current += char
        i += 1
    parts.append(current)
    return parts


def balanced(text: str) -> bool:
    """True when every parenthesis in `text` closes within it."""
    depth = 0
    for char in text:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    return depth == 0


def split_operands(text: str, operator: str) -> list[str]:
    """`text` split on `operator` at depth zero, each operand unwrapped and flattened.

    Redundant grouping is not a contract change: `(a || b) || c` guards exactly what
    `a || b || c` guards. Splitting at depth zero alone would hand back `(a || b)` whole and
    report the guard incomplete, turning a harmless regroup into a red build.

    Recursion only follows a split that made progress. An operator sitting inside a string
    literal survives every unwrap, so recursing on "contains the operator" never terminates.
    """
    operands = []
    for part in depth_zero_split(text, operator):
        part = part.strip()
        while part.startswith("(") and part.endswith(")") and balanced(part[1:-1]):
            part = part[1:-1].strip()
        inner = depth_zero_split(part, operator)
        operands.extend(
            [piece for sub in inner for piece in split_operands(sub, operator)]
            if len(inner) > 1
            else [part]
        )
    return [part for part in operands if part]


def assert_guard_holds(
    source: str, keyword: str, operator: str, required: set[str], *, expected: int
) -> None:
    """Exactly `expected` `keyword` guards must join all of `required` with `operator`.

    Four things this must not do, each of which it did at some point:

    - Take the first `keyword` in the slice: an unrelated earlier `if` gets parsed instead.
    - Stop at the first `)`: a condition holding a call closes a paren of its own, which
      truncates the body.
    - Split on both `||` and `&&`: the operator carries the meaning, since an OR guard
      flipped to AND stops short-circuiting.
    - Accept the first guard that matches, or demand every guard mentioning a condition
      holds them all. These slices carry the guard twice, so accepting one lets the other
      rot; but nearby guards legitimately test a subset, so requiring all of them is wrong
      too. Counting the complete ones catches a dropped condition in either copy and leaves
      the neighbours alone.
    """
    complete = [
        body
        for body in parenthesised_bodies(source, keyword)
        if required <= set(split_operands(body, operator))
    ]
    assert len(complete) == expected, (
        f"expected {expected} {keyword} guards joining {sorted(required)} with {operator!r}, "
        f"found {len(complete)}: {complete}"
    )


def binding_joining(source: str, operator: str, required: set[str]) -> str | None:
    """Name of the first `const NAME = ...` whose operands cover `required`, else None.

    Declarations are LOCATED in the blanked source, so a commented-out or quoted copy
    cannot answer, and then READ from the original at the same offsets, because blanking
    is length-preserving. Reading the blanked text instead would be self-defeating here:
    it empties string literals, and `status === "running"` is a required operand.
    """
    blanked = blank_literals_and_comments(source)
    for match in re.finditer(r"const (\w+) =([^;]*);", blanked):
        operand_text = source[match.start(2) : match.end(2)]
        if required <= set(split_operands(re.sub(r"\s+", " ", operand_text), operator)):
            return match.group(1)
    return None


def gates_the_markup(source: str, name: str) -> bool:
    """True when `name` conditions mounted markup, by `&&` or by a ternary's TRUE arm.

    The ternary arm matters: `{gate ? null : <Shell />}` reads as a guard and inverts one,
    mounting exactly when the gate is false. Accepting any `?` would pass that.
    """
    if re.search(rf"{{\s*{re.escape(name)}\s*&&", source):
        return True
    arm = re.search(rf"{{\s*{re.escape(name)}\s*\?(.*)", source, re.S)
    return bool(arm) and not re.match(r"\s*(null|undefined|false)\b", arm.group(1))
