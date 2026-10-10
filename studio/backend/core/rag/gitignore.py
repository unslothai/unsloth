# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The .gitignore rules a linked-folder scan honours, per https://git-scm.com/docs/gitignore."""

from __future__ import annotations

import re
from dataclasses import dataclass

# A .gitignore is a few KB; a huge one is not worth parsing on every 30 s scan.
MAX_GITIGNORE_BYTES = 256 * 1024
_MAX_GLOBSTARS = 3


@dataclass(frozen = True)
class Rule:
    base: str  # posix directory of the .gitignore, relative to the scan root; "" for the root
    regex: re.Pattern[str]
    negate: bool
    dir_only: bool


def _tokens(pattern: str) -> list[tuple[str, str]]:
    """(kind, regex) pairs: "star", "char" (exactly one non-"/" character), or "other"."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**", i):
                before = i == 0 or pattern[i - 1] == "/"
                after = i + 2 == n or pattern[i + 2] == "/"
                if before and after:
                    if i + 2 == n:
                        # Trailing "/**": everything inside.
                        out.append(("other", ".*"))
                    else:
                        # Leading "**/" or a middle "/**/": zero or more directories.
                        out.append(("other", "(?:.*/)?"))
                        i += 1
                    i += 2
                    continue
            out.append(("star", "[^/]*"))
        elif c == "?":
            out.append(("char", "[^/]"))
        elif c == "[":
            start = i + 1
            if pattern[start : start + 1] in ("!", "^"):
                start += 1
            # A "]" right after the opening (or its negation) is a literal member.
            if pattern[start : start + 1] == "]":
                start += 1
            end = pattern.find("]", start)
            if end == -1:
                out.append(("char", re.escape(c)))
            else:
                body = pattern[i + 1 : end].replace("\\", "\\\\").replace("[", "\\[")
                if body[:1] in ("!", "^"):
                    # Like "*" and "?", a negated class never matches "/".
                    body = "^/" + body[1:]
                out.append(("char", "[" + body + "]"))
                i = end
        else:
            if c == "\\" and i + 1 < n:
                i += 1
                c = pattern[i]
            out.append(("other" if c == "/" else "char", re.escape(c)))
        i += 1
    return out


def _translate(pattern: str) -> str:
    """Glob to regex over a "/"-separated path relative to the rule's base."""
    tokens = _tokens(pattern)
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        kind, regex = tokens[i]
        if kind != "star":
            out.append(regex)
            i += 1
            continue
        while i < n and tokens[i][0] == "star":
            i += 1
        end = i
        while end < n and tokens[end][0] == "char":
            end += 1
        chunk = "".join(regex for _, regex in tokens[i:end])
        if end < n and tokens[end][0] == "star":
            # Another "*" follows in this name, so the earliest fit of the fixed-width chunk is as
            # good as any later one: commit to it (an atomic group, written as fnmatch.translate
            # does for Python < 3.11). Plain [^/]* groups backtrack exponentially on "*a*a*a...b".
            group = f"g{len(out)}"
            out.append(f"(?=(?P<{group}>[^/]*?{chunk}))(?P={group})")
        else:
            out.append("[^/]*" + chunk)
        i = end
    return "".join(out)


def parse(text: str, base: str) -> list[Rule]:
    rules: list[Rule] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        # Trailing spaces are dropped unless escaped with a backslash.
        stripped = line.rstrip(" ")
        if stripped.endswith("\\") and len(stripped) < len(line):
            stripped += " "
        line = stripped
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        elif line.startswith(("\\!", "\\#")):
            line = line[1:]
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        if not line:
            continue
        # A slash at the start or middle anchors the pattern to the .gitignore's directory;
        # otherwise it matches at any depth below it.
        anchored = "/" in line
        line = line.lstrip("/")
        body = _translate(line)
        if not anchored:
            body = "(?:.*/)?" + body
        # Each "**/" tries every directory depth, multiplied across them; real rules use one or two.
        if body.count("(?:.*/)?") > _MAX_GLOBSTARS:
            continue
        try:
            regex = re.compile(f"^{body}$", re.DOTALL)
        except re.error:
            # A malformed class like "[z-a]" matches nothing in git; it must not fail the scan.
            continue
        rules.append(Rule(base, regex, negate, dir_only))
    return rules


def is_ignored(rel: str, is_dir: bool, rules: tuple[Rule, ...]) -> bool:
    """Whether `rel` (posix, relative to the scan root) is ignored; the last matching rule wins."""
    ignored = False
    for rule in rules:
        if rule.dir_only and not is_dir:
            continue
        if rule.base:
            prefix = rule.base + "/"
            if not rel.startswith(prefix):
                continue
            sub = rel[len(prefix) :]
        else:
            sub = rel
        if rule.regex.match(sub):
            ignored = not rule.negate
    return ignored
