# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A helper must be declared before the statement that reaches it.

All of install.ps1 is one function, so its nested `function` statements are ordinary statements
that run in order. A helper declared further down the file does not exist yet when an earlier
statement calls it, and the call raises CommandNotFoundException.

That is not a loud failure here. Every rung in these ladders is wrapped in try/catch precisely so
a rung that cannot run on this host is read as "declined" and the one below it is used, so a
helper placed below its own caller degrades the whole rung silently on every host: the batched
path resolver falls back to one interpreter per path, and the cmdlet launcher that exists for
Constrained Language Mode rejects every interpreter it is given. Observed exactly that way.

Declaration order is invisible in a diff and there is no runtime signal for it, which is why it is
checked statically here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = ROOT / "install.ps1"

CHECKED = (
    "New-StudioChildScriptDirectory",
    "Test-StudioChildScriptDirectoryElevated",
    "Test-StudioInterpreterFileIsAdminOnly",
    "Invoke-StudioEarlyPythonScript",
    "Resolve-StudioFinalPathsInOneChild",
    "Get-StudioPythonFinalPath",
    "Get-StudioPythonProcessImageTable",
    "Read-NvidiaLibraryRawViaPython",
)

SOURCE = INSTALL_PS1.read_text(encoding = "utf-8")


def _blank_here_strings(text: str) -> str:
    """Blank out @' ... '@ and @" ... "@ bodies, keeping every offset where it was.

    Those bodies are Python, and Python has braces of its own. Counted as PowerShell braces they
    swallow the rest of the file, and the containment tests below then read almost everything as
    being inside some helper. Found here as exactly that: 26 KB of 608 KB looked top-level.
    """
    out, inside = [], False
    for line in text.split("\n"):
        if inside:
            if line.strip() in ("'@", '"@'):
                inside = False
                out.append(line)
            else:
                out.append(" " * len(line))
            continue
        stripped = line.rstrip()
        if stripped.endswith("@'") or stripped.endswith('@"'):
            inside = True
        out.append(line)
    return "\n".join(out)


def _strip_comments(text: str) -> str:
    """Blank out comment text, keeping every offset where it was.

    The comments in this file NAME these helpers while explaining them, and a name read out of a
    comment would make a declaration look reachable from a statement that does not call it.
    Replacing rather than deleting keeps the offsets comparable with the raw source.
    """
    out = []
    for line in text.split("\n"):
        hash_at = line.find("#")
        if hash_at == -1:
            out.append(line)
        else:
            out.append(line[:hash_at] + " " * (len(line) - hash_at))
    return "\n".join(out)


CODE = _strip_comments(_blank_here_strings(SOURCE))


def _nested_functions(code: str) -> dict[str, tuple[int, int]]:
    """Every `function NAME {` nested one level inside Install-UnslothStudio, with its extent.

    Brace matching rather than a scan to the next `function` keyword, which overshoots into the
    following helper and makes every containment test below wrong in the permissive direction.
    """
    found: dict[str, tuple[int, int]] = {}
    for match in re.finditer(r"(?m)^    function ([A-Za-z0-9\-]+) \{", code):
        start = match.start()
        depth, i = 0, code.index("{", match.start())
        while True:
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        found[match.group(1)] = (start, i + 1)
    return found


FUNCTIONS = _nested_functions(CODE)


def _calls_within(start: int, end: int) -> set[str]:
    """Which of the file's own helpers this span names, ignoring the declarations inside it."""
    span = CODE[start:end]
    inner = {m.group(1) for m in re.finditer(r"(?m)^\s*function ([A-Za-z0-9\-]+) \{", span)}
    return {
        name
        for name in FUNCTIONS
        if name not in inner
        and re.search(rf"(?<![A-Za-z0-9\-]){re.escape(name)}(?![A-Za-z0-9\-])", span)
    }


CALLS = {name: _calls_within(start, end) for name, (start, end) in FUNCTIONS.items()}


def _guarded_within(start: int, end: int) -> set[str]:
    """Helpers this span calls only behind `Get-Command NAME -CommandType Function`."""
    return set(
        re.findall(r"Get-Command ([A-Za-z0-9\-]+) -CommandType Function", CODE[start:end])
    ) & set(FUNCTIONS)


GUARDED = {name: _guarded_within(start, end) for name, (start, end) in FUNCTIONS.items()}


def _top_level_spans() -> list[tuple[int, int]]:
    """Everything in the file that is not inside one of those nested declarations.

    These are the statements that run in order, so the offset of the first one that can reach a
    helper is the deadline that helper's declaration has to beat.
    """
    holes = sorted(FUNCTIONS.values())
    spans, cursor = [], 0
    for start, end in holes:
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    spans.append((cursor, len(CODE)))
    return spans


def _reaches(
    name: str,
    at: int = len(CODE),
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    """Every helper that calling `name` from offset `at` can end up in, including itself.

    A guarded call runs only once its target is declared, so it is no edge before that.
    """
    if name in seen:
        return set()
    out = {name}
    for callee in CALLS.get(name, ()):  # noqa: SIM118 - CALLS may not carry every name
        if callee in GUARDED.get(name, ()) and FUNCTIONS[callee][0] > at:
            continue
        out |= _reaches(callee, at, seen | {name})
    return out


@pytest.mark.parametrize("name", CHECKED)
def test_the_helper_is_declared_before_anything_that_reaches_it(name: str):
    assert name in FUNCTIONS, f"{name} is not declared inside install.ps1"
    declared = FUNCTIONS[name][0]
    for start, end in _top_level_spans():
        if start > declared:
            continue
        guarded = _guarded_within(start, end)
        for called in sorted(_calls_within(start, end)):
            if called in guarded and FUNCTIONS[called][0] > start:
                continue
            if name in _reaches(called, start):
                line = CODE[:start].count("\n") + 1
                assert declared < start, (
                    f"{name} is declared at line {CODE[:declared].count(chr(10)) + 1}, but the "
                    f"statement at line {line} reaches it through {called}. PowerShell runs these "
                    f"declarations in order, so the call raises CommandNotFoundException and the "
                    f"rung is read as declined."
                )


def test_the_rule_is_not_vacuous():
    """The scan really does find declarations and really does find top-level callers.

    Without this, a regex that matched nothing would leave every row above passing on an empty
    set, which is the failure mode this whole file exists to prevent.
    """
    assert len(FUNCTIONS) > 50, f"only {len(FUNCTIONS)} nested functions found"
    reached = set()
    for start, end in _top_level_spans():
        for called in _calls_within(start, end):
            reached |= _reaches(called)
    assert (
        set(CHECKED) <= reached
    ), f"not reachable from any top-level statement: {sorted(set(CHECKED) - reached)}"
