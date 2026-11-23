# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A helper must be declared before any statement that reaches it, or its try/catch hides the failure."""

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
    """Blank here-string bodies, keeping offsets, so embedded Python braces are not counted as
    PowerShell."""
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
    """Blank comment text in place, so helper names quoted in comments cannot look like calls."""
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
    """Match braces to find each nested function's end; scanning to the next `function` overshoots."""
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
    """Source outside nested declarations, which runs in order and sets each helper's deadline."""
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
    """Helpers reachable by calling `name` at `at`; a call before a helper is declared adds no edge."""
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
    """Guards against the scan finding nothing, which would let every ordering check pass on an
    empty set."""
    assert len(FUNCTIONS) > 50, f"only {len(FUNCTIONS)} nested functions found"
    reached = set()
    for start, end in _top_level_spans():
        for called in _calls_within(start, end):
            reached |= _reaches(called)
    assert (
        set(CHECKED) <= reached
    ), f"not reachable from any top-level statement: {sorted(set(CHECKED) - reached)}"
