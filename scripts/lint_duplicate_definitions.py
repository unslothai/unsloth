#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse a name bound twice in one scope of a Python file.

This exists for one failure mode: merge resolution. When git splits a function into a
conflict over part of its body, the remainder can auto-merge below the resolved region
as an ordinary addition, leaving two complete top-level copies of it. Nothing about the
result looks wrong. Both copies are valid Python, so `compileall` passes. Ruff's rule set
here (E9 / F63 / F7 / F82) does not include F811, and F811 would not fire on a duplicated
constant or on a name repeated inside a single `from x import a, a` anyway. The two copies
routinely differ only in their comments, so reading the diff does not flag it either.

What it costs when it slips through is either everything or nothing, and nothing is worse:

  - everything: the second copy runs a registration decorator again and the whole suite
    fails to COLLECT, which at least stops the branch.
  - nothing: the second copy silently wins, the first is dead code, and the tests that
    covered the first now cover neither.

Two checks, both AST-only. A regex cannot do this job -- a stray paren inside a comment is
enough to make a hand-rolled scanner truncate and under-report, which is how the first of
these landed.

  1. duplicate definition -- def / async def / class / ALL_CAPS assignment bound twice as a
     DIRECT child of the same module or class body.
  2. duplicate import binding -- a name imported twice inside one `from x import a, a`, or
     twice from the same module by two separate statements.

Deliberately narrow, so a finding is always worth acting on:

  - Only DIRECT children of a module or class body count. One definition per branch of an
    if/else, or a fallback import in try/except, is the normal conditional idiom.
  - @typing.overload and the @property / @x.setter / @x.deleter family are exempt: those
    are several defs legitimately sharing one name.
  - Assignments are ALL_CAPS only, and `X = frozenset(X)` style self-transforms are exempt.
    A lowercase module-level name being rebound is common enough to be noise.
  - `import urllib.parse` beside `import urllib.request` both bind `urllib` and are correct,
    so plain imports are keyed on the full dotted path.

Exit codes: 0 = clean, 1 = findings, 2 = usage error.

Run from repo root:
  python3 scripts/lint_duplicate_definitions.py --self-test
  python3 scripts/lint_duplicate_definitions.py unsloth studio        # paths or dirs
  python3 scripts/lint_duplicate_definitions.py --before SHA --after SHA FILE.py [FILE.py ...]

In --before/--after mode a finding fails only if it sits on a line this diff ADDED. A
duplicate that was already there is printed and does not fail, so the gate blocks the
branch that creates one without blocking an unrelated branch that touches the same file.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

OVERLOAD_DECORATORS = {"overload", "typing.overload", "typing_extensions.overload"}
PROPERTY_DECORATORS = {"property", "cached_property", "functools.cached_property"}
PROPERTY_ATTRS = {"setter", "deleter", "getter"}
SKIP_DIRS = re.compile(r"(^|/)(unsloth_compiled_cache|node_modules|build|dist|\.git|\.venv)/")
HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def _decorator_name(node) -> str:
    """Dotted name of a decorator expression, ignoring any call arguments."""
    if isinstance(node, ast.Call):
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_exempt_def(node) -> bool:
    """True for the decorator families that legitimately bind one name several times."""
    for decorator in getattr(node, "decorator_list", []):
        name = _decorator_name(decorator)
        if name in OVERLOAD_DECORATORS or name in PROPERTY_DECORATORS:
            return True
        if name.count(".") == 1 and name.split(".")[1] in PROPERTY_ATTRS:
            return True
    return False


def _defined_name(node):
    """(name, kind) bound by a direct child of a module/class body, or None."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None if _is_exempt_def(node) else (node.name, "def")
    if isinstance(node, ast.ClassDef):
        return (node.name, "class")
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.isupper():
            # `X = frozenset(X)` / `X = X + (...)` transforms a constant rather than
            # redefining it. Only a value computed WITHOUT the old one replaces it.
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Name) and sub.id == target.id:
                    return None
            return (target.id, "constant")
    return None


def _scope_duplicates(body, scope, out) -> None:
    """Names bound twice as direct children of one body; recurses into class bodies."""
    seen = {}
    for node in body:
        found = _defined_name(node)
        if found is not None:
            name, kind = found
            if name in seen:
                first_line, first_kind = seen[name]
                out.append(
                    (
                        node.lineno,
                        f"{scope}{name} is defined twice "
                        f"({first_kind} at line {first_line}, {kind} here)",
                    )
                )
            else:
                seen[name] = (node.lineno, kind)
        if isinstance(node, ast.ClassDef):
            _scope_duplicates(node.body, f"{scope}{node.name}.", out)


def _import_duplicates(tree, out) -> None:
    """A name imported twice within one statement, or twice from the same module."""
    seen = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            # `level` carries the leading dots, so `from .a` and `from ..a` stay distinct.
            module = f"{'.' * node.level}{node.module or ''}"
        elif isinstance(node, ast.Import):
            module = None
        else:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            key = (module, alias.name, alias.asname)
            bound = alias.asname or alias.name
            if key in seen:
                where = (
                    "twice in this statement"
                    if seen[key] == node.lineno
                    else f"twice from the same module (first at line {seen[key]})"
                )
                out.append((node.lineno, f"{bound} is imported {where}"))
            else:
                seen[key] = node.lineno


def scan_source(source: str, filename: str = "<unknown>"):
    """[(line, message)] for one Python source string. A file that does not parse is a finding."""
    try:
        tree = ast.parse(source, filename = filename)
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"does not parse ({exc.msg})")]
    found = []
    _scope_duplicates(tree.body, "", found)
    _import_duplicates(tree, found)
    return sorted(found)


def _git(args, cwd = None):
    return subprocess.run(["git", *args], cwd = cwd, capture_output = True, text = True)


def _added_lines(before: str, after: str, path: str):
    """Line numbers of `path` in `after` that this diff introduced, or None if unknown."""
    result = _git(["diff", "-U0", before, after, "--", path])
    if result.returncode != 0:
        return None
    lines = set()
    for line in result.stdout.splitlines():
        match = HUNK.match(line)
        if match:
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            lines.update(range(start, start + count))
    return lines


def _iter_paths(targets):
    for target in targets:
        path = Path(target)
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                if not SKIP_DIRS.search(child.as_posix()):
                    yield child
        elif path.suffix == ".py":
            yield path


_SELF_TEST_CASES = [
    # (expected finding count, source)
    # 1. The real artefact: two top-level copies of one def, differing only in comments.
    (
        1,
        "def toggle(page):\n    # click it\n    page.click('#r')\n\n\n"
        "def toggle(page):\n    # open the panel\n    page.click('#r')\n",
    ),
    # 2. The other real one: a name repeated inside a single ImportFrom.
    (1, "from floor_table import latest_attempt_rows, refuse_collisions, latest_attempt_rows\n"),
    # 3. A constant defined twice.
    (1, "TOGGLE_JS = '() => 1'\nOTHER = 2\nTOGGLE_JS = '() => 2'\n"),
    # 4. A class, and a method inside one.
    (1, "class A:\n    pass\n\n\nclass A:\n    pass\n"),
    (1, "class A:\n    def go(self):\n        pass\n\n    def go(self):\n        pass\n"),
    # 5. The same name imported twice from the same module by two statements.
    (1, "from a import x\nfrom a import x\n"),
    # 6. A stray paren inside a comment must not truncate the scan (the regex version did).
    (1, "def go():  # takes (a, b\n    pass\n\n\ndef go():\n    pass\n"),
    # 7. A conflict marker left in the tree does not parse, and that is a finding.
    (1, "<<<<<<< HEAD\ndef go():\n    pass\n"),
    # Negative controls: each of these is correct code and must report nothing.
    (0, "if FAST:\n    def go():\n        pass\nelse:\n    def go():\n        pass\n"),
    (0, "try:\n    from fast import x\nexcept ImportError:\n    from slow import x\n"),
    (
        0,
        "from typing import overload\n\n\n@overload\ndef go(a: int) -> int: ...\n"
        "@overload\ndef go(a: str) -> str: ...\ndef go(a):\n    return a\n",
    ),
    (
        0,
        "class A:\n    @property\n    def v(self):\n        return self._v\n\n"
        "    @v.setter\n    def v(self, x):\n        self._v = x\n",
    ),
    (0, "import urllib.parse\nimport urllib.request\n"),
    (0, "from a import x\nfrom b import x\n"),
    (0, "NAMES = ['a']\nNAMES = frozenset(n.lower() for n in NAMES)\n"),
    (0, "ROWS = (1,)\nROWS = ROWS + (2,)\n"),
    (0, "value = 1\nvalue = 2\n"),
    (0, "import os\nfrom a import b\n\nX = 1\n\n\ndef go():\n    return os, b, X\n"),
]


def _self_test() -> int:
    failures = 0
    for expected, source in _SELF_TEST_CASES:
        got = len(scan_source(source, "<self-test>"))
        if got != expected:
            failures += 1
            print(
                f"self-test: expected {expected} finding(s), got {got} for:\n{source}",
                file = sys.stderr,
            )
    if failures:
        print(f"self-test FAILED ({failures} case(s))", file = sys.stderr)
        return 1
    print(f"self-test passed ({len(_SELF_TEST_CASES)} cases)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("targets", nargs = "*", help = "Python files or directories to scan")
    parser.add_argument("--self-test", action = "store_true", help = "check the rule, scan nothing")
    parser.add_argument("--before", help = "base revision; findings on unchanged lines only warn")
    parser.add_argument("--after", help = "head revision, used with --before")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if bool(args.before) != bool(args.after):
        print("--before and --after must be given together", file = sys.stderr)
        return 2
    if not args.targets:
        print("nothing to scan (pass files or directories, or --self-test)", file = sys.stderr)
        return 2

    paths = list(_iter_paths(args.targets))
    blocking, existing = [], []
    for path in paths:
        added = None
        if args.before:
            # Read the AFTER revision out of git rather than the working tree. On a
            # pull_request event the checkout is refs/pull/N/merge, whose line numbers
            # do not have to match the head SHA the diff is measured against, and the
            # added-line gate is only meaningful if both ends agree on numbering.
            shown = _git(["show", f"{args.after}:{path}"])
            if shown.returncode != 0:
                continue  # not present at the head revision (deleted, or renamed away)
            source = shown.stdout
            added = _added_lines(args.before, args.after, str(path))
        else:
            try:
                source = path.read_text(encoding = "utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                blocking.append(f"{path}: unreadable ({exc})")
                continue
        for line, message in scan_source(source, str(path)):
            finding = f"{path}:{line}: {message}"
            if added is None or line in added:
                blocking.append(finding)
            else:
                existing.append(finding)

    if existing:
        print(f"pre-existing in files this diff touches ({len(existing)}; not failing):")
        for finding in existing:
            print(f"  {finding}")
    if blocking:
        print(f"duplicate definitions ({len(blocking)}):")
        for finding in blocking:
            print(f"  {finding}")
        print(
            "A name bound twice in one scope is almost always merge damage: delete the "
            "duplicate copy. The later one wins silently, so the first is dead code and "
            "whatever tested it now tests nothing."
        )
        return 1
    print(f"clean: {len(paths)} file(s) checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
