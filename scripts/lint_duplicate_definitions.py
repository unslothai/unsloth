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
  - @typing.overload and the accessor family @x.setter / @x.deleter / @x.getter are exempt:
    those are several defs legitimately sharing one name. A bare @property is NOT, since it
    is the FIRST binding of the name and two of them are ordinary merge damage.
  - Assignments are ALL_CAPS only, and `X = frozenset(X)` style self-transforms are exempt.
    A lowercase module-level name being rebound is common enough to be noise.
  - `import urllib.parse` beside `import urllib.request` both bind `urllib` and are correct,
    so plain imports are keyed on the full dotted path AND the name they bind.

Exit codes: 0 = clean, 1 = findings, 2 = usage error.

Run from repo root:
  python3 scripts/lint_duplicate_definitions.py --self-test
  python3 scripts/lint_duplicate_definitions.py unsloth studio        # paths or dirs
  python3 scripts/lint_duplicate_definitions.py --before SHA --after SHA FILE.py [FILE.py ...]

In --before/--after mode both revisions are scanned and their findings compared by identity,
so a finding fails only if the diff INTRODUCED it. A duplicate that was already there is
printed and does not fail, which keeps the gate on the branch that creates one without
blocking an unrelated branch that touches the same file.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple


class Finding(NamedTuple):
    """`key` identifies WHICH duplicate this is, independently of where it sits.

    Compare mode needs that. Gating on "is this line new" is not sound: a merge that inserts
    the FIRST copy above an existing definition reports on the SECOND, unchanged line, and a
    duplicate alias added to a multi-line import reports at the statement's opening line.
    Both are the exact bug this gate exists for, and both would read as pre-existing. So the
    before and after revisions are scanned and their finding keys compared.
    """

    line: int
    key: str
    message: str


OVERLOAD_DECORATORS = {"overload", "typing.overload", "typing_extensions.overload"}
PROPERTY_ATTRS = {"setter", "deleter", "getter"}
SKIP_DIRS = re.compile(r"(^|/)(unsloth_compiled_cache|node_modules|build|dist|\.git|\.venv)/")


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
    """True for the decorator families that legitimately bind one name several times.

    `@property` itself is NOT one of them. It is the FIRST binding of the name, so
    exempting it hid two complete copies of a property in one class, which is the same
    merge damage as any other duplicated def. What legitimately rebinds is the accessor
    form, `@v.setter` / `@v.deleter` / `@v.getter`, and that stays exempt.
    """
    for decorator in getattr(node, "decorator_list", []):
        name = _decorator_name(decorator)
        if name in OVERLOAD_DECORATORS:
            return True
        if name.count(".") == 1 and name.split(".")[1] in PROPERTY_ATTRS:
            return True
    return False


def _constant_targets(target):
    """The uppercase names a single assignment target binds.

    A tuple target is not one binding but several: `B, H, N, D = 1, 16, 50345, 128` is a
    shape this repo declares module constants with, and duplicating that line rebinds
    every one of them at once.
    """
    if isinstance(target, ast.Name):
        return [target.id] if target.id.isupper() else []
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _constant_targets(element)]
    return []


def _defined_names(node):
    """[(name, kind)] bound by a direct child of a module/class body."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [] if _is_exempt_def(node) else [(node.name, "def")]
    if isinstance(node, ast.ClassDef):
        return [(node.name, "class")]
    # `X: int = 1` is an AnnAssign, not an Assign, and a typed module-level constant is
    # common in this repo (MAX_FUSED_SIZE: int = 65536). Duplicating one is the same bug.
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target, value = node.targets[0], node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target, value = node.target, node.value
    else:
        return []
    # `X = frozenset(X)` / `X = X + (...)` transforms a constant rather than redefining
    # it. Only a value computed WITHOUT the old one replaces it, decided per name so
    # that `A, B = B, A` exempts both while `A, B = 1, 2` exempts neither.
    referenced = {sub.id for sub in ast.walk(value) if isinstance(sub, ast.Name)}
    return [
        (name, "constant") for name in _constant_targets(target) if name not in referenced
    ]


def _scope_duplicates(body, scope, out) -> None:
    """Names bound twice as direct children of one body; recurses into class bodies."""
    seen = {}
    for node in body:
        for name, kind in _defined_names(node):
            if name in seen:
                first_line, first_kind = seen[name]
                out.append(
                    Finding(
                        node.lineno,
                        f"def:{scope}{name}",
                        f"{scope}{name} is defined twice "
                        f"({first_kind} at line {first_line}, {kind} here)",
                    )
                )
            else:
                seen[name] = (node.lineno, kind)
        if isinstance(node, ast.ClassDef):
            _scope_duplicates(node.body, f"{scope}{node.name}.", out)


def _import_duplicates(body, scope, out) -> None:
    """A name imported twice within one statement, or twice from the same module.

    Keyed on the name each alias BINDS, not on the name it comes from: `from m import x as v`
    followed by `from m import y as v` binds `v` twice and the second silently wins, which is
    the dead-binding this is here to catch. Plain `import a.b` keeps the full dotted path in
    its key too, since `import urllib.parse` beside `import urllib.request` is correct.
    """
    seen = {}
    for node in body:
        if isinstance(node, ast.ImportFrom):
            # `level` carries the leading dots, so `from .a` and `from ..a` stay distinct.
            module = f"{'.' * node.level}{node.module or ''}"
        elif isinstance(node, ast.Import):
            module = None  # the source is per-alias for a plain import, not per-statement
        else:
            # Class bodies are scanned too: two `from m import x` inside one class bind x
            # twice in that scope exactly as they would at module level.
            if isinstance(node, ast.ClassDef):
                _import_duplicates(node.body, f"{scope}{node.name}.", out)
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            # `import a.b` binds `a`, not `a.b`; only an `as` clause binds the dotted name.
            # Using `alias.asname` raw here instead missed `import urllib.parse` followed by
            # `import urllib.parse as urllib`, which silently repoints `urllib` from the
            # package to the submodule.
            source = module if module is not None else alias.name
            bound = alias.asname or (alias.name if module is not None else alias.name.split(".")[0])
            # `from m import x as v` and `from m import y as v` both bind v, so a from-import
            # is keyed on what it binds. A plain import is keyed on the dotted path AND the
            # bound name: `import sys as _sys` beside `import sys` binds two different names
            # and is correct, and so is `import urllib.parse` beside `import urllib.request`.
            key = (source, bound)
            if key in seen:
                where = (
                    "twice in this statement"
                    if seen[key] == node.lineno
                    else f"twice from the same module (first at line {seen[key]})"
                )
                out.append(
                    Finding(
                        node.lineno,
                        # The emitted identity must mirror the detection key. Keyed on the
                        # bound name alone, two DIFFERENT plain-import duplicates both read
                        # as `import:None:x`, so compare mode charged a newly introduced one
                        # against a pre-existing counter entry and passed.
                        f"import:{scope}{source}:{bound}",
                        f"{scope}{bound} is imported {where}",
                    )
                )
            else:
                seen[key] = node.lineno


def scan_source(source: str, filename: str = "<unknown>"):
    """Findings for one Python source string. A file that does not parse is itself a finding."""
    try:
        tree = ast.parse(source, filename = filename)
    except SyntaxError as exc:
        return [Finding(exc.lineno or 0, "parse", f"does not parse ({exc.msg})")]
    found = []
    _scope_duplicates(tree.body, "", found)
    _import_duplicates(tree.body, "", found)
    return sorted(found)


def _git(args, cwd = None):
    return subprocess.run(["git", *args], cwd = cwd, capture_output = True, text = True)


def _revision_findings(revision: str, path: str):
    """Findings for `path` at `revision`, or None if the file does not exist there."""
    shown = _git(["show", f"{revision}:{path}"])
    if shown.returncode != 0:
        return None
    return scan_source(shown.stdout, f"{revision}:{path}")


def _rename_map(before: str, after: str):
    """{new path: old path} for the renames in this range.

    The changed-file sweep reports a renamed file under its NEW name, so looking the before
    side up under that name finds nothing and every finding in it reads as introduced. A
    branch that only moves a file carrying one of the duplicates already on main would then
    be blocked for a duplicate it did not write.
    """
    # -z, because git quotes and backslash-escapes any non-ASCII path by default
    # (core.quotePath), and the escaped spelling matches nothing. Under -z each rename is
    # three NUL-terminated records, "R100", old, new -- the status is its own field rather
    # than tab-joined to the paths.
    listed = _git(["diff", "--name-status", "-z", "-M", "--diff-filter=R", before, after])
    if listed.returncode != 0:
        return {}
    fields = [field for field in listed.stdout.split("\0") if field]
    renames = {}
    for index in range(0, len(fields) - 2, 3):
        status, old, new = fields[index : index + 3]
        if status.startswith("R"):
            renames[new] = old
    return renames


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
    # 8. A typed constant is an AnnAssign, and duplicating one is the same bug.
    (1, "MAX_FUSED_SIZE: int = 65536\nMAX_FUSED_SIZE: int = 131072\n"),
    (1, "REGISTRY: dict = {}\nOTHER = 1\nREGISTRY = {'a': 1}\n"),
    # 9. Two aliases from one module landing on the same bound name: the second wins and
    #    the first is dead, which is the silent half of this bug.
    (1, "from m import x as value\nfrom m import y as value\n"),
    (1, "from m import x as value, y as value\n"),
    # 10. A class body is a scope too, for imports as well as for defs.
    (1, "class A:\n    from m import x\n    from m import x\n"),
    # 11. A duplicated @property getter. Exempting @property hid two complete copies.
    (
        1,
        "class A:\n    @property\n    def v(self):\n        return 1\n\n"
        "    @property\n    def v(self):\n        return 2\n",
    ),
    (
        1,
        "import functools\n\n\nclass A:\n    @functools.cached_property\n"
        "    def v(self):\n        return 1\n\n    @functools.cached_property\n"
        "    def v(self):\n        return 2\n",
    ),
    # 12. Constants declared together in one tuple target: duplicating the line rebinds
    #     every one of them, so each is its own finding.
    (4, "B, H, N, D = 1, 16, 50345, 128\nB, H, N, D = 2, 3, 4, 5\n"),
    # 13. A plain import binds the ROOT of its dotted path, so this repoints `urllib` from
    #     the package to the submodule and the first binding is dead.
    (1, "import urllib.parse\nimport urllib.parse as urllib\n"),
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
    (0, "import sys as _sys\nimport sys\n"),
    (0, "from a import x\nfrom b import x\n"),
    (0, "NAMES = ['a']\nNAMES = frozenset(n.lower() for n in NAMES)\n"),
    (0, "COUNT: int = 1\nCOUNT = COUNT + 1\n"),
    (0, "REGISTRY: dict\nREGISTRY = {}\n"),  # a bare annotation binds nothing
    (0, "from m import x\nfrom m import x as other\n"),
    (0, "ROWS = (1,)\nROWS = ROWS + (2,)\n"),
    # A tuple self-transform is decided per name, so a swap exempts both sides.
    (0, "A = 1\nB = 2\nA, B = B, A\n"),
    # Distinct submodules of one package each bind the package name and are correct.
    (0, "import urllib.parse as parse\nimport urllib.request as request\n"),
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
    parser.add_argument("--before", help = "base revision; only findings this diff adds fail")
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
    renames = _rename_map(args.before, args.after) if args.before else {}
    for path in paths:
        if args.before:
            # Read both revisions out of git rather than the working tree. On a
            # pull_request event the checkout is refs/pull/N/merge, which is neither end
            # of the range being judged.
            after = _revision_findings(args.after, str(path))
            if after is None:
                continue  # not present at the head revision (deleted, or renamed away)
            before = _revision_findings(args.before, renames.get(str(path), str(path)))
            # A file the branch ADDS has no before side, so every finding in it is new.
            was = Counter(f.key for f in (before or []))
            for finding in after:
                text = f"{path}:{finding.line}: {finding.message}"
                if was[finding.key] > 0:
                    was[finding.key] -= 1
                    existing.append(text)
                else:
                    blocking.append(text)
        else:
            try:
                source = path.read_text(encoding = "utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                blocking.append(f"{path}: unreadable ({exc})")
                continue
            for finding in scan_source(source, str(path)):
                blocking.append(f"{path}:{finding.line}: {finding.message}")

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
