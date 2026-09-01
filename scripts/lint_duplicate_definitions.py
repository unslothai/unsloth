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

  - Branches are scanned INDEPENDENTLY of each other. One definition per branch of an
    if/else, or a fallback import in try/except, is the normal conditional idiom and stays
    clean; two copies inside ONE branch is the same merge damage as anywhere else.
  - @typing.overload is exempt: any number of copies is legitimate. The accessor family is
    exempt PER KIND -- one @x.setter, one @x.deleter, one @x.getter -- so a second accessor
    of the same kind is still a finding. A bare @property is NOT exempt at all, since it is
    the FIRST binding of the name and two of them are ordinary merge damage.
  - Assignments are ALL_CAPS only, and `X = frozenset(X)` style self-transforms are exempt.
    A lowercase module-level name being rebound is common enough to be noise.
  - Imports are keyed on the name they BIND, with one carve-out: an IMPLICIT binding takes
    whatever its source is called, so two of them from different sources are the ordinary
    `import urllib.parse` / `import urllib.request` and `from a import x` / `from b import x`
    shapes and stay legitimate. An explicit `as` alias is a name the author chose, so a second
    binding of it is always a dead first binding.

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
import io
import re
import subprocess
import sys
import tokenize
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


def _overload_names(tree) -> set:
    """Every spelling of `typing.overload` this module actually binds.

    The exact-name set alone reads the decorator as written, so `import typing as t` with
    `@t.overload`, or `from typing import overload as ov` with `@ov`, was not recognised and
    the overload signatures were reported as duplicate definitions. That is the wrong way for
    a gate to be wrong: a conventional typing pattern would block CI on correct code.

    MODULE-SCOPED, and not by a whole-tree walk. An alias bound INSIDE a function is not in
    effect at module level, so collecting it globally let a local `import typing as t` exempt
    module-level `@t.overload` definitions where `t` is something else entirely -- widening the
    exemption is the one direction that hides merge damage.

    Nested statements are still descended into, because the conventional spellings are wrapped:
    `try: from typing import overload / except ImportError: from typing_extensions import ...`
    is an ast.Try at module level. Function and class bodies are what get skipped.
    """
    return _overload_names_in(tree.body, OVERLOAD_DECORATORS)


def _overload_names_in(body, base) -> set:
    """`base` plus every overload spelling bound directly by THIS body.

    Called once for the module and again for each class body, because a class that does its own
    `import typing as t` binds `t` in the class namespace, where its `@t.overload` methods resolve
    it. Collecting only module-level aliases reported those valid overloads as duplicates, which
    blocks CI on correct code.
    """
    names: set = set(base)
    modules = {"typing", "typing_extensions"}

    def visit(nested) -> None:
        for node in nested:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # a binding in here belongs to a scope of its own, not to this one
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules and alias.asname:
                        names.add(f"{alias.asname}.overload")
            elif isinstance(node, ast.ImportFrom) and node.module in modules and not node.level:
                for alias in node.names:
                    if alias.name == "overload" and alias.asname:
                        names.add(alias.asname)
            for field in ("body", "orelse", "finalbody"):
                visit(getattr(node, field, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                visit(handler.body)
            # Same blind spot as _branch_bodies, mirrored: an alias bound inside a match case would go unseen and its
            # @t.overload defs would read as a duplicate.
            for case in getattr(node, "cases", []) or []:
                visit(case.body)

    visit(body)
    return names


def _branch_bodies(node):
    """Each nested statement body of a control-flow node, to be scanned INDEPENDENTLY.

    One definition per branch of an if/else, or a fallback import in try/except, is the normal
    conditional idiom, so the branches are never compared with each other or with the body around
    them. But two copies of one def inside ONE branch is the same merge damage as anywhere else,
    and skipping the branch entirely meant never looking.

    Functions and classes are excluded: a function body is a scope this gate does not scan, and a
    class body is recursed into separately so it gets its own name prefix and its own aliases.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    for field in ("body", "orelse", "finalbody"):
        nested = getattr(node, field, None)
        if isinstance(nested, list) and nested:
            yield nested
    for handler in getattr(node, "handlers", []) or []:
        if handler.body:
            yield handler.body
    # A match keeps its branches in `cases[*].body`, so the loop above walked straight past them and two copies of one
    # def inside a SINGLE case scanned clean.
    for case in getattr(node, "cases", []) or []:
        if case.body:
            yield case.body


def _is_overload_def(node, overloads) -> bool:
    """True for `@overload`, the one family where any number of copies is legitimate."""
    return any(
        _decorator_name(decorator) in overloads for decorator in getattr(node, "decorator_list", [])
    )


def _accessor_kind(node) -> str:
    """`setter` / `deleter` / `getter` for the accessor form, else "".

    `@property` itself is NOT one of them. It is the FIRST binding of the name, so exempting it
    hid two complete copies of a property in one class, which is the same merge damage as any
    other duplicated def. What legitimately rebinds is `@v.setter` / `@v.deleter` / `@v.getter`.
    """
    for decorator in getattr(node, "decorator_list", []):
        name = _decorator_name(decorator)
        if name.count(".") == 1 and name.split(".")[1] in PROPERTY_ATTRS:
            return name.split(".")[1]
    return ""


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


def _defined_names(node, overloads):
    """[(name, kind)] bound by a direct child of a module/class body."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if _is_overload_def(node, overloads):
            return []
        # ONE ACCESSOR OF EACH KIND IS LEGITIMATE;
        # A SECOND OF THE SAME KIND IS NOT.
        # Discarding every accessor meant a getter followed by two copies of the same `@value.setter` scanned clean,
        kind = _accessor_kind(node)
        if kind:
            return [(f"{node.name}.{kind}", f"{kind} accessor")]
        return [(node.name, "def")]
    if isinstance(node, ast.ClassDef):
        return [(node.name, "class")]
    # `X: int = 1` is an AnnAssign, not an Assign, and a typed module-level constant is common in this repo
    # (MAX_FUSED_SIZE: int = 65536).
    # EVERY target, not a lone one: a chained `N = K = 256` (which this repo writes, in
    # tests/test_grouped_gemm_optional_gather_indices.py) parks two names in one statement, and requiring exactly one
    # target dropped both, so duplicating the line rebound both while the scan reported clean.
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        targets, value = [node.target], node.value
    else:
        return []
    # `X = frozenset(X)` / `X = X + (...)` transforms a constant rather than redefining it.
    # Only a value computed WITHOUT the old one replaces it, decided per name so that `A, B = B, A` exempts both while
    # `A, B = 1, 2` exempts neither.
    referenced = {sub.id for sub in ast.walk(value) if isinstance(sub, ast.Name)}
    names = []
    for target in targets:
        for name in _constant_targets(target):
            # One statement binding a name twice (`X = X = 1`) still binds it once, so it is not the two-copies damage
            if name not in referenced and name not in names:
                names.append(name)
    return [(name, "constant") for name in names]


def _scope_duplicates(body, scope, out, overloads, module_overloads) -> None:
    """Names bound twice as direct children of one body; recurses into class bodies.

    `overloads` is what a decorator in THIS body resolves to; `module_overloads` is what a
    decorator in a fresh class body resolves to. They differ because a class body is not part
    of the scope chain of a class nested inside it.
    """
    seen = {}
    for node in body:
        for name, kind in _defined_names(node, overloads):
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
            # The class gets the MODULE's aliases plus any it binds itself
            # "The scope of names defined in a class block is limited to the class block", and a class body resolves an
            # unbound name in the GLOBAL namespace, so a nested class does not see `import typing as t` from the class
            _scope_duplicates(
                node.body,
                f"{scope}{node.name}.",
                out,
                _overload_names_in(node.body, module_overloads),
                module_overloads,
            )
        for nested in _branch_bodies(node):
            # A control-flow branch is not a scope, so it keeps the aliases of the body it sits in -- only a class body
            _scope_duplicates(nested, scope, out, overloads, module_overloads)


def _import_duplicates(body, scope, out) -> None:
    """A name imported twice within one statement, or twice from the same module.

    Keyed on the name each alias BINDS, not on the name it comes from: `from m import x as v`
    followed by `from m import y as v` binds `v` twice and the second silently wins, which is
    the dead-binding this is here to catch. Plain `import a.b` keeps the full dotted path in
    its key too, since `import urllib.parse` beside `import urllib.request` is correct.
    """
    seen_implicit: dict = {}
    seen_explicit: dict = {}
    for node in body:
        if isinstance(node, ast.ImportFrom):
            # `level` carries the leading dots, so `from .a` and `from ..a` stay distinct.
            module = f"{'.' * node.level}{node.module or ''}"
        elif isinstance(node, ast.Import):
            module = None  # the source is per-alias for a plain import, not per-statement
        else:
            # Class bodies are scanned too:
            if isinstance(node, ast.ClassDef):
                _import_duplicates(node.body, f"{scope}{node.name}.", out)
            # Each control-flow branch on its own:
            for nested in _branch_bodies(node):
                _import_duplicates(nested, scope, out)
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            # `import a.b` binds `a`, not `a.b`;
            # Using `alias.asname` raw here instead missed `import urllib.parse` followed by `import urllib.parse as
            # urllib`, which silently repoints `urllib` from the package to the submodule.
            source = module if module is not None else alias.name
            bound = alias.asname or (alias.name if module is not None else alias.name.split(".")[0])
            # Keyed on the BOUND NAME, with a carve-out for IMPLICIT bindings, whose name was never chosen:
            # for the case where the name was never chosen: an IMPLICIT binding takes whatever
            # Keyed on the BOUND NAME, because that is what gets shadowed, with one carve-out for the case where the
            # Keyed on the bound name alone, two DIFFERENT plain-import duplicates both read as `import:None:x`, so
            # compare mode charged a newly introduced one against a pre-existing counter entry and passed. DELIMITED,
            # because `scope` ends in a dot and `source` may contain them: a module-level `from A.m import x` and a
            implicit = alias.asname is None
            # Keyed on the BOUND NAME, because that is what gets shadowed, with one carve-out for the case where the
            # name was never chosen: an IMPLICIT binding takes whatever the source happens to be called, so two of them
            # from different sources are the ordinary `import urllib.parse` / `import urllib.request` and `from a import
            # x` / `from b import x` shapes and stay legitimate.
            # An EXPLICIT `as` alias is a name the author picked, so a second binding of it is always dead
            first = seen_explicit.get(bound)
            if implicit:
                first = seen_implicit.get((bound, source), first)
            elif first is None:
                first = min(
                    (line for (name, _), line in seen_implicit.items() if name == bound),
                    default = None,
                )
            if first is not None:
                where = (
                    "twice in this statement"
                    if first == node.lineno
                    else f"twice from the same module (first at line {first})"
                )
                out.append(
                    Finding(
                        node.lineno,
                        # The emitted identity must mirror the detection key:
                        f"import:{scope}|{source}:{bound}",
                        f"{scope}{bound} is imported {where}",
                    )
                )
            elif implicit:
                seen_implicit[(bound, source)] = node.lineno
            else:
                seen_explicit[bound] = node.lineno


def scan_source(source: str, filename: str = "<unknown>"):
    """Findings for one Python source string. A file that does not parse is itself a finding."""
    try:
        tree = ast.parse(source, filename = filename)
    except SyntaxError as exc:
        return [Finding(exc.lineno or 0, "parse", f"does not parse ({exc.msg})")]
    found = []
    module_overloads = _overload_names(tree)
    _scope_duplicates(tree.body, "", found, module_overloads, module_overloads)
    _import_duplicates(tree.body, "", found)
    return sorted(found)


def _git(args, cwd = None):
    return subprocess.run(["git", *args], cwd = cwd, capture_output = True, text = True)


def _decode_source(data: bytes) -> str:
    """Decode Python source by ITS OWN declared encoding, the way the parser would.

    `text = True` decodes with the runner's locale, so a valid `# coding: cp1252` file holding
    a non-UTF-8 byte raised UnicodeDecodeError out of `subprocess` and took the whole run down
    -- on an unrelated edit, and on a file `compileall` and ruff both accept. PEP 263 says the
    declaration decides, and `tokenize.detect_encoding` is the parser's own reader for it.
    """
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    except SyntaxError:
        encoding = "utf-8"
    return data.decode(encoding)


def _revision_findings(revision: str, path: str):
    """Findings for `path` at `revision`, or None if the file does not exist there."""
    shown = subprocess.run(["git", "show", f"{revision}:{path}"], capture_output = True)
    if shown.returncode != 0:
        return None
    try:
        source = _decode_source(shown.stdout)
    except (UnicodeDecodeError, LookupError) as exc:
        return [Finding(0, "parse", f"does not decode ({exc})")]
    return scan_source(source, f"{revision}:{path}")


def _rename_map(before: str, after: str):
    """{new path: old path} for the renames in this range.

    The changed-file sweep reports a renamed file under its NEW name, so looking the before
    side up under that name finds nothing and every finding in it reads as introduced. A
    branch that only moves a file carrying one of the duplicates already on main would then
    be blocked for a duplicate it did not write.
    """
    # -z, because git quotes and backslash-escapes any non-ASCII path by default (core.quotePath), and the escaped
    # spelling matches nothing.
    # Under -z each rename is three NUL-terminated records, "R100", old, new
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
    (
        1,
        "def toggle(page):\n    # click it\n    page.click('#r')\n\n\n"
        "def toggle(page):\n    # open the panel\n    page.click('#r')\n",
    ),
    (1, "from floor_table import latest_attempt_rows, refuse_collisions, latest_attempt_rows\n"),
    (1, "TOGGLE_JS = '() => 1'\nOTHER = 2\nTOGGLE_JS = '() => 2'\n"),
    (1, "class A:\n    pass\n\n\nclass A:\n    pass\n"),
    (1, "class A:\n    def go(self):\n        pass\n\n    def go(self):\n        pass\n"),
    (1, "from a import x\nfrom a import x\n"),
    # 6. A stray paren inside a comment must not truncate the scan (the regex version did).
    (1, "def go():  # takes (a, b\n    pass\n\n\ndef go():\n    pass\n"),
    (1, "<<<<<<< HEAD\ndef go():\n    pass\n"),
    (1, "MAX_FUSED_SIZE: int = 65536\nMAX_FUSED_SIZE: int = 131072\n"),
    (1, "REGISTRY: dict = {}\nOTHER = 1\nREGISTRY = {'a': 1}\n"),
    # Two aliases from one module landing on the same bound name:
    (1, "from m import x as value\nfrom m import y as value\n"),
    (1, "from m import x as value, y as value\n"),
    # A class body is a scope too, for imports as well as for defs.
    (1, "class A:\n    from m import x\n    from m import x\n"),
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
    # Constants declared together in one tuple target:
    (4, "B, H, N, D = 1, 16, 50345, 128\nB, H, N, D = 2, 3, 4, 5\n"),
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
    (0, "REGISTRY: dict\nREGISTRY = {}\n"),
    (0, "from m import x\nfrom m import x as other\n"),
    (0, "ROWS = (1,)\nROWS = ROWS + (2,)\n"),
    # A tuple self-transform is decided per name, so a swap exempts both sides.
    (0, "A = 1\nB = 2\nA, B = B, A\n"),
    # Distinct submodules of one package each bind the package name and are correct.
    (0, "import urllib.parse as parse\nimport urllib.request as request\n"),
    (0, "value = 1\nvalue = 2\n"),
    (0, "import os\nfrom a import b\n\nX = 1\n\n\ndef go():\n    return os, b, X\n"),
    # A chained assignment binds every name in the chain, so duplicating the line rebinds every one of them.
    # `N = K = 256` is written in this repo;
    (2, "N = K = 256\nN = K = 512\n"),
    (0, "N = K = 256\nM = 512\n"),
    (
        0,
        "X = X = 1\n",
    ),  # one statement, one live binding Overload signatures reached through an import alias are still
    (
        0,
        "import typing as t\n@t.overload\ndef f(x: int) -> int: ...\n"
        "@t.overload\ndef f(x: str) -> str: ...\ndef f(x):\n    return x\n",
    ),
    (
        0,
        "from typing import overload as ov\n@ov\ndef f(x: int) -> int: ...\n"
        "@ov\ndef f(x: str) -> str: ...\ndef f(x):\n    return x\n",
    ),
    # ...but the alias has to be typing's.
    (1, "import types as t\n@t.overload\ndef f(x): ...\n@t.overload\ndef f(x): ...\n"),
    # One explicit alias, two sources:
    (1, "import urllib.parse as client\nimport urllib.request as client\n"),
    (1, "from os import path as v\nfrom sys import modules as v\n"),
    # The shape that legitimately binds one name twice, and must stay legitimate:
    (0, "import urllib.parse\nimport urllib.request\n"),
    # A plain import binds the ROOT of its dotted path, so this repoints `urllib` from the package to the submodule and
    # Still caught: the same package root rebound from the package to a submodule.
    (1, "import urllib.parse\nimport urllib.parse as urllib\n"),
    # EVERY IMPLICIT SOURCE IS REMEMBERED, not just the first.
    # Keeping one entry per bound name compared the third statement with the FIRST source, which looked like the
    # legitimate different-source shape again, so an exact repeat went unreported.
    (1, "from a import x\nfrom b import x\nfrom b import x\n"),
    (1, "import urllib.parse\nimport urllib.request\nimport urllib.request\n"),
    (0, "from a import x\nfrom b import x\nfrom c import x\n"),
    # An explicit alias collides with a name already bound implicitly, in either order.
    (1, "import urllib.parse as urllib\nimport urllib.request\n"),
    (
        1,
        "import types as t\n\n\ndef helper():\n    import typing as t\n    return t\n\n\n"
        "@t.overload\ndef f(x): ...\n@t.overload\ndef f(x): ...\n",
    ),
    (
        0,
        "try:\n    from typing import overload as ov\nexcept ImportError:\n"
        "    from typing_extensions import overload as ov\n"
        "@ov\ndef f(x: int) -> int: ...\n@ov\ndef f(x: str) -> str: ...\ndef f(x):\n    return x\n",
    ),
    # A duplicated @property getter.
    (
        1,
        "class C:\n    @property\n    def v(self):\n        return 1\n"
        "    @v.setter\n    def v(self, x):\n        pass\n"
        "    @v.setter\n    def v(self, x):\n        pass\n",
    ),
    (
        0,
        "class C:\n    @property\n    def v(self):\n        return 1\n"
        "    @v.setter\n    def v(self, x):\n        pass\n"
        "    @v.deleter\n    def v(self):\n        pass\n",
    ),
    (
        0,
        "class C:\n    import typing as t\n"
        "    @t.overload\n    def f(self, x: int): ...\n"
        "    @t.overload\n    def f(self, x: str): ...\n    def f(self, x):\n        return x\n",
    ),
    # ...but that alias stops at ITS OWN class body.
    # A class nested inside it resolves an unbound name in the MODULE namespace, not in the class around it, so
    # `@t.overload` in the inner class is `types.overload` here and the two copies are ordinary merge damage.
    (
        1,
        "import types as t\n\n\nclass Outer:\n    import typing as t\n\n    class Inner:\n"
        "        @t.overload\n        def f(self, x: int): ...\n"
        "        @t.overload\n        def f(self, x: str): ...\n",
    ),
    (
        0,
        "import types as t\n\n\nclass Outer:\n    class Inner:\n        import typing as t\n"
        "        @t.overload\n        def f(self, x: int): ...\n"
        "        @t.overload\n        def f(self, x: str): ...\n",
    ),
    (
        0,
        "import typing as t\n\n\nclass Outer:\n    class Inner:\n"
        "        @t.overload\n        def f(self, x: int): ...\n"
        "        @t.overload\n        def f(self, x: str): ...\n",
    ),
    (
        1,
        "import os\nif os.name:\n    def go():\n        return 1\n    def go():\n        return 2\n",
    ),
    # A typing alias bound INSIDE a function is not in effect at module level, so it may not exempt a module-level
    # ...but the conventional wrapped spellings are still found, since they are module level.
    # ONE accessor of each kind is legitimate;
    # A class binds its own names, so a class-local typing alias resolves its own decorators.
    # A nested class binding the alias ITSELF still resolves its own decorators.
    # ...and a MODULE-level alias reaches every class body at every depth, so narrowing the inner class to its own
    # Each control-flow branch scanned on its own:
    (
        0,
        "import os\nif os.name:\n    def go():\n        return 1\nelse:\n"
        "    def go():\n        return 2\n",
    ),
    (0, "try:\n    import json\nexcept ImportError:\n    import json\n"),
    (1, "try:\n    import json\n    import json\nexcept ImportError:\n    pass\n"),
]

# Guarded, not inlined above: `match` is 3.10 syntax and this tool supports 3.9, where ast.parse raises SyntaxError and
# every case below would report a parse finding instead.
if sys.version_info >= (3, 10):
    _SELF_TEST_CASES += [
        (
            1,
            "import os\nmatch os.name:\n    case 'posix':\n        def go():\n"
            "            return 1\n        def go():\n            return 2\n"
            "    case _:\n        pass\n",
        ),
        (
            1,
            "match 1:\n    case 1:\n        import json\n        import json\n"
            "    case _:\n        pass\n",
        ),
        (
            0,
            "match 1:\n    case 1:\n        def go():\n            return 1\n"
            "    case 2:\n        def go():\n            return 2\n",
        ),
        # One definition per case is the conditional idiom, exactly as for if/else.
        # The alias walk reaches into a case too, so this stays an overload pair.
        (
            0,
            "match 1:\n    case 1:\n        import typing as t\n\n        @t.overload\n"
            "        def f(x: int): ...\n\n        @t.overload\n        def f(x: str): ...\n",
        ),
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
            # Read both revisions out of git rather than the working tree.
            # On a pull_request event the checkout is refs/pull/N/merge, which is neither end of the range being judged.
            after = _revision_findings(args.after, str(path))
            if after is None:
                continue
            old = renames.get(str(path), str(path))
            before = _revision_findings(args.before, old) if old.endswith(".py") else None
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
                source = _decode_source(path.read_bytes())
            except (OSError, UnicodeDecodeError) as exc:
                blocking.append(f"{path}: unreadable ({exc})")
                continue  # not present at the head revision (deleted, or renamed away) Only follow a rename back to a
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
