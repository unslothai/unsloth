#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse a TypeScript file that binds the same import name twice.

#8470 landed a duplicate import line in `model-selector.tsx`. `tsc -b` rejects
it with TS2300, so the frontend build fails and every job that builds it goes
red behind it (eleven, on main and on every open PR branch). eslint is not run
in CI at all, so nothing saw it until tsc did, inside the slowest job in the
matrix. This is the cheap Source lint check instead: scan the import prologue of
every committed `.ts`/`.tsx` file, no npm, no node, no type information.

What counts as a binding, per the ES module grammar:

    import Default from "m"                 -> Default
    import * as ns from "m"                 -> ns
    import { a, b as c } from "m"           -> a, c
    import Default, { a } from "m"          -> Default, a
    import type { T } from "m"              -> T
    import "m"                              -> nothing

`import type` collides with a value import of the same name, so the two are not
distinguished. Two imports of one module are legal when the names differ, so the
module path is not compared.

Exit codes: 0 = clean, 1 = findings, 2 = usage error.
Run from repo root: python3 scripts/lint_no_duplicate_ts_imports.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The frontend root, not `src`: `tsc -b` also builds tsconfig.node.json
# (`vite.config.ts`) and tsconfig.test.json (`tests`), so scanning only `src`
# leaves those two able to red the build with the very error this catches.
DEFAULT_SCAN_DIR = REPO_ROOT / "studio" / "frontend"
SKIP_PARTS = frozenset({"node_modules", "dist", "build", ".venv", "venv", "__pycache__"})

# One import declaration, anchored at the cursor. `re.DOTALL` so a clause prettier wrapped
# across lines, as the #8470 duplicate was, is still one match.
_IMPORT_AT = re.compile(
    r"\Aimport\b[ \t]*(?P<clause>[^;'\"]*?)[ \t]*from[ \t]*(['\"])[^'\"]*\2[ \t]*;?",
    re.DOTALL,
)
# A bare `import "./styles.css"` binds nothing, but it is part of the prologue and has to be
# stepped over rather than ending it.
_BARE_IMPORT_AT = re.compile(r"\Aimport\b[ \t]*(['\"])[^'\"]*\1[ \t]*;?")
# Whitespace, comments, and the directive prologue that may precede the imports. Five files
# here open with `"use client";`, and without this their imports would not be read at all.
_TRIVIA_AT = re.compile(
    r"\A(?:[ \t\r\n]+|//[^\n]*|/\*.*?\*/|(['\"])use [a-z ]+\1[ \t]*;?)+",
    re.DOTALL,
)

# The `type` modifier, clause-level or per specifier. `\s+`, not a literal space:
# any whitespace is legal after it, and a literal-space test on `type\n  Foo`
# leaves the modifier in place and records `type` itself as the binding -- so two
# such specifiers read as a duplicate and fail CI on a file tsc accepts.
_TYPE_MODIFIER = re.compile(r"^type\s+")


def _bindings(clause: str) -> list[str]:
    """Local names a single import clause introduces, in source order."""
    # Prettier keeps comments inside a long import list. Left in, `piece.split()`
    # reads `//` as the binding, losing the real name and reporting the next
    # commented import as a duplicate of it -- a CI fail on ordinary TypeScript.
    clause = re.sub(r"/\*.*?\*/", " ", clause, flags=re.DOTALL)
    clause = re.sub(r"//[^\n]*", " ", clause)
    clause = _TYPE_MODIFIER.sub("", clause.strip(), count=1)

    names: list[str] = []
    braced = re.search(r"\{(?P<inner>.*)\}", clause, re.DOTALL)
    if braced:
        for piece in braced.group("inner").split(","):
            piece = piece.strip()
            if not piece:
                continue
            # `a as b` binds b; `type T` binds T; `type T as U` binds U.
            piece = _TYPE_MODIFIER.sub("", piece, count=1).strip()
            parts = piece.split()
            # `as` as a token, not `" as "`: DOTALL lets a specifier wrap around
            # the keyword, and a literal-space test then records the imported
            # name instead of the alias.
            names.append(parts[-1] if "as" in parts[:-1] else parts[0])
        clause = clause[: braced.start()]

    # What is left is the default and/or namespace part, comma separated.
    for piece in clause.split(","):
        piece = piece.strip().rstrip(",").strip()
        if not piece:
            continue
        if piece.startswith("*"):
            parts = piece.split()
            names.append(parts[-1])  # `* as ns`
        elif piece.isidentifier():
            names.append(piece)
    return names


def _prologue_clauses(source: str) -> list[tuple[int, str]]:
    """(offset, clause) for each import declaration in the module's import prologue.

    Only the prologue, which is the run of imports, comments and directives at the top of the
    file, ending at the first statement that is not one. That bound is what makes this
    readable without a TypeScript lexer, and it is the whole design: the prologue cannot
    contain a template literal, a regex, or JSX, so none of the things that merely LOOK like
    imports can appear in it.

    Scanning the whole file instead was tried and produced a false positive every time it was
    patched: TypeScript quoted inside a test fixture, a backtick inside a regex pairing with a
    later template, a code sample rendered as JSX text, a module-level JSX initializer with no
    wrapping parentheses, a nested template, JSX text after an expression container. Every one
    of those is valid TypeScript that the gate would have failed, and this gate runs
    unconditionally on Source lint, so a false positive stops work on code that is correct.

    The cost is stated rather than hidden: an import written after other top-level code is not
    read, so a duplicate involving one is missed. Across this frontend that is 611 of 9801
    import declarations, in 19 files. That is a false NEGATIVE, which leaves the build exactly
    where it was before this check existed, and tsc still catches it. Given the choice between
    missing those and failing CI on valid code, this misses those.
    """
    found: list[tuple[int, str]] = []
    position = 0
    while position < len(source):
        rest = source[position:]
        trivia = _TRIVIA_AT.match(rest)
        if trivia:
            position += trivia.end()
            continue
        declaration = _IMPORT_AT.match(rest)
        if declaration:
            found.append((position, declaration.group("clause")))
            position += declaration.end()
            continue
        bare = _BARE_IMPORT_AT.match(rest)
        if bare:
            position += bare.end()
            continue
        break
    return found


def duplicates_in(source: str) -> list[tuple[int, str]]:
    """(line number, name) for every binding this file introduces twice."""
    seen: dict[str, int] = {}
    found: list[tuple[int, str]] = []
    for offset, clause in _prologue_clauses(source):
        line = source.count("\n", 0, offset) + 1
        for name in _bindings(clause):
            if name in seen:
                found.append((line, name))
            else:
                seen[name] = line
    return found


def scan_paths(root: Path) -> tuple[list[tuple[str, int, str]], int]:
    found: list[tuple[str, int, str]] = []
    scanned = 0
    # `--path` may be relative or outside the repo, and `REPO_ROOT` is absolute,
    # so resolve before relativizing and fall back to the full path.
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".ts", ".tsx"):
            continue
        if SKIP_PARTS & set(path.parts) or path.name.startswith("._"):
            continue
        scanned += 1
        try:
            shown = str(path.relative_to(REPO_ROOT))
        except ValueError:
            shown = str(path)
        for line, name in duplicates_in(path.read_text(encoding="utf-8", errors="replace")):
            found.append((shown, line, name))
    return found, scanned


def _self_test() -> int:
    """The rule has to fail on the real thing and pass on the legal ones."""
    cases: list[tuple[str, str, list[str]]] = [
        (
            "the #8470 regression, verbatim",
            'import { ModelConfigPage } from "./model-config-page";\n'
            'import { HubModelPicker, hasDownloadedModels } from "./model-selector/pickers";\n'
            'import {\n  type ExternalConnectionRef,\n} from "./model-selector/missing";\n'
            'import { HubModelPicker, hasDownloadedModels } from "./model-selector/pickers";\n',
            ["HubModelPicker", "hasDownloadedModels"],
        ),
        (
            "two imports of one module under different names are legal",
            'import { a } from "m";\nimport { b } from "m";\n',
            [],
        ),
        (
            "an alias makes the second binding distinct",
            'import { a } from "m";\nimport { a as b } from "n";\n',
            [],
        ),
        (
            "an alias colliding with an earlier plain name is not",
            'import { a } from "m";\nimport { z as a } from "n";\n',
            ["a"],
        ),
        (
            "a type-only import collides with a value import of that name",
            'import { Foo } from "m";\nimport type { Foo } from "n";\n',
            ["Foo"],
        ),
        (
            "default, namespace and named forms all bind",
            'import D from "m";\nimport * as D from "n";\n',
            ["D"],
        ),
        (
            "a default plus named clause on one line",
            'import D, { a } from "m";\nimport { D } from "n";\n',
            ["D"],
        ),
        (
            "a side-effect import binds nothing and does not end the prologue",
            'import { a } from "m";\nimport "./styles.css";\nimport { a } from "n";\n',
            ["a"],
        ),
        (
            "comments between and inside clauses do not end the prologue",
            'import { a } from "m";\n// why\n/* and why */\nimport {\n  // keep\n  a,\n} from "n";\n',
            ["a"],
        ),
        (
            "a directive prologue comes before the imports",
            '"use client";\n\nimport { a } from "m";\nimport { a } from "n";\n',
            ["a"],
        ),
        (
            "a file that omits semicolons still reports its duplicates",
            'import { A } from "m"\nimport { A } from "n"\n',
            ["A"],
        ),
        (
            "a wrapped specifier list aliased around the line break",
            'import {\n  alpha as\n    beta,\n} from "m";\nimport { beta } from "n";\n',
            ["beta"],
        ),
        # What the prologue bound buys. Each of these is valid TypeScript that a whole-file
        # scan reported as a duplicate, and each was found only after the previous one was
        # patched. They are kept as cases because the bound is what makes them impossible,
        # and a later change that widens the scan has to fail here rather than in CI.
        (
            "TypeScript quoted as a fixture in a template literal",
            'import { createServer } from "vite";\n'
            "const FIXTURE = String.raw`\n"
            'import { createServer } from "vite";\n'
            'import { renderToStaticMarkup } from "react-dom/server";\n'
            "`;\n",
            [],
        ),
        (
            "a nested template literal holding the same shape",
            'import { a } from "m";\n'
            "const OUTER = `${`\n"
            'import { a } from "m";\n'
            'import { a } from "m";\n'
            "`}`;\n",
            [],
        ),
        (
            "a backtick inside a regex",
            'import { a } from "m";\n'
            "const BACKTICK = /`/g;\n"
            "const FIXTURE = `\n"
            'import { a } from "m";\n'
            "`;\n",
            [],
        ),
        (
            "a code sample rendered as JSX text",
            'import { Fragment } from "react";\n'
            "export function Sample() {\n"
            "  return (\n"
            "    <pre>\n"
            'import Widget from "a";\n'
            'import Widget from "b";\n'
            "    </pre>\n"
            "  );\n"
            "}\n",
            [],
        ),
        (
            "a module-level JSX initializer without wrapping parentheses",
            'import { Fragment } from "react";\n'
            "const sample = <pre>\n"
            'import Widget from "a";\n'
            'import Widget from "b";\n'
            "</pre>;\n",
            [],
        ),
        (
            "JSX text following an expression container",
            'import { Fragment } from "react";\n'
            "const sample = <pre>\n"
            "{/* heading */}\n"
            'import Widget from "a";\n'
            'import Widget from "b";\n'
            "</pre>;\n",
            [],
        ),
        # The stated cost, pinned so it is a decision rather than a surprise. An import after
        # other top-level code is outside the prologue and is not read.
        (
            "an import after other top-level code is deliberately not read",
            'import { A } from "m";\nregisterResolver();\nimport { A } from "n";\n',
            [],
        ),
    ]
    failures = 0
    for label, source, expected in cases:
        got = [name for _, name in duplicates_in(source)]
        if got != expected:
            print(f"SELF-TEST FAIL: {label}: expected {expected}, got {got}", file=sys.stderr)
            failures += 1
    if failures:
        return 1
    print(f"self-test ok ({len(cases)} cases)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="check the rule, scan nothing")
    parser.add_argument("--path", type=Path, default=DEFAULT_SCAN_DIR, help="directory to scan")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not args.path.is_dir():
        print(f"ERROR: {args.path} is not a directory", file=sys.stderr)
        return 2

    found, scanned = scan_paths(args.path)
    if not scanned:
        print(f"ERROR: no TypeScript files under {args.path}", file=sys.stderr)
        return 2
    if found:
        for filename, line, name in found:
            print(
                f"::error file={filename},line={line}::'{name}' is imported twice in this file. "
                f"tsc rejects it with TS2300 and the frontend build fails, taking every job "
                f"that builds it down.",
            )
        print(f"{len(found)} duplicate import binding(s)", file=sys.stderr)
        return 1
    print(f"no duplicate import bindings (scanned {scanned} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
