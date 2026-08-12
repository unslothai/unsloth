#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse a TypeScript file that binds the same import name twice.

#8470 landed a second copy of

    import { HubModelPicker, hasDownloadedModels } from "./model-selector/pickers";

into `model-selector.tsx`, seven lines below the first. `tsc -b` rejects it with
TS2300 `Duplicate identifier`, so `npm run build` fails, so the frontend build
step fails -- and every job that builds the frontend goes red with it. On the
commit that landed it that was eleven jobs across Studio, Chat UI, the GGUF
smoke and the wheel, on main and on every open PR branch.

Nothing caught it earlier. eslint is not run in CI at all, and its config does
not enable `no-duplicate-imports` in any case. The first thing to notice was a
`tsc` error inside the slowest job in the matrix.

This is the cheap check that belongs in Source lint instead: scan the import
prologue of every committed `.ts`/`.tsx` file and report any local binding name
introduced twice. It needs no npm, no node and no type information.

What counts as a binding, per the ES module grammar:

    import Default from "m"                 -> Default
    import * as ns from "m"                 -> ns
    import { a, b as c } from "m"           -> a, c
    import Default, { a } from "m"          -> Default, a
    import type { T } from "m"              -> T
    import "m"                              -> nothing

`import type` and a value import of the same name collide exactly as two value
imports do, so they are not distinguished. Two imports of the same *module* are
legal as long as the names differ, so the module path is not what is compared.

Exit codes: 0 = clean, 1 = findings, 2 = usage error.
Run from repo root: python3 scripts/lint_no_duplicate_ts_imports.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIR = REPO_ROOT / "studio" / "frontend" / "src"
SKIP_PARTS = frozenset({"node_modules", "dist", "build", ".venv", "venv", "__pycache__"})

# One import statement, up to its module specifier. `re.DOTALL` so a clause
# broken across lines -- which is how prettier formats anything long, and how
# the #8470 duplicate was formatted -- is still one match.
_IMPORT = re.compile(
    r"^[ \t]*import[ \t]+(?P<clause>[^;'\"]*?)[ \t]*from[ \t]*['\"][^'\"]+['\"]",
    re.MULTILINE | re.DOTALL,
)
# A bare `import "./styles.css"` binds nothing and is skipped by the `from`
# requirement above.


def _bindings(clause: str) -> list[str]:
    """Local names a single import clause introduces, in source order."""
    clause = clause.strip()
    if clause.startswith("type "):
        clause = clause[len("type ") :]

    names: list[str] = []
    braced = re.search(r"\{(?P<inner>.*)\}", clause, re.DOTALL)
    if braced:
        for piece in braced.group("inner").split(","):
            piece = piece.strip()
            if not piece:
                continue
            # `a as b` binds b; `type T` binds T; `type T as U` binds U.
            if piece.startswith("type "):
                piece = piece[len("type ") :].strip()
            parts = piece.split()
            names.append(parts[-1] if " as " in f" {piece} " else parts[0])
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


def duplicates_in(source: str) -> list[tuple[int, str]]:
    """(line number, name) for every binding this file introduces twice."""
    seen: dict[str, int] = {}
    found: list[tuple[int, str]] = []
    for match in _IMPORT.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        for name in _bindings(match.group("clause")):
            if name in seen:
                found.append((line, name))
            else:
                seen[name] = line
    return found


def scan_paths(root: Path) -> tuple[list[tuple[str, int, str]], int]:
    found: list[tuple[str, int, str]] = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".ts", ".tsx"):
            continue
        if SKIP_PARTS & set(path.parts) or path.name.startswith("._"):
            continue
        scanned += 1
        for line, name in duplicates_in(path.read_text(encoding = "utf-8", errors = "replace")):
            found.append((str(path.relative_to(REPO_ROOT)), line, name))
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
            "a side-effect import binds nothing",
            'import "./a.css";\nimport "./a.css";\n',
            [],
        ),
        (
            "a multi-line clause is one statement",
            'import {\n  alpha,\n  beta,\n} from "m";\nimport { beta } from "n";\n',
            ["beta"],
        ),
        (
            "the word import inside a string is not an import",
            'const s = "import { a } from \'m\'";\nimport { a } from "m";\n',
            [],
        ),
    ]
    failures = 0
    for label, source, expected in cases:
        got = [name for _, name in duplicates_in(source)]
        if got != expected:
            print(f"SELF-TEST FAIL: {label}: expected {expected}, got {got}", file = sys.stderr)
            failures += 1
    if failures:
        return 1
    print(f"self-test ok ({len(cases)} cases)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--self-test", action = "store_true", help = "check the rule, scan nothing")
    parser.add_argument("--path", type = Path, default = DEFAULT_SCAN_DIR, help = "directory to scan")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not args.path.is_dir():
        print(f"ERROR: {args.path} is not a directory", file = sys.stderr)
        return 2

    found, scanned = scan_paths(args.path)
    if not scanned:
        print(f"ERROR: no TypeScript files under {args.path}", file = sys.stderr)
        return 2
    if found:
        for filename, line, name in found:
            print(
                f"::error file={filename},line={line}::'{name}' is imported twice in this file. "
                f"tsc rejects it with TS2300 and the frontend build fails, taking every job "
                f"that builds it down.",
            )
        print(f"{len(found)} duplicate import binding(s)", file = sys.stderr)
        return 1
    print(f"no duplicate import bindings (scanned {scanned} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
