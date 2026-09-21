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

# One import statement, up to its module specifier. `re.DOTALL` so a clause
# prettier wrapped across lines, as the #8470 duplicate was, is still one match.
_IMPORT = re.compile(
    r"^[ \t]*import[ \t]+(?P<clause>[^;'\"]*?)[ \t]*from[ \t]*['\"][^'\"]+['\"]",
    re.MULTILINE | re.DOTALL,
)
# A bare `import "./styles.css"` binds nothing; the `from` requirement skips it.

# The `type` modifier, clause-level or per specifier. `\s+`, not a literal space:
# any whitespace is legal after it, and a literal-space test on `type\n  Foo`
# leaves the modifier in place and records `type` itself as the binding -- so two
# such specifiers read as a duplicate and fail CI on a file tsc accepts.
_TYPE_MODIFIER = re.compile(r"^type\s+")

# Words that end in identifier characters but are not values, so a `/` after one of them
# opens a regex rather than dividing.
_OPERATOR_KEYWORDS = frozenset(
    {
        "await",
        "case",
        "delete",
        "do",
        "else",
        "in",
        "instanceof",
        "new",
        "of",
        "return",
        "throw",
        "typeof",
        "void",
        "yield",
    }
)


def _bindings(clause: str) -> list[str]:
    """Local names a single import clause introduces, in source order."""
    # Prettier keeps comments inside a long import list. Left in, `piece.split()`
    # reads `//` as the binding, losing the real name and reporting the next
    # commented import as a duplicate of it -- a CI fail on ordinary TypeScript.
    clause = re.sub(r"/\*.*?\*/", " ", clause, flags = re.DOTALL)
    clause = re.sub(r"//[^\n]*", " ", clause)
    clause = _TYPE_MODIFIER.sub("", clause.strip(), count = 1)

    names: list[str] = []
    braced = re.search(r"\{(?P<inner>.*)\}", clause, re.DOTALL)
    if braced:
        for piece in braced.group("inner").split(","):
            piece = piece.strip()
            if not piece:
                continue
            # `a as b` binds b; `type T` binds T; `type T as U` binds U.
            piece = _TYPE_MODIFIER.sub("", piece, count = 1).strip()
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


def _without_embedded_source(source: str) -> str:
    """`source` with comment and string bodies blanked, character count and lines preserved.

    A test fixture holds TypeScript as data, and a template literal is how it holds it:

        const RENDER_SOURCE = String.raw`
        import { createServer } from "vite";
        ...

    Those lines begin with optional indentation and `import`, so a scan of the raw text reads
    them as real imports of the enclosing file. Two fixtures quoting the same module then look
    like a duplicate, and the gate fails CI on a file tsc accepts. Both frontend test files
    that do this were reported before this masking existed.

    A block comment is the same problem: commented-out imports are not imports. A `//` line
    is already safe, since the pattern anchors `import` to the start of the line, but it is
    handled here too rather than left to that coincidence.

    Blanked, not deleted, so offsets and line numbers still point at the real source. Module
    specifiers survive as whitespace inside their quotes, which the pattern still matches, so
    masking cannot hide a genuine import.

    Regex literals are recognised, because leaving them out did NOT fail open as an earlier
    version of this comment claimed. `const BACKTICK = /`/g;` is real code in
    studio/frontend/src/lib/release-notes-preview.ts, and an unrecognised backtick inside it
    pairs with the opener of the next real template, so the masking lands on the gap between
    them and leaves the fixture exposed. That invents a duplicate rather than missing one.

    Telling a regex from a division needs the previous token, since `/` is both. The test is
    the usual one: after a value, so an identifier, a literal, or a closing bracket, `/`
    divides; anywhere else it opens a regex. `return`, `typeof` and the other operator
    keywords end in identifier characters but are not values, so they are listed.
    """
    out = list(source)
    length = len(source)

    def blank(start: int, stop: int) -> None:
        for index in range(max(start, 0), min(stop, length)):
            if out[index] != "\n":
                out[index] = " "

    def opens_a_regex(before: int) -> bool:
        """True when a `/` at `before` starts a regex rather than dividing."""
        cursor = before - 1
        while cursor >= 0 and source[cursor] in " \t\r\n":
            cursor -= 1
        if cursor < 0:
            return True
        previous = source[cursor]
        if previous in ")]":
            # `(a + b) / 2` divides. `if (x) /re/.test(s)` does not, and is not written here.
            return False
        if previous.isalnum() or previous in "_$":
            word = re.search(r"[A-Za-z_$][\w$]*$", source[: cursor + 1])
            return bool(word) and word.group(0) in _OPERATOR_KEYWORDS
        return True

    index = 0
    while index < length:
        char = source[index]
        pair = source[index : index + 2]
        if char == "/" and pair not in ("//", "/*") and opens_a_regex(index):
            cursor = index + 1
            in_class = False
            while cursor < length:
                here = source[cursor]
                if here == "\\":
                    cursor += 2
                    continue
                if here == "\n":
                    # Unterminated: a regex cannot span a line, so this was a division after
                    # all. Leave the slash alone rather than masking to the end of the file.
                    cursor = index
                    break
                if here == "[":
                    in_class = True
                elif here == "]":
                    in_class = False
                elif here == "/" and not in_class:
                    break
                cursor += 1
            if cursor > index:
                blank(index + 1, min(cursor, length))
                index = min(cursor, length) + 1
                continue
            index += 1
        elif pair == "//":
            stop = source.find("\n", index)
            stop = length if stop == -1 else stop
            blank(index, stop)
            index = stop
        elif pair == "/*":
            stop = source.find("*/", index + 2)
            stop = length if stop == -1 else stop + 2
            blank(index, stop)
            index = stop
        elif char in "'\"`":
            cursor = index + 1
            while cursor < length and source[cursor] != char:
                # A single- or double-quoted string cannot span a line; treating one that
                # reaches a newline as unterminated stops a stray apostrophe in prose from
                # swallowing the rest of the file.
                if char != "`" and source[cursor] == "\n":
                    break
                cursor += 2 if source[cursor] == "\\" else 1
            blank(index + 1, cursor)
            index = cursor + 1
        else:
            index += 1
    return "".join(out)


def _top_level(source: str) -> list[bool]:
    """Per character, whether it sits outside every bracket.

    An `import` declaration is only legal at the top level of a module, so anything that
    looks like one inside a bracket is something else wearing the shape. The case that
    matters is a component rendering a code sample as element text:

        export function Sample() {
          return (
            <pre>
        import Widget from "a";
            </pre>
          );
        }

    TypeScript reads those lines as JSX text and accepts the file. They are unindented,
    because indentation would show up in what the page renders, so anchoring to the start of
    the line does not separate them; what does is that they are inside the function body and
    the parenthesised return. Recognising JSX properly would need a real lexer, and this does
    not pretend to be one: it just declines to read a declaration anywhere the language would
    not allow one.

    Read after masking, so brackets inside strings, comments and regexes are already gone.
    """
    depths, depth = [], 0
    for char in source:
        if char in ")]}":
            depth -= 1
        depths.append(depth <= 0)
        if char in "([{":
            depth += 1
    return depths


def duplicates_in(source: str) -> list[tuple[int, str]]:
    """(line number, name) for every binding this file introduces twice."""
    seen: dict[str, int] = {}
    found: list[tuple[int, str]] = []
    source = _without_embedded_source(source)
    outside = _top_level(source)
    for match in _IMPORT.finditer(source):
        if not outside[match.start()]:
            continue
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
        for line, name in duplicates_in(path.read_text(encoding = "utf-8", errors = "replace")):
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
            "a code sample rendered as JSX text is not a declaration",
            'import { Fragment } from "react";\n'
            "\n"
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
            "a real duplicate after such a component is still caught",
            'import { A } from "m";\n'
            'function C() { return (<pre>\nimport X from "z";\n</pre>); }\n'
            'import { A } from "n";\n',
            ["A"],
        ),
        (
            "a backtick inside a regex does not open a template",
            # `const BACKTICK = /`/g;` is real code in release-notes-preview.ts. Unrecognised,
            # its backtick pairs with the opener of the next real template and exposes the
            # fixture between them.
            "const BACKTICK = /`/g;\n"
            "const FIXTURE = `\n"
            'import { A } from "m";\n'
            'import { A } from "n";\n'
            "`;\n",
            [],
        ),
        (
            "a quote inside a regex does not open a string",
            "const QUOTE = /'/g;\n"
            "const FIXTURE = `\n"
            'import { A } from "m";\n'
            'import { A } from "n";\n'
            "`;\n",
            [],
        ),
        (
            "a division is not a regex",
            'const half = total / 2;\nimport { A } from "m";\nimport { A } from "n";\n',
            ["A"],
        ),
        (
            "a division after a closing paren is not a regex",
            'const x = (a + b) / 2;\nimport { A } from "m";\nimport { A } from "n";\n',
            ["A"],
        ),
        (
            "a regex after an operator keyword is still a regex",
            'function f() { return /`/.test(s); }\nimport { A } from "m";\nimport { A } from "n";\n',
            ["A"],
        ),
        (
            "a slash inside a character class does not end the regex",
            'const re = /[/`]/g;\nimport { A } from "m";\nimport { A } from "n";\n',
            ["A"],
        ),
        (
            "TypeScript quoted as a fixture in a template literal is data, not imports",
            'import { createServer } from "vite";\n'
            "const FIXTURE = String.raw`\n"
            'import { createServer } from "vite";\n'
            'import { renderToStaticMarkup } from "react-dom/server";\n'
            "`;\n"
            "const OTHER = `\n"
            'import { createServer } from "vite";\n'
            "`;\n",
            [],
        ),
        (
            "a real duplicate after a fixture is still caught",
            'import { createServer } from "vite";\n'
            'const FIXTURE = `\nimport { unrelated } from "m";\n`;\n'
            'import { createServer } from "vite";\n',
            ["createServer"],
        ),
        (
            "imports commented out in a block are not imports",
            'import { a } from "m";\n/*\nimport { a } from "m";\nimport { a } from "n";\n*/\n',
            [],
        ),
        (
            "a backtick inside a string does not open a template",
            'const tick = "`";\nimport { a } from "m";\nimport { a } from "n";\n',
            ["a"],
        ),
        (
            "an escaped backtick does not close a template",
            'const FIXTURE = `\\`\nimport { a } from "m";\n`;\n'
            'import { b } from "m";\nimport { b } from "n";\n',
            ["b"],
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
            "an alias split across lines still binds the alias",
            'import {\n  Foo\n  as\n  Bar,\n} from "m";\nimport { Foo } from "n";\n',
            [],
        ),
        (
            "a comment inside the import list is not a binding",
            'import {\n  // the primary widget\n  alpha,\n} from "m";\n'
            'import {\n  /* helpers */\n  beta,\n} from "n";\n',
            [],
        ),
        (
            "a commented import list still reports its real duplicate",
            'import {\n  // the primary widget\n  alpha,\n} from "m";\n'
            'import {\n  // again\n  alpha,\n} from "n";\n',
            ["alpha"],
        ),
        (
            "an inline type modifier may be followed by any whitespace",
            'import {\n  type\n  Foo,\n  type\tBar,\n} from "m";\n',
            [],
        ),
        (
            "a clause-level type modifier may be followed by any whitespace",
            'import type\n{ Foo } from "m";\nimport type\n{ Bar } from "n";\n',
            [],
        ),
        (
            "a wrapped type modifier still reports its real duplicate",
            'import {\n  type\n  Foo,\n} from "m";\nimport { Foo } from "n";\n',
            ["Foo"],
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
