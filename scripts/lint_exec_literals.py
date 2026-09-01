#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail CI when `exec`, `eval` or `compile` is handed something that is not a literal.

Two of the holes fixed in unsloth-zoo#1108 and unsloth#9777 were exactly this shape:
`exec` of an HTTPS response body, and `exec` of a string built from a downloaded
config. GitHub's CodeQL runs on these repositories with Python enabled and raises
other injection findings here, `py/path-injection` among them, but it has never once
raised `py/code-injection` - so this class is not covered by what is already running.

The rule is deliberately the blunt one: the first argument to one of those three
builtins must be a written-out string. Anything else - a name, a call, an f-string
with a placeholder - is reported. That is coarse, and it is meant to be. The realistic
failure is a contributor writing `exec(f"...")` without thinking about where the
pieces came from, not somebody engineering a way past a checker they could equally
well delete.

Existing call sites are recorded in a baseline beside this script, so the gate starts
green and only new ones fail. A baseline entry carries the call's own text rather than
its line number, so moving code does not churn it, and it carries a count, so a new
call cannot hide behind a removed one.

    python scripts/lint_exec_literals.py             # check, exit 1 on a new site
    python scripts/lint_exec_literals.py --update    # rewrite the baseline
    python scripts/lint_exec_literals.py --self-test # prove the rule still fires
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

# Only the bare names. `re.compile(pattern)` and `model.eval()` are not these builtins, and matching on the attribute
SINKS = ("exec", "eval", "compile")

# Notebooks are scanned too:
SUFFIXES = (".py", ".ipynb")

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "exec_literals_baseline.json"

# Not part of any commit, or not ours to fail on.
EXCLUDED_PARTS = frozenset(
    {
        "tests",
        "node_modules",
        "build",
        "dist",
        ".venv",
        "venv",
        "site-packages",
        ".git",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        ".ipynb_checkpoints",
        ".eggs",
    }
)


def _is_written_out(node: ast.AST) -> bool:
    """Whether this argument is a string the file spells out in full."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    if isinstance(node, ast.JoinedStr):
        # An f-string with no placeholder is just a string.
        return not any(isinstance(part, ast.FormattedValue) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # `"import " + "torch"` is still written out;
        return _is_written_out(node.left) and _is_written_out(node.right)
    return False


def _key(relative: str, call: ast.Call) -> dict:
    """How a call is remembered: its file, its sink, and a digest of its own source.

    The digest is over `ast.unparse`, so reformatting and moving the call leave the
    entry alone while changing what is passed does not.
    """
    return {
        "file": relative,
        "sink": call.func.id,
        "digest": hashlib.sha256(ast.unparse(call).encode()).hexdigest()[:16],
    }


def _notebook_source(path: Path, relative: str) -> str:
    """The Python of a notebook's code cells, one blank line apart.

    A cell is not a module, so they are joined rather than parsed one at a time; a
    `%magic` or `!command` line is not Python at all and is blanked, which keeps the
    line count and therefore the reported line numbers honest.
    """
    try:
        document = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{relative}: could not be read ({error.__class__.__name__})")
    lines: list[str] = []
    for cell in document.get("cells") or []:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source") or []
        text = source if isinstance(source, str) else "".join(source)
        for line in text.splitlines():
            lines.append("" if line.lstrip().startswith(("%", "!")) else line)
        lines.append("")
    return "\n".join(lines)


def scan_file(path: Path, relative: str) -> list[dict]:
    if path.suffix == ".ipynb":
        source: str | bytes = _notebook_source(path, relative)
    else:
        source = path.read_bytes()
    try:
        tree = ast.parse(source, filename = str(path))
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        if path.suffix == ".ipynb":
            # A notebook that does not parse as one module is ordinary:
            return []
        # A .py file that will not parse has not been checked, and reporting it clean is the bypass this whole gate
        raise SystemExit(f"{relative}: could not be parsed ({error.__class__.__name__})")
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in SINKS:
            continue
        if _is_written_out(node.args[0]):
            continue
        entry = _key(relative, node)
        entry["line"] = node.lineno
        found.append(entry)
    return found


def collect(targets: list[str]) -> list[dict]:
    found = []
    for target in targets:
        root = REPO_ROOT / target
        if root.is_file() and root.suffix in SUFFIXES:
            paths = [root]
        elif root.is_dir():
            paths = sorted(q for s in SUFFIXES for q in root.rglob(f"*{s}"))
        else:
            # A target that resolves to nothing means the gate covers less than it claims to, which is a silent hole
            raise SystemExit(
                f"{target}: scan target does not exist, so nothing under it was checked"
            )
        for path in paths:
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                # `--paths` may name a file outside the checkout, which is how the tests drive the gate.
                relative = path.as_posix()
            if EXCLUDED_PARTS & set(Path(relative).parts):
                continue
            found.extend(scan_file(path, relative))
    return found


def _counted(entries: list[dict]) -> dict:
    """Identity -> how many times it appears. Identity AND count, so a new call cannot
    take the place of one that was removed."""
    counts: dict = {}
    for entry in entries:
        counts[(entry["file"], entry["sink"], entry["digest"])] = (
            counts.get((entry["file"], entry["sink"], entry["digest"]), 0) + 1
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--update", action = "store_true", help = "rewrite the baseline")
    parser.add_argument("--self-test", action = "store_true", help = "check the rule still fires")
    parser.add_argument("--paths", nargs = "*", help = "scan these instead of the defaults")
    arguments = parser.parse_args()

    if arguments.self_test:
        return self_test()

    document = json.loads(BASELINE_PATH.read_text(encoding = "utf-8"))
    targets = arguments.paths or document["targets"]
    found = collect(targets)

    if arguments.update:
        # Existing reasons are carried over.
        reasons = {
            (e["file"], e["sink"], e["digest"]): e.get("reason", "") for e in document["entries"]
        }
        document["entries"] = sorted(
            (
                {
                    "file": f,
                    "sink": s,
                    "digest": d,
                    "count": n,
                    "reason": reasons.get((f, s, d), "REVIEW ME"),
                }
                for (f, s, d), n in _counted(found).items()
            ),
            key = lambda e: (e["file"], e["sink"], e["digest"]),
        )
        BASELINE_PATH.write_text(json.dumps(document, indent = 2) + "\n", encoding = "utf-8")
        print(f"baseline: {len(document['entries'])} entries, {len(found)} call sites")
        return 0

    allowed = {(e["file"], e["sink"], e["digest"]): e["count"] for e in document["entries"]}
    observed = _counted(found)
    lines = {(e["file"], e["sink"], e["digest"]): e["line"] for e in found}

    new = [k for k, n in observed.items() if n > allowed.get(k, 0)]
    if new:
        print(f"{len(new)} dynamic-execution call site(s) not in the baseline:\n")
        for f, s, d in sorted(new):
            print(f"  {f}:{lines[(f, s, d)]}  {s}(...) is handed a value that is not written out")
        print(
            "\nEither pass a written-out string, or - if the value really is trusted - "
            "record it with `--update` and say why in the pull request."
        )
        return 1

    unreviewed = sorted(
        (e["file"], e["sink"])
        for e in document["entries"]
        if not e.get("reason") or e["reason"] == "REVIEW ME"
    )
    if unreviewed:
        print(f"{len(unreviewed)} baseline entr(y/ies) carry no justification:\n")
        for f, s in unreviewed:
            print(f"  {f}  {s}")
        print("\nSay why the value is trusted in the entry's `reason` field.")
        return 1

    stale = sorted(k for k in allowed if k not in observed)
    if stale:
        # A baseline that outlives its call site quietly re-permits whatever lands on that digest next, so it is an
        print(f"{len(stale)} baseline entr(y/ies) no longer match any call. Run --update:\n")
        for f, s, d in stale:
            print(f"  {f}  {s}  {d}")
        return 1

    print(f"ok: {len(found)} dynamic-execution call site(s), all recorded")
    return 0


_BAD = """
def f(name):
    exec(f"import {name}")
    eval("torch." + name)
    compile(source, "<x>", "exec")
    exec("import MODULE".replace("MODULE", name))
"""

_GOOD = """
import re
def f(name, source):
    exec("import torch")
    exec(f"no placeholders here")
    eval("1 + 1")
    re.compile(name)          # not the builtin
    model.eval()              # not the builtin
    exec()                    # no arguments
"""


def self_test() -> int:
    """The rule has to fire on the bad shapes and stay quiet on the good ones.

    Written out here rather than in the test suite as well, so the gate can prove
    itself on a runner that installs nothing but Python.
    """
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory() as directory:
        for label, source, expected in (("bad", _BAD, 4), ("good", _GOOD, 0)):
            path = Path(directory) / f"{label}.py"
            path.write_text(source, encoding = "utf-8")
            count = len(scan_file(path, f"{label}.py"))
            if count != expected:
                failures.append(f"{label}: expected {expected} finding(s), got {count}")
    if failures:
        print("self-test FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
