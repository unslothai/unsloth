#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse silent llama-server parallel-slot downgrades.

#7717 clamped `--parallel` to 1 whenever MTP resolved, so a batched API caller
lost 4x throughput with nothing but a log line to show for it. The slot count is
a user-facing setting: a load may launch fewer slots only when a real resource
or capability limit forces it, and never as a policy choice.

Flagged inside `studio/backend` (tests excluded):

1.  `n_parallel = 1` in a function body -- a downgrade, not a default.
2.  `n_parallel = <name>` where the name is a saved pre-clamp count, which is
    how a clamp hands slots back to itself across a retry.

Parameter defaults, class-body annotations, `self.<attr>` assignments and
`max(1, ...)` / `getattr(..., 1)` floors are structurally distinct and pass.

A genuine capability or resource limit is allowed with a trailing
`# allow-slot-clamp: <reason>` comment on the assignment line.

Exit codes: 0 = clean, 1 = findings, 2 = usage error.
Run from repo root: python3 scripts/lint_no_parallel_clamp.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIR = REPO_ROOT / "studio" / "backend"

SLOT_NAMES = frozenset({"n_parallel"})
ALLOW_MARKER = "# allow-slot-clamp:"
SKIP_PARTS = frozenset({"tests", ".venv", "venv", "build", "dist", "node_modules", "__pycache__"})


def _is_slot_target(node: ast.AST) -> bool:
    """A bare `n_parallel`, not `self._n_parallel` or `d["n_parallel"]`."""
    return isinstance(node, ast.Name) and node.id in SLOT_NAMES


def _restores_a_saved_count(value: ast.AST) -> bool:
    """`n_parallel = _mtp_clamped_slots`: a name holding a pre-clamp count. Only a
    clamp saves one, so the restore is proof the clamp is back."""
    return isinstance(value, ast.Name) and value.id.endswith("_clamped_slots")


def _findings_for_tree(tree: ast.AST, lines: list[str], filename: str) -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign) or not any(
                _is_slot_target(t) for t in node.targets
            ):
                continue
            if isinstance(node.value, ast.Constant) and node.value.value == 1:
                what = "clamped to a single slot"
            elif _restores_a_saved_count(node.value):
                what = f"restores the pre-clamp count `{node.value.id}`"
            else:
                continue
            if ALLOW_MARKER in lines[node.lineno - 1]:
                continue
            found.append((filename, node.lineno, what))
    return found


def scan_source(source: str, filename: str) -> list[tuple[str, int, str]]:
    """(file, line, reason) per finding. Unparseable files are left to compileall."""
    try:
        tree = ast.parse(source, filename = filename)
    except SyntaxError:
        return []
    return _findings_for_tree(tree, source.splitlines(), filename)


def scan_paths(root: Path) -> tuple[list[tuple[str, int, str]], int]:
    found: list[tuple[str, int, str]] = []
    scanned = 0
    for path in sorted(root.rglob("*.py")):
        if SKIP_PARTS & set(path.parts):
            continue
        scanned += 1
        try:
            source = path.read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            continue
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        found += scan_source(source, str(rel))
    return found, scanned


_SELF_TEST_CASES: tuple[tuple[str, int], ...] = (
    # (source, expected finding count)
    ("def load():\n    n_parallel = 1\n", 1),
    ("def load():\n    n_parallel = 1  # allow-slot-clamp: no --kv-unified\n", 0),
    ("def load():\n    n_parallel = _mtp_clamped_slots\n", 1),
    ("def load(n_parallel: int = 1):\n    return n_parallel\n", 0),
    ("class A:\n    n_parallel: int = 1\n", 0),
    ("def load():\n    self._requested_n_parallel = 1\n", 0),
    ("def load(x):\n    n_parallel = max(1, x)\n", 0),
    ("def load(s):\n    n_parallel = getattr(s, 'llama_parallel_slots', 1)\n", 0),
    ("def load(r):\n    n_parallel = r.n_parallel\n", 0),
    ("n_parallel = 1\n", 0),  # module scope is configuration, not a launch decision
)


def _self_test() -> int:
    failures = 0
    for source, expected in _SELF_TEST_CASES:
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
        print(f"ERROR: no Python files under {args.path}", file = sys.stderr)
        return 2
    if found:
        for filename, lineno, what in found:
            print(
                f"::error file={filename},line={lineno}::parallel slots {what}. The slot count is "
                f"a user setting: launch fewer only for a real capability or VRAM limit, and mark "
                f"it '{ALLOW_MARKER} <reason>'.",
            )
        print(f"{len(found)} silent parallel-slot downgrade(s)", file = sys.stderr)
        return 1
    print(f"no silent parallel-slot downgrades (scanned {scanned} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
