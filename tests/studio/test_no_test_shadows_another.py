# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A test name defined twice at module or class level silently never runs; Python keeps the last."""

import ast
import collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("tests", "studio/backend/tests", "unsloth_cli/tests")
SKIP_PARTS = ("vendor", "node_modules", "__pycache__", ".venv")


def _shadowed() -> list[str]:
    found = []
    for root in ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("test_*.py")):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding = "utf-8", errors = "replace"))
            except SyntaxError:
                continue
            scopes = [("module", tree.body)]
            scopes += [(n.name, n.body) for n in tree.body if isinstance(n, ast.ClassDef)]
            for scope, body in scopes:
                defined = [
                    n
                    for n in body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name.startswith("test")
                ]
                counts = collections.Counter(n.name for n in defined)
                for name, count in counts.items():
                    if count > 1:
                        lines = [n.lineno for n in defined if n.name == name]
                        found.append(
                            f"{path.relative_to(REPO)}::{scope}::{name} at lines {lines} "
                            f"(only line {lines[-1]} runs)"
                        )
    return found


def test_no_test_is_overwritten_by_a_later_definition():
    offenders = _shadowed()
    assert not offenders, (
        "these tests are shadowed by a later definition of the same name, so every one "
        "of them but the last is dead code that pytest never collects:\n  "
        + "\n  ".join(offenders)
        + "\n\nRename them if they test different things, or delete the redundant copy. "
        "Leaving it is the worst option: it reads as coverage and provides none."
    )
