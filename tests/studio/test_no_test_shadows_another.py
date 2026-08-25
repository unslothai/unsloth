# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No test function is defined twice in one scope.

Python keeps the last definition. A test that shares a name with a later one in the same
module or class is simply overwritten before pytest ever collects it: it appears in the
file, it is maintained, it is reviewed, and it never runs. Nothing reports this. It is
not a failure, not a skip, and not an error -- the test is absent, and absence is what
green looks like.

Found by sweeping all 23,895 test bodies in the repo for duplicates, which turned up
exactly one: TestParser.test_xml_param_preserves_leading_indentation in
test_safetensors_tool_loop.py, defined at lines 120 and 156, where the first was dead.
One in the whole tree is a good result, and it is cheap to keep it at one.

Scoped to module level and class level, which is where pytest collects from. A function
nested inside another function is not collected either way, so it is not this test's
business.
"""

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
                continue  # a file that does not parse is a different problem, loudly
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
