# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A test module other tests import top-level must not need its package for its own helpers.

Several tests put this directory on ``sys.path`` and run ``from test_diffusion_backend import ...``. That
imports the file as a top-level module with no parent package, so a bare ``from ._helper import x`` in it
raises ``ImportError: attempted relative import with no known parent package`` and every importer fails to
collect. The relative import has to sit in a ``try`` whose ``except ImportError`` imports the helper top-level.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS = Path(__file__).resolve().parent


def _statements(tree: ast.Module):
    """Every node except expressions and what lies below them. Imports are statements and an
    expression never contains a statement, so this finds the same imports as ast.walk."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(c for c in ast.iter_child_nodes(node) if not isinstance(c, ast.expr))


def _sibling_imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in _statements(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").startswith("test_")
        ):
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.startswith("test_"))
    return names


def _bare_relative_imports(tree: ast.Module) -> list[int]:
    """Top-level relative imports not guarded by a try whose handler catches ImportError."""
    guarded = set()
    for node in tree.body:
        if isinstance(node, ast.Try) and any(
            isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError")
            for h in node.handlers
        ):
            guarded.update(id(child) for child in node.body)
    return [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level > 0 and id(node) not in guarded
    ]


def test_modules_imported_top_level_import_their_helpers_without_a_package():
    # One parse per file, keeping only what the checks need: holding every tree at once cost
    # most of this test's time in allocation and garbage collection.
    siblings, relative = {}, {}
    for path in _TESTS.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        siblings[path.stem] = _sibling_imports(tree)
        relative[path.stem] = _bare_relative_imports(tree)
    imported = set().union(*siblings.values())
    assert (
        "test_diffusion_backend" in imported
    ), "the sibling-import scan found none of the known importers"
    bad = {name: lines for name in sorted(imported & relative.keys()) if (lines := relative[name])}
    assert (
        not bad
    ), f"bare relative imports in modules other tests import top-level (line numbers): {bad}"


def test_the_check_tells_a_guarded_import_from_a_bare_one():
    bare = ast.parse("from ._helper import x\n")
    guarded = ast.parse(
        "try:\n    from ._helper import x\nexcept ImportError:\n    from _helper import x\n"
    )
    assert _bare_relative_imports(bare) == [1]
    assert _bare_relative_imports(guarded) == []
