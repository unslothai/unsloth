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


def _sibling_imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
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
    trees = {
        path.stem: ast.parse(path.read_text(encoding = "utf-8")) for path in _TESTS.glob("test_*.py")
    }
    imported = set().union(*(_sibling_imports(tree) for tree in trees.values()))
    assert (
        "test_diffusion_backend" in imported
    ), "the sibling-import scan found none of the known importers"
    bad = {
        name: lines
        for name in sorted(imported & trees.keys())
        if (lines := _bare_relative_imports(trees[name]))
    }
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
