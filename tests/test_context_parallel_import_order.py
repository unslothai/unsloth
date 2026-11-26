# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""unsloth.context_parallel defines get_cp_manager before any module-level import from unsloth.utils.

Importing any unsloth.utils submodule first runs unsloth/utils/__init__.py, which imports attention_dispatch,
which does `from ..context_parallel import get_cp_manager`. When context_parallel is the first unsloth module a
process imports (a spawned CPU / gloo rank), that name must already exist on the partially initialised module,
or the rank dies with "cannot import name 'get_cp_manager' from partially initialized module".
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "context_parallel.py"


def _module_level_utils_imports(tree):
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and (node.module or "").split(".")[0] == "utils"
        ):
            yield node
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").startswith("unsloth.utils")
        ):
            yield node


def test_get_cp_manager_is_defined_before_any_utils_import():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_cp_manager"
    )
    early = [
        f"line {node.lineno}: from {'.' * node.level}{node.module} import ..."
        for node in _module_level_utils_imports(tree)
        if node.lineno < definition.lineno
    ]
    assert not early, (
        f"these module-level imports run before get_cp_manager (line {definition.lineno}) is defined, so a process "
        f"that imports unsloth.context_parallel first hits a circular import: {early}"
    )


def test_the_check_sees_the_utils_imports():
    """Not vacuous: the module does import from unsloth.utils at module level."""
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    assert any(True for _ in _module_level_utils_imports(tree))
