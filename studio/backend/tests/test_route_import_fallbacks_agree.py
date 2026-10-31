# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fallback import block must bind every name the primary binds; has_resume_state was missed once."""

import ast
import inspect
from pathlib import Path

import pytest


def _import_map(node: ast.AST) -> dict[str, set[str]]:
    """module -> imported names, for every ``from x import ...`` under *node*."""
    out: dict[str, set[str]] = {}
    for sub in ast.walk(node):
        if isinstance(sub, ast.ImportFrom) and sub.module:
            out.setdefault(sub.module, set()).update(
                alias.asname or alias.name for alias in sub.names
            )
    return out


def _try_blocks(source: str) -> list[ast.Try]:
    tree = ast.parse(source)
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(isinstance(h.type, ast.Name) and h.type.id == "ImportError" for h in node.handlers)
    ]


_ROUTES = Path(__file__).resolve().parent.parent / "routes"


@pytest.mark.parametrize(
    "module_path",
    sorted(p for p in _ROUTES.glob("*.py") if p.name != "__init__.py"),
    ids = lambda p: p.name,
)
def test_the_import_fallback_binds_the_same_names(module_path):
    source = module_path.read_text(encoding = "utf-8")
    for block in _try_blocks(source):
        primary = _import_map(ast.Module(body = block.body, type_ignores = []))
        for handler in block.handlers:
            fallback = _import_map(ast.Module(body = handler.body, type_ignores = []))
            for module, names in primary.items():
                if module not in fallback:
                    continue
                missing = names - fallback[module]
                assert not missing, (
                    f"{module_path.name}: the ImportError fallback imports {module} but "
                    f"omits {sorted(missing)}, so those names are undefined whenever the "
                    f"fallback path runs"
                )


def test_has_resume_state_is_bound_on_the_module():
    """The specific regression, checked against the imported module rather than text."""
    from routes import training as training_routes

    assert hasattr(training_routes, "has_resume_state")
    source = inspect.getsource(training_routes)
    assert (
        source.count("has_resume_state,") >= 2
    ), "has_resume_state should appear in both the primary and the fallback import list"
