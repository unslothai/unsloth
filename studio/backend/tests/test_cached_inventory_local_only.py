# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
from pathlib import Path


_backend_root = Path(__file__).resolve().parent.parent
_models_src = _backend_root / "routes" / "models.py"


def _function_node(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(_models_src.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    raise AssertionError(f"Function not found: {name}")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        fn = child.func
        if isinstance(fn, ast.Name):
            names.add(fn.id)
        elif isinstance(fn, ast.Attribute):
            names.add(fn.attr)
    return names


def test_cached_inventory_endpoints_do_not_call_hf_metadata():
    for endpoint in ("list_cached_gguf", "list_cached_models"):
        calls = _called_names(_function_node(endpoint))
        assert "_classify_cached_repos" not in calls
        assert "model_info" not in calls
