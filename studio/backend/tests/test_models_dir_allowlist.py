# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import os
from pathlib import Path

import pytest


_backend_root = Path(__file__).resolve().parent.parent
_models_src = _backend_root / "routes" / "models.py"


def _load_allowlist_helpers():
    """Load the real helpers without importing the full FastAPI route module."""
    tree = ast.parse(_models_src.read_text())
    wanted = {"_path_is_same_or_child", "_resolve_allowed_models_dir"}
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body = body, type_ignores = [])
    ast.fix_missing_locations(module)
    ns: dict = {"Path": Path, "os": os}
    exec(compile(module, f"<extracted {_models_src}>", "exec"), ns)
    return ns["_resolve_allowed_models_dir"]


resolve_allowed_models_dir = _load_allowlist_helpers()


def test_allowed_subfolder_is_not_widened_to_root(tmp_path):
    allowed = tmp_path / "models"
    requested = allowed / "nested"
    requested.mkdir(parents = True)

    assert resolve_allowed_models_dir(str(requested), [allowed]) == requested.resolve()


def test_sibling_with_shared_prefix_is_not_allowed(tmp_path):
    allowed = tmp_path / "models"
    sibling = tmp_path / "models-other"
    allowed.mkdir()
    sibling.mkdir()

    with pytest.raises(ValueError, match = "Directory not allowed"):
        resolve_allowed_models_dir(str(sibling), [allowed])


def test_symlink_inside_allowed_root_to_outside_is_not_allowed(tmp_path):
    allowed = tmp_path / "models"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    link = allowed / "external"
    try:
        link.symlink_to(outside, target_is_directory = True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink unavailable on this platform: {exc}")

    with pytest.raises(ValueError, match = "Directory not allowed"):
        resolve_allowed_models_dir(str(link), [allowed])
