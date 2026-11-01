# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Only Vite's outdated pre-bundle 504 is a reload signal; every other failed asset still fails a smoke."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _playwright_robust import is_vite_outdated_dep  # noqa: E402

VITE_DIST = HERE.parents[1] / "studio" / "frontend" / "node_modules" / "vite" / "dist" / "node"


@pytest.mark.parametrize(
    ("status", "status_text", "expected"),
    [
        (504, "Outdated Optimize Dep", True),
        (504, "Optimize Deps Processing Error", False),
        (504, "Gateway Timeout", False),
        (404, "Not Found", False),
        (500, "Internal Server Error", False),
        (200, "OK", False),
    ],
)
def test_only_the_outdated_pre_bundle_answer_is_a_reload(status, status_text, expected):
    response = SimpleNamespace(status = status, status_text = status_text)
    assert is_vite_outdated_dep(response) is expected


def test_the_titlebar_smoke_still_fails_on_other_asset_errors():
    tree = ast.parse((HERE / "playwright_titlebar_blur.py").read_text(encoding = "utf-8"))
    conditions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BoolOp)
        and any(
            isinstance(v, ast.Compare)
            and isinstance(v.left, ast.Attribute)
            and v.left.attr == "status"
            and isinstance(v.ops[0], ast.GtE)
            and getattr(v.comparators[0], "value", None) == 400
            for v in node.values
        )
    ]
    assert len(conditions) == 1, "the asset response guard moved; re-point this test at it"
    exempt = [
        v
        for v in conditions[0].values
        if isinstance(v, ast.UnaryOp)
        and isinstance(v.op, ast.Not)
        and isinstance(v.operand, ast.Call)
        and getattr(v.operand.func, "id", None) == "is_vite_outdated_dep"
    ]
    # One narrow exemption joined with `and`, so a 404 or a processing-error 504 is still recorded.
    assert isinstance(conditions[0].op, ast.And)
    assert len(exempt) == 1


@pytest.mark.skipif(not VITE_DIST.is_dir(), reason = "frontend dependencies are not installed")
def test_vite_still_names_the_outdated_answer_this_way():
    sources = "".join(p.read_text(encoding = "utf-8") for p in VITE_DIST.rglob("*.js"))
    assert 'res.statusMessage = "Outdated Optimize Dep"' in sources
