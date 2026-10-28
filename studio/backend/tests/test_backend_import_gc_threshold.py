# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
from pathlib import Path

_RUN = Path(__file__).resolve().parent.parent / "run.py"


def _run_server() -> ast.FunctionDef:
    tree = ast.parse(_RUN.read_text(encoding = "utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_server"
    )


def _is_main_import(node) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "main"


def test_main_is_imported_inside_a_try_that_restores_the_threshold():
    tries = [
        node
        for node in ast.walk(_run_server())
        if isinstance(node, ast.Try) and any(_is_main_import(n) for n in node.body)
    ]
    assert len(tries) == 1
    restore = ast.unparse(tries[0].finalbody)
    assert "gc.set_threshold(*_gc_threshold)" in restore


def test_the_threshold_is_raised_before_the_import_and_only_when_gc_runs():
    src = ast.unparse(_run_server())
    raised = src.index("gc.set_threshold(max(_gc_threshold[0], 100000)")
    assert src.index("_gc_threshold = gc.get_threshold()") < raised
    assert src.index("if _gc_threshold[0]:") < raised < src.index("from main import app")
    assert src.count("gc.set_threshold(") == 2
