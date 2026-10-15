# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bitsandbytes must be imported before the conftest cuda spoof; a failed import cannot be recovered."""

from __future__ import annotations

import ast
from pathlib import Path

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def _accelerator_guard_body(tree: ast.Module) -> list[ast.stmt]:
    for node in tree.body:
        if isinstance(node, ast.If) and "_has_real_accelerator" in ast.dump(node.test):
            return node.body
    raise AssertionError("tests/conftest.py has no `if not _has_real_accelerator():` block")


def _called_names(body: list[ast.stmt]) -> list[str]:
    names = []
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.append(node.func.id)
    return names


def test_conftest_defines_the_bitsandbytes_preimport():
    tree = ast.parse(CONFTEST.read_text(encoding = "utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_preimport_bitsandbytes" in defined, (
        "tests/conftest.py must define _preimport_bitsandbytes(); without it a "
        "bitsandbytes import inside the CUDA spoof window permanently breaks "
        "`import unsloth` for the rest of the process"
    )


def test_bitsandbytes_is_preimported_before_the_cuda_spoof():
    tree = ast.parse(CONFTEST.read_text(encoding = "utf-8"))
    called = _called_names(_accelerator_guard_body(tree))

    assert "_preimport_bitsandbytes" in called, (
        "_preimport_bitsandbytes() is never called inside the "
        "`if not _has_real_accelerator():` block"
    )
    assert "_preload_device_type" in called, "conftest no longer calls _preload_device_type"
    assert called.index("_preimport_bitsandbytes") < called.index("_preload_device_type"), (
        "_preimport_bitsandbytes() must run BEFORE _preload_device_type(), which is what "
        "patches torch.cuda.is_available; importing bitsandbytes inside that window makes "
        "it take its CUDA backend on a CPU-only torch and poisons sys.modules"
    )


def test_preimport_swallows_a_genuinely_missing_wheel():
    """An absent bitsandbytes stays unsloth's own degradation path, not a collection error."""
    tree = ast.parse(CONFTEST.read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_preimport_bitsandbytes"
    )
    assert any(isinstance(node, ast.Try) for node in ast.walk(fn)), (
        "_preimport_bitsandbytes() must guard its import with try/except so a missing or "
        "broken wheel does not turn into a collection error"
    )
