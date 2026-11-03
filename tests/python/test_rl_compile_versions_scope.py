# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""patch_trl_rl_trainers reads package versions once per pass, and imports against an older unsloth_zoo."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest


RL_PATH = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "rl.py"


def _tree():
    return ast.parse(RL_PATH.read_text(encoding = "utf-8"))


def _scope_import_block(tree):
    for node in tree.body:
        if isinstance(node, ast.Try):
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module == "unsloth_zoo.compiler":
                    if any(a.name == "_compile_versions_scope" for a in stmt.names):
                        return node
    raise AssertionError("guarded import of _compile_versions_scope not found in rl.py")


def _run_import_block(monkeypatch, fake_compiler):
    block = _scope_import_block(_tree())
    module = ast.fix_missing_locations(ast.Module(body = [block], type_ignores = []))
    zoo = types.ModuleType("unsloth_zoo")
    zoo.compiler = fake_compiler
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.compiler", fake_compiler)
    namespace = {}
    exec(compile(module, str(RL_PATH), "exec"), namespace)
    return namespace["_compile_versions_scope"]


def test_patch_pass_is_wrapped_in_versions_scope():
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == "patch_trl_rl_trainers":
            names = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
            assert "_compile_versions_scope" in names
            return
    raise AssertionError("patch_trl_rl_trainers not found in rl.py")


def test_older_zoo_without_scope_falls_back_to_identity(monkeypatch):
    scope = _run_import_block(monkeypatch, types.ModuleType("unsloth_zoo.compiler"))

    def f(a, b = 2):
        return a + b

    assert scope(f) is f


def test_zoo_scope_is_used_when_present(monkeypatch):
    fake = types.ModuleType("unsloth_zoo.compiler")

    def zoo_scope(function):
        return function

    fake._compile_versions_scope = zoo_scope
    assert _run_import_block(monkeypatch, fake) is zoo_scope


def test_one_version_read_set_per_patch_pass(monkeypatch):
    try:
        from unsloth.models import rl as _rl
        import unsloth_zoo.compiler as zc
    except Exception as e:
        pytest.skip(f"unsloth.models.rl not importable here: {e}")
    if not hasattr(zc, "_compile_versions_scope"):
        pytest.skip("unsloth_zoo predates _compile_versions_scope")
    reads = []
    real = zc.importlib_version

    def counted(package):
        reads.append(package)
        return real(package)

    generated = []

    def fake_generate(trainer_file):
        generated.append((trainer_file, zc._get_compile_versions()))

    monkeypatch.setattr(zc, "importlib_version", counted)
    monkeypatch.setattr(_rl, "_patch_trl_rl_trainers", fake_generate)
    monkeypatch.setattr(_rl, "_import_trl_experimental_trainers", lambda: {})
    _rl.patch_trl_rl_trainers()
    assert len(generated) > 1
    assert len(reads) == len(zc._resolve_compile_versions()), reads
    assert len({versions for _, versions in generated}) == 1
    reads.clear()
    zc._get_compile_versions()
    assert reads
