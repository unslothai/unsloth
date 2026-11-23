# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Branches forcing UNSLOTH_COMPILE_DISABLE=partial keep an explicit full disable full."""

import ast
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

import unsloth.models.loader as loader_module


def _loader_tree():
    return ast.parse(Path(loader_module.__file__).read_text(encoding = "utf-8"))


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "YES"])
def test_explicit_full_disable_is_never_downgraded(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", value)
    loader_module._set_partial_compile_disable()
    assert os.environ["UNSLOTH_COMPILE_DISABLE"] == "1"


@pytest.mark.parametrize("value", [None, "0", "partial", "no", "false"])
def test_unset_or_non_full_values_still_get_partial(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", value)
    loader_module._set_partial_compile_disable()
    assert os.environ["UNSLOTH_COMPILE_DISABLE"] == "partial"


def _function_of(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def test_partial_is_written_only_inside_the_helper():
    tree = _loader_tree()
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(c, ast.Constant) and c.value == "partial" for c in ast.walk(node.value))
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value == "UNSLOTH_COMPILE_DISABLE"
            for t in node.targets
        )
    ]
    assert writes, "no partial write found"
    assert all(
        getattr(_function_of(w, parents), "name", None) == "_set_partial_compile_disable"
        for w in writes
    )


def test_all_six_model_branches_call_the_helper():
    """The five original branches plus Inkling 2, which partly disables compile for every new sequence length."""
    calls = [
        node
        for node in ast.walk(_loader_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_partial_compile_disable"
    ]
    assert len(calls) == 6, f"expected the six branches to call the helper, found {len(calls)}"
