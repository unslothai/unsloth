# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The model branches that force UNSLOTH_COMPILE_DISABLE=partial must not downgrade an
explicit full disable (1/true/yes): readers that only check for "1" would silently
re-enable their compile paths."""

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
    assert os.environ["UNSLOTH_COMPILE_DISABLE"] == value


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
    """A direct `os.environ["UNSLOTH_COMPILE_DISABLE"] = "partial"` anywhere else is the bug."""
    tree = _loader_tree()
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    writes = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value == "partial"
        ):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and (
                    (
                        isinstance(target.value, ast.Attribute)
                        and target.value.attr == "environ"
                    )
                    or (isinstance(target.value, ast.Name) and target.value.id == "environ")
                )
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "UNSLOTH_COMPILE_DISABLE"
            ):
                writes.append(node)
    assert len(writes) == 1, f"expected one guarded write, found {len(writes)}"
    enclosing = _function_of(writes[0], parents)
    assert enclosing is not None and enclosing.name == "_set_partial_compile_disable"


def test_all_five_model_branches_call_the_helper():
    """gemma3-on-RDNA, deepseek_v41, csm, DISABLE_COMPILE_MODEL_NAMES and 4.x-T5 each route
    their partial disable through the helper, so all of them preserve an explicit "1"."""
    calls = [
        node
        for node in ast.walk(_loader_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_set_partial_compile_disable"
    ]
    assert len(calls) == 5, f"expected the five branches to call the helper, found {len(calls)}"
