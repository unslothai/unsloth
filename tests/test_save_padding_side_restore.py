# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A save/push that raises must not leave the tokenizer's `padding_side` stuck at "left"."""

import ast
import contextlib
import types
from pathlib import Path

import pytest


_SAVE_PY = Path(__file__).resolve().parent.parent / "unsloth" / "save.py"
_SOURCE = _SAVE_PY.read_text(encoding = "utf-8")
_TREE = ast.parse(_SOURCE)


def _guard():
    node = next(
        n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_left_padding_side"
    )
    node.decorator_list = []
    namespace = {}
    exec(compile(ast.Module([node], []), str(_SAVE_PY), "exec"), namespace)
    return contextlib.contextmanager(namespace["_left_padding_side"])


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("processor", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_left_padding_side_restores(padding_side, processor, fail):
    source = types.SimpleNamespace(padding_side = padding_side)
    tokenizer = types.SimpleNamespace(tokenizer = source) if processor else source
    seen = []

    def write():
        seen.append(source.padding_side)
        if fail:
            raise OSError("No space left on device")

    with pytest.raises(OSError) if fail else contextlib.nullcontext():
        with _guard()(tokenizer):
            write()
    assert seen == ["left"]
    assert source.padding_side == padding_side


def test_every_left_flip_goes_through_the_guard():
    flips = []
    for fn in ast.walk(_TREE):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Attribute) and t.attr == "padding_side" for t in node.targets
                )
                and isinstance(node.value, ast.Constant)
                and node.value.value == "left"
            ):
                flips.append(fn.name)
    assert set(flips) == {"_left_padding_side"}, flips
    uses = [
        n
        for n in ast.walk(_TREE)
        if isinstance(n, ast.withitem)
        and isinstance(n.context_expr, ast.Call)
        and getattr(n.context_expr.func, "id", None) == "_left_padding_side"
    ]
    assert len(uses) >= 4
