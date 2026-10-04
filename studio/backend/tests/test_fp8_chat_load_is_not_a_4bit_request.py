# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat and export loads must not pass `load_in_4bit = True`: Unsloth reads a passed flag as an
explicit request to requantize fp8 checkpoints to NF4, and the chat UI always sends True."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FILES = ("core/inference/inference.py", "core/export/export.py")


def _helper(path):
    tree = ast.parse((_BACKEND / path).read_text(encoding = "utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_in_4bit_kwargs"
    )
    scope = {}
    exec(compile(ast.Module([node], []), path, "exec"), scope)
    return scope["_load_in_4bit_kwargs"]


@pytest.mark.parametrize("path", _FILES)
def test_true_is_left_to_the_loader_default(path):
    helper = _helper(path)
    assert helper(True) == {}
    assert helper(False) == {"load_in_4bit": False}


@pytest.mark.parametrize("path", _FILES)
def test_no_unsloth_loader_call_forwards_the_flag_verbatim(path):
    tree = ast.parse((_BACKEND / path).read_text(encoding = "utf-8"))
    forwarded = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.startswith("Fast")
        and any(kw.arg == "load_in_4bit" and isinstance(kw.value, ast.Name) for kw in node.keywords)
    ]
    assert not forwarded, f"{path}: from_pretrained forwards load_in_4bit at lines {forwarded}"
