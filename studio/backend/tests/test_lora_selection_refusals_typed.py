# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every generation-time LoRA selection refusal raises LoraSelectionError, so /v1/images/generations answers it as a
400 (param loras) instead of a sanitized 500; /images/generate still sees a ValueError."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.inference.diffusion_lora import LoraSelectionError

_INFERENCE = Path(__file__).resolve().parent.parent / "core" / "inference"

# (file, function): the generate-time paths that refuse a LoRA selection the loaded model cannot take.
_REFUSERS = [
    ("diffusion.py", "_remerge_quant_loras"),
    ("diffusion.py", "_refuse_weight_change_on_streamed_graph"),
    ("diffusion.py", "_resolve_lora_set"),
    ("diffusion.py", "_apply_loras"),
    ("diffusion.py", "_adjust_baked_loras"),
    ("diffusion.py", "_check_h3_image_loras"),
    ("sd_cpp_backend.py", "generate"),
]


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert found, f"{name} not found"
    return found[0]


def _raised_names(fn: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            names.append(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    return names


def _mentions_lora(fn: ast.FunctionDef, raise_node: ast.Raise) -> bool:
    # A raise chained from an internal failure ("Failed to apply LoRA: ...") is a server error and stays sanitized.
    if raise_node.cause is not None:
        return False
    text = ast.unparse(raise_node).lower()
    return "lora" in text or "adapter" in text or "blocked" in text or "unavailable" in text


@pytest.mark.parametrize("filename, function", _REFUSERS)
def test_lora_refusals_are_typed(filename, function):
    tree = ast.parse((_INFERENCE / filename).read_text(encoding = "utf-8"))
    fn = _function(tree, function)
    plain = [
        ast.unparse(node)[:120]
        for node in ast.walk(fn)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and getattr(node.exc.func, "id", None) == "ValueError"
        and _mentions_lora(fn, node)
    ]
    assert (
        not plain
    ), f"{filename}:{function} still refuses a LoRA selection with a plain ValueError: {plain}"
    # LoraReloadRequiredError is the LoraSelectionError subclass for refusals a reload resolves.
    assert {"LoraSelectionError", "LoraReloadRequiredError"} & set(_raised_names(fn))


def test_the_typed_refusal_is_still_a_value_error():
    assert issubclass(LoraSelectionError, ValueError)
