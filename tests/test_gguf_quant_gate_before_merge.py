# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""save_pretrained_gguf must reject a bad quant name, or an IQ quant without an imatrix, before the 16-bit merge."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"
SAVE_SRC = SAVE_PY.read_text(encoding = "utf-8")
SAVE_TREE = ast.parse(SAVE_SRC, filename = str(SAVE_PY))

# Everything the gate needs; stdlib-only, so it runs without save.py's GPU import chain.
_NAMES = (
    "ALLOWED_QUANTS",
    "IMATRIX_QUANTS",
    "_normalize_quantization_methods",
    "_imatrix_is_enabled",
    "_check_quant_method",
    "_check_gguf_quantization_methods",
)


def _node(name):
    for node in SAVE_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node
    raise AssertionError(f"{name} not found in {SAVE_PY.name}")


def _gate():
    namespace = {}
    for name in _NAMES:
        exec(compile(ast.get_source_segment(SAVE_SRC, _node(name)), name, "exec"), namespace)
    return namespace["_check_gguf_quantization_methods"]


def _calls(func_name):
    """(lineno, callee) for every call inside func_name, callee as `name` or `obj.attr`."""
    out = []
    for node in ast.walk(_node(func_name)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.append((node.lineno, f.id))
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                out.append((node.lineno, f"{f.value.id}.{f.attr}"))
    return out


@pytest.mark.parametrize(
    "quantization_method, imatrix_file, match",
    [
        ("q4km", None, "not supported"),
        ("q4_k_m ", None, "not supported"),
        (["q4_k_m", "q8km"], None, "not supported"),
        ("iq2_xxs", None, "importance matrix"),
        ("IQ2_XXS", False, "importance matrix"),
        (["q4_k_m", "iq4_xs"], None, "importance matrix"),
    ],
)
def test_bad_quant_is_rejected(quantization_method, imatrix_file, match):
    with pytest.raises(RuntimeError, match = match):
        _gate()(quantization_method, imatrix_file)


@pytest.mark.parametrize(
    "quantization_method, imatrix_file",
    [
        ("fast_quantized", None),
        ("not_quantized", None),
        ("quantized", None),
        (None, None),
        ([None], None),
        ([], None),
        ("Q4_K_M", None),
        (["q4_k_m", "q8_0"], None),
        (("q5_k_m",), None),
        ("bf16", None),
        ("f32", None),
        ("q2_k_l", None),
        ("iq2_xxs", True),
        ("iq2_xxs", "/x/imatrix.gguf"),
    ],
)
def test_supported_quant_is_accepted(quantization_method, imatrix_file):
    _gate()(quantization_method, imatrix_file)


def test_every_listed_quant_is_accepted():
    gate = _gate()
    namespace = {}
    for name in ("ALLOWED_QUANTS", "IMATRIX_QUANTS"):
        exec(compile(ast.get_source_segment(SAVE_SRC, _node(name)), name, "exec"), namespace)
    gate(list(namespace["ALLOWED_QUANTS"]), None)
    gate(list(namespace["IMATRIX_QUANTS"]), True)


def test_gate_runs_before_anything_is_merged_or_downloaded():
    calls = _calls("unsloth_save_pretrained_gguf")
    gate = [line for line, name in calls if name == "_check_gguf_quantization_methods"]
    assert gate, "unsloth_save_pretrained_gguf never calls _check_gguf_quantization_methods"
    expensive = [
        line
        for line, name in calls
        if name
        in (
            "unsloth_generic_save",
            "self.save_pretrained",
            "_stage_reused_checkpoint",
            "_resolve_imatrix_file",
            "_preflight_gguf_disk",
        )
    ]
    assert expensive, "expected merge / imatrix / preflight calls in unsloth_save_pretrained_gguf"
    assert min(gate) < min(expensive)


def test_save_to_gguf_uses_the_same_gate():
    assert "_check_quant_method" in {name for _, name in _calls("save_to_gguf")}
