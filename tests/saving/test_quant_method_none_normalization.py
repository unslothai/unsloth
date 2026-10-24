"""None quant methods must map to q8_0 before .lower() runs, or the loop raises AttributeError."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"
SAVE_SRC = SAVE_PY.read_text(encoding = "utf-8")
SAVE_TREE = ast.parse(SAVE_SRC, filename = str(SAVE_PY))

TARGETS = (
    ("unsloth_save_pretrained_gguf", "quantization_methods"),
    ("save_to_gguf_generic", "new_quantization_methods"),
)


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {SAVE_PY.name}")


def _quant_loop(func_name):
    func = _func(SAVE_TREE, func_name)
    for node in ast.walk(func):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
            and isinstance(node.iter.args[0], ast.Name)
            and node.iter.args[0].id == "quantization_method"
        ):
            return node
    raise AssertionError(f"quant-normalization loop not found in {func_name}")


def _run_loop(func_name, out_var, quantization_method):
    loop_src = ast.get_source_segment(SAVE_SRC, _quant_loop(func_name))
    namespace = {out_var: [], "quantization_method": quantization_method}
    exec(loop_src, {"__builtins__": __builtins__}, namespace)
    return namespace[out_var]


@pytest.mark.parametrize("func_name, out_var", TARGETS)
def test_none_element_maps_to_q8_0(func_name, out_var):
    assert _run_loop(func_name, out_var, [None]) == ["q8_0"]


@pytest.mark.parametrize("func_name, out_var", TARGETS)
def test_none_mixed_with_strings(func_name, out_var):
    assert _run_loop(func_name, out_var, ["Q4_K_M", None]) == ["q4_k_m", "q8_0"]


@pytest.mark.parametrize("func_name, out_var", TARGETS)
def test_string_methods_unchanged(func_name, out_var):
    methods = ["not_quantized", "fast_quantized", "quantized", "Q8_0"]
    assert _run_loop(func_name, out_var, methods) == ["f16", "q8_0", "q4_k_m", "q8_0"]
