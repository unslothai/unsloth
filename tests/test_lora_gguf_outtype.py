# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_SAVE_PY = Path(__file__).resolve().parent.parent / "unsloth" / "save.py"


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning_once(self, message):
        self.warnings.append(message)


def _load_helper():
    # Importing save.py needs the full ML stack, so lift the helper out with ast.
    tree = ast.parse(_SAVE_PY.read_text(encoding = "utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_lora_gguf_outtype")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_LORA_GGUF_OUTTYPES" for t in node.targets)
        )
    ]
    if not any(isinstance(node, ast.FunctionDef) for node in nodes):
        raise AssertionError("unsloth/save.py does not define _lora_gguf_outtype")
    logger = _Logger()
    namespace = {"logger": logger}
    exec(compile(ast.Module(nodes, []), str(_SAVE_PY), "exec"), namespace)
    return namespace["_lora_gguf_outtype"], logger


@pytest.mark.parametrize(
    "quantization_method, expected",
    [
        ("Q8_0", "q8_0"),
        ("BF16", "bf16"),
        ("F32", "f32"),
        (["Q8_0"], "q8_0"),
        (("BF16",), "bf16"),
    ],
)
def test_lora_outtype_is_case_insensitive(quantization_method, expected):
    outtype, logger = _load_helper()
    assert outtype(quantization_method) == expected
    assert logger.warnings == []


@pytest.mark.parametrize("quantization_method", ["q8_0", "f16", "bf16", "f32", "auto"])
def test_lora_outtype_keeps_valid_lowercase_names(quantization_method):
    outtype, logger = _load_helper()
    assert outtype(quantization_method) == quantization_method
    assert logger.warnings == []


@pytest.mark.parametrize("quantization_method", [None, "fast_quantized", ["fast_quantized"]])
def test_lora_outtype_default_is_f16_without_warning(quantization_method):
    outtype, logger = _load_helper()
    assert outtype(quantization_method) == "f16"
    assert logger.warnings == []


@pytest.mark.parametrize("quantization_method", ["q4_k_m", "Q4_K_M", ["q8_0", "f16"]])
def test_lora_outtype_unsupported_falls_back_to_f16_with_warning(quantization_method):
    outtype, logger = _load_helper()
    assert outtype(quantization_method) == "f16"
    assert len(logger.warnings) == 1


class _ReachedPastValidation(Exception):
    pass


class _Peft:
    pass


def _load_writer():
    tree = ast.parse(_SAVE_PY.read_text(encoding = "utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_unsloth_save_lora_gguf")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_LORA_GGUF_OUTTYPES" for t in node.targets)
        )
    ]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []

    def _stop(token):
        raise _ReachedPastValidation

    namespace = {
        "PeftModelForCausalLM": _Peft,
        "PeftModel": _Peft,
        "_clean_save_token": _stop,
    }
    exec(compile(ast.Module(nodes, []), str(_SAVE_PY), "exec"), namespace)
    return namespace["_unsloth_save_lora_gguf"]


@pytest.mark.parametrize("outtype", ["Q8_0", "BF16", "F16", "f32"])
def test_lora_writer_accepts_any_case_outtype(outtype):
    with pytest.raises(_ReachedPastValidation):
        _load_writer()(_Peft(), None, "out", outtype = outtype)


@pytest.mark.parametrize("outtype", ["Q4_K_M", "q4_k_m"])
def test_lora_writer_still_rejects_unsupported_outtype(outtype):
    with pytest.raises(ValueError, match = "outtype must be one of"):
        _load_writer()(_Peft(), None, "out", outtype = outtype)
