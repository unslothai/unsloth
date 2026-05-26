# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
from pathlib import Path
from typing import Optional


_backend_root = Path(__file__).resolve().parent.parent
_llama_cpp_src = _backend_root / "core" / "inference" / "llama_cpp.py"


def _load_ollama_error_helper():
    """Load the real helper without importing the full inference stack."""
    tree = ast.parse(_llama_cpp_src.read_text())
    fn = next(
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_ollama_llama_cpp_load_error"
        )
    )
    module = ast.Module(body = [fn], type_ignores = [])
    ast.fix_missing_locations(module)
    ns: dict = {"Optional": Optional}
    exec(compile(module, f"<extracted {_llama_cpp_src}>", "exec"), ns)
    return ns["_ollama_llama_cpp_load_error"]


ollama_llama_cpp_load_error = _load_ollama_error_helper()


def test_qwen35_rope_error_suggests_updating_llama_cpp():
    message = ollama_llama_cpp_load_error(
        "error loading model hyperparameters: key "
        "qwen35.rope.dimension_sections has wrong array length; expected 4, got 3"
    )

    assert message is not None
    assert "unsloth studio update" in message
    assert "Ollama connection" in message


def test_generic_failed_ollama_load_suggests_update_or_ollama_runtime():
    message = ollama_llama_cpp_load_error(
        "llama_model_load_from_file_impl: failed to load model"
    )

    assert message is not None
    assert "Studio's llama.cpp bridge" in message
    assert "Ollama connection" in message


def test_unrelated_server_output_has_no_ollama_specific_message():
    assert ollama_llama_cpp_load_error("address already in use") is None
