# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-safe policy tests for FastSentenceTransformer torch.compile handling.

The full Unsloth package cannot import on every CPU runner, so these tests extract and
execute the production method itself. The assertions are behavioral: fake model/config
objects expose the same public contracts as Transformers and torch.compile is replaced
with a call-counting seam.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path


_SOURCE_PATH = (
    Path(__file__).resolve().parents[2] / "unsloth" / "models" / "sentence_transformer.py"
)


class _CompileRecorder:
    def __init__(self):
        self.calls = []

    def compile(self, model, *, mode):
        compiled = types.SimpleNamespace(compiled_from=model, mode=mode)
        self.calls.append((model, mode, compiled))
        return compiled


def _load_apply_torch_compile(recorder):
    source = _SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SOURCE_PATH))
    fast_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FastSentenceTransformer"
    )
    method = next(
        node
        for node in fast_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_torch_compile"
    )
    method.decorator_list = []
    isolated = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"torch": recorder}
    exec(compile(isolated, str(_SOURCE_PATH), "exec"), namespace)
    return namespace["_apply_torch_compile"]


class _TransformerModule:
    def __init__(self, auto_model):
        self.auto_model = auto_model


class _SentenceTransformer:
    def __init__(self, inner):
        self.module = _TransformerModule(inner)

    def __getitem__(self, index):
        assert index == 0
        return self.module


def test_qwen35_config_disables_compile_for_sentence_transformer():
    recorder = _CompileRecorder()
    apply_compile = _load_apply_torch_compile(recorder)
    inner = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen3_5"))
    model = _SentenceTransformer(inner)

    returned = apply_compile(model, mode="default")

    assert returned is model
    assert recorder.calls == []
    assert model[0].auto_model is inner
    assert "_orig_mod" not in model.__dict__


def test_qwen35_class_fallback_disables_compile_when_config_is_absent():
    recorder = _CompileRecorder()
    apply_compile = _load_apply_torch_compile(recorder)
    qwen_class = type("Qwen3_5Model", (), {})
    inner = qwen_class()
    model = _SentenceTransformer(inner)

    returned = apply_compile(model)

    assert returned is model
    assert recorder.calls == []
    assert model[0].auto_model is inner


def test_precompiled_qwen35_model_is_not_compiled_again():
    recorder = _CompileRecorder()
    apply_compile = _load_apply_torch_compile(recorder)
    original = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen3_5"))
    precompiled = types.SimpleNamespace(_orig_mod=original)

    returned = apply_compile(precompiled, mode="reduce-overhead")

    assert returned is precompiled
    assert recorder.calls == []


def test_non_qwen_sentence_transformer_compiles_once_and_keeps_wrapper():
    recorder = _CompileRecorder()
    apply_compile = _load_apply_torch_compile(recorder)
    inner = types.SimpleNamespace(config=types.SimpleNamespace(model_type="bert"))
    model = _SentenceTransformer(inner)

    returned = apply_compile(model, mode="default")

    assert returned is model
    assert len(recorder.calls) == 1
    assert recorder.calls[0][:2] == (inner, "default")
    assert model[0].auto_model is recorder.calls[0][2]
    assert model.__dict__["_orig_mod"] is model


def test_non_sentence_transformer_control_preserves_compiled_return_value():
    recorder = _CompileRecorder()
    apply_compile = _load_apply_torch_compile(recorder)
    model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen3"))

    returned = apply_compile(model, mode="max-autotune")

    assert len(recorder.calls) == 1
    assert recorder.calls[0][:2] == (model, "max-autotune")
    assert returned is recorder.calls[0][2]
