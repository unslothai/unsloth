# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""FastLanguageModel's LoRA / QLoRA path installs the per-block compile (_block_compile.py) like
FastModel, with its own kill switch and off before torch 2.11 (fused LoRA is not traceable there)."""

import ast
import pathlib

import pytest

_LLAMA = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "llama.py"


def test_patch_peft_model_installs_block_compile():
    tree = ast.parse(_LLAMA.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastLlamaModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "patch_peft_model")
    final = [n for n in fn.body if isinstance(n, ast.Return)][-1]
    before = fn.body[fn.body.index(final) - 1]
    assert isinstance(before, ast.Expr) and ast.unparse(before) == "_legacy_block_compile(model)"


@pytest.fixture
def llama(monkeypatch):
    mod = pytest.importorskip("unsloth.models.llama")
    from unsloth.models import _block_compile
    calls = []
    monkeypatch.setattr(_block_compile, "patch_block_compile", lambda model: calls.append(model) or ["X"])
    monkeypatch.delenv("UNSLOTH_BLOCK_COMPILE_LEGACY", raising = False)
    return mod, calls


@pytest.mark.parametrize(
    "env, traceable, installed",
    [
        (None, True, True),
        (None, False, False),
        ("1", False, True),
        ("0", True, False),
    ],
)
def test_legacy_gate(llama, monkeypatch, env, traceable, installed):
    mod, calls = llama
    from unsloth.kernels import fast_lora
    monkeypatch.setattr(fast_lora, "TRACE_LORA_FUNCTIONS", traceable)
    if env is not None:
        monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE_LEGACY", env)
    model = object()
    assert mod._legacy_block_compile(model) == (["X"] if installed else [])
    assert calls == ([model] if installed else [])
