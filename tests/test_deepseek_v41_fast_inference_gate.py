# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DeepSeek-V4.1 fast_inference: served by vLLM as the native text model, refused before vLLM allocates when the
weights cannot be shared, and its vLLM-shared norms are not replaced by float32 copies."""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

VISION_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"
SOURCE = VISION_PATH.read_text(encoding = "utf-8")
TREE = ast.parse(SOURCE)


def _function(name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {}
            exec(
                compile(ast.Module(body = [node], type_ignores = []), str(VISION_PATH), "exec"),
                namespace,
            )
            return namespace[name]
    raise AssertionError(f"{name} not found")


def _with_zoo(monkeypatch, refusal):
    zoo = types.ModuleType("unsloth_zoo")
    module = types.ModuleType("unsloth_zoo.vllm_deepseek_v41")
    module.deepseek_v41_fast_inference_refusal = refusal
    zoo.vllm_deepseek_v41 = module
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_deepseek_v41", module)


def test_refusal_from_the_zoo_is_raised_with_guidance(monkeypatch):
    seen = {}

    def refusal(
        config,
        load_in_4bit = False,
        load_in_8bit = False,
    ):
        seen.update(load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)
        return "the checkpoint is quantized (fp8)"

    _with_zoo(monkeypatch, refusal)
    check = _function("_deepseek_v41_fast_inference_check")
    with pytest.raises(
        NotImplementedError, match = "(?s)quantized \\(fp8\\).*fast_inference = False"
    ):
        check(SimpleNamespace(model_type = "deepseek_v41"), load_in_4bit = True)
    assert seen == {"load_in_4bit": True, "load_in_8bit": False}


def test_shareable_load_passes(monkeypatch):
    _with_zoo(monkeypatch, lambda config, **kw: None)
    _function("_deepseek_v41_fast_inference_check")(SimpleNamespace(model_type = "deepseek_v41"))


def test_old_zoo_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_deepseek_v41", None)
    with pytest.raises(NotImplementedError, match = "newer unsloth_zoo"):
        _function("_deepseek_v41_fast_inference_check")(SimpleNamespace(model_type = "deepseek_v41"))


def test_deepseek_v41_skips_the_vlm_allowlist_on_the_vllm_path():
    gate = SOURCE.index(
        'if fast_inference and any(mt.startswith("deepseek_v41") for mt in model_types):'
    )
    allowlist = SOURCE.index("if is_vlm_config and fast_inference:")
    assert gate < allowlist
    assert "is_vlm_config = False" in SOURCE[gate:allowlist]
    assert "_deepseek_v41_fast_inference_check(" in SOURCE[gate:allowlist]


def test_vllm_shared_norms_are_not_upcast():
    start = SOURCE.index(
        'if os.environ.get("UNSLOTH_HIGH_PRECISION_LAYERNORM", "0") == "1" and not ('
    )
    assert 'getattr(model, "_unsloth_vllm_shared_norms", False)' in SOURCE[start : start + 300]


def test_the_upcast_gate_skips_a_model_marked_shared(monkeypatch):
    import os

    gate = next(
        node.test
        for node in ast.walk(TREE)
        if isinstance(node, ast.If)
        and "UNSLOTH_HIGH_PRECISION_LAYERNORM" in ast.unparse(node.test)
        and "_unsloth_vllm_shared_norms" in ast.unparse(node.test)
    )
    test = compile(ast.Expression(gate), str(VISION_PATH), "eval")
    monkeypatch.setenv("UNSLOTH_HIGH_PRECISION_LAYERNORM", "1")
    shared = SimpleNamespace(_unsloth_vllm_shared_norms = True)
    for model, upcast in ((shared, False), (SimpleNamespace(), True)):
        scope = {"os": os, "fast_inference": True, "model_types": ["deepseek_v41"], "model": model}
        assert eval(test, scope) is upcast
