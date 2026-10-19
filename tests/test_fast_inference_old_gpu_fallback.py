# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import sys
import types
from pathlib import Path

import pytest

LOADER = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "loader.py"
GEMMA4_TYPES = ["gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"]


def _helper(monkeypatch, device_type, capability):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(get_device_capability = capability)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_vllm_needs_newer_gpu":
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "VLLM_NEEDS_SM80_MODEL_TYPES" for t in node.targets
        ):
            wanted.append(node)
    assert len(wanted) == 2
    namespace = {"DEVICE_TYPE": device_type}
    exec(compile(ast.Module(body = wanted, type_ignores = []), "loader.py", "exec"), namespace)
    return namespace["_vllm_needs_newer_gpu"]


@pytest.mark.parametrize("model_type", GEMMA4_TYPES)
def test_gemma4_falls_back_below_sm80(monkeypatch, model_type):
    f = _helper(monkeypatch, "cuda", lambda: (7, 5))
    assert f([model_type]) == model_type


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (12, 0)])
def test_gemma4_keeps_vllm_on_sm80_and_newer(monkeypatch, capability):
    f = _helper(monkeypatch, "cuda", lambda: capability)
    assert f(GEMMA4_TYPES) is None


def test_other_models_keep_vllm_on_t4(monkeypatch):
    f = _helper(monkeypatch, "cuda", lambda: (7, 5))
    for model_types in (["qwen3"], ["llama"], ["gemma3", "gemma3_text"], ["qwen3_5"]):
        assert f(model_types) is None


def test_non_cuda_devices_untouched(monkeypatch):
    def boom():
        raise AssertionError("capability must not be probed off CUDA")

    for device_type in ("hip", "xpu", "mps", "cpu"):
        f = _helper(monkeypatch, device_type, boom)
        assert f(["gemma4"]) is None


def test_failed_probe_keeps_vllm(monkeypatch):
    def broken():
        raise RuntimeError("no device")

    f = _helper(monkeypatch, "cuda", broken)
    assert f(["gemma4"]) is None


def test_fallback_drops_max_num_batched_tokens(capsys):
    # load_vllm takes it; transformers' from_pretrained rejects it once vLLM is switched off.
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_drop_vllm_engine_kwargs")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_VLLM_ENGINE_KWARGS" for t in node.targets)
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body = wanted, type_ignores = []), "loader.py", "exec"), namespace)
    kwargs = {"max_num_batched_tokens": 4096, "token": "x"}
    namespace["_drop_vllm_engine_kwargs"](kwargs)
    assert kwargs == {"token": "x"}
