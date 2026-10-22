# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""vLLM refuses bf16 on an Arc A770 (vllm/platforms/xpu.py check_if_supports_dtype). When vLLM will
own the weights, the HF load takes fp16 as well, so the engine and the shared weights agree."""

import ast
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"


def _vllm_gate(name):
    tree = ast.parse((MODELS / name).read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "xpu_vllm_refuses_bf16()" in ast.unparse(node.test):
            return compile(ast.Module(body = [node], type_ignores = []), name, "exec")
    raise AssertionError(f"vLLM bf16 gate not found in {name}")


def _resolve(
    name,
    device_type,
    dtype,
    fast_inference,
    refuses,
    vllm_available = True,
    env = None,
):
    namespace = {
        "torch": torch,
        "dtype": dtype,
        "DEVICE_TYPE": device_type,
        "fast_inference": fast_inference,
        "num_labels": None,
        "_vllm_will_load_weights": lambda fi, nl: bool(fi) and nl is None and vllm_available,
        "is_vLLM_available": lambda: vllm_available,
        "xpu_vllm_refuses_bf16": lambda: refuses,
        "logger": types.SimpleNamespace(warning_once = lambda *a, **k: None),
        "os": types.SimpleNamespace(environ = dict(env or {})),
    }
    exec(_vllm_gate(name), namespace)
    return namespace["dtype"]


@pytest.mark.parametrize("name", ["llama.py", "vision.py"])
def test_refused_bf16_becomes_fp16_only_for_a_vllm_load_on_xpu(name):
    assert _resolve(name, "xpu", torch.bfloat16, True, refuses = True) is torch.float16
    # vLLM accepts it, vLLM is not loading the weights, or not XPU: unchanged.
    assert _resolve(name, "xpu", torch.bfloat16, True, refuses = False) is torch.bfloat16
    assert _resolve(name, "xpu", torch.bfloat16, False, refuses = True) is torch.bfloat16
    assert (
        _resolve(name, "xpu", torch.bfloat16, True, refuses = True, vllm_available = False)
        is torch.bfloat16
    )
    assert _resolve(name, "cuda", torch.bfloat16, True, refuses = True) is torch.bfloat16
    assert _resolve(name, "xpu", torch.float16, True, refuses = True) is torch.float16
    assert _resolve(name, "xpu", torch.float32, True, refuses = True) is torch.float32


@pytest.mark.parametrize("name", ["llama.py", "vision.py"])
def test_a_forced_float32_model_refuses_to_load_in_fp16(name):
    # Gemma 3/4, Qwen3.5: float16 gives NaN gradients, so a refused bf16 fails early instead.
    with pytest.raises(RuntimeError, match = "fast_inference = False"):
        _resolve(name, "xpu", torch.bfloat16, True, refuses = True, env = {"UNSLOTH_FORCE_FLOAT32": "1"})
    assert (
        _resolve(name, "xpu", torch.bfloat16, True, refuses = False, env = {"UNSLOTH_FORCE_FLOAT32": "1"})
        is torch.bfloat16
    )


def test_helper_follows_the_zoo_check(monkeypatch):
    from unsloth import device_type as dt

    zoo = pytest.importorskip("unsloth_zoo.device_type")
    monkeypatch.setattr(dt, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(zoo, "_xpu_vllm_refuses_bf16", lambda: True, raising = False)
    assert dt.xpu_vllm_refuses_bf16() is True
    monkeypatch.setattr(zoo, "_xpu_vllm_refuses_bf16", lambda: False, raising = False)
    assert dt.xpu_vllm_refuses_bf16() is False

    def _boom():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(zoo, "_xpu_vllm_refuses_bf16", _boom, raising = False)
    assert dt.xpu_vllm_refuses_bf16() is False
    # An older unsloth_zoo without the check refuses nothing.
    monkeypatch.delattr(zoo, "_xpu_vllm_refuses_bf16", raising = False)
    assert dt.xpu_vllm_refuses_bf16() is False
    monkeypatch.setattr(zoo, "_xpu_vllm_refuses_bf16", lambda: True, raising = False)
    monkeypatch.setattr(dt, "DEVICE_TYPE", "cuda")
    assert dt.xpu_vllm_refuses_bf16() is False
