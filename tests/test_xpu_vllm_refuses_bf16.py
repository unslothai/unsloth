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


def test_a_stale_forced_float32_env_does_not_fail_a_language_model_load():
    # FastLanguageModel never sets UNSLOTH_FORCE_FLOAT32; an earlier FastModel load (Gemma 3) can
    # leave it at "1" in the process, which must not refuse a later Llama-path load.
    assert (
        _resolve(
            "llama.py",
            "xpu",
            torch.bfloat16,
            True,
            refuses = True,
            env = {"UNSLOTH_FORCE_FLOAT32": "1"},
        )
        is torch.float16
    )


@pytest.mark.parametrize("name", ["vision.py"])
def test_a_forced_float32_model_refuses_to_load_in_fp16(name):
    # Gemma 3/4, Qwen3.5: float16 gives NaN gradients, so a refused bf16 fails early instead.
    with pytest.raises(RuntimeError, match = "fast_inference = False"):
        _resolve(
            name, "xpu", torch.bfloat16, True, refuses = True, env = {"UNSLOTH_FORCE_FLOAT32": "1"}
        )
    assert (
        _resolve(
            name, "xpu", torch.bfloat16, True, refuses = False, env = {"UNSLOTH_FORCE_FLOAT32": "1"}
        )
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


def _loader_family_gate():
    """FastModel.from_pretrained's FORCE_FLOAT32 loop through the vLLM gate after it."""
    tree = ast.parse((MODELS / "loader.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.For) and ast.unparse(stmt.iter) == "FORCE_FLOAT32":
                for j in range(i + 1, len(body)):
                    if isinstance(body[j], ast.If) and "xpu_vllm_refuses_bf16()" in ast.unparse(
                        body[j].test
                    ):
                        module = ast.Module(body = body[i : j + 1], type_ignores = [])
                        return compile(module, "loader.py", "exec")
    raise AssertionError("FORCE_FLOAT32 family gate not found in loader.py")


def _load_family(
    arch,
    dtype,
    supports_bf16,
    refuses,
    device_type = "xpu",
    fast_inference = True,
):
    env = {}
    namespace = {
        "torch": torch,
        "os": types.SimpleNamespace(environ = env),
        "dtype": dtype,
        "model_type_arch": arch,
        "model_types_all": arch + ",",
        "FORCE_FLOAT32": ["gemma3,", "qwen3_5"],
        "SUPPORTS_BFLOAT16": supports_bf16,
        "do_forced_float32": False,
        "fast_inference": fast_inference,
        "DEVICE_TYPE": device_type,
        "is_vLLM_available": lambda: True,
        "xpu_vllm_refuses_bf16": lambda: refuses,
    }
    exec(_loader_family_gate(), namespace)
    return namespace["dtype"], env.get("UNSLOTH_FORCE_FLOAT32")


@pytest.mark.parametrize(
    "dtype,supports_bf16",
    [(torch.bfloat16, False), (torch.bfloat16, True), (None, True)],
    ids = ["explicit-bf16-emulated", "explicit-bf16-native", "auto-native"],
)
def test_a_fp16_unsafe_family_refuses_vllm_fp16_when_bf16_is_not_forced(dtype, supports_bf16):
    # These loads skip forced float32, so the FastBaseModel gate alone would drop Gemma 3 to fp16.
    with pytest.raises(RuntimeError, match = "fast_inference = False"):
        _load_family("gemma3", dtype, supports_bf16, refuses = True)
    assert _load_family("gemma3", dtype, supports_bf16, refuses = False)[0] is dtype
    assert _load_family("gemma3", dtype, supports_bf16, True, fast_inference = False)[0] is dtype
    assert _load_family("gemma3", dtype, supports_bf16, True, device_type = "cuda")[0] is dtype
    # Families that run in fp16 still switch quietly in FastBaseModel.
    assert _load_family("llama", dtype, supports_bf16, refuses = True)[0] is dtype


def test_forced_float32_family_is_left_to_the_fast_base_model_gate():
    assert _load_family("gemma3", torch.float16, True, refuses = True) == (torch.bfloat16, "1")
    assert _load_family("qwen3_5", None, False, refuses = True) == (torch.bfloat16, "1")
