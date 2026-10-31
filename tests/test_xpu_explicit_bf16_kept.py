# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An explicit bf16 request on XPU stays bf16 (emulated if need be), as before the native-bf16 probe."""

import ast
import os
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"
FILES = ["loader.py", "llama.py", "vision.py", "diffusion.py", "sentence_transformer.py"]


def _override_branch(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        test = ast.unparse(node.test) if isinstance(node, ast.If) else ""
        if test.startswith("dtype == torch.bfloat16 and") and "SUPPORTS_BFLOAT16" in test:
            return compile(ast.Module(body = [node], type_ignores = []), str(path), "exec")
    raise AssertionError(f"bf16 override not found in {path.name}")


def _resolve(
    path,
    device_type,
    dtype,
    supports,
    zoo_keeps = True,
):
    namespace = {
        "xpu_keeps_explicit_bf16": lambda: device_type == "xpu" and zoo_keeps,
        "torch": torch,
        "os": types.SimpleNamespace(environ = {}),
        "dtype": dtype,
        "SUPPORTS_BFLOAT16": supports,
        "DEVICE_TYPE": device_type,
        "logger": types.SimpleNamespace(warning_once = lambda *a, **k: None),
        "print": lambda *a, **k: None,
    }
    exec(_override_branch(path), namespace)
    return namespace["dtype"]


@pytest.mark.parametrize("name", FILES)
def test_explicit_bf16_kept_on_xpu_only(name):
    path = MODELS / name
    assert _resolve(path, "xpu", torch.bfloat16, False) is torch.bfloat16
    assert _resolve(path, "cuda", torch.bfloat16, False) is torch.float16
    assert _resolve(path, "hip", torch.bfloat16, False) is torch.float16
    assert _resolve(path, "xpu", torch.bfloat16, True) is torch.bfloat16
    # Beside a zoo whose vLLM path still downgrades, the load downgrades too, so both agree.
    assert _resolve(path, "xpu", torch.bfloat16, False, zoo_keeps = False) is torch.float16


def test_explicit_bf16_on_xpu_follows_the_paired_zoo(monkeypatch):
    from unsloth import device_type as dt

    zoo = pytest.importorskip("unsloth_zoo.device_type")
    monkeypatch.setattr(dt, "DEVICE_TYPE", "xpu")
    monkeypatch.delattr(zoo, "XPU_KEEPS_EXPLICIT_BF16", raising = False)
    assert dt.xpu_keeps_explicit_bf16() is False
    monkeypatch.setattr(zoo, "XPU_KEEPS_EXPLICIT_BF16", True, raising = False)
    assert dt.xpu_keeps_explicit_bf16() is True
    monkeypatch.setattr(dt, "DEVICE_TYPE", "cuda")
    assert dt.xpu_keeps_explicit_bf16() is False


def _forced_float32_dtype_gate():
    tree = ast.parse((MODELS / "loader.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "FORCE_FLOAT32"
        ):
            test = node.body[0].test
            return compile(ast.Expression(test.values[1]), "gate", "eval")
    raise AssertionError("FORCE_FLOAT32 gate not found")


@pytest.mark.parametrize(
    "dtype, supports, forced",
    [
        (torch.float16, True, True),
        (torch.float16, False, True),
        (torch.bfloat16, True, False),
        (torch.bfloat16, False, False),  # explicit bf16 kept on XPU: no fp16 + float32 path
        (torch.float32, False, True),
        (torch.float32, True, False),
    ],
)
def test_forced_float32_only_for_a_float16_load(dtype, supports, forced):
    gate = _forced_float32_dtype_gate()
    assert (
        bool(eval(gate, {"torch": torch, "dtype": dtype, "SUPPORTS_BFLOAT16": supports})) is forced
    )
