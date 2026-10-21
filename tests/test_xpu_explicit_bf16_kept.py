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


def _resolve(path, device_type, dtype, supports):
    namespace = {
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
