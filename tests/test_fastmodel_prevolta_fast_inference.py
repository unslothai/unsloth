# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel(fast_inference = True) on compute capability < 7 falls back to Unsloth inference, as
FastLanguageModel does, instead of reaching zoo's "Your GPU is too old!". Sliced with `ast`:
importing the loader needs a GPU.
"""

import ast
import types
from pathlib import Path

import pytest

LOADER_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "loader.py"


def _fast_inference_blocks():
    """FastModel.from_pretrained's pre-Volta gate and the `if fast_inference:` block after it (GB10)."""
    tree = ast.parse(LOADER_PATH.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    for parent in ast.walk(fn):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for i, node in enumerate(body):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "fast_inference"
                and "GB10" in ast.unparse(node)
            ):
                gate = body[i - 1]
                assert isinstance(gate, ast.If) and "get_device_capability" in ast.unparse(gate.test)
                return [gate, node]
    raise AssertionError("FastModel.from_pretrained has no pre-Volta gate before the GB10 block")


def _run(device_type, capability, name = "Tesla P100-PCIE-16GB", has_vllm = True):
    device_type_torch = {"cuda": "cuda", "hip": "cuda", "xpu": "xpu"}[device_type]
    cuda = types.SimpleNamespace(
        get_device_name = lambda i = 0: name,
        get_device_capability = lambda *a: capability,
    )
    ns = {
        "fast_inference": True,
        "importlib": types.SimpleNamespace(util = types.SimpleNamespace(find_spec = lambda m: object() if has_vllm else None)),
        "_vllm_unavailable_error": lambda: ImportError("no vllm"),
        "DEVICE_TYPE": device_type,
        "DEVICE_TYPE_TORCH": device_type_torch,
        "DEVICE_COUNT": 1,
        "torch": types.SimpleNamespace(cuda = cuda),
    }
    exec(compile(ast.Module(_fast_inference_blocks(), []), str(LOADER_PATH), "exec"), ns)
    return ns["fast_inference"]


@pytest.mark.parametrize("capability", [(6, 1), (6, 0), (5, 2)])
def test_pre_volta_cuda_falls_back(capability, capsys):
    assert _run("cuda", capability) is False
    assert "vLLM does not work on older GPUs" in capsys.readouterr().out


@pytest.mark.parametrize("device_type,capability", [
    ("cuda", (7, 0)),
    ("cuda", (7, 5)),
    ("cuda", (9, 0)),
    ("hip", (9, 4)),
    ("hip", (6, 0)),  # gfx arch major, not a CUDA capability: left to vLLM
    ("xpu", (0, 0)),
])
def test_other_devices_keep_fast_inference(device_type, capability, capsys):
    assert _run(device_type, capability) is True
    assert "older GPUs" not in capsys.readouterr().out


def test_gb10_still_falls_back():
    assert _run("cuda", (12, 1), name = "NVIDIA GB10") is False


def test_pre_volta_falls_back_without_requiring_vllm(capsys):
    assert _run("cuda", (6, 1), has_vllm = False) is False
    assert "vLLM does not work on older GPUs" in capsys.readouterr().out


def test_volta_still_requires_vllm():
    with pytest.raises(ImportError):
        _run("cuda", (7, 0), has_vllm = False)
