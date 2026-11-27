# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ROCm gates in these files go through is_rocm_torch (torch.version.hip, with the +rocm version tag as a fallback)."""

import ast
import importlib.util
import pathlib
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]

# A raw torch.version.hip read here is a regression. An inlined `_is_rocm_torch` helper may read it.
_GUARDED = (
    "unsloth/kernels/fp8.py",
    "unsloth/kernels/int4_packed.py",
    "unsloth/kernels/large_head_attention.py",
    "unsloth/kernels/multi_lora.py",
    "unsloth/kernels/nvfp4.py",
    "unsloth/models/_uma_safetensors.py",
    "unsloth/models/_windows_safetensors.py",
)
_HELPERS = {"is_rocm_torch", "_is_rocm_torch"}


def _raw_hip_reads(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in _HELPERS:
            allowed |= {id(n) for n in ast.walk(node)}
    lines = []
    for node in ast.walk(tree):
        if id(node) in allowed:
            continue
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "hip"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "version"
        ):
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "hip"
        ):
            lines.append(node.lineno)
    return sorted(lines)


def test_guarded_files_never_read_version_hip_directly():
    raw = {p: _raw_hip_reads(ROOT / p) for p in _GUARDED}
    assert not any(raw.values()), f"use unsloth.rocm_build.is_rocm_torch: {raw}"


def test_the_raw_read_check_can_fail(tmp_path):
    p = tmp_path / "x.py"
    p.write_text(
        "import torch\n"
        "a = torch.version.hip is None\n"
        "b = getattr(torch.version, 'hip', None)\n"
        "def _is_rocm_torch(torch):\n"
        "    return torch.version.hip\n",
        encoding = "utf-8",
    )
    assert _raw_hip_reads(p) == [2, 3]


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _torch(hip, version):
    return types.SimpleNamespace(version = types.SimpleNamespace(hip = hip), __version__ = version)


def test_is_rocm_torch_sees_sdk_wheels_without_version_hip():
    helpers = [
        _load("unsloth/rocm_build.py").is_rocm_torch,
        _load("unsloth/models/_uma_safetensors.py")._is_rocm_torch,
        _load("unsloth/models/_windows_safetensors.py")._is_rocm_torch,
    ]
    for is_rocm in helpers:
        assert is_rocm(_torch(None, "2.11.0+rocm7.14.0"))
        assert is_rocm(_torch("7.1.0", "2.9.0"))
        assert not is_rocm(_torch(None, "2.11.0+cu130"))
        assert not is_rocm(_torch(None, "2.11.0"))


def test_is_rocm_torch_traces_under_fullgraph():
    # Called from compiled code (fp8 gates, routers): torch.__version__ is a TorchVersion whose C str() Dynamo
    # cannot trace, so the version-tag fallback must read the plain-str torch.version.__version__.
    import torch
    helpers = [
        _load("unsloth/rocm_build.py").is_rocm_torch,
        _load("unsloth/models/_uma_safetensors.py")._is_rocm_torch,
        _load("unsloth/models/_windows_safetensors.py")._is_rocm_torch,
    ]
    for is_rocm in helpers:
        torch._dynamo.reset()

        def fn(x):
            return x + int(is_rocm(torch))

        x = torch.zeros(2)
        assert torch.equal(torch.compile(fn, backend = "eager", fullgraph = True)(x), fn(x))
