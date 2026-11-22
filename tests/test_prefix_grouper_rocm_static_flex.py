# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PrefixGrouper on ROCm: flex calls whose backward runs use a static compile.

flex_attention compiled with a symbolic sequence length returned inf / NaN gradients on gfx1151
(torch 2.11, ROCm 7.13 / 7.14), so on a ROCm build a gradient-bearing call on the GPU compiles per
padded length. No-grad calls, CPU tensors and CUDA builds keep the dynamic=True compile. Runs on
CPU: the kernel module is loaded with torch.version.hip simulated and torch.compile recorded."""

import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn.attention.flex_attention")

KERNEL = (
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "prefix_grouper_kernel.py"
)


def _load(
    monkeypatch,
    hip,
    env = None,
):
    compiles = []

    def recording_compile(fn, **kwargs):
        compiles.append((fn, kwargs))

        def compiled(query, key, value, **kw):
            return (fn.__name__, kwargs.get("dynamic"))

        return compiled

    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch, "compile", recording_compile)
    if env is None:
        monkeypatch.delenv("UNSLOTH_ROCM_FLEX_STATIC", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_ROCM_FLEX_STATIC", env)
    name = "_pg_rocm_static_kernel_under_test"
    spec = importlib.util.spec_from_file_location(name, KERNEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module, compiles


def _operand(device = "cuda", grad = True):
    # Stands in for a GPU tensor: the dispatcher reads only .device and .requires_grad.
    return types.SimpleNamespace(device = torch.device(device), requires_grad = grad)


def test_rocm_gradient_calls_compile_static(monkeypatch):
    kernel, compiles = _load(monkeypatch, "7.13.0")
    q = _operand()
    _, dynamic = kernel._flex_attention_compiled(q, q, q)
    assert dynamic is False
    from torch.nn.attention.flex_attention import flex_attention

    static_entries = [fn for fn, kwargs in compiles if kwargs.get("dynamic") is False]
    assert len(static_entries) == 1
    assert static_entries[0].__code__ is not flex_attention.__code__, "a code object of its own"


def test_rocm_no_grad_and_cpu_calls_keep_the_dynamic_compile(monkeypatch):
    kernel, _ = _load(monkeypatch, "7.13.0")
    q = _operand(grad = False)
    assert kernel._flex_attention_compiled(q, q, q)[1] is True
    with torch.no_grad():
        q = _operand()
        assert kernel._flex_attention_compiled(q, q, q)[1] is True
    q = _operand(device = "cpu")
    assert kernel._flex_attention_compiled(q, q, q)[1] is True


@pytest.mark.parametrize("hip, env", [(None, None), ("", None), ("7.13.0", "0")])
def test_cuda_builds_and_the_opt_out_keep_the_dynamic_compile(monkeypatch, hip, env):
    kernel, compiles = _load(monkeypatch, hip, env)
    q = _operand()
    assert kernel._flex_attention_compiled(q, q, q)[1] is True
    assert all(kwargs.get("dynamic") is not False for _, kwargs in compiles)
