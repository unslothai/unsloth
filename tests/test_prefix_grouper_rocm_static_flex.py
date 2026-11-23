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


def _load_real_compile(monkeypatch):
    # Real Dynamo; flex swapped for a differentiable math stand-in so the graphs run on CPU.
    pytest.importorskip("unsloth_zoo.rocm_flex_static")
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_FLEX_STATIC", raising = False)
    name = "_pg_rocm_static_kernel_real_compile"
    spec = importlib.util.spec_from_file_location(name, KERNEL)
    kernel = importlib.util.module_from_spec(spec)
    # Kept registered for the test: Dynamo looks the module up by name while tracing (torch 2.11).
    monkeypatch.setitem(sys.modules, name, kernel)
    spec.loader.exec_module(kernel)
    graphs = []

    def record(gm, inputs):
        # Per graph: the sequence size (dim 2) of each 4-D tensor input.
        graphs.append(
            [
                n.meta["example_value"].shape[2]
                for n in gm.graph.nodes
                if n.op == "placeholder"
                and isinstance(n.meta.get("example_value"), torch.Tensor)
                and n.meta["example_value"].dim() == 4
            ]
        )
        return gm.forward

    def math_flex(query, key, value, **kwargs):
        scores = query @ key.transpose(-2, -1)
        return scores.softmax(-1) @ value

    kernel.flex_attention = math_flex
    kernel._flex_attention_static = torch.compile(
        kernel._rocm_static_flex_entry, backend = record, dynamic = False
    )
    # CPU tensors stand in for ROCm GPU tensors.
    monkeypatch.setattr(
        kernel,
        "_uses_static_flex",
        lambda q, k, v: torch.is_grad_enabled() and q.requires_grad,
    )
    torch._dynamo.reset()
    return kernel, graphs, record


def test_rocm_static_length_inside_an_enclosing_graph(monkeypatch):
    # The dynamic = False compile is inlined into an enclosing graph, so without the trace-time
    # specialization one symbolic graph served every padded length.
    kernel, graphs, record = _load_real_compile(monkeypatch)

    def call(t):
        return kernel._flex_attention_compiled(t, t, t)

    outer = torch.compile(call, fullgraph = True, dynamic = True, backend = record)
    try:
        for n in (128, 256, 384):
            t = torch.randn(1, 2, n, 8, requires_grad = True)
            outer(t).sum().backward()
            assert t.grad is not None and torch.isfinite(t.grad).all()
    finally:
        torch._dynamo.reset()
    # One graph per length, each with its length concrete (a symbol prints as s<N>).
    assert [str(g[0]) for g in graphs] == ["128", "256", "384"], graphs


def test_rocm_static_entry_ignores_a_callers_dynamic_marks(monkeypatch):
    # mark_dynamic on an input wins over dynamic = False: the static entry got one symbolic graph.
    kernel, graphs, _ = _load_real_compile(monkeypatch)
    try:
        for n in (128, 256, 384):
            t = torch.randn(1, 2, n, 8, requires_grad = True)
            torch._dynamo.mark_dynamic(t, 2)
            kernel._flex_attention_compiled(t, t, t).sum().backward()
    finally:
        torch._dynamo.reset()
    assert [str(g[0]) for g in graphs] == ["128", "256", "384"], graphs
