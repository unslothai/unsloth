# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Compiled decoder blocks run Inductor's C++ wrapper where it is available (unsloth/models/_block_compile.py).

CPU: who gets it (UNSLOTH_BLOCK_CPP_WRAPPER, NVIDIA CUDA, Linux, torch floor, a C++ compiler), that the
graphs-off region is the C++ one while CUDA graph regions keep the Python wrapper, that auto mode times
launches on the Python wrapper first, and that a failed C++ build compiles the block with the Python
wrapper in the same mode instead of demoting it. GPU: one compiled block forward and backward match the
Python wrapper bit for bit.
"""

import importlib.util
import os
import sys

import pytest
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

_DIR = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    for key in (
        "UNSLOTH_BLOCK_COMPILE",
        "UNSLOTH_COMPILE_DISABLE",
        "UNSLOTH_FORCE_FLOAT32",
        "UNSLOTH_BLOCK_CPP_WRAPPER",
        "UNSLOTH_BLOCK_CPP_WRAPPER_AFTER",
        "CXX",
    ):
        monkeypatch.delenv(key, raising = False)
    # The toolchain probe installs a wrapper on torch's lazy C++ launch; restore it after each test.
    try:
        from torch._inductor.runtime import triton_lazy_compile as lazy
    except Exception:
        return
    monkeypatch.setattr(
        lazy, "run_triton_kernel_with_autotune", lazy.run_triton_kernel_with_autotune
    )


def _load():
    spec = importlib.util.spec_from_file_location(
        "_u_block_compile_cpp", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # The fake-compile tests must not depend on this torch having the lazy C++ launch (torch < 2.14 has
    # none); test_lazy_cpp_launch_reads_the_params_its_own_kernel_saved runs the real install.
    mod._real_install_cpp_wrapper_param_fix = mod._install_cpp_wrapper_param_fix
    mod._install_cpp_wrapper_param_fix = lambda: True
    # A long Trainer run in progress; test_auto_switch_waits_for_enough_steps_left covers the gate.
    mod._cpp_wrapper_steps_left = lambda: 10**6
    return mod


class BackendCompilerFailed(Exception):
    """Named like torch's: _backend_compile_failed matches the class name in the MRO."""


def _toy_class():
    class ToyDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
        ):
            return self.proj(hidden_states) + hidden_states

    return ToyDecoderLayer


def _fake_compile(calls, fail_cpp = False):
    def fake_compile(fn, **kwargs):
        cpp = bool((kwargs.get("options") or {}).get("cpp_wrapper"))

        # fail_cpp: the block's C++ build fails while the toolchain probe passes (an op it cannot wrap).
        probe = getattr(fn, "__name__", "") == "_cpp_wrapper_probe_fn"

        def compiled(*args, **kw):
            if not probe:  # the toolchain probe passes and is not a block call
                calls.append("cpp" if cpp else "python")
            if cpp and fail_cpp and not probe:
                raise BackendCompilerFailed("C++ compile error: g++ not found")
            return fn(*args, **kw)

        compiled.cpp = cpp
        return compiled

    return fake_compile


def test_reason_follows_the_switch_platform_torch_and_compiler(monkeypatch):
    M = _load()
    monkeypatch.setenv("UNSLOTH_BLOCK_CPP_WRAPPER", "0")
    assert M._cpp_wrapper_reason() == "UNSLOTH_BLOCK_CPP_WRAPPER=0"
    monkeypatch.setenv("UNSLOTH_BLOCK_CPP_WRAPPER", "1")
    assert M._cpp_wrapper_reason() is None  # forced on wherever Inductor has the option
    monkeypatch.delenv("UNSLOTH_BLOCK_CPP_WRAPPER")

    import shutil

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch, "__version__", "2.14.1+cu130")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA L4")
    assert M._cpp_wrapper_reason() is None
    # Auto mode only on the GPU class measured to gain; forcing it on still works anywhere.
    for name in (
        "NVIDIA L40S",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA B200",
        "Tesla T4",
    ):
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, n = name: n)
        assert "measured to gain" in M._cpp_wrapper_reason(), name
        monkeypatch.setenv("UNSLOTH_BLOCK_CPP_WRAPPER", "1")
        assert M._cpp_wrapper_reason() is None
        monkeypatch.delenv("UNSLOTH_BLOCK_CPP_WRAPPER")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA L4")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert M._cpp_wrapper_reason() == "no C++ compiler"
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(torch, "__version__", "2.13.1+cu130")
    assert M._cpp_wrapper_reason() == "torch < 2.14.0"
    monkeypatch.setattr(torch, "__version__", "2.14.0+cu128")
    assert M._cpp_wrapper_reason() is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert M._cpp_wrapper_reason() == "no CUDA device"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "platform", "win32")
    assert M._cpp_wrapper_reason() == "not Linux"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert M._cpp_wrapper_reason() == "not an NVIDIA CUDA build"
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", None)
    assert M._cpp_wrapper_reason() == "not an NVIDIA CUDA build"


def test_graphs_off_region_is_the_cpp_one(monkeypatch):
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
    M = _load()
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 0)
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls))
    cls = _toy_class()
    original = cls.forward
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    layer = cls()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), original(layer, x))
        layer(x)
    assert calls and set(calls) == {"cpp"}, calls
    assert M.CPP_WRAPPER_STATS == {"regions": 1, "fallbacks": 0}


def test_cpp_region_waits_for_a_long_run(monkeypatch, capsys):
    """Short runs never pay the C++ build: until the layers average the threshold in calls, the
    Python region runs."""
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
    M = _load()
    assert M._CPP_WRAPPER_AFTER == 2000
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 3)
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls))
    cls = _toy_class()
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    first, second = cls(), cls()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        # Each layer's first call also runs the grad-mode trial: 2 + 2 + 1 + 1 counted calls, 2 layers.
        first(x)
        second(x)
        first(x)
        second(x)
    assert "cpp" not in calls and M.CPP_WRAPPER_STATS["regions"] == 0, calls
    calls.clear()
    with torch.no_grad():
        first(x)
        second(x)
    assert calls and set(calls) == {"cpp"}, calls
    assert M.CPP_WRAPPER_STATS == {"regions": 1, "fallbacks": 0}
    assert "C++ wrapper (once, about a minute)" in capsys.readouterr().out
    # A deleted first model does not stall the count for the next one.
    calls.clear()
    M2 = _load()
    monkeypatch.setattr(M2, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M2, "_CPP_WRAPPER_AFTER", 1)
    cls2 = _toy_class()
    M2._wrap(cls2, dict(M2._COMPILE_OPTIONS))
    with torch.no_grad():
        layer = cls2()
        layer(x)
        del layer
        calls.clear()
        fresh = cls2()
        for _ in range(4):
            fresh(x)
    assert "cpp" in calls, calls


def test_auto_switch_waits_for_enough_steps_left(monkeypatch, capsys):
    """Past the call threshold, auto mode builds the C++ region only if the Trainer run has enough
    optimizer steps left to win the cold build back; unknown steps left keep the Python wrapper."""
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
    M = _load()
    assert M._CPP_WRAPPER_MIN_REMAINING == 700
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 1)
    left = [None]
    monkeypatch.setattr(M, "_cpp_wrapper_steps_left", lambda: left[0])
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls))
    cls = _toy_class()
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    layer = cls()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        for steps_left in (None, 699):
            left[0] = steps_left
            calls.clear()
            for _ in range(4):
                layer(x)
            assert set(calls) == {"python"} and M.CPP_WRAPPER_STATS["regions"] == 0, (
                steps_left,
                calls,
            )
        assert "keeping Inductor's Python wrapper" in capsys.readouterr().out
        left[0] = 700  # not latched: a later, longer run still switches
        calls.clear()
        for _ in range(2):
            layer(x)
    assert calls[-1] == "cpp" and M.CPP_WRAPPER_STATS["regions"] == 1, calls


def test_active_trainer_state_follows_trainer_train(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_u_gc_train_scope", os.path.join(_DIR, "_gc_train_scope.py")
    )
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    seen = []

    class State:
        max_steps, global_step = 1000, 250

    class Trainer:
        state = State()

        def train(self):
            seen.append(G.active_trainer_state())
            return "trained"

    assert G.patch_trainer_train_gc_scope(Trainer)
    assert G.active_trainer_state() is None
    assert Trainer().train() == "trained"
    assert seen == [Trainer.state] and G.active_trainer_state() is None


def test_forcing_it_on_builds_at_once(monkeypatch):
    monkeypatch.setenv("UNSLOTH_BLOCK_CPP_WRAPPER", "1")
    M = _load()
    assert M._CPP_WRAPPER_AFTER == 0


def test_wrapped_oom_is_not_a_failed_cpp_build(monkeypatch):
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
    M = _load()
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 0)

    def fake_compile(fn, **kwargs):
        cpp = bool((kwargs.get("options") or {}).get("cpp_wrapper"))

        def compiled(*args, **kw):
            if cpp:
                e = BackendCompilerFailed("compile ran out of memory")
                e.inner_exception = torch.OutOfMemoryError("CUDA out of memory")
                raise e
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    with torch.no_grad():
        cls()(torch.randn(2, 4, 8))
    assert M.CPP_WRAPPER_STATS["fallbacks"] == 0


def _fake_compile_backward_fails(calls):
    """C++ regions run their forward, then fail like a lazily lowered backward whose C++ build breaks."""

    class _FailingBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            raise BackendCompilerFailed("C++ compile error in the backward graph")

    def fake_compile(fn, **kwargs):
        cpp = bool((kwargs.get("options") or {}).get("cpp_wrapper"))

        def compiled(*args, **kw):
            calls.append("cpp" if cpp else "python")
            out = fn(*args, **kw)
            return _FailingBackward.apply(out) if cpp and torch.is_grad_enabled() else out

        return compiled

    return fake_compile


def test_backward_build_failure_never_reaches_training(monkeypatch, capsys):
    """Flip test: the C++ backward fails to build (AOTAutograd builds it lazily, at loss.backward()).
    The switch-time probe catches it, every block keeps the Python wrapper, and training steps go on
    with the same gradients as the Python wrapper alone."""
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
    M = _load()
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 2)
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile_backward_fails(calls))
    cls = _toy_class()
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    layer = cls()
    for step in range(6):
        x = torch.randn(2, 4, 8, generator = torch.Generator().manual_seed(step), requires_grad = True)
        layer.zero_grad()
        layer(x).square().sum().backward()  # must not raise, before or after the switch
        assert x.grad is not None and layer.proj.weight.grad is not None
    # The switch-time probe (a small forward + backward built with the C++ wrapper) caught it: no block
    # region was built, every block keeps the Python wrapper.
    assert M.CPP_WRAPPER_STATS == {"regions": 0, "fallbacks": 1}
    assert M._CPP_WRAPPER_BROKEN
    assert "failed a forward + backward probe" in capsys.readouterr().out
    # The trial touched no parameter gradient: one more step equals the plain forward's.
    layer.zero_grad()
    x = torch.randn(2, 4, 8)
    layer(x).square().sum().backward()
    want = torch.autograd.grad(
        cls.forward._unsloth_block_original(layer, x).square().sum(), layer.proj.weight
    )[0]
    torch.testing.assert_close(layer.proj.weight.grad, want)
    # Later calls (and a new class) stay on the Python wrapper.
    calls.clear()
    other = _toy_class()
    M._wrap(other, dict(M._COMPILE_OPTIONS))
    for _ in range(4):
        other()(torch.randn(2, 4, 8, requires_grad = True)).sum().backward()
        layer(torch.randn(2, 4, 8, requires_grad = True)).sum().backward()
    assert "cpp" not in calls and "probe" not in calls, calls


def test_lazy_cpp_launch_reads_the_params_its_own_kernel_saved(monkeypatch):
    """Kernel names repeat across graphs and CudaKernelParamCache is keyed by name: a kernel object
    that already ran (shared through Inductor's source cache) took run()'s fast path, saved nothing,
    and the lazy C++ launch read another graph's same-named kernel (invalid argument, segfault, NaN)."""
    lazy = pytest.importorskip("torch._inductor.runtime.triton_lazy_compile")
    from torch._inductor.codecache import CudaKernelParamCache

    M = _load()
    seen = []

    class Launcher:
        def __init__(self, store_cubin):
            self.store_cubin = store_cubin

    class Kernel:
        def __init__(self, store_cubin):
            self.launchers = [Launcher(store_cubin)]
            self._cached_launcher = object()  # it already ran in another C++ module

    def original(pending, name, stream, args):
        kernel = pending[name]
        seen.append((name, kernel._cached_launcher, name in CudaKernelParamCache.cache))
        return "launched"

    monkeypatch.setattr(lazy, "run_triton_kernel_with_autotune", original)
    monkeypatch.setattr(
        CudaKernelParamCache, "cache", {"k_0": {"from": "another graph"}, "k_1": {"x": 1}}
    )
    assert M._real_install_cpp_wrapper_param_fix() is True
    assert M._real_install_cpp_wrapper_param_fix() is True  # idempotent
    patched = lazy.run_triton_kernel_with_autotune
    assert patched is not original and patched._unsloth_saves_params
    # The stale same-named entry is gone and run() cannot take the fast path, so it saves its own.
    assert patched({"k_0": Kernel(True)}, "k_0", 0, []) == "launched"
    assert seen[-1] == ("k_0", None, False)
    # A kernel whose launchers would not save keeps torch's behaviour (nothing to read otherwise).
    keep = Kernel(False)
    fast = keep._cached_launcher
    patched({"k_1": keep}, "k_1", 0, [])
    assert seen[-1] == ("k_1", fast, True)


def test_probe_refuses_a_torch_without_the_lazy_launch(monkeypatch):
    M = _load()
    monkeypatch.setattr(M, "_install_cpp_wrapper_param_fix", lambda: False)
    why = M._cpp_wrapper_probe(torch.device("cpu"))
    assert why and "lazy" in why
    assert M._CPP_WRAPPER_BROKEN and M.CPP_WRAPPER_STATS["fallbacks"] == 1


def test_kill_switch_keeps_the_python_wrapper(monkeypatch):
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
    monkeypatch.setenv("UNSLOTH_BLOCK_CPP_WRAPPER", "0")
    M = _load()
    assert M._CPP_WRAPPER_REASON == "UNSLOTH_BLOCK_CPP_WRAPPER=0"
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls))
    cls = _toy_class()
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    with torch.no_grad():
        cls()(torch.randn(2, 4, 8))
    assert calls and set(calls) == {"python"}, calls
    assert M.CPP_WRAPPER_STATS == {"regions": 0, "fallbacks": 0}


def test_cuda_graph_regions_keep_the_python_wrapper(monkeypatch):
    M = _load()
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 0)
    seen = []

    def fake_compile(fn, **kwargs):
        seen.append(dict(kwargs.get("options") or {}))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    M._wrap(_toy_class(), {**M._COMPILE_OPTIONS, "triton.cudagraphs": True})
    assert seen and not any(o.get("cpp_wrapper") for o in seen), seen


def test_auto_mode_times_launches_on_the_python_wrapper(monkeypatch):
    """Auto mode's launch / GPU ratio thresholds were measured on the Python wrapper: while it is
    undecided the block runs that region; packed rows (never graphed) take the C++ one at once."""
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "auto")
    M = _load()
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 0)
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls))
    cls = _toy_class()
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    layer = cls()
    x = torch.randn(2, 4, 8)
    M._CUDAGRAPH_STATE["auto"]["phase"] = "probe"
    monkeypatch.setattr(M, "_auto_observe", lambda *a, **k: "plain")
    with torch.no_grad():
        layer(x)
    assert set(calls) == {"python"}, calls
    calls.clear()
    monkeypatch.setattr(M, "_packed_kwargs", lambda kwargs: True)
    with torch.no_grad():
        layer(x)
    assert calls and set(calls) == {"cpp"}, calls
    calls.clear()
    monkeypatch.setattr(M, "_packed_kwargs", lambda kwargs: False)
    M._CUDAGRAPH_STATE["auto"]["phase"] = "off"
    with torch.no_grad():
        layer(x)
    assert calls and set(calls) == {"cpp"}, calls


def test_failed_cpp_build_compiles_with_the_python_wrapper(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
    M = _load()
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 0)
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls, fail_cpp = True))
    cls = _toy_class()
    original = cls.forward
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    layer = cls()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), original(layer, x))
        layer(x)
    # The block keeps its compiled mode: only the wrapper changed.
    assert layer._unsloth_block_mode == M._FULL
    assert calls[0] == "cpp" and set(calls[1:]) == {"python"}, calls
    assert M.CPP_WRAPPER_STATS == {"regions": 1, "fallbacks": 1}
    assert "C++ wrapper failed (C++ compile error: g++ not found)" in capsys.readouterr().out
    # A sibling of the same class never tries the C++ build again.
    calls.clear()
    with torch.no_grad():
        cls()(x)
    assert calls and set(calls) == {"python"}, calls


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_cpp_wrapper_block_matches_the_python_wrapper_on_cuda(monkeypatch):
    """A real Inductor block (norm, projections, SiLU gate, residual) with dynamic sequence length:
    forward and input / weight gradients from the C++ wrapper match the Python wrapper's."""
    if torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA only")
    M = _load()
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) < Version(M._CPP_WRAPPER_MIN_TORCH):
        pytest.skip("below the auto mode torch floor")
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
    M = _load()
    monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None)
    monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 0)

    class MlpDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.RMSNorm(64)
            self.up = torch.nn.Linear(64, 128, bias = False)
            self.gate = torch.nn.Linear(64, 128, bias = False)
            self.down = torch.nn.Linear(128, 64, bias = False)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
        ):
            h = self.norm(hidden_states)
            return hidden_states + self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))

    torch.manual_seed(0)
    reference = MlpDecoderLayer().cuda()
    plain = {k: v.clone() for k, v in reference.state_dict().items()}
    outs = {}
    for arm in ("python", "cpp"):
        torch._dynamo.reset()
        monkeypatch.setattr(M, "_CPP_WRAPPER_REASON", None if arm == "cpp" else "off")
        monkeypatch.setattr(M, "_CPP_WRAPPER_AFTER", 0)
        cls = type(f"MlpDecoderLayer{arm.title()}", (MlpDecoderLayer,), {})
        M._wrap(cls, dict(M._COMPILE_OPTIONS))
        layer = cls().cuda()
        layer.load_state_dict(plain)
        got = []
        for length in (5, 9, 13):
            g = torch.Generator(device = "cuda").manual_seed(length)
            x = torch.randn(2, length, 64, generator = g, device = "cuda", requires_grad = True)
            y = layer(x)
            y.square().sum().backward()
            got.append((y.detach(), x.grad, layer.up.weight.grad.clone()))
            layer.zero_grad()
        outs[arm] = got
    assert M.CPP_WRAPPER_STATS["regions"] == 1 and M.CPP_WRAPPER_STATS["fallbacks"] == 0
    # Two compiles may tune reductions apart, so equal up to fp32 rounding rather than bit for bit.
    for a, b in zip(outs["python"], outs["cpp"]):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol = 1e-5, rtol = 1e-5)


_SAME_NAME_KERNELS = r"""
import importlib.util, os, sys
import torch
spec = importlib.util.spec_from_file_location("_bc", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["_bc"] = m; spec.loader.exec_module(m)
assert m._install_cpp_wrapper_param_fix()
opts = {"cpp_wrapper": True}
def fa(a, b): return a * b + 1          # triton_poi_fused_add_mul_0, 3 pointers
def fc(a, b, c): return a * b + c       # a different kernel under the same name, 4 pointers
def fb(a, b, z): return a * b + 1, z.cos()  # fa's kernel again, byte-identical
cfa, cfc, cfb = (torch.compile(f, options = opts) for f in (fa, fc, fb))
g = torch.Generator(device = "cuda").manual_seed(0)
a, b, c = (torch.randn(1024, device = "cuda", generator = g) for _ in range(3))
z = torch.randn(2048, device = "cuda", generator = g)
close = lambda x, y: torch.allclose(x, y, atol = 1e-5, rtol = 1e-5)
assert close(cfa(a, b), fa(a, b)) and close(cfc(a, b, c), fc(a, b, c))
for _ in range(3):
    (o1, o2), (r1, r2) = cfb(a, b, z), fb(a, b, z)
    assert close(o1, r1) and close(o2, r2)
torch.cuda.synchronize()
print("SAME_NAME_OK")
"""


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.parametrize("warm", [False, True])
def test_same_named_kernels_of_two_graphs_launch_their_own_cubin(tmp_path, warm):
    """pytorch#199612: with one compile thread (or a warm cache) the second call of fb launched fc's
    same-named kernel and segfaulted. Run in a subprocess: the unfixed failure kills the process."""
    import subprocess

    if torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA only")
    pytest.importorskip("torch._inductor.runtime.triton_lazy_compile")
    script = tmp_path / "same_name.py"
    script.write_text(_SAME_NAME_KERNELS)
    env = {
        **os.environ,
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
    }
    for _ in range(2 if warm else 1):  # warm: the second process loads the first one's cache
        run = subprocess.run(
            [sys.executable, str(script), os.path.join(_DIR, "_block_compile.py")],
            env = env,
            capture_output = True,
            text = True,
            timeout = 600,
        )
        assert run.returncode == 0 and "SAME_NAME_OK" in run.stdout, run.stderr[-2000:]
