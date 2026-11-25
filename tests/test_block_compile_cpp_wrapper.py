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
    ):
        monkeypatch.delenv(key, raising = False)


def _load():
    spec = importlib.util.spec_from_file_location(
        "_u_block_compile_cpp", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
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

        def compiled(*args, **kw):
            calls.append("cpp" if cpp else "python")
            if cpp and fail_cpp:
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

    import torch._inductor.cpp_builder as cpp_builder

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch, "__version__", "2.14.1+cu130")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_builder, "get_cpp_compiler", lambda: "g++")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    assert M._cpp_wrapper_reason() is None
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert M._cpp_wrapper_reason() == "no C++ compiler"
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(torch, "__version__", "2.9.1+cu128")
    assert M._cpp_wrapper_reason() == "torch < 2.10.0"
    monkeypatch.setattr(torch, "__version__", "2.10.0+cu128")
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
        fresh = cls2()
        for _ in range(4):
            fresh(x)
    assert "cpp" in calls, calls


def test_forcing_it_on_builds_at_once(monkeypatch):
    monkeypatch.setenv("UNSLOTH_BLOCK_CPP_WRAPPER", "1")
    M = _load()
    assert M._CPP_WRAPPER_AFTER == 0


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
