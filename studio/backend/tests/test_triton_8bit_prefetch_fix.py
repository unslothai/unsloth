# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""triton-lang/triton#12486 fix: the sm_8x prefetch pass is skipped only for 8-bit tl.dot modules."""

from __future__ import annotations

import ast
import sys
import threading
import types
from pathlib import Path

import pytest

import core.inference.diffusion_triton_prefetch as patches

_CALLS = []


class _Passes:
    pass


_passes = types.SimpleNamespace(ttgpuir = _Passes())


def _stock_add_prefetch(pm):
    _CALLS.append(("prefetch", pm))


_HOOK = {"fn": None}


def _stock_make_ttgir(mod, metadata, opt, capability):
    pm = object()
    if capability // 10 == 8:
        passes.ttgpuir.add_prefetch(pm)  # noqa: F821 - resolved from the fake module globals
    if _HOOK["fn"] is not None:
        _HOOK["fn"](mod)
    _CALLS.append(("ttgir", capability))
    return mod


class _Mod:
    def __init__(self, text):
        self.text = text

    def str_nodebug(self):
        return self.text


_I8 = "%5 = tt.dot %3, %4, %cst : tensor<128x64xi8> * tensor<64x256xi8> -> tensor<128x256xi32>"
_F8 = "%5 = tt.dot %3, %4, %cst, inputPrecision = tf32 : tensor<128x64xf8E4M3FN> * tensor<64x256xf8E4M3FN> -> tensor<128x256xf32>"
_F16 = "%5 = tt.dot %3, %4, %cst : tensor<128x64xf16> * tensor<64x256xf16> -> tensor<128x256xf32>"
_LOAD_I8 = "%3 = tt.load %2 : tensor<128x64x!tt.ptr<i8>>\n" + _F16


@pytest.fixture
def fake_triton(monkeypatch):
    _CALLS.clear()
    _HOOK["fn"] = None
    _passes.ttgpuir.add_prefetch = _stock_add_prefetch
    _stock_make_ttgir.__globals__["passes"] = _passes
    backend = type("CUDABackend", (), {"make_ttgir": staticmethod(_stock_make_ttgir)})
    triton = types.ModuleType("triton")
    triton.__version__ = "3.8.0"
    libtriton = types.ModuleType("triton._C.libtriton")
    libtriton.passes = _passes
    c = types.ModuleType("triton._C")
    c.libtriton = libtriton
    nvidia = types.ModuleType("triton.backends.nvidia")
    compiler = types.ModuleType("triton.backends.nvidia.compiler")
    compiler.CUDABackend = backend
    nvidia.compiler = compiler
    backends = types.ModuleType("triton.backends")
    backends.nvidia = nvidia
    for name, mod in {
        "triton": triton,
        "triton._C": c,
        "triton._C.libtriton": libtriton,
        "triton.backends": backends,
        "triton.backends.nvidia": nvidia,
        "triton.backends.nvidia.compiler": compiler,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delenv(patches._TRITON_PREFETCH_ENV, raising = False)
    yield types.SimpleNamespace(triton = triton, backend = backend)


def _prefetches():
    return sum(1 for kind, _ in _CALLS if kind == "prefetch")


def _compile(backend, text, capability):
    _CALLS.clear()
    backend.make_ttgir(_Mod(text), {}, None, capability)
    return _prefetches()


@pytest.mark.parametrize("text", [_I8, _F8])
def test_8bit_dot_on_sm8x_skips_the_prefetch(fake_triton, text):
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    for capability in (80, 86, 89):
        assert _compile(fake_triton.backend, text, capability) == 0


@pytest.mark.parametrize("text", [_F16, _LOAD_I8, ""])
def test_other_dots_keep_the_prefetch(fake_triton, text):
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    assert _compile(fake_triton.backend, text, 89) == 1


@pytest.mark.parametrize("capability", [75, 90, 100, 120])
def test_other_cards_are_untouched(fake_triton, capability):
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    assert _compile(fake_triton.backend, _I8, capability) == 0  # stock adds no prefetch there
    assert ("ttgir", capability) in _CALLS


def test_kill_switch(fake_triton, monkeypatch):
    monkeypatch.setenv(patches._TRITON_PREFETCH_ENV, "0")
    assert patches.fix_triton_sm8x_8bit_prefetch() is False
    assert fake_triton.backend.__dict__["make_ttgir"].__func__ is _stock_make_ttgir
    assert _compile(fake_triton.backend, _I8, 89) == 1


def test_kill_switch_after_install_restores_the_stock_pipeline(fake_triton, monkeypatch):
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    monkeypatch.setenv(patches._TRITON_PREFETCH_ENV, "0")
    assert _compile(fake_triton.backend, _I8, 89) == 1


@pytest.mark.parametrize("version", ["3.7.1", "3.10.0", "4.0.0", "garbage"])
def test_unchecked_triton_versions_are_left_alone(fake_triton, version):
    fake_triton.triton.__version__ = version
    assert patches.fix_triton_sm8x_8bit_prefetch() is False
    assert _compile(fake_triton.backend, _I8, 89) == 1


def test_force_overrides_the_version_list(fake_triton, monkeypatch):
    fake_triton.triton.__version__ = "3.10.0"
    monkeypatch.setenv(patches._TRITON_PREFETCH_ENV, "force")
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    assert _compile(fake_triton.backend, _I8, 89) == 0


def test_a_moved_prefetch_pass_is_left_alone(fake_triton):
    def make_ttgir(mod, metadata, opt, capability):
        return mod

    fake_triton.backend.make_ttgir = staticmethod(make_ttgir)
    assert patches.fix_triton_sm8x_8bit_prefetch() is False
    assert fake_triton.backend.__dict__["make_ttgir"].__func__ is make_ttgir


def test_install_is_idempotent(fake_triton):
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    first = fake_triton.backend.__dict__["make_ttgir"].__func__
    wrapper = _passes.ttgpuir.add_prefetch
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    assert fake_triton.backend.__dict__["make_ttgir"].__func__ is first
    assert _passes.ttgpuir.add_prefetch is wrapper
    assert _compile(fake_triton.backend, _I8, 89) == 0
    assert _compile(fake_triton.backend, _F16, 89) == 1


def test_flag_is_restored_after_a_failing_compile(fake_triton):
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    _HOOK["fn"] = (
        lambda mod: (_ for _ in ()).throw(RuntimeError("compile failed"))
        if "BOOM" in mod.text
        else None
    )
    with pytest.raises(RuntimeError):
        _compile(fake_triton.backend, _I8 + "\nBOOM", 89)
    assert _compile(fake_triton.backend, _F16, 89) == 1


def test_skip_does_not_leak_to_other_threads(fake_triton):
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    inside, released, counts = threading.Event(), threading.Event(), []

    def hook(mod):
        if "HOLD" in mod.text:
            inside.set()
            released.wait(5)

    _HOOK["fn"] = hook

    def other():
        inside.wait(5)
        calls = []
        _CALLS.clear()
        fake_triton.backend.make_ttgir(_Mod(_F16), {}, None, 89)
        calls.extend(_CALLS)
        counts.append(sum(1 for kind, _ in calls if kind == "prefetch"))
        released.set()

    t = threading.Thread(target = other)
    t.start()
    fake_triton.backend.make_ttgir(_Mod(_I8 + "\nHOLD"), {}, None, 89)
    t.join(5)
    assert counts == [1]


def test_no_triton_is_a_no_op(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    assert patches.fix_triton_sm8x_8bit_prefetch() is False


def test_int8_gemm_kernels_install_the_fix():
    source = (Path(patches.__file__).parent / "diffusion_int8_gemm.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    kernels = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_kernels")
    calls = [
        c.func.id
        for c in ast.walk(kernels)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    ]
    assert "fix_triton_sm8x_8bit_prefetch" in calls


def _real_triton():
    try:
        import triton
        from triton._C.libtriton import passes as real_passes
        from triton.backends.nvidia import compiler as nvidia_compiler
    except Exception:
        return None
    return triton, real_passes, nvidia_compiler


@pytest.fixture
def real_triton(monkeypatch):
    found = _real_triton()
    if found is None:
        pytest.skip("Triton with the NVIDIA backend is not installed")
    triton, real_passes, nvidia_compiler = found
    import re

    match = re.match(r"(\d+)\.(\d+)", triton.__version__)
    if (int(match.group(1)), int(match.group(2))) not in patches._TRITON_PREFETCH_VERSIONS:
        pytest.skip(f"Triton {triton.__version__} is outside the patched versions")
    backend = nvidia_compiler.CUDABackend
    saved_stage = backend.__dict__["make_ttgir"]
    saved_prefetch = real_passes.ttgpuir.add_prefetch
    # Another test (or the int8 GEMM) may have installed the fix already: start from stock Triton.
    stage = getattr(saved_stage, "__func__", saved_stage)
    if getattr(stage, patches._TRITON_PREFETCH_SENTINEL, False):
        backend.make_ttgir = staticmethod(stage.__wrapped__)
    if getattr(saved_prefetch, patches._TRITON_PREFETCH_SENTINEL, None) is not None:
        real_passes.ttgpuir.add_prefetch = saved_prefetch.__wrapped__
    monkeypatch.delenv(patches._TRITON_PREFETCH_ENV, raising = False)
    monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)
    yield triton
    backend.make_ttgir = saved_stage
    real_passes.ttgpuir.add_prefetch = saved_prefetch


def _repro_kernel(triton):
    import triton.language as tl

    @triton.jit
    def matmul(a_ptr, b_ptr, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm, rn, rk = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BK)
        a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
        b_ptrs = b_ptr + rk[:, None] + rn[None, :] * K
        acc = tl.zeros((BM, BN), c_ptr.dtype.element_ty)
        for _ in range(0, K, BK):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
            a_ptrs += BK
            b_ptrs += BK
        tl.store(c_ptr + rm[:, None] * BN + rn[None, :], acc)

    return matmul


def _ptx(triton, kernel, ab, c, arch):
    from triton.backends.compiler import GPUTarget

    src = triton.compiler.ASTSource(
        fn = kernel,
        signature = {
            "a_ptr": ab,
            "b_ptr": ab,
            "c_ptr": c,
            "K": "i32",
            "BM": "constexpr",
            "BN": "constexpr",
            "BK": "constexpr",
        },
        constexprs = {"BM": 128, "BN": 128, "BK": 64},
        attrs = {(i,): [["tt.divisibility", 16]] for i in range(4)},
    )
    target = GPUTarget("cuda", arch, 32)
    return triton.compile(src, target = target, options = {"num_warps": 4, "num_stages": 3}).asm["ptx"]


def test_real_triton_compile_only_changes_8bit_dots_on_sm8x(real_triton, monkeypatch):
    """CPU only: compiles for the targets without a GPU."""
    kernel = _repro_kernel(real_triton)
    stock = {
        (ab, arch): _ptx(real_triton, kernel, ab, c, arch)
        for ab, c in (("*i8", "*i32"), ("*fp16", "*fp32"))
        for arch in (89, 90)
    }
    assert stock[("*i8", 89)].count("prmt.b32") > 0  # the bug this fix is for
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    for (ab, arch), ptx in stock.items():
        c = "*i32" if ab == "*i8" else "*fp32"
        patched = _ptx(real_triton, kernel, ab, c, arch)
        if (ab, arch) == ("*i8", 89):
            assert patched.count("prmt.b32") == 0
        else:
            assert patched == ptx, f"{ab} on sm_{arch} changed"
    monkeypatch.setenv(patches._TRITON_PREFETCH_ENV, "0")
    assert _ptx(real_triton, kernel, "*i8", "*i32", 89) == stock[("*i8", 89)]
