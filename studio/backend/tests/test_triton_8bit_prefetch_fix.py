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


@pytest.mark.parametrize("version", ["3.7.1", "2.3.0", "garbage"])
def test_triton_before_the_regression_is_left_alone(fake_triton, version):
    fake_triton.triton.__version__ = version
    assert patches.fix_triton_sm8x_8bit_prefetch() is False
    assert _compile(fake_triton.backend, _I8, 89) == 1


@pytest.mark.parametrize("version", ["3.8.0", "3.9.0", "3.10.0", "4.0.0+git12ab"])
def test_every_later_triton_with_the_pass_is_fixed(fake_triton, version):
    """Default-on from 3.8 for as long as the installed pipeline still adds the pass for sm_8x."""
    fake_triton.triton.__version__ = version
    assert patches.fix_triton_sm8x_8bit_prefetch() is True
    assert _compile(fake_triton.backend, _I8, 89) == 0


def test_force_overrides_the_version_floor(fake_triton, monkeypatch):
    fake_triton.triton.__version__ = "3.7.1"
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


def test_concurrent_first_installs_share_one_skip_flag(fake_triton, monkeypatch):
    """Two threads installing at once must not leave make_ttgir and add_prefetch on different flags."""
    both_read, second_done = threading.Barrier(2, timeout = 0.5), threading.Event()
    stores = []

    class _RacyGPUIR:
        def __init__(self, fn):
            self.__dict__["_fn"] = fn

        def __getattr__(self, name):
            if name != "add_prefetch":
                raise AttributeError(name)
            fn = self.__dict__["_fn"]
            if threading.current_thread() is not threading.main_thread() and not stores:
                try:
                    both_read.wait()  # without the lock both threads read the stock pass
                except threading.BrokenBarrierError:
                    pass
            return fn

        def __setattr__(self, name, value):
            self.__dict__["_fn"] = value
            stores.append(threading.current_thread().name)
            if len(stores) == 1:
                second_done.wait(0.5)  # the first writer finishes last when nothing serialises them

    monkeypatch.setattr(_passes, "ttgpuir", _RacyGPUIR(_stock_add_prefetch))
    results = []

    def install(name):
        results.append(patches.fix_triton_sm8x_8bit_prefetch())
        if stores and stores[0] != name:
            second_done.set()

    threads = [threading.Thread(target = install, args = (n,), name = n) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert results == [True, True]
    assert _compile(fake_triton.backend, _I8, 89) == 0
    assert _compile(fake_triton.backend, _F16, 89) == 1


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
    if (int(match.group(1)), int(match.group(2))) < patches._TRITON_PREFETCH_FIRST:
        pytest.skip(f"Triton {triton.__version__} predates the regression")
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


@pytest.fixture
def clean_meta_path(monkeypatch):
    monkeypatch.setattr(
        sys,
        "meta_path",
        [f for f in sys.meta_path if not getattr(f, patches._TRITON_PREFETCH_SENTINEL, False)],
    )
    yield


def test_install_is_a_no_op_under_the_kill_switch(clean_meta_path, monkeypatch):
    monkeypatch.setenv(patches._TRITON_PREFETCH_ENV, "0")
    before = list(sys.meta_path)
    assert patches.install_triton_prefetch_fix() is False
    assert sys.meta_path == before


def test_install_hooks_modules_as_they_import(clean_meta_path, monkeypatch, tmp_path):
    pkg = tmp_path / "fake_prefetch_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "compiler.py").write_text("VALUE = 41\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    monkeypatch.setattr(
        patches,
        "_POST_IMPORT",
        {"fake_prefetch_pkg.compiler": lambda module: seen.append(module.VALUE)},
    )
    monkeypatch.delenv(patches._TRITON_PREFETCH_ENV, raising = False)
    assert patches.install_triton_prefetch_fix() is True
    assert patches.install_triton_prefetch_fix() is True
    assert (
        sum(1 for f in sys.meta_path if getattr(f, patches._TRITON_PREFETCH_SENTINEL, False)) == 1
    )
    import importlib

    module = importlib.import_module("fake_prefetch_pkg.compiler")
    assert module.VALUE == 41 and seen == [41]
    monkeypatch.delitem(sys.modules, "fake_prefetch_pkg.compiler")
    monkeypatch.delitem(sys.modules, "fake_prefetch_pkg")


def test_install_patches_already_imported_modules_now(clean_meta_path, monkeypatch):
    seen = []
    module = types.ModuleType("already_here")
    monkeypatch.setitem(sys.modules, "already_here", module)
    monkeypatch.setattr(patches, "_POST_IMPORT", {"already_here": lambda m: seen.append(m)})
    before = list(sys.meta_path)
    assert patches.install_triton_prefetch_fix() is True
    assert seen == [module] and sys.meta_path == before  # nothing left to wait for


class _FakePool:
    def __init__(
        self,
        nprocs,
        pickler = None,
        kind = None,
        quiesce = False,
    ):
        self.pickler = pickler


@pytest.fixture
def fake_pool(monkeypatch):
    module = types.SimpleNamespace(SubprocPool = type("SubprocPool", (_FakePool,), {}))
    sentinel = object()
    monkeypatch.setattr(patches, "_worker_pickler_class", lambda: (lambda: sentinel))
    monkeypatch.delenv(patches._TRITON_PREFETCH_ENV, raising = False)
    return module, sentinel


def test_compile_pool_gets_the_fixing_pickler(fake_pool, monkeypatch):
    module, sentinel = fake_pool
    monkeypatch.setattr(patches, "_workers_can_import_this_module", lambda: True)
    assert patches._cover_inductor_workers(module) is True
    assert patches._cover_inductor_workers(module) is False  # once
    assert module.SubprocPool(4).pickler is sentinel
    given = object()
    assert module.SubprocPool(4, pickler = given).pickler is given
    assert module.SubprocPool(4, given).pickler is given


def test_compile_pool_is_left_alone_when_workers_cannot_import_the_fix(fake_pool, monkeypatch):
    module, _ = fake_pool
    monkeypatch.setattr(patches, "_workers_can_import_this_module", lambda: False)
    patches._cover_inductor_workers(module)
    assert module.SubprocPool(4).pickler is None


def test_compile_pool_kill_switch(fake_pool, monkeypatch):
    module, _ = fake_pool
    monkeypatch.setattr(patches, "_workers_can_import_this_module", lambda: True)
    patches._cover_inductor_workers(module)
    monkeypatch.setenv(patches._TRITON_PREFETCH_ENV, "0")
    assert module.SubprocPool(4).pickler is None


def test_this_module_is_importable_from_the_test_path():
    assert patches._workers_can_import_this_module() is True


def test_worker_pickler_is_found_by_name_and_is_a_subproc_pickler():
    pytest.importorskip("torch")
    from torch._inductor.compile_worker.subproc_pool import SubprocPickler

    cls = getattr(patches, patches._WORKER_PICKLER)
    assert issubclass(cls, SubprocPickler) and cls.__module__ == patches.__name__
    assert cls.__name__ == patches._WORKER_PICKLER


def test_real_inductor_compile_worker_runs_with_the_fix(real_triton, monkeypatch):
    """CPU only: a real Inductor SubprocPool sidecar reports the fix installed in its forked worker."""
    pytest.importorskip("torch")
    from torch._inductor.compile_worker import subproc_pool

    saved = subproc_pool.SubprocPool.__dict__["__init__"]
    # an earlier test (any Studio diffusion entry point) may already have wrapped the pool: start from stock
    stock = saved.__wrapped__ if getattr(saved, patches._TRITON_PREFETCH_SENTINEL, False) else saved
    monkeypatch.setattr(subproc_pool.SubprocPool, "__init__", stock)
    assert patches._cover_inductor_workers(subproc_pool) is True
    results = {}
    for mode in ("1", "0"):
        monkeypatch.setenv(patches._TRITON_PREFETCH_ENV, mode)
        pool = subproc_pool.SubprocPool(1)
        try:
            results[mode] = pool.submit(patches.triton_prefetch_fix_installed).result(timeout = 120)
        finally:
            pool.shutdown()
    assert results == {"1": True, "0": False}
