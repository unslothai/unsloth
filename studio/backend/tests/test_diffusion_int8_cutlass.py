# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the prebuilt CUTLASS int8 GEMM (``diffusion_int8_cutlass``) and its dispatch in ``diffusion_int8_gemm``.
The loader, gates, reference, plan dispatch and build generator run anywhere; the kernel tests need NVIDIA CUDA and
the built library (``UNSLOTH_DIFFUSION_INT8_CUTLASS_LIB`` or the ``unsloth-studio-i8gemm`` wheel)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")

_BUILD = Path(__file__).resolve().parents[2] / "native_kernels" / "int8_gemm" / "build.py"
# read before the autouse fixture clears it, for the GPU tests at the bottom
_REAL_LIB = os.environ.get(cut.CUTLASS_LIB_ENV)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for env in (cut.CUTLASS_ENV, cut.CUTLASS_LIB_ENV, g8.INT8_GEMM_ENV):
        monkeypatch.delenv(env, raising = False)
    cut.library.cache_clear()
    cut.tiles.cache_clear()
    cut._PROBED.clear()
    cut._LOAD_ERROR[0] = None
    g8._CUT_PLAN.clear()
    g8._CUT_FAMILY.clear()
    g8._CUT_RACES.clear()
    g8._CUT_DONE.clear()
    yield
    monkeypatch.undo()
    cut.library.cache_clear()
    cut.tiles.cache_clear()
    cut._PROBED.clear()
    g8._CUT_PLAN.clear()
    g8._CUT_FAMILY.clear()


@pytest.mark.parametrize(
    "cap, fam",
    [
        ((7, 5), 75),
        ((8, 0), 80),
        ((8, 6), 80),
        ((8, 7), 80),
        ((8, 9), 80),
        ((9, 0), None),
        ((10, 0), None),
        ((12, 0), None),
        ((7, 0), None),
        (None, None),
    ],
)
def test_family_covers_turing_ampere_ada_only(cap, fam):
    assert cut.family(cap) == fam


@pytest.mark.parametrize(
    "raw, want", [("0", "off"), ("off", "off"), ("1", "force"), ("", "auto"), ("auto", "auto")]
)
def test_mode(monkeypatch, raw, want):
    monkeypatch.setenv(cut.CUTLASS_ENV, raw)
    assert cut.mode() == want


def test_missing_or_broken_library_keeps_todays_path(monkeypatch, tmp_path):
    bad = tmp_path / "libunsloth_i8gemm.so"
    bad.write_bytes(b"not an ELF")
    monkeypatch.setenv(cut.CUTLASS_LIB_ENV, str(bad))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cut.library() is None
    assert cut.load_error() and str(bad) in cut.load_error()
    assert cut.tiles(80) == ()
    assert not g8.cutlass_wanted((8, 9))


def test_kill_switch_never_loads(monkeypatch):
    monkeypatch.setenv(cut.CUTLASS_ENV, "0")
    calls = []
    monkeypatch.setattr(cut, "_candidate_paths", lambda: calls.append(1) or [])
    assert cut.library() is None
    assert calls == []
    assert not g8.cutlass_wanted((8, 0))


def test_rocm_never_loads(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    monkeypatch.setattr(cut, "_candidate_paths", lambda: pytest.fail("probed on ROCm"))
    assert cut.library() is None


def test_wheel_package_is_found_without_importing_it(monkeypatch, tmp_path):
    pkg = tmp_path / cut.WHEEL_MODULE
    (pkg / "lib").mkdir(parents = True)
    (pkg / "__init__.py").write_text("raise RuntimeError('must not be imported')\n")
    lib = pkg / "lib" / cut.LIB_NAMES[0]
    lib.write_bytes(b"x")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert cut._candidate_paths() == [str(lib)]
    assert cut.WHEEL_MODULE not in sys.modules


def test_explicit_path_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(cut.CUTLASS_LIB_ENV, str(tmp_path / "x.so"))
    assert cut._candidate_paths() == [str(tmp_path / "x.so")]


def test_cutlass_wanted_for_turing_ampere_ada_only(monkeypatch):
    monkeypatch.setattr(cut, "library", lambda: object())
    assert g8.cutlass_wanted((8, 9)) and g8.cutlass_wanted((8, 6)) and g8.cutlass_wanted((8, 0))
    assert g8.cutlass_wanted((7, 5))
    assert (
        not g8.cutlass_wanted((9, 0))
        and not g8.cutlass_wanted((12, 0))
        and not g8.cutlass_wanted((10, 0))
    )
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "0")
    assert not g8.cutlass_wanted((8, 9))


def _operands(
    m,
    n,
    k,
    xs_dt,
    ws_dt,
    bias,
    seed = 0,
    out = torch.bfloat16,
):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8)
    w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8)
    xs = (torch.rand(m, generator = g) * 0.02 + 1e-4).to(xs_dt)
    ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(ws_dt)
    b = (torch.randn(n, generator = g) * 0.1).to(out) if bias else None
    return a, w, xs, ws, b


def _cpu_int_mm_ok():
    try:
        torch._int_mm(torch.ones(17, 16, dtype = torch.int8), torch.ones(16, 8, dtype = torch.int8))
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cpu_int_mm_ok(), reason = "torch._int_mm needs CPU support")
@pytest.mark.parametrize("xs_dt", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ws_dt", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("m", [3, 33])
def test_generic_reference_is_studios_bf16_reference(xs_dt, ws_dt, bias, m):
    a, w, xs, ws, b = _operands(m, 48, 64, xs_dt, ws_dt, bias)
    got = cut.reference(a, w, xs, ws, b, torch.bfloat16)
    want = g8.reference(a, w, xs, ws, b)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.skipif(not _cpu_int_mm_ok(), reason = "torch._int_mm needs CPU support")
@pytest.mark.parametrize("ws_dt", [torch.float16, torch.bfloat16, torch.float32])
def test_fp16_reference_is_the_fused_paths_torchao_epilogue(ws_dt):
    from core.inference.diffusion_int8_fused import _torchao_epilogue

    a, w, xs, ws, b = _operands(40, 32, 64, torch.float16, ws_dt, True, out = torch.float16)
    xs = xs * 0.01
    c = torch._int_mm(a, w.t())
    want = _torchao_epilogue(c, xs, ws, b, torch.float16)
    got = cut.reference(a, w, xs, ws, b, torch.float16)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.skipif(not _cpu_int_mm_ok(), reason = "torch._int_mm needs CPU support")
def test_op_body_runs_the_race_winner_then_falls_back(monkeypatch):
    a, w, xs, ws, b = _operands(1100, 64, 64, torch.bfloat16, torch.bfloat16, False)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    g8._CUT_FAMILY[None] = 80
    monkeypatch.setattr(cut, "eligible", lambda *a_: True)
    calls = []

    def fake_launch(a_, w_, xs_, ws_, b_, fam, tile, out):
        calls.append((fam, tile, a_.shape[0]))
        return torch.zeros(a_.shape[0], w_.shape[0], dtype = out)

    monkeypatch.setattr(cut, "launch", fake_launch)
    y = g8._run(a, w, xs, ws, b)
    assert calls == [(80, 3, 1100)] and not y.any()
    # small M lost its race: today's path
    small = a[:200]
    y = g8._run(small, w, xs[:200], ws, b)
    assert len(calls) == 1
    assert torch.equal(
        y.view(torch.int16), g8.reference(small, w, xs[:200], ws, b).view(torch.int16)
    )

    def broken(*a_):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(cut, "launch", broken)
    y = g8._run(a, w, xs, ws, b)
    assert None not in g8._CUT_PLAN  # dropped on this card
    assert torch.equal(y.view(torch.int16), g8.reference(a, w, xs, ws, b).view(torch.int16))


def test_fused_needs_the_large_m_win():
    key = g8._cut_key(64, 128, torch.bfloat16, torch.float32, True)
    g8._CUT_PLAN[0] = {key: {512: 2, 4096: None}}
    assert not g8._cutlass_fused(0, 64, 128, torch.bfloat16, torch.float32, True)
    g8._CUT_PLAN[0] = {key: {512: None, 4096: 1}}
    assert g8._cutlass_fused(0, 64, 128, torch.bfloat16, torch.float32, True)
    assert not g8._cutlass_fused(
        0, 64, 128, torch.bfloat16, torch.float32, False
    )  # bias is part of the key
    assert not g8._cutlass_fused(1, 64, 128, torch.bfloat16, torch.float32, True)  # per card


def test_race_margin_keeps_todays_path_on_a_tie(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = None: (8, 9))
    monkeypatch.setattr(cut, "probed_tiles", lambda i: (0, 1))
    monkeypatch.setattr(cut, "has_kernel", lambda *a: True)
    monkeypatch.setattr(cut, "eligible", lambda *a: True)
    monkeypatch.setattr(torch.cuda, "device", lambda d: __import__("contextlib").nullcontext())
    results = {}

    def fake_time(
        fns,
        rounds = 3,
        reps = 3,
        warm = 1,
    ):
        return {k: results[k] for k in fns}

    monkeypatch.setattr(cut, "time_us", fake_time)
    w = torch.zeros(64, 64, dtype = torch.int8)
    ws = torch.ones(64, dtype = torch.bfloat16)
    results.update({0: 100.0, 1: 99.0, "today": 100.0})
    r = cut.race(0, w, ws, None, torch.bfloat16, lambda *a: None, 64)
    assert r["tile"] is None  # 1% faster is inside the margin
    results.update({0: 90.0, 1: 97.0, "today": 100.0})
    r = cut.race(0, w, ws, None, torch.bfloat16, lambda *a: None, 64)
    assert r["tile"] == 0


def _load_build():
    spec = importlib.util.spec_from_file_location("i8gemm_build", _BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_generator_instantiates_every_variant_per_tile():
    b = _load_build()
    for name, fam in b.FAMILIES.items():
        n_var = len(b.variants(fam))
        native = len(fam["out"]) if fam.get("native") else 0
        assert n_var == len(fam["out"]) * len(fam["xs"]) * len(fam["ws"]) * 2 + native
        for i in range(len(fam["tiles"])):
            src = b.tile_tu(name, fam, i)
            assert src.count("return &ui8::FusedGemm<") == n_var
            assert f"ui8_{name}_t{i}(" in src
    api = b.api_tu()
    for sym in ("ui8_abi_version", "ui8_num_tiles", "ui8_tile_info", "ui8_has_kernel", "ui8_gemm"):
        assert f"UI8_API int {sym}(" in api
    assert f"return {cut.ABI_VERSION};" in api  # loader and library agree on the ABI
    assert "int epilogue" in api
    assert b.FAMILIES["sm75"].get("native") and not b.FAMILIES["sm80"].get("native")
    assert any(e == cut.EPI_NATIVE for *_, e in b.variants(b.FAMILIES["sm75"]))


def test_python_kinds_match_the_library_kinds():
    b = _load_build()
    assert cut._kind(torch.float32) == b.KIND["float"]
    assert cut._kind(torch.bfloat16) == b.KIND["cutlass::bfloat16_t"]
    assert cut._kind(torch.float16) == b.KIND["cutlass::half_t"]
    assert set(b.FAMILY_ID) == set(b.FAMILIES) and set(b.FAMILY_ID.values()) == {75, 80}
    assert cut.family((7, 5)) == b.FAMILY_ID["sm75"] and cut.family((8, 9)) == b.FAMILY_ID["sm80"]


@pytest.mark.skipif(not _cpu_int_mm_ok(), reason = "torch._int_mm needs CPU support")
def test_native_reference_is_the_native_linears_epilogue():
    a, w, _xs, _ws, _ = _operands(40, 32, 64, torch.float32, torch.float32, False)
    xs = torch.rand(40, 1) * 1e-3 + 1e-6
    ws = torch.rand(32) * 1e-3 + 1e-6
    acc = torch._int_mm(a, w.t())
    want = (
        torch.mul(acc, xs).mul_(ws).to(torch.float16)
    )  # NativeWeightOnlyLinear._int_mm_rows, no bias
    got = cut.reference_native(a, w, xs, ws, torch.float16)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_native_plan_needs_fp16_no_bias_and_a_won_race(monkeypatch):
    w = torch.zeros(64, 64, dtype = torch.int8)
    cut._NATIVE_PLAN.clear()
    cut.native_prepare(w, torch.float16, False)  # CPU weight: nothing raced
    cut.native_prepare(w, torch.bfloat16, False)
    assert cut._NATIVE_PLAN == {}
    cut._NATIVE_PLAN[(None, 64, 64)] = {512: None, 4096: 2}
    monkeypatch.setattr(cut, "eligible", lambda *a: True)
    xs = torch.ones(2000, 1)
    assert cut.native_tile(torch.zeros(2000, 64, dtype = torch.int8), w, xs, torch.ones(64)) == 2
    assert (
        cut.native_tile(torch.zeros(100, 64, dtype = torch.int8), w, xs[:100], torch.ones(64)) is None
    )
    assert (
        cut.native_tile(torch.zeros(2000, 64, dtype = torch.int8), w, xs.half(), torch.ones(64))
        is None
    )
    cut._NATIVE_PLAN.clear()


# ------------------------------------------------------------------------------------------------- GPU (needs the lib)

_GPU = pytest.mark.skipif(
    not (torch.cuda.is_available() and _REAL_LIB),
    reason = "needs CUDA and UNSLOTH_DIFFUSION_INT8_CUTLASS_LIB",
)


@pytest.fixture
def real_lib(monkeypatch):
    monkeypatch.setenv(cut.CUTLASS_LIB_ENV, _REAL_LIB)
    cut.library.cache_clear()
    cut.tiles.cache_clear()
    cut._PROBED.clear()


@_GPU
def test_library_probe_and_bit_exact_launch(real_lib):
    cap = torch.cuda.get_device_capability(0)
    fam = cut.family(cap)
    if fam is None:
        pytest.skip("arch outside the library")
    assert cut.library() is not None, cut.load_error()
    ok = cut.probed_tiles(0)
    assert ok, "no tile passed the bit-exact probe"
    out = cut.out_dtype_for(fam)
    for xs_dt, ws_dt in cut._scale_dtypes(fam):
        for bias in (False, True):
            a, w, xs, ws, b = _operands(1031, 528, 1040, xs_dt, ws_dt, bias, out = out)
            if fam == 75:
                xs = xs * 0.01
            a, w, xs, ws = a.cuda(), w.cuda(), xs.cuda(), ws.cuda()
            b = b.cuda() if b is not None else None
            ref = cut.reference(a, w, xs, ws, b, out)
            for tile in ok:
                y = cut.launch(a, w, xs, ws, b, fam, tile, out)
                assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), (
                    tile,
                    xs_dt,
                    ws_dt,
                    bias,
                )
    if fam == 75:  # the native W8A8 Linear's epilogue
        a, w, _x, _w, _ = _operands(1031, 528, 1040, torch.float32, torch.float32, False, out = out)
        a, w = a.cuda(), w.cuda()
        xs = (torch.rand(1031) * 1e-4 + 1e-6).cuda()
        ws = (torch.rand(528) * 1e-3 + 1e-6).cuda()
        ref = cut.reference_native(a, w, xs, ws, out)
        for tile in ok:
            y = cut.launch(a, w, xs, ws, None, fam, tile, out, cut.EPI_NATIVE)
            assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), tile


@_GPU
def test_launch_records_into_a_cuda_graph(real_lib):
    fam = cut.family(torch.cuda.get_device_capability(0))
    if fam is None or not cut.probed_tiles(0):
        pytest.skip("no probed tile")
    out = cut.out_dtype_for(fam)
    xs_dt, ws_dt = cut._scale_dtypes(fam)[0]
    a, w, xs, ws, _ = _operands(256, 256, 256, xs_dt, ws_dt, False, out = out)
    a, w, xs, ws = a.cuda(), w.cuda(), (xs * 0.01).cuda(), ws.cuda()
    tile = cut.probed_tiles(0)[0]
    cut.launch(a, w, xs, ws, None, fam, tile, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = cut.launch(a, w, xs, ws, None, fam, tile, out)
    a.copy_(torch.randint(-127, 128, a.shape, dtype = torch.int8, device = "cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        y.view(torch.int16), cut.reference(a, w, xs, ws, None, out).view(torch.int16)
    )


def test_a_later_load_plans_only_its_new_shapes(monkeypatch):
    class _W:
        def __init__(self, n, k):
            self.qdata = torch.zeros(n, k, dtype = torch.int8)
            self.scale = torch.ones(n, 1, dtype = torch.bfloat16)

    def _rec(n, k):
        return (torch.nn.Linear(k, n, bias = False), ("v2", None, None, _W(n, k)))

    a, b = _rec(64, 32), _rec(128, 64)
    key_a = g8._cut_key(64, 32, torch.float32, torch.bfloat16, False)
    key_b = g8._cut_key(128, 64, torch.float32, torch.bfloat16, False)
    table = {"ck": {repr(key_a): {"512": 1, "4096": 1}, repr(key_b): {"512": 2, "4096": 2}}}
    monkeypatch.setattr(g8, "cutlass_wanted", lambda cap: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(cut, "probed_tiles", lambda index: (1, 2))
    monkeypatch.setattr(g8, "_cut_cache_key", lambda index: "ck")
    monkeypatch.setattr(g8, "_cut_cache_read", lambda: table)
    monkeypatch.setattr(g8, "_cut_cache_write", lambda t: None)
    monkeypatch.setattr(g8, "_xs_dtype", lambda *a_: torch.float32)
    races = []
    monkeypatch.setattr(cut, "race", lambda *a_, **k_: races.append(a_) or {"tile": None})

    assert set(g8._cutlass_plan(0, [a], True)) == {key_a}
    assert set(g8._cutlass_plan(0, [a, b], True)) == {key_a, key_b}
    assert g8._CUT_PLAN[0][key_b] == {512: 2, 4096: 2} and not races


def test_the_kernel_sources_do_not_shadow_the_kernels_package():
    # CI and Studio put studio/ on sys.path, where a studio/kernels folder imports as the HF ``kernels`` package.
    studio = Path(__file__).resolve().parents[2]
    assert _BUILD.is_file() and not (studio / "kernels").exists()
