# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eager int8 / stored-fp8 widening without full float32 temporaries: bitwise equal to the whole-layer formula on
the chunked torch path and the Triton kernel (under TRITON_INTERPRET=1), kill switch, compile unchanged."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_comfy_quant as cq  # noqa: E402
import core.inference.diffusion_native_quant as nq  # noqa: E402

# The tree ``core`` was imported from (the subprocess imports the same one).
BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(nq.__file__))))
ENV = "UNSLOTH_DIFFUSION_FP8_DEQUANT_FUSED"
FP8 = ("float8_e4m3fn", "float8_e5m2")
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# 1, extremes, the e4m3 max-ish, non-powers of two (rounding), a subnormal scale, a negative one (signed zero)
SCALES = (1.0, 2.0**-20, 2.0**20, 448.0, 1 / 448, 0.7, 3.3e-3, 1e-40, 2.0**-140, -1.5, 2.0**100)


def _fd():
    import core.inference.diffusion_fp8_dequant as fd
    return fd


def test_imports_own_tree():
    """The modules under test are this test's own tree."""
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    for mod in (nq, cq):
        assert os.path.realpath(mod.__file__).startswith(root + os.sep), mod.__file__


def _old(
    codes,
    scale,
    dtype,
    fmt = None,
):
    """Today's eager formula, independent of the tree."""
    wide = (
        codes.to(torch.float32)
        if fmt is None
        else codes.view(getattr(torch, fmt)).to(torch.float32)
    )
    return (wide * scale[:, None]).to(dtype)


def _bits(t):
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _assert_bitwise(new, ref):
    assert new.dtype == ref.dtype and new.shape == ref.shape
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(new), nan)
    assert torch.equal(_bits(new)[~nan], _bits(ref)[~nan])


def _layer(
    codes,
    scale,
    scheme,
    dtype = torch.bfloat16,
    bias = False,
):
    lin = torch.nn.Linear(codes.shape[1], codes.shape[0], bias = bias, dtype = dtype)
    return nq.native_linear_class()(lin, scheme, codes = codes, scale = scale)


def _exhaustive(fmt):
    codes = torch.arange(256, dtype = torch.uint8).repeat(len(SCALES), 1).contiguous()
    if fmt is None:
        codes = codes.view(torch.int8)
    return codes, torch.tensor(SCALES, dtype = torch.float32)


@pytest.fixture
def tiny_chunks(monkeypatch):
    """One row per chunk on CPU, so every row runs the chunked path."""
    monkeypatch.delenv(ENV, raising = False)
    monkeypatch.setattr(_fd(), "CHUNK_SCRATCH_BYTES_CPU", 4)


@pytest.mark.parametrize("fmt", (*FP8, None))
@pytest.mark.parametrize("dtype", DTYPES)
def test_every_code_and_scale_is_bitwise_on_the_chunked_path(tiny_chunks, fmt, dtype):
    codes, scale = _exhaustive(fmt)
    store = codes.view(getattr(torch, fmt)) if fmt else codes
    layer = _layer(store, scale, nq.NATIVE_FP8 if fmt else nq.NATIVE_INT8)
    ref = _old(codes, scale, dtype, fmt)
    _assert_bitwise(layer._stored_weight(dtype), ref)
    # Same per-element ops as today, so NaN bits match too.
    assert torch.equal(_bits(layer._stored_weight(dtype)), _bits(ref))


@pytest.mark.parametrize("fmt", FP8)
def test_integer_decode_matches_torch_cast_bits(fmt):
    """The bit decode the kernel mirrors: fp8_bits_to_float32 for every value, torch's NaN bits in the kernel."""
    codes = torch.arange(256, dtype = torch.uint8)
    cast = codes.view(getattr(torch, fmt)).float()
    _assert_bitwise(nq.fp8_bits_to_float32(codes, fmt), cast)


_INTERP = textwrap.dedent(
    """
    import json, os, sys
    os.environ["TRITON_INTERPRET"] = "1"
    sys.path.insert(0, sys.argv[1])
    import torch
    import core.inference.diffusion_fp8_dequant as fd
    assert os.path.realpath(fd.__file__).startswith(os.path.realpath(sys.argv[1]))
    scales = torch.tensor(json.loads(sys.argv[2]), dtype = torch.float32)
    res = []
    for fmt in ("float8_e4m3fn", "float8_e5m2", None):
        codes = torch.arange(256, dtype = torch.uint8).repeat(len(scales), 1).contiguous()
        if fmt is None:
            codes = codes.view(torch.int8)
        wide = codes.to(torch.float32) if fmt is None else codes.view(getattr(torch, fmt)).to(torch.float32)
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            ref = (wide * scales[:, None]).to(dt)
            out = torch.full(codes.shape, 7, dtype = dt)
            fd.launch_triton(codes, scales, out, fmt)
            iv = {2: torch.int16, 4: torch.int32}[out.element_size()]
            nan = torch.isnan(ref)
            res.append(dict(
                fmt = fmt, dtype = str(dt),
                nan_pos = bool(torch.equal(torch.isnan(out), nan)),
                values = bool(torch.equal(out.view(iv)[~nan], ref.view(iv)[~nan])),
                nan_bits_equal = bool(torch.equal(out.view(iv)[nan], ref.view(iv)[nan])),
                n_nan = int(nan.sum()),
                n = int(ref.numel()),
            ))
    # Masked tail and int64 offsets: odd widths.
    for cols in (1, 3, 1023, 1025):
        codes = torch.randint(0, 256, (5, cols), dtype = torch.uint8)
        s = torch.rand(5) + 0.1
        ref = (codes.view(torch.float8_e4m3fn).float() * s[:, None]).to(torch.bfloat16)
        out = torch.empty(codes.shape, dtype = torch.bfloat16)
        fd.launch_triton(codes, s, out, "float8_e4m3fn")
        nan = torch.isnan(ref)
        res.append(dict(fmt = "tail", dtype = str(cols), nan_pos = bool(torch.equal(torch.isnan(out), nan)),
                        values = bool(torch.equal(out.view(torch.int16)[~nan], ref.view(torch.int16)[~nan])),
                        nan_bits_equal = None, n_nan = int(nan.sum()), n = int(ref.numel())))
    print("RESULT " + json.dumps(res))
    """
)


def test_triton_kernel_is_bitwise_under_the_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, "-c", _INTERP, BACKEND, json.dumps(SCALES)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not line and "interpret" in proc.stderr.lower() and "not supported" in proc.stderr.lower():
        # skip because this Triton build has no interpreter for the op; the GPU check covers the kernel
        pytest.skip(f"Triton interpreter cannot run here: {proc.stderr[-400:]}")
    assert line, proc.stderr[-4000:]
    res = json.loads(line[0][len("RESULT ") :])
    print(json.dumps(res, indent = 1))
    for r in res:
        assert r["nan_pos"] and r["values"], r
    # NaN bits too, against CPU torch (the bfloat16 NaN constant is chosen per backend).
    for r in res:
        if r["fmt"] != "tail":
            assert r["nan_bits_equal"], r


@pytest.mark.parametrize("rows", (1, 17, 1023, 1024, 1025))
@pytest.mark.parametrize("fmt", ("float8_e4m3fn", None))
def test_odd_row_counts_over_several_chunks(monkeypatch, rows, fmt):
    monkeypatch.delenv(ENV, raising = False)
    cols = 64
    # 100 rows per chunk: 1025 rows = 11 chunks with a 25-row tail.
    monkeypatch.setattr(_fd(), "CHUNK_SCRATCH_BYTES_CPU", 100 * cols * 4)
    if rows * cols * 8 <= 100 * cols * 4:
        assert _fd().chunk_rows(rows, cols, torch.device("cpu")) == 0
    else:
        assert _fd().chunk_rows(rows, cols, torch.device("cpu")) == 100
    g = torch.Generator().manual_seed(rows)
    codes = torch.randint(0, 256, (rows, cols), dtype = torch.uint8, generator = g)
    if fmt is None:
        codes = codes.view(torch.int8)
    scale = torch.rand(rows, generator = g) * 0.01 + 1e-4
    store = codes.view(getattr(torch, fmt)) if fmt else codes
    layer = _layer(store, scale, nq.NATIVE_FP8 if fmt else nq.NATIVE_INT8)
    for dtype in DTYPES:
        _assert_bitwise(layer._stored_weight(dtype), _old(codes, scale, dtype, fmt))


def test_kill_switch_takes_the_old_path(monkeypatch):
    calls = []
    real = nq.dequant_rows

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(nq, "dequant_rows", spy)
    codes, scale = _exhaustive("float8_e4m3fn")
    layer = _layer(codes.view(torch.float8_e4m3fn), scale, nq.NATIVE_FP8)
    monkeypatch.setenv(ENV, "0")
    old = layer._stored_weight(torch.bfloat16)
    assert calls == []
    monkeypatch.setenv(ENV, "1")
    new = layer._stored_weight(torch.bfloat16)
    assert calls == [1]
    assert torch.equal(_bits(old), _bits(new))

    seen = []
    monkeypatch.setattr(cq, "dequant_rows", lambda *a, **k: seen.append(1) or real(*a, **k))
    monkeypatch.setattr(_fd(), "CHUNK_SCRATCH_BYTES_CPU", 4)
    monkeypatch.setenv(ENV, "0")
    cq._dequant(codes.view(torch.float8_e4m3fn), scale, 0, torch.bfloat16)
    assert seen == []
    monkeypatch.setenv(ENV, "1")
    cq._dequant(codes.view(torch.float8_e4m3fn), scale, 0, torch.bfloat16)
    assert seen == [1]


def test_triton_failure_latches_and_falls_back(monkeypatch):
    """A launch failure never reaches the caller: chunked torch, and Triton is not tried again."""
    fd = _fd()
    tries = []

    def boom(*a, **k):
        tries.append(1)
        raise RuntimeError("no triton")

    monkeypatch.setattr(fd, "launch_triton", boom)
    monkeypatch.setattr(fd, "_TRITON_STATE", {"kernel": None, "failed": None})
    monkeypatch.setattr(fd, "_triton_ok", lambda c, d: fd._TRITON_STATE["failed"] is None)
    codes, scale = _exhaustive("float8_e5m2")
    widen = lambda c: c.view(torch.float8_e5m2).to(torch.float32)
    for _ in range(2):
        out = fd.dequant_rows(codes, scale, torch.float16, widen, "float8_e5m2")
        _assert_bitwise(out, _old(codes, scale, torch.float16, "float8_e5m2"))
    assert tries == [1] and fd._TRITON_STATE["failed"]


class _Fp32Full(torch.utils._python_dispatch.TorchDispatchMode):
    """Counts op outputs that are full-size float32 [rows, cols] matrices."""

    def __init__(self, numel):
        super().__init__()
        self.numel, self.hits = numel, 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.numel() >= self.numel:
                self.hits += 1
        return out


@pytest.mark.parametrize("scheme", (nq.NATIVE_FP8, nq.NATIVE_INT8))
def test_eager_widen_holds_no_full_fp32_matrix(monkeypatch, scheme):
    """Base: the float32 codes and their float32 product (2 full matrices). Head: one bounded chunk."""
    monkeypatch.delenv(ENV, raising = False)
    rows, cols = 2048, 4096  # 32 MiB of float32 per matrix, over the 16 MiB CPU budget
    g = torch.Generator().manual_seed(0)
    raw = torch.randint(-127, 128, (rows, cols), dtype = torch.int8, generator = g)
    codes = raw.view(torch.uint8).view(torch.float8_e4m3fn) if scheme == nq.NATIVE_FP8 else raw
    if scheme == nq.NATIVE_FP8:
        codes = torch.where(torch.isnan(codes.float()), 0, codes.float()).to(torch.float8_e4m3fn)
    layer = _layer(codes, torch.rand(rows) + 0.5, scheme)
    with _Fp32Full(rows * cols) as mode:
        layer._stored_weight(torch.bfloat16)
    assert mode.hits == 0
    monkeypatch.setenv(ENV, "0")
    with _Fp32Full(rows * cols) as mode:
        layer._stored_weight(torch.bfloat16)
    assert mode.hits == 2


@pytest.mark.parametrize("fmt", (*FP8, "int8"))
@pytest.mark.parametrize("per_row", (True, False))
def test_comfy_dequant_chunked_is_bitwise_and_charges_the_true_peak(monkeypatch, fmt, per_row):
    monkeypatch.delenv(ENV, raising = False)
    import core.inference.diffusion_single_file_converters as sfc

    charged = []
    monkeypatch.setattr(sfc, "charge_host_ram", lambda n: charged.append(int(n)))
    rows, cols = 1025, 96
    monkeypatch.setattr(_fd(), "CHUNK_SCRATCH_BYTES_CPU", 64 * cols * 4)
    g = torch.Generator().manual_seed(1)
    raw = torch.randint(0, 256, (rows, cols), dtype = torch.uint8, generator = g)
    codes = raw.view(torch.int8) if fmt == "int8" else raw.view(getattr(torch, fmt))
    scale = (torch.rand(rows if per_row else 1, generator = g) * 0.02).to(torch.float32)
    for dtype in DTYPES:
        charged.clear()
        new = cq._dequant(codes, scale, 0, dtype)
        new_charge = charged[-1]
        monkeypatch.setenv(ENV, "0")
        charged.clear()
        old = cq._dequant(codes, scale, 0, dtype)
        old_charge = charged[-1]
        monkeypatch.delenv(ENV)
        ref = (codes.to(torch.float32) * scale.reshape(-1, 1).expand(rows, 1)).to(dtype)
        _assert_bitwise(new, ref)
        assert torch.equal(_bits(new), _bits(old))
        out_size = torch.empty((), dtype = dtype).element_size()
        assert new_charge == rows * cols * out_size + 64 * cols * 4
        assert old_charge == rows * cols * max(8, 4 + out_size) > new_charge


def test_comfy_convrot_dequant_is_unchanged(monkeypatch):
    """ConvRot (group > 0) is not row-chunked: same expression, same charge, either setting."""
    monkeypatch.setattr(_fd(), "CHUNK_SCRATCH_BYTES_CPU", 4)
    import core.inference.diffusion_single_file_converters as sfc

    charged = []
    monkeypatch.setattr(sfc, "charge_host_ram", lambda n: charged.append(int(n)))
    codes = torch.randint(-127, 128, (33, 64), dtype = torch.int8)
    scale = torch.rand(33)
    monkeypatch.setenv(ENV, "0")
    old = cq._dequant(codes, scale, 16, torch.bfloat16)
    monkeypatch.setenv(ENV, "1")
    new = cq._dequant(codes, scale, 16, torch.bfloat16)
    assert torch.equal(_bits(old), _bits(new))
    assert charged == [33 * 64 * 12] * 2


@pytest.mark.parametrize("scheme", (nq.NATIVE_FP8, nq.NATIVE_INT8))
def test_forward_is_bitwise_new_vs_old(monkeypatch, scheme):
    monkeypatch.setattr(_fd(), "CHUNK_SCRATCH_BYTES_CPU", 8 * 48 * 4)
    g = torch.Generator().manual_seed(2)
    lin = torch.nn.Linear(48, 40, bias = True, dtype = torch.bfloat16)
    layer = nq.native_linear_class()(lin, scheme)
    x = torch.randn(3, 5, 48, generator = g).to(torch.bfloat16)
    monkeypatch.setenv(ENV, "1")
    new = layer(x)
    w_new = layer.weight
    monkeypatch.setenv(ENV, "0")
    old = layer(x)
    w_old = layer.weight
    assert torch.equal(_bits(new), _bits(old))
    assert torch.equal(_bits(w_new), _bits(w_old))


def _compile_counts(layer, x, backend):
    from torch._dynamo.testing import CompileCounterWithBackend

    torch._dynamo.reset()
    cnt = CompileCounterWithBackend(backend)
    fn = torch.compile(layer, backend = cnt, fullgraph = True)
    out = fn(x)
    out2 = fn(x)
    return cnt.frame_count, cnt.op_count, out, out2


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_stored_fp8_layer_graph_is_unchanged(monkeypatch, backend):
    """The compiled branch is today's: one fullgraph frame, the same op count with the switch on or off."""
    lin = torch.nn.Linear(64, 32, bias = True, dtype = torch.float32)
    layer = nq.native_linear_class()(lin, nq.NATIVE_FP8)
    x = torch.randn(4, 64)
    counts = {}
    for flag in ("1", "0"):
        monkeypatch.setenv(ENV, flag)
        try:
            frames, ops, out, out2 = _compile_counts(layer, x, backend)
        except Exception as exc:  # noqa: BLE001 - no C++ toolchain for CPU inductor
            if backend == "inductor":
                # skip because CPU inductor needs a C++ toolchain; the aot_eager leg still runs
                pytest.skip(f"CPU inductor unavailable: {exc!r}"[:300])
            raise
        assert frames == 1
        assert torch.equal(out, out2)
        counts[flag] = (frames, ops)
    print("compile counts", backend, counts)
    assert counts["1"] == counts["0"]
    torch._dynamo.reset()


def test_cuda_graph_capture_takes_only_a_warm_kernel(monkeypatch):
    """No first-launch JIT / NaN probe sync inside a capture: chunked torch until the specialisation ran eager."""
    import types

    fd = _fd()
    monkeypatch.setattr(fd, "_TRITON_STATE", {"kernel": None, "failed": None})
    monkeypatch.setattr(fd, "_WARM", set())
    codes = types.SimpleNamespace(
        is_cuda = True,
        is_contiguous = lambda: True,
        dim = lambda: 2,
        shape = (4, 32),
        device = "cuda:0",
        dtype = torch.uint8,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert fd._triton_ok(codes, torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not fd._triton_ok(codes, torch.bfloat16)
    fd._WARM.add(fd._warm_key(codes, torch.bfloat16))
    assert fd._triton_ok(codes, torch.bfloat16)
    assert not fd._triton_ok(codes, torch.float16)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "needs a CUDA / ROCm GPU to run the Triton widen"
)
@pytest.mark.parametrize("fmt", (*FP8, None))
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_triton_widen_on_the_gpu_matches_the_old_formula(monkeypatch, fmt, dtype):
    monkeypatch.delenv(ENV, raising = False)
    codes, scale = _exhaustive(fmt)
    codes = torch.cat(
        [codes] * 8
    ).contiguous()  # several rows per scale, every column block masked or full
    scale = torch.cat([scale] * 8)
    store = codes.view(getattr(torch, fmt)) if fmt else codes
    layer = _layer(store, scale, nq.NATIVE_FP8 if fmt else nq.NATIVE_INT8).to("cuda")
    launches = []
    real = _fd().launch_triton
    monkeypatch.setattr(_fd(), "launch_triton", lambda *a: launches.append(1) or real(*a))
    got = layer._stored_weight(dtype).cpu()
    assert launches and _fd()._TRITON_STATE["failed"] is None, _fd()._TRITON_STATE["failed"]
    ref = _old(codes, scale, dtype, fmt)
    _assert_bitwise(got, ref)


def test_the_triton_capture_probe_and_launch_run_on_the_operands_device(monkeypatch):
    """Triton launches on the current device's current stream: both the capture probe and the launch must see the
    codes' GPU current, not the one the caller left current."""
    import contextlib
    import types

    fd = _fd()
    monkeypatch.setattr(fd, "_TRITON_STATE", {"kernel": None, "failed": None})
    monkeypatch.setattr(fd, "_WARM", set())
    current = ["cuda:0"]
    seen = {}

    @contextlib.contextmanager
    def device(dev):
        prev, current[0] = current[0], str(dev)
        try:
            yield
        finally:
            current[0] = prev

    def capturing():
        seen["probe"] = current[0]
        return False

    def launch(codes, scale, out, fmt):
        seen["launch"] = current[0]

    out = object()
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capturing)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: out)
    monkeypatch.setattr(fd, "launch_triton", launch)
    codes = types.SimpleNamespace(
        is_cuda = True,
        is_contiguous = lambda: True,
        dim = lambda: 2,
        shape = (4, 32),
        device = "cuda:1",
        dtype = torch.uint8,
    )
    got = fd.dequant_rows(codes, torch.ones(4), torch.bfloat16, lambda c: c, None)
    assert got is out and seen == {"probe": "cuda:1", "launch": "cuda:1"}
    assert current == ["cuda:0"] and fd._TRITON_STATE["failed"] is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason = "needs two CUDA / ROCm GPUs",
)
@pytest.mark.parametrize("fmt", (*FP8, None))
def test_the_triton_widen_of_a_layer_on_a_non_current_gpu(monkeypatch, fmt):
    monkeypatch.delenv(ENV, raising = False)
    fd = _fd()
    monkeypatch.setattr(fd, "_TRITON_STATE", {"kernel": fd._TRITON_STATE["kernel"], "failed": None})
    codes, scale = _exhaustive(fmt)
    store = codes.view(getattr(torch, fmt)) if fmt else codes
    layer = _layer(store, scale, nq.NATIVE_FP8 if fmt else nq.NATIVE_INT8).to("cuda:1")
    launches = []
    real = fd.launch_triton
    monkeypatch.setattr(fd, "launch_triton", lambda *a: launches.append(1) or real(*a))
    with torch.cuda.device(0):
        got = layer._stored_weight(torch.bfloat16)
        assert torch.cuda.current_device() == 0
    torch.cuda.synchronize(1)
    assert launches and fd._TRITON_STATE["failed"] is None, fd._TRITON_STATE["failed"]
    assert got.device == torch.device("cuda", 1)
    _assert_bitwise(got.cpu(), _old(codes, scale, torch.bfloat16, fmt))


def test_one_element_scale_broadcasts_on_kernel_and_chunked_paths(monkeypatch):
    # A per-tensor scale (ComfyUI scaled checkpoints) must reach the kernel as one value per row, since it reads
    # scale[row]; a 1-element buffer there would be read out of bounds on CUDA / ROCm.
    import core.inference.diffusion_fp8_dequant as fd

    codes = torch.randint(-127, 127, (6, 5), dtype = torch.int8)
    scale = torch.tensor([0.5], dtype = torch.float32)
    want = (codes.float() * 0.5).to(torch.bfloat16)
    seen = []

    def fake_launch(codes_, scale_, out, fmt):
        seen.append(scale_.numel())
        out.copy_((codes_.float() * scale_[: codes_.shape[0], None]).to(out.dtype))

    monkeypatch.setattr(fd, "_triton_ok", lambda *a: True)
    monkeypatch.setattr(fd, "launch_triton", fake_launch)
    got = fd.dequant_rows(codes, scale, torch.bfloat16, lambda c: c.float())
    assert seen == [6] and torch.equal(got, want)

    monkeypatch.setattr(fd, "chunk_rows", lambda *a, **k: 2)
    got = fd.dequant_rows(codes, scale, torch.bfloat16, lambda c: c.float(), triton_ok = False)
    assert torch.equal(got, want)
    with pytest.raises(ValueError):
        fd.dequant_rows(codes, torch.ones(3), torch.bfloat16, lambda c: c.float(), triton_ok = False)
