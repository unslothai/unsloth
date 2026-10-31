# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compiled W8A8 linear: gates, fallback, and bit-equality with the eager kernels."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_bg_compile as bg
import core.inference.diffusion_native_quant as nq

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests" / "_shared"))
from real_accelerator import has_real_cuda  # noqa: E402


@pytest.fixture(autouse = True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(nq, "_FUSED_STATE", {"fn": None, "failed": None})
    monkeypatch.delenv(nq.NATIVE_INT8_FUSED_ENV, raising = False)


def _layer(
    rot = 256,
    bias = True,
    in_f = 512,
    out_f = 128,
):
    torch.manual_seed(0)
    lin = torch.nn.Linear(in_f, out_f, bias = bias).to(torch.bfloat16)
    return nq.native_linear_class()(lin, "int8", act_int8 = True, rot_group = rot)


def _int_mm_on_cpu():
    try:
        torch._int_mm(torch.ones(32, 8, dtype = torch.int8), torch.ones(8, 8, dtype = torch.int8))
        return True
    except (RuntimeError, AttributeError, NotImplementedError):
        return False


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_kill_switch_disables_the_compiled_linear(monkeypatch, value):
    monkeypatch.setenv(nq.NATIVE_INT8_FUSED_ENV, value)
    assert nq.int8_fused_disabled()
    assert nq._fused_w8a8() is None


def test_no_inductor_runtime_or_rocm_keeps_eager(monkeypatch):
    import core.inference.diffusion_speed as sp

    monkeypatch.setattr(sp, "torch_compile_runtime_available", lambda: False)
    assert nq._fused_w8a8() is None
    assert nq.fused_w8a8_state()["failed"] == "no inductor runtime"
    monkeypatch.setattr(nq, "_FUSED_STATE", {"fn": None, "failed": None})
    monkeypatch.setattr(sp, "torch_compile_runtime_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    assert nq._fused_w8a8() is None
    assert nq.fused_w8a8_state()["failed"] == "ROCm"
    # AMD SDK / Radeon wheels leave torch.version.hip unset and tag __version__ instead.
    monkeypatch.setattr(nq, "_FUSED_STATE", {"fn": None, "failed": None})
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.9.0+rocm7.0")
    assert nq._fused_w8a8() is None
    assert nq.fused_w8a8_state()["failed"] == "ROCm"


def test_cpu_input_never_takes_the_compiled_linear(monkeypatch):
    layer = _layer()
    monkeypatch.setattr(
        layer, "_forward_fused", lambda x: pytest.fail("compiled linear on a CPU input")
    )
    layer(torch.randn(32, 512, dtype = torch.bfloat16))


def test_too_few_rows_or_switched_off_returns_none(monkeypatch):
    layer = _layer()
    monkeypatch.setattr(nq, "_fused_w8a8", lambda: pytest.fail("no compile for a 16-row input"))
    assert layer._forward_fused(torch.randn(16, 512, dtype = torch.bfloat16)) is None
    monkeypatch.setattr(nq, "_fused_w8a8", lambda: None)
    assert layer._forward_fused(torch.randn(64, 512, dtype = torch.bfloat16)) is None


def test_a_failing_compile_falls_back_once_and_stays_eager(monkeypatch):
    layer = _layer()

    def boom(*_a):
        raise RuntimeError("inductor exploded")

    nq._FUSED_STATE["fn"] = boom
    assert layer._forward_fused(torch.randn(64, 512, dtype = torch.bfloat16)) is None
    state = nq.fused_w8a8_state()
    assert "inductor exploded" in state["failed"]
    assert nq._fused_w8a8() is None


def test_an_oom_propagates_and_keeps_the_compiled_linear(monkeypatch):
    layer = _layer()

    def oom(*_a):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    nq._FUSED_STATE["fn"] = oom
    with pytest.raises(torch.cuda.OutOfMemoryError):
        layer._forward_fused(torch.randn(64, 512, dtype = torch.bfloat16))
    assert nq.fused_w8a8_state()["failed"] is None


def test_compiled_linear_gets_detached_non_parameter_inputs(monkeypatch):
    layer = _layer()
    seen = []

    def spy(x2, rot_h, wq, ws, bias):
        seen.append((x2._base, isinstance(bias, torch.nn.Parameter), rot_h is layer.rot_h))
        return (
            nq._w8a8_rows(x2, rot_h, wq, ws, bias)
            if _int_mm_on_cpu()
            else x2.new_zeros(x2.shape[0], 128)
        )

    nq._FUSED_STATE["fn"] = spy
    out = layer._forward_fused(torch.randn(2, 40, 512, dtype = torch.bfloat16))
    assert out.shape == (2, 40, 128)
    assert seen == [(None, False, True)]
    assert nq.fused_w8a8_state()["calls"] == 1


def test_large_inputs_are_chunked_on_the_accumulator(monkeypatch):
    layer = _layer()
    monkeypatch.setattr(nq, "_INT_MM_FUSED_CHUNK_ELEMS", 512 * 40)  # 40 rows per chunk
    rows = []

    def spy(x2, rot_h, wq, ws, bias):
        rows.append(x2.shape[0])
        return x2.new_zeros(x2.shape[0], 128)

    nq._FUSED_STATE["fn"] = spy
    assert layer._forward_fused(torch.randn(100, 512, dtype = torch.bfloat16)).shape == (100, 128)
    assert sum(rows) == 100 and max(rows) <= 40


def test_chunks_fill_one_output_in_row_order(monkeypatch):
    layer = _layer()
    monkeypatch.setattr(nq, "_INT_MM_FUSED_CHUNK_ELEMS", 512 * 40)
    nq._FUSED_STATE["fn"] = lambda x2, *_a: x2[:, :128] * 2
    x = torch.randn(100, 512, dtype = torch.bfloat16)
    assert torch.equal(layer._forward_fused(x), x[:, :128] * 2)


@pytest.mark.parametrize("in_f, out_f", [(3072, 14336), (14336, 3072), (512, 1536), (5120, 5120)])
def test_fused_chunk_never_outgrows_the_eager_chunk(in_f, out_f):
    # Same row budget as eager keeps the compiled peak <= eager's (#15183 used 4x and OOMed).
    wide = max(in_f, out_f)
    fused_rows = nq._INT_MM_FUSED_CHUNK_ELEMS // wide
    eager_rows = nq._INT_MM_CHUNK_ELEMS // wide
    assert fused_rows <= eager_rows
    for rows in (eager_rows - 1, eager_rows + 1, 3 * eager_rows + 7, 9 * eager_rows):
        if rows <= fused_rows:
            fused, eager_peak = 6 * rows * out_f, 8 * rows * out_f
        else:
            n_f, n_e = -(-rows // fused_rows), -(-rows // eager_rows)
            fused = 2 * rows * out_f + 6 * -(-rows // n_f) * out_f
            last = rows - (n_e - 1) * (rows // n_e)
            eager_peak = max(2 * (rows - last) * out_f + 8 * last * out_f, 4 * rows * out_f)
        assert fused <= eager_peak, (rows, fused, eager_peak)


def test_eager_forced_pass_never_enters_the_compiled_linear(monkeypatch):
    layer = _layer()
    monkeypatch.setattr(
        nq, "_fused_w8a8", lambda: pytest.fail("compiled linear inside a force-eager pass")
    )
    x = torch.randn(64, 512, dtype = torch.bfloat16)
    with bg.auxiliary_pass():
        assert layer._forward_fused(x) is None
    with bg.force_eager():
        assert layer._forward_fused(x) is None
    assert nq.fused_w8a8_state()["calls"] == 0


@pytest.mark.skipif(not _int_mm_on_cpu(), reason = "torch._int_mm has no CPU kernel in this build")
@pytest.mark.parametrize("rot", [256, 0])
@pytest.mark.parametrize("bias", [True, False])
def test_fused_maths_tracks_the_eager_kernels(rot, bias):
    # CPU eager divides, so only ulp-level agreement; the CUDA test checks bit-equality.
    layer = _layer(rot = rot, bias = bias)
    x = torch.randn(48, 512, dtype = torch.bfloat16)
    x_rot = layer._rotate(x) if rot else x
    ref = layer._int_mm_rows(x_rot).to(x.dtype)
    got = nq._w8a8_rows(
        x, layer.rot_h if rot else None, layer.weight_q, layer.weight_scale, layer.bias
    )
    assert got.dtype == torch.bfloat16 and got.shape == ref.shape
    assert ((got.float() - ref.float()).norm() / ref.float().norm()).item() < 2e-2


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA torch._int_mm and inductor")
@pytest.mark.parametrize("rot", [256, 0])
def test_cuda_compiled_linear_is_bit_identical_to_eager(monkeypatch, rot):
    torch.manual_seed(0)
    cls = nq.native_linear_class()
    layers = [
        cls(
            torch.nn.Linear(i, o, bias = b).to("cuda", torch.bfloat16),
            "int8",
            act_int8 = True,
            rot_group = rot,
        )
        for i, o, b in ((512, 1536, True), (512, 512, False), (1536, 512, True))
    ]
    torch._dynamo.reset()
    with torch.no_grad():
        for shape in ((1, 700), (2, 333), (4096,)):
            for layer in layers:
                x = torch.randn(*shape, layer.in_features, device = "cuda", dtype = torch.bfloat16) * 3
                monkeypatch.setenv(nq.NATIVE_INT8_FUSED_ENV, "0")
                ref = layer(x)
                monkeypatch.delenv(nq.NATIVE_INT8_FUSED_ENV)
                calls = nq.fused_w8a8_state()["calls"]
                got = layer(x)
                assert nq.fused_w8a8_state()["calls"] == calls + 1
                assert torch.equal(got, ref)
    assert nq.fused_w8a8_state()["failed"] is None


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA torch._int_mm and inductor")
@pytest.mark.parametrize("rot", [256, 0])
def test_cuda_compiled_linear_peak_stays_at_or_below_eager(monkeypatch, rot):
    # 8192 rows x 16384 outputs: eager takes four chunks, the old 4x budget one call (768 vs 576 MiB).
    torch.manual_seed(0)
    layer = nq.native_linear_class()(
        torch.nn.Linear(512, 16384).to("cuda", torch.bfloat16), "int8", act_int8 = True, rot_group = rot
    )
    x = torch.randn(8192, 512, device = "cuda", dtype = torch.bfloat16)
    torch._dynamo.reset()

    def peak():
        layer(x)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = layer(x)
        torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated() - base

    with torch.no_grad():
        monkeypatch.setenv(nq.NATIVE_INT8_FUSED_ENV, "0")
        ref, eager_peak = peak()
        monkeypatch.delenv(nq.NATIVE_INT8_FUSED_ENV)
        calls = nq.fused_w8a8_state()["calls"]
        got, fused_peak = peak()
    assert nq.fused_w8a8_state()["calls"] == calls + 2
    assert torch.equal(got, ref)
    assert fused_peak <= eager_peak, (fused_peak, eager_peak)


def test_chunked_output_keeps_the_compiled_dtype(monkeypatch):
    # Under autocast the compiled function can return another dtype than the input; chunking must not recast it.
    layer = _layer()
    monkeypatch.setattr(nq, "_INT_MM_FUSED_CHUNK_ELEMS", 512 * 40)
    nq._FUSED_STATE["fn"] = lambda x2, *_a: (x2[:, :128] * 2).to(torch.float16)
    x = torch.randn(100, 512, dtype = torch.float32)
    got = layer._forward_fused(x)
    assert got.dtype == torch.float16
    assert torch.equal(got, (x[:, :128] * 2).to(torch.float16))


def test_chunks_are_stored_once_into_their_output_rows(monkeypatch):
    # Each chunk is written in place into its rows of the one output: no chunk result, no second copy.
    layer = _layer()
    monkeypatch.setattr(nq, "_INT_MM_FUSED_CHUNK_ELEMS", 512 * 40)
    writes = []

    def into(x2, rot_h, wq, ws, bias, out):
        writes.append((out.data_ptr(), out.shape[0], out.untyped_storage().data_ptr()))
        out.copy_(x2[:, :128] * 2)

    def fn(*_a):
        raise AssertionError("the chunked path must not build a chunk result")

    nq._FUSED_STATE.update(fn = fn, into = into)
    x = torch.randn(100, 512, dtype = torch.bfloat16)
    got = layer._forward_fused(x)
    assert torch.equal(got, x[:, :128] * 2)
    base = got.untyped_storage().data_ptr()
    assert [w[2] for w in writes] == [base] * len(writes) and len(writes) == 3
    step = got.element_size() * 128
    assert [w[0] for w in writes] == [got.data_ptr() + step * r for r in (0, 34, 67)]
    assert sum(w[1] for w in writes) == 100


def test_rotation_under_autocast_keeps_the_returning_path(monkeypatch):
    layer = _layer()
    monkeypatch.setattr(nq, "_INT_MM_FUSED_CHUNK_ELEMS", 512 * 40)

    def into(*_a):
        raise AssertionError(
            "autocast may change the rotated dtype; the output dtype must come from a chunk"
        )

    nq._FUSED_STATE.update(fn = lambda x2, *_a: (x2[:, :128] * 2).to(torch.float16), into = into)
    x = torch.randn(100, 512, dtype = torch.float32)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        got = layer._forward_fused(x)
    assert got.dtype == torch.float16
    assert torch.equal(got, (x[:, :128] * 2).to(torch.float16))


@pytest.mark.skipif(not _int_mm_on_cpu(), reason = "torch._int_mm unavailable on CPU")
@pytest.mark.parametrize("rot", [0, 256])
def test_compiled_into_slice_is_one_graph_and_bit_identical(monkeypatch, rot):
    from torch._dynamo.testing import CompileCounterWithBackend

    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    layer = _layer(rot = rot)
    cnt = CompileCounterWithBackend("aot_eager")
    nq._FUSED_STATE["row_major"] = nq._row_major_op()
    fn = torch.compile(nq._w8a8_rows, dynamic = True, fullgraph = True, backend = "aot_eager")
    into = torch.compile(nq._w8a8_rows_into, dynamic = True, fullgraph = True, backend = cnt)
    nq._FUSED_STATE.update(fn = fn, into = into, calls = 0)  # as _fused_w8a8 leaves it
    monkeypatch.setattr(nq, "_INT_MM_FUSED_CHUNK_ELEMS", 512 * 40)
    with torch.no_grad():
        for rows in (100, 130, 77):
            x = torch.randn(rows, 512, dtype = torch.bfloat16)
            rot_h = layer.rot_h if layer.rot_group else None
            ref = nq._w8a8_rows(x, rot_h, layer.weight_q, layer.weight_scale, layer.bias.detach())
            assert torch.equal(layer._forward_fused(x), ref)
    # Every slice offset and chunk height reuses one graph: no recompile, no graph break.
    assert cnt.frame_count == 1
    assert not counters["graph_break"]
    torch._dynamo.reset()


def test_factory_creates_every_state_key_before_the_first_trace(monkeypatch):
    # torch <= 2.6 dynamo guards on len(_FUSED_STATE): a key added after the first call recompiles.
    import core.inference.diffusion_speed as sp

    monkeypatch.setattr(sp, "torch_compile_runtime_available", lambda: True)
    assert nq._fused_w8a8() is not None
    assert {"fn", "into", "row_major", "calls", "failed"} <= set(nq._FUSED_STATE)
