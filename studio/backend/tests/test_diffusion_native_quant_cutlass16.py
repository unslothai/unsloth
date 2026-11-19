# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sm75 fp16 native W8A8 linears: the compiled codes feed the prebuilt CUTLASS GEMM where its race won (routing only;
the kernel's bit-equality with the stock epilogue is the load-time probe in diffusion_int8_cutlass)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_int8_cutlass as cut
import core.inference.diffusion_native_quant as nq


@pytest.fixture(autouse = True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(nq, "_FUSED_STATE", {"fn": None, "failed": None, "calls": 0})
    monkeypatch.setattr(nq, "_CUT16_CALLS", [0])
    monkeypatch.setattr(cut, "_NATIVE_PLAN", {})
    monkeypatch.delenv(nq.NATIVE_INT8_FUSED_ENV, raising = False)


def _layer(
    bias = False,
    rot = 256,
    in_f = 512,
    out_f = 128,
):
    torch.manual_seed(0)
    lin = torch.nn.Linear(in_f, out_f, bias = bias).to(torch.float16)
    return nq.native_linear_class()(lin, "int8", act_int8 = True, rot_group = rot)


def _x2(
    rows = 64,
    in_f = 512,
    dtype = torch.float16,
):
    return torch.randn(rows, in_f).to(dtype)


def _plan(layer, small, large):
    cut._NATIVE_PLAN[(None, int(layer.weight_q.shape[0]), int(layer.weight_q.shape[1]))] = {
        min(cut._RACE_M): small,
        max(cut._RACE_M): large,
    }


def _codes_spy(calls):
    def codes(x2, rot_h):
        calls.append((tuple(x2.shape), rot_h is not None))
        return torch.zeros(x2.shape[0], x2.shape[1], dtype = torch.int8), torch.ones(x2.shape[0], 1)

    return codes


def test_engages_where_the_race_won(monkeypatch):
    layer = _layer()
    _plan(layer, 3, 4)
    calls = []
    nq._FUSED_STATE["codes"] = _codes_spy(calls)
    seen = []
    monkeypatch.setattr(cut, "native_tile", lambda xq, wq, xs, ws: 3)

    def gemm(xq, wq, xs, ws, tile):
        seen.append((tuple(xq.shape), tuple(wq.shape), xs.dtype, ws.dtype, tile))
        return torch.zeros(xq.shape[0], wq.shape[0], dtype = torch.float16)

    monkeypatch.setattr(cut, "native_gemm", gemm)
    out = layer._forward_cutlass16(_x2(), layer.rot_h, 64)
    assert out.shape == (64, 128) and out.dtype == torch.float16
    assert calls == [((64, 512), True)]
    assert seen == [((64, 512), (128, 512), torch.float32, torch.float32, 3)]
    assert nq._CUT16_CALLS[0] == 1
    assert nq.fused_w8a8_state()["cutlass16_calls"] == 1


@pytest.mark.parametrize(
    "case",
    ["bias", "bf16", "no_codes", "no_plan", "small_lost", "large_lost"],
)
def test_declines_without_touching_the_codes(monkeypatch, case):
    layer = _layer(bias = case == "bias")
    rows = 2048 if case == "large_lost" else 64
    if case != "no_plan":
        _plan(layer, None if case == "small_lost" else 2, None if case == "large_lost" else 4)
    calls = []
    if case != "no_codes":
        nq._FUSED_STATE["codes"] = _codes_spy(calls)
    monkeypatch.setattr(cut, "native_gemm", lambda *a: pytest.fail("no CUTLASS launch"))
    x2 = _x2(rows, dtype = torch.bfloat16 if case == "bf16" else torch.float16)
    assert layer._forward_cutlass16(x2, layer.rot_h, rows) is None
    assert calls == []


def test_row_count_picks_the_race_slot(monkeypatch):
    layer = _layer()
    _plan(layer, None, 4)  # small race kept cuBLAS, large race won
    calls = []
    nq._FUSED_STATE["codes"] = _codes_spy(calls)
    monkeypatch.setattr(cut, "native_tile", lambda *a: 4)
    monkeypatch.setattr(
        cut,
        "native_gemm",
        lambda xq, wq, xs, ws, t: torch.zeros(xq.shape[0], wq.shape[0], dtype = torch.float16),
    )
    assert layer._forward_cutlass16(_x2(64), layer.rot_h, 64) is None
    rows = cut._NATIVE_LARGE_M
    assert layer._forward_cutlass16(_x2(rows), layer.rot_h, rows) is not None
    assert calls == [((rows, 512), True)]


def test_a_failing_codes_compile_switches_the_path_off(monkeypatch):
    layer = _layer()
    _plan(layer, 2, 4)

    def boom(*_a):
        raise RuntimeError("inductor exploded")

    nq._FUSED_STATE["codes"] = boom
    assert layer._forward_cutlass16(_x2(), layer.rot_h, 64) is None
    assert nq._FUSED_STATE["codes"] is None
    assert nq._FUSED_STATE["failed"] is None  # the compiled _w8a8_rows keeps serving


def test_an_oom_in_the_codes_propagates(monkeypatch):
    layer = _layer()
    _plan(layer, 2, 4)

    def oom(*_a):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    nq._FUSED_STATE["codes"] = oom
    with pytest.raises(torch.cuda.OutOfMemoryError):
        layer._forward_cutlass16(_x2(), layer.rot_h, 64)
    assert nq._FUSED_STATE["codes"] is oom


def _int_mm_on_cpu():
    try:
        torch._int_mm(torch.ones(32, 8, dtype = torch.int8), torch.ones(8, 8, dtype = torch.int8))
        return True
    except (RuntimeError, AttributeError, NotImplementedError):
        return False


@pytest.mark.skipif(not _int_mm_on_cpu(), reason = "torch._int_mm has no CPU kernel in this torch")
def test_ineligible_operands_run_the_stock_maths_on_the_same_codes(monkeypatch):
    layer = _layer()
    _plan(layer, 2, 4)
    nq._FUSED_STATE["codes"] = nq._w8a8_codes
    monkeypatch.setattr(cut, "native_tile", lambda *a: None)
    monkeypatch.setattr(cut, "native_gemm", lambda *a: pytest.fail("no CUTLASS launch"))
    x2 = _x2(64)
    rot_h = layer.rot_h.to(x2.dtype)
    got = layer._forward_cutlass16(x2, rot_h, 64)
    want = nq._w8a8_rows(x2, rot_h, layer.weight_q, layer.weight_scale, None)
    assert torch.equal(got, want)


@pytest.mark.skipif(not _int_mm_on_cpu(), reason = "torch._int_mm has no CPU kernel in this torch")
@pytest.mark.parametrize("rot", [True, False])
def test_codes_are_the_rows_prologue(rot):
    torch.manual_seed(1)
    x2 = torch.randn(48, 512).to(torch.float16)
    rot_h = torch.linalg.qr(torch.randn(256, 256))[0].to(torch.float16) if rot else None
    wq = torch.randint(-127, 128, (64, 512), dtype = torch.int8)
    ws = (torch.rand(64) * 1e-3 + 1e-5).float()
    xq, xs = nq._w8a8_codes(x2, rot_h)
    assert xq.dtype == torch.int8 and xq.is_contiguous() and xs.dtype == torch.float32
    acc = torch._int_mm(xq, wq.t())
    assert torch.equal(
        torch.mul(acc, xs).mul_(ws).to(torch.float16), nq._w8a8_rows(x2, rot_h, wq, ws, None)
    )


def _gemm_rows(seen):
    def gemm(xq, wq, xs, ws, tile):
        seen.append(xq.shape[0])
        # Row i of the output carries its input row's first value, so stitching order is checked too.
        return xq[:, :1].to(torch.float16).expand(-1, wq.shape[0]).contiguous()

    return gemm


def test_large_inputs_keep_the_row_budget(monkeypatch):
    """Codes and output are made per chunk, as the fused path budgets them, never for every row at once."""
    layer = _layer()
    _plan(layer, 2, 4)
    monkeypatch.setattr(
        nq, "_INT_MM_FUSED_CHUNK_ELEMS", 512 * 1536
    )  # 1536 rows per chunk at 512 features
    calls = []
    nq._FUSED_STATE["codes"] = lambda x2, rot_h: (
        calls.append(x2.shape[0]) or x2[:, :].to(torch.int8),
        torch.ones(x2.shape[0], 1),
    )
    seen = []
    monkeypatch.setattr(cut, "native_tile", lambda *a: 4)
    monkeypatch.setattr(cut, "native_gemm", _gemm_rows(seen))
    rows = 4000
    x2 = torch.arange(rows).remainder(100).to(torch.float16)[:, None].expand(rows, 512).contiguous()
    out = layer._forward_cutlass16(x2, layer.rot_h, rows)
    assert calls == seen == [1334, 1333, 1333]
    assert max(calls) <= 1536
    assert out.shape == (rows, 128) and torch.equal(out[:, 0], x2[:, 0])
    assert nq._CUT16_CALLS[0] == 3


def test_a_chunk_whose_race_kept_cublas_declines_before_any_code(monkeypatch):
    """Chunks below the large-M race read the small slot: if it kept cuBLAS, the whole call declines up front."""
    layer = _layer()
    _plan(layer, None, 4)
    monkeypatch.setattr(nq, "_INT_MM_FUSED_CHUNK_ELEMS", 512 * 800)  # chunks of under 1024 rows
    calls = []
    nq._FUSED_STATE["codes"] = _codes_spy(calls)
    monkeypatch.setattr(cut, "native_gemm", lambda *a: pytest.fail("no CUTLASS launch"))
    rows = 2048
    assert layer._forward_cutlass16(_x2(rows), layer.rot_h, rows) is None
    assert calls == []
