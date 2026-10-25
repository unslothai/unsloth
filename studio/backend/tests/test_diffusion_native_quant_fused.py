# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compiled W8A8 linear: gates, fallback, and bit-equality with the eager kernels."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_native_quant as nq


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA torch._int_mm and inductor")
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
