# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted ModelOpt FP8 (nvidia/Cosmos3-*-FP8) with dynamic activations on a torch build without the row-wise
_scaled_mm (Windows wheels): the probe must route the dynamic GEMM through the unit-scale fallback instead of
dropping to static scales, and leave builds with the row-wise kernel exactly as they were (CPU, emulated card)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_fp8_unit_scale as us  # noqa: E402
from core.inference import diffusion_modelopt_fp8 as mo  # noqa: E402

ROWWISE_ERROR = "Rowwise scaling is not currently supported on your device"
_REAL_ONES = torch.ones


def _fake_scaled_mm(
    rowwise_ok: bool,
    calls: list,
    fp32_ok: bool = True,
):
    """torch._scaled_mm as a Windows build has it (``rowwise_ok=False``): tensorwise (scalar scales) runs, row-wise
    (2-D scales) raises the error the Windows wheels raise."""

    def scaled_mm(
        a,
        b,
        scale_a,
        scale_b,
        out_dtype = None,
        use_fast_accum = False,
        bias = None,
        **_,
    ):
        rowwise = scale_a.numel() > 1 or scale_b.numel() > 1
        calls.append(
            "rowwise" if rowwise else "tensorwise32" if out_dtype == torch.float32 else "tensorwise"
        )
        if rowwise and not rowwise_ok:
            raise RuntimeError(ROWWISE_ERROR)
        if out_dtype == torch.float32 and not fp32_ok:
            raise RuntimeError("out_dtype float32 is not supported")
        acc = (a.float() * scale_a.float()) @ (b.float() * scale_b.float())
        if bias is not None:
            acc = acc + bias.float()
        return acc.to(out_dtype)

    return scaled_mm


@pytest.fixture
def card(monkeypatch):
    """An sm_120 CUDA card on a CPU host; ``card(rowwise_ok=...)`` installs its _scaled_mm and returns the call log."""
    monkeypatch.setattr(mo, "_GEMM_PROBE", {})
    monkeypatch.delenv(mo.GEMM_ENV, raising = False)
    monkeypatch.delenv(mo.ACT_SCALE_ENV, raising = False)
    monkeypatch.delenv(us.UNIT_SCALE_ENV, raising = False)
    monkeypatch.setattr(us, "_FP32_OUT", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (12, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "ones", lambda *a, device = None, **k: _REAL_ONES(*a, **k))
    from core.inference import rocm_bf16

    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda _torch = None: False)

    def make(rowwise_ok: bool, fp32_ok: bool = True) -> list:
        calls: list = []
        monkeypatch.setattr(torch, "_scaled_mm", _fake_scaled_mm(rowwise_ok, calls, fp32_ok))
        return calls

    return make


def test_without_the_rowwise_kernel_dynamic_runs_the_unit_scale_fallback(card, monkeypatch):
    card(rowwise_ok = False)
    probe = mo._probe_gemms(0)
    assert probe == {
        "static": True,
        "dynamic": False,
        "dynamic_unit_scale": True,
        "unit_scale_fp32": True,
    }
    mo._GEMM_PROBE[0] = probe
    assert mo.fp8_gemm_runs(None, device = "cuda:0", act = "dynamic") is True
    # the dynamic runtime is kept (before: it silently fell back to the static activation scales)
    monkeypatch.setenv(mo.ACT_SCALE_ENV, "dynamic")
    assert mo.modelopt_runtime(None) == mo.RUNTIME_DYNAMIC
    assert mo.gemm_route(0) == {
        "device_index": 0,
        "static": "tensorwise",
        "dynamic": "unit_scale_fallback",
        "accumulator": "fp32",
    }


def test_with_the_rowwise_kernel_nothing_changes(card, monkeypatch):
    card(rowwise_ok = True)
    probe = mo._probe_gemms(0)
    assert probe["dynamic"] is True and probe["dynamic_unit_scale"] is False
    mo._GEMM_PROBE[0] = probe
    monkeypatch.setenv(mo.ACT_SCALE_ENV, "dynamic")
    assert mo.modelopt_runtime(None) == mo.RUNTIME_DYNAMIC
    assert mo.gemm_route(0)["dynamic"] == "rowwise"


def test_the_kill_switch_keeps_the_old_static_fallback(card, monkeypatch):
    card(rowwise_ok = False)
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "0")
    probe = mo._probe_gemms(0)
    assert probe["dynamic"] is False and probe["dynamic_unit_scale"] is False
    mo._GEMM_PROBE[0] = probe
    monkeypatch.setenv(mo.ACT_SCALE_ENV, "dynamic")
    assert mo.modelopt_runtime(None) == mo.RUNTIME_STATIC


def test_forced_fallback_takes_the_unit_scale_path_where_rowwise_runs(card):
    card(rowwise_ok = True)
    import os

    os.environ[us.UNIT_SCALE_ENV] = "fallback"
    try:
        probe = mo._probe_gemms(0)
    finally:
        del os.environ[us.UNIT_SCALE_ENV]
    assert probe["dynamic"] is False and probe["dynamic_unit_scale"] is True


def test_an_oom_or_other_error_is_not_read_as_a_missing_kernel(card, monkeypatch):
    card(rowwise_ok = True)
    real = torch._scaled_mm

    def flaky(a, b, scale_a, scale_b, **k):
        if scale_a.numel() > 1:
            raise RuntimeError("CUDA out of memory")
        return real(a, b, scale_a, scale_b, **k)

    monkeypatch.setattr(torch, "_scaled_mm", flaky)
    probe = mo._probe_gemms(0)
    assert probe["dynamic"] is False and probe["dynamic_unit_scale"] is False


@pytest.mark.parametrize("fp32_ok", [True, False])
def test_the_unit_scale_forward_matches_the_rowwise_answer(card, monkeypatch, fp32_ok):
    calls = card(rowwise_ok = False, fp32_ok = fp32_ok)
    probe = mo._probe_gemms(0)
    assert probe["unit_scale_fp32"] is fp32_ok
    cls = mo.fp8_linear_class()
    torch.manual_seed(0)
    k, n = 64, 48
    w = torch.randn(n, k)
    w_scale = float(w.abs().max()) / 448
    codes = (w / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    bias = torch.randn(n)
    layer = cls(k, n, codes, w_scale, 1.0, bias = bias, act = "dynamic", fast_accum = False)
    # the CPU operands index as None: the layer sees this probe and believes it is on the card
    mo._GEMM_PROBE[None] = probe
    monkeypatch.setattr(layer, "_gemm", lambda x2: True)
    x = torch.randn(2, 300, k).to(torch.bfloat16)
    calls.clear()
    y = layer(x).float()
    # never the row-wise kernel at forward time, only the tensorwise one at scale 1
    assert calls == ["tensorwise32" if fp32_ok else "tensorwise"]
    assert layer.gemm_calls == 1 and layer.unit_scale_calls == 1
    assert mo.fp8_gemm_calls(layer)["unit_scale"] == 1
    # reference: the row-wise kernel's answer on the same codes
    x2 = x.reshape(-1, k)
    xq, s_x = layer._quantize_dynamic(x2)
    ref = (xq.float() * s_x) @ (codes.float() * w_scale).T + bias
    ref = ref.to(torch.bfloat16).float().reshape(2, 300, n)
    tol = 1e-2 if fp32_ok else 2e-2
    assert torch.allclose(y, ref, rtol = tol, atol = tol * ref.abs().max().item())
