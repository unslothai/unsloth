# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fused static e4m3 activation quantiser of the ModelOpt FP8 Linear: when it declines (CPU, traced, kill switch,
strided input) the eager formula runs, so the Linear's output is unchanged; on CUDA sm_89+ its codes equal the eager
formula's byte for byte (the device check runs every bf16 / fp16 bit pattern)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_fp8_act_quant as fq  # noqa: E402
from core.inference import diffusion_modelopt_fp8 as mo  # noqa: E402


def _linear(policy = None):
    torch.manual_seed(0)
    w = torch.randn(48, 32) * 0.05
    ws = float(w.abs().max() / 448)
    codes = (w / ws).clamp(-448, 448).to(torch.float8_e4m3fn)
    return mo.fp8_linear_class()(32, 48, codes, ws, 0.0123, bias = torch.randn(48), policy = policy)


def _eager_codes(x2, scale):
    return (x2.to(torch.float32) / scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)


def test_reference_is_the_eager_chunk_formula():
    x = torch.randn(37, 32, dtype = torch.bfloat16) * 9
    s = torch.tensor([0.0123], dtype = torch.float32)
    assert torch.equal(fq.reference(x, s).view(torch.uint8), _eager_codes(x, s).view(torch.uint8))
    lin = _linear()
    assert torch.equal(
        lin._quantize_static(x, fast = True).view(torch.uint8), _eager_codes(x, s).view(torch.uint8)
    )


def test_declines_off_cuda_and_when_switched_off(monkeypatch):
    x = torch.randn(4, 32, dtype = torch.bfloat16)
    s = torch.tensor([0.5], dtype = torch.float32)
    assert fq.quantize_static(x, s) is None  # CPU
    monkeypatch.setenv(fq.FAST_QUANT_ENV, "0")
    assert not fq.enabled()
    monkeypatch.setenv(fq.FAST_QUANT_ENV, "1")
    assert fq.enabled()


def test_cpu_linear_output_unchanged():
    lin = _linear()
    x = torch.randn(2, 5, 32, dtype = torch.bfloat16) * 3
    calls = fq.describe()["calls"]
    y = lin(x)
    assert fq.describe()["calls"] == calls
    xq = _eager_codes(x.reshape(-1, 32), torch.tensor([0.0123]))
    ref = torch.nn.functional.linear(
        (xq.to(torch.float32) * 0.0123).to(torch.bfloat16),
        lin.dequantized_weight(torch.bfloat16),
        lin.bias.to(torch.bfloat16),
    ).reshape(2, 5, 48)
    assert torch.equal(y, ref)


def test_fast_flag_follows_the_gemm(monkeypatch):
    lin = _linear()
    seen = []
    orig = lin._quantize_static

    def spy(x2, fast = False):
        seen.append(fast)
        return orig(x2, fast = fast)

    monkeypatch.setattr(lin, "_quantize_static", spy)
    lin(torch.randn(3, 32, dtype = torch.bfloat16))
    assert seen == [False]  # CPU: no W8A8 GEMM, so the codes stay on the eager formula
    monkeypatch.setattr(lin, "_gemm", lambda x2: True)
    monkeypatch.setattr(
        torch, "_scaled_mm", lambda *a, **k: torch.zeros(3, 48, dtype = torch.bfloat16)
    )
    lin(torch.randn(3, 32, dtype = torch.bfloat16))
    assert seen == [False, True]


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 9)),
    reason = "needs CUDA sm_89+",
)
def test_cuda_codes_byte_identical():
    dev = torch.device("cuda", 0)
    assert fq.ready(dev), fq.describe()
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        x = (torch.randn(1031, 2048, device = dev) * 40).to(dt)
        x[0, :4] = torch.tensor([float("nan"), -float("nan"), float("inf"), -float("inf")])
        s = torch.tensor([0.0371], device = dev)
        assert torch.equal(
            fq.quantize_static(x, s).view(torch.uint8), fq.reference(x, s).view(torch.uint8)
        )


def _fake_cards(monkeypatch, capturing_on):
    """Card 0 current; ``is_current_stream_capturing`` answers for whichever card is selected."""
    import contextlib

    selected = [0]

    @contextlib.contextmanager
    def device(index):
        prev, selected[0] = selected[0], int(getattr(index, "index", index))
        try:
            yield
        finally:
            selected[0] = prev

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: selected[0] in capturing_on
    )


def test_cold_check_never_runs_inside_a_capture(monkeypatch):
    # The cold check syncs the host; inside a recording it must not run, and no verdict (pass or fail) is recorded,
    # so the first eager call outside the capture still verifies the card.
    monkeypatch.setattr(fq, "_STATE", {"verified": {}, "calls": 0, "reason": ""})
    monkeypatch.setattr(fq, "triton", object())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    calls = []

    def verify(device):
        calls.append(device)
        if capture_on:
            raise RuntimeError("operation not permitted when stream is capturing")
        return True

    monkeypatch.setattr(fq, "_verify", verify)
    capture_on = {1}
    _fake_cards(monkeypatch, capture_on)
    dev = torch.device("cuda", 1)
    assert not fq.ready(dev)  # recording on the operand's card while card 0 is current
    assert calls == [] and fq._STATE["verified"] == {}
    capture_on.clear()
    assert fq.ready(dev) and len(calls) == 1 and fq._STATE["verified"] == {1: True}


def test_gemm_probe_never_runs_inside_a_capture(monkeypatch):
    lin = _linear()
    probed = []
    monkeypatch.setattr(mo, "_GEMM_PROBE", {})
    monkeypatch.setattr(mo, "fp8_gemm_runs", lambda **k: probed.append(k))
    _fake_cards(monkeypatch, {1})

    class _X:
        is_cuda = True
        shape = (4, 32)
        device = torch.device("cuda", 1)

    from types import SimpleNamespace

    holder = SimpleNamespace(weight_q = SimpleNamespace(is_cuda = True), act = lin.act)
    assert type(lin)._gemm(holder, _X()) is False
    assert probed == [] and mo._GEMM_PROBE == {}
