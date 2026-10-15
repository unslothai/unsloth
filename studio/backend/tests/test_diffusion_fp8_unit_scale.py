# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sm_89 fp8 unit-scale dispatch: install gating, kill switch, fallback, and the epilogue math (CPU, emulated GEMM)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization.quantize_.workflows.float8 import float8_tensor as f8t  # noqa: E402

from core.inference import diffusion_fp8_unit_scale as us  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    us.uninstall()
    monkeypatch.delenv(us.UNIT_SCALE_ENV, raising = False)
    monkeypatch.delenv(us.MIN_ROWS_ENV, raising = False)
    for k in us._COUNTS:
        us._COUNTS[k] = 0
    original = f8t.addmm_float8_unwrapped_inference
    yield
    us.uninstall()
    f8t.addmm_float8_unwrapped_inference = original


def _caps(monkeypatch, caps):
    monkeypatch.setattr(us, "_device_caps", lambda: dict(caps))


def test_kill_switch_never_installs(monkeypatch):
    _caps(monkeypatch, {0: (8, 9)})
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "0")
    before = f8t.addmm_float8_unwrapped_inference
    assert us.install() is False
    assert f8t.addmm_float8_unwrapped_inference is before


@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (9, 0), (10, 0), (12, 0)])
def test_auto_leaves_other_arches_untouched(monkeypatch, cap):
    _caps(monkeypatch, {0: cap})
    before = f8t.addmm_float8_unwrapped_inference
    assert us.install() is False
    assert f8t.addmm_float8_unwrapped_inference is before
    assert us.stats()["installed"] is False


def test_auto_installs_on_ada_idempotent_and_reversible(monkeypatch):
    _caps(monkeypatch, {0: (8, 9)})
    before = f8t.addmm_float8_unwrapped_inference
    assert us.install() is True
    wrapped = f8t.addmm_float8_unwrapped_inference
    assert wrapped is not before and wrapped._unsloth_orig is before
    assert us.install() is True
    assert f8t.addmm_float8_unwrapped_inference is wrapped
    us.uninstall()
    assert f8t.addmm_float8_unwrapped_inference is before


def _operands(
    m,
    k = 64,
    n = 48,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator = g) * 3).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator = g) * 3).to(torch.float8_e4m3fn).t()
    sa = torch.rand(m, 1, generator = g) + 0.1
    sb = torch.rand(1, n, generator = g) + 0.1
    return a, sa, b, sb


def test_non_cuda_calls_fall_back_to_torchao(monkeypatch):
    _caps(monkeypatch, {0: (8, 9)})
    seen = []
    monkeypatch.setattr(
        f8t, "addmm_float8_unwrapped_inference", lambda *a, **k: seen.append((a, k)) or "orig"
    )
    assert us.install() is True
    a, sa, b, sb = _operands(2048)
    out = f8t.addmm_float8_unwrapped_inference(a, sa, b, sb, torch.bfloat16, use_fast_accum = True)
    assert out == "orig" and len(seen) == 1 and seen[0][1]["use_fast_accum"] is True
    assert us.stats()["unit_scale"] == 0 and us.stats()["rowwise"] == 1


class _FakeDevice:
    type = "cuda"
    index = 0


class _Dev:
    def __init__(self, t):
        self.t = t
        self.device = _FakeDevice()
        self.shape = t.shape

    def dim(self):
        return self.t.dim()


def _emulated_scaled_mm(
    a,
    b,
    scale_a,
    scale_b,
    out_dtype,
    use_fast_accum = False,
    **_,
):
    a = a.t if isinstance(a, _Dev) else a
    acc = (a.float() * scale_a.float()) @ (b.float() * scale_b.float())
    return acc.to(out_dtype)


@pytest.mark.parametrize("bias", [False, True])
def test_epilogue_matches_rowwise_math(monkeypatch, bias):
    _caps(monkeypatch, {0: (8, 9)})
    rowwise_calls = []

    def rowwise(
        a,
        sa,
        b,
        sb,
        output_dtype,
        output_scale = None,
        bias = None,
        use_fast_accum = False,
    ):
        rowwise_calls.append(a.shape[0])
        a = a.t if isinstance(a, _Dev) else a
        out = (a.float() @ b.float()) * sa * sb
        return (out + bias.float() if bias is not None else out).to(output_dtype)

    monkeypatch.setattr(f8t, "addmm_float8_unwrapped_inference", rowwise)
    monkeypatch.setattr(torch, "_scaled_mm", _emulated_scaled_mm)
    monkeypatch.setattr(
        torch, "ones", lambda *a, device = None, dtype = None: torch.tensor(1.0, dtype = dtype)
    )
    monkeypatch.setenv(us.MIN_ROWS_ENV, "1024")
    assert us.install() is True
    fn = f8t.addmm_float8_unwrapped_inference
    a, sa, b, sb = _operands(2048)
    bv = torch.randn(48).to(torch.bfloat16) if bias else None
    got = fn(_Dev(a), sa, b, sb, torch.bfloat16, bias = bv)
    want = rowwise(a, sa, b, sb, torch.bfloat16, bias = bv)
    assert us.stats()["unit_scale"] == 1
    # One extra bf16 rounding of the unscaled accumulator (bias adds after it, so measure against the output range).
    err = (got.float() - want.float()).abs().max()
    assert err <= 2**-7 * want.float().abs().max()
    assert torch.isfinite(got).all()

    small = _operands(256)
    fn(_Dev(a), sa, b, sb, torch.bfloat16, use_fast_accum = True)
    fn(_Dev(small[0]), small[1], small[2], small[3], torch.bfloat16)
    fn(_Dev(a), sa, b, sb, torch.float32)
    fn(_Dev(a), sa, b, sb, torch.bfloat16, output_scale = torch.tensor(1.0))
    assert us.stats()["unit_scale"] == 1 and len(rowwise_calls) == 5


def test_min_rows_env(monkeypatch):
    monkeypatch.setenv(us.MIN_ROWS_ENV, "4096")
    assert us._min_rows() == 4096
    monkeypatch.setenv(us.MIN_ROWS_ENV, "junk")
    assert us._min_rows() == us._DEFAULT_MIN_ROWS


@pytest.mark.parametrize("container", ["pickle", "safetensors"])
def test_every_prequant_container_installs_the_dispatch(monkeypatch, container):
    # The pickle container and the safetensors rebuild-off path never reach prequant_native's torchao API.
    from core.inference import diffusion_fp8_unit_scale as us
    from core.inference import diffusion_prequant as dp
    from core.inference import prequant_safetensors as ps

    calls = []
    monkeypatch.setattr(us, "install", lambda logger = None: calls.append(container) or True)
    monkeypatch.setattr(ps, "is_safetensors_checkpoint", lambda path: container == "safetensors")
    monkeypatch.setattr(ps, "load_prequant_safetensors", lambda path, mmap = False: {"format": "fp8"})
    monkeypatch.setattr(dp, "_torch_load_prequant", lambda path, **kwargs: {"format": "fp8"})
    assert dp._load_prequant_checkpoint("model.bin", map_location = "cpu") == {"format": "fp8"}
    assert calls == [container]
