# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVFP4 weight quantisers (max6 / 4over6): bit-identity with torchao, layout, round trip, patch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nvfp4_tensor = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")

from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, pack_uint4, unpack_uint4
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import from_blocked

from core.inference import diffusion_nvfp4_weight_quant as wq


def _weight(
    n = 256,
    k = 512,
    seed = 0,
    dtype = torch.bfloat16,
):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator = g) * (torch.rand(n, 1, generator = g) * 0.1)
    w[0, :16] = 0
    w[1, 16:32] = -1e-7
    w[2, 3] = 0.9
    return w.to(dtype)


def _torchao_weight(w):
    return NVFP4Tensor.to_nvfp4(
        w,
        per_tensor_scale = per_tensor_amax_to_scale(torch.max(torch.abs(w))),
        is_swizzled_scales = True,
        act_quant_kwargs = QuantizeTensorToNVFP4Kwargs(is_swizzled_scales = True),
    )


def test_e2m1_round_and_pack_match_torchao():
    x = torch.linspace(-7.0, 7.0, 28001)
    x = torch.cat(
        [x, torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -5.0, -0.0, 0.0])]
    )
    x = x[: x.numel() // 2 * 2]
    want = pack_uint4(f32_to_f4_unpacked(x.clamp(-6, 6)))
    assert torch.equal(wq.pack_e2m1(wq.e2m1_round(x)), want)
    assert torch.equal(wq.unpack_e2m1(want), wq.e2m1_round(x))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(256, 512), (77, 3840), (300, 2560), (16, 64)])
def test_max6_is_bit_identical_to_torchao(shape, dtype):
    w = _weight(*shape, dtype = dtype)
    ref = _torchao_weight(w)
    q = wq.quantize_nvfp4_weight(w, "max6")
    assert torch.equal(q.qdata, ref.qdata)
    got = wq.to_torchao_tensor(q, w.dtype)
    assert got.scale.shape == ref.scale.shape
    assert torch.equal(got.scale.view(torch.uint8), ref.scale.view(torch.uint8))
    assert float(q.per_tensor_scale) == float(ref.per_tensor_scale)
    assert torch.equal(got.dequantize(torch.float32), ref.dequantize(torch.float32))


def test_4over6_layout_scale_top_and_error():
    w = _weight(512, 1024)
    q = wq.quantize_nvfp4_weight(w, "4over6")
    amax = float(w.float().abs().max())
    assert float(q.per_tensor_scale) == pytest.approx(amax / (6 * 256), rel = 1e-7)
    assert float(q.scale.float().max()) <= 448.0
    assert 0 < q.blocks_at_4 < q.scale.numel()
    t = wq.to_torchao_tensor(q, w.dtype)
    assert (
        t.is_swizzled_scales and t.qdata.dtype == torch.uint8 and tuple(t.shape) == tuple(w.shape)
    )
    assert torch.equal(
        from_blocked(t.scale, 512, 1024 // 16).view(torch.uint8), q.scale.view(torch.uint8)
    )
    assert torch.equal(t.dequantize(torch.float32), wq.dequantize_nvfp4_weight(q))
    e6 = ((wq.fake_quantize_nvfp4_weight(w.float(), "max6") - w.float()) ** 2).sum()
    e46 = ((wq.dequantize_nvfp4_weight(q) - w.float()) ** 2).sum()
    assert e46 < 0.95 * e6


def test_4over6_per_block_choice_is_never_worse():
    w = _weight(128, 256).float()
    pts = wq.per_tensor_scale_for(w, "4over6")
    xb = w.reshape(128, 16, 16)
    c46, s46, _ = wq.quantize_blocks(xb, pts, "4over6")
    c6, s6, _ = wq.quantize_blocks(xb, pts, "max6")
    e46 = ((c46 * (s46.float() * pts).unsqueeze(-1) - xb) ** 2).sum(-1)
    e6 = ((c6 * (s6.float() * pts).unsqueeze(-1) - xb) ** 2).sum(-1)
    assert bool((e46 <= e6).all())


def test_activation_weighted_choice_uses_weights():
    w = _weight(64, 256, seed = 3).float()
    ew = torch.ones(256)
    ew[::16] = 1e6
    q = wq.quantize_nvfp4_weight(w, "4over6", err_weight = ew)
    dq = wq.dequantize_nvfp4_weight(q)
    plain = wq.dequantize_nvfp4_weight(wq.quantize_nvfp4_weight(w, "4over6"))
    assert ((dq - w)[:, ::16] ** 2).sum() <= ((plain - w)[:, ::16] ** 2).sum()


def test_chunked_equals_unchunked(monkeypatch):
    w = _weight(300, 512)
    full = wq.quantize_nvfp4_weight(w, "4over6")
    monkeypatch.setattr(wq, "_CHUNK_ELEMS", 512 * 7)
    part = wq.quantize_nvfp4_weight(w, "4over6")
    assert torch.equal(full.qdata, part.qdata)
    assert torch.equal(full.scale.view(torch.uint8), part.scale.view(torch.uint8))


def test_resolve_method(monkeypatch):
    monkeypatch.delenv(wq.WEIGHT_QUANT_ENV, raising = False)
    assert wq.resolve_method() == wq.DEFAULT_METHOD
    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "max6")
    assert wq.resolve_method() == "max6"
    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "0")
    assert wq.resolve_method() == "max6"
    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "4over6")
    assert wq.resolve_method() == "4over6"
    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "garbage")
    assert wq.resolve_method() == wq.DEFAULT_METHOD


def test_patch_routes_weights_only(monkeypatch):
    inner = NVFP4Tensor.to_nvfp4
    calls = []
    w = _weight(256, 512)
    ref = _torchao_weight(w).qdata

    def spy(*args, **kwargs):
        calls.append(args[6] if len(args) > 6 else kwargs.get("act_quant_kwargs"))
        return inner(*args, **kwargs)

    monkeypatch.setattr(NVFP4Tensor, "to_nvfp4", staticmethod(spy))
    wq.install_nvfp4_weight_quant_patch()
    wq.install_nvfp4_weight_quant_patch()
    assert getattr(NVFP4Tensor.to_nvfp4, "_unsloth_weight_quant", False)
    assert NVFP4Tensor.to_nvfp4._unsloth_inner is spy
    kw = QuantizeTensorToNVFP4Kwargs(is_swizzled_scales = True)
    s = per_tensor_amax_to_scale(torch.max(torch.abs(w)))

    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "4over6")
    t = NVFP4Tensor.to_nvfp4(w, 16, s, None, True, False, kw)
    want = wq.quantize_nvfp4_weight(w, "4over6")
    assert isinstance(t, NVFP4Tensor) and t.act_quant_kwargs is kw and t.is_swizzled_scales
    assert t.per_tensor_scale.shape == s.shape
    assert float(t.per_tensor_scale) == float(want.per_tensor_scale)
    assert torch.equal(t.qdata, want.qdata)
    assert calls == []

    a = NVFP4Tensor.to_nvfp4(w, 16, s, None, True, False, None)
    assert calls == [None] and torch.equal(a.qdata, ref)
    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "max6")
    m = NVFP4Tensor.to_nvfp4(w, 16, s, None, True, False, kw)
    assert len(calls) == 2 and torch.equal(m.qdata, ref)
    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "4over6")
    NVFP4Tensor.to_nvfp4(w, 16, s, None, False, False, kw)
    assert len(calls) == 3
    wq.uninstall_nvfp4_weight_quant_patch()
    assert NVFP4Tensor.to_nvfp4 is spy


def test_prequant_safetensors_round_trip(tmp_path):
    pytest.importorskip("safetensors")
    from core.inference.prequant_safetensors import (
        safetensors_prequant_supported,
        save_prequant_safetensors,
    )

    if not safetensors_prequant_supported():
        pytest.skip("this torchao cannot write NVFP4 safetensors")
    from core.inference.diffusion_prequant import _load_prequant_checkpoint

    w = _weight(256, 512)
    sd = {
        "a.weight": wq.to_torchao_tensor(wq.quantize_nvfp4_weight(w, "4over6"), w.dtype),
        "b.weight": wq.to_torchao_tensor(wq.quantize_nvfp4_weight(w, "max6"), w.dtype),
    }
    path = tmp_path / "t.safetensors"
    save_prequant_safetensors(str(path), fmt = "nvfp4", state_dict = sd, metadata = {"scheme": "nvfp4"})
    ck = _load_prequant_checkpoint(str(path), scheme = "nvfp4")
    for key, want in sd.items():
        got = ck["state_dict"][key]
        assert type(got).__name__ == "NVFP4Tensor" and got.is_swizzled_scales
        assert torch.equal(got.qdata.view(torch.uint8), want.qdata)
        assert torch.equal(
            got.scale.reshape(-1).view(torch.uint8), want.scale.reshape(-1).view(torch.uint8)
        )
        assert float(got.per_tensor_scale) == float(want.per_tensor_scale)


def test_make_quant_config_installs_the_weight_patch():
    from core.inference.diffusion_transformer_quant import _make_quant_config
    try:
        _make_quant_config("nvfp4")
        assert getattr(NVFP4Tensor.to_nvfp4, "_unsloth_weight_quant", False)
    finally:
        wq.uninstall_nvfp4_weight_quant_patch()


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(512, 512)
        self.b = torch.nn.Linear(512, 528)
        self.c = torch.nn.Linear(528, 512)

    def forward(self, x):
        return self.c(torch.relu(self.b(self.a(x))))


@pytest.mark.parametrize("method", ["max6", "4over6"])
def test_an_all_zero_weight_round_trips_to_zeros(method):
    w = torch.zeros(64, 128, dtype = torch.bfloat16)
    q = wq.quantize_nvfp4_weight(w, method)
    assert float(q.per_tensor_scale) > 0
    out = wq.dequantize_nvfp4_weight(q)
    assert torch.isfinite(out).all() and not out.any()
