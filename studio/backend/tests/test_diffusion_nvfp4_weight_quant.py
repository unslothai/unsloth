# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVFP4 weight quantisers (max6 / 4over6 / GPTQ): bit-identity with torchao, layout, round trip, patch."""

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
    w[0, :16] = 0  # an all-zero block
    w[1, 16:32] = -1e-7  # a block that rounds to (signed) zero
    w[2, 3] = 0.9  # an outlier that sets the tensor amax
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
    # the largest block's /4 candidate stays inside E4M3, never saturates
    assert float(q.scale.float().max()) <= 448.0
    assert 0 < q.blocks_at_4 < q.scale.numel()
    t = wq.to_torchao_tensor(q, w.dtype)
    assert (
        t.is_swizzled_scales and t.qdata.dtype == torch.uint8 and tuple(t.shape) == tuple(w.shape)
    )
    # swizzle round trip: unswizzled scales equal what the quantiser chose
    assert torch.equal(
        from_blocked(t.scale, 512, 1024 // 16).view(torch.uint8), q.scale.view(torch.uint8)
    )
    # torchao's own dequantize == the documented kernel math
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
    ew[::16] = 1e6  # one dominant column per block: the choice must minimise that column's error
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


def test_gptq_beats_rtn_on_output_error_and_is_deterministic():
    g = torch.Generator().manual_seed(1)
    k, n = 256, 128
    x = torch.randn(2048, k, generator = g) @ (torch.eye(k) + 0.3 * torch.randn(k, k, generator = g))
    w = torch.randn(n, k, generator = g) * 0.05
    H = x.t() @ x / x.shape[0]
    for method in ("max6", "4over6"):
        q = wq.gptq_nvfp4(w, H, method)
        q2 = wq.gptq_nvfp4(w, H, method)
        assert torch.equal(q.qdata, q2.qdata) and torch.equal(
            q.scale.view(torch.uint8), q2.scale.view(torch.uint8)
        )
        rtn = wq.quantize_nvfp4_weight(w, method)
        out = lambda d: float((((d - w) @ H) * (d - w)).sum())
        assert out(wq.dequantize_nvfp4_weight(q)) < 0.8 * out(wq.dequantize_nvfp4_weight(rtn))
        t = wq.to_torchao_tensor(q, torch.bfloat16)
        assert torch.equal(t.dequantize(torch.float32), wq.dequantize_nvfp4_weight(q))


def test_gptq_identity_hessian_with_max6_equals_rtn():
    w = _weight(64, 128).float()
    q = wq.gptq_nvfp4(w, torch.eye(128), "max6", damp = 0.0)
    rtn = wq.quantize_nvfp4_weight(w, "max6")
    assert torch.equal(q.qdata, rtn.qdata)
    assert torch.equal(q.scale.view(torch.uint8), rtn.scale.view(torch.uint8))


def test_rotate_hessian_matches_explicit():
    from core.inference.diffusion_convrot import build_convrot_hadamard

    g = torch.Generator().manual_seed(2)
    x = torch.randn(300, 512, generator = g, dtype = torch.float64)
    h = build_convrot_hadamard(256, dtype = torch.float64)
    xr = (x.reshape(-1, 2, 256) @ h).reshape(300, 512)
    want = xr.t() @ xr
    got = wq.rotate_hessian(x.t() @ x, 256, h)
    assert torch.allclose(got, want, atol = 1e-8)


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
    ref = _torchao_weight(w).qdata  # torchao's own max/6, before the patch

    def spy(*args, **kwargs):
        calls.append(args[6] if len(args) > 6 else kwargs.get("act_quant_kwargs"))
        return inner(*args, **kwargs)

    monkeypatch.setattr(NVFP4Tensor, "to_nvfp4", staticmethod(spy))
    wq.install_nvfp4_weight_quant_patch()
    wq.install_nvfp4_weight_quant_patch()  # idempotent
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
    assert calls == []  # the weight never reached the inner quantiser

    # activation-style call (no act_quant_kwargs) and max6 both go to the inner quantiser unchanged
    a = NVFP4Tensor.to_nvfp4(w, 16, s, None, True, False, None)
    assert calls == [None] and torch.equal(a.qdata, ref)
    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "max6")
    m = NVFP4Tensor.to_nvfp4(w, 16, s, None, True, False, kw)
    assert len(calls) == 2 and torch.equal(m.qdata, ref)
    # unswizzled layout is left to torchao
    monkeypatch.setenv(wq.WEIGHT_QUANT_ENV, "4over6")
    NVFP4Tensor.to_nvfp4(w, 16, s, None, False, False, kw)
    assert len(calls) == 3
    wq.uninstall_nvfp4_weight_quant_patch()
    assert NVFP4Tensor.to_nvfp4 is spy


def test_replace_payload_keeps_fields():
    w = _weight(256, 512)
    rtn = _torchao_weight(w)
    q = wq.quantize_nvfp4_weight(w, "4over6")
    t = wq.replace_payload(rtn, q)
    assert t.act_quant_kwargs is rtn.act_quant_kwargs and t.dtype == rtn.dtype
    assert t.scale.shape == rtn.scale.shape and t.qdata.shape == rtn.qdata.shape
    assert torch.equal(t.dequantize(torch.float32), wq.dequantize_nvfp4_weight(q))


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
    H = torch.eye(512) + 0.1
    sd = {
        "a.weight": wq.to_torchao_tensor(wq.quantize_nvfp4_weight(w, "4over6"), w.dtype),
        "b.weight": wq.to_torchao_tensor(wq.gptq_nvfp4(w, H, "4over6"), w.dtype),
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


def test_gptq_build_solves_rotates_and_swaps_payloads():
    from core.inference import diffusion_nvfp4_gptq_build as gb
    from core.inference.diffusion_convrot import rotate_linears_

    torch.manual_seed(0)
    m = _Tiny().to(torch.bfloat16)
    g = torch.Generator().manual_seed(4)
    xs = [torch.randn(64, 512, generator = g).to(torch.bfloat16) for _ in range(3)]
    calls = []

    def render_all():
        calls.append(1)
        for x in xs:
            m(x)

    logs = []
    # budget for one 512x512 H per pass -> several passes; "a" is solved in the ConvRot basis
    solved = gb.calibrate_and_solve(
        m,
        ["a", "b", "c"],
        render_all,
        method = "4over6",
        rotated = ["a"],
        group = 256,
        budget_bytes = 512 * 512 * 4 + 1,
        log = logs.append,
    )
    assert set(solved) == {"a", "b", "c"} and len(calls) == 3
    # the rotated solve matches a solve on the rotated weight with the rotated-input Hessian
    from core.inference.diffusion_convrot import build_convrot_hadamard

    h = build_convrot_hadamard(256)
    x = torch.cat(xs).float()
    xr = (x.reshape(-1, 2, 256) @ h).reshape(-1, 512)
    wr = gb._rotated_copy(m.a.weight, 256)
    want = wq.gptq_nvfp4(wr, xr.t() @ xr / xr.shape[0], "4over6")
    # R^T H R vs H of the rotated inputs differ only in fp32 summation order; GPTQ amplifies that a little
    assert (
        solved["a"].scale.view(torch.uint8) == want.scale.view(torch.uint8)
    ).float().mean() > 0.97
    assert (solved["a"].qdata == want.qdata).float().mean() > 0.95
    plain = gb.calibrate_and_solve(m, ["a"], render_all, method = "4over6", log = logs.append)["a"]
    assert (
        plain.scale.view(torch.uint8) == want.scale.view(torch.uint8)
    ).float().mean() < 0.5  # basis matters

    # quantize_ stand-in: torchao weights exactly as the runtime builds them, after the build's rotation
    rotate_linears_(m, ["a"], 256)
    assert torch.equal(m.a.weight.data, wr)
    kw = QuantizeTensorToNVFP4Kwargs(is_swizzled_scales = True)
    for name in ("a", "b", "c"):
        lin = getattr(m, name)
        w = lin.weight.data
        t = NVFP4Tensor.to_nvfp4(
            w,
            per_tensor_scale = per_tensor_amax_to_scale(torch.max(torch.abs(w))),
            is_swizzled_scales = True,
            act_quant_kwargs = kw,
        )
        lin.weight = torch.nn.Parameter(t, requires_grad = False)
    assert gb.apply_gptq_payloads(m, solved) == 3
    for name in ("a", "b", "c"):
        t = getattr(m, name).weight
        assert type(t).__name__ == "NVFP4Tensor" and t.act_quant_kwargs is kw
        assert torch.equal(t.dequantize(torch.float32), wq.dequantize_nvfp4_weight(solved[name]))


def test_gptq_build_refuses_unreached_and_non_nvfp4_layers():
    from core.inference import diffusion_nvfp4_gptq_build as gb

    m = _Tiny().to(torch.bfloat16)
    with pytest.raises(RuntimeError, match = "never reached"):
        gb.calibrate_and_solve(m, ["a"], lambda: None, method = "4over6", log = lambda s: None)
    q = wq.quantize_nvfp4_weight(m.a.weight, "4over6")
    with pytest.raises(RuntimeError, match = "not NVFP4Tensor"):
        gb.apply_gptq_payloads(m, {"a": q})
