# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""MXFP4 `mse` shared-exponent search and the magnitude-domain NVFP4 4over6 pick in unsloth.kernels.fp4_quantize
(CPU)."""

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_g_fp4_quantize_search", os.path.join(ROOT, "unsloth/kernels/fp4_quantize.py")
)
fq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fq)

_E2M1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))


def _decode(packed, scale, global_scale):
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1)
    values = _E2M1[(codes & 7).long()] * torch.where((codes & 8).bool(), -1.0, 1.0)
    return values * (scale.float() / global_scale.float()).repeat_interleave(16, 1)


def _values(codes):
    v = _E2M1[(codes & 7).long()]
    return torch.where((codes & 8).bool(), -v, v)


def _weight(
    N = 96,
    K = 512,
    seed = 0,
    dtype = torch.bfloat16,
):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator = g) * torch.logspace(-2.5, -1, K)
    w[0, :3] *= 40
    w[3, :32] = 0
    return w.to(dtype)


def _block_err(w, packed, scale, g, group):
    return ((_decode(packed, scale, g) - w.float()) ** 2).view(w.shape[0], -1, group).sum(-1)


def _bytes(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def _legacy_4over6(w):
    """4over6 as written before the magnitude-domain pick (signed codes, both candidates rounded)."""
    x = w.float()
    N, K = x.shape
    amax = x.abs().amax()
    g = torch.where(amax > 0, (256.0 * 6.0) / amax, torch.ones_like(amax)).clamp(max = 2.0**127)
    blocks = x.view(N, K // 16, 16)
    bmax = blocks.abs().amax(-1)
    out = []
    for top in (6.0, 4.0):
        s = (bmax / top * g).clamp(max = 448.0).to(torch.float8_e4m3fn)
        d = s.float() / g
        q = fq._round_e2m1(
            torch.where(d[..., None] > 0, blocks / d[..., None], torch.zeros_like(blocks))
        )
        out.append((q, s, ((_values(q) * d[..., None] - blocks) ** 2).sum(-1)))
    (q6, s6, e6), (q4, s4, e4) = out
    take4 = e4 < e6
    q = torch.where(take4[..., None], q4, q6)
    s = torch.where(take4, s4.view(torch.uint8), s6.view(torch.uint8))
    return fq._pack_nibbles(q.reshape(N, K)), s, g.reshape(1)


def _mx_full_search(w):
    """Reference MXFP4 `mse` without the clipping shortcut: every block tries e and e + 1 (ties keep e)."""
    x = w.float()
    N, K = x.shape
    b = x.view(N, K // 32, 32)
    bmax = b.abs().amax(-1)
    e = torch.floor(torch.log2(bmax.clamp(min = 2.0**-126))) - 2
    e = torch.where(bmax > 0, e, torch.full_like(e, -127.0)).clamp(-127, 127)
    errs = []
    for ee in (e, (e + 1).clamp(max = 127)):
        sc = torch.exp2(ee)[..., None]
        errs.append(((_values(fq._round_e2m1(b / sc)) * sc - b) ** 2).sum(-1))
    return torch.where(errs[1] < errs[0], (e + 1).clamp(max = 127), e), bmax > 0


def test_method_resolution(monkeypatch):
    monkeypatch.delenv(fq.MXFP4_WEIGHT_QUANT_ENV, raising = False)
    assert fq.mxfp4_weight_quant_method() == "mse"
    for value, want in (
        ("mse", "mse"),
        ("SEARCH", "mse"),
        ("rtn", "rtn"),
        ("OCP", "rtn"),
        ("0", "rtn"),
    ):
        assert fq.mxfp4_weight_quant_method(value) == want
    monkeypatch.setenv(fq.MXFP4_WEIGHT_QUANT_ENV, "rtn")
    assert fq.mxfp4_weight_quant_method() == "rtn"
    assert fq.mxfp4_weight_quant_method("mse") == "mse"  # the argument wins
    with pytest.raises(ValueError, match = "rtn, mse"):
        fq.mxfp4_weight_quant_method("4over6")
    with pytest.raises(ValueError):
        fq.nvfp4_weight_quant_method("mse")  # NVFP4 keeps max6 / 4over6 only


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nvfp4_4over6_bit_identical_to_the_signed_pick(seed):
    w = _weight(seed = seed)
    w[5, 16:32] = (
        w[5, 16:32].float().mul(2.0**-40).to(w.dtype)
    )  # a block near the e4m3 subnormal range
    packed, scale, g = fq.nvfp4_quantize_weight(w, "4over6")
    ref_packed, ref_scale, ref_g = _legacy_4over6(w)
    assert torch.equal(packed, ref_packed)
    assert torch.equal(scale.view(torch.uint8), ref_scale)
    assert torch.equal(g, ref_g)


@pytest.mark.parametrize("spread", ["narrow", "wide"])
def test_mxfp4_mse_matches_the_full_search_and_never_loses(spread):
    w = _weight(N = 64, K = 256, dtype = torch.float32)
    if spread == "wide":
        w[7, 32:64] *= 2.0**-30  # past e4m3's power-of-two span: bfloat16 scales
    rtn = fq.mxfp4_quantize_weight(w, "rtn")
    mse = fq.mxfp4_quantize_weight(w, "mse")
    assert (
        mse[1].dtype
        == rtn[1].dtype
        == (torch.float8_e4m3fn if spread == "narrow" else torch.bfloat16)
    )
    want, live = _mx_full_search(w)
    got = (mse[1].float() / mse[2])[:, ::2]
    assert torch.equal(got[live], torch.exp2(want)[live])  # all-zero blocks take any scale
    e_rtn, e_mse = _block_err(w, *rtn, 32), _block_err(w, *mse, 32)
    assert (e_mse <= e_rtn).all()
    assert e_mse.sum() < 0.97 * e_rtn.sum()
    ratio = got / (rtn[1].float() / rtn[2])[:, ::2]
    assert set(ratio[live].unique().tolist()) == {
        1.0,
        2.0,
    }  # only ever the next exponent up, and it occurs


def test_mxfp4_row_chunks_match_one_pass(monkeypatch):
    w = _weight(N = 40, K = 256)
    whole = fq.mxfp4_quantize_weight(w, "mse")
    monkeypatch.setattr(fq, "_ROW_CHUNK", 7)
    chunked = fq.mxfp4_quantize_weight(w, "mse")
    assert all(torch.equal(_bytes(a), _bytes(b)) for a, b in zip(whole, chunked))


def test_mxfp4_env_reaches_fp4_quantize_weight(monkeypatch):
    w = _weight(N = 64, K = 256)
    monkeypatch.setenv(fq.MXFP4_WEIGHT_QUANT_ENV, "rtn")
    rtn = fq.fp4_quantize_weight(w, "mxfp4", "4over6")  # an NVFP4 rule name: MXFP4 ignores it
    monkeypatch.delenv(fq.MXFP4_WEIGHT_QUANT_ENV)
    default = fq.fp4_quantize_weight(w, "mxfp4")
    assert all(
        torch.equal(_bytes(a), _bytes(b)) for a, b in zip(rtn, fq.mxfp4_quantize_weight(w, "rtn"))
    )
    assert all(
        torch.equal(_bytes(a), _bytes(b))
        for a, b in zip(default, fq.mxfp4_quantize_weight(w, "mse"))
    )
    assert not torch.equal(_bytes(rtn[1]), _bytes(default[1]))


def test_mxfp4_mse_matches_unsloth_zoo_expert_quantizer():
    pe = pytest.importorskip("unsloth_zoo.packed_experts")
    if not hasattr(pe, "mxfp4_weight_quant_method"):
        pytest.skip("unsloth_zoo without the MXFP4 exponent search")
    w = _weight(N = 64, K = 256)
    w[3, :32] = 1e-3  # no all-zero block: those take a different filler scale on each side
    packed, scale, g = fq.mxfp4_quantize_weight(w, "mse")
    ref = pe.quantize_experts_mxfp4(w[None], method = "mse")
    exps = ref.mxfp4_scales[0].to(torch.float32) - 127  # zoo stores e8m0 bytes per 32
    assert torch.equal(scale.float()[:, ::2] / g, torch.exp2(exps))
    assert torch.equal(packed, ref.data[0].reshape(64, -1))


def test_tiny_magnitude_weights_keep_finite_scales_with_search():
    w = torch.zeros(4, 64, dtype = torch.bfloat16)
    w[0, 0] = 2.0**-118
    w[1, 37] = -(2.0**-119)
    packed, scale, global_scale = fq.mxfp4_quantize_weight(w, "mse")
    W = _decode(packed, scale, global_scale)
    assert torch.isfinite(global_scale).all() and torch.isfinite(W).all()
    assert torch.equal(W, w.float())


def test_e2m1_magnitude_equals_the_code_round_trip():
    # Every midpoint (the ties), its fp32 neighbours, the grid, saturation and random values.
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    pts = torch.cat([
        mids, _E2M1, torch.nextafter(mids, torch.tensor(9.0)), torch.nextafter(mids, torch.tensor(-9.0)),
        torch.rand(200_000, generator = torch.Generator().manual_seed(0)) * 8,
        torch.tensor([6.0001, 7.0, 1e30, float("inf"), 0.0, 1e-45]),
    ])  # fmt: skip
    assert torch.equal(fq._e2m1_magnitude(pts), fq._e2m1_to_float(fq._round_e2m1(pts)))
