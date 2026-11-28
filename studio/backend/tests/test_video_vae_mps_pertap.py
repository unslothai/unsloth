# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Apple MPS per-tap Conv3d for video VAEs, on CPU with the device gate pointed at "cpu": equals F.conv3d in fp64 /
fp32, stays inside stock MPS's own bf16 / fp16 error, leaves every VAE's causal padding (zero, replicate, reflect,
feature cache) bitwise unchanged, chunks every op under the bound, and gates on MPS, torch < 2.14, the 2.14 shape
rule, the size guard and the kill switch."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from core.inference import video_vae_mps_pertap as P  # noqa: E402

_needs_diffusers = pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None, reason = "needs diffusers"
)

# (cin, cout, kernel, stride, padding, dilation, groups, T, H, W)
GEOMETRIES = {
    "causal_k3": (16, 24, (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1), 1, 7, 10, 12),
    "k1_shortcut": (16, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, 5, 9, 11),
    "k3_tpad1": (8, 12, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1, 6, 8, 8),
    "k3_tpad2_t1": (8, 12, (3, 3, 3), (1, 1, 1), (2, 1, 1), (1, 1, 1), 1, 1, 8, 8),
    "stride_t2_s2_pad": (8, 8, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), 1, 9, 12, 10),
    "wan_time_conv": (8, 16, (3, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, 5, 6, 6),
    "wan_time_down": (8, 8, (3, 1, 1), (2, 1, 1), (0, 0, 0), (1, 1, 1), 1, 7, 6, 6),
    "dilation_t2_pad": (8, 8, (3, 3, 3), (1, 1, 1), (2, 1, 1), (2, 1, 1), 1, 9, 8, 8),
    "dilation_hw2": (8, 8, (3, 3, 3), (1, 1, 1), (0, 2, 2), (1, 2, 2), 1, 5, 9, 9),
    "groups2": (8, 12, (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1), 2, 6, 8, 8),
    "pad_over_kernel": (4, 4, (2, 3, 3), (1, 1, 1), (3, 1, 1), (1, 1, 1), 1, 2, 5, 5),
}


def _inputs(
    geom: tuple,
    dtype,
    seed: int = 0,
    batch: int = 2,
):
    cin, cout, k, s, p, d, g, T, H, W = geom
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, cin, T, H, W, generator = gen, dtype = torch.float64)
    fan = cin // g * k[0] * k[1] * k[2]
    w = torch.randn(cout, cin // g, *k, generator = gen, dtype = torch.float64) / fan**0.5
    b = torch.randn(cout, generator = gen, dtype = torch.float64)
    return x.to(dtype), w.to(dtype), b.to(dtype)


def _rel_l2(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.fixture
def on_cpu(monkeypatch):
    """Point the per-call device gate at CPU (fp64 allowed, for the exactness oracle) and clear the env switch."""
    monkeypatch.setattr(P, "ENGAGE_DEVICE_TYPES", ("cpu",))
    monkeypatch.setattr(P, "ENGAGE_DTYPES", (*P.ENGAGE_DTYPES, "float64"))
    monkeypatch.delenv(P.PERTAP_ENV, raising = False)
    return P


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
@pytest.mark.parametrize("bias", [True, False])
def test_pertap_matches_conv3d_fp64(name, bias):
    geom = GEOMETRIES[name]
    x, w, b = _inputs(geom, torch.float64)
    b = b if bias else None
    _, _, _, s, p, d, g, *_ = geom
    ref = F.conv3d(x, w, b, s, p, d, g)
    got = P.pertap_conv3d(x, w, b, s, p, d, g)
    assert got.shape == ref.shape and got.dtype == ref.dtype and got.is_contiguous()
    assert torch.allclose(got, ref, rtol = 0, atol = 1e-12), (got - ref).abs().max()


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_pertap_matches_conv3d_fp32(name):
    geom = GEOMETRIES[name]
    x, w, b = _inputs(geom, torch.float32)
    _, _, _, s, p, d, g, *_ = geom
    ref64 = F.conv3d(x.double(), w.double(), b.double(), s, p, d, g)
    stock = F.conv3d(x, w, b, s, p, d, g)
    got = P.pertap_conv3d(x, w, b, s, p, d, g)
    # fp32 accumulation either way: per-tap sits at fp32 rounding, as close to fp64 as stock is
    assert _rel_l2(got, ref64) < 1e-6
    assert _rel_l2(got, ref64) <= 4 * _rel_l2(stock, ref64) + 1e-7


# Stock MPS Conv3d's own error vs a float32 CPU conv3d on M5 Max (Mac task 0034), max over the measured shapes.
MPS_STOCK_REL_L2 = {torch.bfloat16: 2.4e-3, torch.float16: 3e-4}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "geom",
    [
        (64, 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1), 1, 6, 12, 16),
        (128, 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1), 1, 4, 8, 8),
    ],
)
def test_pertap_half_error_within_stock_mps(dtype, geom):
    x, w, b = _inputs(geom, dtype, batch = 1)
    _, _, _, s, p, d, g, *_ = geom
    ref = F.conv3d(x.float(), w.float(), b.float(), s, p, d, g)
    got = P.pertap_conv3d(x, w, b, s, p, d, g)
    assert got.dtype == dtype and torch.isfinite(got).all()
    assert _rel_l2(got, ref) <= MPS_STOCK_REL_L2[dtype]


def test_chunking_bounds_every_op(monkeypatch):
    geom = (8, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1, 11, 12, 12)
    x, w, b = _inputs(geom, torch.float32, batch = 1)
    _, _, _, s, p, d, g, *_ = geom
    ref = F.conv3d(x, w, b, s, p, d, g)
    chunk = 3 * 16 * 12 * 12 * 4  # three fp32 output frames
    sizes = []
    real = F.conv2d

    def spy(inp, *a, **k):
        out = real(inp, *a, **k)
        sizes.append(max(inp.numel(), out.numel()) * 4)
        return out

    monkeypatch.setattr(F, "conv2d", spy)
    got = P.pertap_conv3d(x, w, b, s, p, d, g, chunk_bytes = chunk)
    assert len(sizes) > 3 * 3  # several chunks
    assert max(sizes) <= chunk
    assert torch.allclose(got, ref, atol = 1e-5, rtol = 1e-5)


def test_torch_version_tuple():
    assert P.torch_version_tuple("2.10.0") == (2, 10)
    assert P.torch_version_tuple("2.14.1") == (2, 14)
    assert P.torch_version_tuple("2.14.0.dev20261101+cpu") == (2, 14)
    assert P.torch_version_tuple("garbage") == (0, 0)


# ---- causal padding semantics: the conv sees bitwise the stock input, per VAE conv class ----


def _capture_inputs(module):
    """Record what reaches ``F.conv3d`` (stock) or the per-tap kernel (patched)."""
    seen = []
    real3d, real_pt = F.conv3d, P.pertap_conv3d

    def spy3d(x, *a, **k):
        seen.append(x.detach().clone())
        return real3d(x, *a, **k)

    def spy_pt(x, *a, **k):
        seen.append(x.detach().clone())
        return real_pt(x, *a, **k)

    return seen, spy3d, spy_pt


def _run_both(
    monkeypatch,
    make,
    call,
    dtype = torch.float32,
):
    """Stock vs patched on fresh identical modules; returns (stock out, patched out, stock inputs, patched inputs)."""
    torch.manual_seed(0)
    stock_mod = make().to(dtype).eval()
    patched_mod = make().to(dtype).eval()
    patched_mod.load_state_dict(stock_mod.state_dict())
    root = torch.nn.Module()
    root.inner = patched_mod
    assert P.install(root) >= 1
    seen_s, spy3d, spy_pt = _capture_inputs(stock_mod)
    with torch.no_grad(), monkeypatch.context() as m:
        m.setattr(F, "conv3d", spy3d)
        out_s = call(stock_mod)
    seen_p, spy3d, spy_pt = _capture_inputs(patched_mod)
    with torch.no_grad(), monkeypatch.context() as m:
        m.setattr(F, "conv3d", spy3d)
        m.setattr(P, "pertap_conv3d", spy_pt)
        out_p = call(patched_mod)
    return out_s, out_p, seen_s, seen_p, patched_mod


def _assert_same_padding(out_s, out_p, seen_s, seen_p, patched_mod, dtype):
    assert len(seen_s) == len(seen_p) >= 1
    for a, b in zip(seen_s, seen_p):
        assert a.dtype == b.dtype and torch.equal(a, b)  # padding bit-for-bit
    assert P.pertap_calls(patched_mod) >= 1
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    assert _rel_l2(out_p, out_s) < tol


@_needs_diffusers
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("with_cache", [False, True])
def test_wan_causal_conv_zero_pad_and_cache(on_cpu, monkeypatch, dtype, with_cache):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    x = torch.randn(1, 8, 4, 9, 10, dtype = dtype)
    cache = torch.randn(1, 8, 2, 9, 10, dtype = dtype) if with_cache else None
    outs = _run_both(
        monkeypatch,
        lambda: WanCausalConv3d(8, 12, 3, padding = 1),
        lambda m: m(x, cache),
        dtype,
    )
    _assert_same_padding(*outs, dtype)


@_needs_diffusers
@pytest.mark.parametrize("with_cache", [False, True])
def test_wan_mps_zero_buffer_forward(on_cpu, monkeypatch, with_cache):
    """The MPS frame-pad fix's zero-buffer forward calls Conv3d.forward, so the wrap still sees its buffer."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    from core.inference.diffusion_device import _zero_buffer_causal_conv

    x = torch.randn(1, 8, 3, 7, 6)
    cache = torch.randn(1, 8, 2, 7, 6) if with_cache else None

    def make():
        m = WanCausalConv3d(8, 8, 3, padding = 1)
        m.forward = types.MethodType(_zero_buffer_causal_conv, m)
        return m

    calls = []
    real = torch.nn.Conv3d.forward

    def spy(self, inp):
        calls.append(inp.detach().clone())
        return real(self, inp)

    out_s = None
    with monkeypatch.context() as mp:
        mp.setattr(torch.nn.Conv3d, "forward", spy)
        torch.manual_seed(0)
        stock = make().eval()
        patched = make().eval()
        patched.load_state_dict(stock.state_dict())
        root = torch.nn.Module()
        root.inner = patched
        assert P.install(root) == 1
        with torch.no_grad():
            out_s = stock(x, cache)
            out_p = patched(x, cache)
    assert len(calls) == 2 and torch.equal(calls[0], calls[1])
    assert P.pertap_calls(patched) == 1
    assert _rel_l2(out_p, out_s) < 1e-5
    assert _rel_l2(out_p, WanCausalConv3d.forward(stock, x, cache).detach()) < 1e-5


@_needs_diffusers
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["zeros", "reflect"])
@pytest.mark.parametrize("causal", [True, False])
def test_ltx2_causal_conv_replicate(on_cpu, monkeypatch, dtype, mode, causal):
    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import LTX2VideoCausalConv3d

    x = torch.randn(1, 8, 5, 8, 9, dtype = dtype)
    outs = _run_both(
        monkeypatch,
        lambda: LTX2VideoCausalConv3d(8, 12, 3, spatial_padding_mode = mode),
        lambda m: m(x, causal = causal),
        dtype,
    )
    _assert_same_padding(*outs, dtype)


@_needs_diffusers
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hunyuanvideo15_causal_conv_replicate(on_cpu, monkeypatch, dtype):
    from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
        HunyuanVideo15CausalConv3d,
    )

    x = torch.randn(1, 8, 5, 8, 9, dtype = dtype)
    outs = _run_both(
        monkeypatch,
        lambda: HunyuanVideo15CausalConv3d(8, 12, 3),
        lambda m: m(x),
        dtype,
    )
    _assert_same_padding(*outs, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_minimax_h3_cat_front_conv(on_cpu, monkeypatch, dtype):
    mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3")
    from core.inference.diffusion_device import _cat_front_h3_conv

    x = torch.randn(1, 8, 5, 8, 9, dtype = dtype)

    def make():
        return mod.MiniMaxH3VideoCausalConv3d(
            8,
            12,
            kernel_size = 3,
            spatial_padding = 1,
            temporal_padding = 2,
            spatial_padding_mode = "reflect",
        )

    torch.manual_seed(0)
    stock = make().to(dtype).eval()
    patched = make().to(dtype).eval()
    patched.load_state_dict(stock.state_dict())
    patched.forward = types.MethodType(_cat_front_h3_conv, patched)
    root = torch.nn.Module()
    root.inner = patched
    assert P.install(root) == 1
    with torch.no_grad():
        ref = stock(x)  # diffusers' own F.pad forward
        got = patched(x)
    assert P.pertap_calls(patched) == 1
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    assert _rel_l2(got, ref) < tol


# ---- whole tiny VAEs: decode (and encode) through the wrap ----


def _tiny_vaes():
    from diffusers import AutoencoderKLHunyuanVideo15, AutoencoderKLLTX2Video, AutoencoderKLWan
    return {
        "wan22_residual": (
            lambda: AutoencoderKLWan(
                base_dim = 16,
                z_dim = 4,
                dim_mult = [1, 2, 2],
                num_res_blocks = 1,
                temperal_downsample = [False, True],
                is_residual = True,
                patch_size = 2,
                in_channels = 12,
                out_channels = 12,
                latents_mean = [0.0] * 4,
                latents_std = [1.0] * 4,
            ),
            (1, 3, 5, 32, 32),
        ),
        "ltx2": (
            lambda: AutoencoderKLLTX2Video(
                latent_channels = 8,
                block_out_channels = (16, 16),
                decoder_block_out_channels = (16,),
                layers_per_block = (1, 1, 1),
                decoder_layers_per_block = (1, 1),
                spatio_temporal_scaling = (True, False),
                decoder_spatio_temporal_scaling = (True,),
                decoder_inject_noise = (False, False),
                downsample_type = ("spatiotemporal", "spatial"),
                upsample_type = ("spatiotemporal",),
                upsample_residual = (True,),
                upsample_factor = (2,),
                patch_size = 2,
            ),
            (1, 3, 9, 16, 16),
        ),
        "hunyuanvideo15": (
            lambda: AutoencoderKLHunyuanVideo15(
                latent_channels = 4,
                block_out_channels = (16, 16),
                layers_per_block = 1,
                spatial_compression_ratio = 2,
                temporal_compression_ratio = 2,
            ),
            (1, 3, 5, 8, 8),
        ),
    }


@_needs_diffusers
@pytest.mark.parametrize("name", ["wan22_residual", "ltx2", "hunyuanvideo15"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tiny_vae_roundtrip(on_cpu, name, dtype):
    make, shape = _tiny_vaes()[name]
    torch.manual_seed(0)
    try:
        stock = make().to(dtype).eval()
    except Exception as exc:  # noqa: BLE001 - diffusers config drift
        pytest.skip(f"{name} tiny config does not build on this diffusers: {exc}")
    patched = make().to(dtype).eval()
    patched.load_state_dict(stock.state_dict())
    n = P.install(patched)
    assert n == sum(isinstance(m, torch.nn.Conv3d) for m in patched.modules()) and n > 0
    x = torch.randn(*shape, dtype = dtype)
    with torch.no_grad():
        z_s = stock.encode(x).latent_dist.mode()
        z_p = patched.encode(x).latent_dist.mode()
        y_s = stock.decode(z_s).sample
        y_p = patched.decode(z_s).sample
    tol = 1e-11 if dtype == torch.float64 else 1e-5
    assert _rel_l2(z_p, z_s) < tol
    assert _rel_l2(y_p, y_s) < tol
    assert P.pertap_calls(patched) > 0


# ---- gates ----


def _conv(cin = 8, cout = 8):
    return torch.nn.Conv3d(cin, cout, 3, padding = (0, 1, 1)).eval()


def test_cpu_default_device_untouched(monkeypatch):
    monkeypatch.delenv(P.PERTAP_ENV, raising = False)
    conv = _conv()
    assert P.install(conv) == 1
    with torch.no_grad():
        conv(torch.randn(1, 8, 4, 6, 6))
    assert P.pertap_calls(conv) == 0  # the gate only engages on MPS


def test_kill_switch(on_cpu, monkeypatch):
    conv = _conv()
    assert P.install(conv) == 1
    monkeypatch.setenv(P.PERTAP_ENV, "0")
    with torch.no_grad():
        conv(torch.randn(1, 8, 4, 6, 6))
    assert P.pertap_calls(conv) == 0
    assert P.install(_conv()) == 0  # off at install too
    monkeypatch.delenv(P.PERTAP_ENV)
    with torch.no_grad():
        conv(torch.randn(1, 8, 4, 6, 6))
    assert P.pertap_calls(conv) == 1


def test_new_torch_shape_rule_and_size_guard(on_cpu, monkeypatch):
    narrow = _conv(8, 8)
    wide = _conv(P.WIDE_CHANNELS, 8)
    assert P.install(narrow, torch_version = "2.14.1") == 1
    assert P.install(wide, torch_version = "2.14.1") == 1
    with torch.no_grad():
        narrow(torch.randn(1, 8, 4, 6, 6))
        assert P.pertap_calls(narrow) == 0  # native on >= 2.14
        wide(torch.randn(1, P.WIDE_CHANNELS, 3, 4, 4))
        assert P.pertap_calls(wide) == 1  # 1024ch small frame: per-tap still wins there
        wide(torch.randn(1, P.WIDE_CHANNELS, 3, 64, 104))
        assert P.pertap_calls(wide) == 2  # 64x104 (6656 px): per-tap 1.74x on 2.14.1
        wide(torch.randn(1, P.WIDE_CHANNELS, 3, 92, 92))
        assert P.pertap_calls(wide) == 2  # past 8192 px: native
        monkeypatch.setattr(P, "SIZE_GUARD_BYTES", 8 * 4 * 6 * 6 * 4 - 1)
        narrow(torch.randn(1, 8, 4, 6, 6))
        assert P.pertap_calls(narrow) == 1  # over the size guard: chunked per-tap on any torch
        monkeypatch.setenv(P.PERTAP_ENV, "1")
        monkeypatch.setattr(P, "SIZE_GUARD_BYTES", 2**40)
        narrow(torch.randn(1, 8, 4, 6, 6))
        assert P.pertap_calls(narrow) == 2  # force


def test_old_torch_every_call(on_cpu):
    conv = _conv()
    assert P.install(conv, torch_version = "2.10.0") == 1
    with torch.no_grad():
        conv(torch.randn(1, 8, 4, 6, 6))
        conv(torch.randn(1, 8, 1, 6, 6).to(torch.bfloat16).float())
    assert P.pertap_calls(conv) == 2


def test_grad_autocast_and_dtype_stay_stock(on_cpu, monkeypatch):
    monkeypatch.setattr(
        P, "ENGAGE_DTYPES", ("bfloat16", "float16", "float32")
    )  # Metal has no float64
    conv = _conv()
    P.install(conv, torch_version = "2.10.0")
    conv(torch.randn(1, 8, 4, 6, 6))  # grad enabled, weight requires grad
    with torch.no_grad(), torch.autocast("cpu", dtype = torch.bfloat16):
        conv(torch.randn(1, 8, 4, 6, 6))
    with torch.no_grad():
        conv.double()(torch.randn(1, 8, 4, 6, 6, dtype = torch.float64))
    assert P.pertap_calls(conv) == 0


def test_install_idempotent_and_respects_foreign_patch(on_cpu):
    root = torch.nn.Sequential(_conv(), _conv())
    sentinel = root[1]._conv_forward
    root[1]._conv_forward = sentinel  # someone else's wrap, as an instance attribute
    assert P.install(root) == 1
    assert P.install(root) == 1
    assert root[1]._conv_forward is sentinel


def test_install_on_pipe_only_on_mps(monkeypatch):
    monkeypatch.delenv(P.PERTAP_ENV, raising = False)
    pipe = types.SimpleNamespace(vae = torch.nn.Sequential(_conv()))
    assert P.install_on_pipe(pipe, types.SimpleNamespace(device = "cuda")) == 0
    assert P.install_on_pipe(pipe, types.SimpleNamespace(device = "cpu")) == 0
    log = logging.getLogger("test_mps_pertap")
    assert P.install_on_pipe(pipe, types.SimpleNamespace(device = "mps"), log) == 1
    assert (
        P.install_on_pipe(types.SimpleNamespace(vae = None), types.SimpleNamespace(device = "mps")) == 0
    )


def test_video_loader_wires_install():
    src = (_BACKEND / "core" / "inference" / "video.py").read_text(encoding = "utf-8")
    assert "from .video_vae_mps_pertap import install_on_pipe as install_vae_mps_pertap" in src
    assert "speed_optims += (_MPS_PERTAP,)" in src
    assert P.OPTIM_NAME == "vae_mps_pertap_conv"
