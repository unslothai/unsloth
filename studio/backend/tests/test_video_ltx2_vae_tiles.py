# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2 / 2.3 VAE decode without tile seams (video_ltx2_vae_tiles) and the ltx-2 untiled estimate."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLLTX2Video"):
    pytest.skip("diffusers without LTX-2", allow_module_level = True)

from core.inference import video_ltx2_vae_tiles as vt  # noqa: E402
from core.inference import video_vae_untiled as U  # noqa: E402

GIB = 2**30


def _ltx_vae():
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLLTX2Video(
        in_channels = 3,
        out_channels = 3,
        latent_channels = 8,
        block_out_channels = (8, 8, 8, 8),
        decoder_block_out_channels = (8, 16, 32),
        layers_per_block = (1, 1, 1, 1, 1),
        decoder_layers_per_block = (1, 1, 1, 1),
        spatial_compression_ratio = 32,
        temporal_compression_ratio = 8,
    ).eval()
    assert vae.spatial_compression_ratio == 32
    return vae


def _line_error(x, ref):
    d = (x - ref).abs().mean(1)[0]
    return max(float(d.mean(1).max()), float(d.mean(2).max()))


def _decode(vae, z, *args, **kwargs):
    with torch.no_grad():
        return vae.decode(z, *args, return_dict = False, **kwargs)[0]


@pytest.fixture
def free(monkeypatch):
    """Set the free bytes the planner sees."""
    box = {"bytes": 0}
    monkeypatch.setattr(vt, "_free_bytes", lambda device: box["bytes"])
    monkeypatch.delenv(vt.WIDE_TILES_ENV, raising = False)
    monkeypatch.delenv(U.UNTILED_ENV, raising = False)
    return box


def test_tight_memory_tiles_have_no_seam_lines(free, monkeypatch):
    vae = _ltx_vae()
    z = torch.randn(1, 8, 2, 22, 38, generator = torch.Generator().manual_seed(1))
    vae.use_tiling = False
    untiled = _decode(vae, z)
    vae.enable_tiling()
    stock = _decode(vae, z)
    free["bytes"] = 0
    assert vt.install(vae)
    wide = _decode(vae, z)
    assert vae._unsloth_last_decode_tile == (16, 16)
    assert wide.shape == untiled.shape == stock.shape
    stock_err, wide_err = _line_error(stock, untiled), _line_error(wide, untiled)
    assert stock_err > 3 * wide_err, (stock_err, wide_err)
    monkeypatch.setenv(vt.WIDE_TILES_ENV, "0")
    assert torch.equal(_decode(vae, z), stock)


def test_whole_latent_fits_decodes_untiled(free, monkeypatch):
    vae = _ltx_vae()
    z = torch.randn(1, 8, 2, 16, 24, generator = torch.Generator().manual_seed(2))
    vae.use_tiling = False
    untiled = _decode(vae, z)
    vae.enable_tiling()
    assert vt.install(vae)
    free["bytes"] = 1000 * GIB
    assert torch.equal(_decode(vae, z), untiled)
    assert vae._unsloth_last_decode_tile == (16, 24)
    assert vae._unsloth_wide_tiles_stats["untiled"] == 1
    monkeypatch.setenv(U.UNTILED_ENV, "0")
    _decode(vae, z)
    assert vae._unsloth_last_decode_tile != (16, 24)


def test_more_free_memory_means_fewer_larger_tiles(free, monkeypatch):
    vae = _ltx_vae()
    monkeypatch.setattr(vt, "_itemsize", lambda vae: 2)
    z = torch.zeros(1, 128, 16, 22, 38)
    for fused in (False, True):
        vae._unsloth_vae_fused_installed = int(fused)
        sizes = []
        for gib in (0, 4, 6, 7, 9, 12, 20):
            tile = vt.plan_tiles(vae, z, free = gib * GIB)
            n = len(vt.tile_starts(22, tile[0])) * len(vt.tile_starts(38, tile[1]))
            need = vt.tile_bytes(121, *tile, fused = fused) * vt._MARGIN + vt._MARGIN_BYTES
            accum = 0 if n == 1 else 4 * 3 * 121 * 704 * 1216
            if tile != (16, 16):
                assert need + accum <= gib * GIB
            sizes.append(n)
        assert sizes == sorted(sizes, reverse = True) and sizes[0] > 1 and sizes[-1] == 1, sizes


def test_temb_and_causal_reach_every_tile(free):
    vae = _ltx_vae()
    vae.enable_tiling()
    assert vt.install(vae)
    seen = []
    decoder = vae.decoder
    forward = decoder.forward

    def spy(
        x,
        temb = None,
        causal = None,
    ):
        seen.append((temb, causal))
        return forward(x, temb, causal = causal)

    decoder.forward = spy
    temb = torch.tensor([0.05])
    _decode(vae, torch.randn(1, 8, 1, 16, 40), temb, causal = False)
    assert len(seen) > 1 and all(t is temb and c is False for t, c in seen)


def test_oom_retries_in_stock_size_tiles(free, monkeypatch):
    vae = _ltx_vae()
    z = torch.randn(1, 8, 1, 22, 38)
    vae.enable_tiling()
    assert vt.install(vae)
    free["bytes"] = 1000 * GIB
    calls = []
    real = vt._decode_tiles

    def flaky(
        vae_,
        z_,
        temb,
        causal,
        th,
        tw,
        fp32_accum = True,
    ):
        calls.append((th, tw, fp32_accum))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("fake")
        return real(vae_, z_, temb, causal, th, tw, fp32_accum)

    monkeypatch.setattr(vt, "_decode_tiles", flaky)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    out = _decode(vae, z)
    assert calls == [(22, 38, True), (16, 16, False)] and out.shape[-2:] == (704, 1216)
    assert vae._unsloth_wide_tiles_stats["oom_fallback"] == 1


@pytest.mark.parametrize("free_bytes, fp32_accum", [(0, False), (None, False), (1000 * GIB, True)])
def test_fp32_accumulator_only_when_budgeted(free, monkeypatch, free_bytes, fp32_accum):
    vae = _ltx_vae().to(torch.bfloat16)
    z = torch.randn(1, 8, 1, 22, 38, dtype = torch.bfloat16)
    vae.enable_tiling()
    assert vt.install(vae)
    free["bytes"] = free_bytes
    monkeypatch.setenv(U.UNTILED_ENV, "0")
    seen = []
    zeros = torch.zeros

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dtype"))
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", spy)
    out = _decode(vae, z)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    assert (torch.float32 if fp32_accum else torch.bfloat16) in seen
    assert (torch.bfloat16 if fp32_accum else torch.float32) not in seen


def test_install_is_scoped_idempotent_and_reversible(monkeypatch):
    monkeypatch.delenv(vt.WIDE_TILES_ENV, raising = False)
    vae = _ltx_vae()
    assert vt.install(vae) and vt.install(vae)
    assert "tiled_decode" in vae.__dict__
    vt.uninstall(vae)
    assert "tiled_decode" not in vae.__dict__
    monkeypatch.setenv(vt.WIDE_TILES_ENV, "0")
    assert not vt.install(_ltx_vae())
    monkeypatch.delenv(vt.WIDE_TILES_ENV)
    assert not vt.install(
        diffusers.AutoencoderKLWan(base_dim = 8, z_dim = 4, dim_mult = [1, 1], num_res_blocks = 1)
    )


@pytest.mark.parametrize("length", list(range(1, 130)))
def test_tile_layout_full_tiles_and_weights_sum_to_one(length):
    for tile in sorted({16, 20, 24, 32, max(16, length - 1)}):
        starts = vt.tile_starts(length, tile)
        if length > tile:
            assert starts[0] == 0 and starts[-1] + tile == length
            assert all(tile - (b - a) >= vt.OVERLAP_LATENTS for a, b in zip(starts, starts[1:]))
        weights = vt.axis_weights(starts, tile, length, 4, torch, "cpu")
        total = torch.zeros(length * 4, dtype = torch.float64)
        for s, w in zip(starts, weights):
            total[s * 4 : s * 4 + w.numel()] += w.double()
        assert torch.allclose(total, torch.ones_like(total), atol = 1e-6)


MEASURED = {
    (768, 512, 25): 1011,
    (768, 512, 121): 4755,
    (1216, 704, 49): 4239,
    (1216, 704, 121): 10352,
    (896, 896, 121): 9708,
    (1216, 704, 241): 20541,
}


@pytest.mark.parametrize("shape,peak", list(MEASURED.items()))
def test_ltx2_untiled_estimate_covers_measured_peaks(shape, peak):
    w, h, frames = shape
    latent = (1, 128, (frames - 1) // 8 + 1, h // 32, w // 32)
    need = U.untiled_decode_bytes("ltx-2", latent, itemsize = 2)
    assert need is not None and peak * 2**20 <= need <= 1.2 * peak * 2**20
    assert vt.tile_bytes(frames, h // 32, w // 32) == need
    assert U.untiled_decode_bytes("ltx-2", latent, itemsize = 4) == 2 * need


class _FakeLTX:
    def __init__(self):
        self.decoder = torch.nn.Conv3d(1, 1, 1).to(torch.bfloat16)
        self.use_tiling = True
        self.calls = []

    def decode(
        self,
        z,
        temb = None,
        return_dict = False,
    ):
        self.calls.append((self.use_tiling, temb))
        return ("tiled" if self.use_tiling else "untiled",)


def test_resident_ltx2_decodes_untiled_when_it_fits(monkeypatch):
    import types

    monkeypatch.delenv(U.UNTILED_ENV, raising = False)
    z = torch.zeros(1, 128, 16, 22, 38)
    need = U.untiled_decode_bytes("ltx-2", tuple(z.shape), itemsize = 2)
    vae = _FakeLTX()
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * GIB)
    assert U.install_untiled_decode(types.SimpleNamespace(vae = vae), "ltx-2")
    assert vae.decode(z, "t", return_dict = False) == ("untiled",) and vae.calls == [(False, "t")]
    monkeypatch.setattr(U, "_free_bytes", lambda device: int(need * 1.25 + 2 * GIB) - 1)
    assert vae.decode(z, "t") == ("tiled",)


def test_video_load_installs_wide_tiles_on_every_tiling_load():
    src = open(os.path.join(os.path.dirname(vt.__file__), "video.py"), encoding = "utf-8").read()
    tiling = src.index("pipe.vae.enable_tiling()")
    install = src.index("from .video_ltx2_vae_tiles import install")
    resident = src.index("from .video_vae_untiled import install_untiled_decode")
    assert tiling < install < resident


def test_blend_weights_never_put_float64_on_the_device(monkeypatch):
    made = []
    for name in ("arange", "zeros", "ones"):
        real = getattr(torch, name)

        def spy(
            *args,
            _real = real,
            **kwargs,
        ):
            made.append((kwargs.get("dtype"), str(kwargs.get("device", "cpu"))))
            return _real(*args, **kwargs)

        monkeypatch.setattr(torch, name, spy)
    w = vt.axis_weights(vt.tile_starts(38, 16), 16, 38, 4, torch, torch.device("meta"))
    assert all(x.device.type == "meta" and x.dtype == torch.float32 for x in w)
    assert all(dev == "cpu" for dtype, dev in made if dtype == torch.float64)
