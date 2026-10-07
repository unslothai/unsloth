# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The wide, edge-aligned tiled VAE decode for every image VAE whose stock tiles fall under the floor.

diffusion_vae_tiling reads each VAE's own tile geometry (diffusers' tile and overlap, in latents) and its compression
ratio, and per decode keeps the stock tiles when they meet the floor (every tile >= 32 latents, every overlap >= 16)
and takes the wide tiles when they do not. These pin the rule for 8x, 16x and 32x VAEs."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_memory as dm  # noqa: E402
from core.inference import diffusion_vae_tiling as vt  # noqa: E402


@pytest.fixture(autouse = True)
def _fixed_decode_tiles(monkeypatch):
    """CPU decodes use the floor tiles anyway; pin it so a CUDA box runs the same geometry."""
    monkeypatch.delenv(vt.WIDE_TILES_ENV, raising = False)
    monkeypatch.setenv(vt.MAX_TILE_ENV, "1")


def _need(name):
    cls = getattr(diffusers, name, None)
    if cls is None:
        pytest.skip(f"diffusers without {name}")
    return cls


def _qwen_image():
    """Qwen-Image / Qwen-Image-Edit / Krea-2's VAE class, 8x: stock 32-latent tiles with 8-latent blends."""
    torch.manual_seed(0)
    return _need("AutoencoderKLQwenImage")(
        base_dim = 8,
        z_dim = 4,
        dim_mult = [1, 2, 4, 4],
        num_res_blocks = 1,
        attn_scales = [],
        temperal_downsample = [False, True, True],
        latents_mean = [0.0] * 4,
        latents_std = [1.0] * 4,
    ).eval()


def _qwen_image_21():
    torch.manual_seed(0)
    return _need("AutoencoderKLQwenImage21")(
        base_dim = 8,
        decoder_base_dim = 8,
        z_dim = 4,
        dim_mult = [1, 2, 4, 4, 4],
        num_res_blocks = 1,
        attn_scales = [],
        temperal_downsample = [False, True, True, True],
        in_channels = 3,
        out_channels = 3,
        latents_mean = [0.0] * 4,
        latents_std = [1.0] * 4,
    ).eval()


def _kl(sample_size = 1024, cls = "AutoencoderKL"):
    """FLUX.1 / Z-Image / HiDream / Lumina / SDXL (AutoencoderKL) or FLUX.2 / Ideogram 4 (AutoencoderKLFlux2), 8x."""
    torch.manual_seed(0)
    return _need(cls)(
        in_channels = 3,
        out_channels = 3,
        down_block_types = ("DownEncoderBlock2D",) * 4,
        up_block_types = ("UpDecoderBlock2D",) * 4,
        block_out_channels = (8, 8, 8, 8),
        layers_per_block = 1,
        latent_channels = 4,
        norm_num_groups = 4,
        sample_size = sample_size,
    ).eval()


def _hunyuan_image():
    """HunyuanImage-2.1's VAE, 32x: stock 12-latent (384 px) tiles with 3-latent blends."""
    torch.manual_seed(0)
    return _need("AutoencoderKLHunyuanImage")(
        in_channels = 3,
        out_channels = 3,
        latent_channels = 8,
        block_out_channels = (32,) * 6,
        layers_per_block = 1,
        spatial_compression_ratio = 32,
        sample_size = 384,
    ).eval()


GEOMETRY = [
    (_qwen_image, 8, (32, 8), (32, 16)),
    (_qwen_image_21, 16, (16, 4), (32, 16)),
    (_hunyuan_image, 32, (12, 3), (32, 16)),
    (_kl, 8, (128, 32), (128, 32)),
    (lambda: _kl(cls = "AutoencoderKLFlux2"), 8, (128, 32), (128, 32)),
]


@pytest.mark.parametrize("build, ratio, stock, floor", GEOMETRY)
def test_geometry_is_read_off_each_vae(build, ratio, stock, floor):
    vae = build()
    assert vt.compression_ratio(vae) == ratio
    assert vt.stock_tiles(vae) == stock
    assert vt.floor_tiles(vae) == floor
    assert vt.install(vae)
    assert vae._unsloth_decode_tile_side == floor[0] * ratio
    assert dm.vae_tile_side(vae) == floor[0] * ratio
    vt.uninstall(vae)
    assert "tiled_decode" not in vae.__dict__


@pytest.mark.parametrize(
    "stock, length, ok",
    [
        ((32, 8), 40, False),
        ((32, 8), 128, False),
        ((16, 4), 64, False),
        ((12, 3), 64, False),
        ((128, 32), 128, True),
        ((128, 32), 166, True),
        ((128, 32), 192, True),
        ((128, 32), 200, False),  # 1600 px: an 8-latent last tile
        ((128, 32), 220, False),
        ((128, 32), 224, True),
        ((128, 32), 256, True),
        ((128, 32), 296, False),  # 2368 px: 8 latents
        ((128, 32), 336, True),
    ],
)
def test_stock_layout_floor(stock, length, ok):
    assert vt.stock_layout_ok(stock, length, length) is ok
    assert vt.stock_layout_ok(stock, min(length, stock[0]), length) is ok
    if not ok:
        assert vt.stock_layout_ok(stock, stock[0], length) is False


@pytest.mark.parametrize("tile, overlap", [(32, 16), (128, 32)])
@pytest.mark.parametrize("length", list(range(1, 400, 3)))
def test_wide_tiles_are_full_and_edge_aligned_for_every_floor(tile, overlap, length):
    starts = vt.tile_starts(length, tile, overlap)
    if length <= tile:
        assert starts == [0]
        return
    assert starts[0] == 0 and starts[-1] + tile == length
    assert all(b + overlap <= a + tile for a, b in zip(starts, starts[1:]))
    for ratio in (8, 16, 32):
        weights = vt.axis_weights(starts, tile, length, ratio, torch, "cpu")
        total = torch.zeros(length * ratio)
        for s, w in zip(starts, weights):
            total[s * ratio : s * ratio + w.numel()] += w
        torch.testing.assert_close(total, torch.ones_like(total))


def _line_error(x, ref):
    """Worst full-height column / full-width row mean |x - ref|: what a seam line looks like."""
    d = (x - ref).abs().mean(1)[0]
    if d.dim() == 3:
        d = d[0]
    return max(float(d.mean(0).max()), float(d.mean(1).max()))


def _decode_three_ways(vae, z):
    with torch.no_grad():
        vae.use_tiling = False
        untiled = vae.decode(z).sample
        vae.enable_tiling()
        stock = vae.decode(z).sample
        assert vt.install(vae)
        wide = vae.decode(z).sample
    return untiled, stock, wide


@pytest.mark.parametrize("latents", [40, 64])
def test_qwen_image_tiled_decode_has_no_seam_lines(latents):
    """Fails before this change: the Qwen-Image VAE kept diffusers' 32-latent tiles with 8-latent blends."""
    vae = _qwen_image()
    z = torch.randn(1, 4, 1, latents, latents, generator = torch.Generator().manual_seed(1))
    untiled, stock, wide = _decode_three_ways(vae, z)
    stock_err, wide_err = _line_error(stock, untiled), _line_error(wide, untiled)
    assert stock_err > 0.02
    assert wide_err < stock_err / 2, (stock_err, wide_err)
    assert vae._unsloth_last_decode_tile[:2] == (32, 32)


def test_hunyuan_image_tiled_decode_has_no_seam_lines():
    """32x: stock 12-latent tiles with 3-latent blends; the floor is 32-latent tiles with 16-latent overlaps."""
    vae = _hunyuan_image()
    z = torch.randn(1, 8, 40, 40, generator = torch.Generator().manual_seed(1))
    untiled, stock, wide = _decode_three_ways(vae, z)
    stock_err, wide_err = _line_error(stock, untiled), _line_error(wide, untiled)
    assert stock_err > 0.02
    assert wide_err < stock_err / 4, (stock_err, wide_err)
    assert wide.shape == untiled.shape


def test_kl_vae_keeps_the_stock_tiles_where_they_meet_the_floor():
    """AutoencoderKL with 512 px tiles (64 latents, 16-latent blends): bit-identical to stock on an 80-latent canvas."""
    vae = _kl(sample_size = 512)
    assert vt.stock_tiles(vae) == (64, 16)
    z = torch.randn(1, 4, 80, 80, generator = torch.Generator().manual_seed(2))
    _, stock, wide = _decode_three_ways(vae, z)
    assert torch.equal(wide, stock)


def test_kl_vae_takes_the_wide_tiles_for_an_edge_sliver():
    """A 100-latent side leaves stock a 4-latent last tile; the wide tiles end at the edge at full size."""
    vae = _kl(sample_size = 512)
    z = torch.randn(1, 4, 100, 100, generator = torch.Generator().manual_seed(2))
    untiled, stock, wide = _decode_three_ways(vae, z)
    assert vt.stock_layout_ok(vt.stock_tiles(vae), 100, 100) is False
    stock_err, wide_err = _line_error(stock, untiled), _line_error(wide, untiled)
    assert wide_err < stock_err / 3, (stock_err, wide_err)
    # two 60-latent tiles per side (120 decoded latents, as many as the stock 64 + 52 + 4, in 2 calls instead of 3),
    # not two 64s with a 28-latent overlap (128)
    assert vae._unsloth_last_decode_tile[:2] == (60, 60)


def test_kl_encode_is_untouched_and_qwen_image_encode_is_wide():
    """AutoencoderKL tiles its encode privately (``_tiled_encode``, 1024 px tiles); the Wan family routes ``_encode``
    through ``tiled_encode`` with the stock 8-latent blends."""
    kl = _kl()
    assert vt.install(kl)
    assert "tiled_encode" not in kl.__dict__
    qi = _qwen_image()
    assert vt.install(qi)
    assert "tiled_encode" in qi.__dict__


def test_qwen_image_tiled_encode_is_closer_to_untiled():
    vae = _qwen_image()
    g = torch.Generator().manual_seed(4)
    x = torch.nn.functional.interpolate(
        torch.rand(1, 3, 8, 8, generator = g) * 2 - 1, size = (640, 640), mode = "bicubic"
    ).clamp(-1, 1)[:, :, None]
    with torch.no_grad():
        vae.use_tiling = False
        untiled = vae._encode(x)
        vae.enable_tiling()
        stock = vae._encode(x)
        assert vt.install(vae)
        wide = vae._encode(x)
    assert wide.shape == untiled.shape == stock.shape
    assert (wide - untiled).abs().mean() < (stock - untiled).abs().mean() / 2


@pytest.mark.parametrize("build, ratio, stock, floor", GEOMETRY)
def test_kill_switch_keeps_every_stock_decode(monkeypatch, build, ratio, stock, floor):
    vae = build()
    side = max(floor[0] + 8, stock[0] + 8)
    shape = (
        (1, 4, side, side)
        if not hasattr(vae, "tile_sample_stride_height")
        else (1, 4, 1, side, side)
    )
    if ratio == 32:
        shape = (1, 8, side, side)
    if ratio * side > 1300:
        pytest.skip("too large a CPU decode")
    z = torch.randn(*shape, generator = torch.Generator().manual_seed(3))
    with torch.no_grad():
        vae.enable_tiling()
        before = vae.decode(z).sample
        assert vt.install(vae)
        monkeypatch.setenv(vt.WIDE_TILES_ENV, "0")
        killed = vae.decode(z).sample
    assert torch.equal(killed, before)


def test_keep_stock_list_skips_the_install(monkeypatch):
    monkeypatch.setitem(vt.KEEP_STOCK, "AutoencoderKLQwenImage", "test")
    vae = _qwen_image()
    assert vt.install(vae) is False
    assert "tiled_decode" not in vae.__dict__
    for reason in vt.KEEP_STOCK.values():
        assert reason


def test_unreadable_geometry_keeps_the_stock_decode():
    cls = type(
        "SomeFutureVAE", (), {"_decode": lambda self: None, "tiled_decode": lambda self: None}
    )
    vae = cls()
    vae.use_tiling, vae.spatial_compression_ratio = False, 16
    assert vt.stock_tiles(vae) is None
    assert vt.install(vae) is False


# bf16 decode peak per latent of tile area, (unfused, fused) MiB: one untiled ``_decode`` of a t x t latent, t = 8 to
# 128, on the real diffusers VAE weights (B200, diffusers 0.41.0.dev0, temp calibration run); the worst side per VAE.
_MEASURED_MIB_PER_LATENT = {
    "AutoencoderKLQwenImage": (8, 0.270, 0.179),  # Qwen-Image / Qwen-Image-Edit / Krea-2
    "AutoencoderKLQwenImage21": (
        16,
        1.650,
        0.425,
    ),  # Qwen-Image-2.1 (the 8-latent fused side, 0.59, is never a tile)
    "AutoencoderKLHunyuanImage": (32, 1.255, 1.255),  # HunyuanImage-2.1, no fused path
    "AutoencoderKL": (8, 0.166, 0.086),  # FLUX.1 / Kontext / Z-Image / HiDream / Lumina 2 / SDXL
    "AutoencoderKLFlux2": (8, 0.166, 0.086),  # FLUX.2 / Ideogram 4
}


@pytest.mark.parametrize("cls", sorted(_MEASURED_MIB_PER_LATENT))
def test_calibrated_figures_bound_the_measured_peaks_and_are_tight(cls):
    """Not the old per-pixel scaling (6.8 MiB per latent at 32x, 5x HunyuanImage's peak; 0.43 at 8x, 2.9x
    AutoencoderKL's): within 10% of each VAE's measured peak (Qwen-Image-2.1: #12696's 1.7), so the free VRAM buys
    the largest tiles it really holds."""
    ratio, unfused, fused = _MEASURED_MIB_PER_LATENT[cls]
    assert unfused <= vt.decode_mib_per_latent(ratio, False, cls) <= 1.1 * unfused
    assert fused <= vt.decode_mib_per_latent(ratio, True, cls) <= 1.1 * fused
    # a class not measured takes the worst VAE of its ratio
    worst = max(u for r, u, _ in _MEASURED_MIB_PER_LATENT.values() if r == ratio)
    worst_fused = max(f for r, _, f in _MEASURED_MIB_PER_LATENT.values() if r == ratio)
    assert vt.decode_mib_per_latent(ratio, False, "SomeFutureVAE") >= worst
    assert vt.decode_mib_per_latent(ratio, True, "SomeFutureVAE") >= worst_fused


def test_calibrated_figures_other_ratios():
    # Qwen-Image-2.1's unfused figure is #12696's
    assert vt.decode_mib_per_latent(16) == 1.7
    # an unmeasured ratio keeps the per-pixel scaling of the 16x figure
    assert vt.decode_mib_per_latent(4) == pytest.approx(1.7 / 16)


@pytest.mark.parametrize("fused", [False, True])
def test_budget_uses_the_fused_figure_only_where_the_fused_kernels_are_installed(
    monkeypatch, fused
):
    vae = _hunyuan_image()
    qi = _qwen_image()
    monkeypatch.delenv(vt.MAX_TILE_ENV)
    monkeypatch.setattr(vt, "_free_mib", lambda v, z, **k: (1000.0, 1.0))
    for v in (vae, qi):
        assert vt.install(v)
        if fused:
            v._unsloth_vae_fused_installed = 7
    z = torch.zeros(1, 4, 1, 64, 64)
    want = int(vt.FREE_FRACTION * 1000 / (0.18 if fused else 0.27))
    assert vt.decode_tile_budget(qi, z) == want
    assert vt.decode_tile_budget(vae, torch.zeros(1, 8, 64, 64)) == int(
        vt.FREE_FRACTION * 1000 / 1.3
    )
    kl = _kl()
    assert vt.install(kl)
    if fused:
        kl._unsloth_vae_fused_installed = 3
    assert vt.decode_tile_budget(kl, torch.zeros(1, 4, 64, 64)) == int(
        vt.FREE_FRACTION * 1000 / (0.09 if fused else 0.17)
    )
    qi._unsloth_vae_fused_failed = True
    assert vt.decode_tile_budget(qi, z) == int(vt.FREE_FRACTION * 1000 / 0.27)


def test_only_large_stock_tile_vaes_size_past_the_floor_from_device_free_memory(monkeypatch):
    """AutoencoderKL / FLUX.2 take the wide tiles only to drop a sliver: their tiles grow past the floor only into the
    device's free memory, not the allocator's cached blocks (an untiled FLUX.1 decode out of the denoiser's cache
    stalled on cache flushes at 8 GB: 0.21 s median against stock's 0.08). The Wan family and HunyuanImage count the
    cache, as #12696 does for Qwen-Image-2.1 (it measured faster there)."""
    monkeypatch.delenv(vt.MAX_TILE_ENV)
    seen = {}

    def free(
        v,
        z,
        cached = True,
    ):
        seen[type(v).__name__] = cached
        return 1000.0 if cached else 100.0, 1.0

    monkeypatch.setattr(vt, "_free_mib", free)
    for build, z in (
        (_kl, torch.zeros(1, 4, 64, 64)),
        (lambda: _kl(cls = "AutoencoderKLFlux2"), torch.zeros(1, 4, 64, 64)),
        (_qwen_image, torch.zeros(1, 4, 1, 64, 64)),
        (_qwen_image_21, torch.zeros(1, 4, 1, 64, 64)),
        (_hunyuan_image, torch.zeros(1, 8, 64, 64)),
    ):
        vae = build()
        assert vt.install(vae)
        vt.decode_tile_budget(vae, z)
    assert seen == {
        "AutoencoderKL": False,
        "AutoencoderKLFlux2": False,
        "AutoencoderKLQwenImage": True,
        "AutoencoderKLQwenImage21": True,
        "AutoencoderKLHunyuanImage": True,
    }
    # and the floor-fit check (stock fallback) still counts the cache for every VAE
    vae = _hunyuan_image()
    assert vt.install(vae)
    seen.clear()
    vt.floor_shortfall(vae, torch.zeros(1, 8, 64, 64))
    assert seen == {"AutoencoderKLHunyuanImage": True}


def _stock_decoded(length, tile, overlap):
    """Latents diffusers' stock loop decodes along one side (a tile every tile - overlap, the last one cut short)."""
    if length <= tile:
        return length
    return sum(min(tile, length - s) for s in range(0, length, tile - overlap))


@pytest.mark.parametrize("length", list(range(129, 420)))
@pytest.mark.parametrize("max_area", [None, 23_200, 10**6])
def test_large_floor_tiles_never_decode_more_than_stock(length, max_area):
    """FLUX.1 / SDXL / FLUX.2 sliver sizes. Before, the floor kept 128-latent tiles and spread them (two 128s with a
    56-latent overlap on a 200-latent side: 7% more decode than stock, 9% slower). Now, per side, no more decoded
    latents than the stock loop (up to rounding where only the smallest covering side exists), fewer decoder calls,
    no sliver, every overlap >= 32, and the widest tile that allows (two 120s on a 200-latent side)."""
    if vt.stock_layout_ok((128, 32), length, length):
        return  # no sliver: these sizes decode through the stock tiles
    th, tw = vt.choose_tiles(length, length, max_area, 128, 32, 128)
    assert th * tw <= max(max_area or 0, 128 * 128) and min(th, tw) >= vt.TILE_LATENTS
    stock_calls = len(range(0, length, 96))
    span = _stock_decoded(length, 128, 32)
    for side in (th, tw):
        starts = vt.tile_starts(length, side, 32)
        assert starts[0] == 0 and starts[-1] + side == length or starts == [0]
        assert all(b + 32 <= a + side for a, b in zip(starts, starts[1:]))
        assert len(starts) < stock_calls
        assert len(starts) * side <= span + len(starts) - 1
    if length == 200:
        assert (th, tw) == {None: (120, 120), 23_200: (116, 200), 10**6: (200, 200)}[max_area]
        if max_area is None:
            assert vt.tile_starts(200, 120, 32) == [0, 80]


def test_qwen_family_layout_ignores_the_large_stock_rule():
    """Only a VAE whose stock tile is the floor tile takes the stock-work rule; the 32-latent floor VAEs keep #12696's
    fewest-latents choice whatever stock tile is passed."""
    for length in (64, 100, 128, 166, 256):
        for area in (None, 1500, 4000, 10**6):
            assert vt.choose_tiles(length, length, area) == vt.choose_tiles(
                length, length, area, 32, 16, 16
            )
            assert vt.choose_tiles(length, length, area) == vt.choose_tiles(
                length, length, area, 32, 16, 32
            )


@pytest.mark.parametrize("length", list(range(1, 400)))
@pytest.mark.parametrize("max_area", [None, 0, 1024])
def test_the_32_latent_floor_layout_is_unchanged(length, max_area):
    """Qwen-Image, Qwen-Image-2.1 and HunyuanImage at the floor: 32-latent tiles, as #12696 picks."""
    assert vt.choose_tiles(length, length, max_area) == (min(32, length), min(32, length))


def _decode_with(vae, z):
    with torch.no_grad():
        vae.enable_tiling()
        return vae.decode(z).sample


class _Log:
    def __init__(self):
        self.records = []

    def info(self, *a):
        pass

    def debug(self, *a):
        self.records.append(("debug", a[0] % a[1:]))

    def warning(self, *a):
        self.records.append(("warning", a[0] % a[1:]))


def test_floor_that_cannot_fit_decodes_in_the_stock_tiles_and_says_why(monkeypatch):
    """HunyuanImage's 32-latent floor tile peaks at ~1.3 GiB, its 12-latent stock tile at ~0.2: with less free than the
    floor needs, OOM loses to a seam, so the decode takes the stock tiles and logs the reason (once per VAE)."""
    vae = _hunyuan_image()
    z = torch.randn(1, 8, 40, 40, generator = torch.Generator().manual_seed(1))
    stock = _decode_with(vae, z)
    log = _Log()
    assert vt.install(vae, log)
    monkeypatch.delenv(vt.MAX_TILE_ENV)
    monkeypatch.setattr(
        vt, "_free_mib", lambda v, zz, **k: (500.0, 1.0)
    )  # 32x32 x 1.3 MiB = 1,331 MiB needed
    assert "floor tile needs about 1331 MiB" in vt.floor_shortfall(vae, z)
    assert torch.equal(_decode_with(vae, z), stock)
    assert _decode_with(vae, z).shape == stock.shape
    assert [k for k, _ in log.records] == ["warning", "debug"]
    assert "HunyuanImage" in log.records[0][1] and "stock tiles" in log.records[0][1]
    # with room for the floor, the wide tiles again
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz, **k: (2000.0, 1.0))
    assert vt.floor_shortfall(vae, z) is None
    assert not torch.equal(_decode_with(vae, z), stock)
    # unknown free memory (CPU) and an explicit tile cap are never a shortfall
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz, **k: None)
    assert vt.floor_shortfall(vae, z) is None
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz, **k: (1.0, 1.0))
    monkeypatch.setenv(vt.MAX_TILE_ENV, "32")
    assert vt.floor_shortfall(vae, z) is None


@pytest.mark.parametrize("build", [_qwen_image, _kl, lambda: _kl(cls = "AutoencoderKLFlux2")])
def test_no_stock_fallback_where_the_stock_tile_is_as_large(monkeypatch, build):
    """Qwen-Image (32-latent stock tiles) and AutoencoderKL / FLUX.2 (128): the stock tiles need as much memory as the
    floor, so a low free VRAM keeps the wide tiles (the FLUX.1 sliver fix held at 2 GiB free only after this)."""
    vae = build()
    assert vt.install(vae)
    vae._unsloth_vae_fused_installed = 3
    monkeypatch.delenv(vt.MAX_TILE_ENV)
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz, **k: (1.0, 1.0))
    z = (
        torch.zeros(1, 4, 1, 200, 200)
        if hasattr(vae, "tile_sample_stride_height")
        else torch.zeros(1, 4, 200, 200)
    )
    assert vt.floor_shortfall(vae, z) is None


def test_floor_fit_uses_the_fused_peak_so_qwen_image_21_keeps_its_wide_floor(monkeypatch):
    """Qwen-Image-2.1's fused floor tile peaks at ~0.43 GiB: it stays on the wide floor down to that (where #12696
    would OOM), not at the 1.7 GiB unfused figure."""
    vae = _qwen_image_21()
    assert vt.install(vae)
    vae._unsloth_vae_fused_installed = 5
    monkeypatch.delenv(vt.MAX_TILE_ENV)
    z = torch.zeros(1, 4, 1, 64, 64)
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz, **k: (500.0, 1.0))
    assert vt.floor_shortfall(vae, z) is None
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz, **k: (400.0, 1.0))
    assert vt.floor_shortfall(vae, z)
    vae._unsloth_vae_fused_failed = True
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz, **k: (1500.0, 1.0))
    assert vt.floor_shortfall(vae, z)


def test_oom_retries_at_the_floor_then_in_the_stock_tiles(monkeypatch):
    vae = _qwen_image()
    z = torch.randn(1, 4, 1, 64, 64, generator = torch.Generator().manual_seed(1))
    stock = _decode_with(vae, z)
    assert vt.install(vae, _Log())
    floor = _decode_with(vae, z)  # the autouse 1-latent cap: the floor tiles
    real = vt.tiled_decode
    calls = []

    def flaky(
        v,
        zz,
        return_dict = True,
        max_area = "auto",
        fail = ("auto",),
    ):
        calls.append(max_area)
        if max_area in fail:
            v._unsloth_last_decode_tile = (48, 48, 2304) if max_area == "auto" else (32, 32, 0)
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")
        return real(v, zz, return_dict = return_dict, max_area = max_area)

    monkeypatch.setattr(vt, "tiled_decode", flaky)
    assert torch.equal(_decode_with(vae, z), floor)
    assert calls == ["auto", 0]
    calls.clear()
    monkeypatch.setattr(vt, "tiled_decode", lambda *a, **k: flaky(*a, **k, fail = ("auto", 0)))
    assert torch.equal(_decode_with(vae, z), stock)
    assert calls == ["auto", 0]
    # an error that is not an OOM propagates
    monkeypatch.setattr(
        vt, "tiled_decode", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(ValueError):
        _decode_with(vae, z)


def test_geometry_and_blend_weights_are_cached_per_vae(monkeypatch):
    vae = _qwen_image()
    assert vt.install(vae)
    assert vae._unsloth_wide_geometry == (8, 32, 16)
    built = []
    real = vt.axis_weights
    monkeypatch.setattr(vt, "axis_weights", lambda *a, **k: built.append(a[:3]) or real(*a, **k))
    monkeypatch.setattr(vt, "compression_ratio", lambda v: pytest.fail("ratio re-read per decode"))
    z = torch.randn(1, 4, 1, 48, 48, generator = torch.Generator().manual_seed(1))
    first = _decode_with(vae, z)
    assert len(built) == 1
    second = _decode_with(vae, z)
    assert len(built) == 1 and torch.equal(first, second)
    vt.uninstall(vae)
    assert (
        "_unsloth_wide_weights" not in vae.__dict__ and "_unsloth_wide_geometry" not in vae.__dict__
    )


def _image_family_vae_classes():
    """Every image family's VAE class, from the diffusers pipeline's own ``vae`` annotation."""
    from core.inference.diffusion_families import _FAMILIES

    out = {}
    for fam in _FAMILIES:
        pipe = getattr(diffusers, fam.pipeline_class, None)
        if pipe is None:
            continue
        ann = inspect.signature(pipe.__init__).parameters.get("vae")
        if ann is None:
            continue
        hint = ann.annotation
        name = hint if isinstance(hint, str) else getattr(hint, "__name__", str(hint))
        out[fam.name] = name.split(".")[-1].split("|")[0].strip()
    return out


def test_every_image_family_vae_is_covered_or_kept_stock_on_purpose():
    """A new image family (or a diffusers that renames a VAE) must land on a VAE whose tile geometry the rule reads,
    or be listed in KEEP_STOCK with the measured reason: a VAE the rule cannot read keeps stock tiles silently."""
    readable = {
        "AutoencoderKL",
        "AutoencoderKLFlux2",
        "AutoencoderKLQwenImage",
        "AutoencoderKLQwenImage21",
        "AutoencoderKLHunyuanImage",
    }
    found = _image_family_vae_classes()
    assert found, "no image family resolved its VAE class"
    for fam, cls in found.items():
        assert cls in readable or cls in vt.KEEP_STOCK, (fam, cls)


def test_load_installs_for_every_image_family():
    """The load path installs unconditionally (the rule decides per VAE), not behind a family check."""
    from core.inference import diffusion

    src = textwrap.dedent(
        inspect.getsource(inspect.unwrap(diffusion.DiffusionBackend.load_pipeline))
    )
    calls = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "install_wide_vae_tiles"
    ]
    assert calls
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If) and any(c in ast.walk(node) for c in calls):
            assert (
                "fam" not in ast.unparse(node.test) and "vae" not in ast.unparse(node.test).lower()
            )


def test_stock_fallback_keeps_the_fused_batched_loop_on_the_wan_family(monkeypatch):
    """Under the wide tiles the fused VAE layer cannot take ``tiled_decode`` (they own it), so before this the stock
    fallback (kill switch, or a floor that cannot fit) ran diffusers' one-tile-per-call loop, slower than main's
    batched one. Now the batched loop backs the fallback, and is ``tiled_decode`` again after an uninstall."""
    from core.inference import diffusion_vae_fused as fused

    vae = _qwen_image_21()
    assert vt.install(vae)
    ours = vae.__dict__["tiled_decode"]
    assert fused.install_wan_tile_batch(vae) is False
    assert vae.__dict__["tiled_decode"] is ours
    batched = vae.__dict__["_unsloth_wide_stock_decode"]
    assert getattr(batched, "_unsloth_vae_fused", False)
    calls = []
    vae._unsloth_wide_stock_decode = lambda z, return_dict = True: calls.append(z.shape) or batched(
        z, return_dict = return_dict
    )
    z = torch.randn(1, 4, 1, 48, 48, generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        vae.enable_tiling()
        monkeypatch.setenv(vt.WIDE_TILES_ENV, "0")
        killed = vae.decode(z).sample
        monkeypatch.delenv(vt.WIDE_TILES_ENV)
        monkeypatch.delenv(vt.MAX_TILE_ENV)
        monkeypatch.setattr(vt, "_free_mib", lambda v, zz, **k: (1.0, 1.0))
        short = vae.decode(z).sample
    assert len(calls) == 2 and torch.equal(killed, short)
    vae._unsloth_wide_stock_decode = batched
    vt.uninstall(vae)
    assert vae.__dict__.get("tiled_decode") is batched
    # without the wide tiles the batched loop installs on ``tiled_decode`` as before
    plain = _qwen_image_21()
    assert fused.install_wan_tile_batch(plain)
    assert getattr(plain.__dict__["tiled_decode"], "_unsloth_vae_fused", False)
    assert "_unsloth_wide_stock_decode" not in plain.__dict__
