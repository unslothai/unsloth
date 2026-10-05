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


# (builder, ratio, stock (tile, overlap), floor (tile, overlap)) with each VAE's real tile settings
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
    # the memory estimates see the wide floor tile; it never shrinks below the stock tile
    assert vae._unsloth_decode_tile_side == floor[0] * ratio
    assert dm.vae_tile_side(vae) == floor[0] * ratio
    vt.uninstall(vae)
    assert "tiled_decode" not in vae.__dict__


@pytest.mark.parametrize(
    "stock, length, ok",
    [
        # Qwen-Image (8x), Qwen-Image-2.1 (16x), HunyuanImage-2.1 (32x): blends under 16 latents, every size
        ((32, 8), 40, False),
        ((32, 8), 128, False),
        ((16, 4), 64, False),
        ((12, 3), 64, False),
        # AutoencoderKL / FLUX.2 (8x, 1024 px tiles, 768 px stride): fine unless the last tile is a sliver
        ((128, 32), 128, True),
        ((128, 32), 166, True),  # 1328 px
        ((128, 32), 192, True),  # 1536 px
        ((128, 32), 200, False),  # 1600 px: an 8-latent last tile
        ((128, 32), 220, False),  # 1760 px: 28 latents
        ((128, 32), 224, True),  # 1792 px: 32 latents
        ((128, 32), 256, True),  # 2048 px
        ((128, 32), 296, False),  # 2368 px: 8 latents
        ((128, 32), 336, True),  # 2688 px
    ],
)
def test_stock_layout_floor(stock, length, ok):
    assert vt.stock_layout_ok(stock, length, length) is ok
    assert vt.stock_layout_ok(stock, min(length, stock[0]), length) is ok
    # one tiled axis is enough to fail
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
    assert vae._unsloth_last_decode_tile[:2] == (64, 64)


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
        assert reason  # every kept class states its measured reason


def test_unreadable_geometry_keeps_the_stock_decode():
    cls = type(
        "SomeFutureVAE", (), {"_decode": lambda self: None, "tiled_decode": lambda self: None}
    )
    vae = cls()
    vae.use_tiling, vae.spatial_compression_ratio = False, 16
    assert vt.stock_tiles(vae) is None
    assert vt.install(vae) is False


@pytest.mark.parametrize("ratio", [8, 16, 32])
def test_decode_budget_scales_with_the_tile_pixels(ratio):
    """1.7 MiB per latent at 16x, the per-pixel figure for every ratio, so Qwen-Image-2.1's budget is unchanged."""
    per_latent = vt.DECODE_MIB_PER_LATENT * (ratio / vt.DECODE_MIB_RATIO) ** 2
    assert per_latent / ratio**2 == pytest.approx(1.7 / 256)
    if ratio == 16:
        assert per_latent == vt.DECODE_MIB_PER_LATENT


# Unfused bf16 decode peak over the weights with floor tiles, real diffusers VAE weights on a B200, worst over 1024 to
# 2048 px canvases (fp32 output accumulator included): the per-pixel budget figure must bound each covered VAE.
_MEASURED_FLOOR_TILE_PEAK_MIB = {
    (8, 32): 314,  # Qwen-Image / Qwen-Image-Edit / Krea-2 (AutoencoderKLQwenImage)
    (16, 32): 1_697,  # Qwen-Image-2.1 (AutoencoderKLQwenImage21, #12696)
    (32, 32): 1_332,  # HunyuanImage-2.1 (AutoencoderKLHunyuanImage)
    (
        8,
        128,
    ): 2_486,  # FLUX.1 / FLUX.2 / SDXL (AutoencoderKL, AutoencoderKLFlux2) at an edge-sliver size
}


def test_budget_figure_bounds_the_measured_floor_tile_peaks():
    for (ratio, tile), peak in _MEASURED_FLOOR_TILE_PEAK_MIB.items():
        budget = vt.DECODE_MIB_PER_LATENT * (ratio / vt.DECODE_MIB_RATIO) ** 2 * tile * tile
        assert budget >= peak, (ratio, tile, budget, peak)


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
        # not nested under an `if` on the family / VAE class
        if isinstance(node, ast.If) and any(c in ast.walk(node) for c in calls):
            assert (
                "fam" not in ast.unparse(node.test) and "vae" not in ast.unparse(node.test).lower()
            )
