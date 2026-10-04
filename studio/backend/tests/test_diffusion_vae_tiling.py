# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 tiled VAE decode without seam lines (diffusion_vae_tiling).

The stock geometry tiles this 16x VAE in 16-latent tiles with 4-latent blends and a 4-latent sliver at the right /
bottom edge, which draws thin vertical / horizontal lines on every tiled (low-VRAM) decode."""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_memory as dm  # noqa: E402
from core.inference import diffusion_vae_tiling as vt  # noqa: E402


@pytest.mark.parametrize("length", list(range(1, 200)) + [256, 257, 333])
def test_tile_starts_are_full_edge_aligned_tiles_with_wide_overlaps(length):
    starts = vt.tile_starts(length)
    tile, overlap = vt.TILE_LATENTS, vt.OVERLAP_LATENTS
    assert starts[0] == 0
    assert starts == sorted(set(starts))
    if length <= tile:
        assert starts == [0]
        return
    assert starts[-1] + tile == length  # the last tile ends at the edge at full size: no sliver
    assert all(b - a <= tile - overlap for a, b in zip(starts, starts[1:]))
    assert len(starts) == math.ceil((length - overlap) / (tile - overlap))


# Every size Studio's Images page offers for Qwen-Image-2.1 (32 px grid, 256 to 2752 px): the 1:1 / 3:2 / 4:3 / 16:9 /
# 21:9 ratios at the default 1024 width and flipped, the official 2K presets, the smallest side, a 512 px side that
# fits one tile, and custom sizes off the presets.
UI_SIZES = [
    (1024, 1024), (1024, 672), (672, 1024), (1024, 768), (768, 1024), (1024, 576), (576, 1024), (1024, 448),
    (448, 1024), (2048, 2048), (2400, 1792), (1792, 2400), (2528, 1696), (1696, 2528), (2752, 1536), (1536, 2752),
    (1344, 1344), (1312, 1312), (1184, 864), (512, 1536), (256, 256), (256, 2752), (2752, 2752),
]
UI_LENGTHS = sorted({side // 16 for size in UI_SIZES for side in size} | set(range(16, 173, 2)))


@pytest.mark.parametrize("length", UI_LENGTHS + [33, 47, 65, 83])
def test_axis_weights_partition_unity_and_skip_shared_edges(length):
    starts = vt.tile_starts(length)
    tile, scale = vt.TILE_LATENTS, 16
    size = min(tile, length)
    weights = vt.axis_weights(starts, tile, length, scale, torch, "cpu")
    total = torch.zeros(length * scale)
    covered = torch.zeros(length * scale, dtype = torch.bool)
    for s, w in zip(starts, weights):
        assert w.shape == (size * scale,)
        assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0 + 1e-6
        total[s * scale : s * scale + w.numel()] += w
        covered[s * scale : s * scale + w.numel()] = True
        edge = vt.MARGIN_LATENTS * scale
        if s > 0:  # no weight where the tile's decode lacks context, even with three tiles overlapping
            assert float(w[:edge].abs().max()) == 0.0
        if s + tile < length:
            assert float(w[-edge:].abs().max()) == 0.0
    assert bool(covered.all())
    torch.testing.assert_close(total, torch.ones_like(total))


@pytest.mark.parametrize("width, height", UI_SIZES)
def test_every_ui_size_tiles_without_slivers(width, height):
    for side in (width, height):
        length = side // 16
        starts = vt.tile_starts(length)
        if length <= vt.TILE_LATENTS:
            assert starts == [0]
            continue
        assert starts[0] == 0 and starts[-1] + vt.TILE_LATENTS == length
        assert all(b + vt.OVERLAP_LATENTS <= a + vt.TILE_LATENTS for a, b in zip(starts, starts[1:]))


def _diffusers_vae():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "AutoencoderKLQwenImage21", None)
    if cls is None:
        pytest.skip("diffusers without AutoencoderKLQwenImage21")
    torch.manual_seed(0)
    return cls(
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


def _line_error(x, ref):
    """Worst full-height column / full-width row mean |x - ref|: what a seam line looks like."""
    d = (x - ref).abs().mean(1)[0, 0]
    return max(float(d.mean(0).max()), float(d.mean(1).max()))


@pytest.mark.parametrize("latents", [40, 64])
def test_tiled_decode_has_no_seam_lines(latents):
    vae = _diffusers_vae()
    assert vae.spatial_compression_ratio == 16
    z = torch.randn(1, 4, 1, latents, latents, generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        vae.use_tiling = False
        untiled = vae.decode(z).sample
        vae.enable_tiling()
        stock = vae.decode(z).sample
        assert vt.install(vae)
        wide = vae.decode(z).sample
    stock_err, wide_err = _line_error(stock, untiled), _line_error(wide, untiled)
    assert stock_err > 0.05  # the stock geometry's seams, so this test can see the defect
    assert wide_err < stock_err / 4, (stock_err, wide_err)


def test_canvas_within_one_tile_decodes_untiled_bit_identical():
    vae = _diffusers_vae()
    z = torch.randn(1, 4, 1, 32, 24, generator = torch.Generator().manual_seed(2))
    with torch.no_grad():
        vae.use_tiling = False
        untiled = vae.decode(z).sample
        vae.enable_tiling()
        assert vt.install(vae)
        tiled = vae.decode(z).sample
    assert torch.equal(tiled, untiled)
    assert vae.use_tiling is True


def test_kill_switch_keeps_the_stock_tiled_decode(monkeypatch):
    vae = _diffusers_vae()
    z = torch.randn(1, 4, 1, 40, 40, generator = torch.Generator().manual_seed(3))
    with torch.no_grad():
        vae.enable_tiling()
        stock = vae.decode(z).sample
        assert vt.install(vae)
        monkeypatch.setenv(vt.WIDE_TILES_ENV, "0")
        killed = vae.decode(z).sample
    assert torch.equal(killed, stock)


def test_install_is_idempotent_and_uninstall_restores():
    vae = _diffusers_vae()
    before = (vae.tile_sample_min_height, vae.tile_sample_stride_height, dm.vae_tile_side(vae))
    assert vt.install(vae) and vt.install(vae)
    assert "tiled_decode" in vae.__dict__
    # the encode geometry is untouched; the memory estimates see the real decode tile
    assert (vae.tile_sample_min_height, vae.tile_sample_stride_height) == before[:2]
    assert dm.vae_tile_side(vae) == vt.TILE_LATENTS * 16
    vt.uninstall(vae)
    assert "tiled_decode" not in vae.__dict__
    assert dm.vae_tile_side(vae) == before[2]


@pytest.mark.parametrize("name", ["AutoencoderKLWan", "AutoencoderKLQwenImage", "AutoencoderKL"])
def test_other_vaes_keep_the_stock_tiles(name):
    vae = type(name, (), {"spatial_compression_ratio": 16, "use_tiling": False, "_decode": lambda self: None})()
    vae.config = types.SimpleNamespace(patch_size = None)
    assert vt.install(vae) is False
    assert "tiled_decode" not in vae.__dict__


def test_patchified_variant_keeps_the_stock_tiles():
    cls = type("AutoencoderKLQwenImage21", (), {"_decode": lambda self: None})
    vae = cls()
    vae.spatial_compression_ratio, vae.use_tiling = 16, False
    vae.config = types.SimpleNamespace(patch_size = 2)
    assert vt.install(vae) is False


def test_load_installs_the_wide_tiles_before_the_speed_optims():
    """The fused batched tile decode only installs on a VAE without its own tiled_decode, so order matters."""
    from core.inference import diffusion

    src = inspect.getsource(inspect.unwrap(diffusion.DiffusionBackend.load_pipeline))
    tree = ast.parse(textwrap.dedent(src))
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("install_wide_vae_tiles", "apply_speed_optims")
    ]
    install = [ln for ln, name in calls if name == "install_wide_vae_tiles"]
    speed = [ln for ln, name in calls if name == "apply_speed_optims"]
    assert install and speed and min(install) < min(speed)


def test_fused_batched_tile_decode_does_not_replace_the_wide_tiles():
    from core.inference import diffusion_vae_fused

    vae = _diffusers_vae()
    assert vt.install(vae)
    ours = vae.__dict__["tiled_decode"]
    assert diffusion_vae_fused.install_wan_tile_batch(vae) is False
    assert vae.__dict__["tiled_decode"] is ours


def test_kill_switch_at_load_skips_the_install(monkeypatch):
    monkeypatch.setenv(vt.WIDE_TILES_ENV, "0")
    vae = _diffusers_vae()
    assert vt.install(vae) is False
    assert "tiled_decode" not in vae.__dict__


@pytest.mark.parametrize("width, height", UI_SIZES)
def test_per_call_guard_budgets_one_wide_tile_at_every_ui_size(width, height):
    """The tiled-decode estimate charges one 512 px tile whatever the canvas, and that covers the measured
    worst case of one wide tile (1,697 MiB unfused bf16 on top of the weights)."""
    vae = _diffusers_vae()
    assert vt.install(vae)
    side = dm.vae_tile_side(vae)
    assert side == vt.TILE_LATENTS * 16
    tiled = dm.estimate_tiled_image_runtime_mib(
        width = width, height = height, family = "qwen-image-2.1", tile_side = side, vae_sliced = True
    )
    one_tile = dm.estimate_image_runtime_mib(width = side, height = side, family = "qwen-image-2.1")
    assert one_tile >= 1_697
    assert tiled >= one_tile
    act = dm.calibrated_image_activation("qwen-image-2.1")
    assert act.tiled_decode_mib >= 1_697
