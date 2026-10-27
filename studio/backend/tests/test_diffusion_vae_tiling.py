# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stock 16x VAE tiling leaves a 4-latent sliver that draws seam lines; this geometry avoids it."""

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
    assert starts[-1] + tile == length
    assert all(b - a <= tile - overlap for a, b in zip(starts, starts[1:]))
    assert len(starts) == math.ceil((length - overlap) / (tile - overlap))


UI_SIZES = [
    (1024, 1024),
    (1024, 672),
    (672, 1024),
    (1024, 768),
    (768, 1024),
    (1024, 576),
    (576, 1024),
    (1024, 448),
    (448, 1024),
    (2048, 2048),
    (2400, 1792),
    (1792, 2400),
    (2528, 1696),
    (1696, 2528),
    (2752, 1536),
    (1536, 2752),
    (1344, 1344),
    (1312, 1312),
    (1184, 864),
    (512, 1536),
    (256, 256),
    (256, 2752),
    (2752, 2752),
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
        if s > 0:
            assert float(w[:edge].abs().max()) == 0.0
        if s + tile < length:
            assert float(w[-edge:].abs().max()) == 0.0
    assert bool(covered.all())
    torch.testing.assert_close(total, torch.ones_like(total))


def test_axis_weights_keep_float64_off_a_device_without_it(monkeypatch):
    # MPS has no float64: the tiled encode on Apple Silicon raised before the first step (#12935).
    from torch.utils._python_dispatch import TorchDispatchMode

    asked, made = [], []

    class Record(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            out = func(*args, **(kwargs or {}))
            if torch.is_tensor(out):
                made.append((out.dtype, out.device.type))
            return out

    monkeypatch.setattr(vt, "float64_device", lambda device: asked.append(device) or "cpu")
    target = torch.device("meta")
    with Record():
        w = vt.axis_weights(vt.tile_starts(47), vt.TILE_LATENTS, 47, 16, torch, target)
    assert asked == [target]
    assert all(x.device.type == "meta" and x.dtype == torch.float32 for x in w)
    assert (torch.float64, "cpu") in made and (torch.float64, "meta") not in made


@pytest.mark.parametrize("width, height", UI_SIZES)
def test_every_ui_size_tiles_without_slivers(width, height):
    for side in (width, height):
        length = side // 16
        starts = vt.tile_starts(length)
        if length <= vt.TILE_LATENTS:
            assert starts == [0]
            continue
        assert starts[0] == 0 and starts[-1] + vt.TILE_LATENTS == length
        assert all(
            b + vt.OVERLAP_LATENTS <= a + vt.TILE_LATENTS for a, b in zip(starts, starts[1:])
        )


@pytest.fixture(autouse = True)
def _fixed_decode_tiles(monkeypatch):
    """CPU decodes use 32x32 tiles anyway; pin it so a CUDA box runs the same geometry."""
    monkeypatch.setenv(vt.MAX_TILE_ENV, str(vt.TILE_LATENTS))


def _diffusers_vae():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "AutoencoderKLQwenImage21", None)
    if cls is None:
        pytest.skip(
            reason = "needs diffusers >= 0.41 (AutoencoderKLQwenImage21); the 0.40.0 pin predates it"
        )
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
    assert "tiled_decode" in vae.__dict__ and "tiled_encode" in vae.__dict__
    assert (vae.tile_sample_min_height, vae.tile_sample_stride_height) == before[:2]
    assert dm.vae_tile_side(vae) == vt.TILE_LATENTS * 16
    vt.uninstall(vae)
    assert "tiled_decode" not in vae.__dict__ and "tiled_encode" not in vae.__dict__
    assert dm.vae_tile_side(vae) == before[2]


@pytest.mark.parametrize("name", ["AutoencoderKLWan", "AutoencoderKLQwenImage", "AutoencoderKL"])
def test_other_vaes_keep_the_stock_tiles(name):
    vae = type(
        name,
        (),
        {"spatial_compression_ratio": 16, "use_tiling": False, "_decode": lambda self: None},
    )()
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


# measured on B200, unfused bf16: 512 px is one 32-latent tile, 1024+ px use 64-latent tiles
_MEASURED_ENCODE_TILE_MIB = {512: 590, 1024: 2_322, 2048: 2_322}


@pytest.mark.parametrize("width, height", UI_SIZES)
def test_per_call_guard_covers_the_encode_tile_at_every_reference_size(width, height):
    """Qwen-Image-2.1 only encodes edit inputs, at the reference resolution, and the guard charges those as weighted
    condition pixels. The tiled estimate must still cover one 64-latent encode tile at every output size."""
    from core.inference.diffusion_families import detect_family

    fam = detect_family("Qwen/Qwen-Image-2.1")
    assert fam is not None and fam.name == "qwen-image-2.1"
    assert fam.img2img_pipeline_class is None
    for ref in fam.reference_resolutions:
        cond = int(ref * ref * getattr(fam, "condition_pixel_weight", 1.0))
        tiled = dm.estimate_tiled_image_runtime_mib(
            width = width,
            height = height,
            family = fam.name,
            condition_pixels = cond,
            tile_side = vt.TILE_LATENTS * 16,
            vae_sliced = True,
        )
        assert tiled >= _MEASURED_ENCODE_TILE_MIB[ref], (ref, tiled)
    assert vt.ENCODE_TILE_LATENTS * 16 == 1024


def _line_error_latent(x, ref):
    d = (x - ref).abs().mean(1)[0, 0]
    return max(float(d.mean(0).max()), float(d.mean(1).max()))


@pytest.mark.parametrize("height, width", [(640, 640), (256, 1088), (1088, 256)])
def test_tiled_encode_has_no_seam_lines(height, width):
    vae = _diffusers_vae()
    g = torch.Generator().manual_seed(4)
    x = torch.nn.functional.interpolate(
        torch.rand(1, 3, 8, 8, generator = g) * 2 - 1, size = (height, width), mode = "bicubic"
    )
    x = x.clamp(-1, 1)[:, :, None]
    with torch.no_grad():
        vae.use_tiling = False
        untiled = vae._encode(x)
        vae.enable_tiling()
        stock = vae._encode(x)
        assert vt.install(vae)
        wide = vae._encode(x)
    assert wide.shape == untiled.shape == stock.shape
    stock_err, wide_err = _line_error_latent(stock, untiled), _line_error_latent(wide, untiled)
    assert stock_err > 0
    if max(height, width) <= vt.ENCODE_TILE_LATENTS * 16:
        # up to 1024 px a side is one tile, identical to the untiled encode
        assert torch.equal(wide, untiled)
    else:
        assert stock_err > 2 * wide_err, (stock_err, wide_err)


def test_encode_tiles_are_wider_than_decode_tiles():
    assert vt.ENCODE_TILE_LATENTS == 2 * vt.TILE_LATENTS
    assert vt.ENCODE_OVERLAP_LATENTS >= vt.ENCODE_TILE_LATENTS // 2
    for length in range(1, 200):
        starts = vt.tile_starts(length, vt.ENCODE_TILE_LATENTS, vt.ENCODE_OVERLAP_LATENTS)
        w = vt.axis_weights(
            starts,
            vt.ENCODE_TILE_LATENTS,
            length,
            1,
            torch,
            "cpu",
            vt.ENCODE_MARGIN_LATENTS,
            vt.ENCODE_RAMP_LATENTS,
        )
        total = torch.zeros(length)
        for s, wi in zip(starts, w):
            total[s : s + wi.numel()] += wi
        assert torch.allclose(total, torch.ones(length), atol = 1e-6)


def test_encode_within_one_tile_is_untiled_and_kill_switch_keeps_stock(monkeypatch):
    vae = _diffusers_vae()
    x = torch.rand(1, 3, 1, 512, 384, generator = torch.Generator().manual_seed(5)) * 2 - 1
    big = torch.rand(1, 3, 1, 640, 640, generator = torch.Generator().manual_seed(6)) * 2 - 1
    with torch.no_grad():
        vae.use_tiling = False
        untiled = vae._encode(x)
        vae.enable_tiling()
        stock_big = vae._encode(big)
        assert vt.install(vae)
        assert torch.equal(vae._encode(x), untiled)
        monkeypatch.setenv(vt.WIDE_TILES_ENV, "0")
        assert torch.equal(vae._encode(big), stock_big)
    vt.uninstall(vae)
    assert "tiled_encode" not in vae.__dict__


def _decoded_latents(height, width, th, tw):
    return len(vt.tile_starts(height, th)) * th * len(vt.tile_starts(width, tw)) * tw


@pytest.mark.parametrize("width, height", UI_SIZES)
@pytest.mark.parametrize("max_area", [None, 0, 1023, 1024, 1300, 1700, 2600, 5000, 10**6])
def test_budget_sized_tiles_keep_the_invariants_and_never_decode_more(width, height, max_area):
    h, w = height // 16, width // 16
    th, tw = vt.choose_tiles(h, w, max_area)
    floor = (min(vt.TILE_LATENTS, h), min(vt.TILE_LATENTS, w))
    if max_area is None or max_area <= floor[0] * floor[1]:
        assert (th, tw) == floor
    else:
        assert th * tw <= max_area
        assert _decoded_latents(h, w, th, tw) <= _decoded_latents(h, w, *floor)
    if max_area is not None and max_area >= h * w:
        assert (th, tw) == (h, w)
    for length, side in ((h, th), (w, tw)):
        assert side >= min(vt.TILE_LATENTS, length) and side <= length
        starts = vt.tile_starts(length, side)
        assert starts[0] == 0 and starts[-1] + side == length or starts == [0]
        assert all(b + vt.OVERLAP_LATENTS <= a + side for a, b in zip(starts, starts[1:]))
        weights = vt.axis_weights(starts, side, length, 2, torch, "cpu")
        total = torch.zeros(length * 2)
        for s, wgt in zip(starts, weights):
            total[s * 2 : s * 2 + wgt.numel()] += wgt
        torch.testing.assert_close(total, torch.ones_like(total))


def test_budget_comes_from_free_vram_and_the_env_caps_it(monkeypatch):
    vae = _diffusers_vae()
    z = torch.zeros(1, 4, 1, 64, 64)
    assert vt.decode_tile_budget(vae, z) == vt.TILE_LATENTS**2
    monkeypatch.delenv(vt.MAX_TILE_ENV)
    assert vt.decode_tile_budget(vae, z) is None
    assert vt.choose_tiles(64, 64, None) == (32, 32)


def test_larger_tiles_decode_like_untiled_when_the_budget_allows():
    vae = _diffusers_vae()
    z = torch.randn(1, 4, 1, 40, 64, generator = torch.Generator().manual_seed(7))
    with torch.no_grad():
        vae.use_tiling = False
        untiled = vae.decode(z).sample
        assert vt.install(vae)
        whole = vt.tiled_decode(vae, z, return_dict = False, max_area = 40 * 64)[0]
        two = vt.tiled_decode(vae, z, return_dict = False, max_area = 40 * 40)[0]
        small = vt.tiled_decode(vae, z, return_dict = False, max_area = 32 * 32)[0]
    assert torch.equal(whole, untiled)
    assert vae._unsloth_last_decode_tile == (32, 32, 32 * 32)
    assert _line_error(two, untiled) <= _line_error(small, untiled)
