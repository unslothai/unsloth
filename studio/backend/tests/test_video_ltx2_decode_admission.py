# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The request guard prices an LTX-2 tiled decode from the tile it takes (measured peaks plus headroom), not from the
flat three-clips estimate that refused LTX-2.5 1280x704x121 on a 6 GiB card."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLLTX2Video"):
    pytest.skip("diffusers without LTX-2", allow_module_level = True)

from core.inference import video_ltx2_vae_tiles as vt  # noqa: E402
from core.inference import video_vae_untiled as U  # noqa: E402

# Colab L4, stock modules, 1280x704x121 (latent 16 x 22 x 40) at a 6 GiB cap: 16x16 tiles, unbudgeted (bf16 clip),
# 3171 MiB decoder intermediates + 624 MiB clip over the weights. At 8 GiB: 16x19, 3765 MiB intermediates.
MEASURED_6GIB_STOCK_MIB = 3171 + 624
MEASURED_8GIB_STOCK_INTERMEDIATES_MIB = 3765
W, H, F = 1280, 704, 121


@pytest.fixture(autouse = True)
def _envs(monkeypatch):
    monkeypatch.delenv(vt.WIDE_TILES_ENV, raising = False)
    monkeypatch.delenv(U.UNTILED_ENV, raising = False)


def _ltx_vae(fused = False):
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
    ).to(torch.bfloat16)
    vae.enable_tiling()
    assert vt.install(vae)
    if fused:
        vae._unsloth_vae_fused_installed = 1
    return vae


def _price(vae, free_mib, **shape):
    return vt.admission_decode_mib(
        vae,
        width = shape.get("width", W),
        height = shape.get("height", H),
        frames = shape.get("frames", F),
        free_mib = free_mib,
    )


def test_tight_card_prices_the_stock_tile_with_its_bf16_clip_and_headroom():
    vae = _ltx_vae()
    # 3.4 GiB left beside the decode-phase weights: nothing budgets, the decode takes 16x16 with a bf16 clip
    assert vt._plan_shape(vae, 1, 16, 22, 40, 3_465 << 20) == ((16, 16), False)
    need = _price(vae, 3_465)
    assert need >= int(MEASURED_6GIB_STOCK_MIB * 1.15)
    assert need <= int(MEASURED_6GIB_STOCK_MIB * 1.25)
    fused = _price(_ltx_vae(fused = True), 3_465)
    assert fused < need
    # fused intermediates at the planner's fused rate (0.071 measured) plus the bf16 clip, 15% over
    assert fused == int(-(-((0.075 * 121 * 256 + 624.0) * 1.15) // 1)) or abs(fused - 3_389) <= 2


def test_old_flat_estimate_refuses_6gib_and_the_priced_decode_admits_it():
    from core.inference.video import video_offload_shortfall_message

    # streaming tier at a 6 GiB cap: ~5.1 GiB available to the server, 1.7 GiB of decode-phase weights, 2.8 GiB floor
    common = dict(
        family = "ltx-2.5",
        floor_mib = 2_800,
        available_mib = 5_150,
        placement = "'streaming' offload",
        width = W,
        height = H,
        frames = F,
        decode_floor_mib = 1_735,
    )
    assert video_offload_shortfall_message(**common) is not None
    fused = _price(_ltx_vae(fused = True), 5_150 - 1_735)
    assert video_offload_shortfall_message(**common, decode_mib = fused) is None
    # the stock modules' 4.4 GiB decode still does not fit there
    stock = _price(_ltx_vae(), 5_150 - 1_735)
    message = video_offload_shortfall_message(**common, decode_mib = stock)
    assert message is not None and "1280x704 at 121 frames" in message


def test_priced_decode_never_lowers_the_denoise_term():
    from core.inference.video import video_offload_shortfall_message

    # the denoise phase (floor + base overhead) binds: a cheap decode price cannot admit it
    assert (
        video_offload_shortfall_message(
            family = "ltx-2.5",
            floor_mib = 6_000,
            available_mib = 7_000,
            placement = "'group' offload",
            width = 768,
            height = 512,
            frames = 49,
            decode_floor_mib = 1_000,
            decode_mib = 100,
        )
        is not None
    )


def test_budgeted_tile_prices_the_fp32_clip():
    vae = _ltx_vae()
    free = 7_000
    tile, budgeted = vt._plan_shape(vae, 1, 16, 22, 40, free << 20)
    assert budgeted and tile != (22, 40)
    need = _price(vae, free)
    fp32_clip = 4 * 3 * 121 * 704 * 1280 / 2**20
    assert need == int(
        -(-((vt.ADMISSION_MIB_PER_FRAME_LATENT * 121 * tile[0] * tile[1] + fp32_clip) * 1.15) // 1)
    )
    # within the room the planner itself budgeted for it
    assert need <= free * 1.15


def test_measured_8gib_stock_tile_is_covered_with_headroom():
    # the 8 GiB cell's 16x19 intermediates, priced at the admission rate with headroom
    assert vt.ADMISSION_MIB_PER_FRAME_LATENT * 121 * 16 * 19 * vt.ADMISSION_HEADROOM >= (
        MEASURED_8GIB_STOCK_INTERMEDIATES_MIB * 1.15
    )


def test_plan_shape_is_the_decode_plan():
    vae = _ltx_vae()
    z = torch.zeros(1, 8, 16, 22, 40)
    for free in (0, 2 << 30, 5 << 30, 8 << 30, 40 << 30):
        assert vt._plan(vae, z, free) == vt._plan_shape(vae, 1, 16, 22, 40, free)


@pytest.mark.parametrize("why", ["not_installed", "tiling_off", "wide_tiles_off", "unknown_free"])
def test_unpriced_decodes_keep_the_flat_estimate(monkeypatch, why):
    vae = _ltx_vae()
    free = 3_000
    if why == "not_installed":
        vt.uninstall(vae)
    elif why == "tiling_off":
        vae.disable_tiling()
    elif why == "wide_tiles_off":
        monkeypatch.setenv(vt.WIDE_TILES_ENV, "0")
    else:
        free = None
    assert _price(vae, free) is None


def test_request_guard_price_reads_the_decode_floor_and_its_switch(monkeypatch):
    from core.inference import video

    vae = _ltx_vae(fused = True)
    state = types.SimpleNamespace(
        pipe = types.SimpleNamespace(vae = vae), vram_floor_mib = 2_800, vram_decode_floor_mib = 1_735
    )
    monkeypatch.delenv(video.VIDEO_TILED_DECODE_PRICE_ENV, raising = False)
    got = video._video_tiled_decode_mib(state, width = W, height = H, frames = F, available_mib = 5_150)
    assert got == _price(vae, 5_150 - 1_735)
    assert (
        video._video_tiled_decode_mib(state, width = None, height = H, frames = F, available_mib = 5_150)
        is None
    )
    monkeypatch.setenv(video.VIDEO_TILED_DECODE_PRICE_ENV, "0")
    assert (
        video._video_tiled_decode_mib(state, width = W, height = H, frames = F, available_mib = 5_150)
        is None
    )


def test_priced_decode_is_never_stricter_than_the_flat_estimate():
    from core.inference.video import video_offload_shortfall_message

    # a short-side clip whose stock floor tile prices above the flat share: the flat estimate's verdict stands
    common = dict(
        family = "ltx-2.5",
        floor_mib = 2_800,
        placement = "'streaming' offload",
        width = 768,
        height = 512,
        frames = 121,
        decode_floor_mib = 1_735,
    )
    vae = _ltx_vae()
    priced = _price(vae, 3_000, width = 768, height = 512)
    flat_need = 1_735 + 2_048 + (4_096 + int(3.0 * 121 * 768 * 512 * 12 / 2**20) - 4_096)
    assert 1_735 + priced > flat_need
    for available in (flat_need - 1, flat_need, flat_need + 500):
        assert (video_offload_shortfall_message(available_mib = available, **common) is None) == (
            video_offload_shortfall_message(available_mib = available, decode_mib = priced, **common)
            is None
        )


def test_budgeted_tile_is_never_priced_above_the_room_it_was_budgeted_in(monkeypatch):
    vae = _ltx_vae()
    monkeypatch.setattr(vt, "ADMISSION_HEADROOM", 3.0)
    assert vt._plan_shape(vae, 1, 16, 22, 40, 7_000 << 20)[1]
    assert _price(vae, 7_000) == 7_000
    # unbudgeted keeps the full price (the decode may still OOM and is refused up front)
    assert _price(vae, 3_465) > 3_465
