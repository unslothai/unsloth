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
    # the live tiler's reading: everything above the decode-phase weights is reusable here
    monkeypatch.setattr(U, "_free_bytes", lambda device: (5_150 - 1_735) << 20)
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


# ── load: whole-module offload refined to streaming with the connectors streamed ──


class LTX2TextConnectors(torch.nn.Module):
    def __init__(self, mib):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(mib * 2**19, dtype = torch.bfloat16))


def _mib_module(mib):
    m = torch.nn.Module()
    m.w = torch.nn.Parameter(torch.zeros(mib * 2**19, dtype = torch.bfloat16))
    return m


def _ltx_pipe():
    import types as _t

    pipe = _t.SimpleNamespace()
    # small-card shape: a 24 MiB DiT, 6 MiB connectors, 2 MiB VAE, against an 8 MiB budget
    pipe.transformer = _mib_module(24)
    pipe.connectors = LTX2TextConnectors(6)
    pipe.vae = _mib_module(2)
    pipe.components = {
        "transformer": pipe.transformer,
        "connectors": pipe.connectors,
        "vae": pipe.vae,
    }
    return pipe


def _model_plan(budget_mib):
    import core.inference.diffusion_memory as dm
    return dm.MemoryPlan(
        requested_mode = "auto",
        offload_policy = dm.OFFLOAD_MODEL,
        vae_tiling = True,
        vae_slicing = True,
        device_memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib, budget_mib),
        estimates = {"safe_device_budget_mib": budget_mib},
        reasons = (),
    )


def test_small_card_load_streams_instead_of_counting_the_connectors_resident(monkeypatch):
    import core.inference.diffusion_memory as dm

    monkeypatch.delenv(dm.STREAM_RUN_ONCE_COMPANIONS_ENV, raising = False)
    out = dm.refine_memory_plan_for_components(_ltx_pipe(), _model_plan(7))
    # streamed: only the 2 MiB VAE stays; the 6 MiB connectors fit only off the card, pinned or not
    assert out.offload_policy == dm.OFFLOAD_STREAMING
    assert out.estimates["streaming_resident_mib"] == 2
    assert dm.streams_companions_unpinned(out)
    # the streaming switch keeps them resident, and then streaming cannot fit either: the plan stands
    monkeypatch.setenv(dm.STREAM_RUN_ONCE_COMPANIONS_ENV, "0")
    assert dm.refine_memory_plan_for_components(_ltx_pipe(), _model_plan(7)).offload_policy == (
        dm.OFFLOAD_MODEL
    )


def test_decode_guard_never_credits_memory_a_cuda_graph_pool_holds(monkeypatch):
    from core.inference import video

    # 3,465 MiB driver free, 1,735 MiB of decode-phase weights, a 2,000 MiB CUDA-graph private pool: the request reading
    # (free + reserved) is 7,200 MiB, the tiler's (free + reusable cache) 3,465 MiB
    vae = _ltx_vae()
    state = types.SimpleNamespace(
        pipe = types.SimpleNamespace(vae = vae), vram_floor_mib = 2_800, vram_decode_floor_mib = 1_735
    )
    monkeypatch.delenv(video.VIDEO_TILED_DECODE_PRICE_ENV, raising = False)
    monkeypatch.setattr(U, "_free_bytes", lambda device: 3_465 << 20)
    decode_mib = video._video_tiled_decode_mib(
        state, width = W, height = H, frames = F, available_mib = 7_200
    )
    # priced at the unbudgeted 16x16 tile the decode really takes, plus the 2,000 MiB it cannot reuse
    assert decode_mib == _price(vae, 3_465) + 2_000
    assert (
        video.video_offload_shortfall_message(
            family = "ltx-2.5",
            floor_mib = 2_800,
            available_mib = 7_200,
            placement = "'streaming' offload",
            width = W,
            height = H,
            frames = F,
            decode_floor_mib = 1_735,
            decode_mib = decode_mib,
        )
        is not None
    )


def test_a_fused_module_that_fell_back_prices_the_stock_rate():
    vae = _ltx_vae(fused = True)
    fused = _price(vae, 3_465)
    next(iter(vae.decoder.modules()))._unsloth_vae_fused_failed = True
    assert _price(vae, 3_465) == _price(_ltx_vae(), 3_465) > fused


def _stub_streaming(
    monkeypatch,
    budget_mib,
    prefetch = True,
):
    import sys

    import core.inference.diffusion_memory as dm

    pytest.importorskip("diffusers.hooks")
    calls = []

    def apply_group_offloading(
        module,
        onload_device = None,
        offload_device = None,
        offload_type = None,
        num_blocks_per_group = None,
        use_stream = False,
        non_blocking = False,
        record_stream = False,
        low_cpu_mem_usage = False,
    ):
        calls.append((module, offload_type, use_stream, low_cpu_mem_usage))

    monkeypatch.setattr(
        sys.modules["diffusers.hooks"], "apply_group_offloading", apply_group_offloading
    )
    for name in (
        dm.GROUP_OFFLOAD_PIN_ENV,
        dm.BACKGROUND_PIN_ENV,
        dm.STREAM_RUN_ONCE_COMPANIONS_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    for name, value in (
        ("host_pinning_allowed", True),
        ("_streaming_prefetch_enabled", prefetch),
        ("_pinned_memory_capped", False),
        ("_pin_top_level_group", True),
    ):
        monkeypatch.setattr(dm, name, lambda *a, _v = value, **k: _v)
    for name in (
        "install_group_offload_buffer_restore",
        "install_group_offload_hooks_eager",
        "install_group_pin_wait",
        "install_group_prefetch",
        "_skip_top_level_copy_back",
        "_pin_vision_embedding_device",
    ):
        monkeypatch.setattr(dm, name, lambda *a, **k: None)
    monkeypatch.setattr(
        dm,
        "_apply_group_offloading_packed",
        lambda apply, module, kwargs, logger: apply(module, **kwargs),
    )
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: budget_mib)
    return calls


@pytest.mark.parametrize("why", ["pin_budget", "no_prefetch"])
def test_refined_streaming_load_never_places_the_connectors_on_the_card(monkeypatch, why):
    """The refinement counts the connectors streamed; the installer must not move them on the card when it cannot pin
    them: they stream unpinned (no copy stream, nothing pinned) instead."""
    import core.inference.diffusion_memory as dm

    calls = _stub_streaming(
        monkeypatch, 1 if why == "pin_budget" else 1 << 20, prefetch = why != "no_prefetch"
    )
    pipe = _ltx_pipe()
    moved = []
    pipe.vae.to = lambda *a, **k: pipe.vae
    pipe.connectors.to = lambda *a, **k: moved.append(a) or pipe.connectors
    out = dm.refine_memory_plan_for_components(pipe, _model_plan(7))
    assert out.offload_policy == dm.OFFLOAD_STREAMING
    dm._apply_streaming_offload(
        pipe, "cuda", None, stream_companions_unpinned = dm.streams_companions_unpinned(out)
    )
    hooked = {id(m): (t, stream, low) for m, t, stream, low in calls}
    assert hooked[id(pipe.connectors)] == ("leaf_level", False, True)
    assert moved == []


def test_streaming_connectors_stay_resident_only_when_switched_off(monkeypatch):
    import core.inference.diffusion_memory as dm

    calls = _stub_streaming(monkeypatch, 1)
    monkeypatch.setenv(dm.STREAM_RUN_ONCE_COMPANIONS_ENV, "0")
    pipe = _ltx_pipe()
    moved = []
    pipe.vae.to = lambda *a, **k: pipe.vae
    pipe.connectors.to = lambda *a, **k: moved.append(a) or pipe.connectors
    dm._apply_streaming_offload(pipe, "cuda", None)
    assert all(m is not pipe.connectors for m, *_ in calls)
    assert moved


def test_a_card_the_connectors_fit_keeps_them_resident_when_it_cannot_pin_them(monkeypatch):
    """Main's placement wherever main's plan works: the connectors fit beside the rest (9 MiB budget, 8 MiB resident),
    so the plan carries no unpinned request and a short pin budget keeps them on the card."""
    import core.inference.diffusion_memory as dm

    calls = _stub_streaming(monkeypatch, 1)
    pipe = _ltx_pipe()
    moved = []
    pipe.vae.to = lambda *a, **k: pipe.vae
    pipe.connectors.to = lambda *a, **k: moved.append(a) or pipe.connectors
    out = dm.refine_memory_plan_for_components(pipe, _model_plan(9))
    assert out.offload_policy == dm.OFFLOAD_STREAMING
    assert out.estimates["streaming_resident_mib"] == 8
    assert not dm.streams_companions_unpinned(out)
    dm._apply_streaming_offload(
        pipe, "cuda", None, stream_companions_unpinned = dm.streams_companions_unpinned(out)
    )
    assert all(m is not pipe.connectors for m, *_ in calls)
    assert moved


def test_apply_memory_plan_hands_the_unpinned_request_to_the_streaming_installer(monkeypatch):
    import core.inference.diffusion_memory as dm

    seen = []
    monkeypatch.setattr(
        dm,
        "_apply_streaming_offload",
        lambda *a, **k: seen.append(k.get("stream_companions_unpinned")),
    )
    monkeypatch.setattr(dm, "_enable_vae_saver", lambda *a, **k: False)
    pipe = _ltx_pipe()
    for flagged in (True, False):
        plan = _model_plan(7)
        estimates = dict(plan.estimates)
        if flagged:
            estimates[dm.STREAM_COMPANIONS_UNPINNED_KEY] = 1
        import dataclasses

        plan = dataclasses.replace(
            plan,
            offload_policy = dm.OFFLOAD_STREAMING,
            estimates = estimates,
            vae_tiling = False,
            vae_slicing = False,
        )
        try:
            dm.apply_memory_plan(pipe, plan, device = "cuda")
        except Exception:  # noqa: BLE001 -- only the installer call is under test
            pass
    assert seen == [True, False]
