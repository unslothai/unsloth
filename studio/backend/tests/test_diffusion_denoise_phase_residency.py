# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A partially streamed denoiser sizes its resident blocks from the denoise phase; CPU only.

Numbers from a Qwen-Image-2.1 int8 load on an A100 under an 8 GiB budget: 7482 MiB free, 6922 MiB DiT, 644 MiB VAE,
625 MiB prefetch window, tiled + sliced VAE."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_measured_peaks as mp
import core.inference.diffusion_memory as dm

LOADED = {
    "transformer": (6922, "dit"),
    "text_encoder": (8917, "text_encoder"),
    "vae": (644, "other"),
}
WINDOW = 625
FREE_8G = 7482
QUANT = "torchao:Int8Tensor"


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _flat_plan(free_mib: int, total_mib: int):
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(free_mib - reserve, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "qwen-image-2.1"
        ),
        requested_mode = None,
        model_dense_mib = 19630,
        companion_dense_mib = 12182,
        text_encoder_dense_mib = 10847,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return plan


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "peaks.json"
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(path))
    for env in (
        mp.LEARNED_PEAKS_ENV,
        mp.PHASE_PEAKS_ENV,
        dm.MEASURED_ACTIVATION_ENV,
        dm.PARTIAL_RESIDENT_ENV,
        dm.DENOISE_PHASE_RESIDENCY_ENV,
        dm.STREAMED_RESIDENCY_ENV,
        dm.RESIDENT_DIT_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(mp, "current_device_name", lambda: "NVIDIA A100-SXM4-40GB")
    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA A100-SXM4-40GB")
    return path


@pytest.fixture
def pipe(monkeypatch, store):
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    monkeypatch.setattr(dm, "denoiser_quant_label", lambda pipe: QUANT)
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: WINDOW)
    monkeypatch.setattr(dm, "planned_vae_saver_tag", lambda pipe, plan: "tiled+sliced")
    return types.SimpleNamespace(_unsloth_phase_seeds = True)


def _refine(
    pipe,
    plan,
    family = "qwen-image-2.1",
    speed = "default",
):
    return dm.refine_plan_from_loaded_weights(
        pipe, plan, family = family, speed_mode = speed, attention = None
    )


def _record(peak, phases = None):
    mp.record(
        family = "qwen-image-2.1",
        quant = QUANT,
        speed_mode = "default",
        width = 1024,
        height = 1024,
        batch_size = 1,
        peak_mib = peak,
        device_name = "NVIDIA A100-SXM4-40GB",
        phases = phases or {},
    )


def _old_room() -> int:
    """The whole-render placement: the 1849 MiB seed x margin (2304) instead of the denoise phase."""
    return FREE_8G - 1024 - 644 - 2304 - WINDOW


def test_8gib_sizes_resident_blocks_from_the_denoise_phase(pipe):
    plan = _flat_plan(FREE_8G, 8192)
    assert plan.offload_policy == dm.OFFLOAD_STREAMING
    new = _refine(pipe, plan)
    # denoise seed 666 MiB x 1.15, rounded up to 256 MiB; the window is charged once beside it
    assert new.estimates["measured_denoise_phase_headroom_mib"] == 768
    assert new.resident_transformer_mib == FREE_8G - 1024 - 644 - 768 - WINDOW == 4421
    assert new.resident_transformer_mib > _old_room() == 2885
    # the whole-render reserve is unchanged (request pricing, encode room)
    assert new.estimates["measured_runtime_headroom_mib"] == 2304
    assert (
        new.estimates["encode_resident_transformer_mib"]
        == plan.estimates["safe_device_budget_mib"] - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 644
    )
    # the decode borrows the blocks back: tiled decode seed, 1 MP bucket, linear
    assert dm.decode_lend_from_plan(new) == (4421, 1697, 1024 * 1024, False)
    assert "denoise phase" in new.reasons[-1]


@pytest.mark.parametrize("top, resident", [(3000, 4421), (4421, 4421), (4422, None)])
def test_a_streamed_top_level_group_is_charged_beside_the_window(pipe, monkeypatch, top, resident):
    """The top-level group goes resident first; one larger than the room is onloaded whole every forward."""
    monkeypatch.setattr(dm, "_top_level_group_mib", lambda pipe: top)
    new = _refine(pipe, _flat_plan(FREE_8G, 8192))
    if resident is None:
        assert "measured_denoise_phase_headroom_mib" not in new.estimates
        assert new.resident_transformer_mib == _old_room()
    else:
        assert new.resident_transformer_mib == resident


def test_kill_switch_restores_the_whole_render_placement(pipe, monkeypatch):
    monkeypatch.setenv(dm.DENOISE_PHASE_RESIDENCY_ENV, "0")
    new = _refine(pipe, _flat_plan(FREE_8G, 8192))
    assert new.resident_transformer_mib == _old_room()
    assert "measured_denoise_phase_headroom_mib" not in new.estimates
    assert "decode_lend_transformer_mib" not in new.estimates
    assert dm.decode_lend_from_plan(new) is None


def test_untiled_decode_reads_the_untiled_seed(pipe, monkeypatch):
    monkeypatch.setattr(dm, "planned_vae_saver_tag", lambda pipe, plan: "full")
    new = _refine(pipe, _flat_plan(FREE_8G, 8192))
    assert dm.decode_lend_from_plan(new)[1] == 7648


def test_learned_phases_charge_the_window_once(pipe):
    # what the 8 GiB render recorded: pre-decode 1189 (window included), decode 1739
    _record(
        1739,
        {
            mp.PHASE_PRE_DECODE: 1189,
            mp.PHASE_DECODE: 1739,
            mp.PHASE_DECODE_UNPADDED: 1739,
            mp.PHASE_UNPADDED: 1739,
        },
    )
    new = _refine(pipe, _flat_plan(FREE_8G, 8192))
    # 1189 x 1.15 = 1368 less the depth-1 window the sample surely holds (2 of 3 blocks: 416) = 952 -> 1024
    assert new.estimates["measured_denoise_phase_headroom_mib"] == 1024
    assert new.resident_transformer_mib == FREE_8G - 1024 - 644 - 1024 - WINDOW == 4165
    assert dm.decode_lend_from_plan(new) == (4165, 1739, 1024 * 1024, False)


def test_a_deeper_prefetch_never_credits_more_window_than_the_sample_held(pipe, monkeypatch):
    """A sample recorded at depth 2 read under depth 8: only the depth-1 share of the current window is credited."""
    import core.inference.diffusion_offload_prefetch as pf

    _record(
        1739,
        {
            mp.PHASE_PRE_DECODE: 1189,
            mp.PHASE_DECODE: 1739,
            mp.PHASE_DECODE_UNPADDED: 1739,
            mp.PHASE_UNPADDED: 1739,
        },
    )
    window = 208 * 9  # the same blocks at depth 8
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: window)
    monkeypatch.setenv(pf.PREFETCH_DEPTH_ENV, "8")
    new = _refine(pipe, _flat_plan(FREE_8G, 8192))
    # credited 1872 x 2 // 9 = 416 MiB, not the whole 1872 MiB window
    assert new.estimates["measured_denoise_phase_headroom_mib"] == 1024
    assert new.resident_transformer_mib == FREE_8G - 1024 - 644 - 1024 - window


def test_a_larger_learned_pre_decode_peak_raises_the_reserve(pipe):
    _record(
        3000,
        {
            mp.PHASE_PRE_DECODE: 2400,
            mp.PHASE_DECODE: 3000,
            mp.PHASE_DECODE_UNPADDED: 3000,
            mp.PHASE_UNPADDED: 3000,
        },
    )
    new = _refine(pipe, _flat_plan(FREE_8G, 8192))
    # ceil(2400 x 1.15) - 416 = 2344, rounded up to 2560, against a 3584 MiB whole-render reserve (3000 x 1.15)
    assert new.estimates["measured_runtime_headroom_mib"] == 3584
    assert new.estimates["measured_denoise_phase_headroom_mib"] == 2560
    assert new.resident_transformer_mib == FREE_8G - 1024 - 644 - 2560 - WINDOW


def test_learned_peak_without_phases_keeps_the_whole_render_placement(pipe):
    _record(1739)
    new = _refine(pipe, _flat_plan(FREE_8G, 8192))
    assert "measured_denoise_phase_headroom_mib" not in new.estimates
    assert new.resident_transformer_mib == _old_room()


def test_unseeded_target_keeps_the_whole_render_placement(pipe):
    pipe._unsloth_phase_seeds = False
    new = _refine(pipe, _flat_plan(FREE_8G, 8192))
    assert "measured_denoise_phase_headroom_mib" not in new.estimates


def test_unmeasured_family_keeps_its_placement(pipe):
    plan = _flat_plan(FREE_8G, 8192)
    new = _refine(pipe, plan, family = "qwen-image")
    assert "measured_denoise_phase_headroom_mib" not in new.estimates


def test_12gib_whole_transformer_is_untouched(pipe):
    new = _refine(pipe, _flat_plan(11578, 12288))
    assert new.resident_transformer_mib == 6922
    assert "measured_denoise_phase_headroom_mib" not in new.estimates
    assert dm.decode_lend_from_plan(new) is None


def _reserved_pipe(
    phase,
    learned = None,
    cfg_off = False,
):
    reserve = (
        2304,
        "qwen-image-2.1",
        "default",
        None,
        None,
        learned,
        None,
        None,
        True,
        cfg_off,
        False,
        False,
        False,
        phase,
    )
    return types.SimpleNamespace(
        _unsloth_measured_reserve = reserve,
        _unsloth_phase_seeds = True,
        _unsloth_peak_identity = None,
    )


@pytest.mark.parametrize("phase", [None, 768])
def test_a_request_within_the_reserve_releases_nothing(store, phase):
    assert dm.measured_request_extra_mib(_reserved_pipe(phase), width = 1024, height = 1024) == 0


def test_a_larger_request_first_returns_the_phase_blocks(store):
    old = dm.measured_request_extra_mib(_reserved_pipe(None), width = 2048, height = 2048)
    assert old == 8704 - 2304
    new = dm.measured_request_extra_mib(_reserved_pipe(768), width = 2048, height = 2048)
    assert new == old + (2304 - 768)
    # a reserve tuple written by an older load prices as before
    short = _reserved_pipe(None)
    short._unsloth_measured_reserve = short._unsloth_measured_reserve[:13]
    assert dm.measured_request_extra_mib(short, width = 2048, height = 2048) == old


def test_a_conditioned_request_returns_the_phase_blocks(store):
    old = dm.measured_request_extra_mib(
        _reserved_pipe(None), width = 1024, height = 1024, condition_pixels = 1024 * 1024
    )
    new = dm.measured_request_extra_mib(
        _reserved_pipe(768), width = 1024, height = 1024, condition_pixels = 1024 * 1024
    )
    assert old > 0 and new == old + 2304 - 768


def test_a_smaller_canvas_releases_nothing(store):
    assert dm.measured_request_extra_mib(_reserved_pipe(768), width = 512, height = 512) == 0


def test_another_workflow_is_priced_against_the_denoise_reserve(store):
    pipe = _reserved_pipe(768)
    pipe._unsloth_peak_identity = ("qwen-image-2.1", QUANT, "default")
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, workflow = "txt2img") == 0
    assert (
        dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, workflow = "img2img")
        == 2304 - 768
    )
    pipe = _reserved_pipe(None)
    pipe._unsloth_peak_identity = ("qwen-image-2.1", QUANT, "default")
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, workflow = "img2img") == 0


def test_the_flat_fallback_is_priced_against_the_denoise_reserve(store):
    """Placed from a CFG-off sample, a CFG request takes the flat estimate: it must clear the denoise reserve, not the
    whole-render one the phase sizing no longer holds during the steps."""
    flat = dm.estimate_image_runtime_mib(width = 1024, height = 1024, family = "qwen-image-2.1")
    old = dm.measured_request_extra_mib(
        _reserved_pipe(None, learned = 1739, cfg_off = True), width = 1024, height = 1024, cfg = True
    )
    new = dm.measured_request_extra_mib(
        _reserved_pipe(768, learned = 1739, cfg_off = True), width = 1024, height = 1024, cfg = True
    )
    assert old == max(0, flat - 2304)
    assert new == max(0, flat - 768) > old


def _deferring_pipe(phase):
    return _reserved_pipe(phase)


def _held(calls, name, released_mib):
    def restore():
        calls.append(name)

    restore.released_mib = released_mib
    return restore


def test_an_oversized_request_holds_its_restore_until_a_request_needs_the_blocks(store):
    calls = []
    pipe = _deferring_pipe(768)
    assert dm.defer_resident_restore(pipe, _held(calls, "edit", 4631))
    assert calls == []
    # the next edit releases about as much: the groups stay streamed, nothing re-pinned or released
    carried, extra = dm.take_deferred_restore(pipe, 4550)
    assert carried is not None and extra == 0 and calls == []
    assert getattr(pipe, dm.DEFERRED_RESTORE_ATTR) is None
    assert dm.defer_resident_restore(pipe, carried)
    # a text-to-image request the placement was sized for pins them back before it runs
    carried, extra = dm.take_deferred_restore(pipe, 0)
    assert carried is None and extra == 0 and calls == ["edit"]
    assert dm.take_deferred_restore(pipe, 0) == (None, 0) and calls == ["edit"]


def test_a_larger_release_carries_the_held_groups_and_releases_the_rest(store):
    calls = []
    pipe = _deferring_pipe(768)
    dm.defer_resident_restore(pipe, _held(calls, "held", 3000))
    carried, extra = dm.take_deferred_restore(pipe, 4000)
    assert carried is not None and extra == 1000 and calls == []


def test_a_much_smaller_release_pins_the_held_groups_back_first(store):
    calls = []
    pipe = _deferring_pipe(768)
    dm.defer_resident_restore(pipe, _held(calls, "held", 4631))
    assert dm.take_deferred_restore(pipe, 1000) == (None, 1000)
    assert calls == ["held"]


def test_held_restores_chain_and_both_run(store):
    calls = []
    pipe = _deferring_pipe(768)
    dm.defer_resident_restore(pipe, _held(calls, "a", 1000))
    dm.defer_resident_restore(pipe, _held(calls, "b", 2000))
    assert getattr(pipe, dm.DEFERRED_RESTORE_ATTR).released_mib == 3000
    dm.take_deferred_restore(pipe, 0)
    assert calls == ["a", "b"]


@pytest.mark.parametrize("phase, env", [(None, None), (768, "0")])
def test_whole_render_placements_restore_at_once(store, monkeypatch, phase, env):
    if env is not None:
        monkeypatch.setenv(dm.DENOISE_PHASE_RESIDENCY_ENV, env)
    pipe = _deferring_pipe(phase)
    assert not dm.defer_resident_restore(pipe, _held([], "x", 4631))
    assert dm.take_deferred_restore(pipe, 4631) == (None, 4631)


def test_a_request_counts_the_room_whole_groups_left_unused(store, monkeypatch):
    """The L4 8 GiB edit: room 4718 MiB, 24 groups kept 4631 MiB. Priced at 4550 MiB against the room, the request
    needs 4463 MiB of groups streamed out, which leaves the top-level group resident (as the whole-render placement
    did), not 4550, which reached into it."""
    mib = 1024 * 1024
    groups = [types.SimpleNamespace(_unsloth_resident_bytes = 131 * mib)] + [
        types.SimpleNamespace(_unsloth_resident_bytes = 4500 * mib // 23) for _ in range(23)
    ]
    dit = types.SimpleNamespace(_unsloth_resident_room = 4718)
    monkeypatch.setattr(dm, "_offload_groups", lambda m: groups if m is dit else [])
    pipe = _reserved_pipe(768)
    pipe.components = {"transformer": dit}
    assert dm._unused_room_mib(pipe) == 4718 - (131 + 4499)
    assert dm.take_deferred_restore(pipe, 4550) == (None, 4550 - 88)
    # the whole-render placement prices as before
    pipe = _reserved_pipe(None)
    pipe.components = {"transformer": dit}
    assert dm.take_deferred_restore(pipe, 4550) == (None, 4550)
