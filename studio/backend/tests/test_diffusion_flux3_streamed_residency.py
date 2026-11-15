# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Klein int8 at a 6 GiB budget keeps more of the streamed transformer resident; CPU only.

Numbers from the Colab L4 load log: 5732 MiB free, 6144 MiB int8 DiT, 4619 MiB int8 text encoder, 161 MiB VAE,
703 MiB prefetch window, 647 MiB top-level group, tiled VAE."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_decode_segments as seg
import core.inference.diffusion_measured_peaks as mp
import core.inference.diffusion_memory as dm

LOADED = {
    "transformer": (6144, "dit"),
    "text_encoder": (4619, "text_encoder"),
    "vae": (161, "other"),
}
FREE_6G = 5732
WINDOW = 703


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    for env in (
        mp.LEARNED_PEAKS_ENV,
        dm.MEASURED_ACTIVATION_ENV,
        dm.DENOISE_PHASE_RESIDENCY_ENV,
        dm.STREAMED_RESIDENCY_ENV,
        "UNSLOTH_DIFFUSION_STREAMED_PHASE_SEED",
    ):
        monkeypatch.delenv(env, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    monkeypatch.setattr(dm, "denoiser_quant_label", lambda pipe: "torchao:Int8Tensor")
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: WINDOW)
    monkeypatch.setattr(dm, "_top_level_group_mib", lambda pipe: 647)
    monkeypatch.setattr(dm, "planned_vae_saver_tag", lambda pipe, plan: "tiled")
    return types.SimpleNamespace(_unsloth_phase_seeds = True)


def _place(pipe, speed = "default"):
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", FREE_6G, 6144)
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = memory,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "flux.3-klein"
        ),
        requested_mode = None,
        model_dense_mib = 11105,
        companion_dense_mib = 5283,
        text_encoder_dense_mib = 5092,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = "flux.3-klein", speed_mode = speed)


def test_6gib_int8_sizes_resident_blocks_from_the_streamed_phases(pipe):
    new = _place(pipe)
    # streamed denoise 308 MiB x 1.15, rounded up to 256; slack and window unchanged
    assert new.estimates["measured_denoise_phase_headroom_mib"] == 512
    assert new.estimates["resident_dit_slack_mib"] == 1024
    assert new.resident_transformer_mib == FREE_6G - 1024 - 161 - 512 - WINDOW == 3332
    # the decode lends from the measured tiled decode, not the untiled 2437
    assert new.estimates["measured_decode_peak_mib"] == 904


def test_kill_switch_restores_the_resident_table(pipe, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_STREAMED_PHASE_SEED", "0")
    new = _place(pipe)
    assert new.estimates["measured_denoise_phase_headroom_mib"] == 1280
    assert new.estimates["measured_decode_peak_mib"] == 2437
    assert new.resident_transformer_mib == 2564


def test_unmeasured_cases_keep_the_resident_table():
    assert dm._streamed_phase_seed("flux.3-klein", "default", "torchao:Int8Tensor", True) == (
        308,
        904,
    )
    assert dm._streamed_phase_seed("flux.3-klein", "max", "torchao:Int8Tensor", True) is None
    assert dm._streamed_phase_seed("flux.3-klein", "default", "torchao:Int8Tensor", False) is None
    # an int8 text encoder in the label does not make an fp8 denoiser int8
    assert (
        dm._streamed_phase_seed("flux.3-klein", "default", "torchao:Float8Tensor|te:int8", True)
        is None
    )
    assert (
        dm._streamed_phase_seed("flux.3-klein-base", "default", "torchao:Int8Tensor", True) is None
    )
    assert dm._streamed_phase_seed("qwen-image-2.1", "default", "torchao:Int8Tensor", True) is None


@pytest.fixture
def allocator(monkeypatch):
    calls = []
    fake = types.SimpleNamespace()
    monkeypatch.setattr(seg, "_torch", lambda: fake)
    monkeypatch.setattr(seg, "_busy", lambda torch: False)
    monkeypatch.setattr(seg, "_decide", lambda torch: True)
    monkeypatch.setattr(seg, "_live_settings", lambda torch: "")
    monkeypatch.setattr(seg, "_setter", lambda torch: calls.append)
    monkeypatch.setattr(seg, "_usable", None)
    monkeypatch.setattr(seg, "_streamed", None, raising = False)
    monkeypatch.setattr(seg, "_depth", 0)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_STREAMED_EXPANDABLE", raising = False)
    return calls


def test_streamed_placement_engages_expandable_segments_until_released(allocator):
    assert seg.engage_streamed()
    assert seg.engage_streamed()
    assert allocator == ["expandable_segments:True"]
    assert seg.release_streamed()
    assert allocator == ["expandable_segments:True", "expandable_segments:False", ""]
    assert not seg.release_streamed()


def test_streamed_expandable_kill_switch(allocator, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_STREAMED_EXPANDABLE", "0")
    assert not seg.engage_streamed()
    assert allocator == []


def test_streamed_expandable_leaves_a_configured_allocator(allocator, monkeypatch):
    monkeypatch.setattr(seg, "_live_settings", lambda torch: None)
    assert not seg.engage_streamed()
    assert allocator == []


def test_a_rolled_back_load_releases_streamed_expandable_segments():
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    rollbacks = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.UnaryOp)
        and isinstance(n.test.op, ast.Not)
        and isinstance(n.test.operand, ast.Name)
        and n.test.operand.id == "state_committed"
    ]
    assert rollbacks
    for block in rollbacks:
        called = {
            c.func.id
            for c in ast.walk(block)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        assert "_release_streamed_expandable" in called
