# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Low VRAM keeps a video denoiser streamed for every request.

Low VRAM streams the denoiser in block groups. The per-request residency of Wan / HunyuanVideo-1.5 / LTX-2 / LTX-2.5
(video_stream_residency) fills free memory with resident groups under any group plan: Wan2.1 14B int8 at 832x480x33
on an A100 40 GB kept all 41 groups (13774 MiB) resident and peaked at 23.0 GiB under Low VRAM, above the 17.0 GiB
whole-module offload had."""

from __future__ import annotations

import inspect

import pytest

import core.inference.diffusion_memory as dm
import core.inference.video as video
import core.inference.video_stream_residency as vsr


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(vsr.VIDEO_DIT_RESIDENT_ENV, raising = False)
    monkeypatch.delenv(dm.LOW_VRAM_PROMOTE_ENV, raising = False)


def _applies(memory_mode, family = "wan2.1-t2v-14b"):
    return vsr.applies(
        family, is_moe = False, offload_policy = "group", device = "cuda:0", memory_mode = memory_mode
    )


@pytest.mark.parametrize(
    "family", ["wan2.1-t2v-14b", "wan2.2-ti2v-5b", "hunyuanvideo-1.5", "ltx-2", "ltx-2.5"]
)
def test_low_vram_keeps_the_denoiser_streamed(family):
    assert _applies(dm.MEMORY_MODE_BALANCED, family)
    assert _applies(None, family)
    assert _applies("auto", family)
    assert not _applies(dm.MEMORY_MODE_LOW_VRAM, family)
    assert not _applies("low-vram", family)


def test_the_promote_switch_restores_residency_under_low_vram(monkeypatch):
    monkeypatch.setenv(dm.LOW_VRAM_PROMOTE_ENV, "1")
    assert _applies(dm.MEMORY_MODE_LOW_VRAM)


def test_an_unreadable_mode_keeps_residency():
    assert _applies("not-a-mode")
    assert not dm.keeps_denoiser_streamed("not-a-mode")


def test_the_render_passes_the_loaded_memory_mode():
    src = inspect.getsource(video.VideoBackend.generate)
    call = src.split("video_stream_residency.applies(", 1)[1].split(")\n", 1)[0]
    assert 'memory_mode = getattr(state, "memory_mode", None)' in call
