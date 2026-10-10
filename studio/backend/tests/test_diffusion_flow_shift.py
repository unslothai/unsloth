# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Families sample at ComfyUI's static sigma shift: the loader rebuilds the shipped scheduler."""

from __future__ import annotations

import types

from core.inference.diffusion_families import comfy_flow_shift_for, detect_family
from core.inference.diffusion_flow_shift import (
    apply_comfy_flow_shift,
    flow_shift_overrides,
    install_sample_sigmas,
    pipe_sample_sigmas,
    sample_sigmas_for_steps,
)
from core.inference.video_families import detect_video_family


class _Config(dict):
    pass


class _FakeScheduler:
    def __init__(self, **config):
        self.config = _Config(config)

    @classmethod
    def from_config(cls, config, **overrides):
        return cls(**{**config, **overrides})


# Shipped configs (the fields that matter) of the Qwen-Image, Wan2.2-TI2V-5B and HunyuanVideo-1.5 720p repos.
_QWEN = dict(
    shift = 1.0, use_dynamic_shifting = True, shift_terminal = 0.02, time_shift_type = "exponential"
)
_WAN = dict(flow_shift = 5.0, use_flow_sigmas = True, solver_order = 2)
_HV15 = dict(shift = 9.0, use_dynamic_shifting = False, shift_terminal = None)


def test_flow_match_gets_a_static_shift_without_terminal_stretch():
    pipe = types.SimpleNamespace(scheduler = _FakeScheduler(**_QWEN))
    assert apply_comfy_flow_shift(pipe, 3.1) is True
    cfg = pipe.scheduler.config
    assert (cfg["shift"], cfg["use_dynamic_shifting"], cfg["shift_terminal"]) == (3.1, False, None)
    assert cfg["time_shift_type"] == "exponential"  # untouched keys survive
    assert apply_comfy_flow_shift(pipe, 3.1) is False  # idempotent


def test_unipc_flow_shift_and_static_euler():
    wan = types.SimpleNamespace(scheduler = _FakeScheduler(**_WAN))
    assert apply_comfy_flow_shift(wan, 8.0) and wan.scheduler.config["flow_shift"] == 8.0
    hv = types.SimpleNamespace(scheduler = _FakeScheduler(**_HV15))
    assert apply_comfy_flow_shift(hv, 7.0) and hv.scheduler.config["shift"] == 7.0


def test_no_shift_or_unknown_scheduler_is_a_no_op():
    pipe = types.SimpleNamespace(scheduler = _FakeScheduler(**_QWEN))
    assert apply_comfy_flow_shift(pipe, None) is False
    other = types.SimpleNamespace(scheduler = _FakeScheduler(beta_schedule = "scaled_linear"))
    assert apply_comfy_flow_shift(other, 3.0) is False
    assert flow_shift_overrides(_Config(beta_schedule = "x"), 3.0) is None
    assert apply_comfy_flow_shift(types.SimpleNamespace(), 3.0) is False


def test_family_shifts_match_comfy_defaults():
    assert detect_family("Qwen/Qwen-Image").comfy_flow_shift == 3.1
    assert detect_family("Qwen/Qwen-Image-Edit-2511").comfy_flow_shift == 3.1
    assert detect_family("Tongyi-MAI/Z-Image").comfy_flow_shift == 3.0
    # Families whose shipped schedule already matches keep it.
    assert detect_family("black-forest-labs/FLUX.1-schnell").comfy_flow_shift is None
    assert detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers").comfy_flow_shift == 8.0
    assert detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers").comfy_flow_shift == 5.0
    for repo in (
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    ):
        assert detect_video_family(repo).comfy_flow_shift == 7.0


def test_qwen_image_edit_2509_keeps_its_template_shift():
    # Comfy-Org image_qwen_image_edit_2509 uses ModelSamplingAuraFlow 3, the 2511 template 3.1.
    fam = detect_family("Qwen/Qwen-Image-Edit-2509")
    assert fam.name == "qwen-image-edit"
    assert comfy_flow_shift_for(fam, None, "Qwen/Qwen-Image-Edit-2509", None) == 3.0
    assert (
        comfy_flow_shift_for(
            fam, "qwen-image-edit-2509-Q4_K_M.gguf", "unsloth/Qwen-Image-Edit-2509-GGUF", None
        )
        == 3.0
    )
    assert comfy_flow_shift_for(fam, None, "Qwen/Qwen-Image-Edit-2511", None) == 3.1
    assert comfy_flow_shift_for(fam, None, "unsloth/Qwen-Image-Edit-2511-GGUF", None) == 3.1
    assert comfy_flow_shift_for(detect_family("Qwen/Qwen-Image"), "Qwen/Qwen-Image") == 3.1


def test_sample_sigmas_follow_the_checkpoint_grid():
    grid = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
    assert sample_sigmas_for_steps(grid, 8) == list(grid)
    for steps in (2, 4, 6, 12, 16):
        out = sample_sigmas_for_steps(grid, steps)
        assert len(out) == steps and out[0] == grid[0] and abs(out[-1] - grid[-1]) < 1e-12
        assert all(b < a for a, b in zip(out, out[1:]))
    assert sample_sigmas_for_steps(grid, 1) == [1.0]

    def _pipe(config = None):
        return types.SimpleNamespace(config = _Config(config or {}))

    pipe = _pipe()
    assert install_sample_sigmas(pipe, list(grid)) == grid
    assert pipe_sample_sigmas(pipe) == grid and pipe.config == {}
    assert install_sample_sigmas(_pipe(), None) is None
    # A malformed grid is refused rather than handed to the scheduler, and the pipe stays untouched.
    for bad in ([0.5, 0.8], [1.5, 0.5], [1.0, "x"], [], 0.5):
        pipe = _pipe()
        assert install_sample_sigmas(pipe, bad) is None
        assert pipe_sample_sigmas(pipe) is None
    # A pipeline that already carries one (diffusers with #14950) keeps it without reading the file.
    assert install_sample_sigmas(_pipe({"sample_sigmas": [1.0, 0.5]}), None) == (1.0, 0.5)
