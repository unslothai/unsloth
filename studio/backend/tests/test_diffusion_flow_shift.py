# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Families sample at ComfyUI's static sigma shift: the loader rebuilds the shipped scheduler. The Cosmos3 rows rebuild
it with their model cards' settings instead (scheduler_overrides)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from core.inference.diffusion_families import comfy_flow_shift_for, detect_family
from core.inference.diffusion_flow_shift import (
    apply_comfy_flow_shift,
    apply_scheduler_overrides,
    flow_shift_overrides,
    install_sample_sigmas,
    pipe_sample_sigmas,
    sample_sigmas_for_steps,
)
from core.inference.video_families import VideoFamily, detect_video_family


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


# ── Cosmos3 ──────────────────────────────────────────────────────────────────

# The shipped scheduler_config.json of nvidia/Cosmos3-Nano@e59a53c2 and nvidia/Cosmos3-Edge@344d602b (public Hub
# metadata, fixtures/cosmos3_checkpoints.json): UniPC at flow_shift 1.0 with Karras sigmas on. Each row carries its
# card's from_config values as scheduler_overrides; only Edge's (Karras off, shift 12.0) change the sigmas.
_COSMOS3_CHECKPOINTS = json.loads(
    (Path(__file__).parent / "fixtures" / "cosmos3_checkpoints.json").read_text(encoding = "utf-8")
)
# Keyed "<repo>@<revision>", one revision per repo.
_COSMOS3_SCHEDULERS = {
    key.partition("@")[0]: entry.get("scheduler_config")
    for key, entry in _COSMOS3_CHECKPOINTS.items()
}
_COSMOS3_NANO = _COSMOS3_SCHEDULERS["nvidia/Cosmos3-Nano"]
_COSMOS3_EDGE = _COSMOS3_SCHEDULERS["nvidia/Cosmos3-Edge"]


class _Log:
    """Records ``logger.<level>(fmt, *args)``; the backend logger is structlog, which caplog does not see."""

    def __init__(self):
        self.lines = []

    def info(self, fmt, *args, **kwargs):
        self.lines.append(("info", fmt % args if args else fmt))

    def warning(self, fmt, *args, **kwargs):
        self.lines.append(("warning", fmt % args if args else fmt))


class _RejectingScheduler(_FakeScheduler):
    @classmethod
    def from_config(cls, config, **overrides):
        # What UniPC raises for a value it does not implement (solver_type="bh3", for one). The message does not name
        # the overrides, so a warning that names them has to do so itself.
        raise NotImplementedError(f"bh3 is not implemented for {cls}")


@pytest.mark.parametrize(
    "name, config",
    [("cosmos3-nano", _COSMOS3_NANO), ("cosmos3-edge", _COSMOS3_EDGE)],
    ids = ["nano", "edge"],
)
def test_cosmos3_scheduler_overrides_rebuild_the_shipped_scheduler_once(name, config):
    # UniPC has no set_shift, and both shipped scheduler_config.json keep flow_shift 1.0 with Karras sigmas on, so
    # the cards' settings arrive by rebuilding the scheduler once at load.
    overrides = detect_video_family("", override = name).scheduler_overrides
    shipped = _FakeScheduler(**config)
    pipe = types.SimpleNamespace(scheduler = shipped)
    log = _Log()
    assert apply_scheduler_overrides(pipe, overrides, log) is True
    rebuilt = pipe.scheduler
    assert type(rebuilt) is _FakeScheduler and rebuilt is not shipped
    # The overrides land and every other shipped key survives.
    assert rebuilt.config == {**config, **dict(overrides)}
    assert [level for level, _line in log.lines] == ["info"]
    assert all(key in log.lines[0][1] for key, _value in overrides)
    # Idempotent: applied again, the rebuilt scheduler stays.
    assert apply_scheduler_overrides(pipe, overrides, log) is False
    assert pipe.scheduler is rebuilt and len(log.lines) == 1


def test_cosmos3_scheduler_overrides_already_in_the_config_change_nothing():
    overrides = detect_video_family("nvidia/Cosmos3-Edge").scheduler_overrides
    held = _FakeScheduler(**{**_COSMOS3_EDGE, "flow_shift": 12.0, "use_karras_sigmas": False})
    pipe = types.SimpleNamespace(scheduler = held)
    assert apply_scheduler_overrides(pipe, overrides) is False
    assert pipe.scheduler is held
    # Half of Edge's pair is not enough: with Karras on the shift is never read, so it still rebuilds.
    half = _FakeScheduler(**{**_COSMOS3_EDGE, "flow_shift": 12.0})
    pipe = types.SimpleNamespace(scheduler = half)
    assert apply_scheduler_overrides(pipe, overrides) is True
    assert pipe.scheduler.config["use_karras_sigmas"] is False


def test_empty_cosmos3_overrides_and_pipes_without_a_scheduler_are_no_ops():
    shipped = _FakeScheduler(**_COSMOS3_NANO)
    # The field's default, which every non-Cosmos3 family keeps: the shipped scheduler stays.
    default = VideoFamily.__dataclass_fields__["scheduler_overrides"].default_factory()
    for overrides in (default, (), None):
        pipe = types.SimpleNamespace(scheduler = shipped)
        assert apply_scheduler_overrides(pipe, overrides) is False
        assert pipe.scheduler is shipped
    overrides = detect_video_family("nvidia/Cosmos3-Nano").scheduler_overrides
    for pipe in (
        types.SimpleNamespace(),
        types.SimpleNamespace(scheduler = None),
        types.SimpleNamespace(scheduler = object()),  # no config to rebuild from
    ):
        assert apply_scheduler_overrides(pipe, overrides) is False


def test_a_scheduler_rejecting_cosmos3_overrides_keeps_the_shipped_schedule():
    overrides = detect_video_family("nvidia/Cosmos3-Edge").scheduler_overrides
    shipped = _RejectingScheduler(**_COSMOS3_EDGE)
    pipe = types.SimpleNamespace(scheduler = shipped)
    log = _Log()
    # The load has already taken minutes: keep the shipped schedule and say why, rather than fail it.
    assert apply_scheduler_overrides(pipe, overrides, log) is False
    assert pipe.scheduler is shipped
    assert [level for level, _line in log.lines] == ["warning"]
    warning = log.lines[0][1]
    assert all(key in warning for key, _value in overrides) and "not implemented" in warning
    assert apply_scheduler_overrides(pipe, overrides) is False  # no logger: the same, silently
    assert pipe.scheduler is shipped
