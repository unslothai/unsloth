# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Images page sampler choice and the v-prediction markers of single-file SDXL checkpoints."""

from __future__ import annotations

import types

import pytest

diffusers = pytest.importorskip("diffusers")

from core.inference.diffusion_sampler import (
    apply_checkpoint_prediction_type,
    build_scheduler,
    checkpoint_prediction_overrides,
    sampler_options,
)
from core.inference.image_gallery import _params_text

# stabilityai/stable-diffusion-xl-base-1.0 scheduler/scheduler_config.json
_SDXL = dict(
    beta_end = 0.012,
    beta_schedule = "scaled_linear",
    beta_start = 0.00085,
    clip_sample = False,
    interpolation_type = "linear",
    num_train_timesteps = 1000,
    prediction_type = "epsilon",
    sample_max_value = 1.0,
    set_alpha_to_one = False,
    skip_prk_steps = True,
    steps_offset = 1,
    timestep_spacing = "leading",
    trained_betas = None,
    use_karras_sigmas = False,
)


def _sdxl(**overrides):
    return diffusers.EulerDiscreteScheduler.from_config({**_SDXL, **overrides})


def test_sdxl_offers_the_popular_samplers():
    options = {o["id"]: o["schedule_types"] for o in sampler_options(_sdxl())}
    assert list(options) == ["default", "euler", "euler_a", "dpmpp_2m", "dpmpp_2m_sde"]
    assert options["dpmpp_2m"] == ["automatic", "uniform", "karras", "exponential"]
    # EulerAncestralDiscreteScheduler has no sigma-schedule flags.
    assert options["euler_a"] == ["automatic", "uniform"]


def test_flow_matching_offers_nothing():
    assert sampler_options(diffusers.FlowMatchEulerDiscreteScheduler()) == []
    assert sampler_options(None) == []


def test_default_keeps_the_shipped_scheduler():
    assert build_scheduler(_sdxl(), None, None) == (None, None)
    assert build_scheduler(_sdxl(), "default", "automatic") == (None, None)


@pytest.mark.parametrize(
    "sampler, schedule, cls, algorithm, flags",
    [
        ("dpmpp_2m", "automatic", "DPMSolverMultistepScheduler", "dpmsolver++", "karras"),
        ("dpmpp_2m_sde", None, "DPMSolverMultistepScheduler", "sde-dpmsolver++", "exponential"),
        ("dpmpp_2m", "uniform", "DPMSolverMultistepScheduler", "dpmsolver++", "uniform"),
        ("euler", "karras", "EulerDiscreteScheduler", None, "karras"),
        ("default", "exponential", "EulerDiscreteScheduler", None, "exponential"),
    ],
)
def test_build_scheduler(sampler, schedule, cls, algorithm, flags):
    built, resolved = build_scheduler(_sdxl(use_karras_sigmas = True), sampler, schedule)
    assert type(built).__name__ == cls
    assert resolved == flags
    assert built.config.use_karras_sigmas is (flags == "karras")
    assert built.config.use_exponential_sigmas is (flags == "exponential")
    if algorithm:
        assert built.config.algorithm_type == algorithm


def test_euler_a_automatic_keeps_the_config_spacing():
    built, resolved = build_scheduler(_sdxl(), "euler_a", None)
    assert type(built).__name__ == "EulerAncestralDiscreteScheduler"
    assert resolved is None
    assert built.config.timestep_spacing == "leading"


def test_prediction_settings_carry_over():
    vpred = _sdxl(
        prediction_type = "v_prediction",
        rescale_betas_zero_snr = True,
        timestep_spacing = "trailing",
    )
    for sampler in ("euler_a", "dpmpp_2m", "dpmpp_2m_sde"):
        built, _ = build_scheduler(vpred, sampler, None)
        assert built.config.prediction_type == "v_prediction"
        assert built.config.rescale_betas_zero_snr is True
        assert built.config.timestep_spacing == "trailing"


def test_zero_snr_automatic_avoids_exponential():
    vpred = _sdxl(prediction_type = "v_prediction", rescale_betas_zero_snr = True)
    built, resolved = build_scheduler(vpred, "dpmpp_2m_sde", "automatic")
    assert resolved == "karras"
    assert built.config.use_karras_sigmas is True
    # An explicit choice is honoured.
    assert build_scheduler(vpred, "dpmpp_2m_sde", "exponential")[1] == "exponential"


def test_unavailable_choices_are_refused():
    with pytest.raises(ValueError, match = "does not apply"):
        build_scheduler(_sdxl(), "euler_a", "karras")
    with pytest.raises(ValueError, match = "not available"):
        build_scheduler(diffusers.FlowMatchEulerDiscreteScheduler(), "euler", None)
    with pytest.raises(ValueError, match = "Unknown sampler"):
        build_scheduler(_sdxl(), "lms", None)


def _checkpoint(path, *markers):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    tensors = {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(2)}
    tensors.update({m: torch.zeros(0) for m in markers})
    save_file(tensors, str(path))
    return str(path)


def test_vpred_ztsnr_markers(tmp_path):
    pipe = types.SimpleNamespace(scheduler = _sdxl())
    path = _checkpoint(tmp_path / "noob.safetensors", "v_pred", "ztsnr")
    assert apply_checkpoint_prediction_type(pipe, path) is True
    config = pipe.scheduler.config
    assert type(pipe.scheduler).__name__ == "EulerDiscreteScheduler"
    assert config.prediction_type == "v_prediction"
    assert config.rescale_betas_zero_snr is True
    assert config.timestep_spacing == "trailing"
    # Idempotent: a second pass changes nothing.
    assert apply_checkpoint_prediction_type(pipe, path) is False


def test_vpred_marker_without_ztsnr(tmp_path):
    path = _checkpoint(tmp_path / "vpred.safetensors", "v_pred")
    assert checkpoint_prediction_overrides(path) == {"prediction_type": "v_prediction"}


def test_unmarked_and_unreadable_checkpoints_keep_epsilon(tmp_path):
    pipe = types.SimpleNamespace(scheduler = _sdxl())
    plain = _checkpoint(tmp_path / "sdxl.safetensors")
    assert apply_checkpoint_prediction_type(pipe, plain) is False
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    assert apply_checkpoint_prediction_type(pipe, str(junk)) is False
    assert apply_checkpoint_prediction_type(pipe, None) is False
    assert pipe.scheduler.config.prediction_type == "epsilon"


def _meta(**extra):
    return dict(
        prompt = "a cat", steps = 25, guidance = 7, seed = 1, width = 1024, height = 1024, model = "m", **extra
    )


def test_parameters_text_names_the_sampler():
    text = _params_text(_meta(sampler = "dpmpp_2m", schedule_type = "karras"))
    assert "Steps: 25, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7," in text
    assert "Sampler" not in _params_text(_meta())
    assert "Schedule type" not in _params_text(_meta(sampler = "euler_a"))


def test_route_refuses_a_sampler_the_engine_cannot_run():
    from routes.inference import _sampler_kwargs

    def native_generate(*, prompt):
        return None

    def diffusers_generate(
        *,
        prompt,
        sampler = None,
        schedule_type = None,
    ):
        return None

    assert _sampler_kwargs(native_generate, None, None) == {}
    assert _sampler_kwargs(native_generate, "default", "automatic") == {}
    with pytest.raises(ValueError, match = "diffusers engine only"):
        _sampler_kwargs(native_generate, "dpmpp_2m", None)
    assert _sampler_kwargs(diffusers_generate, "euler", "karras") == {
        "sampler": "euler",
        "schedule_type": "karras",
    }
