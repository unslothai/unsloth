# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The vLLM importance sampling correction is on by default for GRPO.

History: from 2025-10-29 (unsloth #3390, unsloth-zoo #308) every use_vllm GRPO run passed the vLLM
sampling logprobs to grpo_compute_loss, which weighted the loss by clamp(exp(old - sampling), max = cap).
On 2026-01-15 the chunked logprob refactor (unsloth #3628, unsloth-zoo #357) gated those logprobs on
vllm_importance_sampling_correction, whose Unsloth default was False, so the correction silently
stopped. These tests fail if the default, or the wiring that carries the rollout logprobs from
generation to the loss, is dropped again."""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import math
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)

import torch


def _grpo_config_cls():
    import unsloth  # noqa: F401
    import trl

    cls = trl.GRPOConfig
    if "vllm_importance_sampling_correction" not in {f.name for f in dataclasses.fields(cls)}:
        pytest.skip("this TRL has no vLLM importance sampling correction")
    if "_unsloth_patched_rl_config" not in cls.__dict__ and not cls.__name__.startswith("Unsloth"):
        pytest.skip("Unsloth does not patch GRPOConfig here")
    return cls


def test_default_is_on_and_explicit_false_stays_off():
    cls = _grpo_config_cls()
    assert cls(output_dir = "unused").vllm_importance_sampling_correction is True
    assert (
        cls(
            output_dir = "unused", vllm_importance_sampling_correction = False
        ).vllm_importance_sampling_correction
        is False
    )


def test_cap_comes_from_trl_args():
    import trl

    cls = _grpo_config_cls()
    trl_fields = {f.name: f for f in dataclasses.fields(trl.trainer.grpo_config.GRPOConfig)}
    for name in ("vllm_importance_sampling_clip_max", "vllm_importance_sampling_cap"):
        if name in trl_fields:
            assert getattr(cls(output_dir = "unused"), name) == trl_fields[name].default
            assert getattr(cls(output_dir = "unused", **{name: 1.5}), name) == 1.5


def _zoo_loss():
    rl_replacements = pytest.importorskip("unsloth_zoo.rl_replacements")
    loss = getattr(rl_replacements, "grpo_compute_loss", None)
    if loss is None:
        pytest.skip("unsloth_zoo has no grpo_compute_loss")
    return loss


def _loss(trainer, sampling_ratio):
    """grpo_compute_loss fed as grpo_accumulated_loss feeds it from this trainer, on-policy (old = new)."""
    torch.manual_seed(0)
    new = -torch.rand(2, 4, dtype = torch.float64, requires_grad = True)
    old = new.detach().clone()
    sampling = old - math.log(sampling_ratio)
    args = trainer.args
    mode = getattr(args, "vllm_importance_sampling_mode", None) or "token_truncate"
    cap = getattr(args, "vllm_importance_sampling_cap", None)
    clip_min = getattr(args, "vllm_importance_sampling_clip_min", None)
    clip_max = getattr(args, "vllm_importance_sampling_clip_max", None)
    if clip_max is None and cap is not None:
        # grpo_accumulated_loss maps TRL < 1.6's single cap onto the clip bounds the same way.
        clip_min, clip_max = 0, cap
    out = _zoo_loss()(
        None,
        new,
        old,
        sampling,
        torch.zeros(2, 4, dtype = torch.long),
        torch.ones(2, 4),
        0.0,
        torch.tensor([1.0, -0.5], dtype = torch.float64),
        loss_type = "bnpo",
        use_vllm = trainer.use_vllm,
        vllm_importance_sampling_correction = getattr(
            trainer, "vllm_importance_sampling_correction", False
        ),
        vllm_importance_sampling_mode = mode,
        vllm_importance_sampling_cap = cap,
        vllm_importance_sampling_clip_min = clip_min,
        vllm_importance_sampling_clip_max = clip_max,
    )
    return out[0].detach()


def _default_trainer(use_vllm = True, **config):
    cfg = _grpo_config_cls()(output_dir = "unused", **config)
    # TRL's GRPOTrainer.__init__ copies the flag from args.
    return SimpleNamespace(
        args = cfg,
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = cfg.vllm_importance_sampling_correction,
    )


def _cap(trainer):
    args = trainer.args
    cap = getattr(args, "vllm_importance_sampling_clip_max", None)
    return cap if cap is not None else getattr(args, "vllm_importance_sampling_cap", None)


@pytest.mark.parametrize("ratio", [0.5, 2.5, 50.0])
def test_default_weights_the_loss_by_the_truncated_ratio(ratio):
    trainer = _default_trainer()
    unweighted = _loss(trainer, 1.0)
    weight = min(ratio, _cap(trainer))
    torch.testing.assert_close(_loss(trainer, ratio), unweighted * weight)


@pytest.mark.parametrize("case", ["explicit_off", "no_vllm"])
def test_paths_without_the_correction_are_unweighted(case):
    trainer = (
        _default_trainer(vllm_importance_sampling_correction = False)
        if case == "explicit_off"
        else _default_trainer(use_vllm = False)
    )
    torch.testing.assert_close(_loss(trainer, 50.0), _loss(trainer, 1.0))


def test_rollout_logprobs_reach_the_loss():
    """The 2026-01-15 regression dropped these links; each must hold for the default to act."""
    from unsloth.models.rl_replacements import (
        grpo_trainer__generate_and_score_completions,
        grpo_trainer_compute_loss,
    )

    generate = inspect.getsource(grpo_trainer__generate_and_score_completions)
    assert 'output["sampling_per_token_logps"] = sampling_per_token_logps' in generate
    assert (
        'if self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False):'
        in generate
    )

    loss = grpo_trainer_compute_loss("compute_loss", "def compute_loss(self):\n    pass\n")
    assert 'sampling_per_token_logps = inputs.get("sampling_per_token_logps", None)' in loss
    assert "sampling_per_token_logps = sampling_per_token_logps," in loss

    rl_replacements = pytest.importorskip("unsloth_zoo.rl_replacements")
    accumulated = inspect.getsource(rl_replacements.grpo_accumulated_loss)
    assert (
        'kwargs["vllm_importance_sampling_correction"] = getattr(trainer, "vllm_importance_sampling_correction", False)'
        in accumulated
    )


def test_async_grpo_config_keeps_trls_default():
    """AsyncGRPO has its own objective and is not patched by rl.py: its config default must not move."""
    import unsloth  # noqa: F401

    try:
        from trl.trainer.async_grpo_config import AsyncGRPOConfig
    except ImportError:
        try:
            from trl.experimental.async_grpo import AsyncGRPOConfig
        except ImportError:
            pytest.skip("this TRL has no AsyncGRPO")
    fields = {f.name: f for f in dataclasses.fields(AsyncGRPOConfig)}
    if "vllm_importance_sampling_correction" not in fields:
        pytest.skip("AsyncGRPOConfig has no vllm_importance_sampling_correction")
    assert "Unsloth" not in AsyncGRPOConfig.__name__
    assert (
        AsyncGRPOConfig(output_dir = "unused").vllm_importance_sampling_correction
        == fields["vllm_importance_sampling_correction"].default
    )
