# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The vLLM importance sampling correction is on by default, with token-level truncated importance sampling.

TRL's default vllm_importance_sampling_mode = "sequence_mask" weights every token of a completion by
the product of its token ratios. Unsloth's vLLM logprobs are renormalised over the min_p / top_p kept
set while the trainer scores the full vocabulary, so each token ratio sits below 1 and the product
decays with length. Unsloth resolves an unset mode to "token_truncate"; an explicit mode is kept. The
correction itself defaults on (as in TRL): colocated vLLM and the trainer score the same weights
differently, so even num_iterations = 1 rollouts are slightly off-policy."""

from __future__ import annotations

import dataclasses
import importlib.util
import math

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)

import torch


def _grpo_config():
    import unsloth  # noqa: F401
    import trl
    from packaging.version import Version

    if Version(trl.__version__) < Version("0.20.0"):
        pytest.skip("unsloth refuses GRPO on this trl")
    cls = trl.GRPOConfig
    fields = {f.name for f in dataclasses.fields(cls)}
    if "vllm_importance_sampling_mode" not in fields:
        pytest.skip(
            "this TRL has no vllm_importance_sampling_mode (the zoo loss already uses token_truncate)"
        )
    if "_unsloth_patched_rl_config" not in cls.__dict__ and not cls.__name__.startswith("Unsloth"):
        pytest.skip("Unsloth does not patch GRPOConfig here")
    return cls


def test_unset_mode_resolves_to_token_truncate():
    cfg = _grpo_config()(output_dir = "unused")
    assert cfg.vllm_importance_sampling_mode == "token_truncate"
    assert cfg.vllm_importance_sampling_correction is True
    assert cfg.vllm_importance_sampling_clip_max == 3.0
    assert cfg.vllm_importance_sampling_clip_min is None


def test_correction_on_without_mode_uses_token_truncate_and_says_so(capsys):
    cfg = _grpo_config()(output_dir = "unused", vllm_importance_sampling_correction = True)
    assert cfg.vllm_importance_sampling_correction is True
    assert cfg.vllm_importance_sampling_mode == "token_truncate"
    assert "token_truncate" in capsys.readouterr().out


@pytest.mark.parametrize(
    "mode", ["sequence_mask", "sequence_truncate", "token_mask", "token_truncate"]
)
def test_explicit_mode_is_kept(mode, capsys):
    cfg = _grpo_config()(
        output_dir = "unused",
        vllm_importance_sampling_correction = True,
        vllm_importance_sampling_mode = mode,
    )
    assert cfg.vllm_importance_sampling_mode == mode
    assert "on by default" not in capsys.readouterr().out


def test_correction_off_prints_nothing(capsys):
    cfg = _grpo_config()(output_dir = "unused", vllm_importance_sampling_correction = False)
    assert cfg.vllm_importance_sampling_correction is False
    assert "on by default" not in capsys.readouterr().out


def _is_weights(
    mode,
    d,
    length,
    clip_max = 3.0,
):
    """Mean flat_is_ratio and normalised ESS via unsloth_zoo.rl_replacements.grpo_compute_loss."""
    import unsloth_zoo.rl_replacements as rr

    rows = len(length)
    cols = max(length)
    mask = torch.zeros(rows, cols, dtype = torch.long)
    for i, n in enumerate(length):
        mask[i, :n] = 1
    new = torch.full((rows, cols), -1.0, dtype = torch.float64)
    sampling = new - d
    out = rr.grpo_compute_loss(
        None,
        new,
        new.clone(),
        sampling,
        torch.zeros_like(mask),
        mask,
        0.0,
        torch.linspace(-1, 1, rows, dtype = torch.float64),
        loss_type = "dapo",
        epsilon_low = 0.2,
        epsilon_high = 0.28,
        max_completion_length = cols,
        num_items_in_batch = mask.sum(),
        current_gradient_accumulation_steps = 1,
        steps_per_generation = 1,
        num_processes = 1,
        importance_sampling_level = "token",
        use_vllm = True,
        vllm_importance_sampling_correction = True,
        vllm_importance_sampling_mode = mode,
        vllm_importance_sampling_clip_min = None,
        vllm_importance_sampling_clip_max = clip_max,
    )
    w = torch.broadcast_to(out[4].double(), mask.shape)[mask.bool()]
    return w.mean().item(), (w.sum() ** 2 / (w.numel() * (w * w).sum())).item()


def test_default_keeps_signal_where_sequence_mask_decays():
    """The measured gap: -0.035 nats per token (trainer below vLLM) on 30-300 token completions."""
    lengths = [30, 60, 100, 150, 300]
    seq_w, seq_ess = _is_weights("sequence_mask", -0.035, lengths)
    tok_w, tok_ess = _is_weights("token_truncate", -0.035, lengths)
    assert math.isclose(tok_w, math.exp(-0.035), rel_tol = 1e-9)
    assert tok_ess == pytest.approx(1.0)
    assert seq_w < 0.15 and seq_ess < 0.5
