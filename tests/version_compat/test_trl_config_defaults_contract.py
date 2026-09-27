# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Unsloth's generated TRL configs may only move the defaults Unsloth means to move.

Every other field must equal TRL's own default for the installed TRL. A default
Unsloth writes over the signature silently goes stale when TRL changes its own:
top_k=None meant "TRL default" until trl#4695 (0.27) made the default 0, after
which None let transformers 5 fall back to the model's top_k; the `seed` rewrite
also hit `data_seed`. Both only showed up as different samples and data order.

CPU only, no downloads: builds each config with no arguments.
"""

from __future__ import annotations

import dataclasses
import importlib
import math

import pytest


# Fields Unsloth changes on purpose (rl.py replacements, extra_args, or TRL deriving them from those).
INTENDED = {
    "auto_find_batch_size",
    "beta",
    "bf16",
    "dataloader_pin_memory",
    "dataset_num_proc",
    "eval_accumulation_steps",
    "fp16",
    "generation_kwargs",
    "gradient_accumulation_steps",
    "gradient_checkpointing_kwargs",
    "include_num_input_tokens_seen",
    "include_tokens_per_second",
    "learning_rate",
    "logging_nan_inf_filter",
    "logging_steps",
    "loss_type",
    "num_generations",
    "optim",
    "padding_free",
    "per_device_eval_batch_size",
    "per_device_train_batch_size",
    "report_to",
    "router_aux_loss_coef",
    "seed",
    "steps_per_generation",
    "torch_empty_cache_steps",
    "vllm_importance_sampling_correction",
    "vllm_mode",
    "warmup_ratio",
    "warmup_steps",
    "weight_decay",
}

CONFIGS = [
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
    "RLOOConfig",
    "KTOConfig",
    "ORPOConfig",
    "CPOConfig",
    "RewardConfig",
    "GKDConfig",
    "OnlineDPOConfig",
]


def _pristine(cls):
    while "_unsloth_patched_rl_config" in cls.__dict__ or cls.__name__.startswith("Unsloth"):
        cls = cls.__mro__[1]
    return cls


def _config_cls(name):
    import unsloth  # noqa: F401
    import trl

    cls = getattr(trl, name, None)
    if cls is None:
        try:
            cls = getattr(
                importlib.import_module("trl.experimental." + name[: -len("Config")].lower()), name
            )
        except Exception:
            return None
    return cls


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b)
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


@pytest.mark.parametrize("name", CONFIGS)
def test_only_intended_defaults_differ_from_trl(name):
    cls = _config_cls(name)
    if cls is None:
        pytest.skip(f"this TRL has no {name}")
    pristine = _pristine(cls)
    if pristine is cls:
        pytest.skip(f"Unsloth does not patch {name} on this TRL")
    ours, theirs = cls(output_dir = "unused"), pristine(output_dir = "unused")
    moved = {
        f.name: (getattr(ours, f.name, None), getattr(theirs, f.name, None))
        for f in dataclasses.fields(theirs)
        if f.name not in INTENDED
        and not f.name.endswith("_dir")
        and not _same(getattr(ours, f.name, None), getattr(theirs, f.name, None))
    }
    assert not moved, f"{name} defaults moved away from TRL's (unsloth, trl): {moved}"


@pytest.mark.parametrize("name", ["GRPOConfig", "RLOOConfig", "OnlineDPOConfig"])
def test_top_k_default_is_trls(name):
    cls = _config_cls(name)
    if cls is None or not hasattr(_pristine(cls), "top_k"):
        pytest.skip(f"this TRL has no {name}.top_k")
    assert cls(output_dir = "unused").top_k == _pristine(cls)(output_dir = "unused").top_k


def test_seed_default_does_not_leak_into_data_seed():
    cls = _config_cls("SFTConfig")
    assert cls(output_dir = "unused").data_seed is None
    assert cls(output_dir = "unused", seed = 1).data_seed is None


def _grpo(**kwargs):
    return _config_cls("GRPOConfig")(output_dir = "unused", **kwargs)


def test_dapo_fills_only_unset_recommendations():
    cfg = _grpo(loss_type = "dapo")
    assert cfg.epsilon_high == 0.28 and cfg.mask_truncated_completions is True
    cfg = _grpo(loss_type = "dapo", epsilon_high = 0.2, mask_truncated_completions = False)
    assert cfg.epsilon_high == 0.2, "an explicit epsilon_high was overwritten"
    assert (
        cfg.mask_truncated_completions is False
    ), "an explicit mask_truncated_completions was overwritten"


@pytest.mark.parametrize("loss_type", ["bnpo", "grpo", "dr_grpo"])
def test_other_loss_types_keep_trl_mask_default(loss_type):
    assert _grpo(loss_type = loss_type).mask_truncated_completions is False


def test_trl_fields_the_overrides_key_on_are_unchanged():
    """The loss-type overrides compare against these TRL defaults; a new spelling needs the overrides revisited."""
    fields = {f.name: f for f in dataclasses.fields(_pristine(_config_cls("GRPOConfig")))}
    assert fields["scale_rewards"].default in (True, "group"), fields["scale_rewards"].default
    assert fields["epsilon_high"].default is None, fields["epsilon_high"].default
    assert fields["mask_truncated_completions"].default is False
    assert set(("dr_grpo", "dapo")) <= set(
        _documented_loss_types()
    ), "TRL dropped a loss_type Unsloth special-cases"


def _documented_loss_types():
    import inspect
    src = inspect.getsource(_pristine(_config_cls("GRPOConfig")))
    return [
        lt
        for lt in ("grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo")
        if f'"{lt}"' in src
    ]
