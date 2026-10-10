# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Unsloth's generated TRL configs may only move the defaults Unsloth means to move.

A stale override drifts silently when TRL changes its own default (top_k after trl#4695, data_seed via the seed rewrite).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import math

import pytest


# daily-fresh-fetch collects this directory with only pytest installed.
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)

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


# Unsloth refuses GRPO below this TRL (unsloth/models/rl.py): it crashes on the first step there.
GRPO_TRL_FLOOR = "0.20.0"


def _skip_if_trl_is_the_mlx_shim(trl):
    # On Apple Silicon unsloth swaps trl.SFTConfig for an MLX alias (and stubs trl itself when it is
    # absent), so there are no TRL config defaults to compare against.
    if getattr(getattr(trl, "SFTConfig", None), "__name__", "") == "_MLXSFTConfig":
        pytest.skip("trl is unsloth's MLX shim on this platform")


def _grpo_refused():
    import trl
    from packaging.version import Version
    return Version(trl.__version__) < Version(GRPO_TRL_FLOOR)


def _config_cls(name):
    import unsloth  # noqa: F401
    import trl

    _skip_if_trl_is_the_mlx_shim(trl)
    if name == "GRPOConfig" and _grpo_refused():
        pytest.skip(f"unsloth refuses GRPO on trl {trl.__version__} (< {GRPO_TRL_FLOOR})")

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
    import trl
    from packaging.version import Version

    special = {"dr_grpo", "dapo"} if Version(trl.__version__) >= Version("0.22.0") else {"dr_grpo"}
    assert special <= set(_documented_loss_types()), "TRL dropped a loss_type Unsloth special-cases"


def _documented_loss_types():
    import inspect
    src = inspect.getsource(_pristine(_config_cls("GRPOConfig")))
    return [
        lt
        for lt in ("grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo")
        if f'"{lt}"' in src
    ]


def _needs_loss_type(loss_type):
    if loss_type not in _documented_loss_types():
        pytest.skip(f"this TRL has no loss_type={loss_type!r}")


def test_default_loss_type_is_trls_dapo():
    cfg = _grpo()
    if "dapo" not in _documented_loss_types():
        assert (
            cfg.loss_type == "bnpo" and cfg.beta == 0.001
        ), "TRL <= 0.21 has no dapo to default to"
        return
    assert cfg.loss_type == "dapo" and cfg.beta == 0.0


def test_default_dapo_keeps_trl_clip_and_truncation():
    """Only an explicit loss_type="dapo" gets the paper's settings: masking truncated rows zeroes every update when all are."""
    _needs_loss_type("dapo")
    trl_default = _pristine(_config_cls("GRPOConfig"))(output_dir = "unused")
    cfg = _grpo()
    assert cfg.epsilon_high == trl_default.epsilon_high
    assert cfg.mask_truncated_completions is trl_default.mask_truncated_completions is False


@pytest.mark.parametrize(
    "loss_type, beta", [("dapo", 0.0), ("dr_grpo", 0.0), ("bnpo", 0.001), ("grpo", 0.001)]
)
def test_unset_beta_follows_loss_type(loss_type, beta):
    _needs_loss_type(loss_type)
    assert _grpo(loss_type = loss_type).beta == beta


@pytest.mark.parametrize("loss_type", ["dapo", "dr_grpo", "bnpo", "grpo"])
@pytest.mark.parametrize("beta", [0.0, 0.001, 0.04])
def test_explicit_beta_is_kept(loss_type, beta):
    assert _grpo(loss_type = loss_type, beta = beta).beta == beta


def test_cispo_caps_the_is_weight_at_scalerl_epsilon_high():
    """TRL clamps the CISPO weight at epsilon_high itself; its epsilon fallback (0.2) would cap every weight below 1."""
    _needs_loss_type("cispo")
    cfg = _grpo(loss_type = "cispo")
    assert cfg.epsilon_high == 5.0
    assert cfg.mask_truncated_completions is False, "the dapo recommendations leaked into cispo"
    assert cfg.beta == 0.001
    assert _grpo(loss_type = "cispo", epsilon_high = 3.0).epsilon_high == 3.0


@pytest.mark.parametrize("loss_type", ["bnpo", "grpo", "dr_grpo"])
def test_other_loss_types_keep_trl_epsilon_high(loss_type):
    assert (
        _grpo(loss_type = loss_type).epsilon_high
        == _pristine(_config_cls("GRPOConfig"))(
            output_dir = "unused", loss_type = loss_type
        ).epsilon_high
    )


def test_grpo_below_its_trl_floor_is_refused_with_an_upgrade_hint():
    import unsloth  # noqa: F401
    import trl

    _skip_if_trl_is_the_mlx_shim(trl)
    if not _grpo_refused():
        pytest.skip(f"trl {trl.__version__} supports GRPO")
    with pytest.raises(ImportError, match = "GRPO needs trl >= 0.20.0"):
        trl.GRPOConfig(output_dir = "unused")
