# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO loss must resolve `importance_sampling_level` on every TRL in the declared window.

`grpo_trainer_compute_loss` hands both loss paths -- `grpo_compute_loss_slow` (no-grad) and
`grpo_accumulated_loss` (gradient) -- an `importance_sampling_level`. GSPO introduced that knob
in TRL 0.20.0, as a `GRPOConfig` field the trainer copies onto itself in `__init__`:

    0.18.2, 0.19.1   neither the config field nor the trainer attribute: token level by construction
    0.20.0 - 1.13.0  `GRPOConfig.importance_sampling_level`, copied to `self.importance_sampling_level`

Reading `self.importance_sampling_level` unconditionally therefore died at the bottom of the
declared `trl>=0.18.2,!=0.19.0` window:

    unsloth_compiled_cache/UnslothGRPOTrainer.py: in compute_loss
        importance_sampling_level = self.importance_sampling_level,
    E   AttributeError: 'UnslothGRPOTrainer' object has no attribute 'importance_sampling_level'

The table above is read off the real sources; `test_the_window_still_binds_what_this_loss_expects`
re-derives it from upstream, so a TRL that moves the knob again fails here rather than in a user's
training loop.
"""

from __future__ import annotations

import re

import pytest

from unsloth.models.rl_replacements import grpo_trainer_compute_loss


class _Stop(Exception):
    """Raised by the loss stubs once the kwarg under test has been captured."""


class _Args:
    loss_type = "grpo"
    max_completion_length = 16
    delta = None
    temperature = 1.0
    unsloth_num_chunks = -1


class _Accelerator:
    num_processes = 1


def _trainer(
    *,
    on_trainer = None,
    on_config = None,
    with_logps: bool,
):
    """A stand-in GRPOTrainer binding `importance_sampling_level` where the TRL under test does."""
    import torch

    args = _Args()
    if on_config is not None:
        args.importance_sampling_level = on_config

    class _Trainer:
        pass

    trainer = _Trainer()
    trainer.args = args
    trainer.accelerator = _Accelerator()
    trainer.beta = 0.0
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    if on_trainer is not None:
        trainer.importance_sampling_level = on_trainer
    # `with_logps` picks the branch: a tensor takes the no-grad slow path, None the gradient one.
    trainer._get_per_token_logps = lambda *a, **k: (torch.zeros((1, 2)) if with_logps else None)
    return trainer


def _captured_level(
    *,
    on_trainer = None,
    on_config = None,
    with_logps: bool,
) -> str:
    """Compile the real rewritten compute_loss and read the level it hands the loss."""
    torch = pytest.importorskip("torch")
    source = grpo_trainer_compute_loss("compute_loss", None)
    captured = {}

    def _loss_stub(*args, **kwargs):
        captured["level"] = kwargs.get("importance_sampling_level", "<not passed>")
        raise _Stop

    namespace = {
        "torch": torch,
        "inspect": __import__("inspect"),
        "grpo_compute_loss_slow": _loss_stub,
        "grpo_accumulated_loss": _loss_stub,
        "_unsloth_grpo_vision_inputs": lambda inputs: {},
        "_unsloth_fix_mm_token_type_ids": lambda *a, **k: None,
        "_unsloth_get_model_config": lambda model: object(),
        "_unsloth_get_final_logit_softcapping": lambda model: 0,
        "detect_logit_transforms": None,
        "sanitize_logprob": lambda x: x,
    }
    exec(
        compile(re.sub(r"^    ", "", source, flags = re.MULTILINE), "<rewritten>", "exec"), namespace
    )

    inputs = {
        "prompt_ids": torch.zeros((1, 2), dtype = torch.long),
        "prompt_mask": torch.ones((1, 2), dtype = torch.long),
        "completion_ids": torch.zeros((1, 2), dtype = torch.long),
        "completion_mask": torch.ones((1, 2), dtype = torch.long),
        "advantages": torch.zeros((1,)),
    }
    trainer = _trainer(on_trainer = on_trainer, on_config = on_config, with_logps = with_logps)
    with pytest.raises(_Stop):
        namespace["compute_loss"](trainer, object(), inputs)
    return captured["level"]


@pytest.mark.parametrize("with_logps", [True, False], ids = ["no_grad_path", "gradient_path"])
def test_the_floor_binds_neither_and_still_reaches_the_loss(with_logps: bool) -> None:
    """0.18.2 and 0.19.1: no field, no attribute, and token level is what they implement."""
    assert _captured_level(with_logps = with_logps) == "token"


@pytest.mark.parametrize("with_logps", [True, False], ids = ["no_grad_path", "gradient_path"])
def test_the_trainer_attribute_still_wins_where_trl_sets_it(with_logps: bool) -> None:
    """0.20.0 and up: unchanged behaviour, the trainer's own value is forwarded verbatim."""
    assert (
        _captured_level(
            on_trainer = "sequence",
            on_config = "token",
            with_logps = with_logps,
        )
        == "sequence"
    )


@pytest.mark.parametrize("with_logps", [True, False], ids = ["no_grad_path", "gradient_path"])
def test_the_config_field_is_used_when_only_it_is_bound(with_logps: bool) -> None:
    """A config carrying the knob is honoured even if __init__ never copied it across."""
    assert _captured_level(on_config = "sequence", with_logps = with_logps) == "sequence"


def test_the_window_still_binds_what_this_loss_expects() -> None:
    """Re-derive the table in the docstring from upstream, so a moved knob fails here."""
    from tests.version_compat._fetch import fetch_text

    expected = {
        "v0.18.2": False,
        "v0.19.1": False,
        "v0.20.0": True,
        "v0.24.0": True,
        "v1.13.0": True,
    }
    for tag, bound in expected.items():
        config = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_config.py")
        trainer = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
        assert config is not None and trainer is not None, f"{tag}: GRPO sources not found"
        has_field = bool(re.search(r"^\s{4}importance_sampling_level\s*:", config, re.M))
        has_attr = bool(
            re.search(
                r"^\s+self\.importance_sampling_level\s*=\s*args\.importance_sampling_level",
                trainer,
                re.M,
            )
        )
        assert has_field == bound, f"{tag}: GRPOConfig field bound={has_field}, expected {bound}"
        assert has_attr == bound, f"{tag}: trainer attribute bound={has_attr}, expected {bound}"
