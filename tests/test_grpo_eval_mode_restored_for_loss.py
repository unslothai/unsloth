# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Asserts model.training is still False at compute_loss, since generation flips eval batches."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rl_source import load_rl_wrapper  # noqa: E402


RL_REPLACEMENTS_PATH = (
    Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
)
HELPER_NAME = "_unsloth_grpo_accumulation_steps"
WRAPPER_NAME = "_wrap_grpo_generate_and_score"


def _load_divisor_helper():
    tree = ast.parse(RL_REPLACEMENTS_PATH.read_text(encoding = "utf-8"))
    found = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == HELPER_NAME
    ]
    assert len(found) == 1, f"expected one module-level def {HELPER_NAME}, found {len(found)}"
    namespace: dict = {}
    exec(
        compile(ast.Module(body = found, type_ignores = []), str(RL_REPLACEMENTS_PATH), "exec"),
        namespace,
    )
    return namespace[HELPER_NAME]


def _load_generate_wrapper():
    return load_rl_wrapper((WRAPPER_NAME,))[WRAPPER_NAME]


class _Model:
    """Only the surface `_wrap_grpo_generate_and_score` touches."""

    def __init__(self, training):
        self.training = training
        self.calls = []

    def for_training(self, use_gradient_checkpointing = True):
        self.calls.append("for_training")
        self.training = True

    def for_inference(self):
        self.calls.append("for_inference")
        self.training = False


class _Trainer:
    current_gradient_accumulation_steps = 4

    def __init__(
        self,
        training,
        raises = False,
    ):
        self.model = _Model(training)
        self.raises = raises

    def _generate_and_score_completions(self, *args, **kwargs):
        # Mirrors the injection at rl_replacements.py: unconditional, eval batches included.
        self.model.for_training(use_gradient_checkpointing = True)
        if self.raises:
            raise RuntimeError("generation blew up")
        return "completions"


def _wrapped_trainer_cls(**kwargs):
    cls = type("Wrapped", (_Trainer,), {})
    _load_generate_wrapper()(cls)
    return cls(**kwargs)


def test_eval_mode_survives_generation_so_the_divisor_is_one():
    """The property the fix rests on, asserted end to end."""
    trainer = _wrapped_trainer_cls(training = False)
    trainer._generate_and_score_completions()
    assert trainer.model.training is False, "generation left the model in training mode"
    assert _load_divisor_helper()(trainer) == 1


def test_training_mode_survives_generation_so_the_window_is_kept():
    trainer = _wrapped_trainer_cls(training = True)
    trainer._generate_and_score_completions()
    assert trainer.model.training is True
    assert _load_divisor_helper()(trainer) == 4


def test_generation_really_does_flip_the_flag_midway():
    """Guard the guard: if the injection ever stops firing, this file proves nothing."""
    trainer = _wrapped_trainer_cls(training = False)
    trainer._generate_and_score_completions()
    assert "for_training" in trainer.model.calls
    assert trainer.model.calls[-1] == "for_inference"


def test_eval_mode_is_restored_even_when_generation_raises():
    trainer = _wrapped_trainer_cls(training = False, raises = True)
    with pytest.raises(RuntimeError):
        trainer._generate_and_score_completions()
    assert trainer.model.training is False
    assert _load_divisor_helper()(trainer) == 1


def test_wrapping_twice_does_not_stack():
    cls = type("Twice", (_Trainer,), {})
    wrap = _load_generate_wrapper()
    wrap(cls)
    first = cls._generate_and_score_completions
    wrap(cls)
    assert cls._generate_and_score_completions is first
    assert getattr(first, "_unsloth_restore_training_wrapped", False) is True


def test_a_trainer_without_the_method_is_left_alone():
    cls = type("Bare", (), {})
    _load_generate_wrapper()(cls)
    assert not hasattr(cls, "_generate_and_score_completions")
