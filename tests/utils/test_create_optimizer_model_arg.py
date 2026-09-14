# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# transformers 5.x passes model positionally when optimizer creation is delayed (FSDP);
# 4.x passes nothing. The override must satisfy both.

import inspect

import pytest

import unsloth  # noqa: F401  (must precede transformers/trl)
from unsloth.trainer import UnslothTrainer


def test_create_optimizer_accepts_a_positional_model():
    parameters = inspect.signature(UnslothTrainer.create_optimizer).parameters
    assert "model" in parameters, (
        "UnslothTrainer.create_optimizer must accept `model`; transformers 5.x calls "
        "self.create_optimizer(model) positionally on the delayed-creation path."
    )
    assert parameters["model"].default is None, (
        "`model` must default to None so transformers 4.x, which calls "
        "create_optimizer() with no argument, keeps working."
    )


def test_create_optimizer_is_compatible_with_the_installed_transformers():
    from transformers import Trainer

    base = inspect.signature(Trainer.create_optimizer).parameters
    ours = inspect.signature(UnslothTrainer.create_optimizer).parameters
    for name in base:
        if name == "self":
            continue
        assert name in ours, (
            f"transformers Trainer.create_optimizer takes `{name}` but the Unsloth "
            f"override does not, so transformers can call it in a way we reject."
        )


def test_create_optimizer_does_not_raise_typeerror_on_a_positional_model():
    """A bare object() suffices: the arity TypeError fired before self was ever touched."""
    try:
        UnslothTrainer.create_optimizer(object(), "prepared-model")
    except TypeError as error:
        message = str(error)
        if "positional argument" in message and "create_optimizer" in message:
            pytest.fail(f"create_optimizer rejected a positional model: {message}")
    except Exception:
        pass  # reached the body and failed on the fake self: the expected outcome
