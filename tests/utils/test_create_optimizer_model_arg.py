# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# UnslothTrainer.create_optimizer must accept the model transformers 5.x passes it.
#
# transformers 5.x declares `Trainer.create_optimizer(self, model=None)` and, on the
# delayed optimizer creation path (FSDP, SageMaker MP, FSDP-XLA), calls
# `self.create_optimizer(model)` POSITIONALLY with the accelerator-prepared model.
# transformers 4.x declares `create_optimizer(self)` and calls it with no argument.
#
# Before the fix, the override took only `self`, so on transformers 5.x every FSDP run
# died with "create_optimizer() takes 1 positional argument but 2 were given" before the
# first step. These tests pin both halves of that contract.

import inspect

import pytest

import unsloth  # noqa: F401  (must precede transformers/trl)
from unsloth.trainer import UnslothTrainer


def test_create_optimizer_accepts_a_positional_model():
    """The signature transformers 5.x calls against."""
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
    """Whatever the base class declares, our override must be callable the same way."""
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
    """The actual failure, reproduced without building a real Trainer.

    A bare object() is enough: the TypeError happened at call time, before any attribute
    on self was touched. Any later failure means the signature is fine, which is what
    this test is about, so only TypeError about arity is treated as a failure.
    """
    try:
        UnslothTrainer.create_optimizer(object(), "prepared-model")
    except TypeError as error:
        message = str(error)
        if "positional argument" in message and "create_optimizer" in message:
            pytest.fail(f"create_optimizer rejected a positional model: {message}")
    except Exception:
        # Reached the body and failed on the fake self, which is the expected outcome.
        pass
