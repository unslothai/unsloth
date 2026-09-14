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


def test_q_galore_refuses_a_model_with_no_projectable_parameters():
    """FSDP1 hands back 1-D views, which match nothing, so the run must not quietly
    downgrade to ordinary AdamW."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace
    from unsloth.trainer import QGaloreConfig

    flattened = nn.Module()
    flattened.register_parameter("_flat_param", nn.Parameter(torch.ones(64)))
    args = SimpleNamespace(
        learning_rate = 1e-3,
        weight_decay = 0.0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
    )
    trainer = SimpleNamespace(args = args, model = flattened, optimizer = None)
    with pytest.raises(ValueError, match = "no parameter matched"):
        UnslothTrainer._create_q_galore_optimizer(
            trainer,
            QGaloreConfig(rank = 8, weight_quant = False),
            None,
        )


def test_q_galore_still_builds_when_parameters_are_projectable():
    """The guard must not fire on an ordinary unwrapped model."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace
    from unsloth.trainer import QGaloreConfig

    model = nn.Sequential()
    model.add_module("q_proj", nn.Linear(64, 64, bias = False))
    args = SimpleNamespace(
        learning_rate = 1e-3,
        weight_decay = 0.0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
    )
    trainer = SimpleNamespace(args = args, model = model, optimizer = None)
    optimizer = UnslothTrainer._create_q_galore_optimizer(
        trainer,
        QGaloreConfig(rank = 8, weight_quant = False),
        None,
    )
    assert any("rank" in group for group in optimizer.param_groups)


def test_embedding_lr_is_rejected_when_wrapping_hid_the_embeddings():
    """FSDP renames parameters, so the modules_to_save match finds nothing and the
    requested embedding LR would be dropped in silence."""
    import torch
    import torch.nn as nn
    from unsloth.trainer import _create_unsloth_optimizer

    inner = nn.Module()
    inner.register_parameter("_flat_param", nn.Parameter(torch.ones(64)))
    wrapped = nn.Module()
    wrapped.add_module("_fsdp_wrapped_module", inner)
    assert [n for n, _ in wrapped.named_parameters()] == ["_fsdp_wrapped_module._flat_param"]
    with pytest.raises(ValueError, match = "no embedding parameter matched"):
        _create_unsloth_optimizer(
            wrapped,
            torch.optim.AdamW,
            {"lr": 1e-3},
            5e-5,
            require_embedding_match = True,
        )


def test_embedding_lr_without_embeddings_is_still_fine_off_the_delayed_path():
    """The pre-existing behaviour: a model that simply does not train its embeddings is
    ordinary, and must not start raising."""
    import torch
    import torch.nn as nn
    from unsloth.trainer import _create_unsloth_optimizer

    plain = nn.Linear(8, 8, bias = False)
    optimizer = _create_unsloth_optimizer(plain, torch.optim.AdamW, {"lr": 1e-3}, 5e-5)
    assert optimizer.param_groups[1]["params"] == []
