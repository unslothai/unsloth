# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# `embedding_learning_rate` routes optimizer creation through
# `_create_unsloth_optimizer`, which read the decay from `optimizer_kwargs`.
# transformers keeps the decay on the param groups and only puts it in
# optimizer_kwargs for schedule-free and stable_adamw, so the `.get(..., 0.0)`
# default always won and the whole run trained with no weight decay, silently,
# with the value the user set still on the config.

from types import MethodType, SimpleNamespace

import pytest

import unsloth  # noqa: F401  (must precede transformers/trl)
from unsloth.trainer import UnslothTrainer


EMBEDDING = "model.embed_tokens.modules_to_save.default.weight"


def _model(torch, nn):
    model = nn.Module()
    model.proj = nn.Linear(4, 4)  # .weight decays, .bias does not
    model.norm = nn.LayerNorm(4)  # neither weight nor bias decays
    embed = nn.Module()
    embed.modules_to_save = nn.ModuleDict({"default": nn.Linear(4, 4, bias = False)})
    inner = nn.Module()
    inner.embed_tokens = embed
    model.model = inner
    return model


def _trainer(torch, weight_decay):
    from transformers import Trainer
    from trl import SFTConfig

    args = SFTConfig(
        output_dir = "/tmp/unsloth-weight-decay-test",
        weight_decay = weight_decay,
        learning_rate = 2e-4,
        optim = "adamw_torch",
        report_to = [],
    )
    args.embedding_learning_rate = 5e-5
    trainer = SimpleNamespace(args = args, model = None, optimizer = None)
    trainer.get_decay_parameter_names = MethodType(Trainer.get_decay_parameter_names, trainer)
    return trainer


def _groups(torch, nn, weight_decay):
    model = _model(torch, nn)
    trainer = _trainer(torch, weight_decay)
    trainer.model = model
    optimizer = UnslothTrainer.create_optimizer(trainer, model)
    named = {id(p): n for n, p in model.named_parameters()}
    return [
        {
            "names": sorted(named[id(p)] for p in group["params"]),
            "weight_decay": group["weight_decay"],
            "lr": group["lr"],
        }
        for group in optimizer.param_groups
    ]


def test_weight_decay_from_the_config_reaches_the_param_groups():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    groups = _groups(torch, nn, weight_decay = 0.1)
    decayed = {name for g in groups if g["weight_decay"] == 0.1 for name in g["names"]}

    assert "proj.weight" in decayed, f"weight_decay=0.1 never reached the optimizer: {groups}"
    assert EMBEDDING in decayed, "the embedding group must carry the decay too"


def test_biases_and_norms_stay_out_of_the_decay():
    # Trainer.get_decay_parameter_names excludes them on the path this replaces.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    groups = _groups(torch, nn, weight_decay = 0.1)
    undecayed = {name for g in groups if g["weight_decay"] == 0.0 for name in g["names"]}

    assert {"proj.bias", "norm.weight", "norm.bias"} <= undecayed, groups


def test_the_embedding_group_keeps_its_own_learning_rate():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    groups = _groups(torch, nn, weight_decay = 0.1)
    for group in groups:
        if EMBEDDING in group["names"]:
            assert group["lr"] == 5e-5, groups
        elif group["names"]:
            assert group["lr"] == 2e-4, groups


def test_zero_weight_decay_is_still_zero():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    groups = _groups(torch, nn, weight_decay = 0.0)
    assert all(group["weight_decay"] == 0.0 for group in groups), groups
