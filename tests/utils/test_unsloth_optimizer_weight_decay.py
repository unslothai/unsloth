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


def _lora_shaped(torch, nn):
    """A LoRA run trains no bias and no norm, so both no-decay groups come back empty."""
    model = _model(torch, nn)
    model.proj.bias.requires_grad_(False)
    model.norm.weight.requires_grad_(False)
    model.norm.bias.requires_grad_(False)
    return model


def _optimizer(
    torch,
    model,
    weight_decay,
    optim = "adamw_torch",
):
    trainer = _trainer(torch, weight_decay)
    trainer.args.optim = optim
    trainer.model = model
    return UnslothTrainer.create_optimizer(trainer, model)


def test_no_empty_param_groups():
    # An empty group is not free. AdafactorSchedule reads group["params"][0] with no
    # guard, and torch's load_state_dict insists the checkpoint have the same number
    # of groups, so an empty one is a crash and a broken resume for nothing.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    for model in (_model(torch, nn), _lora_shaped(torch, nn)):
        optimizer = _optimizer(torch, model, 0.1)
        assert all(group["params"] for group in optimizer.param_groups), optimizer.param_groups


def test_adafactor_schedule_survives_the_split():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from transformers.optimization import AdafactorSchedule

    model = _lora_shaped(torch, nn)
    optimizer = _optimizer(torch, model, 0.1, optim = "adafactor")
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)
    optimizer.step()

    assert AdafactorSchedule(optimizer).get_lr()  # IndexError if any group is empty


def test_a_lora_run_started_before_this_can_still_resume():
    # The old two-group build is what every checkpoint out there was written with.
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = _lora_shaped(torch, nn)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    old = torch.optim.AdamW(
        [
            {
                "params": [p for name, p in trainable if name != EMBEDDING],
                "weight_decay": 0.0,
                "lr": 2e-4,
            },
            {
                "params": [p for name, p in trainable if name == EMBEDDING],
                "weight_decay": 0.0,
                "lr": 5e-5,
            },
        ],
        lr = 2e-4,
    )
    for _, param in trainable:
        param.grad = torch.randn_like(param)
    old.step()

    new = _optimizer(torch, model, 0.1)
    new.load_state_dict(old.state_dict())  # ValueError if the group count moved
