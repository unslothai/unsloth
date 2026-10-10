# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Under full finetuning the embedding and lm_head are plain parameters with no PEFT
# `modules_to_save` copy, so the name match found nothing and `embedding_learning_rate`
# was dropped without a word: they trained at the base learning rate.

from types import MethodType, SimpleNamespace

import pytest

import unsloth  # noqa: F401  (must precede transformers/trl)
from unsloth.trainer import UnslothTrainer


LR, EMBEDDING_LR = 2e-4, 5e-5


def _model(nn, tied):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(10, 4)
            self.model.proj = nn.Linear(4, 4)
            self.lm_head = nn.Linear(4, 10, bias = False)
            if tied:
                self.lm_head.weight = self.model.embed_tokens.weight

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    return Tiny()


def _optimizer(model):
    from transformers import Trainer
    from trl import SFTConfig

    args = SFTConfig(
        output_dir = "/tmp/unsloth-full-ft-embedding-lr-test",
        learning_rate = LR,
        weight_decay = 0.01,
        optim = "adamw_torch",
        report_to = [],
    )
    args.embedding_learning_rate = EMBEDDING_LR
    trainer = SimpleNamespace(args = args, model = model, optimizer = None)
    trainer.get_decay_parameter_names = MethodType(Trainer.get_decay_parameter_names, trainer)
    return UnslothTrainer.create_optimizer(trainer), trainer.get_decay_parameter_names(model)


def _lr_by_name(torch, model):
    optimizer, _ = _optimizer(model)
    named = {id(p): n for n, p in model.named_parameters()}
    return {named[id(p)]: g["lr"] for g in optimizer.param_groups for p in g["params"]}


@pytest.mark.parametrize("tied", [False, True])
def test_full_finetune_embeddings_get_embedding_learning_rate(tied):
    torch = pytest.importorskip("torch")
    lrs = _lr_by_name(torch, _model(torch.nn, tied))

    assert lrs["model.embed_tokens.weight"] == EMBEDDING_LR, lrs
    if not tied:
        assert lrs["lm_head.weight"] == EMBEDDING_LR, lrs
    assert lrs["model.proj.weight"] == LR, lrs
    assert lrs["model.proj.bias"] == LR, lrs


def test_frozen_embeddings_stay_out_of_the_optimizer():
    torch = pytest.importorskip("torch")
    model = _model(torch.nn, tied = False)
    model.model.embed_tokens.weight.requires_grad_(False)
    lrs = _lr_by_name(torch, model)

    assert "model.embed_tokens.weight" not in lrs, lrs
    assert lrs["lm_head.weight"] == EMBEDDING_LR, lrs


def test_a_checkpoint_from_before_resumes_with_its_own_moments():
    # The previous layout put the embeddings in the ordinary decay groups, so the new
    # groups differ; every parameter must get its own moments back, and the embeddings
    # their own learning rate at the saved point of the schedule.
    torch = pytest.importorskip("torch")
    from torch.optim.lr_scheduler import LambdaLR
    from unsloth.trainer import _install_legacy_scheduler_resume

    schedule = lambda step: 0.5**step
    model = _model(torch.nn, tied = False)
    named = {id(p): n for n, p in model.named_parameters()}
    _, decay = _optimizer(model)
    params = list(model.named_parameters())
    old = torch.optim.AdamW(
        [
            {"params": [p for n, p in params if n in decay], "weight_decay": 0.01},
            {"params": [p for n, p in params if n not in decay], "weight_decay": 0.0},
        ],
        lr = LR,
    )
    old_scheduler = LambdaLR(old, schedule)
    for index, (_, param) in enumerate(params):
        param.grad = torch.full_like(param, float(index + 1))
    old.step()
    old_scheduler.step()
    saved, saved_scheduler = old.state_dict(), old_scheduler.state_dict()
    flat = [p for group in old.param_groups for p in group["params"]]
    want = {named[id(p)]: float(saved["state"][i]["exp_avg"].sum()) for i, p in enumerate(flat)}

    new, _ = _optimizer(model)
    assert len(new.param_groups) != len(saved["param_groups"]), "no migration exercised"
    scheduler = _install_legacy_scheduler_resume(LambdaLR(new, schedule), new)
    new.load_state_dict(saved)
    scheduler.load_state_dict(saved_scheduler)

    state = new.state_dict()["state"]
    flat = [p for group in new.param_groups for p in group["params"]]
    for index, param in enumerate(flat):
        assert float(state[index]["exp_avg"].sum()) == pytest.approx(want[named[id(param)]])
    lr_of = {named[id(p)]: g["lr"] for g in new.param_groups for p in g["params"]}
    assert lr_of["model.embed_tokens.weight"] == pytest.approx(EMBEDDING_LR * 0.5), lr_of
    assert lr_of["model.proj.weight"] == pytest.approx(LR * 0.5), lr_of

    new.step()
    scheduler.step()
    lr_of = {named[id(p)]: g["lr"] for g in new.param_groups for p in g["params"]}
    assert lr_of["lm_head.weight"] == pytest.approx(EMBEDDING_LR * 0.25), lr_of
    assert lr_of["model.proj.bias"] == pytest.approx(LR * 0.25), lr_of
