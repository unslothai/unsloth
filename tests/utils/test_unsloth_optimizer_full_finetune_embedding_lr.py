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


def _lr_by_name(torch, model):
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
    optimizer = UnslothTrainer.create_optimizer(trainer)
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
