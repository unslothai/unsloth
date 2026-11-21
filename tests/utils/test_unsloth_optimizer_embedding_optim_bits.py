# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from types import MethodType, SimpleNamespace

import pytest

import unsloth  # noqa: F401
from unsloth.trainer import UnslothTrainer


def _model(nn):
    model = nn.Module()
    model.proj = nn.Linear(128, 128)  # above bnb's 4096-element floor, so it stays 8-bit
    embed = nn.Module()
    embed.modules_to_save = nn.ModuleDict({"default": nn.Embedding(8000, 64)})
    inner = nn.Module()
    inner.embed_tokens = embed
    model.model = inner
    return model


def _optimizer(model, embedding_learning_rate):
    from transformers import Trainer
    from trl import SFTConfig

    args = SFTConfig(
        output_dir = "/tmp/unsloth-embedding-optim-bits-test",
        learning_rate = 2e-4,
        optim = "adamw_8bit",
        report_to = [],
        use_cpu = True,
    )
    args.embedding_learning_rate = embedding_learning_rate
    trainer = SimpleNamespace(args = args, model = model, optimizer = None)
    trainer.get_decay_parameter_names = MethodType(Trainer.get_decay_parameter_names, trainer)
    trainer.get_optimizer_cls_and_kwargs = Trainer.get_optimizer_cls_and_kwargs
    trainer.optimizer_cls_and_kwargs = None
    # No model argument: Transformers 4.x's create_optimizer takes none; both use trainer.model.
    if embedding_learning_rate is None:
        return Trainer.create_optimizer(trainer)
    return UnslothTrainer.create_optimizer(trainer)


@pytest.mark.parametrize("embedding_learning_rate", [None, 5e-5])
def test_embedding_state_stays_32_bit_under_adamw_8bit(embedding_learning_rate):
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")

    model = _model(torch.nn)
    optimizer = _optimizer(model, embedding_learning_rate)
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    optimizer.step()

    embedding = model.model.embed_tokens.modules_to_save["default"].weight
    assert optimizer.state[embedding]["state1"].dtype == torch.float32
    assert optimizer.state[model.proj.weight]["state1"].dtype == torch.uint8
