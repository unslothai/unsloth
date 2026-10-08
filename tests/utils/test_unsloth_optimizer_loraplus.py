# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# LoRA+ (arXiv 2402.12354) via `UnslothTrainingArguments(loraplus_lr_ratio = ...)`:
# lora_B trains at learning_rate * ratio, everything else as before.

from types import MethodType, SimpleNamespace

import pytest

import unsloth  # noqa: F401  (must precede transformers/trl)
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments

LR = 2e-4


def _peft_model(
    torch,
    nn,
    embeddings = False,
):
    from peft import LoraConfig, get_peft_model

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.q_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)

        def forward(self, x):
            return self.v_proj(self.q_proj(self.embed_tokens(x)))

    torch.manual_seed(0)
    config = LoraConfig(
        r = 2,
        target_modules = ["q_proj", "v_proj"],
        modules_to_save = ["embed_tokens"] if embeddings else None,
    )
    return get_peft_model(Tiny(), config)


def _trainer(
    model,
    weight_decay = 0.0,
    **extra,
):
    from transformers import Trainer
    from trl import SFTConfig

    args = SFTConfig(
        output_dir = "/tmp/unsloth-loraplus-test",
        learning_rate = LR,
        weight_decay = weight_decay,
        optim = "adamw_torch",
        report_to = [],
    )
    args.embedding_learning_rate = None
    args.loraplus_lr_ratio = None
    for key, value in extra.items():
        setattr(args, key, value)
    trainer = SimpleNamespace(args = args, model = model, optimizer = None)
    trainer.get_decay_parameter_names = MethodType(Trainer.get_decay_parameter_names, trainer)
    return trainer


def _lr_of(model, optimizer):
    named = {id(p): n for n, p in model.named_parameters()}
    return {named[id(p)]: group["lr"] for group in optimizer.param_groups for p in group["params"]}


def test_lora_b_gets_the_ratio_and_nothing_else_does():
    torch = pytest.importorskip("torch")
    model = _peft_model(torch, torch.nn)
    optimizer = UnslothTrainer.create_optimizer(_trainer(model, loraplus_lr_ratio = 16.0), model)

    lrs = _lr_of(model, optimizer)
    lora_b = {n: lr for n, lr in lrs.items() if ".lora_B." in n}
    others = {n: lr for n, lr in lrs.items() if ".lora_B." not in n}
    assert len(lora_b) == 2 and len(others) == 2, lrs
    assert all(lr == pytest.approx(LR * 16) for lr in lora_b.values()), lrs
    assert all(lr == pytest.approx(LR) for lr in others.values()), lrs
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(lrs) == trainable, "every trainable parameter must be in exactly one group"


def test_loraplus_composes_with_embedding_lr_and_weight_decay():
    torch = pytest.importorskip("torch")
    model = _peft_model(torch, torch.nn, embeddings = True)
    trainer = _trainer(model, 0.1, loraplus_lr_ratio = 4.0, embedding_learning_rate = 5e-6)
    optimizer = UnslothTrainer.create_optimizer(trainer, model)

    lrs = _lr_of(model, optimizer)
    embedding = [n for n in lrs if "modules_to_save" in n]
    assert embedding and all(lrs[n] == pytest.approx(5e-6) for n in embedding), lrs
    assert all(lrs[n] == pytest.approx(LR * 4) for n in lrs if ".lora_B." in n), lrs
    assert any(g["weight_decay"] == 0.1 for g in optimizer.param_groups)
    assert all(g["params"] for g in optimizer.param_groups), "no empty groups"


def test_embeddings_stay_at_the_base_lr_when_only_loraplus_is_set():
    torch = pytest.importorskip("torch")
    model = _peft_model(torch, torch.nn, embeddings = True)
    optimizer = UnslothTrainer.create_optimizer(_trainer(model, loraplus_lr_ratio = 16.0), model)

    lrs = _lr_of(model, optimizer)
    assert all(lr == pytest.approx(LR) for n, lr in lrs.items() if "modules_to_save" in n), lrs


def test_without_the_ratio_the_layout_is_unchanged():
    # Existing embedding_learning_rate runs must keep their group layout so checkpoints resume.
    torch = pytest.importorskip("torch")
    model = _peft_model(torch, torch.nn, embeddings = True)
    optimizer = UnslothTrainer.create_optimizer(
        _trainer(model, embedding_learning_rate = 5e-6), model
    )
    assert optimizer._unsloth_group_roles == ["non_embeddings", "embeddings"]
    assert sorted({g["lr"] for g in optimizer.param_groups}) == [5e-6, LR]


def test_a_custom_optimizer_cls_and_kwargs_is_kept():
    torch = pytest.importorskip("torch")
    model = _peft_model(torch, torch.nn)
    trainer = _trainer(model, loraplus_lr_ratio = 16.0)
    trainer.optimizer_cls_and_kwargs = (torch.optim.SGD, {"momentum": 0.5})
    optimizer = UnslothTrainer.create_optimizer(trainer, model)

    assert type(optimizer) is torch.optim.SGD
    assert all(g["momentum"] == 0.5 for g in optimizer.param_groups)
    assert sorted({g["lr"] for g in optimizer.param_groups}) == [LR, pytest.approx(LR * 16)]


def test_a_ratio_with_no_lora_is_refused():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match = "loraplus_lr_ratio"):
        UnslothTrainer.create_optimizer(_trainer(model, loraplus_lr_ratio = 16.0), model)


def test_a_loraplus_checkpoint_resumes_with_its_learning_rates():
    torch = pytest.importorskip("torch")
    model = _peft_model(torch, torch.nn)
    first = UnslothTrainer.create_optimizer(_trainer(model, loraplus_lr_ratio = 16.0), model)
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)
    first.step()

    second = UnslothTrainer.create_optimizer(_trainer(model, loraplus_lr_ratio = 16.0), model)
    second.load_state_dict(first.state_dict())
    assert _lr_of(model, second) == _lr_of(model, first)


def test_a_legacy_checkpoint_is_refused_not_misread_under_loraplus():
    # The pre-split two-group layout has no LoRA+ group; torch's own size check must decide.
    torch = pytest.importorskip("torch")
    model = _peft_model(torch, torch.nn, embeddings = True)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    legacy = torch.optim.AdamW(
        [
            {"params": [p for n, p in trainable if "modules_to_save" not in n], "lr": LR},
            {"params": [p for n, p in trainable if "modules_to_save" in n], "lr": 5e-6},
        ],
        lr = LR,
    )
    trainer = _trainer(model, loraplus_lr_ratio = 16.0, embedding_learning_rate = 5e-6)
    optimizer = UnslothTrainer.create_optimizer(trainer, model)
    with pytest.raises(ValueError):
        optimizer.load_state_dict(legacy.state_dict())


def test_training_arguments_store_and_validate_the_ratio():
    pytest.importorskip("torch")
    if UnslothTrainingArguments.__module__ != "unsloth.trainer":
        pytest.skip(
            "MLX maps loraplus_lr_ratio onto lora_plus_ratio (test_mlx_public_trainer_api.py)"
        )
    args = UnslothTrainingArguments(output_dir = "/tmp/unsloth-loraplus-test", loraplus_lr_ratio = 16)
    assert args.loraplus_lr_ratio == 16
    assert (
        UnslothTrainingArguments(output_dir = "/tmp/unsloth-loraplus-test").loraplus_lr_ratio is None
    )
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match = "loraplus_lr_ratio"):
            UnslothTrainingArguments(output_dir = "/tmp/unsloth-loraplus-test", loraplus_lr_ratio = bad)
