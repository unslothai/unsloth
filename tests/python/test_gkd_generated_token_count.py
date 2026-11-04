# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""GKD counts num_items_in_batch from the generated labels the loss sees, not the dataset answers."""

from __future__ import annotations

import importlib.util
import os
import random
import textwrap

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

# TRL <= 1.6 runs seq_kd first; TRL >= 1.7 runs the on-policy branch first, seq_kd as its elif.
SEQ_KD_FIRST = """
    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.seq_kd:
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
"""
LMBDA_FIRST = """
    def training_step(self, model, inputs, num_items_in_batch=None):
        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        elif self.seq_kd:
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
"""

GENERATED_LEN = 3


class _Parent:
    def get_batch_samples(
        self,
        epoch_iterator,
        num_batches,
        device = None,
    ):
        batches = [next(epoch_iterator) for _ in range(num_batches)]
        return batches, sum(int((b["labels"][..., 1:] != -100).sum()) for b in batches)

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch = None,
    ):
        self.seen.append((num_items_in_batch, sorted(inputs)))
        return torch.tensor(0.0)


class _NoGen:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _build(source, lmbda, seq_kd):
    rewritten = rl.gkd_trainer_generate_before_counting("training_step", source)
    namespace = {
        "random": random,
        "torch": torch,
        "unwrap_model_for_generation": _NoGen,
        "_Parent": _Parent,
    }
    exec("class Trainer(_Parent):\n" + rewritten, namespace)
    trainer = namespace["Trainer"]()
    trainer.lmbda, trainer.seq_kd, trainer.seen, trainer.generated = lmbda, seq_kd, [], []
    trainer.teacher_model = trainer.accelerator = trainer.generation_config = None
    trainer.processing_class = type("P", (), {"pad_token_id": 0})()
    trainer.generated_len, trainer.mask_generated = GENERATED_LEN, False

    def generate(model, inputs, *args):
        trainer.generated.append(id(inputs))
        ids = torch.ones(1, inputs["prompts"].shape[1] + trainer.generated_len, dtype = torch.long)
        labels = ids.clone()
        labels[:, : inputs["prompts"].shape[1]] = -100
        if trainer.mask_generated:
            labels[:] = -100
        return ids, torch.ones_like(ids), labels

    trainer.generate_on_policy_outputs = generate
    return rewritten, trainer


def _window(answer_lengths, prompt_len = 4):
    batches = []
    for n in answer_lengths:
        labels = torch.full((1, prompt_len + n), 5)
        labels[:, :prompt_len] = -100
        batches.append(
            {
                "prompts": torch.ones(1, prompt_len, dtype = torch.long),
                "input_ids": torch.ones(1, prompt_len + n, dtype = torch.long),
                "attention_mask": torch.ones(1, prompt_len + n, dtype = torch.long),
                "labels": labels,
            }
        )
    return batches


def _train_window(trainer, answer_lengths):
    batches, count = trainer.get_batch_samples(iter(_window(answer_lengths)), len(answer_lengths))
    for batch in batches:
        trainer.training_step(None, batch, count)


@pytest.mark.parametrize("source", [SEQ_KD_FIRST, LMBDA_FIRST], ids = ["seq_kd_first", "lmbda_first"])
@pytest.mark.parametrize("lmbda, seq_kd", [(1.0, False), (0.0, True)])
def test_generated_labels_set_the_count(source, lmbda, seq_kd):
    rewritten, trainer = _build(source, lmbda, seq_kd)
    assert rewritten != source
    _train_window(trainer, [40, 7, 19])
    # Each microbatch generated once; the shared count is the generated tokens, whatever the dataset answers were.
    assert len(trainer.generated) == len(set(trainer.generated)) == 3
    assert [n for n, _ in trainer.seen] == [3 * GENERATED_LEN] * 3
    assert all(not any(k.startswith("_unsloth") for k in keys) for _, keys in trainer.seen)
    # A short final window and a second window recount on their own.
    trainer.seen.clear()
    _train_window(trainer, [11])
    assert [n for n, _ in trainer.seen] == [GENERATED_LEN]


@pytest.mark.parametrize("source", [SEQ_KD_FIRST, LMBDA_FIRST], ids = ["seq_kd_first", "lmbda_first"])
def test_mixed_window_draws_the_same_random_numbers(source):
    random.seed(1)
    expected = [random.random() <= 0.5 for _ in range(6)]
    random.seed(1)
    _, trainer = _build(source, 0.5, False)
    _train_window(trainer, [40, 7, 19])
    _train_window(trainer, [8, 9, 10])
    assert len(trainer.generated) == sum(expected)
    answers = [40, 7, 19, 8, 9, 10]
    windows = [expected[:3], expected[3:]]
    totals = [
        sum(GENERATED_LEN if gen else n for gen, n in zip(flags, answers[i * 3 : i * 3 + 3]))
        for i, flags in enumerate(windows)
    ]
    assert [n for n, _ in trainer.seen] == [totals[0]] * 3 + [totals[1]] * 3


@pytest.mark.parametrize("source", [SEQ_KD_FIRST, LMBDA_FIRST], ids = ["seq_kd_first", "lmbda_first"])
def test_off_policy_keeps_the_dataset_count(source):
    _, trainer = _build(source, 0.0, False)
    _train_window(trainer, [40, 7, 19])
    assert trainer.generated == []
    assert [n for n, _ in trainer.seen] == [66] * 3


def test_unrecognised_training_step_is_left_alone():
    source = SEQ_KD_FIRST.replace("loss = super().training_step(", "loss = self._step(")
    assert rl.gkd_trainer_generate_before_counting("training_step", source) == source
    assert rl.gkd_trainer_generate_before_counting("compute_loss", SEQ_KD_FIRST) == SEQ_KD_FIRST


def test_installed_trl_training_step_is_rewritten():
    """Read TRL's file, not the class: importing unsloth may have rebound the class to the patched one."""
    import ast

    for name in ("trl.experimental.gkd.gkd_trainer", "trl.trainer.gkd_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        with open(spec.origin, encoding = "utf-8") as file:
            tree = ast.parse(file.read())
        lines = open(spec.origin, encoding = "utf-8").read().splitlines(keepends = True)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GKDTrainer":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "training_step":
                        source = "".join(lines[item.lineno - 1 : item.end_lineno])
                        rewritten = rl.gkd_trainer_generate_before_counting("training_step", source)
                        assert rewritten != source
                        ast.parse(textwrap.dedent("class T:\n" + rewritten))
                        return
    pytest.skip("installed trl has no GKDTrainer")


@pytest.mark.parametrize("tensor_count", [False, True])
def test_all_masked_generations_count_one_not_zero(tensor_count):
    # TRL 1.7-1.8 mask a pad == eos completion entirely; a zero count would make the JSD sum / 0 NaN.
    _, trainer = _build(LMBDA_FIRST, 1.0, False)
    trainer.mask_generated = True
    if tensor_count:
        parent = _Parent.get_batch_samples

        def tensor_counts(self, *args, **kwargs):
            batches, count = parent(self, *args, **kwargs)
            return batches, torch.tensor(count)

        type(trainer).__bases__[0].get_batch_samples = tensor_counts
    try:
        _train_window(trainer, [40, 7])
    finally:
        if tensor_count:
            type(trainer).__bases__[0].get_batch_samples = parent
    assert [int(n) for n, _ in trainer.seen] == [1, 1]
