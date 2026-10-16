# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_TOKEN_BUDGET keeps TRL's per-forward ratio metrics (token accuracy, entropy) equal to an
unsplit step when a micro-batch is split into unequal chunks. CPU only."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "token_budget.py"
_spec = importlib.util.spec_from_file_location("_unsloth_token_budget_metrics", _PATH)
tb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tb
_spec.loader.exec_module(tb)


class _Accelerator:
    distributed_type = "NO"
    num_processes = 1

    @contextlib.contextmanager
    def no_sync(self, model):
        yield


def _token_stats(input_ids):
    # Stand-ins for the model: a token is "predicted correctly" when even, its entropy is its id.
    return (input_ids % 2 == 0).double(), input_ids.double()


class _TRL022Trainer:
    """TRL 0.22.2 SFTTrainer.compute_loss metrics: accuracy over shifted labels != -100, entropy over
    the attention mask."""

    num_virtual_tokens = 0

    def __init__(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.accelerator = _Accelerator()
        self.args = type("Args", (), {"gradient_accumulation_steps": 1})()
        self.model_accepts_loss_kwargs = True
        self.label_smoother = None

    def compute_loss(self, inputs):
        correct, entropy = _token_stats(inputs["input_ids"])
        mask = inputs["attention_mask"].double()
        self._metrics["train"]["entropy"].append(float((entropy * mask).sum() / mask.sum()))
        self._metrics["train"]["num_tokens"] = [0]
        shift_labels = inputs["labels"][..., 1:]
        valid = shift_labels != -100
        hits = (correct[..., 1:].bool() & valid).sum()
        self._metrics["train"]["mean_token_accuracy"].append(float(hits / valid.sum()))

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch = None,
    ):
        self.compute_loss(inputs)
        return torch.tensor(1.0)


class _TRL1Trainer(_TRL022Trainer):
    """TRL 1.x SFTTrainer.compute_loss metrics: both ratios over the fused head's label_mask."""

    def compute_loss(self, inputs):
        correct, entropy = _token_stats(inputs["input_ids"])
        label_mask = inputs["labels"][..., 1:] != -100
        num_valid = label_mask.sum()
        entropy_sum = (entropy[..., 1:] * label_mask).sum()
        num_correct = (correct[..., 1:] * label_mask).sum()
        self._metrics["train"]["entropy"].append(float(entropy_sum / num_valid))
        self._metrics["train"]["mean_token_accuracy"].append(float(num_correct / num_valid))


def _padded_batch():
    lengths = [2, 3, 3, 12]
    width = max(lengths)
    gen = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, 50, (len(lengths), width), generator = gen)
    attention_mask = torch.zeros(len(lengths), width, dtype = torch.long)
    for row, n in enumerate(lengths):
        attention_mask[row, :n] = 1
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    labels[:, 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _logged(trainer):
    return {k: sum(v) / len(v) for k, v in trainer._metrics["train"].items()}


@pytest.mark.parametrize("trainer_cls", [_TRL022Trainer, _TRL1Trainer])
def test_split_metrics_match_unsplit_step(monkeypatch, trainer_cls):
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    batch = _padded_batch()

    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    reference = trainer_cls()
    tb.install_token_budget(reference)
    reference.training_step(None, dict(batch), 10)
    expected = _logged(reference)

    monkeypatch.setenv("UNSLOTH_TOKEN_BUDGET", "36")
    trainer = trainer_cls()
    tb.install_token_budget(trainer)
    chunks = tb.split_batch_by_token_budget(dict(batch), 36)
    assert len(chunks) > 1 and len({c["input_ids"].shape[0] for c in chunks}) > 1
    # Two steps: one entry per micro-batch, each weighted like the unsplit forward.
    trainer.training_step(None, dict(batch), 10)
    trainer.training_step(None, dict(batch), 10)
    assert trainer._unsloth_token_budget_splits == 2
    for key in ("entropy", "mean_token_accuracy"):
        assert len(trainer._metrics["train"][key]) == 2
    logged = _logged(trainer)
    for key in ("entropy", "mean_token_accuracy"):
        assert logged[key] == pytest.approx(expected[key], rel = 1e-6)
    assert logged.get("num_tokens") == expected.get("num_tokens")


def test_unsplit_steps_leave_metrics_untouched(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    monkeypatch.setenv("UNSLOTH_TOKEN_BUDGET", "100000")
    trainer = _TRL1Trainer()
    tb.install_token_budget(trainer)
    trainer.training_step(None, _padded_batch(), 10)
    assert trainer._unsloth_token_budget_splits == 0
    assert len(trainer._metrics["train"]["entropy"]) == 1


def test_trainers_without_metrics_still_split(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    monkeypatch.setenv("UNSLOTH_TOKEN_BUDGET", "36")
    trainer = _TRL1Trainer()
    del trainer._metrics
    seen = []
    trainer.training_step = lambda model, inputs, n = None: seen.append(inputs) or torch.tensor(1.0)
    tb.install_token_budget(trainer)
    loss = trainer.training_step(None, _padded_batch(), 10)
    assert len(seen) > 1 and float(loss) == len(seen)


class _DelegatingTRL1Trainer(_TRL1Trainer):
    """A user subclass of a TRL 1.x trainer whose compute_loss only delegates to super()."""

    def compute_loss(self, inputs):
        return super().compute_loss(inputs)


def test_delegating_subclass_uses_parent_metric_code(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    batch = _padded_batch()
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    reference = _DelegatingTRL1Trainer()
    tb.install_token_budget(reference)
    reference.training_step(None, dict(batch), 10)
    expected = _logged(reference)

    monkeypatch.setenv("UNSLOTH_TOKEN_BUDGET", "36")
    trainer = _DelegatingTRL1Trainer()
    tb.install_token_budget(trainer)
    trainer.training_step(None, dict(batch), 10)
    assert trainer._unsloth_token_budget_splits == 1
    logged = _logged(trainer)
    for key in ("entropy", "mean_token_accuracy"):
        assert len(trainer._metrics["train"][key]) == 1
        assert logged[key] == pytest.approx(expected[key], rel = 1e-6)


def _per_chunk_entries(trainer_cls, batch, **attrs):
    # What TRL logs with no folding: one entry per chunk.
    trainer = trainer_cls()
    for name, value in attrs.items():
        setattr(trainer, name, value)
    for chunk in tb.split_batch_by_token_budget(dict(batch), 36):
        trainer.compute_loss(chunk)
    return {k: list(trainer._metrics["train"][k]) for k in ("entropy", "mean_token_accuracy")}


def _split_once(monkeypatch, trainer_cls, batch, **attrs):
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    monkeypatch.setenv("UNSLOTH_TOKEN_BUDGET", "36")
    trainer = trainer_cls()
    for name, value in attrs.items():
        setattr(trainer, name, value)
    tb.install_token_budget(trainer)
    trainer.training_step(None, dict(batch), 10)
    assert trainer._unsloth_token_budget_splits == 1
    return {k: list(trainer._metrics["train"][k]) for k in ("entropy", "mean_token_accuracy")}


def test_prompt_learning_leaves_entries_unfolded(monkeypatch):
    batch = _padded_batch()
    expected = _per_chunk_entries(_TRL022Trainer, batch, num_virtual_tokens = 2)
    assert len(expected["entropy"]) > 1
    assert _split_once(monkeypatch, _TRL022Trainer, batch, num_virtual_tokens = 2) == expected


def test_ambiguous_label_source_leaves_entries_unfolded(monkeypatch):
    batch = _padded_batch()
    batch["shift_labels"] = batch["labels"][..., 1:].contiguous()
    batch["shift_labels"] = torch.nn.functional.pad(batch["shift_labels"], (0, 1), value = -100)
    expected = _per_chunk_entries(_TRL1Trainer, batch)
    assert len(expected["entropy"]) > 1
    assert _split_once(monkeypatch, _TRL1Trainer, batch) == expected


def test_multi_process_leaves_entries_unfolded(monkeypatch):
    batch = _padded_batch()
    expected = _per_chunk_entries(_TRL022Trainer, batch)
    accelerator = _Accelerator()
    accelerator.num_processes = 2
    assert _split_once(monkeypatch, _TRL022Trainer, batch, accelerator = accelerator) == expected
