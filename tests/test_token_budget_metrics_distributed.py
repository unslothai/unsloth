# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_TOKEN_BUDGET under several processes: TRL gathers each forward's metric sums across ranks,
so split-chunk entries must be folded with token counts summed over ranks to match an unsplit step.
Two CPU ranks over gloo."""

from __future__ import annotations

import contextlib
import importlib.util
import multiprocessing
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

if not dist.is_available() or not dist.is_gloo_available() or sys.platform != "linux":
    pytest.skip("needs torch.distributed with gloo on Linux (fork)", allow_module_level = True)

_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "token_budget.py"
_spec = importlib.util.spec_from_file_location("_unsloth_token_budget_dist", _PATH)
tb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tb
_spec.loader.exec_module(tb)

_BUDGET = 36
_WORLD = 2


class _Accelerator:
    distributed_type = "MULTI_GPU"
    num_processes = _WORLD
    device = torch.device("cpu")

    @contextlib.contextmanager
    def no_sync(self, model):
        yield

    def gather_sum(self, value):
        # accelerator.gather_for_metrics(x).sum() for a scalar.
        value = torch.as_tensor(value, dtype = torch.float64).clone()
        dist.all_reduce(value, op = dist.ReduceOp.SUM)
        return value


class _TRL115Trainer:
    """TRL 1.15 SFTTrainer.compute_loss (sft_trainer.py:1638-1642): accuracy and entropy are global
    sums over valid labels divided by the global valid-label count."""

    num_virtual_tokens = 0

    def __init__(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.accelerator = _Accelerator()
        self.args = type("Args", (), {"gradient_accumulation_steps": 1})()
        self.model_accepts_loss_kwargs = True
        self.label_smoother = None

    def compute_loss(self, inputs):
        ids = inputs["input_ids"]
        correct, entropy = (ids % 2 == 0).double(), ids.double()
        label_mask = inputs["labels"][..., 1:] != -100
        num_valid = self.accelerator.gather_sum(label_mask.sum())
        entropy_sum = self.accelerator.gather_sum((entropy[..., 1:] * label_mask).sum())
        num_correct = self.accelerator.gather_sum((correct[..., 1:] * label_mask).sum())
        self._metrics["train"]["entropy"].append(float(entropy_sum / num_valid))
        self._metrics["train"]["mean_token_accuracy"].append(float(num_correct / num_valid))

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch = None,
    ):
        self.compute_loss(inputs)
        return torch.tensor(1.0)


class _TRL022Trainer(_TRL115Trainer):
    """TRL 0.22.2 (sft_trainer.py:1056-1061, 1099-1105): entropy is the mean of per-rank attention
    ratios (not refoldable), accuracy is global correct over global shifted labels."""

    def compute_loss(self, inputs):
        ids = inputs["input_ids"]
        correct, entropy = (ids % 2 == 0).double(), ids.double()
        attention_mask = inputs["attention_mask"].double()
        local_entropy = (entropy * attention_mask).sum() / attention_mask.sum()
        entropy = self.accelerator.gather_sum(local_entropy) / _WORLD
        self._metrics["train"]["entropy"].append(float(entropy))
        shift_labels = inputs["labels"][..., 1:]
        valid = shift_labels != -100
        correct_tokens = self.accelerator.gather_sum((correct[..., 1:].bool() & valid).sum())
        total_tokens = self.accelerator.gather_sum(valid.sum())
        self._metrics["train"]["mean_token_accuracy"].append(float(correct_tokens / total_tokens))


_TRAINERS = {"trl115": _TRL115Trainer, "trl022": _TRL022Trainer}


def _rank_batch(rank):
    # Unequal row lengths per rank, so chunks differ in token count within and across ranks.
    lengths = [[2, 3, 3, 12], [9, 2, 12, 5]][rank]
    width = 12
    gen = torch.Generator().manual_seed(rank)
    input_ids = torch.randint(1, 50, (len(lengths), width), generator = gen)
    attention_mask = torch.zeros(len(lengths), width, dtype = torch.long)
    for row, n in enumerate(lengths):
        attention_mask[row, :n] = 1
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    labels[:, 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _worker(
    rank,
    init_file,
    trainer_name,
    queue,
    shift_labels_on_rank = None,
):
    try:
        dist.init_process_group(
            "gloo", init_method = f"file://{init_file}", rank = rank, world_size = _WORLD
        )
        trainer_cls = _TRAINERS[trainer_name]
        batch = _rank_batch(rank)
        if rank == shift_labels_on_rank:
            # A second label source makes only this rank decline to fold.
            batch["shift_labels"] = batch["labels"][..., 1:].contiguous()

        os.environ.pop("UNSLOTH_TOKEN_BALANCE", None)
        os.environ.pop("UNSLOTH_TOKEN_BUDGET", None)
        reference = trainer_cls()
        tb.install_token_budget(reference)
        reference.training_step(None, dict(batch), 10)

        os.environ["UNSLOTH_TOKEN_BUDGET"] = str(_BUDGET)
        trainer = trainer_cls()
        tb.install_token_budget(trainer)
        trainer.training_step(None, dict(batch), 10)
        n_chunks = len(tb.split_batch_by_token_budget(dict(batch), _BUDGET))
        queue.put(
            (
                rank,
                {
                    "reference": {k: list(v) for k, v in reference._metrics["train"].items()},
                    "split": {k: list(v) for k, v in trainer._metrics["train"].items()},
                    "splits": trainer._unsloth_token_budget_splits,
                    "n_chunks": n_chunks,
                },
            )
        )
        dist.destroy_process_group()
    except BaseException as error:
        queue.put((rank, repr(error)))
        raise


def _run(trainer_name, shift_labels_on_rank = None):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg")
        procs = [
            ctx.Process(
                target = _worker,
                args = (rank, init_file, trainer_name, queue, shift_labels_on_rank),
            )
            for rank in range(_WORLD)
        ]
        for proc in procs:
            proc.start()
        results = dict(queue.get(timeout = 120) for _ in procs)
        for proc in procs:
            proc.join(timeout = 60)
    for rank, result in results.items():
        assert isinstance(result, dict), f"rank {rank} failed: {result}"
    return results


def _mean(values):
    return sum(values) / len(values)


@pytest.mark.parametrize("trainer_name", ["trl115", "trl022"])
def test_multi_process_split_accuracy_matches_unsplit(trainer_name):
    results = _run(trainer_name)
    for result in results.values():
        assert result["splits"] == 1 and result["n_chunks"] > 1
        reference, split = result["reference"], result["split"]
        assert len(split["mean_token_accuracy"]) == 1
        assert _mean(split["mean_token_accuracy"]) == pytest.approx(
            _mean(reference["mean_token_accuracy"]), rel = 1e-9
        )
    # Both ranks log the same gathered value, as TRL does.
    assert results[0]["split"] == results[1]["split"]


def test_multi_process_split_entropy_matches_unsplit_on_label_weighted_trl():
    for result in _run("trl115").values():
        reference, split = result["reference"], result["split"]
        assert len(split["entropy"]) == 1
        assert _mean(split["entropy"]) == pytest.approx(_mean(reference["entropy"]), rel = 1e-9)


def test_multi_process_attention_entropy_left_unfolded():
    # Older TRL averages per-rank entropy ratios; a token-weighted fold cannot reproduce that.
    for result in _run("trl022").values():
        assert len(result["split"]["entropy"]) == result["n_chunks"]


def test_ranks_that_disagree_on_folding_neither_fold_nor_hang():
    # The fold adds a collective; if one rank cannot fold, every rank must skip it (else the others hang).
    results = _run("trl115", shift_labels_on_rank = 1)
    for result in results.values():
        assert result["splits"] == 1
        assert len(result["split"]["mean_token_accuracy"]) == result["n_chunks"] > 1
