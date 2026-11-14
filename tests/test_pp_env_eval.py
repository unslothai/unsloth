# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import types

import pytest
import torch
import trl

from unsloth.models import pp_env


class _Trainer:
    def __init__(self, average_tokens_across_devices = True):
        self.args = types.SimpleNamespace(
            device = torch.device("cpu"),
            average_tokens_across_devices = average_tokens_across_devices,
        )

    def compute_loss(self, *args, **kwargs):
        raise AssertionError("pipeline evaluation does not call compute_loss")

    def _get_num_items_in_batch(self, batches, device):
        assert not self.args.average_tokens_across_devices
        return sum((batch["labels"] != -100).sum() for batch in batches).to(device)


class _CpuTrainer(_Trainer):
    def compute_loss(self, *args, **kwargs):
        return super().compute_loss(*args, **kwargs)

    compute_loss._unsloth_cpu = True


@pytest.mark.parametrize(
    "labels,expected",
    [([[1, 2, -100], [3, 4, 5]], 3), ([[1, -100, -100]], 0)],
)
def test_cpu_eval_counts_only_shifted_targets(monkeypatch, labels, expected):
    monkeypatch.setattr(trl, "__version__", "1.15.0")
    labels = torch.tensor(labels)
    trainer = _CpuTrainer()
    assert pp_env._eval_items(trainer, {"labels": labels}, labels) == expected
    assert trainer.args.average_tokens_across_devices is True


@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("trainer_cls,version", [(_Trainer, "1.15.0"), (_CpuTrainer, "1.14.0")])
def test_other_trainers_keep_their_eval_token_count(monkeypatch, trainer_cls, version, average):
    monkeypatch.setattr(trl, "__version__", version)
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    trainer = trainer_cls(average_tokens_across_devices = average)
    assert pp_env._eval_items(trainer, {"labels": labels}, labels) == 5
    assert trainer.args.average_tokens_across_devices is average
