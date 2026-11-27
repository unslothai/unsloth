# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Token balancing is the default under multi-process data parallel (WORLD_SIZE > 1).

The DDP gradient all-reduce waits for the rank with the most tokens. With UNSLOTH_TOKEN_BALANCE unset
each optimizer step's rows are now spread over the ranks with equal token sums; every step keeps the
same rows, so the loss is unchanged. One process, `UNSLOTH_TOKEN_BALANCE=0`, or no cross-rank token
count (`average_tokens_across_devices` off) leaves the sampler alone. CPU only.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
accelerate_data_loader = pytest.importorskip("accelerate.data_loader")

_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "token_budget.py"
_spec = importlib.util.spec_from_file_location("_unsloth_token_balance_ddp_default", _PATH)
tb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tb
_spec.loader.exec_module(tb)

BS, GA, WS = 2, 4, 2


def _dataset(n = 64, seed = 0):
    rng = random.Random(seed)
    lengths = [rng.choice([8, 8, 8, 600]) for _ in range(n)]
    return datasets.Dataset.from_dict({"input_ids": [[1] * k for k in lengths]}), lengths


def _rank_batches(
    dataset,
    rank,
    average_tokens = True,
    **trainer_extra,
):
    sampler = torch.utils.data.SequentialSampler(dataset)
    inner = torch.utils.data.BatchSampler(sampler, BS, drop_last = False)
    shard = accelerate_data_loader.BatchSamplerShard(
        inner, num_processes = WS, process_index = rank, split_batches = False
    )
    loader = SimpleNamespace(batch_sampler = shard, dataset = dataset)
    trainer = SimpleNamespace(
        training_step = None,
        get_train_dataloader = lambda: loader,
        args = SimpleNamespace(
            gradient_accumulation_steps = GA, average_tokens_across_devices = average_tokens
        ),
        **{"model_accepts_loss_kwargs": True, **trainer_extra},
    )
    tb.install_token_budget(trainer)
    trainer.get_train_dataloader()
    return list(shard), bool(getattr(trainer, "_unsloth_token_balance_active", False))


def _rank_tokens(per_rank, lengths, step):
    return [
        sum(lengths[i] for b in per_rank[r][step * GA : (step + 1) * GA] for i in b)
        for r in range(WS)
    ]


@pytest.fixture
def ddp_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    monkeypatch.setenv("WORLD_SIZE", str(WS))
    return monkeypatch


def test_unset_under_ddp_balances_ranks_with_the_same_rows(ddp_env, capsys):
    dataset, lengths = _dataset()
    ddp_env.setenv("UNSLOTH_TOKEN_BALANCE", "0")
    plain = [_rank_batches(dataset, r) for r in range(WS)]
    ddp_env.delenv("UNSLOTH_TOKEN_BALANCE")
    balanced = [_rank_batches(dataset, r) for r in range(WS)]
    assert not any(active for _, active in plain)
    assert all(active for _, active in balanced)
    assert "equal token counts" in capsys.readouterr().out
    plain = [b for b, _ in plain]
    balanced = [b for b, _ in balanced]
    steps = len(plain[0]) // GA
    worst_plain = worst_balanced = 0
    for s in range(steps):
        rows = lambda per_rank: sorted(
            i for r in range(WS) for b in per_rank[r][s * GA : (s + 1) * GA] for i in b
        )
        # Same multiset of rows in every optimizer step: the loss and gradient are unchanged.
        assert rows(balanced) == rows(plain)
        t_plain, t_bal = _rank_tokens(plain, lengths, s), _rank_tokens(balanced, lengths, s)
        worst_plain = max(worst_plain, max(t_plain) - min(t_plain))
        worst_balanced = max(worst_balanced, max(t_bal) - min(t_bal))
    assert worst_balanced < worst_plain
    assert worst_balanced <= max(lengths)


def test_single_process_default_stays_off(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    monkeypatch.delenv("WORLD_SIZE", raising = False)
    loader = object()
    trainer = SimpleNamespace(training_step = None, get_train_dataloader = lambda: loader)
    get = trainer.get_train_dataloader
    tb.install_token_budget(trainer)
    assert trainer.get_train_dataloader is get


@pytest.mark.parametrize("value", ["0", "off", "false"])
def test_opt_out_under_ddp(ddp_env, value):
    ddp_env.setenv("UNSLOTH_TOKEN_BALANCE", value)
    loader = object()
    trainer = SimpleNamespace(training_step = None, get_train_dataloader = lambda: loader)
    get = trainer.get_train_dataloader
    tb.install_token_budget(trainer)
    assert trainer.get_train_dataloader is get


def test_default_needs_a_cross_rank_token_count(ddp_env, capsys):
    # Without average_tokens_across_devices rows cannot move between ranks; the default then does
    # nothing (no per-rank slot reordering) and says nothing.
    dataset, _ = _dataset()
    ddp_env.setenv("UNSLOTH_TOKEN_BALANCE", "0")
    plain = [_rank_batches(dataset, r, average_tokens = False)[0] for r in range(WS)]
    ddp_env.delenv("UNSLOTH_TOKEN_BALANCE")
    got = [_rank_batches(dataset, r, average_tokens = False) for r in range(WS)]
    assert [b for b, _ in got] == plain
    assert not any(active for _, active in got)
    assert "UNSLOTH_TOKEN_BALANCE" not in capsys.readouterr().out


def test_default_steps_aside_quietly_when_the_loss_is_a_per_batch_mean(ddp_env, capsys):
    dataset, _ = _dataset()
    got = [_rank_batches(dataset, r, model_accepts_loss_kwargs = False) for r in range(WS)]
    ddp_env.setenv("UNSLOTH_TOKEN_BALANCE", "0")
    plain = [_rank_batches(dataset, r)[0] for r in range(WS)]
    assert [b for b, _ in got] == plain
    assert not any(active for _, active in got)
    assert "ignored" not in capsys.readouterr().out


def test_explicit_one_still_balances_a_single_process(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TOKEN_BALANCE", "1")
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    monkeypatch.delenv("WORLD_SIZE", raising = False)
    assert tb._balance_mode() == "exact"


@pytest.mark.parametrize("transform", ["plain", "shuffle", "select", "filter"])
def test_arrow_lengths_match_the_token_lists(transform):
    dataset, _ = _dataset(97, seed = 3)
    if transform == "shuffle":
        dataset = dataset.shuffle(seed = 1)
    elif transform == "select":
        dataset = dataset.select([5, 1, 90, 33, 33, 0])
    elif transform == "filter":
        dataset = dataset.filter(lambda row: len(row["input_ids"]) > 8)
    expected = [len(ids) for ids in dataset["input_ids"]]
    assert tb._arrow_lengths(dataset) == expected
    assert tb._dataset_lengths(dataset) == expected


def test_output_format_keeps_the_arrow_lengths_and_a_transform_does_not():
    dataset, lengths = _dataset(16)
    # numpy / torch output formats only change how rows are returned, not the stored lists.
    assert tb._arrow_lengths(dataset.with_format("numpy")) == lengths
    # A transform may compute input_ids, so the stored column is not trusted.
    transformed = dataset.with_transform(lambda batch: batch)
    assert tb._arrow_lengths(transformed) is None


@pytest.mark.parametrize("flag", ["is_deepspeed_enabled", "is_fsdp_enabled"])
def test_default_leaves_deepspeed_and_accelerate_fsdp_alone(ddp_env, flag):
    dataset, _ = _dataset()
    got = [_rank_batches(dataset, r, **{flag: True}) for r in range(WS)]
    ddp_env.setenv("UNSLOTH_TOKEN_BALANCE", "0")
    plain = [_rank_batches(dataset, r)[0] for r in range(WS)]
    assert [b for b, _ in got] == plain
    assert not any(active for _, active in got)
    # An explicit UNSLOTH_TOKEN_BALANCE=1 still applies there, as before.
    ddp_env.setenv("UNSLOTH_TOKEN_BALANCE", "1")
    assert all(_rank_batches(dataset, r, **{flag: True})[1] for r in range(WS))
