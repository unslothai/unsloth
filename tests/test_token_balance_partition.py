# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Plain DDP token balancing: `equal_size_partition` (equal-size differencing or longest-first, then
1-for-1 swaps) evens per-rank tokens on long-tailed lengths where equal-size differencing alone leaves
the slowest rank 4-17% over the mean. Ranks in lockstep (FSDP, PP, TP, CP, EP) keep the differencing
layout. Every optimizer step keeps its rows, so a two-rank gloo DDP run has the same losses and
weights with either layout. CPU only."""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import random
import statistics
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "token_budget.py"
_spec = importlib.util.spec_from_file_location("_unsloth_token_balance_partition", _PATH)
tb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tb
_spec.loader.exec_module(tb)


def _chat_lengths(rng, n):
    return [min(4096, max(16, int(rng.lognormvariate(6.0, 1.1)))) for _ in range(n)]


def _excess(values, groups):
    sums = [sum(values[i] for i in g) for g in groups]
    return max(sums) / (sum(values) / len(groups)) - 1


@pytest.mark.parametrize("ws,base_above", [(2, 0.03), (4, 0.09), (8, 0.14)])
def test_partition_evens_long_tailed_ranks_where_differencing_does_not(ws, base_above):
    rng = random.Random(ws)
    differencing, partition = [], []
    for _ in range(100):
        values = _chat_lengths(rng, ws * 8)
        differencing.append(_excess(values, tb.karmarkar_karp(values, ws, equal_size = True)))
        partition.append(_excess(values, tb.equal_size_partition(values, ws)))
    assert statistics.mean(differencing) > base_above
    assert statistics.mean(partition) < 0.01


@pytest.mark.parametrize("seed", range(40))
def test_partition_keeps_sizes_and_rows_and_is_never_heavier(seed):
    rng = random.Random(seed)
    k = rng.choice([2, 3, 4, 8])
    values = _chat_lengths(rng, k * rng.randint(1, 12))
    if seed % 5 == 0:
        values = [rng.choice([7, 7, 7, 900]) for _ in values]  # ties
    groups = tb.equal_size_partition(values, k)
    assert len(groups) == k and all(len(g) == len(values) // k for g in groups)
    assert sorted(i for g in groups for i in g) == list(range(len(values)))
    reference = tb.karmarkar_karp(values, k, equal_size = True)
    sums = [sum(values[i] for i in g) for g in groups]
    assert max(sums) <= max(sum(values[i] for i in g) for g in reference)
    assert sums == sorted(sums, reverse = True)
    # Every rank computes the same plan from the same lengths.
    assert tb.equal_size_partition(list(values), k) == groups


def test_partition_edge_cases():
    assert tb.equal_size_partition([5, 1, 3], 1) == [[0, 1, 2]]
    assert sorted(map(sorted, tb.equal_size_partition([5, 1], 2))) == [[0], [1]]
    with pytest.raises(ValueError):
        tb.equal_size_partition([1, 2, 3], 2)


def _layout(order, bs, ga, ws):
    batches = [order[i : i + bs] for i in range(0, len(order), bs)]
    return [[batches[g * ws + r] for g in range(ga)] for r in range(ws)]


@pytest.mark.parametrize("ws,ga,bs", [(2, 4, 2), (4, 2, 2), (8, 8, 1), (2, 1, 4)])
def test_ddp_layout_keeps_the_step_and_evens_rank_tokens(ws, ga, bs):
    rng = random.Random(ws * 100 + ga * 10 + bs)
    n = ws * ga * bs
    worse = 0
    for _ in range(50):
        lengths = _chat_lengths(rng, n)
        order = list(range(n))
        rng.shuffle(order)
        lockstep = tb.balance_step_indices(order, lengths, bs, ga, ws, True)
        assert lockstep == tb.balance_step_indices(order, lengths, bs, ga, ws, True, True)
        ddp = tb.balance_step_indices(order, lengths, bs, ga, ws, True, False)
        assert sorted(ddp) == sorted(order)
        layout = _layout(ddp, bs, ga, ws)
        assert all(len(b) == bs for r in layout for b in r)

        def heaviest(out):
            return max(sum(lengths[i] for b in r for i in b) for r in _layout(out, bs, ga, ws))

        worse += heaviest(ddp) > heaviest(lockstep)
    assert worse == 0


def test_lockstep_is_only_off_for_plain_ddp(monkeypatch):
    names = (
        "UNSLOTH_FSDP2",
        "UNSLOTH_TP_SIZE",
        "UNSLOTH_PP_SIZE",
        "UNSLOTH_CP_SIZE",
        "UNSLOTH_EP_SIZE",
    )
    for name in names:
        monkeypatch.delenv(name, raising = False)
    plain = type("T", (), {})()
    assert tb._lockstep_ranks(plain) is False
    monkeypatch.setenv("UNSLOTH_FSDP2", "0")
    monkeypatch.setenv("UNSLOTH_TP_SIZE", "1")
    assert tb._lockstep_ranks(plain) is False
    for name, value in (
        ("UNSLOTH_FSDP2", "1"),
        ("UNSLOTH_TP_SIZE", "2"),
        ("UNSLOTH_PP_SIZE", "2"),
        ("UNSLOTH_CP_SIZE", "2"),
        ("UNSLOTH_EP_SIZE", "4"),
    ):
        monkeypatch.setenv(name, value)
        assert tb._lockstep_ranks(plain) is True
        monkeypatch.delenv(name)
    for flag in ("is_fsdp_enabled", "is_deepspeed_enabled"):
        trainer = type("T", (), {flag: True})()
        assert tb._lockstep_ranks(trainer) is True


# Two-rank gloo DDP: same losses and weights whichever layout the sampler picks.
_WS, _GA, _BS, _STEPS, _VOCAB = 2, 4, 2, 4, 32


def _ddp_worker(rank, init_file, lockstep, queue):
    import torch.distributed as dist
    try:
        dist.init_process_group(
            "gloo", init_method = f"file://{init_file}", rank = rank, world_size = _WS
        )
        torch.manual_seed(0)
        rng = random.Random(0)
        n = _WS * _GA * _BS * _STEPS
        lengths = _chat_lengths(rng, n)
        lengths = [max(2, length // 64) for length in lengths]
        gen = torch.Generator().manual_seed(1)
        rows = [torch.randint(0, _VOCAB, (length,), generator = gen) for length in lengths]
        model = torch.nn.Sequential(
            torch.nn.Embedding(_VOCAB, 8), torch.nn.Linear(8, _VOCAB)
        ).double()
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.SGD(ddp.parameters(), lr = 0.1)
        sampler = tb.TokenBalancedSampler(
            torch.utils.data.SequentialSampler(range(n)),
            lengths,
            _BS,
            _GA,
            _WS,
            True,
            lockstep = lockstep,
        )
        order = list(sampler)
        batches = [order[i : i + _BS] for i in range(0, n, _BS)]
        mine = batches[rank::_WS]  # accelerate BatchSamplerShard
        losses, rank_tokens = [], []
        for step in range(_STEPS):
            micro = mine[step * _GA : (step + 1) * _GA]
            step_rows = [
                i for g in range(step * _GA * _WS, (step + 1) * _GA * _WS) for i in batches[g]
            ]
            items = sum(len(rows[i]) - 1 for i in step_rows)  # average_tokens_across_devices
            total = torch.zeros((), dtype = torch.float64)
            tokens = 0
            for g, batch in enumerate(micro):
                ids = torch.cat([rows[i][:-1] for i in batch])
                target = torch.cat([rows[i][1:] for i in batch])
                tokens += sum(len(rows[i]) for i in batch)
                loss = (
                    torch.nn.functional.cross_entropy(ddp(ids), target, reduction = "sum")
                    / items
                    * _WS
                )
                if g < _GA - 1:
                    with ddp.no_sync():
                        loss.backward()
                else:
                    loss.backward()
                total += loss.detach() / _WS
            dist.all_reduce(total)
            opt.step()
            opt.zero_grad()
            losses.append(total.item())
            rank_tokens.append(tokens)
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        queue.put((rank, {"losses": losses, "params": params.tolist(), "tokens": rank_tokens}))
        dist.destroy_process_group()
    except BaseException as error:  # noqa: BLE001
        queue.put((rank, repr(error)))
        raise


def _ddp_run(lockstep):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg")
        procs = [
            ctx.Process(target = _ddp_worker, args = (r, init_file, lockstep, queue))
            for r in range(_WS)
        ]
        for proc in procs:
            proc.start()
        results = dict(queue.get(timeout = 120) for _ in procs)
        for proc in procs:
            proc.join(timeout = 60)
    for rank, result in results.items():
        assert isinstance(result, dict), f"rank {rank} failed: {result}"
    return results


@pytest.mark.skipif(sys.platform != "linux", reason = "fork + gloo")
def test_gloo_ddp_losses_and_weights_match_across_layouts():
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("needs gloo")
    old, new = _ddp_run(True), _ddp_run(False)
    assert old[0]["losses"] == pytest.approx(new[0]["losses"], rel = 1e-12, abs = 1e-12)
    assert old[1]["losses"] == pytest.approx(new[0]["losses"], rel = 1e-12, abs = 1e-12)
    for rank in range(_WS):
        assert old[rank]["params"] == pytest.approx(new[rank]["params"], rel = 1e-10, abs = 1e-12)
    # The layouts really differ, and the new one is at least as even.
    assert [old[r]["tokens"] for r in range(_WS)] != [new[r]["tokens"] for r in range(_WS)]
    heaviest = lambda res: sum(max(res[r]["tokens"][s] for r in range(_WS)) for s in range(_STEPS))
    assert heaviest(new) <= heaviest(old)


@pytest.mark.parametrize("fsdp2", [None, "1"])
def test_default_ddp_sampler_uses_the_ddp_layout(monkeypatch, fsdp2):
    datasets = pytest.importorskip("datasets")
    accelerate_data_loader = pytest.importorskip("accelerate.data_loader")
    from types import SimpleNamespace

    for name in (
        "UNSLOTH_TOKEN_BALANCE",
        "UNSLOTH_TOKEN_BUDGET",
        "UNSLOTH_FSDP2",
        "UNSLOTH_TP_SIZE",
        "UNSLOTH_PP_SIZE",
        "UNSLOTH_CP_SIZE",
        "UNSLOTH_EP_SIZE",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    if fsdp2:
        monkeypatch.setenv("UNSLOTH_FSDP2", fsdp2)
    dataset = datasets.Dataset.from_dict(
        {"input_ids": [[1] * k for k in _chat_lengths(random.Random(0), 64)]}
    )
    inner = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(dataset), 2, drop_last = False
    )
    shard = accelerate_data_loader.BatchSamplerShard(
        inner, num_processes = 2, process_index = 0, split_batches = False
    )
    loader = SimpleNamespace(batch_sampler = shard, dataset = dataset)
    trainer = SimpleNamespace(
        training_step = None,
        get_train_dataloader = lambda: loader,
        args = SimpleNamespace(gradient_accumulation_steps = 4, average_tokens_across_devices = True),
        model_accepts_loss_kwargs = True,
    )
    tb.install_token_budget(trainer)
    trainer.get_train_dataloader()
    assert trainer._unsloth_token_balance_active
    assert inner.sampler.lockstep is bool(fsdp2)
