# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_TOKEN_BALANCE / UNSLOTH_TOKEN_BUDGET: partitioner, step invariants, exact split loss. CPU only."""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "token_budget.py"
_spec = importlib.util.spec_from_file_location("_unsloth_token_budget", _PATH)
tb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tb
_spec.loader.exec_module(tb)


def _sums(values, groups):
    return [sum(values[i] for i in g) for g in groups]


def _best_spread(values, k):
    best = None
    for labels in itertools.product(range(k), repeat = len(values)):
        sums = [0] * k
        for v, g in zip(values, labels):
            sums[g] += v
        spread = max(sums) - min(sums)
        best = spread if best is None else min(best, spread)
    return best


@pytest.mark.parametrize("seed", range(20))
def test_karmarkar_karp_is_a_partition_close_to_optimal(seed):
    rng = random.Random(seed)
    values = [rng.randint(1, 100) for _ in range(rng.randint(3, 9))]
    k = rng.choice([2, 3])
    groups = tb.karmarkar_karp(values, k)
    assert sorted(i for g in groups for i in g) == list(range(len(values)))
    sums = _sums(values, groups)
    assert max(sums) - min(sums) <= _best_spread(values, k) + max(values)


def test_karmarkar_karp_equal_size_and_classic_example():
    groups = tb.karmarkar_karp([8, 7, 6, 5, 4], 2)
    assert sorted(_sums([8, 7, 6, 5, 4], groups)) == [14, 16]
    rng = random.Random(0)
    values = [rng.randint(1, 4096) for _ in range(64)]
    groups = tb.karmarkar_karp(values, 8, equal_size = True)
    assert all(len(g) == 8 for g in groups)
    assert sorted(i for g in groups for i in g) == list(range(64))
    sums = _sums(values, groups)
    assert max(sums) - min(sums) < 0.05 * (sum(values) / 8)
    with pytest.raises(ValueError):
        tb.karmarkar_karp(values[:63], 8, equal_size = True)


def _layout(order, bs, ga, ws):
    """accelerate BatchSamplerShard: global batch b -> rank b % ws, the rank's (b // ws)-th batch."""
    batches = [order[i : i + bs] for i in range(0, len(order), bs)]
    return [[batches[g * ws + r] for g in range(ga)] for r in range(ws)]


def _num_items(labels, batches):
    # HF / zoo count: shifted labels that are not -100, summed over the step's micro-batches.
    return sum(int((labels[i][1:] != -100).sum()) for batch in batches for i in batch)


@pytest.mark.parametrize("ws,ga,bs", [(1, 4, 2), (2, 2, 3), (4, 2, 2), (2, 1, 4)])
@pytest.mark.parametrize("cross_rank", [True, False])
def test_step_rows_and_num_items_are_unchanged(ws, ga, bs, cross_rank):
    rng = random.Random(ws * 100 + ga * 10 + bs)
    n = ws * ga * bs
    lengths = [rng.choice([16, 32, 512, 2048]) for _ in range(n)]
    labels = []
    for length in lengths:
        row = torch.randint(0, 50, (length,))
        row[: rng.randint(0, length - 1)] = -100
        labels.append(row)
    order = list(range(n))
    rng.shuffle(order)
    balanced = tb.balance_step_indices(order, lengths, bs, ga, ws, cross_rank)
    assert sorted(balanced) == sorted(order)

    before = _layout(order, bs, ga, ws)
    after = _layout(balanced, bs, ga, ws)
    assert sum(_num_items(labels, r) for r in before) == sum(_num_items(labels, r) for r in after)
    assert all(len(b) == bs for r in after for b in r)
    if not cross_rank or ws == 1:
        for r in range(ws):
            assert sorted(sum(before[r], [])) == sorted(sum(after[r], []))
            assert _num_items(labels, before[r]) == _num_items(labels, after[r])


def test_balancing_evens_out_rank_and_slot_tokens():
    bs, ga, ws = 4, 4, 4
    lengths = [4096] * 8 + [64] * (bs * ga * ws - 8)
    order = list(range(len(lengths)))
    before = _layout(order, bs, ga, ws)
    after = _layout(tb.balance_step_indices(order, lengths, bs, ga, ws, True), bs, ga, ws)

    def rank_tokens(layout):
        return [sum(lengths[i] for b in r for i in b) for r in layout]

    def slot_max(layout):
        return max(sum(lengths[i] for i in b) for r in layout for b in r)

    assert max(rank_tokens(before)) - min(rank_tokens(before)) > 10_000
    assert max(rank_tokens(after)) - min(rank_tokens(after)) == 0
    assert slot_max(after) < slot_max(before)


def test_sampler_keeps_the_index_multiset_and_forwards_set_epoch():
    lengths = [random.Random(i).randint(1, 999) for i in range(23)]
    base = torch.utils.data.RandomSampler(range(23), generator = torch.Generator().manual_seed(0))
    sampler = tb.TokenBalancedSampler(base, lengths, batch_size = 2, grad_accum = 2, world_size = 2)
    out = list(sampler)
    assert len(sampler) == 23 and sorted(out) == list(range(23))
    calls = []
    base.set_epoch = calls.append
    sampler.set_epoch(3)
    assert calls == [3]
    assert sampler.generator is base.generator


def _padding_free_batch(segments, mask_boundaries = True):
    total = sum(segments)
    torch.manual_seed(0)
    input_ids = torch.randint(1, 30, (1, total))
    labels = input_ids.clone()
    position_ids = torch.cat([torch.arange(s) for s in segments]).unsqueeze(0)
    if mask_boundaries:
        labels[position_ids == 0] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "packed_seq_lengths": torch.tensor(segments, dtype = torch.int32),
    }


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)

    def loss(self, batch, num_items):
        logits = self.head(torch.tanh(self.emb(batch["input_ids"])))
        shift_logits = logits[:, :-1].reshape(-1, 32)
        shift_labels = batch["labels"][:, 1:].reshape(-1)
        return (
            torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index = -100, reduction = "sum"
            )
            / num_items
        )


def _grads(model, batches, num_items):
    model.zero_grad()
    total = 0.0
    for batch in batches:
        loss = model.loss(batch, num_items)
        loss.backward()
        total += float(loss)
    return total, [p.grad.clone() for p in model.parameters()]


def test_padding_free_split_is_exact_and_respects_the_budget():
    batch = _padding_free_batch([5, 9, 3, 12, 7, 4])
    chunks = tb.split_batch_by_token_budget(batch, 16)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["input_ids"].shape[1] <= 16 or len(chunk["packed_seq_lengths"]) == 1
        assert int(chunk["packed_seq_lengths"].sum()) == chunk["input_ids"].shape[1]
        assert int(chunk["position_ids"][0, 0]) == 0
    for key in ("input_ids", "labels", "position_ids"):
        assert torch.equal(torch.cat([c[key] for c in chunks], dim = 1), batch[key])

    num_items = int((batch["labels"][:, 1:] != -100).sum())
    model = _TinyLM()
    full_loss, full_grads = _grads(model, [batch], num_items)
    split_loss, split_grads = _grads(model, chunks, num_items)
    assert abs(full_loss - split_loss) < 1e-5
    for a, b in zip(full_grads, split_grads):
        torch.testing.assert_close(a, b, atol = 1e-6, rtol = 1e-5)


def test_split_refuses_when_a_cut_would_drop_a_real_target():
    batch = _padding_free_batch([5, 9, 3, 12], mask_boundaries = False)
    assert tb.split_batch_by_token_budget(batch, 10) == [batch]


def test_split_leaves_small_unknown_and_vision_batches_alone():
    batch = _padding_free_batch([5, 6])
    assert tb.split_batch_by_token_budget(batch, 64) == [batch]
    vision = {"input_ids": torch.ones(4, 8, dtype = torch.long), "pixel_values": torch.ones(7, 3)}
    assert tb.split_batch_by_token_budget(vision, 8) == [vision]


def test_padded_rows_split_is_exact():
    torch.manual_seed(1)
    input_ids = torch.randint(1, 30, (6, 10))
    labels = input_ids.clone()
    labels[:, :3] = -100
    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}
    chunks = tb.split_batch_by_token_budget(batch, 20)
    assert [c["input_ids"].shape[0] for c in chunks] == [2, 2, 2]
    num_items = int((labels[:, 1:] != -100).sum())
    model = _TinyLM()
    full_loss, full_grads = _grads(model, [batch], num_items)
    split_loss, split_grads = _grads(model, chunks, num_items)
    assert abs(full_loss - split_loss) < 1e-5
    for a, b in zip(full_grads, split_grads):
        torch.testing.assert_close(a, b, atol = 1e-6, rtol = 1e-5)


class _FakeAccelerator:
    distributed_type = "DistributedType.MULTI_GPU"

    def __init__(self):
        self.no_sync_calls = 0

    @contextlib.contextmanager
    def no_sync(self, model):
        self.no_sync_calls += 1
        yield


def _fake_trainer():
    seen = []

    def training_step(
        model,
        inputs,
        num_items_in_batch = None,
    ):
        seen.append((inputs["input_ids"].shape[1], num_items_in_batch))
        return torch.tensor(float(inputs["input_ids"].shape[1]) / num_items_in_batch)

    trainer = SimpleNamespace(
        training_step = training_step,
        get_train_dataloader = lambda: None,
        accelerator = _FakeAccelerator(),
        args = SimpleNamespace(gradient_accumulation_steps = 1),
        model_accepts_loss_kwargs = True,
        label_smoother = None,
    )
    return trainer, seen


def test_install_is_a_no_op_without_the_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    trainer, _ = _fake_trainer()
    step, loader = trainer.training_step, trainer.get_train_dataloader
    tb.install_token_budget(trainer)
    assert trainer.training_step is step and trainer.get_train_dataloader is loader


def test_budgeted_training_step_sums_chunks_and_syncs_once(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TOKEN_BUDGET", "16")
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    trainer, seen = _fake_trainer()
    tb.install_token_budget(trainer)
    batch = _padding_free_batch([5, 9, 3, 12, 7, 4])
    loss = trainer.training_step(None, batch, 100)
    assert len(seen) > 1 and all(n == 100 for _, n in seen)
    assert sum(width for width, _ in seen) == 40
    assert abs(float(loss) - 0.40) < 1e-6
    assert trainer.accelerator.no_sync_calls == len(seen) - 1
    assert trainer._unsloth_token_budget_splits == 1
    # Without a token count the loss is a per-call mean, so the batch is not split.
    seen.clear()
    with pytest.raises(TypeError):
        # The fake divides by num_items_in_batch: reaching it unsplit with None proves the pass-through.
        trainer.training_step(None, batch, None)
    assert seen == [(40, None)]


def test_balanced_dataloader_end_to_end_with_accelerate_shards(monkeypatch):
    accelerate_data_loader = pytest.importorskip("accelerate.data_loader")
    monkeypatch.setenv("UNSLOTH_TOKEN_BALANCE", "1")
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    datasets = pytest.importorskip("datasets")
    rng = random.Random(0)
    lengths = [rng.choice([8, 8, 8, 600]) for _ in range(64)]
    dataset = datasets.Dataset.from_dict({"input_ids": [[1] * n for n in lengths]})
    bs, ga, ws = 2, 4, 2

    def make(rank):
        sampler = torch.utils.data.SequentialSampler(dataset)
        inner = torch.utils.data.BatchSampler(sampler, bs, drop_last = False)
        shard = accelerate_data_loader.BatchSamplerShard(
            inner, num_processes = ws, process_index = rank, split_batches = False
        )
        loader = SimpleNamespace(batch_sampler = shard, dataset = dataset)
        trainer = SimpleNamespace(
            training_step = None,
            get_train_dataloader = lambda: loader,
            args = SimpleNamespace(
                gradient_accumulation_steps = ga, average_tokens_across_devices = True
            ),
            model_accepts_loss_kwargs = True,
        )
        tb.install_token_budget(trainer)
        trainer.get_train_dataloader()
        assert trainer._unsloth_token_balance_active
        return list(shard)

    per_rank = [make(r) for r in range(ws)]
    steps = len(per_rank[0]) // ga
    for s in range(steps):
        step_rows = [i for r in range(ws) for b in per_rank[r][s * ga : (s + 1) * ga] for i in b]
        assert sorted(step_rows) == list(range(s * ga * ws * bs, (s + 1) * ga * ws * bs))
        tokens = [
            sum(lengths[i] for b in per_rank[r][s * ga : (s + 1) * ga] for i in b)
            for r in range(ws)
        ]
        assert max(tokens) - min(tokens) <= max(lengths)


class _MeanLabelSmoother:
    # transformers < 5 LabelSmoother.__call__ signature: no num_items_in_batch, returns a per-call mean.
    def __call__(
        self,
        model_output,
        labels,
        shift_labels = False,
    ):
        raise AssertionError("not called")


class _CountLabelSmoother:
    def __call__(
        self,
        model_output,
        labels,
        shift_labels = False,
        num_items_in_batch = None,
    ):
        raise AssertionError("not called")


def _moe_model(coef):
    config = SimpleNamespace(output_router_logits = True, router_aux_loss_coef = coef)
    return SimpleNamespace(config = config)


@pytest.mark.parametrize(
    "override,splits",
    [
        ({}, True),
        ({"label_smoother": _CountLabelSmoother()}, True),
        ({"label_smoother": _MeanLabelSmoother()}, False),
        ({"model_accepts_loss_kwargs": False}, False),
        ({"aux_loss_enabled": True}, False),
        ({"model": _moe_model(0.01)}, False),
        ({"model": _moe_model(0.0)}, True),
    ],
)
def test_split_steps_aside_when_chunk_losses_do_not_sum_exactly(
    monkeypatch, capsys, override, splits
):
    monkeypatch.setenv("UNSLOTH_TOKEN_BUDGET", "16")
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    trainer, seen = _fake_trainer()
    for key, value in override.items():
        setattr(trainer, key, value)
    tb.install_token_budget(trainer)
    batch = _padding_free_batch([5, 9, 3, 12, 7, 4])
    trainer.training_step(None, batch, 100)
    trainer.training_step(None, batch, 100)
    assert (len(seen) > 2) == splits
    if not splits:
        assert seen == [(40, 100), (40, 100)]
        assert capsys.readouterr().out.count("UNSLOTH_TOKEN_BUDGET not splitting") == 1


def test_split_refuses_media_keys_and_per_sample_metadata():
    rows = {
        "input_ids": torch.ones(4, 8, dtype = torch.long),
        "labels": torch.ones(4, 8, dtype = torch.long),
    }
    assert len(tb.split_batch_by_token_budget(rows, 8)) == 4
    # Flattened images that happen to total B rows still must not be paired row by row.
    for extra in (
        {"pixel_values": torch.ones(4, 3)},
        {"image_grid_thw": torch.ones(4, 3, dtype = torch.long)},
        {"input_values": torch.ones(4, 16000)},
        {"num_images": [0, 2, 1, 1]},
    ):
        batch = {**rows, **extra}
        assert tb.split_batch_by_token_budget(batch, 8) == [batch]
    packed = {**_padding_free_batch([5, 9, 3, 12]), "pixel_values": torch.ones(1, 29, 3)}
    assert tb.split_batch_by_token_budget(packed, 10) == [packed]


def _shard_steps(n, bs, ga, ws, balance, **trainer_kwargs):
    accelerate_data_loader = pytest.importorskip("accelerate.data_loader")
    datasets = pytest.importorskip("datasets")
    rng = random.Random(1)
    lengths = [rng.choice([8, 8, 8, 600]) for _ in range(n)]
    dataset = datasets.Dataset.from_dict({"input_ids": [[1] * k for k in lengths]})
    per_rank, active = [], []
    for rank in range(ws):
        base = torch.utils.data.RandomSampler(dataset, generator = torch.Generator().manual_seed(3))
        inner = torch.utils.data.BatchSampler(base, bs, drop_last = False)
        shard = accelerate_data_loader.BatchSamplerShard(
            inner, num_processes = ws, process_index = rank, split_batches = False, even_batches = True
        )
        if balance:
            loader = SimpleNamespace(batch_sampler = shard, dataset = dataset)
            kwargs = {"model_accepts_loss_kwargs": True, **trainer_kwargs}
            optim = kwargs.pop("args_optim", "adamw_torch")
            trainer = SimpleNamespace(
                training_step = None,
                get_train_dataloader = lambda: loader,
                args = SimpleNamespace(
                    gradient_accumulation_steps = ga, average_tokens_across_devices = True, optim = optim
                ),
                **kwargs,
            )
            tb.install_token_budget(trainer)
            trainer.get_train_dataloader()
            active.append(trainer._unsloth_token_balance_active)
        per_rank.append(list(shard))
    steps = []
    for s in range(-(-len(per_rank[0]) // ga)):
        steps.append(
            sorted(i for r in range(ws) for b in per_rank[r][s * ga : (s + 1) * ga] for i in b)
        )
    return steps, active


@pytest.mark.parametrize("n", [61, 64, 67])
def test_balance_keeps_every_step_including_the_even_batches_tail(monkeypatch, n):
    monkeypatch.setenv("UNSLOTH_TOKEN_BALANCE", "1")
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    bs, ga, ws = 2, 4, 2
    plain, _ = _shard_steps(n, bs, ga, ws, balance = False)
    balanced, active = _shard_steps(n, bs, ga, ws, balance = True)
    assert all(active)
    assert balanced == plain


@pytest.mark.parametrize(
    "override",
    [
        {"model_accepts_loss_kwargs": False},
        {"label_smoother": _MeanLabelSmoother()},
        {"aux_loss_enabled": True},
    ],
)
def test_balance_steps_aside_when_slot_grouping_changes_the_loss(monkeypatch, capsys, override):
    monkeypatch.setenv("UNSLOTH_TOKEN_BALANCE", "1")
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    plain, _ = _shard_steps(64, 2, 4, 1, balance = False)
    balanced, active = _shard_steps(64, 2, 4, 1, balance = True, **override)
    assert active == [False] and balanced == plain
    assert "UNSLOTH_TOKEN_BALANCE ignored" in capsys.readouterr().out


def test_real_label_smoother_is_classified_by_its_signature():
    trainer_pt_utils = pytest.importorskip("transformers.trainer_pt_utils")
    import transformers

    smoother = trainer_pt_utils.LabelSmoother(epsilon = 0.1)
    trainer = SimpleNamespace(model_accepts_loss_kwargs = True, label_smoother = smoother)
    uses_count = tb._loss_uses_step_count(trainer)
    if int(transformers.__version__.split(".")[0]) < 5:
        assert not uses_count
    assert uses_count == (
        "num_items_in_batch" in __import__("inspect").signature(smoother.__call__).parameters
    )


def test_split_refuses_pairwise_masks():
    batch = _padding_free_batch([5, 9, 3, 12])
    total = batch["input_ids"].shape[1]
    batch["attention_mask"] = torch.ones(1, total, total, dtype = torch.bool).tril()
    assert tb.split_batch_by_token_budget(batch, 10) == [batch]


class _CountingTensor(torch.Tensor):
    host_reads = 0

    @classmethod
    def __torch_function__(
        cls,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func in (
            torch.Tensor.__int__,
            torch.Tensor.item,
            torch.Tensor.tolist,
            torch.Tensor.__bool__,
        ):
            cls.host_reads += 1
        return super().__torch_function__(func, types, args, kwargs or {})


def test_split_reads_boundary_labels_in_one_transfer():
    batch = _padding_free_batch([3] * 40)
    batch["labels"] = batch["labels"].as_subclass(_CountingTensor)
    _CountingTensor.host_reads = 0
    chunks = tb.split_batch_by_token_budget(batch, 6)
    assert len(chunks) == 20
    assert _CountingTensor.host_reads <= 1


def test_lomo_steps_aside_for_split_and_balance(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_TOKEN_BUDGET", "16")
    monkeypatch.delenv("UNSLOTH_TOKEN_BALANCE", raising = False)
    for optim, neftune in (
        ("lomo", None),
        ("adalomo", None),
        ("galore_adamw_layerwise", None),
        ("adamw_torch", 5.0),
    ):
        trainer, seen = _fake_trainer()
        trainer.args.optim = optim
        trainer.args.neftune_noise_alpha = neftune
        tb.install_token_budget(trainer)
        batch = _padding_free_batch([5, 9, 3, 12, 7, 4])
        trainer.training_step(None, batch, 100)
        trainer.training_step(None, batch, 100)
        assert seen == [(40, 100), (40, 100)]
        assert capsys.readouterr().out.count("UNSLOTH_TOKEN_BUDGET not splitting") == 1
    monkeypatch.setenv("UNSLOTH_TOKEN_BALANCE", "1")
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    plain, _ = _shard_steps(64, 2, 4, 1, balance = False)
    balanced, active = _shard_steps(64, 2, 4, 1, balance = True, args_optim = "lomo")
    assert active == [False] and balanced == plain


def test_balance_steps_aside_for_multi_process_dispatched_dataloaders(monkeypatch, capsys):
    accelerate_data_loader = pytest.importorskip("accelerate.data_loader")
    datasets = pytest.importorskip("datasets")
    monkeypatch.setenv("UNSLOTH_TOKEN_BALANCE", "1")
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    dataset = datasets.Dataset.from_dict({"input_ids": [[1] * (1 + i % 7) for i in range(32)]})
    loader = accelerate_data_loader.DataLoaderDispatcher(dataset, batch_size = 2)
    loader.state = SimpleNamespace(num_processes = 2)
    trainer = SimpleNamespace(
        training_step = None,
        get_train_dataloader = lambda: loader,
        args = SimpleNamespace(gradient_accumulation_steps = 4, average_tokens_across_devices = True),
        model_accepts_loss_kwargs = True,
    )
    tb.install_token_budget(trainer)
    trainer.get_train_dataloader()
    assert trainer._unsloth_token_balance_active is False
    assert not isinstance(loader.batch_sampler.sampler, tb.TokenBalancedSampler)
    assert "UNSLOTH_TOKEN_BALANCE ignored" in capsys.readouterr().out


def test_split_checks_boundary_labels_even_with_shift_labels():
    # Unsloth's fast causal LM path builds its loss from labels, not shift_labels.
    batch = _padding_free_batch([5, 9, 3, 12, 7, 4], mask_boundaries = False)
    batch["shift_labels"] = torch.roll(batch["labels"], -1, dims = 1)
    assert len(tb.split_batch_by_token_budget(batch, 16)) == 1


def test_dataset_lengths_are_read_in_batches():
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_dict(
        {"input_ids": [[1] * (1 + i % 7) for i in range(10_000)], "x": [0] * 10_000}
    )
    dataset = dataset.shuffle(seed = 0)
    assert tb._dataset_lengths(dataset) == [len(ids) for ids in dataset["input_ids"]]


_DDP_WORKER = r"""
import contextlib, importlib.util, os, sys
from types import SimpleNamespace
import torch, torch.distributed as dist

spec = importlib.util.spec_from_file_location("_tb", sys.argv[1])
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
dist.init_process_group("gloo")
rank = dist.get_rank()
segments = [[5, 9, 3, 12, 7, 4], [int(x) for x in sys.argv[2].split(",")]][rank]
total = sum(segments)
input_ids = torch.randint(1, 30, (1, total))
position_ids = torch.cat([torch.arange(s) for s in segments]).unsqueeze(0)
labels = input_ids.clone()
labels[position_ids == 0] = -100
batch = dict(input_ids = input_ids, labels = labels, position_ids = position_ids,
             packed_seq_lengths = torch.tensor(segments, dtype = torch.int32))
calls = []

def training_step(model, inputs, num_items_in_batch = None):
    calls.append(inputs["input_ids"].shape[1])
    # Stands in for TRL's per-compute_loss metric gather: one collective per call.
    dist.all_reduce(torch.ones(1))
    return torch.tensor(0.0)

accelerator = SimpleNamespace(
    distributed_type = "DistributedType.MULTI_GPU", num_processes = 2, device = torch.device("cpu"),
    no_sync = lambda model: contextlib.nullcontext(),
)
trainer = SimpleNamespace(
    training_step = training_step, get_train_dataloader = lambda: None, accelerator = accelerator,
    args = SimpleNamespace(gradient_accumulation_steps = 1), model_accepts_loss_kwargs = True,
    label_smoother = None,
)
os.environ["UNSLOTH_TOKEN_BUDGET"] = "16"
tb.install_token_budget(trainer)
trainer.training_step(None, batch, 100)
print("CALLS", rank, len(calls), flush = True)
dist.destroy_process_group()
"""


@pytest.mark.parametrize("other, expected", [("8,8", (1, 1)), ("4,5,3,6,2,4,6,4,6", (3, 3))])
def test_ddp_ranks_split_only_when_every_rank_needs_the_same_count(tmp_path, other, expected):
    import socket
    import subprocess

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        __import__("os").environ,
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        WORLD_SIZE = "2",
    )
    env.pop("UNSLOTH_TOKEN_BALANCE", None)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _DDP_WORKER, str(_PATH), other],
            env = dict(env, RANK = str(rank)),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
        )
        for rank in range(2)
    ]
    outputs = []
    try:
        for p in procs:
            outputs.append(p.communicate(timeout = 120)[0])
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    calls = {}
    for out in outputs:
        for line in out.splitlines():
            if line.startswith("CALLS"):
                _, rank, n = line.split()
                calls[int(rank)] = int(n)
    assert (calls.get(0), calls.get(1)) == expected, outputs


@pytest.mark.parametrize("key", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("kind", ["longrope", "dynamic"])
def test_length_dependent_rope_steps_aside_for_split_and_balance(key, kind):
    trainer, _ = _fake_trainer()
    assert tb._exactness_blocker(trainer) is None
    config = SimpleNamespace(**{key: {"rope_type": kind, "original_max_position_embeddings": 4096}})
    trainer.model = SimpleNamespace(config = config)
    assert kind in tb._exactness_blocker(trainer)
    # Per layer type (transformers 5 rope_parameters).
    nested = {"full_attention": {"rope_type": kind}, "sliding_attention": {"rope_type": "default"}}
    trainer.model = SimpleNamespace(config = SimpleNamespace(**{key: nested}))
    assert kind in tb._exactness_blocker(trainer)
    trainer.model = SimpleNamespace(config = SimpleNamespace(**{key: {"rope_type": "llama3"}}))
    assert tb._exactness_blocker(trainer) is None


def test_balance_steps_aside_for_padded_batches(monkeypatch, capsys):
    # Padded rows: balancing token sums spreads the long rows over every micro-batch, so the
    # padded token count grows while the longest micro-batch, and so the peak, stays the same.
    datasets = pytest.importorskip("datasets")
    monkeypatch.setenv("UNSLOTH_TOKEN_BALANCE", "1")
    monkeypatch.delenv("UNSLOTH_TOKEN_BUDGET", raising = False)
    rng = random.Random(0)
    lengths = [rng.choice([8, 8, 8, 600]) for _ in range(64)]
    dataset = datasets.Dataset.from_dict({"input_ids": [[1] * n for n in lengths]})
    bs, ga = 2, 4

    def padded_tokens(padding_free, packing = False):
        sampler = torch.utils.data.SequentialSampler(dataset)
        inner = torch.utils.data.BatchSampler(sampler, bs, drop_last = False)
        loader = SimpleNamespace(batch_sampler = inner, dataset = dataset)
        trainer = SimpleNamespace(
            training_step = None,
            get_train_dataloader = lambda: loader,
            args = SimpleNamespace(
                gradient_accumulation_steps = ga,
                average_tokens_across_devices = True,
                padding_free = padding_free,
                packing = packing,
            ),
            model_accepts_loss_kwargs = True,
        )
        tb.install_token_budget(trainer)
        trainer.get_train_dataloader()
        return trainer._unsloth_token_balance_active, sum(
            len(b) * max(lengths[i] for i in b) for b in inner
        )

    plain = sum(
        len(lengths[s : s + bs]) * max(lengths[s : s + bs]) for s in range(0, len(lengths), bs)
    )
    active, tokens = padded_tokens(padding_free = False)
    assert active is False and tokens == plain
    assert "batches are padded" in capsys.readouterr().out
    assert padded_tokens(padding_free = True)[0] is True
    assert padded_tokens(padding_free = False, packing = True)[0] is True
