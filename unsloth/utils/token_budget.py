# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in token-aware SFT micro-batches, off unless their env var is set.

UNSLOTH_TOKEN_BALANCE=1 spreads each optimizer step's rows over data-parallel ranks and
accumulation slots with equal token counts; every step keeps its rows, so num_items_in_batch
is unchanged. UNSLOTH_TOKEN_BUDGET=<tokens> runs a larger micro-batch as several passes over
whole sequences, each divided by the same num_items_in_batch, so they sum to the unsplit loss.
"""

from __future__ import annotations

import contextlib
import heapq
import itertools
import os
from collections.abc import Mapping

import torch

__all__ = [
    "karmarkar_karp",
    "balance_step_indices",
    "TokenBalancedSampler",
    "split_batch_by_token_budget",
    "install_token_budget",
]

_TOKEN_BALANCE_ENV = "UNSLOTH_TOKEN_BALANCE"
_TOKEN_BUDGET_ENV = "UNSLOTH_TOKEN_BUDGET"


def karmarkar_karp(values, k, equal_size = False):
    """Split indices of `values` into `k` groups with near-equal sums (largest differencing method).

    With `equal_size`, `len(values)` must be a multiple of `k` and every group gets the same count.
    Deterministic: ties break on index. Mirrors verl/utils/seqlen_balancing.py:karmarkar_karp.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n = len(values)
    if equal_size and n % k != 0:
        raise ValueError(f"equal_size needs len(values) % k == 0, got {n} % {k}")
    if k == 1:
        return [list(range(n))]
    order = sorted(range(n), key = lambda i: (values[i], i))
    counter = itertools.count()
    heap = []

    def push(sets):
        sets.sort(key = lambda s: (s[0], len(s[1]), s[1]), reverse = True)
        heapq.heappush(heap, (-(sets[0][0] - sets[-1][0]), next(counter), sets))

    if equal_size:
        for start in range(0, n, k):
            push([[values[i], [i]] for i in order[start : start + k]])
    else:
        for i in order:
            push([[values[i], [i]]] + [[0, []] for _ in range(k - 1)])
    if not heap:
        return [[] for _ in range(k)]
    while len(heap) > 1:
        a = heapq.heappop(heap)[2]
        b = heapq.heappop(heap)[2]
        # Largest of one state with the smallest of the other cancels the biggest difference.
        push([[a[j][0] + b[k - 1 - j][0], a[j][1] + b[k - 1 - j][1]] for j in range(k)])
    return [sorted(s[1]) for s in heap[0][2]]


def balance_step_indices(
    indices,
    lengths,
    batch_size,
    grad_accum,
    world_size = 1,
    cross_rank = True,
):
    """Reorder one optimizer step's sampler output (`world_size * grad_accum * batch_size` indices).

    The layout follows accelerate's BatchSamplerShard: consecutive `batch_size` chunks are global
    batches, global batch b goes to rank b % world_size, and rank r's g-th batch of the step is
    global batch g * world_size + r. With `cross_rank` rows move between ranks to equalize per-rank
    tokens; without it each rank keeps its own rows and only its accumulation slots are balanced.
    """
    bs, ga, ws = int(batch_size), int(grad_accum), int(world_size)
    window = bs * ga * ws
    if len(indices) != window:
        raise ValueError(f"expected {window} indices, got {len(indices)}")
    if ga * ws == 1:
        return list(indices)
    if cross_rank and ws > 1:
        groups = karmarkar_karp([lengths[i] for i in indices], ws, equal_size = True)
        rank_rows = [[indices[p] for p in group] for group in groups]
    else:
        rank_rows = [
            [indices[(g * ws + r) * bs + j] for g in range(ga) for j in range(bs)]
            for r in range(ws)
        ]
    out = [None] * window
    for r, rows in enumerate(rank_rows):
        slots = karmarkar_karp([lengths[i] for i in rows], ga, equal_size = True)
        for g, slot in enumerate(slots):
            base = (g * ws + r) * bs
            for j, p in enumerate(slot):
                out[base + j] = rows[p]
    return out


class TokenBalancedSampler(torch.utils.data.Sampler):
    """Applies `balance_step_indices` to each full step window; a partial tail passes through."""

    def __init__(
        self,
        sampler,
        lengths,
        batch_size,
        grad_accum,
        world_size = 1,
        cross_rank = True,
    ):
        self.sampler = sampler
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.world_size = int(world_size)
        self.cross_rank = bool(cross_rank)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        window = self.batch_size * self.grad_accum * self.world_size
        buffer = []
        for index in self.sampler:
            buffer.append(index)
            if len(buffer) == window:
                yield from balance_step_indices(
                    buffer,
                    self.lengths,
                    self.batch_size,
                    self.grad_accum,
                    self.world_size,
                    self.cross_rank,
                )
                buffer = []
        yield from buffer

    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __getattr__(self, name):
        # accelerate and the Trainer read generator / data_source / epoch off the sampler.
        if name == "sampler":
            raise AttributeError(name)
        return getattr(self.sampler, name)


def _segments_of(batch, total):
    lengths = batch.get("packed_seq_lengths")
    if isinstance(lengths, torch.Tensor):
        lengths = [int(x) for x in lengths.tolist()]
        return lengths if sum(lengths) == total else None
    position_ids = batch.get("position_ids")
    if not isinstance(position_ids, torch.Tensor) or tuple(position_ids.shape) != (1, total):
        return None
    starts = (position_ids[0] == 0).nonzero(as_tuple = False).flatten().tolist()
    if not starts or starts[0] != 0:
        return None
    return [b - a for a, b in zip(starts, starts[1:] + [total])]


def _split_padding_free(batch, budget):
    input_ids = batch["input_ids"]
    total = int(input_ids.shape[1])
    if total <= budget:
        return None
    segments = _segments_of(batch, total)
    if segments is None or len(segments) < 2:
        return None
    for key, value in batch.items():
        if key == "packed_seq_lengths" or not isinstance(value, torch.Tensor):
            continue
        if value.dim() < 2 or value.shape[0] != 1 or value.shape[1] != total:
            return None
    groups, current, used = [], [], 0
    for length in segments:
        if current and used + length > budget:
            groups.append(current)
            current, used = [], 0
        current.append(length)
        used += length
    groups.append(current)
    if len(groups) < 2:
        return None
    labels = batch.get("labels")
    chunks, start = [], 0
    for group in groups:
        stop = start + sum(group)
        # The last token of a chunk loses its next-token target, exact only when that target is masked.
        if (
            isinstance(labels, torch.Tensor)
            and "shift_labels" not in batch
            and stop < total
            and int(labels[0, stop]) != -100
        ):
            return None
        chunk = {}
        for key, value in batch.items():
            if key == "packed_seq_lengths":
                chunk[key] = torch.tensor(group, dtype = value.dtype, device = value.device)
            elif isinstance(value, torch.Tensor):
                chunk[key] = value[:, start:stop]
            else:
                chunk[key] = value
        chunks.append(chunk)
        start = stop
    return chunks


def _split_rows(batch, budget):
    input_ids = batch["input_ids"]
    rows, width = int(input_ids.shape[0]), int(input_ids.shape[1])
    per_chunk = max(1, budget // max(width, 1))
    if rows <= per_chunk:
        return None
    for value in batch.values():
        if isinstance(value, torch.Tensor) and (value.dim() == 0 or value.shape[0] != rows):
            return None
    chunks = []
    for start in range(0, rows, per_chunk):
        chunks.append(
            {
                key: value[start : start + per_chunk] if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
        )
    return chunks


def split_batch_by_token_budget(batch, budget):
    """Split a collated text batch into sub-batches of at most `budget` tokens, else return `[batch]`.

    Padding-free batches ([1, T] with sequence boundaries) split between whole sequences; padded
    batches ([B, L]) split by rows. Anything else (vision tensors, unknown shapes) is not split.
    """
    if not isinstance(batch, Mapping) or budget is None or budget <= 0:
        return [batch]
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
        return [batch]
    if input_ids.shape[0] == 1 and (
        "packed_seq_lengths" in batch or "position_ids" in batch
    ):
        chunks = _split_padding_free(batch, int(budget))
    else:
        chunks = _split_rows(batch, int(budget))
    return chunks or [batch]


def _env_int(name):
    value = os.environ.get(name, "").strip()
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Unsloth: {name} must be an integer, got {value!r}.")


def _dataset_lengths(dataset):
    column_names = getattr(dataset, "column_names", None)
    if column_names is None or "input_ids" not in column_names:
        return None
    return [len(ids) for ids in dataset["input_ids"]]


def _balance_dataloader(trainer, dataloader):
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    world_size = 1
    if hasattr(batch_sampler, "num_processes") and hasattr(batch_sampler, "batch_sampler"):
        # accelerate's BatchSamplerShard with split_batches False deals whole batches round robin.
        if getattr(batch_sampler, "split_batches", False):
            return False
        world_size = int(batch_sampler.num_processes)
        batch_sampler = batch_sampler.batch_sampler
    if not isinstance(batch_sampler, torch.utils.data.BatchSampler):
        return False
    sampler = batch_sampler.sampler
    if isinstance(sampler, TokenBalancedSampler):
        return True
    if not isinstance(sampler, (torch.utils.data.RandomSampler, torch.utils.data.SequentialSampler)):
        # group_by_length / batch_rebalance already pick their own order.
        return False
    lengths = _dataset_lengths(getattr(dataloader, "dataset", None))
    if lengths is None:
        return False
    args = trainer.args
    # Moving rows between ranks keeps the loss exact only when the token count is global.
    cross_rank = world_size > 1 and bool(getattr(args, "average_tokens_across_devices", False))
    batch_sampler.sampler = TokenBalancedSampler(
        sampler,
        lengths,
        batch_sampler.batch_size,
        args.gradient_accumulation_steps,
        world_size,
        cross_rank,
    )
    return True


def install_token_budget(trainer):
    """Install whichever of UNSLOTH_TOKEN_BALANCE / UNSLOTH_TOKEN_BUDGET is set on `trainer`."""
    balance = os.environ.get(_TOKEN_BALANCE_ENV, "0").lower() in ("1", "true", "yes", "on")
    budget = _env_int(_TOKEN_BUDGET_ENV)
    if (not balance and budget <= 0) or getattr(trainer, "_unsloth_token_budget_installed", False):
        return
    trainer._unsloth_token_budget_installed = True

    if balance:
        original_get_train_dataloader = trainer.get_train_dataloader

        def get_train_dataloader(*args, **kwargs):
            dataloader = original_get_train_dataloader(*args, **kwargs)
            trainer._unsloth_token_balance_active = _balance_dataloader(trainer, dataloader)
            if not trainer._unsloth_token_balance_active:
                print("Unsloth: UNSLOTH_TOKEN_BALANCE ignored for this dataloader.")
            return dataloader

        trainer.get_train_dataloader = get_train_dataloader

    if budget > 0:
        original_training_step = trainer.training_step
        trainer._unsloth_token_budget_splits = 0

        def training_step(model, inputs, num_items_in_batch = None, *args, **kwargs):
            accelerator = getattr(trainer, "accelerator", None)
            distributed_type = str(getattr(accelerator, "distributed_type", "NO")).upper()
            # Only None (loss would be a per-chunk mean), plain or DDP: DeepSpeed and FSDP count micro-steps themselves.
            if num_items_in_batch is None or not (
                distributed_type.endswith("NO") or distributed_type.endswith("MULTI_GPU")
            ):
                return original_training_step(model, inputs, num_items_in_batch, *args, **kwargs)
            chunks = split_batch_by_token_budget(inputs, budget)
            if len(chunks) == 1:
                return original_training_step(model, inputs, num_items_in_batch, *args, **kwargs)
            trainer._unsloth_token_budget_splits += 1
            total = None
            for i, chunk in enumerate(chunks):
                context = (
                    accelerator.no_sync(model)
                    if i < len(chunks) - 1 and accelerator is not None
                    else contextlib.nullcontext()
                )
                with context:
                    loss = original_training_step(model, chunk, num_items_in_batch, *args, **kwargs)
                total = loss if total is None else total + loss
            return total

        trainer.training_step = training_step
