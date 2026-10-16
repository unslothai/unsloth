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
import inspect
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


def karmarkar_karp(
    values,
    k,
    equal_size = False,
):
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
    """Applies `balance_step_indices` to each full step window; a partial tail passes through.

    `keep_first_window` passes the first window through unchanged: accelerate's BatchSamplerShard
    (even_batches) pads an uneven tail with the first `world_size * batch_size` indices it saw.
    """

    def __init__(
        self,
        sampler,
        lengths,
        batch_size,
        grad_accum,
        world_size = 1,
        cross_rank = True,
        keep_first_window = False,
    ):
        self.sampler = sampler
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.world_size = int(world_size)
        self.cross_rank = bool(cross_rank)
        self.keep_first_window = bool(keep_first_window)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        window = self.batch_size * self.grad_accum * self.world_size
        buffer = []
        first = self.keep_first_window
        for index in self.sampler:
            buffer.append(index)
            if len(buffer) == window and first:
                first = False
                yield from buffer
                buffer = []
            elif len(buffer) == window:
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
        # A pairwise [1, T, T] mask or bias would need both token axes cut.
        if total in value.shape[2:]:
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
    stops = list(itertools.accumulate(sum(group) for group in groups))
    labels = batch.get("labels")
    # The last token of a chunk loses its next-token target, exact only when that target is masked.
    # Checked even with shift_labels: Unsloth's fast causal LM path builds its loss from labels.
    if isinstance(labels, torch.Tensor):
        cuts = torch.tensor(stops[:-1], device = labels.device)
        if bool((labels[0, cuts] != -100).any()):
            return None
    chunks, start = [], 0
    for group, stop in zip(groups, stops):
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


# Media tensors need not have one row per text row (flattened images, patches), so their batches are never split.
_MEDIA_KEY_PARTS = (
    "pixel",
    "image",
    "video",
    "audio",
    "vision",
    "input_features",
    "input_values",
    "grid_thw",
    "aspect_ratio",
    "cross_attention",
)


def _splittable_keys(batch):
    for key, value in batch.items():
        if any(part in str(key).lower() for part in _MEDIA_KEY_PARTS):
            return False
        # Per-sample metadata in a list would be copied whole into every chunk.
        if isinstance(value, (list, tuple, Mapping)):
            return False
    return True


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
    if not _splittable_keys(batch):
        return [batch]
    if input_ids.shape[0] == 1 and ("packed_seq_lengths" in batch or "position_ids" in batch):
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
    # Batched, so the whole token column is never materialised as Python lists at once.
    if hasattr(dataset, "select_columns") and hasattr(dataset, "iter"):
        lengths = []
        for rows in dataset.select_columns(["input_ids"]).iter(batch_size = 4096):
            lengths.extend(len(ids) for ids in rows["input_ids"])
        return lengths
    return [len(ids) for ids in dataset["input_ids"]]


def _accepts_num_items(fn):
    try:
        return "num_items_in_batch" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _loss_uses_step_count(trainer):
    """True when each micro-batch loss is a sum divided by the step's shared num_items_in_batch."""
    if getattr(trainer, "compute_loss_func", None) is not None:
        return True
    # Without loss kwargs the Trainer counts nothing and every micro-batch loss is its own mean.
    if not getattr(trainer, "model_accepts_loss_kwargs", False):
        return False
    smoother = getattr(trainer, "label_smoother", None)
    # transformers < 5 LabelSmoother returns a per-call mean.
    return smoother is None or _accepts_num_items(smoother.__call__)


def _router_aux_loss_active(trainer):
    """MoE load-balancing loss is a nonlinear per-batch statistic, so regrouping rows changes it."""
    if getattr(trainer, "aux_loss_enabled", False):
        return True
    model = getattr(trainer, "model", None)
    model = getattr(model, "module", model)
    config = getattr(model, "config", None)
    configs = [config]
    if hasattr(config, "get_text_config"):
        configs.append(config.get_text_config())
    for cfg in configs:
        if (
            getattr(cfg, "output_router_logits", False)
            and getattr(cfg, "router_aux_loss_coef", 1.0) != 0
        ):
            return True
    return False


def _exactness_blocker(trainer):
    args = getattr(trainer, "args", None)
    optim = str(getattr(args, "optim", "")).lower()
    # LOMO / AdaLomo and the *_layerwise optimizers update weights inside every backward.
    if "lomo" in optim or "layerwise" in optim:
        return "this optimizer updates the weights on every backward"
    # NEFTune scales its noise by the forward's sequence length, which a split shortens.
    if getattr(args, "neftune_noise_alpha", None):
        return "NEFTune noise depends on the sequence length of each forward"
    if not _loss_uses_step_count(trainer):
        return (
            "the loss is a per-micro-batch mean (no shared num_items_in_batch, or label smoothing)"
        )
    if _router_aux_loss_active(trainer):
        return "the MoE router auxiliary loss is active"
    rope = _length_dependent_rope(trainer)
    if rope is not None:
        return f"{rope} RoPE picks its frequencies from each forward's length"
    return None


def _length_dependent_rope(trainer):
    """LongRoPE or dynamic RoPE anywhere in the text config (flat or per layer type): both pick
    frequencies from each forward's length, which splitting or regrouping changes."""
    config = getattr(getattr(trainer, "model", None), "config", None)
    config = getattr(config, "text_config", None) or config

    def kinds(value):
        if isinstance(value, Mapping):
            kind = value.get("rope_type", value.get("type"))
            if kind is not None:
                yield str(kind).lower()
            for item in value.values():
                yield from kinds(item)

    for name in ("rope_scaling", "rope_parameters"):
        for kind in kinds(getattr(config, name, None)):
            if kind in ("longrope", "dynamic"):
                return kind
    return None


def _note_once(trainer, message):
    notes = getattr(trainer, "_unsloth_token_budget_notes", None)
    if notes is None:
        notes = set()
        trainer._unsloth_token_budget_notes = notes
    if message not in notes:
        notes.add(message)
        print(message)


def _padded_batches(trainer):
    """True when the SFT collator pads rows: neither padding-free nor packing is on."""
    args = getattr(trainer, "args", None)
    return (
        hasattr(args, "padding_free")
        and not args.padding_free
        and not getattr(args, "packing", False)
    )


def _balance_dataloader(trainer, dataloader):
    if _exactness_blocker(trainer) is not None:
        return False
    # A padded micro-batch costs rows x its longest row: equal token sums put a long row in every
    # micro-batch, which adds padding without lowering the peak.
    if _padded_batches(trainer):
        return False
    # accelerate's DataLoaderDispatcher deals successive inner batches to ranks behind a plain BatchSampler.
    if any(c.__name__ == "DataLoaderDispatcher" for c in type(dataloader).__mro__):
        if int(getattr(getattr(dataloader, "state", None), "num_processes", 1)) > 1:
            return False
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    world_size = 1
    keep_first_window = False
    if hasattr(batch_sampler, "num_processes") and hasattr(batch_sampler, "batch_sampler"):
        # accelerate's BatchSamplerShard with split_batches False deals whole batches round robin.
        if getattr(batch_sampler, "split_batches", False):
            return False
        world_size = int(batch_sampler.num_processes)
        keep_first_window = not getattr(batch_sampler, "drop_last", False) and getattr(
            batch_sampler, "even_batches", True
        )
        batch_sampler = batch_sampler.batch_sampler
    if not isinstance(batch_sampler, torch.utils.data.BatchSampler):
        return False
    sampler = batch_sampler.sampler
    if isinstance(sampler, TokenBalancedSampler):
        return True
    if not isinstance(
        sampler, (torch.utils.data.RandomSampler, torch.utils.data.SequentialSampler)
    ):
        # group_by_length / batch_rebalance already pick their own order.
        return False
    lengths = _dataset_lengths(getattr(dataloader, "dataset", None))
    if lengths is None:
        return False
    if keep_first_window and len(sampler) % (world_size * batch_sampler.batch_size) == 0:
        keep_first_window = False
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
        keep_first_window,
    )
    return True


def _same_count_on_every_rank(count, accelerator):
    if accelerator is None or getattr(accelerator, "num_processes", 1) <= 1:
        return True
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return True
    bounds = torch.tensor([count, -count], dtype = torch.int64, device = accelerator.device)
    dist.all_reduce(bounds, op = dist.ReduceOp.MAX)
    return int(bounds[0]) == -int(bounds[1])


def _label_count(chunk):
    """Tokens TRL scores for token accuracy: shifted labels that are not -100."""
    labels = chunk.get("shift_labels")
    if labels is None:
        labels = chunk.get("labels")
        if labels is None:
            return 0
        labels = labels[..., 1:]
    return int((labels != -100).sum())


def _entropy_count(trainer, chunk):
    """Tokens TRL averages entropy over: the label mask on fused-head TRL, else the attention mask
    (plus prompt-tuning virtual tokens) or every position when padding-free."""
    style = getattr(trainer, "_unsloth_token_budget_entropy_style", None)
    if style is None:
        try:
            source = inspect.getsource(type(trainer).compute_loss)
        except Exception:
            source = ""
        style = "labels" if "label_mask" in source else "attention"
        trainer._unsloth_token_budget_entropy_style = style
    if style == "labels":
        return _label_count(chunk)
    if "attention_mask" in chunk:
        mask = chunk["attention_mask"]
        virtual = int(getattr(trainer, "num_virtual_tokens", 0) or 0)
        return int(mask.sum()) + virtual * mask.shape[0]
    if "position_ids" in chunk:
        return int(chunk["position_ids"].numel())
    return 0


# Per-forward ratio metrics TRL appends once per compute_loss call and later averages per entry.
_RATIO_METRICS = {
    "mean_token_accuracy": lambda trainer, chunk: _label_count(chunk),
    "entropy": _entropy_count,
}


def _metric_lengths(trainer):
    metrics = getattr(trainer, "_metrics", None)
    metrics = metrics.get("train") if isinstance(metrics, Mapping) else None
    if not isinstance(metrics, Mapping):
        return None, None
    lengths = {
        key: len(metrics[key])
        for key in _RATIO_METRICS
        if key in metrics and isinstance(metrics[key], list)
    }
    return metrics, lengths


def _merge_chunk_metrics(trainer, metrics, lengths, chunks, accelerator):
    """Fold the per-chunk ratio entries back into one entry for the micro-batch, weighting each chunk by
    the tokens TRL divided by, so the logged value matches an unsplit step."""
    keys = [key for key in _RATIO_METRICS if key in metrics and isinstance(metrics[key], list)]
    keys = [key for key in keys if len(metrics[key]) - lengths.get(key, 0) == len(chunks)]
    if not keys:
        return
    weights = torch.tensor(
        [[_RATIO_METRICS[key](trainer, chunk) for chunk in chunks] for key in keys],
        dtype = torch.float64,
    )
    if accelerator is not None and getattr(accelerator, "num_processes", 1) > 1:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            # TRL gathers the counts of every rank before dividing.
            weights = weights.to(accelerator.device)
            dist.all_reduce(weights)
            weights = weights.cpu()
    for key, row in zip(keys, weights.tolist()):
        start = lengths.get(key, 0)
        values = metrics[key][start:]
        total = sum(row)
        merged = sum(v * w for v, w in zip(values, row)) / total if total > 0 else 0.0
        metrics[key][start:] = [merged]


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
                reason = _exactness_blocker(trainer)
                if reason is None and _padded_batches(trainer):
                    reason = "batches are padded, so equal token sums would only add padding"
                reason = reason or "unsupported sampler or dataset"
                _note_once(trainer, f"Unsloth: UNSLOTH_TOKEN_BALANCE ignored: {reason}.")
            return dataloader

        trainer.get_train_dataloader = get_train_dataloader

    if budget > 0:
        original_training_step = trainer.training_step
        trainer._unsloth_token_budget_splits = 0

        def training_step(
            model,
            inputs,
            num_items_in_batch = None,
            *args,
            **kwargs,
        ):
            accelerator = getattr(trainer, "accelerator", None)
            distributed_type = str(getattr(accelerator, "distributed_type", "NO")).upper()
            # Only None (loss would be a per-chunk mean), plain or DDP: DeepSpeed and FSDP count micro-steps themselves.
            if num_items_in_batch is None or not (
                distributed_type.endswith("NO") or distributed_type.endswith("MULTI_GPU")
            ):
                return original_training_step(model, inputs, num_items_in_batch, *args, **kwargs)
            reason = _exactness_blocker(trainer)
            if reason is not None:
                _note_once(trainer, f"Unsloth: UNSLOTH_TOKEN_BUDGET not splitting: {reason}.")
                return original_training_step(model, inputs, num_items_in_batch, *args, **kwargs)
            chunks = split_batch_by_token_budget(inputs, budget)
            if not _same_count_on_every_rank(len(chunks), accelerator):
                # Each chunk runs compute_loss (TRL gathers metrics there) and its own backward:
                # ranks with different counts would issue mismatched collectives and hang.
                _note_once(
                    trainer,
                    "Unsloth: UNSLOTH_TOKEN_BUDGET not splitting steps whose ranks need different chunk counts.",
                )
                return original_training_step(model, inputs, num_items_in_batch, *args, **kwargs)
            if len(chunks) == 1:
                return original_training_step(model, inputs, num_items_in_batch, *args, **kwargs)
            trainer._unsloth_token_budget_splits += 1
            metrics, lengths = _metric_lengths(trainer)
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
            if metrics is not None:
                _merge_chunk_metrics(trainer, metrics, lengths, chunks, accelerator)
            return total

        trainer.training_step = training_step
