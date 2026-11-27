# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Token-aware SFT micro-batches: UNSLOTH_TOKEN_BALANCE=1 keeps num_items_in_batch unchanged;
UNSLOTH_TOKEN_BUDGET passes sum to the unsplit loss.

Under torchrun (WORLD_SIZE > 1) balancing across ranks is the default; UNSLOTH_TOKEN_BALANCE=0 turns it off.

UNSLOTH_TOKEN_BALANCE=labels also regroups rows across UNSLOTH_TOKEN_BALANCE_WINDOW (default 8)
consecutive optimizer steps so every step carries a similar number of supervised tokens (shifted
labels != -100). That changes which rows share a step, so it changes the data order: opt-in only."""

from __future__ import annotations

import bisect
import contextlib
import heapq
import inspect
import itertools
import os
from collections.abc import Mapping

import torch

__all__ = [
    "karmarkar_karp",
    "equal_size_partition",
    "balance_step_indices",
    "balance_label_steps",
    "TokenBalancedSampler",
    "split_batch_by_token_budget",
    "install_token_budget",
]

_TOKEN_BALANCE_ENV = "UNSLOTH_TOKEN_BALANCE"
_TOKEN_BUDGET_ENV = "UNSLOTH_TOKEN_BUDGET"
_TOKEN_BALANCE_WINDOW_ENV = "UNSLOTH_TOKEN_BALANCE_WINDOW"
_DEFAULT_LABEL_WINDOW = 8


def karmarkar_karp(
    values,
    k,
    equal_size = False,
):
    """Split indices of `values` into `k` groups with near-equal sums (largest differencing method).

    With `equal_size`, `len(values)` must be a multiple of `k` and every group gets the same count;
    each group then takes exactly one of every `k` consecutive values in sorted order, which is what
    ranks stepping in lockstep need (see `balance_step_indices`), and groups come heaviest first.
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


def _longest_first(values, k):
    """Equal-size LPT: rows longest first, each to the lightest group that still has room."""
    size = len(values) // k
    groups = [[] for _ in range(k)]
    heap = [(0, j) for j in range(k)]
    for i in sorted(range(len(values)), key = lambda i: (-values[i], i)):
        total, j = heapq.heappop(heap)
        groups[j].append(i)
        if len(groups[j]) < size:
            heapq.heappush(heap, (total + values[i], j))
    return groups


def _swap_refine(values, groups, max_swaps):
    """1-for-1 swaps between the heaviest group and the lightest one that has a helpful swap, taking
    the swap that best halves their gap. Group sizes never change, the heaviest sum never grows and
    the sum of squares strictly falls, so it stops; `max_swaps` bounds the cost."""
    groups = [list(g) for g in groups]
    sums = [sum(values[i] for i in g) for g in groups]
    k = len(groups)
    for _ in range(max_swaps):
        hi = min(range(k), key = lambda j: (-sums[j], j))
        best = None
        for lo in sorted(range(k), key = lambda j: (sums[j], j)):
            gap = sums[hi] - sums[lo]
            if gap <= 0:
                break
            keyed = sorted((values[b], p) for p, b in enumerate(groups[lo]))
            keys = [v for v, _ in keyed]
            for a_pos, a in enumerate(groups[hi]):
                # The best partner b has values[b] nearest values[a] - gap / 2.
                q = bisect.bisect_left(keys, values[a] - gap / 2)
                for c in (q - 1, q):
                    if 0 <= c < len(keys):
                        d = values[a] - keys[c]
                        if 0 < d < gap:
                            score = abs(gap - 2 * d)
                            if best is None or score < best[0]:
                                best = (score, a_pos, keyed[c][1], lo, d)
            if best is not None:
                break
        if best is None:
            break
        _, a_pos, b_pos, lo, d = best
        groups[hi][a_pos], groups[lo][b_pos] = groups[lo][b_pos], groups[hi][a_pos]
        sums[hi] -= d
        sums[lo] += d
    return [sorted(g) for g in groups]


def equal_size_partition(values, k):
    """`k` equal-count groups of the indices of `values` with the smallest heaviest sum this finds:
    the better (heaviest sum, then spread) of equal-size differencing and a size-capped longest-first
    fill, each improved by 1-for-1 swaps. Equal-size differencing alone leaves the heaviest group
    several % (2 groups) to 20%+ (8 groups) over the mean on long-tailed lengths; this gets within
    about 0.1-3%. Unlike `karmarkar_karp`, a group may take several of the largest values, so it is
    for a total per group (ranks that synchronise once per step), not for ranks in lockstep.
    Deterministic: ties break on index; groups come heaviest first. At most 2 * len(values) swaps."""
    differenced = karmarkar_karp(values, k, equal_size = True)
    k = len(differenced)
    if k < 2 or len(values) <= k:
        return differenced

    def key(groups):
        sums = [sum(values[i] for i in g) for g in groups]
        return (max(sums), max(sums) - min(sums))

    max_swaps = 2 * len(values)
    best = min(
        _swap_refine(values, differenced, max_swaps),
        _swap_refine(values, _longest_first(values, k), max_swaps),
        key = key,
    )
    return sorted(best, key = lambda g: (-sum(values[i] for i in g), g))


def _more_even(regrouped, given, weights, size):
    """The partitioners are heuristics: True unless `given` already has the smaller spread of
    per-group sums over consecutive `size` chunks, so a regroup never ends up less even."""

    def spread(order):
        sums = [sum(weights[i] for i in order[s : s + size]) for s in range(0, len(order), size)]
        return max(sums) - min(sums)

    return spread(regrouped) <= spread(given)


def balance_step_indices(
    indices,
    lengths,
    batch_size,
    grad_accum,
    world_size = 1,
    cross_rank = True,
    lockstep = True,
):
    """Reorder one optimizer step's sampler output (`world_size * grad_accum * batch_size` indices).

    The layout follows accelerate's BatchSamplerShard: consecutive `batch_size` chunks are global
    batches, global batch b goes to rank b % world_size, and rank r's g-th batch of the step is
    global batch g * world_size + r. With `cross_rank` rows move between ranks to equalize per-rank
    tokens; without it each rank keeps its own rows and only its accumulation slots are balanced.

    `lockstep`: ranks wait for each other on every micro-batch (FSDP, pipeline / tensor / context
    parallel), so a micro-step costs its longest slot over ranks. Equal-size differencing gives every
    rank one row of each size tier and slots come heaviest first, which lines similar slots up across
    ranks. False (DDP, which synchronises once per step): only each rank's total matters, which
    `equal_size_partition` evens out far better on long-tailed lengths.
    """
    bs, ga, ws = int(batch_size), int(grad_accum), int(world_size)
    window = bs * ga * ws
    if len(indices) != window:
        raise ValueError(f"expected {window} indices, got {len(indices)}")
    if ga * ws == 1:
        return list(indices)
    rank_rows = [
        [indices[(g * ws + r) * bs + j] for g in range(ga) for j in range(bs)] for r in range(ws)
    ]
    if cross_rank and ws > 1:
        values = [lengths[i] for i in indices]
        if lockstep:
            groups = karmarkar_karp(values, ws, equal_size = True)
        else:
            groups = equal_size_partition(values, ws)
        regrouped = [[indices[p] for p in group] for group in groups]
        if _more_even(sum(regrouped, []), sum(rank_rows, []), lengths, ga * bs):
            rank_rows = regrouped
    out = [None] * window
    for r, rows in enumerate(rank_rows):
        slots = karmarkar_karp([lengths[i] for i in rows], ga, equal_size = True)
        slots = [[rows[p] for p in slot] for slot in slots]
        if not _more_even(sum(slots, []), rows, lengths, bs):
            slots = [rows[g * bs : (g + 1) * bs] for g in range(ga)]
        for g, slot in enumerate(slots):
            base = (g * ws + r) * bs
            out[base : base + bs] = slot
    return out


def balance_label_steps(indices, label_counts, step_size):
    """Regroup `len(indices) // step_size` consecutive optimizer steps so each step's summed
    `label_counts` is near-equal; every step keeps `step_size` rows and the multiset is unchanged.

    Steps are emitted in order of their earliest original row and rows keep their original relative
    order inside a step, so the sampler's shuffle still decides the coarse order."""
    step_size = int(step_size)
    if step_size < 1 or len(indices) % step_size != 0:
        raise ValueError(f"expected a multiple of {step_size} indices, got {len(indices)}")
    steps = len(indices) // step_size
    if steps < 2:
        return list(indices)
    groups = equal_size_partition([label_counts[i] for i in indices], steps)
    groups.sort(key = lambda group: group[0])
    out = [indices[p] for group in groups for p in group]
    return out if _more_even(out, indices, label_counts, step_size) else list(indices)


class TokenBalancedSampler(torch.utils.data.Sampler):
    """Applies `balance_step_indices` to each full step window; a partial tail passes through.

    `keep_first_window` passes the first window through unchanged: accelerate's BatchSamplerShard
    (even_batches) pads an uneven tail with the first `world_size * batch_size` indices it saw.

    With `label_counts`, `label_window` consecutive step windows are first regrouped by
    `balance_label_steps`; `balance_lengths` False then leaves each step's rows in that order.
    `lockstep` is passed to `balance_step_indices`.
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
        label_counts = None,
        label_window = 1,
        balance_lengths = True,
        lockstep = True,
    ):
        self.sampler = sampler
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.world_size = int(world_size)
        self.cross_rank = bool(cross_rank)
        self.keep_first_window = bool(keep_first_window)
        self.label_counts = label_counts
        self.label_window = max(int(label_window), 1) if label_counts is not None else 1
        self.balance_lengths = bool(balance_lengths)
        self.lockstep = bool(lockstep)

    def __len__(self):
        return len(self.sampler)

    def _steps(self, buffer, window):
        if self.label_counts is not None:
            buffer = balance_label_steps(buffer, self.label_counts, window)
        if not self.balance_lengths:
            return buffer
        out = []
        for start in range(0, len(buffer), window):
            out.extend(
                balance_step_indices(
                    buffer[start : start + window],
                    self.lengths,
                    self.batch_size,
                    self.grad_accum,
                    self.world_size,
                    self.cross_rank,
                    self.lockstep,
                )
            )
        return out

    def __iter__(self):
        window = self.batch_size * self.grad_accum * self.world_size
        span = window * self.label_window
        buffer = []
        first = self.keep_first_window
        for index in self.sampler:
            buffer.append(index)
            if len(buffer) == window and first:
                first = False
                yield from buffer
                buffer = []
            elif len(buffer) == span:
                yield from self._steps(buffer, window)
                buffer = []
        # A shorter tail still regroups its full steps; the partial step passes through.
        full = len(buffer) - len(buffer) % window
        if full and not first:
            yield from self._steps(buffer[:full], window)
            buffer = buffer[full:]
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
        # The host tuple the packing collators attach (packing._carry_host_lengths): no device read.
        host = getattr(lengths, "_unsloth_lengths_tuple", None)
        if isinstance(host, tuple) and len(host) == lengths.numel():
            lengths = list(host)
        else:
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
                # Each chunk's forward reads its lengths from this tuple instead of the device.
                chunk[key]._unsloth_lengths_tuple = tuple(group)
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
    """Splits a batch to at most budget tokens: padding-free [1, T] batches between whole sequences,
    padded [B, L] by rows; other shapes are never split."""
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


def _arrow_lengths(dataset):
    """Row lengths of `input_ids` from the Arrow list offsets (no token is read), or None."""
    try:
        import pyarrow.compute as pc

        # A transform may not return the stored column.
        if getattr(dataset, "_format_type", None) == "custom":
            return None
        lengths = pc.list_value_length(dataset.data.column("input_ids"))
        indices = getattr(dataset, "_indices", None)
        if indices is not None:
            lengths = pc.take(lengths, indices.column(0))
        lengths = lengths.to_pylist()
    except Exception:
        return None
    if len(lengths) != len(dataset) or any(length is None for length in lengths):
        return None
    return lengths


def _dataset_lengths(dataset):
    column_names = getattr(dataset, "column_names", None)
    if column_names is None or "input_ids" not in column_names:
        return None
    lengths = _arrow_lengths(dataset)
    if lengths is not None:
        return lengths
    # Batched, so the whole token column is never materialised as Python lists at once.
    if hasattr(dataset, "select_columns") and hasattr(dataset, "iter"):
        lengths = []
        for rows in dataset.select_columns(["input_ids"]).iter(batch_size = 4096):
            lengths.extend(len(ids) for ids in rows["input_ids"])
        return lengths
    return [len(ids) for ids in dataset["input_ids"]]


def _supervised_count(
    labels,
    completion_mask,
    assistant_mask,
    seq_lengths = None,
):
    # Shifted-loss positions surviving the collator masks; packed rows mask each later document's
    # first token (packing._mask_padding_free_boundary_labels); set_format arrays need tolist().
    labels, completion_mask, assistant_mask, seq_lengths = (
        value.tolist() if hasattr(value, "tolist") else value
        for value in (labels, completion_mask, assistant_mask, seq_lengths)
    )
    n = (
        len(labels)
        if labels is not None
        else len(completion_mask if completion_mask is not None else assistant_mask)
    )
    starts = set()
    offset = 0
    for length in seq_lengths if seq_lengths is not None else ():
        starts.add(offset)
        offset += int(length)
    count = 0
    for k in range(1, n):
        if k in starts:
            continue
        if labels is not None and labels[k] == -100:
            continue
        if completion_mask is not None and not completion_mask[k]:
            continue
        if assistant_mask is not None and not assistant_mask[k]:
            continue
        count += 1
    return count


def _dataset_label_counts(dataset, completion_only_loss = True):
    """Supervised tokens per row as the loss sees them: shifted labels != -100, and also inside the
    completion_mask (when the collator applies it) and assistant_masks columns, as TRL's collator
    masks every one that is present. None when the dataset has none of these columns."""
    column_names = getattr(dataset, "column_names", None)
    if column_names is None:
        return None
    columns = [
        column
        for column in ("labels", "completion_mask", "assistant_masks")
        if column in column_names and (column != "completion_mask" or completion_only_loss)
    ]
    if not columns:
        return None

    def counts_of(rows):
        n = len(rows[columns[0]])
        return [
            _supervised_count(
                rows["labels"][i] if "labels" in rows else None,
                rows["completion_mask"][i] if "completion_mask" in rows else None,
                rows["assistant_masks"][i] if "assistant_masks" in rows else None,
                rows["seq_lengths"][i] if "seq_lengths" in rows else None,
            )
            for i in range(n)
        ]

    read = columns + (["seq_lengths"] if "seq_lengths" in column_names else [])
    if hasattr(dataset, "select_columns") and hasattr(dataset, "iter"):
        counts = []
        selected = dataset.select_columns(read)
        if hasattr(selected, "with_format"):
            selected = selected.with_format(None)  # plain lists: no tensor conversion per row
        for rows in selected.iter(batch_size = 4096):
            counts.extend(counts_of(rows))
        return counts
    return counts_of({column: dataset[column] for column in read})


def _data_parallel_world():
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _balance_mode():
    value = os.environ.get(_TOKEN_BALANCE_ENV, "auto").strip().lower()
    if value in ("", "auto"):
        return "auto" if _data_parallel_world() > 1 else None
    if value in ("1", "true", "yes", "on", "exact", "tokens"):
        return "exact"
    if value in ("labels", "label"):
        return "labels"
    return None


def _label_window():
    window = _env_int(_TOKEN_BALANCE_WINDOW_ENV)
    return window if window > 0 else _DEFAULT_LABEL_WINDOW


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


def _exactness_blocker(trainer, step_loss = True):
    """Why regrouping / splitting this trainer's rows would change the objective, or None. Shared with
    pipeline parallel GA folding (models/pp_env.py), which passes step_loss = False: its loss always
    divides by the window's num_items_in_batch."""
    args = getattr(trainer, "args", None)
    optim = str(getattr(args, "optim", "")).lower()
    # LOMO / AdaLomo and the *_layerwise optimizers update weights inside every backward.
    if "lomo" in optim or "layerwise" in optim:
        return "this optimizer updates the weights on every backward"
    # NEFTune scales its noise by the forward's sequence length, which a split shortens.
    if getattr(args, "neftune_noise_alpha", None):
        return "NEFTune noise depends on the sequence length of each forward"
    if step_loss and not _loss_uses_step_count(trainer):
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


def _balance_dataloader(
    trainer,
    dataloader,
    mode = "exact",
):
    labels_mode = mode == "labels"
    auto = mode == "auto"
    if auto and (
        getattr(trainer, "is_deepspeed_enabled", False)
        or getattr(trainer, "is_fsdp_enabled", False)
    ):
        # DeepSpeed / accelerate FSDP stay opt-in.
        return False
    # The labels mode regroups rows across steps (a data order change), so it does not need the
    # within-step exactness guards; it then length-balances inside each step only when they pass.
    balance_lengths = _exactness_blocker(trainer) is None and not _padded_batches(trainer)
    if not labels_mode and not balance_lengths:
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
    dataset = getattr(dataloader, "dataset", None)
    lengths = _dataset_lengths(dataset)
    if lengths is None:
        return False
    label_counts = None
    if labels_mode:
        # TRL's collator applies completion_mask only with completion_only_loss; others are assumed to apply it.
        collator = getattr(trainer, "data_collator", None)
        label_counts = _dataset_label_counts(
            dataset,
            completion_only_loss = getattr(collator, "completion_only_loss", True) is not False,
        )
        if label_counts is None or len(label_counts) != len(lengths):
            return False
    if keep_first_window and len(sampler) % (world_size * batch_sampler.batch_size) == 0:
        keep_first_window = False
    args = trainer.args
    # Moving rows between ranks keeps the loss exact only when the token count is global.
    cross_rank = world_size > 1 and bool(getattr(args, "average_tokens_across_devices", False))
    if auto and not cross_rank:
        return False
    batch_sampler.sampler = TokenBalancedSampler(
        sampler,
        lengths,
        batch_sampler.batch_size,
        args.gradient_accumulation_steps,
        world_size,
        cross_rank,
        keep_first_window,
        label_counts = label_counts,
        label_window = _label_window() if labels_mode else 1,
        balance_lengths = balance_lengths,
        lockstep = _lockstep_ranks(trainer),
    )
    return True


def _lockstep_ranks(trainer):
    """False only for plain DDP, whose ranks synchronise once per optimizer step. FSDP (native or
    accelerate), DeepSpeed and tensor / pipeline / context / expert parallel ranks also wait for
    each other inside every micro-batch, so they keep the lockstep layout."""
    if getattr(trainer, "is_deepspeed_enabled", False) or getattr(
        trainer, "is_fsdp_enabled", False
    ):
        return True
    if str(os.environ.get("UNSLOTH_FSDP2", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
        "native",
    ):
        return True
    for name in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE", "UNSLOTH_EP_SIZE"):
        value = str(os.environ.get(name, "")).strip()
        if value and value != "1":
            return True
    return False


def _same_count_on_every_rank(
    count,
    accelerator,
    flag = None,
):
    """True when every rank has the same count. With flag, returns (same, flag on every rank) from
    the same collective, so a later collective guarded by flag runs on every rank or on none."""
    if accelerator is None or getattr(accelerator, "num_processes", 1) <= 1:
        return True if flag is None else (True, bool(flag))
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return True if flag is None else (True, bool(flag))
    values = [count, -count] + ([] if flag is None else [0 if flag else 1])
    bounds = torch.tensor(values, dtype = torch.int64, device = accelerator.device)
    dist.all_reduce(bounds, op = dist.ReduceOp.MAX)
    same = int(bounds[0]) == -int(bounds[1])
    return same if flag is None else (same, int(bounds[2]) == 0)


def _label_count(chunk):
    """Tokens TRL scores for token accuracy: shifted labels that are not -100."""
    labels = chunk.get("shift_labels")
    if labels is None:
        labels = chunk.get("labels")
        if labels is None:
            return 0
        labels = labels[..., 1:]
    return int((labels != -100).sum())


def _metric_style(trainer):
    """How the trainer's TRL compute_loss averages entropy: "labels" (TRL >= 1.3, valid labels) or
    "attention" (older TRL, attention mask). None when no class in the MRO computes TRL's metrics
    or the entropy code is not recognised."""
    style = getattr(trainer, "_unsloth_token_budget_metric_style", False)
    if style is not False:
        return style
    style = None
    # Subclasses that only delegate to super() do not hold the metric code: use the class that does.
    for cls in type(trainer).__mro__:
        function = cls.__dict__.get("compute_loss")
        if function is None:
            continue
        try:
            source = inspect.getsource(function)
        except Exception:
            break
        if "mean_token_accuracy" in source:
            # TRL >= 1.3 sums entropy over valid labels (entropy_sum / label_mask); older TRL divides by
            # attention_mask.sum(). Both or neither is unknown, so its entries are left unfolded.
            labels = "label_mask" in source or "entropy_sum" in source
            attention = "attention_mask.sum()" in source
            if labels != attention:
                style = "labels" if labels else "attention"
            break
    trainer._unsloth_token_budget_metric_style = style
    return style


def _entropy_count(trainer, chunk):
    """Tokens TRL averages entropy over: the label mask on fused-head TRL, else the attention mask or
    every position when padding-free."""
    if _metric_style(trainer) == "labels":
        return _label_count(chunk)
    if "attention_mask" in chunk:
        return int(chunk["attention_mask"].sum())
    if "position_ids" in chunk:
        return int(chunk["position_ids"].numel())
    return 0


# Per-forward ratio metrics TRL appends once per compute_loss call and later averages per entry.
_RATIO_METRICS = {
    "mean_token_accuracy": lambda trainer, chunk: _label_count(chunk),
    "entropy": _entropy_count,
}


def _is_prompt_learning(trainer, model):
    if int(getattr(trainer, "num_virtual_tokens", 0) or 0) > 0:
        return True
    for candidate in (model, getattr(trainer, "model", None)):
        config = getattr(candidate, "active_peft_config", None)
        if getattr(config, "is_prompt_learning", False):
            return True
    return False


def _is_multi_process(accelerator):
    return accelerator is not None and int(getattr(accelerator, "num_processes", 1) or 1) > 1


def _can_merge_metrics(trainer, model, chunks, accelerator):
    """Fold per-chunk entries only when token weighting provably reproduces TRL's unsplit value:
    with several processes only under an initialized process group and without Liger (whose older
    TRL paths average per-rank ratios), no prompt-learning PEFT (virtual token logits are counted
    differently per version), one label source, and known TRL metric code."""
    if _is_multi_process(accelerator):
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return False
        if getattr(getattr(trainer, "args", None), "use_liger_kernel", False):
            return False
    if _is_prompt_learning(trainer, model):
        return False
    if any("shift_labels" in chunk and "labels" in chunk for chunk in chunks):
        return False
    return _metric_style(trainer) is not None


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


def _global_chunk_weights(trainer, chunks, accelerator):
    """Multi-process weights sum token counts over ranks; older attention-style TRL averages
    per-rank entropy ratios, which cannot be refolded, so entropy stays unfolded."""
    weights = {
        key: [count(trainer, chunk) for chunk in chunks] for key, count in _RATIO_METRICS.items()
    }
    if not _is_multi_process(accelerator):
        return weights
    import torch.distributed as dist

    # Same shape on every rank (chunk counts already agree), so the collective always matches.
    local = torch.tensor(list(weights.values()), dtype = torch.int64, device = accelerator.device)
    dist.all_reduce(local, op = dist.ReduceOp.SUM)
    weights = {key: [int(w) for w in row] for key, row in zip(weights, local.tolist())}
    if _metric_style(trainer) != "labels":
        weights.pop("entropy", None)
    return weights


def _merge_chunk_metrics(
    trainer,
    metrics,
    lengths,
    chunks,
    weights = None,
):
    """Fold the per-chunk ratio entries back into one entry for the micro-batch, weighting each chunk by
    the tokens TRL divided by, so the logged value matches an unsplit step."""
    if weights is None:
        weights = {
            key: [count(trainer, chunk) for chunk in chunks]
            for key, count in _RATIO_METRICS.items()
        }
    keys = [key for key in weights if key in metrics and isinstance(metrics[key], list)]
    keys = [key for key in keys if len(metrics[key]) - lengths.get(key, 0) == len(chunks)]
    for key in keys:
        row = weights[key]
        start = lengths.get(key, 0)
        values = metrics[key][start:]
        total = sum(row)
        merged = sum(v * w for v, w in zip(values, row)) / total if total > 0 else 0.0
        metrics[key][start:] = [merged]


def install_token_budget(trainer):
    """Install whichever of UNSLOTH_TOKEN_BALANCE / UNSLOTH_TOKEN_BUDGET is set on `trainer`."""
    balance = _balance_mode()
    budget = _env_int(_TOKEN_BUDGET_ENV)
    if (not balance and budget <= 0) or getattr(trainer, "_unsloth_token_budget_installed", False):
        return
    trainer._unsloth_token_budget_installed = True

    if balance:
        original_get_train_dataloader = trainer.get_train_dataloader

        def get_train_dataloader(*args, **kwargs):
            dataloader = original_get_train_dataloader(*args, **kwargs)
            trainer._unsloth_token_balance_active = _balance_dataloader(
                trainer, dataloader, balance
            )
            if trainer._unsloth_token_balance_active and balance == "labels":
                _note_once(
                    trainer,
                    "Unsloth: UNSLOTH_TOKEN_BALANCE=labels regrouping rows over "
                    f"{_label_window()} optimizer steps to even out supervised tokens per step.",
                )
            elif trainer._unsloth_token_balance_active and balance == "auto":
                _note_once(
                    trainer,
                    "Unsloth: data-parallel ranks get equal token counts per optimizer step "
                    "(same rows per step; UNSLOTH_TOKEN_BALANCE=0 turns it off).",
                )
            elif balance == "auto":
                pass
            elif balance == "labels":
                _note_once(
                    trainer,
                    "Unsloth: UNSLOTH_TOKEN_BALANCE=labels ignored: needs a labels, completion_mask "
                    "or assistant_masks column and a plain random or sequential sampler.",
                )
            elif not trainer._unsloth_token_balance_active:
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
            # Folding metrics across ranks adds a collective, so every rank must agree to fold.
            metrics, lengths = (
                _metric_lengths(trainer)
                if len(chunks) > 1 and _can_merge_metrics(trainer, model, chunks, accelerator)
                else (None, None)
            )
            same_count, merge = _same_count_on_every_rank(
                len(chunks), accelerator, flag = metrics is not None
            )
            if not merge:
                metrics, lengths = None, None
            if not same_count:
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
            total = None
            try:
                for i, chunk in enumerate(chunks):
                    partial = i < len(chunks) - 1
                    # sync_gradients stays True per chunk; expert parallel reads this to sum replicas once per step.
                    trainer._unsloth_token_budget_partial = partial
                    context = (
                        accelerator.no_sync(model)
                        if partial and accelerator is not None
                        else contextlib.nullcontext()
                    )
                    with context:
                        loss = original_training_step(
                            model, chunk, num_items_in_batch, *args, **kwargs
                        )
                    total = loss if total is None else total + loss
            finally:
                trainer._unsloth_token_budget_partial = False
            if metrics is not None:
                weights = _global_chunk_weights(trainer, chunks, accelerator)
                _merge_chunk_metrics(trainer, metrics, lengths, chunks, weights)
            return total

        trainer.training_step = training_step
