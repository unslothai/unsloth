# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Chunked contrastive loss (InfoNCE / NTXent) that avoids materializing the
full similarity matrix.  Drop-in replacement for
`sentence_transformers.losses.MultipleNegativesRankingLoss`.

Supports non-square matrices (B_a != B_b) for multi-positive setups.
"""

import os
from collections import deque

import torch
import torch.nn.functional as F
from .utils import torch_amp_custom_fwd, torch_amp_custom_bwd


_PADDED_SEQUENCE_FEATURES = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "special_tokens_mask",
}
_BATCH_INDEPENDENT_PIPELINE_MODULES = {
    "Transformer",
    "Pooling",
    "Normalize",
    "Dense",
    "Dropout",
    "LayerNorm",
    "GuidedProjection",
    "GuidedProjectionPooling",
}
_DECODER_EMBEDDING_MODEL_TYPES = {
    "qwen2",
    "qwen3",
    "llama",
    "mistral",
    "gemma",
    "gemma2",
    "gemma3_text",
}


def _contrastive_chunk_size(candidate_count):
    """Return a safe positive chunk size for the streaming loss loops."""
    try:
        configured = int(os.environ.get("UNSLOTH_CONTRASTIVE_CHUNK_SIZE", 1024))
    except (TypeError, ValueError):
        configured = 1024
    return min(max(configured, 1), candidate_count)


def _has_standard_batch_independent_pipeline(model):
    modules = getattr(model, "_modules", None)
    if not modules:
        return False
    if any(
        module.__class__.__name__ not in _BATCH_INDEPENDENT_PIPELINE_MODULES
        for module in modules.values()
    ):
        return False
    # BatchNorm and SyncBatchNorm deliberately couple rows while training.  The
    # top-level allowlist catches custom SentenceTransformer modules; this nested
    # check also catches a BatchNorm hidden inside an otherwise familiar wrapper.
    batch_norm = torch.nn.modules.batchnorm._BatchNorm
    return not any(isinstance(module, batch_norm) for module in model.modules())


def _has_compiled_encoder(model):
    for module in getattr(model, "_modules", {}).values():
        inner = getattr(module, "auto_model", None)
        if hasattr(inner, "_orig_mod"):
            return True
    return False


def _right_padded_attention_lengths(attention_masks):
    """Return row lengths for binary right-padded masks with one device sync.

    Calling ``.item()`` for every validation predicate serialized the GPU up to
    nine times per MNRL step.  Accumulate all predicates on-device and transfer
    the small length/validity vector once instead.
    """
    if not attention_masks:
        return None
    first = attention_masks[0]
    if not torch.is_tensor(first):
        return None

    length_tensors = []
    valid = torch.ones((), device = first.device, dtype = torch.bool)
    sizes = []
    for attention_mask in attention_masks:
        if (
            not torch.is_tensor(attention_mask)
            or attention_mask.device != first.device
            or attention_mask.ndim != 2
            or attention_mask.shape[0] <= 0
            or attention_mask.shape[1] <= 0
        ):
            return None
        lengths = attention_mask.to(torch.int64).sum(dim = 1)
        present = attention_mask != 0
        valid = valid & (((attention_mask == 0) | (attention_mask == 1)).all())
        valid = valid & (lengths > 0).all()
        valid = valid & ((~present[:, 1:]) | present[:, :-1]).all()
        length_tensors.append(lengths)
        sizes.append(lengths.numel())

    payload = torch.cat(
        [*length_tensors, valid.to(torch.int64).reshape(1)],
        dim = 0,
    ).tolist()
    if not payload[-1]:
        return None

    result = []
    offset = 0
    for size in sizes:
        result.append([int(length) for length in payload[offset : offset + size]])
        offset += size
    return result


def _minimum_cost_length_buckets(
    sorted_lengths,
    original_token_slots,
    max_padding_multiplier = 1.25,
):
    """Find the fewest optimal contiguous buckets within the padding budget.

    For a sorted segment ``[start, end)``, its padded cost is
    ``(end - start) * sorted_lengths[end - 1]``.  Dynamic programming finds the
    minimum total cost for each bucket count.  A monotone convex hull reduces
    each DP layer from quadratic to linear time.
    """
    row_count = len(sorted_lengths)
    if row_count == 0:
        return []

    infinity = float("inf")
    previous_costs = [0] + [infinity] * row_count
    parents = [None]

    def line_value(line, x):
        slope, intercept, _index = line
        return slope * x + intercept

    def middle_line_is_redundant(first, middle, last):
        first_slope, first_intercept, _ = first
        middle_slope, middle_intercept, _ = middle
        last_slope, last_intercept, _ = last
        return (middle_intercept - first_intercept) * (middle_slope - last_slope) >= (
            last_intercept - middle_intercept
        ) * (first_slope - middle_slope)

    for bucket_count in range(1, row_count + 1):
        current_costs = [infinity] * (row_count + 1)
        current_parents = [-1] * (row_count + 1)
        hull = deque()

        for end in range(bucket_count, row_count + 1):
            start = end - 1
            if previous_costs[start] != infinity:
                line = (-start, previous_costs[start], start)
                while len(hull) >= 2 and middle_line_is_redundant(hull[-2], hull[-1], line):
                    hull.pop()
                hull.append(line)

            length = sorted_lengths[end - 1]
            while len(hull) >= 2 and line_value(hull[0], length) >= line_value(hull[1], length):
                hull.popleft()
            best_line = hull[0]
            current_costs[end] = end * length + line_value(best_line, length)
            current_parents[end] = best_line[2]

        parents.append(current_parents)
        if current_costs[row_count] <= original_token_slots * max_padding_multiplier:
            buckets = []
            end = row_count
            for level in range(bucket_count, 0, -1):
                start = parents[level][end]
                buckets.append((start, end))
                end = start
            buckets.reverse()
            return buckets
        previous_costs = current_costs

    # Singleton buckets cost exactly the number of attended tokens, which is
    # never greater than the original separately padded columns.  Keep this
    # defensive fallback in case malformed metadata slips through.
    return [(index, index + 1) for index in range(row_count)]


def _max_combined_padding_multiplier(model):
    """Use a wider one-call budget for calibrated causal embedding models."""
    for module in getattr(model, "_modules", {}).values():
        inner = getattr(module, "auto_model", None)
        if inner is None:
            continue
        while hasattr(inner, "base_model") and getattr(inner, "base_model") is not inner:
            inner = inner.base_model
        config = getattr(inner, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        if model_type in _DECODER_EMBEDDING_MODEL_TYPES:
            # Decoder LoRA kernels and attention have substantially higher
            # per-forward overhead. One combined call stayed faster with up to
            # 2x padded token slots; above that, retain adaptive bucketing.
            return 2.0
        break
    return 1.25


class _ReusableIndex:
    """A view into a lazily materialized device index buffer."""

    __slots__ = ("_pool", "_group")

    def __init__(self, pool, group):
        self._pool = pool
        self._group = group

    def on(self, device):
        return self._pool.on(self._group, device)


class _ReusableIndexPool:
    """Pack all row indices into one allocation per device for the whole step."""

    __slots__ = ("_values", "_spans", "_by_device")

    def __init__(self, groups):
        values = []
        spans = []
        for group in groups:
            start = len(values)
            values.extend(group)
            spans.append((start, len(values)))
        self._values = values
        self._spans = spans
        self._by_device = {}

    def index(self, group):
        return _ReusableIndex(self, group)

    def on(self, group, device):
        buffer = self._by_device.get(device)
        if buffer is None:
            buffer = torch.tensor(self._values, device = device, dtype = torch.long)
            self._by_device[device] = buffer
        start, end = self._spans[group]
        return buffer.narrow(0, start, end - start)


def _prepare_bucket_row_plan(bucket_flat_indices, batch_size, column_count):
    """Precompute bucket row gathers and stable-order permutations."""
    processing_order = []
    bucket_column_rows = []
    bucket_reorders = []

    for flat_indices in bucket_flat_indices:
        processing_order.extend(flat_indices)
        rows_by_column = [[] for _ in range(column_count)]
        for flat_index in flat_indices:
            column, row = divmod(flat_index, batch_size)
            rows_by_column[column].append(row)
        bucket_column_rows.append(rows_by_column)

        grouped_flat_indices = []
        for column, rows in enumerate(rows_by_column):
            grouped_flat_indices.extend(column * batch_size + row for row in rows)

        if grouped_flat_indices == flat_indices:
            bucket_reorders.append(None)
        else:
            grouped_positions = {
                flat_index: position for position, flat_index in enumerate(grouped_flat_indices)
            }
            bucket_reorders.append([grouped_positions[flat_index] for flat_index in flat_indices])

    identity_order = all(
        flat_index == position for position, flat_index in enumerate(processing_order)
    )
    index_groups = []

    def add_index(values):
        group = len(index_groups)
        index_groups.append(values)
        return group

    processing_group = None if identity_order else add_index(processing_order)
    bucket_column_groups = tuple(
        tuple(None if not rows else add_index(rows) for rows in rows_by_column)
        for rows_by_column in bucket_column_rows
    )
    reorder_groups = tuple(
        None if reorder is None else add_index(reorder) for reorder in bucket_reorders
    )
    index_pool = _ReusableIndexPool(index_groups)
    return {
        "processing_order": processing_order,
        "processing_index": (
            None if processing_group is None else index_pool.index(processing_group)
        ),
        "bucket_column_indices": tuple(
            tuple(None if group is None else index_pool.index(group) for group in groups)
            for groups in bucket_column_groups
        ),
        "bucket_reorders": tuple(
            None if group is None else index_pool.index(group) for group in reorder_groups
        ),
        "bucket_sizes": tuple(len(indices) for indices in bucket_flat_indices),
        "index_pool": index_pool,
    }


def _gather_unpadded_bucket_rows(values, row_plan):
    """Reorder a regular batch feature once, then return per-bucket views."""
    flattened = torch.cat(values, dim = 0)
    processing_index = row_plan["processing_index"]
    if processing_index is not None:
        flattened = flattened.index_select(0, processing_index.on(flattened.device))
    return torch.split(flattened, row_plan["bucket_sizes"], dim = 0)


def _gather_single_bucket_rows(
    values,
    sequence_length = None,
    pad_value = 0,
):
    """Concatenate a column-major bucket without indices or row reordering."""
    if sequence_length is None:
        return torch.cat(values, dim = 0)

    pieces = []
    for value in values:
        piece = value[:, :sequence_length]
        if piece.shape[1] < sequence_length:
            piece = F.pad(
                piece,
                (0, sequence_length - piece.shape[1]),
                value = pad_value,
            )
        pieces.append(piece)
    return torch.cat(pieces, dim = 0)


def _gather_padded_bucket_rows(values, row_plan, bucket_lengths, pad_value):
    """Gather bucket-local rows with indices shared by every padded feature."""
    buckets = []
    for bucket_index, sequence_length in enumerate(bucket_lengths):
        pieces = []
        for value, index in zip(values, row_plan["bucket_column_indices"][bucket_index]):
            if index is None:
                continue
            piece = value.index_select(0, index.on(value.device))
            piece = piece[:, :sequence_length]
            if piece.shape[1] < sequence_length:
                piece = F.pad(
                    piece,
                    (0, sequence_length - piece.shape[1]),
                    value = pad_value,
                )
            pieces.append(piece)

        gathered = torch.cat(pieces, dim = 0)
        reorder = row_plan["bucket_reorders"][bucket_index]
        if reorder is not None:
            gathered = gathered.index_select(0, reorder.on(gathered.device))
        buckets.append(gathered)
    return buckets


def _bucketed_sentence_features(model, sentence_features):
    """Build the smallest safe set of length-bucketed encoder batches."""
    if len(sentence_features) < 2:
        return None
    if not _has_standard_batch_independent_pipeline(model):
        return None
    # Compiled encoders are already launch-amortized, and changing their batch
    # shape from B to N*B regressed real training throughput in calibration.
    if _has_compiled_encoder(model):
        return None
    if not all(isinstance(features, dict) for features in sentence_features):
        return None
    if not all(
        "input_ids" in features and "attention_mask" in features for features in sentence_features
    ):
        return None

    keys = tuple(sentence_features[0])
    if any(set(features) != set(keys) for features in sentence_features[1:]):
        return None

    batch_sizes = []
    sequence_lengths = []
    attention_masks = []
    for features in sentence_features:
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]
        if (
            not torch.is_tensor(input_ids)
            or not torch.is_tensor(attention_mask)
            or input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
        ):
            return None
        batch_sizes.append(input_ids.shape[0])
        sequence_lengths.append(input_ids.shape[1])
        attention_masks.append(attention_mask)

    attended_lengths = _right_padded_attention_lengths(attention_masks)
    if attended_lengths is None:
        return None

    if batch_sizes[0] == 0 or any(size != batch_sizes[0] for size in batch_sizes[1:]):
        return None

    pad_token_id = getattr(getattr(model, "tokenizer", None), "pad_token_id", 0)
    pad_token_id = 0 if pad_token_id is None else int(pad_token_id)
    values_by_key = {}

    for key in keys:
        values = [features[key] for features in sentence_features]
        values_by_key[key] = values
        first = values[0]
        if not torch.is_tensor(first):
            if any(value != first for value in values[1:]):
                return None
            continue

        if any(not torch.is_tensor(value) for value in values[1:]):
            return None
        if any(value.device != first.device or value.dtype != first.dtype for value in values[1:]):
            return None
        if any(value.ndim != first.ndim for value in values[1:]):
            return None
        if first.ndim == 0:
            if any(not torch.equal(value, first) for value in values[1:]):
                return None
            continue
        if any(value.shape[0] != size for value, size in zip(values, batch_sizes)):
            return None

        if key == "prompt_length":
            # Pooling consumes prompt_length[0] as a batch-wide value. Sorting
            # rows is only equivalent when every row and column agrees.
            if first.numel() == 0:
                return None
            prompt_length = first.reshape(-1)[0]
            # Validate all columns with one device-to-host decision instead of
            # synchronizing once per feature column.
            flattened = torch.cat([value.reshape(-1) for value in values])
            if not bool((flattened == prompt_length).all().item()):
                return None

        if key in _PADDED_SEQUENCE_FEATURES:
            if any(
                value.ndim != 2 or value.shape[1] != sequence_length
                for value, sequence_length in zip(values, sequence_lengths)
            ):
                return None
            continue

        trailing_shapes = [value.shape[1:] for value in values]
        if any(shape != trailing_shapes[0] for shape in trailing_shapes[1:]):
            return None
        # Unknown sequence-aligned tensors need key-specific padding semantics;
        # falling back is safer than silently trimming them incorrectly.
        if first.ndim >= 2 and all(
            value.shape[1] == sequence_length
            for value, sequence_length in zip(values, sequence_lengths)
        ):
            return None

    batch_size = batch_sizes[0]
    flat_lengths = [length for column in attended_lengths for length in column]
    sorted_flat_indices = sorted(range(len(flat_lengths)), key = flat_lengths.__getitem__)
    sorted_lengths = [flat_lengths[index] for index in sorted_flat_indices]
    separate_token_slots = sum(batch_size * length for length in sequence_lengths)
    segments = _minimum_cost_length_buckets(
        sorted_lengths,
        separate_token_slots,
        _max_combined_padding_multiplier(model),
    )

    # With one bucket, retain column-major order for the existing single-batch
    # helper API and avoid an unnecessary reorder.
    if len(segments) == 1:
        bucket_flat_indices = [list(range(len(flat_lengths)))]
    else:
        bucket_flat_indices = [sorted_flat_indices[start:end] for start, end in segments]

    if len(bucket_flat_indices) == 1:
        bucket_length = max(flat_lengths)
        bucket = {}
        for key in keys:
            values = values_by_key[key]
            first = values[0]
            if not torch.is_tensor(first) or first.ndim == 0:
                bucket[key] = first
            elif key in _PADDED_SEQUENCE_FEATURES:
                pad_value = pad_token_id if key == "input_ids" else 0
                bucket[key] = _gather_single_bucket_rows(
                    values,
                    sequence_length = bucket_length,
                    pad_value = pad_value,
                )
            else:
                bucket[key] = _gather_single_bucket_rows(values)
        # Padding validity was already checked in one consolidated device
        # transfer by _right_padded_attention_lengths.
        bucket["_unsloth_right_padded"] = True
        return [bucket], batch_sizes, None

    row_plan = _prepare_bucket_row_plan(
        bucket_flat_indices,
        batch_size,
        len(sentence_features),
    )
    bucket_lengths = tuple(
        max(flat_lengths[index] for index in flat_indices) for flat_indices in bucket_flat_indices
    )
    bucket_features = [{} for _ in bucket_flat_indices]
    for key in keys:
        values = values_by_key[key]
        first = values[0]
        if not torch.is_tensor(first) or first.ndim == 0:
            for bucket in bucket_features:
                bucket[key] = first
        elif key in _PADDED_SEQUENCE_FEATURES:
            pad_value = pad_token_id if key == "input_ids" else 0
            gathered = _gather_padded_bucket_rows(
                values,
                row_plan,
                bucket_lengths,
                pad_value,
            )
            for bucket, value in zip(bucket_features, gathered):
                bucket[key] = value

        else:
            gathered = _gather_unpadded_bucket_rows(values, row_plan)
            for bucket, value in zip(bucket_features, gathered):
                bucket[key] = value

    for bucket in bucket_features:
        bucket["_unsloth_right_padded"] = True

    if row_plan["processing_index"] is None:
        return bucket_features, batch_sizes, None

    inverse_order = [0] * len(row_plan["processing_order"])
    for position, flat_index in enumerate(row_plan["processing_order"]):
        inverse_order[flat_index] = position
    return bucket_features, batch_sizes, inverse_order


def _combined_sentence_features(model, sentence_features):
    """Return a compatible one-batch encoding, preserving the legacy API."""
    sentence_features = list(sentence_features)
    bucketed = _bucketed_sentence_features(model, sentence_features)
    if bucketed is None:
        return None, None
    bucket_features, batch_sizes, _inverse_order = bucketed
    if len(bucket_features) != 1:
        return None, None
    return bucket_features[0], batch_sizes


def encode_sentence_features(model, sentence_features):
    """Encode compatible MNRL rows in adaptive length buckets."""

    def sentence_embedding_only(features):
        features["_unsloth_sentence_embedding_only"] = True
        try:
            return model(features)["sentence_embedding"]
        finally:
            features.pop("_unsloth_sentence_embedding_only", None)

    sentence_features = list(sentence_features)
    bucketed = _bucketed_sentence_features(model, sentence_features)
    if bucketed is None:
        return [sentence_embedding_only(features) for features in sentence_features]

    bucket_features, batch_sizes, inverse_order = bucketed
    bucket_embeddings = []
    for features in bucket_features:
        embeddings = sentence_embedding_only(features)
        expected_rows = features["input_ids"].shape[0]
        if embeddings.ndim < 1 or embeddings.shape[0] != expected_rows:
            # A custom module changed the batch dimension. This cannot safely be
            # re-run after a training forward (dropout/RNG and side effects).
            raise RuntimeError(
                "Bucketed SentenceTransformer forward changed the batch dimension: "
                f"expected {expected_rows}, got {embeddings.shape[0] if embeddings.ndim else 0}."
            )
        bucket_embeddings.append(embeddings)

    embeddings = torch.cat(bucket_embeddings, dim = 0)
    if inverse_order is not None:
        inverse = torch.tensor(inverse_order, device = embeddings.device, dtype = torch.long)
        embeddings = embeddings.index_select(0, inverse)
    return list(torch.split(embeddings, batch_sizes, dim = 0))


class FusedContrastiveLoss(torch.autograd.Function):
    """
    Chunked forward + backward for contrastive (InfoNCE) loss.

    embeddings_a: (B_a, D) — anchors
    embeddings_b: (B_b, D) — positives (+ extra negatives when B_b > B_a)

    The positive pair for row i is at column i (diagonal).
    Columns beyond B_a are additional negatives.

    The forward is a single streaming pass that never allocates a full
    (B_a, B_b) tensor: each chunk updates a running (max, sum) pair via the
    online log-sum-exp recurrence (Welford-style rescale when the max grows),
    extracting the positive logits on the diagonal in the same loop. This
    replaces the older two-pass approach (separate max pass + lse pass) with
    one matmul pass.
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        embeddings_a,
        embeddings_b,
        scale = 20.0,
    ):
        B_a, _dim = embeddings_a.shape
        B_b = embeddings_b.shape[0]

        if B_a > 0 and B_b == 0:
            raise ValueError("FusedContrastiveLoss requires candidates when anchors are non-empty.")
        if B_a == 0:
            # Save context so backward returns zero grads instead of crashing on
            # an empty ctx.saved_tensors unpack.
            ctx.empty = True
            ctx.save_for_backward(embeddings_a, embeddings_b)
            return embeddings_a.new_zeros(())

        if B_a > B_b:
            raise ValueError(f"FusedContrastiveLoss requires B_a <= B_b, got {B_a} and {B_b}")

        CHUNK = _contrastive_chunk_size(B_b)
        acc_dtype = (
            torch.float32
            if embeddings_a.dtype in (torch.float16, torch.bfloat16)
            else embeddings_a.dtype
        )

        # Online log-sum-exp: one streaming pass instead of two. Keep a running
        # max + sum per row so we don't recompute every chunk's matmul twice.
        running_max = torch.full(
            (B_a,),
            float("-inf"),
            device = embeddings_a.device,
            dtype = acc_dtype,
        )
        running_sum = torch.zeros(B_a, device = embeddings_a.device, dtype = acc_dtype)
        pos_logits = torch.zeros(B_a, device = embeddings_a.device, dtype = acc_dtype)

        for j0 in range(0, B_b, CHUNK):
            j1 = min(j0 + CHUNK, B_b)
            sim = (embeddings_a @ embeddings_b[j0:j1].t()).to(acc_dtype) * scale

            chunk_max = sim.max(dim = 1).values
            new_max = torch.maximum(running_max, chunk_max)
            # First chunk: exp(-inf - finite) == 0, so running_sum starts clean.
            rescale = torch.exp(running_max - new_max)
            running_sum = running_sum * rescale + torch.exp(sim - new_max.unsqueeze(1)).sum(dim = 1)
            running_max = new_max

            # Gather diagonal positives sim[i, i] in one shot (no per-row loop).
            # These are RAW (unshifted) logits; the loss adds row_max back below.
            diag_hi = min(j1, B_a)
            if diag_hi > j0:
                rows = torch.arange(j0, diag_hi, device = sim.device)
                pos_logits[j0:diag_hi] = sim[rows, rows - j0]

        # row_lse is the SHIFTED lse (relative to row_max): log(sum exp(sim - row_max)).
        # backward expects that form. The full log-sum-exp is (row_max + row_lse),
        # which is why the loss adds row_max back to the raw pos_logits here.
        row_max = running_max
        row_lse = running_sum.log()
        loss = (-pos_logits + (row_max + row_lse)).mean()

        ctx.save_for_backward(embeddings_a, embeddings_b, row_max, row_lse)
        ctx.scale = scale

        return loss

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, grad_output):
        if getattr(ctx, "empty", False):
            embeddings_a, embeddings_b = ctx.saved_tensors
            return torch.zeros_like(embeddings_a), torch.zeros_like(embeddings_b), None
        embeddings_a, embeddings_b, row_max, row_lse = ctx.saved_tensors
        scale = ctx.scale

        B_a = embeddings_a.shape[0]
        B_b = embeddings_b.shape[0]
        CHUNK = _contrastive_chunk_size(B_b)
        acc_dtype = row_max.dtype

        grad_a = torch.zeros(embeddings_a.shape, device = embeddings_a.device, dtype = acc_dtype)
        grad_b = torch.zeros(embeddings_b.shape, device = embeddings_b.device, dtype = acc_dtype)

        for j0 in range(0, B_b, CHUNK):
            j1 = min(j0 + CHUNK, B_b)
            b_chunk = embeddings_b[j0:j1]

            sim = (embeddings_a @ b_chunk.t()).to(acc_dtype) * scale
            prob = (sim - row_max.unsqueeze(1) - row_lse.unsqueeze(1)).exp()

            # subtract 1 on the diagonal, vectorized (was a row-by-row loop)
            diag_hi = min(j1, B_a)
            if diag_hi > j0:
                rows = torch.arange(j0, diag_hi, device = prob.device)
                prob[rows, rows - j0] -= 1.0

            prob = prob * (grad_output.to(acc_dtype) * scale / B_a)

            grad_a += prob @ b_chunk.to(acc_dtype)
            grad_b[j0:j1] += prob.t() @ embeddings_a.to(acc_dtype)

        return grad_a.to(embeddings_a.dtype), grad_b.to(embeddings_b.dtype), None


class FastMultipleNegativesRankingLoss(torch.nn.Module):
    """
    Launch-efficient ``MultipleNegativesRankingLoss`` that combines compatible
    text columns into one encoder call. Users normally never instantiate this:
    ``FastSentenceTransformer.from_pretrained`` patches
    ``sentence_transformers.losses.MultipleNegativesRankingLoss.forward``
    (see ``_patch_mnrl_loss`` in ``unsloth/models/sentence_transformer.py``)
    so existing training code picks up the fused path automatically. It is
    kept public for direct use with a plain ``SentenceTransformer``.
    """

    def __init__(
        self,
        model,
        scale = 20.0,
        similarity_fct = None,
    ):
        super().__init__()
        if scale <= 0:
            raise ValueError("Scale must be a positive value.")
        if similarity_fct is None:
            from sentence_transformers.util import cos_sim
            similarity_fct = cos_sim
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct

    def forward(
        self,
        sentence_features,
        labels = None,
    ):
        if labels is not None:
            import warnings
            warnings.warn(
                "Unsloth: labels is ignored by FusedContrastiveLoss (positive pairs are diagonal).",
                stacklevel = 2,
            )
        reps = encode_sentence_features(self.model, sentence_features)
        embeddings_a = reps[0]
        if any(embedding.shape[0] != embeddings_a.shape[0] for embedding in reps[1:]):
            raise ValueError("Every candidate column must match the anchor batch size.")
        embeddings_b = torch.cat(reps[1:], dim = 0)

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        target = torch.arange(embeddings_a.shape[0], device = scores.device)
        return F.cross_entropy(scores, target)
