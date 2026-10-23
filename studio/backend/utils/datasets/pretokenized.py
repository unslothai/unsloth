# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tokenized text datasets: rows that already hold ``input_ids`` (optionally ``attention_mask`` and ``labels``). They train as-is, without a chat template; user ``labels`` are the objective, so response masking applies only to rows without them. Only datasets no other format matched are claimed, so every dataset that trained before keeps its route."""

PRETOKENIZED_FORMAT = "pretokenized"
_TOKEN_COLUMNS = (
    "input_ids",
    "attention_mask",
    "labels",
    "token_type_ids",
    "completion_mask",
    "assistant_masks",
)
# Per-token masks TRL applies to the labels when a row has none of its own: a missing value or a
# length other than input_ids' fails mid-training.
_MASK_COLUMNS = ("completion_mask", "assistant_masks")


def _checked_masks(columns) -> tuple:
    # User labels are the whole objective: the masks are then unused and not checked.
    return () if "labels" in columns else tuple(c for c in _MASK_COLUMNS if c in columns)


def _mask_error(name):
    return ValueError(
        f"Pre-tokenized dataset: '{name}' must have the same length as 'input_ids' in every row."
    )


_SAMPLE_ROWS = 20


def _is_token_list(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    )


def is_pretokenized_dataset(dataset) -> bool:
    from .format_detection import _sample_dataset_rows

    rows = _sample_dataset_rows(dataset, limit = _SAMPLE_ROWS)
    if not rows:
        return False
    for row in rows:
        if not isinstance(row, dict) or not _is_token_list(row.get("input_ids")):
            return False
        labels = row.get("labels")
        if labels is not None and (
            not _is_token_list(labels) or len(labels) != len(row["input_ids"])
        ):
            return False
    return True


def _column_names(dataset) -> list[str]:
    names = getattr(dataset, "column_names", None)
    if names:
        return list(names)
    return list(next(iter(dataset)).keys())


def has_custom_labels(dataset) -> bool:
    return "labels" in _column_names(dataset)


def _check_arrow_table(table, max_seq_length, vocab_size) -> bool:
    import pyarrow.compute as pc

    ids = table.column("input_ids")
    lengths = pc.list_value_length(ids)
    if "labels" in table.column_names:
        labels = table.column("labels")
        if not pc.all(pc.equal(lengths, pc.list_value_length(labels))).as_py():
            raise ValueError(
                "Pre-tokenized dataset: 'labels' must have the same length as 'input_ids' in every row."
            )
    for name in _checked_masks(table.column_names):
        same = pc.equal(lengths, pc.list_value_length(table.column(name)))
        # A row without the mask compares as null: that row fails too.
        if pc.all(same, skip_nulls = False).as_py() is not True:
            raise _mask_error(name)
    if vocab_size:
        flat = pc.list_flatten(ids)
        bounds = pc.min_max(flat).as_py() if len(flat) else {"min": 0, "max": 0}
        if bounds["min"] < 0 or bounds["max"] >= vocab_size:
            bad = bounds["min"] if bounds["min"] < 0 else bounds["max"]
            raise ValueError(
                f"Pre-tokenized dataset: token id {bad} is outside this model's vocabulary "
                f"(size {vocab_size}). The dataset was likely "
                "tokenized for a different model; re-tokenize it with this model's tokenizer."
            )
    longest = pc.max(lengths).as_py() or 0
    return longest > max_seq_length


def prepare_pretokenized_dataset(
    dataset,
    *,
    max_seq_length,
    vocab_size = None,
):
    """Unsloth's SFT preparation passes tokenized rows through untouched: an over-long row would reach the model whole and an id past the embedding table is a device-side assert mid-run. Idempotent."""
    names = _column_names(dataset)
    # With user labels the masks are unused: leave them as they are (they may not even be lists).
    unused = _MASK_COLUMNS if "labels" in names else ()
    columns = [c for c in _TOKEN_COLUMNS if c in names and c not in unused]
    table = getattr(dataset, "data", None)
    needs_truncation = True
    if table is not None and hasattr(table, "column"):
        # Covers the whole backing table, rows a select() hid included: a bad id there is still a bad dataset.
        needs_truncation = _check_arrow_table(table, max_seq_length, vocab_size)
    if not needs_truncation or not max_seq_length:
        return dataset

    if table is None:
        masks = _checked_masks(columns)

        def _truncate(batch):
            # Streamed rows never reach _check_arrow_table: check the masks before slicing.
            for name in masks:
                for ids, mask in zip(batch["input_ids"], batch[name]):
                    if mask is None or ids is None or len(mask) != len(ids):
                        raise _mask_error(name)
            return {
                c: [v[:max_seq_length] if v is not None else v for v in batch[c]] for c in columns
            }

        return dataset.map(_truncate, batched = True)

    import pyarrow.compute as pc

    def _truncate_arrow(batch):
        # Arrow slices in C: 46x faster than slicing Python lists on 52M tokens.
        for c in columns:
            batch = batch.set_column(
                batch.column_names.index(c), c, pc.list_slice(batch.column(c), 0, max_seq_length)
            )
        return batch

    out = dataset.with_format("arrow").map(
        _truncate_arrow, batched = True, desc = "Truncating pre-tokenized rows"
    )
    return out.with_format(dataset.format["type"])
