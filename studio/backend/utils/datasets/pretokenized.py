# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tokenized text datasets: rows that already hold ``input_ids`` (optionally ``attention_mask`` and ``labels``). They train as-is: no chat template, no response masking, since the user's ``labels`` are the objective. Auto-detection only claims datasets no other format matched, so every dataset that trained before keeps its route."""

PRETOKENIZED_FORMAT = "pretokenized"
_TOKEN_COLUMNS = ("input_ids", "attention_mask", "labels", "token_type_ids")
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


def _check_arrow_table(table, max_seq_length, vocab_size) -> bool:
    """Validate ids and label lengths over the backing Arrow table; return whether any row needs truncating."""
    import pyarrow.compute as pc

    ids = table.column("input_ids")
    lengths = pc.list_value_length(ids)
    if "labels" in table.column_names:
        labels = table.column("labels")
        if not pc.all(pc.equal(lengths, pc.list_value_length(labels))).as_py():
            raise ValueError(
                "Pre-tokenized dataset: 'labels' must have the same length as 'input_ids' in every row."
            )
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
    columns = [c for c in _TOKEN_COLUMNS if c in _column_names(dataset)]
    table = getattr(dataset, "data", None)
    needs_truncation = True
    if table is not None and hasattr(table, "column"):
        # Covers the whole backing table, rows a select() hid included: a bad id there is still a bad dataset.
        needs_truncation = _check_arrow_table(table, max_seq_length, vocab_size)
    if not needs_truncation or not max_seq_length:
        return dataset

    def _truncate(batch):
        return {c: [v[:max_seq_length] if v is not None else v for v in batch[c]] for c in columns}

    if table is not None:
        return dataset.map(_truncate, batched = True, desc = "Truncating pre-tokenized rows")
    return dataset.map(_truncate, batched = True)
