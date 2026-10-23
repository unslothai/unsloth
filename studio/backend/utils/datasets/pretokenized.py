# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tokenized text datasets: rows that already hold ``input_ids`` (optionally ``attention_mask`` and ``labels``). They train as-is, without a chat template; user ``labels`` are the objective, so response masking applies only to rows without them. Only datasets no other format matched are claimed, so every dataset that trained before keeps its route."""

PRETOKENIZED_FORMAT = "pretokenized"
# Every per-token column a row may carry. TRL reads completion_mask / assistant_masks position by position (the 0.22
# collator applies them to the labels, 1.x folds them into labels), so they are trimmed and truncated with input_ids.
_TOKEN_COLUMNS = (
    "input_ids",
    "attention_mask",
    "labels",
    "token_type_ids",
    "completion_mask",
    "assistant_masks",
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


def _check_labels_range(flat_labels, vocab_size) -> None:
    import pyarrow.compute as pc

    flat = pc.filter(flat_labels, pc.not_equal(flat_labels, -100))
    if not len(flat):
        return
    bounds = pc.min_max(flat).as_py()
    if bounds["min"] is None:
        return
    if bounds["min"] < 0 or (vocab_size and bounds["max"] >= vocab_size):
        bad = bounds["min"] if bounds["min"] < 0 else bounds["max"]
        _raise_bad_label(bad, vocab_size)


def _raise_bad_label(bad, vocab_size):
    limit = f"in [0, {vocab_size})" if vocab_size else "non-negative"
    raise ValueError(
        f"Pre-tokenized dataset: label {bad} is invalid. Labels must be -100 (ignored) or a token id "
        f"{limit}. The dataset was likely tokenized for a different model or uses another ignore index."
    )


def _raise_bad_mask_values(row_index):
    raise ValueError(
        f"Pre-tokenized dataset: row {row_index} has 'attention_mask' values other than 0 and 1."
    )


def _raise_length_mismatch(column):
    raise ValueError(
        f"Pre-tokenized dataset: '{column}' must have the same length as 'input_ids' in every row."
    )


def _check_arrow_table(table, max_seq_length, vocab_size) -> tuple[bool, bool]:
    """Returns (needs_truncation, needs_trimming)."""
    import pyarrow.compute as pc

    ids = table.column("input_ids")
    lengths = pc.list_value_length(ids)
    needs_trimming = False
    if "attention_mask" in table.column_names:
        flat_mask = pc.list_flatten(table.column("attention_mask"))
        # Exact membership, not a range: a float mask such as [0.5, 1.0] sits inside [0, 1] yet is neither value.
        if pc.any(pc.and_(pc.not_equal(flat_mask, 0), pc.not_equal(flat_mask, 1))).as_py():
            mask = table.column("attention_mask").combine_chunks()
            flat = pc.list_flatten(mask)
            bad = pc.and_(pc.not_equal(flat, 0), pc.not_equal(flat, 1))
            _raise_bad_mask_values(pc.filter(pc.list_parent_indices(mask), bad)[0].as_py())
        bounds = pc.min_max(flat_mask).as_py()
        # Only a mask with zeros changes what trains (TRL rebuilds all-one masks); an all-ones mask is left as is.
        needs_trimming = bounds["min"] == 0
    # Trimming slices every token column at the mask's boundaries and TRL applies the supervision masks position by
    # position, so each must line up with input_ids whether or not a row is trimmed.
    for column in _TOKEN_COLUMNS[1:]:
        if (
            column in table.column_names
            and not pc.all(pc.equal(lengths, pc.list_value_length(table.column(column)))).as_py()
        ):
            _raise_length_mismatch(column)
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
    if "labels" in table.column_names:
        _check_labels_range(pc.list_flatten(table.column("labels")), vocab_size)
    longest = pc.max(lengths).as_py() or 0
    return bool(max_seq_length) and longest > max_seq_length, needs_trimming


def _trim_row(row_index, columns, values):
    """Drops the leading/trailing attention_mask==0 positions of one row from every token column. TRL's collators
    rebuild the mask as all ones (padding-free ones drop it), so trimming is the only way a pad stays unattended."""
    width = len(values[columns.index("input_ids")])
    for column, value in zip(columns, values):
        if value is not None and len(value) != width:
            _raise_length_mismatch(column)
    mask = values[columns.index("attention_mask")] if "attention_mask" in columns else None
    if mask is None:
        return values
    if mask.count(0) + mask.count(1) != len(mask):
        _raise_bad_mask_values(row_index)
    if 0 not in mask:
        return values
    if 1 not in mask:
        raise ValueError(
            f"Pre-tokenized dataset: row {row_index} has an all-zero 'attention_mask', so it has no tokens "
            "to train on. Remove the row."
        )
    start = mask.index(1)
    stop = len(mask) - mask[::-1].index(1)
    if 0 in mask[start:stop]:
        raise ValueError(
            f"Pre-tokenized dataset: row {row_index} has 'attention_mask' zeros between attended tokens. "
            "Only leading or trailing padding can be masked; Studio's trainer attends every kept token, "
            "so remove the masked tokens from input_ids (and labels) instead."
        )
    return [v[start:stop] if v is not None else v for v in values]


def prepare_pretokenized_dataset(
    dataset,
    *,
    max_seq_length,
    vocab_size = None,
):
    """Unsloth's SFT preparation passes tokenized rows through untouched: an over-long row would reach the model whole and an id past the embedding table is a device-side assert mid-run. Rows whose attention_mask zeros are leading/trailing padding lose those positions (the trainer would attend and train on them); zeros between attended tokens are refused. A falsy max_seq_length skips truncation. Idempotent."""
    columns = [c for c in _TOKEN_COLUMNS if c in _column_names(dataset)]
    table = getattr(dataset, "data", None)

    if table is None or not hasattr(table, "column"):
        # Streaming: nothing to scan up front, so rows are checked, trimmed and truncated as they are read.
        def _prepare(batch, indices):
            out = {c: list(batch[c]) for c in columns}
            for i, row_index in enumerate(indices):
                labels = out["labels"][i] if "labels" in out else None
                if labels is not None and len(labels) != len(out["input_ids"][i]):
                    _raise_length_mismatch("labels")
                if labels:
                    high, low = max(labels), min((x for x in labels if x != -100), default = 0)
                    if low < 0 or (vocab_size and high >= vocab_size):
                        _raise_bad_label(low if low < 0 else high, vocab_size)
                values = _trim_row(row_index, columns, [out[c][i] for c in columns])
                if max_seq_length:
                    values = [v[:max_seq_length] if v is not None else v for v in values]
                for c, v in zip(columns, values):
                    out[c][i] = v
            return out

        return dataset.map(_prepare, batched = True, with_indices = True)

    # Covers the whole backing table, rows a select() hid included: a bad id there is still a bad dataset.
    needs_truncation, needs_trimming = _check_arrow_table(table, max_seq_length, vocab_size)
    if needs_trimming:

        def _trim(batch, indices):
            out = {c: list(batch[c]) for c in columns}
            for i, row_index in enumerate(indices):
                trimmed = _trim_row(row_index, columns, [out[c][i] for c in columns])
                for c, v in zip(columns, trimmed):
                    out[c][i] = v
            return out

        fmt = dataset.format
        dataset = dataset.with_format(None).map(
            _trim,
            batched = True,
            with_indices = True,
            desc = "Trimming masked padding from pre-tokenized rows",
        )
        dataset = dataset.with_format(
            fmt["type"],
            columns = fmt["columns"],
            output_all_columns = fmt["output_all_columns"],
            **fmt["format_kwargs"],
        )
    if not needs_truncation or not max_seq_length:
        return dataset

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
