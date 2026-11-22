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
# Fallback masks TRL applies to labels; a missing or wrong-length one fails mid-training.
_MASK_COLUMNS = ("completion_mask", "assistant_masks")


def _collator_masks_user_labels() -> bool:
    """TRL < 1.7's collator applies the masks on top of a row's own labels; 1.7+ uses the labels as they are."""
    try:
        import trl
        from packaging.version import Version
        return Version(trl.__version__) < Version("1.7.0")
    except Exception:
        return True


def _checked_masks(
    columns,
    completion_only_loss = None,
    mlx = False,
) -> tuple:
    """Masks the trainer will apply, the only ones checked and truncated.

    TRL applies `assistant_masks` on presence but `completion_mask` only under
    `completion_only_loss`, which, left unset, it resolves from the dataset shape (prompt plus
    completion columns). Beside user labels the masks are applied only by TRL < 1.7's collator.
    zoo's MLX reader applies `assistant_masks` on presence, labels or not, and `completion_mask`
    only for an explicit `completion_only_loss=True`.
    """
    if mlx:
        return tuple(
            c
            for c in _MASK_COLUMNS
            if c in columns and (completion_only_loss is True or c != "completion_mask")
        )
    if "labels" in columns and not _collator_masks_user_labels():
        return ()
    if completion_only_loss is None:
        completion_only_loss = "prompt" in columns and "completion" in columns
    return tuple(
        c
        for c in _MASK_COLUMNS
        if c in columns and (completion_only_loss or c != "completion_mask")
    )


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
    """Columns the trainer's rows expose: a map-style ``set_format(columns = ...)`` hides the rest."""
    names = getattr(dataset, "column_names", None)
    if names:
        fmt = getattr(dataset, "format", None)
        if (
            isinstance(fmt, dict)
            and fmt.get("columns") is not None
            and not fmt.get("output_all_columns")
        ):
            return [c for c in names if c in fmt["columns"]]
        return list(names)
    return list(next(iter(dataset)).keys())


def resolve_completion_only_loss(train_dataset, completion_only_loss = None) -> bool:
    """TRL resolves an unset `completion_only_loss` once, from the train split's first row, for every split."""
    if completion_only_loss is not None:
        return completion_only_loss
    names = _column_names(train_dataset)
    return "prompt" in names and "completion" in names


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


def _raise_null_values(column, row_index):
    raise ValueError(
        f"Pre-tokenized dataset: row {row_index} has a missing (null) value inside '{column}'. "
        "Every position must hold an integer."
    )


def _check_arrow_table(
    table,
    max_seq_length,
    vocab_size,
    masks = (),
    labels = True,
) -> tuple[bool, bool]:
    """Returns (needs_truncation, needs_trimming)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    ids = table.column("input_ids")
    lengths = pc.list_value_length(ids)
    for name in masks:
        # An all-null or scalar column has no list kernel: reject it with the same error.
        kind = table.column(name).type
        if not (
            pa.types.is_list(kind)
            or pa.types.is_large_list(kind)
            or pa.types.is_fixed_size_list(kind)
        ):
            raise _mask_error(name)
        same = pc.equal(lengths, pc.list_value_length(table.column(name)))
        # A row without the mask compares as null: that row fails too.
        if pc.all(same, skip_nulls = False).as_py() is not True:
            raise _mask_error(name)
    others = [c for c in ("attention_mask", "token_type_ids") if c in table.column_names]
    if labels and "labels" in table.column_names:
        others.append("labels")
    # Arrow reductions (any, all, min_max) skip nulls, so a mask such as [None, 1] or a label hole would pass every
    # check below and reach the trainer; refuse null elements first.
    for column in ["input_ids", *masks, *others]:
        if pc.list_flatten(table.column(column)).null_count:
            values = table.column(column).combine_chunks()
            holes = pc.is_null(pc.list_flatten(values))
            _raise_null_values(column, pc.filter(pc.list_parent_indices(values), holes)[0].as_py())
    # Trimming slices every token column at the mask's boundaries, so each must line up with input_ids.
    for column in others:
        if not pc.all(pc.equal(lengths, pc.list_value_length(table.column(column)))).as_py():
            raise _mask_error(column)
    needs_trimming = False
    if "attention_mask" in table.column_names:
        mask = table.column("attention_mask").combine_chunks()
        flat_mask = pc.list_flatten(mask)
        if pa.types.is_boolean(flat_mask.type):
            # Arrow has no bool-vs-int comparison kernel; True/False are exactly 1/0, so compare them as integers.
            flat_mask = pc.cast(flat_mask, pa.int8())
        # Exact membership, not a range: a float mask such as [0.5, 1.0] sits inside [0, 1] yet is neither value.
        bad = pc.and_(pc.not_equal(flat_mask, 0), pc.not_equal(flat_mask, 1))
        if pc.any(bad).as_py():
            _raise_bad_mask_values(pc.filter(pc.list_parent_indices(mask), bad)[0].as_py())
        # Only a mask with zeros changes what trains (TRL rebuilds all-one masks); an all-ones mask is left as is.
        needs_trimming = pc.min_max(flat_mask).as_py()["min"] == 0
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
    if "labels" in others:
        _check_labels_range(pc.list_flatten(table.column("labels")), vocab_size)
    longest = pc.max(lengths).as_py() or 0
    return bool(max_seq_length) and longest > max_seq_length, needs_trimming


def _trim_row(
    row_index,
    columns,
    values,
    masks = (),
):
    """Drops the leading/trailing attention_mask==0 positions of one row from every token column. TRL's collators
    rebuild the mask as all ones (padding-free ones drop it), so trimming is the only way a pad stays unattended."""
    width = len(values[columns.index("input_ids")])
    for column, value in zip(columns, values):
        # A mask the trainer applies must be present; a scalar (None, int, numpy 0-d) has no len().
        if value is None and column not in masks:
            continue
        try:
            bad = len(value) != width
        except TypeError:
            bad = True
        if bad:
            raise _mask_error(column)
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


def _restore_format(out, fmt):
    # The whole format: the column selection too, so hidden columns stay hidden.
    return out.with_format(
        fmt["type"],
        columns = fmt["columns"],
        output_all_columns = fmt["output_all_columns"],
        **(fmt["format_kwargs"] or {}),
    )


def prepare_pretokenized_dataset(
    dataset,
    *,
    max_seq_length,
    vocab_size = None,
    completion_only_loss = None,
    mlx = False,
):
    """Unsloth's SFT preparation passes tokenized rows through untouched: an over-long row would reach the model whole and an id past the embedding table is a device-side assert mid-run. Rows whose attention_mask zeros are leading/trailing padding lose those positions (the trainer would attend and train on them); zeros between attended tokens are refused. A falsy max_seq_length skips truncation. Idempotent."""
    names = _column_names(dataset)
    if not mlx:
        completion_only_loss = resolve_completion_only_loss(dataset, completion_only_loss)
    # Masks are checked even when set_format hides them: TRL 0.22-0.26 re-exposes every column through
    # with_transform, and 1.7+ reads dataset.column_names and builds labels with map(), which sees them all.
    masks = _checked_masks(
        getattr(dataset, "column_names", None) or names, completion_only_loss, mlx = mlx
    )
    # A mask the trainer will not apply is left as it is (it may not even be a list).
    columns = [c for c in _TOKEN_COLUMNS if (c in names and c not in _MASK_COLUMNS) or c in masks]
    table = getattr(dataset, "data", None)

    if table is None or not hasattr(table, "column"):
        # Streaming: nothing to scan up front, so rows are checked, trimmed and truncated as they are read.
        def _prepare(batch, indices):
            # A formatted stream (numpy, torch) hands over arrays: work on plain lists.
            out = {
                c: [v.tolist() if hasattr(v, "tolist") else v for v in batch[c]] for c in columns
            }
            for i, row_index in enumerate(indices):
                for c in columns:
                    if isinstance(out[c][i], (list, tuple)) and None in out[c][i]:
                        _raise_null_values(c, row_index)
                labels = out["labels"][i] if "labels" in out else None
                if labels is not None and len(labels) != len(out["input_ids"][i]):
                    raise _mask_error("labels")
                if labels:
                    high, low = max(labels), min((x for x in labels if x != -100), default = 0)
                    if low < 0 or (vocab_size and high >= vocab_size):
                        _raise_bad_label(low if low < 0 else high, vocab_size)
                values = _trim_row(row_index, columns, [out[c][i] for c in columns], masks)
                if max_seq_length:
                    values = [v[:max_seq_length] if v is not None else v for v in values]
                for c, v in zip(columns, values):
                    out[c][i] = v
            return out

        return dataset.map(_prepare, batched = True, with_indices = True)

    # Covers the whole backing table, rows a select() hid included: a bad id there is still a bad dataset.
    needs_truncation, needs_trimming = _check_arrow_table(
        table, max_seq_length, vocab_size, masks, labels = "labels" in names
    )
    if needs_trimming:

        def _trim(batch, indices):
            out = {c: list(batch[c]) for c in columns}
            for i, row_index in enumerate(indices):
                trimmed = _trim_row(row_index, columns, [out[c][i] for c in columns], masks)
                for c, v in zip(columns, trimmed):
                    out[c][i] = v
            return out

        fmt = dataset.format
        dataset = _restore_format(
            dataset.with_format(None).map(
                _trim,
                batched = True,
                with_indices = True,
                desc = "Trimming masked padding from pre-tokenized rows",
            ),
            fmt,
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
    return _restore_format(out, dataset.format)
