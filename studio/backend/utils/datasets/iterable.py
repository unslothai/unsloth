# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for dataset iterable detection."""


def is_streaming_dataset(dataset) -> bool:
    """Return True for iterable datasets that do not support eager map kwargs."""
    try:
        from datasets import IterableDataset as HfIterableDataset
        if isinstance(dataset, HfIterableDataset):
            return True
    except ImportError:
        pass

    try:
        from torch.utils.data import IterableDataset as TorchIterableDataset
        return isinstance(dataset, TorchIterableDataset)
    except ImportError:
        return False


def take_rows(
    stream,
    count: int,
    batch_size: int = 1000,
) -> list:
    """``list(stream.take(count))``, read in batches when the stream is typed.

    A typed stream iterated one row at a time rebuilds the Arrow schema and casts every row on its
    own (~5 s per 20k rows); batches give the same rows. Typed rows carry exactly the feature
    columns, so splitting a batch back into rows loses nothing. Untyped streams (rows may differ in
    keys) and batches whose source files disagree on schema read row by row as before.
    """
    taken = stream.take(count)
    features = getattr(stream, "features", None)
    # A formatted stream (with_format) batches into stacked arrays, which do not split back into its rows.
    if (
        not features
        or getattr(taken, "_formatting", "unknown") is not None
        or not callable(getattr(taken, "iter", None))
    ):
        return list(taken)
    import pyarrow as pa

    rows = []
    try:
        for batch in taken.iter(batch_size = batch_size):
            columns = list(batch)
            if not columns:
                return list(stream.take(count))
            values = [batch[column] for column in columns]
            rows.extend(dict(zip(columns, row)) for row in zip(*values))
    except pa.ArrowException:
        return list(stream.take(count))
    return rows
