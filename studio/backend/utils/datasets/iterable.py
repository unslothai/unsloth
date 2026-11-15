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
    """``list(stream.take(count))``, batched for typed streams: per-row reads rebuild the Arrow schema each row.

    Untyped, formatted (stacked-array batches) or schema-mismatched (ArrowException) streams read row by row.
    """
    taken = stream.take(count)
    if (
        not getattr(stream, "features", None)
        or getattr(taken, "_formatting", "unknown") is not None
    ):
        return list(taken)
    import pyarrow as pa

    rows = []
    try:
        for batch in taken.iter(batch_size = batch_size):
            columns = list(batch)
            rows.extend(dict(zip(columns, row)) for row in zip(*batch.values()))
    except pa.ArrowException:
        return list(stream.take(count))
    return rows
