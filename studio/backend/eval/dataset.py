# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class DatasetRef:
    is_local: bool
    path: Optional[str] = None     # local file path (when is_local)
    name: Optional[str] = None     # HF repo id (when not is_local)
    split: str = "train"
    subset: Optional[str] = None


def _coerce(value: Any) -> Any:
    # text columns -> str; structured (dict/list) references pass through.
    if isinstance(value, (dict, list)) or value is None:
        return value
    return str(value)


def load_eval_examples(
    ref: DatasetRef, *, input_col: str, reference_col: str, limit: Optional[int],
) -> list[tuple[str, Any]]:
    """Load (input, reference) pairs from an HF repo or a local file.

    Returns the first `limit` rows (all rows when limit is None).
    """
    from datasets import load_dataset

    if ref.is_local:
        path = Path(ref.path)
        suffix = path.suffix.lower()
        fmt = {".jsonl": "json", ".json": "json", ".csv": "csv",
               ".parquet": "parquet"}.get(suffix)
        if fmt is None:
            raise ValueError(f"Unsupported local dataset file type: {suffix!r}")
        ds = load_dataset(fmt, data_files=str(path), split=ref.split)
    else:
        ds = load_dataset(ref.name, ref.subset, split=ref.split)

    cols = set(ds.column_names)
    for col in (input_col, reference_col):
        if col not in cols:
            raise ValueError(
                f"column {col!r} not in dataset (have: {sorted(cols)})"
            )

    n = len(ds) if limit is None else min(limit, len(ds))
    sliced = ds.select(range(n))
    return [
        (str(row[input_col]) if row[input_col] is not None else "",
         _coerce(row[reference_col]))
        for row in sliced
    ]
