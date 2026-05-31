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


def _load_dataset(ref: DatasetRef):
    """Load the requested split from an HF repo or a local file/dir."""
    from datasets import load_dataset

    if ref.is_local:
        path = Path(ref.path)
        if path.is_dir():
            # Directory of files (e.g. a Data Recipe output: recipe_xxx/
            # parquet-files/*.parquet, or a folder of parquet/jsonl/csv).
            parquet_dir = (
                path / "parquet-files"
                if (path / "parquet-files").exists()
                else path
            )
            parquet_files = sorted(parquet_dir.glob("*.parquet"))
            if parquet_files:
                return load_dataset(
                    "parquet",
                    data_files=[str(p) for p in parquet_files],
                    split=ref.split,
                )
            # Fall back to other supported file types in the top of `path`.
            files: list[Path] = []
            for ext, _ in (
                (".jsonl", "json"), (".json", "json"), (".csv", "csv"),
            ):
                files.extend(sorted(path.glob(f"*{ext}")))
            if not files:
                raise ValueError(
                    f"No loadable files (parquet/json/jsonl/csv) under {path}"
                )
            first_suffix = files[0].suffix.lower()
            fmt = {".jsonl": "json", ".json": "json", ".csv": "csv"}[first_suffix]
            return load_dataset(
                fmt, data_files=[str(p) for p in files], split=ref.split,
            )
        suffix = path.suffix.lower()
        fmt = {".jsonl": "json", ".json": "json", ".csv": "csv",
               ".parquet": "parquet"}.get(suffix)
        if fmt is None:
            raise ValueError(f"Unsupported local dataset file type: {suffix!r}")
        return load_dataset(fmt, data_files=str(path), split=ref.split)
    return load_dataset(ref.name, ref.subset, split=ref.split)


def load_eval_examples(
    ref: DatasetRef, *, input_col: str, reference_col: str, limit: Optional[int],
) -> list[tuple[str, Any]]:
    """Load (input, reference) pairs from an HF repo or a local file.

    Returns the first `limit` rows (all rows when limit is None).
    """
    ds = _load_dataset(ref)

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


def sample_column_values(
    ref: DatasetRef, *, column: str, limit: int,
) -> list[Any]:
    """Return the first `limit` values of `column` — used by schema inference."""
    ds = _load_dataset(ref)
    if column not in set(ds.column_names):
        raise ValueError(
            f"column {column!r} not in dataset (have: {sorted(ds.column_names)})"
        )
    n = min(max(limit, 1), len(ds))
    return [ds[i][column] for i in range(n)]
