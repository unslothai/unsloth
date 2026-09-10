# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export recipe dataset artifacts to downloadable files."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from core.data_recipe.huggingface import (
    RecipeDatasetPublishError,
    _resolve_recipe_artifact_path,
)
from core.data_recipe.jsonable import to_jsonable, to_preview_jsonable

ExportFormat = Literal["jsonl", "parquet"]


class RecipeDatasetExportError(ValueError):
    """Raised when a recipe dataset cannot be exported."""


def _parquet_dir(dataset_path: Path) -> Path:
    parquet_dir = dataset_path / "parquet-files"
    if not parquet_dir.exists():
        raise RecipeDatasetExportError(f"Dataset parquet files missing: {parquet_dir}")
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise RecipeDatasetExportError(f"No parquet files found in {parquet_dir}")
    return parquet_dir


def _read_all_rows_with_duckdb(parquet_dir: Path) -> list[dict[str, Any]] | None:
    parquet_glob = str((parquet_dir / "*.parquet").resolve())
    try:
        import duckdb  # type: ignore
    except Exception:
        return None

    try:
        conn = duckdb.connect(":memory:")
        try:
            dataframe = conn.execute(
                (
                    "SELECT * FROM read_parquet(?, filename=true) "
                    "ORDER BY filename"
                ),
                [parquet_glob],
            ).fetchdf()
        finally:
            conn.close()
    except (RuntimeError, ValueError, duckdb.Error):
        return None

    for helper_col in ("filename",):
        if helper_col in dataframe.columns:
            dataframe = dataframe.drop(columns = [helper_col])

    rows = dataframe.to_dict(orient = "records")
    return [to_preview_jsonable(row) for row in rows]


def _read_all_rows_with_pandas(parquet_dir: Path) -> list[dict[str, Any]] | None:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    try:
        dataframe = pd.concat(
            [pd.read_parquet(path) for path in parquet_files],
            ignore_index = True,
        )
    except Exception:
        return None

    rows = dataframe.to_dict(orient = "records")
    return [to_preview_jsonable(row) for row in rows]


def _read_all_rows_with_data_designer(parquet_dir: Path) -> list[dict[str, Any]]:
    from data_designer.config.utils.io_helpers import read_parquet_dataset

    dataframe = read_parquet_dataset(parquet_dir)
    rows = dataframe.to_dict(orient = "records")
    return [to_preview_jsonable(row) for row in rows]


def _read_all_rows(parquet_dir: Path) -> list[dict[str, Any]]:
    rows = _read_all_rows_with_duckdb(parquet_dir)
    if rows is not None:
        return rows
    rows = _read_all_rows_with_pandas(parquet_dir)
    if rows is not None:
        return rows
    return _read_all_rows_with_data_designer(parquet_dir)


def _write_jsonl_file(rows: list[dict[str, Any]], destination: Path) -> None:
    with destination.open("w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii = False))
            handle.write("\n")


def _safe_filename_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or "recipe-dataset"


def build_dataset_download(
    *,
    artifact_path: str,
    export_format: ExportFormat,
    filename_stem: str,
) -> tuple[Path, str, str]:
    """Return ``(temp_path, media_type, download_filename)``.

    The caller is responsible for deleting ``temp_path`` after the response is sent.
    """
    try:
        dataset_path = _resolve_recipe_artifact_path(artifact_path)
    except RecipeDatasetPublishError as exc:
        raise RecipeDatasetExportError(str(exc)) from exc

    stem = _safe_filename_stem(filename_stem)
    parquet_dir = _parquet_dir(dataset_path)

    if export_format == "parquet":
        tmp = tempfile.NamedTemporaryFile(delete = False, suffix = ".zip")
        tmp.close()
        zip_path = Path(tmp.name)
        with zipfile.ZipFile(zip_path, "w", compression = zipfile.ZIP_DEFLATED) as archive:
            for parquet_file in sorted(parquet_dir.glob("*.parquet")):
                archive.write(parquet_file, arcname = parquet_file.name)
        return zip_path, "application/zip", f"{stem}.parquet.zip"

    rows = _read_all_rows(parquet_dir)
    tmp = tempfile.NamedTemporaryFile(delete = False, suffix = ".jsonl")
    tmp.close()
    jsonl_path = Path(tmp.name)
    _write_jsonl_file(rows, jsonl_path)
    return jsonl_path, "application/x-ndjson", f"{stem}.jsonl"


def build_in_memory_dataset_download(
    rows: list[dict[str, Any]],
    *,
    filename_stem: str,
) -> tuple[Path, str, str]:
    """Export rows already held in memory to a temporary JSONL file."""
    stem = _safe_filename_stem(filename_stem)
    tmp = tempfile.NamedTemporaryFile(delete = False, suffix = ".jsonl")
    tmp.close()
    jsonl_path = Path(tmp.name)
    _write_jsonl_file(rows, jsonl_path)
    return jsonl_path, "application/x-ndjson", f"{stem}.jsonl"
