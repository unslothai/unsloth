# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export recipe dataset artifacts to downloadable files."""

from __future__ import annotations

import json
import math
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

# A DuckDB vector is 2048 rows, so this fetches ~8k rows at a time.
_JSONL_EXPORT_VECTORS_PER_CHUNK = 4
# file_row_number is the row's ordinal inside its own shard, so this is a total order that matches
# the generated artifact. row_number() OVER (PARTITION BY filename) is not: DuckDB leaves a window
# with no ORDER BY undefined, and its parallel parquet scan then numbers the rows differently on
# every query, which paged reads turn into dropped and duplicated rows.
_PARQUET_EXPORT_SQL = (
    "SELECT * EXCLUDE (filename, file_row_number) "
    "FROM read_parquet(?, filename=true, file_row_number=true) "
    "ORDER BY filename, file_row_number"
)


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


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    converted = to_jsonable(value)
    if isinstance(converted, float):
        return None if not math.isfinite(converted) else converted
    if converted is None or isinstance(converted, (str, int, bool)):
        return converted
    if isinstance(converted, dict):
        return {str(key): _sanitize_json_value(item) for key, item in converted.items()}
    if isinstance(converted, (list, tuple, set)):
        return [_sanitize_json_value(item) for item in converted]
    return converted


def _json_dumps_row(row: dict[str, Any]) -> str:
    return json.dumps(
        _sanitize_json_value(row),
        ensure_ascii = False,
        allow_nan = False,
    )


def _write_jsonl_rows(handle, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        handle.write(_json_dumps_row(row))
        handle.write("\n")


def _parquet_glob(parquet_dir: Path) -> str:
    return str((parquet_dir / "*.parquet").resolve())


def _stream_jsonl_from_parquet_with_duckdb(*, parquet_dir: Path, destination: Path) -> bool:
    try:
        import duckdb  # type: ignore
    except Exception:
        return False

    try:
        conn = duckdb.connect(":memory:")
    except Exception:
        return False
    try:
        # One cursor for the whole export: the chunked fetches keep memory bounded, and the single
        # ORDER BY is what keeps every row present exactly once. Re-running the query per page
        # instead re-derives the order each time, so the pages overlap and gap.
        conn.execute(_PARQUET_EXPORT_SQL, [_parquet_glob(parquet_dir)])
        with destination.open("w", encoding = "utf-8") as handle:
            while True:
                dataframe = conn.fetch_df_chunk(_JSONL_EXPORT_VECTORS_PER_CHUNK)
                if dataframe.empty:
                    break
                _write_jsonl_rows(
                    handle,
                    [to_preview_jsonable(row) for row in dataframe.to_dict(orient = "records")],
                )
    except Exception:
        return False
    finally:
        conn.close()
    return True


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


def _write_jsonl_from_parquet(parquet_dir: Path, destination: Path) -> None:
    if _stream_jsonl_from_parquet_with_duckdb(
        parquet_dir = parquet_dir,
        destination = destination,
    ):
        return

    rows = _read_all_rows_with_pandas(parquet_dir)
    if rows is None:
        rows = _read_all_rows_with_data_designer(parquet_dir)
    with destination.open("w", encoding = "utf-8") as handle:
        _write_jsonl_rows(handle, rows)


def _safe_filename_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or "recipe-dataset"


def _add_images_to_archive(archive: zipfile.ZipFile, dataset_path: Path) -> None:
    images_dir = dataset_path / "images"
    if not images_dir.is_dir():
        return
    for image_file in sorted(images_dir.rglob("*")):
        if not image_file.is_file():
            continue
        relative_path = image_file.relative_to(images_dir)
        archive.write(image_file, arcname = str(Path("images") / relative_path))


def build_dataset_download(
    *, artifact_path: str, export_format: ExportFormat, filename_stem: str
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
        # Only the caller of a successful build knows to delete the temp file, so a failed one
        # has to take its own away.
        try:
            with zipfile.ZipFile(zip_path, "w", compression = zipfile.ZIP_DEFLATED) as archive:
                for parquet_file in sorted(parquet_dir.glob("*.parquet")):
                    archive.write(parquet_file, arcname = parquet_file.name)
                _add_images_to_archive(archive, dataset_path)
        except BaseException:
            zip_path.unlink(missing_ok = True)
            raise
        return zip_path, "application/zip", f"{stem}.parquet.zip"

    tmp = tempfile.NamedTemporaryFile(delete = False, suffix = ".jsonl")
    tmp.close()
    jsonl_path = Path(tmp.name)
    try:
        _write_jsonl_from_parquet(parquet_dir, jsonl_path)
    except BaseException:
        jsonl_path.unlink(missing_ok = True)
        raise
    return jsonl_path, "application/x-ndjson", f"{stem}.jsonl"
