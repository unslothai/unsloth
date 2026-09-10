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
from utils.paths.path_utils import drop_appledouble_metadata

ExportFormat = Literal["jsonl", "parquet"]

# A DuckDB vector is 2048 rows, so this fetches ~8k rows at a time.
_JSONL_EXPORT_VECTORS_PER_CHUNK = 4
_JSONL_EXPORT_BATCH_ROWS = 8192
# file_row_number, not row_number() OVER (PARTITION BY filename): a window with no ORDER BY is
# undefined in DuckDB, and its parallel scan renumbered every query.
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
    if not _parquet_files(parquet_dir):
        raise RecipeDatasetExportError(f"No parquet files found in {parquet_dir}")
    return parquet_dir


def _parquet_files(parquet_dir: Path) -> list[Path]:
    """The real shards, without the ``._batch.parquet`` companions a macOS volume leaves beside
    them: the glob matches those but no reader can parse one. Same call the worker makes."""
    return drop_appledouble_metadata(sorted(parquet_dir.glob("*.parquet")))


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
        # One cursor for the whole export: the single ORDER BY is what keeps every row exactly
        # once. A shard list, not a glob DuckDB would re-expand over the companions.
        conn.execute(
            _PARQUET_EXPORT_SQL,
            [[str(path.resolve()) for path in _parquet_files(parquet_dir)]],
        )
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


def _write_jsonl_with_pyarrow(parquet_dir: Path, destination: Path) -> bool:
    """Row group at a time, for a dataset DuckDB will not take. Not shard at a time:
    ``merge_batches`` collapses a whole run into one file. Declines rather than half-write."""
    try:
        import pyarrow.parquet as pyarrow_parquet  # type: ignore
    except Exception:
        return False

    parquet_files = _parquet_files(parquet_dir)
    if not parquet_files:
        return False

    try:
        with destination.open("w", encoding = "utf-8") as handle:
            for path in parquet_files:
                parquet_file = pyarrow_parquet.ParquetFile(path)
                for batch in parquet_file.iter_batches(batch_size = _JSONL_EXPORT_BATCH_ROWS):
                    _write_jsonl_rows(
                        handle,
                        [to_preview_jsonable(row) for row in batch.to_pylist()],
                    )
    except Exception:
        return False
    return True


def _read_all_rows_with_data_designer(parquet_dir: Path) -> list[dict[str, Any]]:
    from data_designer.config.utils.io_helpers import read_parquet_dataset

    dataframe = read_parquet_dataset(parquet_dir)
    rows = dataframe.to_dict(orient = "records")
    return [to_preview_jsonable(row) for row in rows]


def _write_jsonl_from_parquet(parquet_dir: Path, destination: Path) -> None:
    # Only the last of these materializes the dataset; the first two stream.
    if _stream_jsonl_from_parquet_with_duckdb(
        parquet_dir = parquet_dir,
        destination = destination,
    ):
        return
    if _write_jsonl_with_pyarrow(parquet_dir, destination):
        return

    with destination.open("w", encoding = "utf-8") as handle:
        _write_jsonl_rows(handle, _read_all_rows_with_data_designer(parquet_dir))


def _safe_filename_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or "recipe-dataset"


def _artifact_image_files(dataset_path: Path) -> list[Path]:
    images_dir = dataset_path / "images"
    if not images_dir.is_dir():
        return []
    return [
        path for path in drop_appledouble_metadata(sorted(images_dir.rglob("*"))) if path.is_file()
    ]


def download_filename(*, artifact_path: str, export_format: ExportFormat, stem: str) -> str:
    """The name the export will have, and the checks ``build_dataset_download`` will make, so a
    run whose shards are gone is refused while the caller can still be told."""
    try:
        dataset_path = _resolve_recipe_artifact_path(artifact_path)
    except RecipeDatasetPublishError as exc:
        raise RecipeDatasetExportError(str(exc)) from exc
    _parquet_dir(dataset_path)
    if export_format == "parquet":
        return f"{stem}.parquet.zip"
    return f"{stem}.jsonl.zip" if _artifact_image_files(dataset_path) else f"{stem}.jsonl"


def _add_images_to_archive(archive: zipfile.ZipFile, dataset_path: Path) -> None:
    images_dir = dataset_path / "images"
    if not images_dir.is_dir():
        return
    for image_file in drop_appledouble_metadata(sorted(images_dir.rglob("*"))):
        if not image_file.is_file():
            continue
        relative_path = image_file.relative_to(images_dir)
        # Stored, not deflated: deflating already-compressed images bought nothing and cost 7x.
        archive.write(
            image_file,
            arcname = str(Path("images") / relative_path),
            compress_type = zipfile.ZIP_STORED,
        )


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
        # Only a successful build's caller unlinks the temp file, so a failed one takes its own.
        try:
            with zipfile.ZipFile(zip_path, "w", compression = zipfile.ZIP_DEFLATED) as archive:
                for parquet_file in _parquet_files(parquet_dir):
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
        image_files = _artifact_image_files(dataset_path)
        if not image_files:
            return jsonl_path, "application/x-ndjson", f"{stem}.jsonl"
        # Rows reference these by relative path, so a bare JSONL loses every image.
        zip_tmp = tempfile.NamedTemporaryFile(delete = False, suffix = ".zip")
        zip_tmp.close()
        zip_path = Path(zip_tmp.name)
        try:
            with zipfile.ZipFile(zip_path, "w", compression = zipfile.ZIP_DEFLATED) as archive:
                archive.write(jsonl_path, arcname = f"{stem}.jsonl")
                _add_images_to_archive(archive, dataset_path)
        except BaseException:
            zip_path.unlink(missing_ok = True)
            raise
    except BaseException:
        jsonl_path.unlink(missing_ok = True)
        raise
    jsonl_path.unlink(missing_ok = True)
    return zip_path, "application/zip", f"{stem}.jsonl.zip"
