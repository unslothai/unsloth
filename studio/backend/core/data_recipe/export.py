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
    if not _parquet_files(parquet_dir):
        raise RecipeDatasetExportError(f"No parquet files found in {parquet_dir}")
    return parquet_dir


def _parquet_files(parquet_dir: Path) -> list[Path]:
    """The real shards. A macOS volume stores extended attributes in a ``._batch.parquet``
    companion that the glob matches but no reader can parse, so it is dropped here the way the
    worker that writes this directory drops it."""
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
        # One cursor for the whole export: the chunked fetches keep memory bounded, and the single
        # ORDER BY is what keeps every row present exactly once. Re-running the query per page
        # instead re-derives the order each time, so the pages overlap and gap. The shard list
        # rather than a "*.parquet" glob, which DuckDB would expand back over the companions.
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


def _write_jsonl_with_pandas(parquet_dir: Path, destination: Path) -> bool:
    """Shard at a time, so a dataset DuckDB would not take does not have to fit in memory. It
    declines the job outright rather than leaving a half-written file behind."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False

    parquet_files = _parquet_files(parquet_dir)
    if not parquet_files:
        return False

    try:
        with destination.open("w", encoding = "utf-8") as handle:
            for path in parquet_files:
                rows = pd.read_parquet(path).to_dict(orient = "records")
                _write_jsonl_rows(handle, [to_preview_jsonable(row) for row in rows])
    except Exception:
        return False
    return True


def _read_all_rows_with_data_designer(parquet_dir: Path) -> list[dict[str, Any]]:
    from data_designer.config.utils.io_helpers import read_parquet_dataset

    dataframe = read_parquet_dataset(parquet_dir)
    rows = dataframe.to_dict(orient = "records")
    return [to_preview_jsonable(row) for row in rows]


def _write_jsonl_from_parquet(parquet_dir: Path, destination: Path) -> None:
    # DuckDB streams it; pandas streams it a shard at a time when DuckDB will not take the schema
    # (a dataset carrying its own `filename` or `file_row_number` column is one); the Data Designer
    # reader is the last resort and is the only one that materializes everything.
    if _stream_jsonl_from_parquet_with_duckdb(
        parquet_dir = parquet_dir,
        destination = destination,
    ):
        return
    if _write_jsonl_with_pandas(parquet_dir, destination):
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
    """The name the export will actually have. Whether the JSONL comes back zipped depends on the
    artifact, so the server settles it and the client is told rather than guessing."""
    try:
        dataset_path = _resolve_recipe_artifact_path(artifact_path)
    except RecipeDatasetPublishError as exc:
        raise RecipeDatasetExportError(str(exc)) from exc
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
        # Stored, not deflated: on 500 MB of PNGs that cost 8.6s and saved 0 MB, and the desktop
        # downloader gives the whole build 30s before it gives up waiting for headers.
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
        # Only the caller of a successful build knows to delete the temp file, so a failed one
        # has to take its own away.
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
        # Rows reference these by relative path, the way the publish path uploads them, so a bare
        # JSONL would hand over a multimodal dataset whose images are all missing.
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
