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

_JSONL_EXPORT_BATCH_SIZE = 5_000
_PARQUET_ROW_ORDER_SQL = (
    "SELECT * EXCLUDE (__row_num__) FROM ("
    "SELECT *, row_number() OVER (PARTITION BY filename) AS __row_num__ "
    "FROM read_parquet(?, filename=true)"
    ") ORDER BY filename, __row_num__ "
    "LIMIT ? OFFSET ?"
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


def _drop_parquet_helper_columns(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"filename", "__row_num__"}
    }


def _write_jsonl_rows(handle, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        handle.write(_json_dumps_row(row))
        handle.write("\n")


def _parquet_glob(parquet_dir: Path) -> str:
    return str((parquet_dir / "*.parquet").resolve())


def _count_parquet_rows_with_duckdb(parquet_glob: str) -> int | None:
    try:
        import duckdb  # type: ignore
    except Exception:
        return None

    try:
        conn = duckdb.connect(":memory:")
        try:
            total_row = conn.execute(
                "SELECT COUNT(*) FROM read_parquet(?)",
                [parquet_glob],
            ).fetchone()
        finally:
            conn.close()
    except (RuntimeError, ValueError, duckdb.Error):
        return None
    return int(total_row[0] if total_row else 0)


def _fetch_parquet_page_with_duckdb(
    *,
    parquet_glob: str,
    limit: int,
    offset: int,
) -> list[dict[str, Any]] | None:
    try:
        import duckdb  # type: ignore
    except Exception:
        return None

    try:
        conn = duckdb.connect(":memory:")
        try:
            dataframe = conn.execute(
                _PARQUET_ROW_ORDER_SQL,
                [parquet_glob, int(limit), int(offset)],
            ).fetchdf()
        finally:
            conn.close()
    except (RuntimeError, ValueError, duckdb.Error):
        return None

    rows = dataframe.to_dict(orient = "records")
    return [_drop_parquet_helper_columns(to_preview_jsonable(row)) for row in rows]


def _stream_jsonl_from_parquet_with_duckdb(
    *,
    parquet_dir: Path,
    destination: Path,
) -> bool:
    parquet_glob = _parquet_glob(parquet_dir)
    total = _count_parquet_rows_with_duckdb(parquet_glob)
    if total is None:
        return False

    offset = 0
    with destination.open("w", encoding = "utf-8") as handle:
        while offset < total:
            page = _fetch_parquet_page_with_duckdb(
                parquet_glob = parquet_glob,
                limit = _JSONL_EXPORT_BATCH_SIZE,
                offset = offset,
            )
            if page is None:
                return False
            if not page:
                break
            _write_jsonl_rows(handle, page)
            offset += len(page)
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
    return [_drop_parquet_helper_columns(to_preview_jsonable(row)) for row in rows]


def _read_all_rows_with_data_designer(parquet_dir: Path) -> list[dict[str, Any]]:
    from data_designer.config.utils.io_helpers import read_parquet_dataset

    dataframe = read_parquet_dataset(parquet_dir)
    rows = dataframe.to_dict(orient = "records")
    return [_drop_parquet_helper_columns(to_preview_jsonable(row)) for row in rows]


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


def _write_jsonl_file(rows: list[dict[str, Any]], destination: Path) -> None:
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
            _add_images_to_archive(archive, dataset_path)
        return zip_path, "application/zip", f"{stem}.parquet.zip"

    tmp = tempfile.NamedTemporaryFile(delete = False, suffix = ".jsonl")
    tmp.close()
    jsonl_path = Path(tmp.name)
    _write_jsonl_from_parquet(parquet_dir, jsonl_path)
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
