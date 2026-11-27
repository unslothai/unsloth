# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local dataset upload and listing services."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import threading
import uuid
from contextlib import suppress
from pathlib import Path
from fastapi import HTTPException, UploadFile

from hub.schemas.datasets import (
    LocalDatasetItem,
    LocalDatasetsResponse,
    UploadDatasetResponse,
)
from hub.utils.paths import dataset_uploads_root, ensure_dir, recipe_datasets_root
from utils.upload_limits import get_upload_limit_mb, upload_limit_bytes, upload_limit_label
from utils.paths.lazy import LazyPath
from utils.paths.storage_roots import own_entry, within_account
from utils.paths.path_utils import (
    any_not_appledouble_metadata,
    drop_appledouble_metadata,
    is_appledouble_metadata,
)

# Archives load as ImageFolder with synthetic columns, so prefer tabular formats.
_TABULAR_EXTS = (".parquet", ".json", ".jsonl", ".csv", ".tsv", ".arrow")
_ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".gz", ".zst", ".zip", ".txt")
DATA_EXTS = _TABULAR_EXTS + _ARCHIVE_EXTS
LOCAL_FILE_EXTS = (".json", ".jsonl", ".csv", ".parquet")
LOCAL_UPLOAD_EXTS = {".csv", ".json", ".jsonl", ".parquet"}
LOCAL_UPLOAD_CHUNK_BYTES = 1024 * 1024
LOCAL_DATASETS_ROOT = LazyPath(recipe_datasets_root)
DATASET_UPLOAD_DIR = LazyPath(dataset_uploads_root)
_REUSE_LOCK = threading.Lock()
_CSV_CONVERSION_LOCK = threading.Lock()
_PANDAS_JSON_MAX_MB = 8
_PANDAS_JSON_CONVERSION_LOCK = threading.Lock()


def _safe_read_metadata(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _safe_read_rows_from_metadata(payload: dict | None) -> int | None:
    if not payload:
        return None
    for key in ("actual_num_records", "target_num_records"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def _safe_read_metadata_summary(payload: dict | None) -> dict | None:
    if not payload:
        return None

    actual_num_records = (
        payload.get("actual_num_records")
        if isinstance(payload.get("actual_num_records"), int)
        else None
    )
    target_num_records = (
        payload.get("target_num_records")
        if isinstance(payload.get("target_num_records"), int)
        else actual_num_records
    )

    columns: list[str] | None = None
    schema = payload.get("schema")
    if isinstance(schema, dict):
        columns = [str(key) for key in schema.keys()]
    if not columns:
        stats = payload.get("column_statistics")
        if isinstance(stats, list):
            derived = [
                str(item.get("column_name"))
                for item in stats
                if isinstance(item, dict) and item.get("column_name")
            ]
            columns = derived or None

    parquet_files_count = None
    file_paths = payload.get("file_paths")
    if isinstance(file_paths, dict):
        parquet_files = file_paths.get("parquet-files")
        if isinstance(parquet_files, list):
            parquet_files_count = len(parquet_files)

    total_num_batches = (
        payload.get("total_num_batches")
        if isinstance(payload.get("total_num_batches"), int)
        else parquet_files_count
    )
    num_completed_batches = (
        payload.get("num_completed_batches")
        if isinstance(payload.get("num_completed_batches"), int)
        else total_num_batches
    )

    return {
        "actual_num_records": actual_num_records,
        "target_num_records": target_num_records,
        "total_num_batches": total_num_batches,
        "num_completed_batches": num_completed_batches,
        "columns": columns,
    }


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _is_upload_id(prefix: str) -> bool:
    return len(prefix) == 32 and all(c in "0123456789abcdef" for c in prefix)


def _display_uploaded_dataset_name(path: Path) -> str:
    stem = path.stem
    prefix, sep, rest = stem.partition("_")
    if sep and _is_upload_id(prefix):
        return f"{rest}{path.suffix}"
    return path.name


def _build_recipe_dataset_items() -> list[LocalDatasetItem]:
    if not LOCAL_DATASETS_ROOT.exists():
        return []

    items: list[LocalDatasetItem] = []
    for entry in LOCAL_DATASETS_ROOT.iterdir():
        if not entry.is_dir() or not entry.name.startswith("recipe_"):
            continue
        if not within_account(entry):
            continue
        parquet_dir = entry / "parquet-files"
        if not parquet_dir.exists() or not any_not_appledouble_metadata(
            parquet_dir.glob("*.parquet")
        ):
            continue

        rows = None
        metadata_summary = None
        metadata_path = entry / "metadata.json"
        if own_entry(metadata_path):
            metadata_payload = _safe_read_metadata(metadata_path)
            rows = _safe_read_rows_from_metadata(metadata_payload)
            metadata_summary = _safe_read_metadata_summary(metadata_payload)

        items.append(
            LocalDatasetItem(
                id = entry.name,
                label = entry.name,
                path = str(parquet_dir.resolve()),
                source = "recipe",
                rows = rows,
                updated_at = _safe_mtime(entry),
                metadata = metadata_summary,
            )
        )

    return items


def _build_uploaded_dataset_items() -> list[LocalDatasetItem]:
    if not DATASET_UPLOAD_DIR.exists():
        return []

    items: list[LocalDatasetItem] = []
    for path in DATASET_UPLOAD_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in LOCAL_UPLOAD_EXTS:
            continue
        if is_appledouble_metadata(path) or not within_account(path):
            continue
        try:
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue
        label = _display_uploaded_dataset_name(path)
        items.append(
            LocalDatasetItem(
                id = path.name,
                label = label,
                path = str(path.resolve()),
                source = "upload",
                updated_at = _safe_mtime(path),
            )
        )
    return items


def _build_local_dataset_items() -> list[LocalDatasetItem]:
    items = _build_recipe_dataset_items() + _build_uploaded_dataset_items()
    items.sort(key = lambda item: item.updated_at or 0, reverse = True)
    return items


# The datasets json loader types its first batch of this many bytes; the preview types the same window.
_JSONL_SCHEMA_BYTES = 10 << 20


def _first_significant_byte(f) -> bytes:
    """First byte that ``bytes.strip`` would keep, or ``b""`` for an empty / all-whitespace file."""
    while chunk := f.read(1 << 16):
        if stripped := chunk.lstrip():
            return stripped[:1]
    return b""


def _jsonl_head(path: Path, preview_size: int) -> tuple[list[dict], bytes] | None:
    """First ``preview_size`` JSON Lines objects and the bytes of the loader's first batch. ``None`` if not JSON Lines."""
    rows, keys, window, scanned = [], {}, [], 0
    with path.open("rb") as f:
        # Only an object decodes to a dict: a minified JSON array is one huge line, so don't read or parse it here.
        if preview_size > 0 and _first_significant_byte(f) not in (b"", b"{"):
            return None
        f.seek(0)
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line.decode("utf-8"))
                except ValueError:
                    row = None
                if not isinstance(row, dict):
                    # Within the preview rows the loader decides; past them the window ends.
                    if len(rows) < preview_size:
                        return None
                    break
                keys.update(dict.fromkeys(row))
                if len(rows) < preview_size:
                    rows.append(row)
            window.append(line)
            scanned += len(line)
            if len(rows) >= preview_size and scanned >= _JSONL_SCHEMA_BYTES:
                break
    # Columns are the union over the window, as the Arrow reader builds them (from_list reads row 0's keys).
    return [{key: row.get(key) for key in keys} for row in rows], b"".join(window)


def _jsonl_rows_dataset(
    rows: list[dict],
    window: bytes,
    dates_as_written: bool = False,
):
    import io

    import pyarrow as pa
    import pyarrow.json as paj
    from datasets import Dataset

    from utils.datasets.json_rows import has_timestamp, timestamps_as_strings

    try:
        # The loader's reader types as training does; one block, no threads: #15681 segfaults only threaded.
        options = paj.ReadOptions(use_threads = False, block_size = len(window))
        table = paj.read_json(io.BytesIO(window), read_options = options)
        if dates_as_written and has_timestamp(table.schema):
            # Local JSON training keeps date strings as written, so its preview does too.
            parse = paj.ParseOptions(explicit_schema = timestamps_as_strings(table.schema))
            table = paj.read_json(io.BytesIO(window), read_options = options, parse_options = parse)
        return Dataset.from_list(table.slice(0, len(rows)).to_pylist())
    except pa.ArrowException as error:
        # A field that is a string in one row and an object in another (#15681): show the rows as written.
        try:
            return Dataset.from_list(rows)
        except (TypeError, ValueError, OverflowError):
            raise error from None


def _stream_file_preview_slice(
    path: Path,
    preview_size: int,
    *,
    dates_as_written: bool = False,
    **load_kwargs,
):
    """Stream the first ``preview_size`` rows so a large file is never fully parsed into Arrow; returns ``(Dataset, None)`` or ``None`` if empty/unsupported."""
    from itertools import islice

    from datasets import Dataset, load_dataset

    name = path.name.lower()
    if name.endswith((".json", ".jsonl")):
        if not load_kwargs:
            head = _jsonl_head(path, preview_size)
            if head is not None:
                return (_jsonl_rows_dataset(*head, dates_as_written), None) if head[0] else None
        loader = "json"
    elif name.endswith((".csv", ".tsv")):
        loader = "csv"
    elif name.endswith(".parquet"):
        loader = "parquet"
    elif name.endswith(".arrow"):
        loader = "arrow"
    elif name.endswith(".txt"):
        loader = "text"
    else:
        return None

    streamed = load_dataset(
        loader,
        data_files = str(path),
        split = "train",
        streaming = True,
        **load_kwargs,
    )
    rows = list(islice(streamed, preview_size))
    if not rows:
        return None
    preview = Dataset.from_list(rows, features = load_kwargs.get("features"))
    if dates_as_written and loader == "json" and not load_kwargs:
        import pyarrow as pa
        from datasets import Features

        from utils.datasets.json_rows import has_timestamp, head_rows, timestamps_as_strings

        schema = preview.features.arrow_schema
        if has_timestamp(schema):
            # The streamed schema keeps the loader's columns and types; only dates read as written.
            try:
                written = head_rows(path, preview_size)
                preview = Dataset.from_list(
                    [{key: row.get(key) for key in schema.names} for row in written],
                    features = Features.from_arrow_schema(timestamps_as_strings(schema)),
                )
            except (
                OSError,
                ValueError,
                TypeError,
                OverflowError,
                AttributeError,
                pa.ArrowException,
            ):
                pass
    return preview, None


def _parquet_preview_slice(files: list[Path], preview_size: int):
    """Leading ``preview_size`` rows + footer row count, typed as the datasets Parquet builder types them,
    without the full load's cache write. ``None`` (full load decides) on differing schemas, no rows, or any error."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import Dataset, DatasetInfo, Features
        from datasets.table import InMemoryTable, table_cast

        schema, total_rows = None, 0
        for path in files:
            metadata = pq.read_metadata(path)
            file_schema = metadata.schema.to_arrow_schema()
            if schema is None:
                schema = file_schema
            elif not file_schema.equals(schema, check_metadata = True):
                # The full load casts later files to the first one's features or raises; leave both to it.
                return None
            total_rows += metadata.num_rows
        if total_rows <= 0:
            return None

        features = Features.from_arrow_schema(schema)
        arrow_schema = features.arrow_schema
        wanted = min(preview_size, total_rows)
        tables, decoded_rows = [], 0
        # Decode every page (damage past the preview rows must still fail), one row group per batch at most.
        for path in files:
            # A file object, so the handle is closed here (Windows would keep the file locked otherwise).
            with open(path, "rb") as handle:
                parquet_file = pq.ParquetFile(handle, pre_buffer = False, buffer_size = 1 << 20)
                metadata = parquet_file.metadata
                for index in range(metadata.num_row_groups):
                    batch_size = max(1, min(8192, metadata.row_group(index).num_rows))
                    for batch in parquet_file.iter_batches(
                        batch_size = batch_size, row_groups = [index], use_threads = False
                    ):
                        decoded_rows += batch.num_rows
                        if wanted > 0:
                            table = pa.Table.from_batches([batch.slice(0, wanted)])
                            tables.append(table_cast(table, arrow_schema))
                            wanted -= tables[-1].num_rows
        if decoded_rows != total_rows or wanted > 0:
            return None
        table = pa.concat_tables(tables).combine_chunks()
        # The full load writes its rows under the features' own Arrow schema (list child names included).
        table = table.cast(arrow_schema)
        return Dataset(InMemoryTable(table), info = DatasetInfo(features = features)), total_rows
    except Exception:
        return None


def _load_local_preview_slice(*, dataset_path: Path, train_split: str, preview_size: int):
    # Non-streaming loads take the cached builder lock; use the EACCES-safe wrapper.
    from utils.datasets.cache_safe import load_dataset_cache_safe as load_dataset
    from utils.datasets.cells import csv_as_text_kwargs

    single_file = True
    if dataset_path.is_dir():
        from utils.paths import dataset_files_in_dir

        try:
            files = dataset_files_in_dir(dataset_path)
        except ValueError:
            raise HTTPException(
                status_code = 400,
                detail = "Unsupported local dataset directory (expected parquet/json/jsonl/csv files)",
            ) from None
        if files[0].suffix.lower() == ".parquet":
            if train_split == "train":
                preview = _parquet_preview_slice(files, preview_size)
                if preview is not None:
                    return preview
            dataset = load_dataset(
                "parquet",
                data_files = [str(path) for path in files],
                split = train_split,
            )
            total_rows = len(dataset)
            preview_slice = dataset.select(range(min(preview_size, total_rows)))
            return preview_slice, total_rows
        single_file = len(files) == 1
        dataset_path = files[0]

    suffix = dataset_path.suffix.lower()
    # Parquet/Arrow give a cheap exact total_rows from their metadata; JSON/JSONL/CSV stream the preview,
    # then a bounded scan of one file counts its rows (None over the size caps or for several files).
    if suffix == ".parquet":
        if train_split == "train":
            preview = _parquet_preview_slice([dataset_path], preview_size)
            if preview is not None:
                return preview
        dataset = load_dataset("parquet", data_files = str(dataset_path), split = train_split)
        total_rows = len(dataset)
        preview_slice = dataset.select(range(min(preview_size, total_rows)))
        return preview_slice, total_rows

    if suffix in (".json", ".jsonl", ".csv"):
        load_kwargs = csv_as_text_kwargs([dataset_path])
        if "features" in load_kwargs:
            # String features fix every column type; csv_as_text_kwargs already parsed 10k rows, so bad rows still raise.
            load_kwargs["chunksize"] = max(1, preview_size)
        preview = _stream_file_preview_slice(
            dataset_path,
            preview_size,
            dates_as_written = True,
            **load_kwargs,
        )
        if preview is None:
            raise HTTPException(
                status_code = 400,
                detail = "Dataset appears to be empty or could not be read",
            )
        preview_slice, total_rows = preview
        if total_rows is None and single_file and train_split == "train":
            from hub.services.datasets.row_counts import count_local_file_rows
            total_rows = count_local_file_rows(dataset_path)
        return preview_slice, total_rows

    raise HTTPException(status_code = 400, detail = f"Unsupported file format: {dataset_path.suffix}")


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "dataset_upload"
    return name


def _upload_too_large(limit_label: str) -> HTTPException:
    return HTTPException(
        status_code = 413,
        detail = f"Training dataset upload too large. Maximum is {limit_label}.",
    )


def _upload_destination(filename: str) -> tuple[str, Path, int, str]:
    filename = _sanitize_filename(filename)
    ext = Path(filename).suffix.lower()
    if ext not in LOCAL_UPLOAD_EXTS:
        allowed = ", ".join(sorted(LOCAL_UPLOAD_EXTS))
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported file type: {ext}. Allowed: {allowed}",
        )

    limit_mb = get_upload_limit_mb()
    max_bytes = upload_limit_bytes(limit_mb)
    max_label = upload_limit_label(limit_mb)
    ensure_dir(DATASET_UPLOAD_DIR)
    stem = Path(filename).stem
    stored_name = f"{uuid.uuid4().hex}_{stem}{ext}"
    return filename, DATASET_UPLOAD_DIR / stored_name, max_bytes, max_label


def _native_upload_dataset_response(
    native_path_lease: str,
    json_columns: bool = False,
    csv_delimiter: str = ",",
) -> UploadDatasetResponse:
    from core.training.account_jobs import account_path
    from utils.native_path_leases import NativePathLeaseError, verify_native_path_lease

    try:
        grant = verify_native_path_lease(
            native_path_lease,
            operation = "dataset-import",
            expected_kind = "dataset",
            expected_path_type = "file",
            allowed_suffixes = sorted(LOCAL_UPLOAD_EXTS),
        )
    except NativePathLeaseError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc

    # A managed account may only import files inside its own workspace, as the RAG/seed/library drops check.
    account_path(grant.canonical_path)
    filename, stored_path, max_bytes, max_label = _upload_destination(grant.canonical_path.name)
    _validate_csv_delimiter(stored_path, csv_delimiter)
    _validate_json_columns_upload(stored_path, json_columns)
    input_bytes, input_label = _json_columns_input_limit(max_bytes, max_label, json_columns)
    if grant.size_bytes is not None and grant.size_bytes > input_bytes:
        raise _upload_too_large(input_label)

    partial_path = _partial_upload_path(stored_path)
    written = 0
    upload_complete = False
    try:
        with open(grant.canonical_path, "rb") as source, open(partial_path, "wb") as target:
            while chunk := source.read(LOCAL_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > input_bytes:
                    raise _upload_too_large(input_label)
                target.write(chunk)
        upload_complete = True
    except OSError as exc:
        raise HTTPException(
            status_code = 400,
            detail = "Dropped dataset could not be read.",
        ) from exc
    finally:
        if not upload_complete:
            with suppress(OSError):
                partial_path.unlink(missing_ok = True)

    if written == 0:
        partial_path.unlink(missing_ok = True)
        raise HTTPException(status_code = 400, detail = "Dropped dataset is empty")

    published = _finish_upload(
        partial_path, stored_path, max_bytes, max_label, json_columns, csv_delimiter
    )
    return UploadDatasetResponse(filename = filename, stored_path = str(published))


async def upload_dataset_response(
    file: UploadFile | None,
    native_path_lease: str | None = None,
    *,
    json_columns: bool = False,
    csv_delimiter: str = ",",
) -> UploadDatasetResponse:
    if native_path_lease:
        return await asyncio.to_thread(
            _native_upload_dataset_response,
            native_path_lease,
            json_columns,
            csv_delimiter,
        )
    if file is None:
        raise HTTPException(status_code = 400, detail = "No dataset file was provided")

    filename, stored_path, max_bytes, max_label = _upload_destination(
        file.filename or "dataset_upload"
    )
    _validate_csv_delimiter(stored_path, csv_delimiter)
    _validate_json_columns_upload(stored_path, json_columns)
    input_bytes, input_label = _json_columns_input_limit(max_bytes, max_label, json_columns)
    declared_size = getattr(file, "size", None)
    if isinstance(declared_size, int) and declared_size > input_bytes:
        raise _upload_too_large(input_label)

    partial_path = _partial_upload_path(stored_path)
    written = 0
    upload_complete = False
    try:
        with open(partial_path, "wb") as f:
            while chunk := await file.read(LOCAL_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > input_bytes:
                    raise _upload_too_large(input_label)
                await asyncio.to_thread(f.write, chunk)
        upload_complete = True
    finally:
        if not upload_complete:
            with suppress(OSError):
                partial_path.unlink(missing_ok = True)

    if written == 0:
        partial_path.unlink(missing_ok = True)
        raise HTTPException(status_code = 400, detail = "Empty upload payload")

    reused_path = await asyncio.to_thread(
        _finish_upload,
        partial_path,
        stored_path,
        max_bytes,
        max_label,
        json_columns,
        csv_delimiter,
    )
    return UploadDatasetResponse(filename = filename, stored_path = str(reused_path))


def _validate_csv_delimiter(path: Path, csv_delimiter: str) -> None:
    if csv_delimiter != "," and (csv_delimiter != ";" or path.suffix != ".csv"):
        raise HTTPException(status_code = 400, detail = "Semicolon import requires a .csv file.")


def _validate_json_columns_upload(path: Path, json_columns: bool) -> None:
    if json_columns and path.suffix != ".json":
        raise HTTPException(status_code = 400, detail = "Pandas column import requires a .json file.")


def _json_columns_input_limit(
    max_bytes: int, max_label: str, json_columns: bool
) -> tuple[int, str]:
    # Decoding a column object keeps every cell in memory until the transpose is written.
    pandas_limit = upload_limit_bytes(_PANDAS_JSON_MAX_MB)
    if json_columns and pandas_limit < max_bytes:
        return pandas_limit, f"{upload_limit_label(_PANDAS_JSON_MAX_MB)} for pandas column JSON"
    return max_bytes, max_label


def _unique_json_object(items):
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("JSON objects must have unique keys.")
    return result


def _finish_upload(
    partial_path: Path,
    stored_path: Path,
    max_bytes: int,
    max_label: str,
    json_columns: bool,
    csv_delimiter: str,
) -> Path:
    if json_columns:
        return _finish_pandas_json_upload(partial_path, stored_path, max_bytes, max_label)
    if csv_delimiter != ",":
        return _finish_semicolon_csv_upload(
            partial_path, stored_path, max_bytes, max_label, csv_delimiter
        )
    return _publish_upload(partial_path, stored_path)


def _finish_semicolon_csv_upload(
    partial_path: Path, stored_path: Path, max_bytes: int, max_label: str, csv_delimiter: str
) -> Path:
    converted_path = partial_path.with_suffix(".csv.part")
    try:
        with _CSV_CONVERSION_LOCK:
            field_limit = csv.field_size_limit()
            try:
                source_bytes = partial_path.stat().st_size
                if source_bytes > field_limit:
                    try:
                        csv.field_size_limit(source_bytes)
                    except OverflowError:
                        # Python versions that store this as C long cap it at 2 GiB on Windows.
                        csv.field_size_limit(2**31 - 1)
                _convert_semicolon_csv(
                    partial_path, converted_path, max_bytes, max_label, csv_delimiter
                )
            finally:
                csv.field_size_limit(field_limit)
        return _publish_upload(converted_path, stored_path)
    except UnicodeDecodeError as exc:
        # The ANSI code page depends on the exporting locale, so a guessed one would publish mojibake.
        raise HTTPException(
            status_code = 400,
            detail = "Invalid semicolon CSV. The file is not UTF-8; save it from Excel as CSV UTF-8.",
        ) from exc
    except (csv.Error, ValueError) as exc:
        raise HTTPException(status_code = 400, detail = f"Invalid semicolon CSV. {exc}") from exc
    finally:
        with suppress(OSError):
            partial_path.unlink(missing_ok = True)
        with suppress(OSError):
            converted_path.unlink(missing_ok = True)


def _finish_pandas_json_upload(
    partial_path: Path, stored_path: Path, max_bytes: int, max_label: str
) -> Path:
    converted_path = partial_path.with_suffix(".rows.part")
    try:
        with _PANDAS_JSON_CONVERSION_LOCK:
            _write_pandas_json_rows(partial_path, converted_path, max_bytes, max_label)
        destination = stored_path.with_suffix(".jsonl")
        if len(destination.name.encode("utf-8")) > 255:
            destination = destination.with_name(f"{destination.stem[:-1]}.jsonl")
        return _publish_upload(converted_path, destination)
    except (ValueError, RecursionError) as exc:
        raise HTTPException(status_code = 400, detail = f"Invalid pandas column JSON. {exc}") from exc
    finally:
        with suppress(OSError):
            partial_path.unlink(missing_ok = True)
        with suppress(OSError):
            converted_path.unlink(missing_ok = True)


def _csv_records(source, delimiter):
    last_line = ""

    def lines():
        nonlocal last_line
        for line in source:
            last_line = line
            yield line

    reader = csv.reader(lines(), delimiter = delimiter, strict = True)
    end = 0
    for row in reader:
        line_count = reader.line_num - end
        end = reader.line_num
        if line_count == 1 and not last_line.strip():
            continue
        yield row


class _ByteCountingWriter:
    __slots__ = ("file", "written")

    def __init__(self, file):
        self.file = file
        self.written = 0

    def write(self, text):
        data = text.encode("utf-8")
        self.written += len(data)
        return self.file.write(data)


def _convert_semicolon_csv(partial_path, converted_path, max_bytes, max_label, delimiter):
    with (
        partial_path.open(encoding = "utf-8-sig", newline = "") as source,
        converted_path.open("wb") as target,
    ):
        # Excel also accepts a delimiter directive before the header.
        if source.readline().rstrip("\r\n").lower() != "sep=;":
            source.seek(0)
        rows = _csv_records(source, delimiter)
        header = next(rows, None)
        if header is None:
            raise ValueError("Missing CSV header.")
        out = _ByteCountingWriter(target)
        writer = csv.writer(out, lineterminator = "\r\n")
        writer.writerow(header)
        for row in rows:
            missing = len(header) - len(row)
            if missing < 0:
                raise ValueError("A row has more fields than the header.")
            writer.writerow(row + [""] * missing if missing else row)
            if out.written > max_bytes:
                raise _upload_too_large(max_label)
        if out.written > max_bytes:
            raise _upload_too_large(max_label)


def _write_pandas_json_rows(
    source_path: Path, converted_path: Path, max_bytes: int, max_label: str
) -> None:
    with source_path.open(encoding = "utf-8-sig") as source:
        columns = json.load(source, object_pairs_hook = _unique_json_object)
    if not isinstance(columns, dict) or not columns:
        raise ValueError("Expected an object containing named columns.")
    index = next(iter(columns.values()))
    if not isinstance(index, dict) or not index:
        raise ValueError("Expected a non-empty object of row indices in each column.")
    if any(
        not isinstance(values, dict) or values.keys() != index.keys() for values in columns.values()
    ):
        raise ValueError("Every column must contain the same row indices.")

    with converted_path.open("wb") as target:
        for key in index:
            row = {name: values[key] for name, values in columns.items()}
            encoded = json.dumps(
                row, ensure_ascii = False, allow_nan = False, separators = (",", ":")
            ).encode("utf-8")
            if target.tell() + len(encoded) + 1 > max_bytes:
                raise _upload_too_large(max_label)
            target.write(encoded + b"\n")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(LOCAL_UPLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _partial_upload_path(stored_path: Path) -> Path:
    # Only the upload id: appending to a long original name can exceed the 255-byte filename limit.
    return stored_path.with_name(f"{stored_path.name.split('_', 1)[0]}.part")


def _publish_upload(partial_path: Path, stored_path: Path) -> Path:
    try:
        with _REUSE_LOCK:
            existing = _find_identical_upload(partial_path, stored_path)
            if existing is not None:
                return existing
            partial_path.replace(stored_path)
            return stored_path
    finally:
        with suppress(OSError):
            partial_path.unlink(missing_ok = True)


def _find_identical_upload(partial_path: Path, stored_path: Path) -> Path | None:
    suffix = stored_path.name.split("_", 1)[-1]
    try:
        size = partial_path.stat().st_size
        candidates = [
            path
            for path in DATASET_UPLOAD_DIR.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and within_account(path)
            and _is_upload_id(path.name.split("_", 1)[0])
            and path.name.split("_", 1)[-1] == suffix
            and path.stat().st_size == size
        ]
        if not candidates:
            return None
        new_digest = _file_digest(partial_path)
        for existing in candidates:
            if _file_digest(existing) == new_digest:
                return existing
    except OSError:
        pass
    return None


def delete_uploaded_dataset_response(dataset_id: str) -> dict:
    name = Path(dataset_id or "").name
    if not name or name != dataset_id or Path(name).suffix.lower() not in LOCAL_UPLOAD_EXTS:
        raise HTTPException(status_code = 400, detail = "Invalid uploaded dataset id")
    path = DATASET_UPLOAD_DIR / name
    with _REUSE_LOCK:
        if not path.is_file() or path.is_symlink() or not within_account(path):
            raise HTTPException(status_code = 404, detail = "Uploaded dataset not found")
        try:
            path.unlink()
        except OSError as exc:
            raise HTTPException(
                status_code = 500, detail = "Could not delete the uploaded dataset"
            ) from exc
    return {"status": "deleted", "id": name}


def list_local_datasets_response() -> LocalDatasetsResponse:
    return LocalDatasetsResponse(datasets = _build_local_dataset_items())
