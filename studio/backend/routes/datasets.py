"""
Datasets API routes
"""
import base64
import io
import json
import sys
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, UploadFile
import logging

# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import dataset utilities
from utils.datasets import check_dataset_format
from auth.authentication import get_current_subject

router = APIRouter()
logger = logging.getLogger(__name__)

# Configure logger
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


from models.datasets import (
    CheckFormatRequest,
    CheckFormatResponse,
    LocalDatasetItem,
    LocalDatasetsResponse,
    UploadDatasetResponse,
)


def _serialize_preview_value(value):
    """make it json safe for client preview ⊂(◉‿◉)つ"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    try:
        from PIL.Image import Image as PILImage
        if isinstance(value, PILImage):
            buffer = io.BytesIO()
            value.convert("RGB").save(buffer, format="JPEG", quality=85)
            return {
                "type": "image",
                "mime": "image/jpeg",
                "width": value.width,
                "height": value.height,
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(key): _serialize_preview_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_preview_value(item) for item in value]

    return str(value)


def _serialize_preview_rows(rows):
    return [
        {str(key): _serialize_preview_value(value) for key, value in dict(row).items()}
        for row in rows
    ]


# --- Endpoints ---

# Recognized data-file extensions for the single-file fallback approach.
DATA_EXTS = (
    '.parquet',
    '.json', '.jsonl',
    '.csv', '.tsv',
    '.txt',
    '.arrow',
    '.tar', '.tar.gz', '.tgz',
    '.gz', '.zst',
    '.zip',
)
LOCAL_FILE_EXTS = ('.json', '.jsonl', '.csv', '.parquet')
LOCAL_UPLOAD_EXTS = {".csv", ".json", ".jsonl", ".parquet"}
DATASET_UPLOAD_DIR = (
    Path.home() / ".cache" / "unsloth" / "training" / "dataset-uploads"
)
BACKEND_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATASETS_ROOT = BACKEND_ROOT / "assets" / "datasets"


def _safe_read_metadata(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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


def _build_local_dataset_items() -> list[LocalDatasetItem]:
    if not LOCAL_DATASETS_ROOT.exists():
        return []

    items: list[LocalDatasetItem] = []
    for entry in LOCAL_DATASETS_ROOT.iterdir():
        if not entry.is_dir() or not entry.name.startswith("recipe_"):
            continue
        parquet_dir = entry / "parquet-files"
        if not parquet_dir.exists() or not any(parquet_dir.glob("*.parquet")):
            continue

        rows = None
        metadata_summary = None
        metadata_path = entry / "metadata.json"
        if metadata_path.exists():
            metadata_payload = _safe_read_metadata(metadata_path)
            rows = _safe_read_rows_from_metadata(metadata_payload)
            metadata_summary = _safe_read_metadata_summary(metadata_payload)

        try:
            updated_at = entry.stat().st_mtime
        except OSError:
            updated_at = None

        items.append(
            LocalDatasetItem(
                id=entry.name,
                label=entry.name,
                path=str(parquet_dir.resolve()),
                rows=rows,
                updated_at=updated_at,
                metadata=metadata_summary,
            )
        )

    items.sort(key=lambda item: item.updated_at or 0, reverse=True)
    return items


def _load_local_preview_slice(*, dataset_path: Path, train_split: str, preview_size: int):
    from datasets import load_dataset

    if dataset_path.is_dir():
        parquet_dir = dataset_path / "parquet-files" if (dataset_path / "parquet-files").exists() else dataset_path
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        if parquet_files:
            dataset = load_dataset(
                "parquet",
                data_files=[str(path) for path in parquet_files],
                split=train_split,
            )
            total_rows = len(dataset)
            preview_slice = dataset.select(range(min(preview_size, total_rows)))
            return preview_slice, total_rows
        else:
            candidate_files: list[Path] = []
            for ext in LOCAL_FILE_EXTS:
                candidate_files.extend(sorted(dataset_path.glob(f"*{ext}")))
            if not candidate_files:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported local dataset directory (expected parquet/json/jsonl/csv files)",
                )
            dataset_path = candidate_files[0]

    if dataset_path.suffix in ['.json', '.jsonl']:
        dataset = load_dataset('json', data_files=str(dataset_path), split=train_split)
    elif dataset_path.suffix == '.csv':
        dataset = load_dataset('csv', data_files=str(dataset_path), split=train_split)
    elif dataset_path.suffix == '.parquet':
        dataset = load_dataset('parquet', data_files=str(dataset_path), split=train_split)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {dataset_path.suffix}"
        )

    total_rows = len(dataset)
    preview_slice = dataset.select(range(min(preview_size, total_rows)))
    return preview_slice, total_rows


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name:
        return "dataset_upload"
    return name


@router.post("/upload", response_model=UploadDatasetResponse)
async def upload_dataset(
    file: UploadFile,
    current_subject: str = Depends(get_current_subject),
) -> UploadDatasetResponse:
    filename = _sanitize_filename(file.filename or "dataset_upload")
    ext = Path(filename).suffix.lower()
    if ext not in LOCAL_UPLOAD_EXTS:
        allowed = ", ".join(sorted(LOCAL_UPLOAD_EXTS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed}",
        )

    max_size_bytes = 512 * 1024 * 1024
    DATASET_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    stored_name = f"{uuid4().hex}_{stem}{ext}"
    stored_path = DATASET_UPLOAD_DIR / stored_name

    # Stream file to disk in chunks to avoid holding entire file in memory
    size = 0
    with open(stored_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_size_bytes:
                stored_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (max 512MB)")
            f.write(chunk)

    if size == 0:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload payload")

    return UploadDatasetResponse(filename=filename, stored_path=str(stored_path))


@router.get("/local", response_model=LocalDatasetsResponse)
def list_local_datasets(
    current_subject: str = Depends(get_current_subject),
) -> LocalDatasetsResponse:
    return LocalDatasetsResponse(datasets=_build_local_dataset_items())


@router.post("/check-format", response_model=CheckFormatResponse)
def check_format(
    request: CheckFormatRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a dataset requires manual column mapping.

    Strategy for HuggingFace datasets:
      1. list_repo_files → pick the first data file → load_dataset(data_files=[…])
         Avoids resolving thousands of files; typically ~2-4 s.
      2. Full streaming load_dataset as a last-resort fallback.

    Local files are loaded directly.

    Using a plain `def` (not async) so FastAPI runs this in a thread-pool,
    preventing any blocking IO from freezing the event loop.
    """
    try:
        from itertools import islice
        from datasets import Dataset, load_dataset
        from utils.datasets import format_dataset

        PREVIEW_SIZE = 10

        logger.info(f"Checking format for dataset: {request.dataset_name}")

        dataset_path = Path(request.dataset_name)
        total_rows = None

        if dataset_path.exists():
            # ── Local file ──────────────────────────────────────────
            train_split = request.train_split or "train"
            preview_slice, total_rows = _load_local_preview_slice(
                dataset_path=dataset_path,
                train_split=train_split,
                preview_size=PREVIEW_SIZE,
            )
        else:
            # ── HuggingFace dataset ─────────────────────────────────
            # Tier 1: list_repo_files → load only the first data file
            preview_slice = None

            try:
                from huggingface_hub import HfApi
                api = HfApi()
                repo_files = api.list_repo_files(
                    request.dataset_name,
                    repo_type="dataset",
                    token=request.hf_token or None,
                )
                data_files = [f for f in repo_files if any(f.endswith(ext) for ext in DATA_EXTS)]

                if data_files:
                    first_file = data_files[0]
                    logger.info(f"Tier 1: loading single file {first_file}")
                    load_kwargs = {
                        "path": request.dataset_name,
                        "data_files": [first_file],
                        "split": "train",
                        "streaming": True,
                    }
                    if request.hf_token:
                        load_kwargs["token"] = request.hf_token

                    streamed_ds = load_dataset(**load_kwargs)
                    rows = list(islice(streamed_ds, PREVIEW_SIZE))
                    if rows:
                        preview_slice = Dataset.from_list(rows)
            except Exception as e:
                logger.warning(f"Tier 1 (single-file) failed: {e}")

            if preview_slice is None:
                # Tier 2: full streaming (resolves all files — slow for large repos)
                logger.info("Tier 2: falling back to full streaming load_dataset")
                load_kwargs = {"path": request.dataset_name, "split": request.train_split, "streaming": True}
                if request.subset:
                    load_kwargs["name"] = request.subset
                if request.hf_token:
                    load_kwargs["token"] = request.hf_token

                streamed_ds = load_dataset(**load_kwargs)

                rows = list(islice(streamed_ds, PREVIEW_SIZE))
                if not rows:
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset appears to be empty or could not be streamed"
                    )

                preview_slice = Dataset.from_list(rows)
            total_rows = None

        # Run lightweight format check on the preview slice
        result = check_dataset_format(preview_slice, is_vlm=request.is_vlm)

        logger.info(f"Format check result: requires_mapping={result['requires_manual_mapping']}, format={result['detected_format']}, is_image={result.get('is_image', False)}")

        # Generate preview samples
        preview_samples = None
        if not result["requires_manual_mapping"]:
            if result.get("suggested_mapping"):
                # Heuristic-detected: show raw data so columns match the API response.
                # Processing (column stripping) happens at training time, not preview.
                preview_samples = _serialize_preview_rows(preview_slice)
            else:
                try:
                    format_result = format_dataset(
                        preview_slice,
                        format_type="auto",
                        num_proc=1,  # Only 10 preview rows — no need for multiprocessing
                    )
                    processed = format_result["dataset"]
                    preview_samples = _serialize_preview_rows(processed)
                except Exception as e:
                    logger.warning(f"Processed preview generation failed (non-fatal): {e}")
                    preview_samples = _serialize_preview_rows(preview_slice)
        else:
            preview_samples = _serialize_preview_rows(preview_slice)

        # Lightweight URL-based image detection for VLM datasets
        warning = None
        image_col = result.get("detected_image_column")
        if image_col and image_col in (result.get("columns") or []):
            try:
                sample_val = preview_slice[0][image_col]
                if isinstance(sample_val, str) and sample_val.startswith(("http://", "https://")):
                    warning = (
                        "This dataset contains image URLs instead of embedded images. "
                        "Images will be downloaded during training, which may be slow for large datasets."
                    )
                    logger.info(f"URL-based image column detected: {image_col}")
            except Exception:
                pass

        return CheckFormatResponse(
            requires_manual_mapping=result["requires_manual_mapping"],
            detected_format=result["detected_format"],
            columns=result["columns"],
            is_image=result.get("is_image", False),
            is_audio=result.get("is_audio", False),
            multimodal_columns=result.get("multimodal_columns"),
            suggested_mapping=result.get("suggested_mapping"),
            detected_image_column=result.get("detected_image_column"),
            detected_audio_column=result.get("detected_audio_column"),
            detected_text_column=result.get("detected_text_column"),
            detected_speaker_column=result.get("detected_speaker_column"),
            preview_samples=preview_samples,
            total_rows=total_rows,
            warning=warning,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking dataset format: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check dataset format: {str(e)}"
        )
