"""
Datasets API routes
"""
import base64
import io
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
import logging

# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import dataset utilities
from utils.datasets import check_dataset_format

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


from models.datasets import CheckFormatRequest, CheckFormatResponse


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


@router.post("/check-format", response_model=CheckFormatResponse)
def check_format(request: CheckFormatRequest):
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
            if dataset_path.suffix in ['.json', '.jsonl']:
                dataset = load_dataset('json', data_files=str(dataset_path), split=request.train_split)
            elif dataset_path.suffix == '.csv':
                dataset = load_dataset('csv', data_files=str(dataset_path), split=request.train_split)
            elif dataset_path.suffix == '.parquet':
                dataset = load_dataset('parquet', data_files=str(dataset_path), split=request.train_split)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {dataset_path.suffix}"
                )
            total_rows = len(dataset)
            preview_slice = dataset.select(range(min(PREVIEW_SIZE, total_rows)))
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
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking dataset format: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check dataset format: {str(e)}"
        )
