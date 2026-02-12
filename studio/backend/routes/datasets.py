"""
Datasets API routes
"""
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


# --- Endpoints ---

@router.post("/check-format", response_model=CheckFormatResponse)
async def check_format(request: CheckFormatRequest):
    """
    Check if a dataset requires manual column mapping.
    
    This is a lightweight check that loads only the first 10 rows,
    runs format detection, and (if processable) returns processed
    preview samples. The full dataset is re-processed at training time.
    """
    try:
        from datasets import load_dataset
        from utils.datasets import format_dataset
        
        PREVIEW_SIZE = 10
        
        logger.info(f"Checking format for dataset: {request.dataset_name}")
        
        # Load dataset
        dataset_path = Path(request.dataset_name)
        
        if dataset_path.exists():
            # Local dataset
            if dataset_path.suffix in ['.json', '.jsonl']:
                dataset = load_dataset('json', data_files=str(dataset_path), split=request.split)
            elif dataset_path.suffix == '.csv':
                dataset = load_dataset('csv', data_files=str(dataset_path), split=request.split)
            elif dataset_path.suffix == '.parquet':
                dataset = load_dataset('parquet', data_files=str(dataset_path), split=request.split)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {dataset_path.suffix}"
                )
        else:
            # HuggingFace dataset
            load_kwargs = {"path": request.dataset_name, "split": request.split}
            if request.hf_token:
                load_kwargs["token"] = request.hf_token
            dataset = load_dataset(**load_kwargs)
        
        # Slice to top N rows — all detection and preview runs on this subset
        total_rows = len(dataset)
        preview_slice = dataset.select(range(min(PREVIEW_SIZE, total_rows)))
        
        # Run lightweight format check on the preview slice
        result = check_dataset_format(preview_slice, is_vlm=request.is_vlm)
        
        logger.info(f"Format check result: requires_mapping={result['requires_manual_mapping']}, format={result['detected_format']}")
        
        # Generate preview samples
        preview_samples = None
        if not result["requires_manual_mapping"]:
            # Format detected — return processed preview
            try:
                format_result = format_dataset(
                    preview_slice,
                    format_type="auto",
                    custom_format_mapping=result.get("suggested_mapping"),
                )
                processed = format_result["dataset"]
                preview_samples = [dict(row) for row in processed]
            except Exception as e:
                logger.warning(f"Processed preview generation failed (non-fatal): {e}")
                # Fall back to raw samples so frontend still has something
                preview_samples = [dict(row) for row in preview_slice]
        else:
            # Format detection failed — return raw samples so user can
            # see actual data and map columns in the frontend
            preview_samples = [dict(row) for row in preview_slice]
        
        return CheckFormatResponse(
            requires_manual_mapping=result["requires_manual_mapping"],
            detected_format=result["detected_format"],
            columns=result["columns"],
            suggested_mapping=result.get("suggested_mapping"),
            detected_image_column=result.get("detected_image_column"),
            detected_text_column=result.get("detected_text_column"),
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
