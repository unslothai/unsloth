"""
Dataset-related Pydantic models for API requests and responses.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CheckFormatRequest(BaseModel):
    """Request for dataset format check"""
    dataset_name: str  # HuggingFace dataset name or local path
    is_vlm: bool = False
    hf_token: Optional[str] = None
    subset: Optional[str] = None
    train_split: Optional[str] = "train"

    @model_validator(mode="before")
    @classmethod
    def _compat_split(cls, values: Any) -> Any:
        """Accept legacy 'split' field as alias for 'train_split'."""
        if isinstance(values, dict) and "split" in values:
            values.setdefault("train_split", values.pop("split"))
        return values


class CheckFormatResponse(BaseModel):
    """Response for dataset format check"""
    requires_manual_mapping: bool
    detected_format: str
    columns: List[str]
    is_image: bool = False
    is_audio: bool = False
    multimodal_columns: Optional[List[str]] = None
    suggested_mapping: Optional[Dict[str, str]] = None
    detected_image_column: Optional[str] = None
    detected_audio_column: Optional[str] = None
    detected_text_column: Optional[str] = None
    detected_speaker_column: Optional[str] = None
    preview_samples: Optional[List[Dict]] = None
    total_rows: Optional[int] = None
    warning: Optional[str] = None


class UploadDatasetResponse(BaseModel):
    """Response with stored dataset path for training."""
    filename: str = Field(..., description="Original filename")
    stored_path: str = Field(..., description="Absolute path stored on backend")


class LocalDatasetItem(BaseModel):
    class Metadata(BaseModel):
        actual_num_records: Optional[int] = None
        target_num_records: Optional[int] = None
        total_num_batches: Optional[int] = None
        num_completed_batches: Optional[int] = None
        columns: Optional[List[str]] = None

    id: str
    label: str
    path: str
    rows: Optional[int] = None
    updated_at: Optional[float] = None
    metadata: Optional[Metadata] = None


class LocalDatasetsResponse(BaseModel):
    datasets: List[LocalDatasetItem] = Field(default_factory=list)
