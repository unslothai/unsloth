"""
Dataset-related Pydantic models for API requests and responses.
"""
from pydantic import BaseModel
from typing import Optional, Dict, List


class CheckFormatRequest(BaseModel):
    """Request for dataset format check"""
    dataset_name: str  # HuggingFace dataset name or local path
    is_vlm: bool = False
    hf_token: Optional[str] = None
    split: Optional[str] = "train"


class CheckFormatResponse(BaseModel):
    """Response for dataset format check"""
    requires_manual_mapping: bool
    detected_format: str
    columns: List[str]
    is_multimodal: bool = False
    multimodal_columns: Optional[List[str]] = None
    suggested_mapping: Optional[Dict[str, str]] = None
    detected_image_column: Optional[str] = None
    detected_text_column: Optional[str] = None
    preview_samples: Optional[List[Dict]] = None
    total_rows: Optional[int] = None
