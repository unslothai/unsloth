# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset Pydantic models for the legacy API aliases."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CheckFormatRequest(BaseModel):
    dataset_name: str
    is_vlm: bool = False
    hf_token: Optional[str] = None
    subset: Optional[str] = None
    train_split: Optional[str] = "train"

    @model_validator(mode = "before")
    @classmethod
    def _compat_split(cls, values: Any) -> Any:
        if isinstance(values, dict) and "split" in values:
            merged = {**values}
            merged.setdefault("train_split", merged.pop("split"))
            return merged
        return values


class CheckFormatResponse(BaseModel):
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
    chat_column: Optional[str] = None
    preview_samples: Optional[List[Dict]] = None
    total_rows: Optional[int] = None
    warning: Optional[str] = None


class AiAssistMappingRequest(BaseModel):
    columns: List[str]
    samples: List[Dict[str, Any]]
    dataset_name: Optional[str] = None
    hf_token: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None


class AiAssistMappingResponse(BaseModel):
    success: bool
    suggested_mapping: Optional[Dict[str, str]] = None
    warning: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    assistant_template: Optional[str] = None
    label_mapping: Optional[Dict[str, Dict[str, str]]] = None
    dataset_type: Optional[str] = None
    is_conversational: Optional[bool] = None
    user_notification: Optional[str] = None


class UploadDatasetResponse(BaseModel):
    """Response with stored dataset path for training."""

    filename: str = Field(..., description = "Original filename")
    stored_path: str = Field(..., description = "Absolute path stored on backend")


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
    datasets: List[LocalDatasetItem] = Field(default_factory = list)
