# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Dataset-related Pydantic models for API requests and responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Round 24 P1 #11: reuse the chat / diffusion / export identifier
# hardening so dataset routes also reject control characters and
# URL-embedded HF tokens in user-controlled identifiers.
from models.inference import _no_control_chars, _reject_embedded_hf_token


class CheckFormatRequest(BaseModel):
    """Request for dataset format check"""

    dataset_name: str  # HuggingFace dataset name or local path
    is_vlm: bool = False
    hf_token: Optional[str] = None
    subset: Optional[str] = None
    train_split: Optional[str] = "train"

    @model_validator(mode = "before")
    @classmethod
    def _compat_split(cls, values: Any) -> Any:
        """Accept legacy 'split' field as alias for 'train_split'."""
        if isinstance(values, dict) and "split" in values:
            values.setdefault("train_split", values.pop("split"))
        return values

    # Round 27 P1 #6: subset / train_split also flow into HF dataset
    # APIs and errors/responses, so they need the same hardening.
    @field_validator("dataset_name", "subset", "train_split")
    @classmethod
    def _no_dataset_name_control_chars(cls, v, info):
        return _no_control_chars(v, info.field_name)

    @field_validator("dataset_name", "subset", "train_split")
    @classmethod
    def _no_dataset_name_embedded_hf_tokens(cls, v, info):
        return _reject_embedded_hf_token(v, info.field_name)


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


class AiAssistMappingRequest(BaseModel):
    """Request for LLM-assisted column classification (user-triggered)."""

    columns: List[str]
    samples: List[Dict[str, Any]]  # Preview rows already loaded in the dialog
    dataset_name: Optional[str] = None  # For LLM context
    hf_token: Optional[str] = None  # For fetching dataset card
    model_name: Optional[str] = None
    model_type: Optional[str] = None

    @field_validator("dataset_name", "model_name")
    @classmethod
    def _no_identifier_control_chars(cls, v, info):
        return _no_control_chars(v, info.field_name)

    @field_validator("dataset_name", "model_name")
    @classmethod
    def _no_identifier_embedded_hf_tokens(cls, v, info):
        return _reject_embedded_hf_token(v, info.field_name)


class AiAssistMappingResponse(BaseModel):
    """Response from LLM-assisted column classification and conversion advice."""

    success: bool
    suggested_mapping: Optional[Dict[str, str]] = None
    warning: Optional[str] = None
    # Conversion advisor fields
    system_prompt: Optional[str] = None
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
