# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset Pydantic models for API requests and responses."""

from typing import List, Optional

from pydantic import BaseModel, Field


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
