# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic response schemas for endpoints that previously returned raw dicts.
These are small response models for training and model management routes.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# --- Training route response models ---


class TrainingStopResponse(BaseModel):
    """Response for stopping a training job"""

    status: str = Field(..., description = "Current status: 'stopped' or 'idle'")
    message: str = Field(..., description = "Human-readable status message")


class TrainingMetricsResponse(BaseModel):
    """Response for training metrics history"""

    loss_history: List[float] = Field(
        default_factory = list, description = "Loss values per step"
    )
    lr_history: List[float] = Field(
        default_factory = list, description = "Learning rate per step"
    )
    step_history: List[int] = Field(default_factory = list, description = "Step numbers")
    grad_norm_history: List[float] = Field(
        default_factory = list, description = "Gradient norm values"
    )
    grad_norm_step_history: List[int] = Field(
        default_factory = list, description = "Step numbers for gradient norm values"
    )
    current_loss: Optional[float] = Field(None, description = "Most recent loss value")
    current_lr: Optional[float] = Field(None, description = "Most recent learning rate")
    current_step: Optional[int] = Field(None, description = "Most recent step number")


# --- Model management route response models ---


class LoRABaseModelResponse(BaseModel):
    """Response for getting a LoRA's base model"""

    lora_path: str = Field(..., description = "Path to the LoRA adapter")
    base_model: str = Field(..., description = "Base model identifier")


class VisionCheckResponse(BaseModel):
    """Response for checking if a model is a vision model"""

    model_name: str = Field(..., description = "Model identifier")
    is_vision: bool = Field(..., description = "Whether the model is a vision model")


class EmbeddingCheckResponse(BaseModel):
    """Response for checking if a model is an embedding model"""

    model_name: str = Field(..., description = "Model identifier")
    is_embedding: bool = Field(
        ..., description = "Whether the model is an embedding/sentence-transformer model"
    )
