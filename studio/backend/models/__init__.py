# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Pydantic models for API request/response schemas
"""

from .training import (
    TrainingStartRequest,
    TrainingJobResponse,
    TrainingStatus,
    TrainingProgress,
)
from .models import (
    CheckpointInfo,
    ModelCheckpoints,
    CheckpointListResponse,
    ModelDetails,
    LocalModelInfo,
    LocalModelListResponse,
    LoRAInfo,
    LoRAScanResponse,
    ModelListResponse,
)
from .auth import (
    AuthSetupRequest,
    AuthLoginRequest,
    RefreshTokenRequest,
    AuthStatusResponse,
)
from .export import (
    LoadCheckpointRequest,
    ExportStatusResponse,
    ExportOperationResponse,
    ExportMergedModelRequest,
    ExportBaseModelRequest,
    ExportGGUFRequest,
    ExportLoRAAdapterRequest,
)
from .users import Token
from .datasets import (
    CheckFormatRequest,
    CheckFormatResponse,
)
from .inference import (
    LoadRequest,
    UnloadRequest,
    GenerateRequest,
    LoadResponse,
    UnloadResponse,
    InferenceStatusResponse,
)
from .responses import (
    TrainingStopResponse,
    TrainingMetricsResponse,
    LoRABaseModelResponse,
    VisionCheckResponse,
    EmbeddingCheckResponse,
)
from .data_recipe import (
    RecipePayload,
    PreviewResponse,
    ValidateError,
    ValidateResponse,
    JobCreateResponse,
)

__all__ = [
    # Training schemas
    "TrainingStartRequest",
    "TrainingJobResponse",
    "TrainingStatus",
    "TrainingProgress",
    # Model management schemas
    "ModelDetails",
    "LocalModelInfo",
    "LocalModelListResponse",
    "LoRAInfo",
    "LoRAScanResponse",
    "ModelListResponse",
    # Auth schemas
    "AuthSetupRequest",
    "AuthLoginRequest",
    "RefreshTokenRequest",
    "AuthStatusResponse",
    # Export schemas
    "CheckpointInfo",
    "ModelCheckpoints",
    "CheckpointListResponse",
    "LoadCheckpointRequest",
    "ExportStatusResponse",
    "ExportOperationResponse",
    "ExportMergedModelRequest",
    "ExportBaseModelRequest",
    "ExportGGUFRequest",
    "ExportLoRAAdapterRequest",
    "Token",
    # Dataset schemas
    "CheckFormatRequest",
    "CheckFormatResponse",
    # Inference schemas
    "LoadRequest",
    "UnloadRequest",
    "GenerateRequest",
    "LoadResponse",
    "UnloadResponse",
    "InferenceStatusResponse",
    # Response schemas
    "TrainingStopResponse",
    "TrainingMetricsResponse",
    "LoRABaseModelResponse",
    "VisionCheckResponse",
    "EmbeddingCheckResponse",
    # Data recipe
    "RecipePayload",
    "PreviewResponse",
    "ValidateError",
    "ValidateResponse",
    "JobCreateResponse",
]
