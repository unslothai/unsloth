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
    ModelDetails,
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
    CheckpointInfo,
    CheckpointListResponse,
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
)

__all__ = [
    # Training schemas
    "TrainingStartRequest",
    "TrainingJobResponse",
    "TrainingStatus",
    "TrainingProgress",
    # Model management schemas
    "ModelDetails",
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
]
