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
    "LoRAInfo",
    "LoRAScanResponse",
    "ModelListResponse",
    # Auth schemas
    "AuthSetupRequest",
    "AuthLoginRequest",
    "RefreshTokenRequest",
    "AuthStatusResponse",
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
    # Data recipe
    "RecipePayload",
    "PreviewResponse",
    "ValidateError",
    "ValidateResponse",
    "JobCreateResponse",
]
