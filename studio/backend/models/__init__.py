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
)
from .auth import (
    AuthSetupRequest,
    AuthLoginRequest,
    AuthStatusResponse,
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
    # Auth schemas
    "AuthSetupRequest",
    "AuthLoginRequest",
    "AuthStatusResponse",
]

