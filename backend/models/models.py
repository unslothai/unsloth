"""
Pydantic schemas for Model Management API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ModelSearchRequest(BaseModel):
    """Request schema for searching HuggingFace models"""
    query: str = Field(..., description="Search query")
    hf_token: Optional[str] = Field(None, description="HuggingFace token for authenticated searches")


class ModelInfo(BaseModel):
    """Model information"""
    id: str = Field(..., description="Model identifier")
    name: Optional[str] = Field(None, description="Display name")
    description: Optional[str] = Field(None, description="Model description")
    size: Optional[str] = Field(None, description="Model size")
    is_vision: bool = Field(False, description="Whether model is a vision model")
    is_lora: bool = Field(False, description="Whether model is a LoRA adapter")


class ModelSearchResponse(BaseModel):
    """Response schema for model search"""
    models: List[ModelInfo] = Field(default_factory=list, description="List of matching models")
    total: int = Field(0, description="Total number of results")


class ModelListResponse(BaseModel):
    """Response schema for listing available models"""
    models: List[ModelInfo] = Field(default_factory=list, description="List of available models")
    default_models: List[str] = Field(default_factory=list, description="List of default model IDs")


class ModelConfigResponse(BaseModel):
    """Response schema for model configuration"""
    model_name: str = Field(..., description="Model identifier")
    config: Dict[str, Any] = Field(..., description="Model configuration dictionary")
    is_vision: bool = Field(False, description="Whether model is a vision model")
    is_lora: bool = Field(False, description="Whether model is a LoRA adapter")
    base_model: Optional[str] = Field(None, description="Base model if this is a LoRA adapter")


class LoRAInfo(BaseModel):
    """LoRA adapter information"""
    display_name: str = Field(..., description="Display name for the LoRA")
    adapter_path: str = Field(..., description="Path to the LoRA adapter")
    base_model: Optional[str] = Field(None, description="Base model identifier")


class LoRAScanResponse(BaseModel):
    """Response schema for scanning trained LoRA adapters"""
    loras: List[LoRAInfo] = Field(default_factory=list, description="List of found LoRA adapters")
    outputs_dir: str = Field(..., description="Directory that was scanned")

