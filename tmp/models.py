"""
Pydantic schemas for Model Management API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ModelDetails(BaseModel):
    """Detailed model configuration and metadata"""
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
