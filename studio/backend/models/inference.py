"""
Pydantic schemas for Inference API
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class LoadRequest(BaseModel):
    """Request to load a model for inference"""
    model_path: str = Field(..., description="Model identifier or local path")
    hf_token: Optional[str] = Field(None, description="HuggingFace token for gated models")
    max_seq_length: int = Field(2048, ge=128, le=32768, description="Maximum sequence length")
    load_in_4bit: bool = Field(True, description="Load model in 4-bit quantization")
    is_lora: bool = Field(False, description="Whether this is a LoRA adapter")


class UnloadRequest(BaseModel):
    """Request to unload a model"""
    model_path: str = Field(..., description="Model identifier to unload")


class GenerateRequest(BaseModel):
    """Request for text generation"""
    messages: List[dict] = Field(..., description="Chat messages in OpenAI format")
    system_prompt: str = Field("You are a helpful AI assistant.", description="System prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(40, ge=1, le=100, description="Top-k sampling")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image for vision models")


class LoadResponse(BaseModel):
    """Response after loading a model"""
    status: str = Field(..., description="Load status")
    model: str = Field(..., description="Model identifier")
    display_name: str = Field(..., description="Display name of the model")
    is_vision: bool = Field(False, description="Whether model is a vision model")
    is_lora: bool = Field(False, description="Whether model is a LoRA adapter")


class UnloadResponse(BaseModel):
    """Response after unloading a model"""
    status: str = Field(..., description="Unload status")
    model: str = Field(..., description="Model identifier that was unloaded")


class InferenceStatusResponse(BaseModel):
    """Current inference backend status"""
    active_model: Optional[str] = Field(None, description="Currently active model identifier")
    is_vision: bool = Field(False, description="Whether the active model is a vision model")
    loading: List[str] = Field(default_factory=list, description="Models currently being loaded")
    loaded: List[str] = Field(default_factory=list, description="Models currently loaded")
