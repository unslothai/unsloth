# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Model Management API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

ModelType = Literal["text", "vision", "audio", "embeddings"]


class CheckpointInfo(BaseModel):
    """Information about a discovered checkpoint directory."""

    display_name: str = Field(
        ..., description = "User-friendly checkpoint name (folder name)"
    )
    path: str = Field(..., description = "Full path to the checkpoint directory")
    loss: Optional[float] = Field(None, description = "Training loss at this checkpoint")


class ModelCheckpoints(BaseModel):
    """A training run and its associated checkpoints."""

    name: str = Field(..., description = "Training run folder name")
    checkpoints: List[CheckpointInfo] = Field(
        default_factory = list,
        description = "List of checkpoints for this training run (final + intermediate)",
    )
    base_model: Optional[str] = Field(
        None,
        description = "Base model name from adapter_config.json or config.json",
    )
    peft_type: Optional[str] = Field(
        None,
        description = "PEFT type (e.g. LORA) if adapter training, None for full fine-tune",
    )
    lora_rank: Optional[int] = Field(
        None,
        description = "LoRA rank (r) if applicable",
    )
    is_quantized: bool = Field(
        False,
        description = "Whether the model uses BNB quantization (e.g. bnb-4bit)",
    )


class CheckpointListResponse(BaseModel):
    """Response for listing available checkpoints in an outputs directory."""

    outputs_dir: str = Field(..., description = "Directory that was scanned")
    models: List[ModelCheckpoints] = Field(
        default_factory = list,
        description = "List of training runs with their checkpoints",
    )


class ModelDetails(BaseModel):
    """Detailed model configuration and metadata - can be used for both list and detail views"""

    id: str = Field(..., description = "Model identifier")
    model_name: Optional[str] = Field(
        None, description = "Model identifier (alias for id, for backward compatibility)"
    )
    name: Optional[str] = Field(None, description = "Display name for the model")
    config: Optional[Dict[str, Any]] = Field(
        None, description = "Model configuration dictionary"
    )
    is_vision: bool = Field(False, description = "Whether model is a vision model")
    is_embedding: bool = Field(
        False, description = "Whether model is an embedding/sentence-transformer model"
    )
    is_lora: bool = Field(False, description = "Whether model is a LoRA adapter")
    is_gguf: bool = Field(
        False, description = "Whether model is a GGUF model (llama.cpp format)"
    )
    is_audio: bool = Field(False, description = "Whether model is a TTS audio model")
    audio_type: Optional[str] = Field(
        None, description = "Audio codec type: snac, csm, bicodec, dac"
    )
    has_audio_input: bool = Field(
        False, description = "Whether model accepts audio input (ASR)"
    )
    model_type: Optional[ModelType] = Field(
        None, description = "Collapsed model modality: text, vision, audio, or embeddings"
    )
    base_model: Optional[str] = Field(
        None, description = "Base model if this is a LoRA adapter"
    )
    max_position_embeddings: Optional[int] = Field(
        None, description = "Maximum context length supported by the model"
    )
    model_size_bytes: Optional[int] = Field(
        None, description = "Total size of model weight files in bytes"
    )
    vram_estimate_qlora_gb: Optional[float] = Field(
        None, description = "Estimated training VRAM (GB) for QLoRA (4-bit) with default params"
    )
    vram_estimate_lora_gb: Optional[float] = Field(
        None, description = "Estimated training VRAM (GB) for LoRA (fp16) with default params"
    )
    vram_estimate_full_gb: Optional[float] = Field(
        None, description = "Estimated training VRAM (GB) for full fine-tuning with default params"
    )


class LoRAInfo(BaseModel):
    """LoRA adapter or exported model information"""

    display_name: str = Field(..., description = "Display name for the LoRA")
    adapter_path: str = Field(
        ..., description = "Path to the LoRA adapter or exported model"
    )
    base_model: Optional[str] = Field(None, description = "Base model identifier")
    source: Optional[str] = Field(None, description = "'training' or 'exported'")
    export_type: Optional[str] = Field(
        None, description = "'lora', 'merged', or 'gguf' (for exports)"
    )


class LoRAScanResponse(BaseModel):
    """Response schema for scanning trained LoRA adapters"""

    loras: List[LoRAInfo] = Field(
        default_factory = list, description = "List of found LoRA adapters"
    )
    outputs_dir: str = Field(..., description = "Directory that was scanned")


class ModelListResponse(BaseModel):
    """Response schema for listing models"""

    models: List[ModelDetails] = Field(
        default_factory = list, description = "List of models"
    )
    default_models: List[str] = Field(
        default_factory = list, description = "List of default model IDs"
    )


class GgufVariantDetail(BaseModel):
    """A single GGUF quantization variant in a HuggingFace repo."""

    filename: str = Field(
        ..., description = "GGUF filename (e.g., 'gemma-3-4b-it-Q4_K_M.gguf')"
    )
    quant: str = Field(..., description = "Quantization label (e.g., 'Q4_K_M')")
    size_bytes: int = Field(0, description = "File size in bytes")
    downloaded: bool = Field(
        False, description = "Whether this variant is already in the local HF cache"
    )


class GgufVariantsResponse(BaseModel):
    """Response for listing GGUF quantization variants in a HuggingFace repo."""

    repo_id: str = Field(..., description = "HuggingFace repo ID")
    variants: List[GgufVariantDetail] = Field(
        default_factory = list, description = "Available GGUF variants"
    )
    has_vision: bool = Field(
        False, description = "Whether the model has vision support (mmproj files)"
    )
    default_variant: Optional[str] = Field(
        None, description = "Recommended default quantization variant"
    )


class LocalModelInfo(BaseModel):
    """Discovered local model candidate."""

    id: str = Field(..., description = "Identifier to use for loading/training")
    display_name: str = Field(..., description = "Display label")
    path: str = Field(..., description = "Local path where model data was discovered")
    source: Literal["models_dir", "hf_cache", "lmstudio", "custom"] = Field(
        ...,
        description = "Discovery source",
    )
    model_id: Optional[str] = Field(
        None,
        description = "HF repo id for cached models, e.g. org/model",
    )
    updated_at: Optional[float] = Field(
        None,
        description = "Unix timestamp of latest observed update",
    )


class LocalModelListResponse(BaseModel):
    """Response schema for listing local/cached models."""

    models_dir: str = Field(
        ..., description = "Directory scanned for custom local models"
    )
    hf_cache_dir: Optional[str] = Field(
        None,
        description = "HF cache root that was scanned",
    )
    lmstudio_dirs: List[str] = Field(
        default_factory = list,
        description = "LM Studio model directories that were scanned",
    )
    models: List[LocalModelInfo] = Field(
        default_factory = list,
        description = "Discovered local/cached models",
    )


class AddScanFolderRequest(BaseModel):
    """Request body for adding a custom scan folder."""

    path: str = Field(
        ..., description = "Absolute or relative directory path to scan for models"
    )


class ScanFolderInfo(BaseModel):
    """A registered custom model scan folder."""

    id: int = Field(..., description = "Database row ID")
    path: str = Field(..., description = "Normalized absolute path")
    created_at: str = Field(..., description = "ISO 8601 creation timestamp")
