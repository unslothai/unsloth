# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Export API.
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any


def _validate_save_directory(value: str) -> str:
    """Reject save_directory values that escape the export root."""
    if value is None:
        raise ValueError("save_directory is required")
    raw = str(value).strip()
    if not raw:
        raise ValueError("save_directory must not be empty")
    if "\x00" in raw:
        raise ValueError("save_directory may not contain null bytes")
    if any(ch in raw for ch in ("\r", "\n")):
        raise ValueError("save_directory may not contain control characters")
    if len(raw) > 255:
        raise ValueError("save_directory must be <= 255 characters")
    path = Path(raw).expanduser()
    if path.is_absolute():
        raise ValueError(
            "save_directory must be a name or relative path under the "
            "export root; absolute paths are rejected"
        )
    if ".." in path.parts:
        raise ValueError("save_directory may not contain '..' segments")
    return raw


class LoadCheckpointRequest(BaseModel):
    """Request for loading a checkpoint into the export backend."""

    checkpoint_path: str = Field(..., description = "Path to the checkpoint directory")
    max_seq_length: int = Field(
        2048,
        ge = 128,
        le = 32768,
        description = "Maximum sequence length for loading the model",
    )
    load_in_4bit: bool = Field(
        True,
        description = "Whether to load the model in 4-bit quantization",
    )
    trust_remote_code: bool = Field(
        False,
        description = "Allow loading models with custom code. Only enable for checkpoints/base models you trust.",
    )


class ExportStatusResponse(BaseModel):
    """Current export backend status."""

    current_checkpoint: Optional[str] = Field(
        None,
        description = "Path to the currently loaded checkpoint, if any",
    )
    is_vision: bool = Field(
        False,
        description = "True if the loaded checkpoint is a vision model",
    )
    is_peft: bool = Field(
        False,
        description = "True if the loaded checkpoint is a PEFT (LoRA) model",
    )


class ExportOperationResponse(BaseModel):
    """Generic response for export operations."""

    success: bool = Field(..., description = "True if the operation succeeded")
    message: str = Field(..., description = "Human-readable status or error message")
    details: Optional[Dict[str, Any]] = Field(
        default = None,
        description = "Optional extra details about the operation",
    )


class ExportCommonOptions(BaseModel):
    """Common options for export operations that save locally and/or push to Hub."""

    save_directory: str = Field(
        ...,
        description = "Local directory where the exported artifacts will be written",
    )

    @field_validator("save_directory", mode = "before")
    @classmethod
    def _check_save_directory(cls, v):
        return _validate_save_directory(v)

    push_to_hub: bool = Field(
        False,
        description = "If True, also push the exported model to the Hugging Face Hub",
    )
    repo_id: Optional[str] = Field(
        None,
        description = "Hugging Face Hub repository ID (username/model-name)",
    )
    hf_token: Optional[str] = Field(
        None,
        description = "Hugging Face access token used for Hub operations",
    )
    private: bool = Field(
        False,
        description = "If True, create a private repository on the Hub (where applicable)",
    )
    base_model_id: Optional[str] = Field(
        None,
        description = "HuggingFace model ID of the base model (for model card metadata)",
    )


class ExportMergedModelRequest(ExportCommonOptions):
    """Request for exporting a merged PEFT model."""

    format_type: Literal["16-bit (FP16)", "4-bit (FP4)"] = Field(
        "16-bit (FP16)",
        description = "Export precision / format for the merged model",
    )


class ExportBaseModelRequest(ExportCommonOptions):
    """Request for exporting a non-PEFT (base) model."""

    # Uses fields from ExportCommonOptions only


class ExportGGUFRequest(BaseModel):
    """Request for exporting the current model to GGUF format."""

    save_directory: str = Field(
        ...,
        description = "Directory where GGUF files will be saved",
    )

    @field_validator("save_directory", mode = "before")
    @classmethod
    def _check_save_directory(cls, v):
        return _validate_save_directory(v)

    quantization_method: str = Field(
        "Q4_K_M",
        description = 'GGUF quantization method (e.g. "Q4_K_M")',
    )
    push_to_hub: bool = Field(
        False,
        description = "If True, also push GGUF artifacts to the Hugging Face Hub",
    )
    repo_id: Optional[str] = Field(
        None,
        description = "Hugging Face Hub repository ID for GGUF upload",
    )
    hf_token: Optional[str] = Field(
        None,
        description = "Hugging Face token for GGUF upload",
    )


class ExportLoRAAdapterRequest(ExportCommonOptions):
    """Request for exporting only the LoRA adapter (not merged)."""

    # Uses fields from ExportCommonOptions only
