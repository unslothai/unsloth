# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for Export API."""

from pathlib import Path, PureWindowsPath

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any, Union


def _validate_save_directory(value: str) -> str:
    """Validate save_directory — allows absolute paths (user may want a different drive)."""
    if value is None:
        raise ValueError("save_directory is required")
    raw = str(value).strip()
    if not raw:
        raise ValueError("save_directory must not be empty")
    if "\x00" in raw:
        raise ValueError("save_directory may not contain null bytes")
    if any(ch in raw for ch in ("\r", "\n")):
        raise ValueError("save_directory may not contain control characters")
    path = Path(raw).expanduser()
    path_parts = (*path.parts, *PureWindowsPath(raw).parts, *raw.replace("\\", "/").split("/"))
    if any(len(part) > 255 for part in path_parts if part not in ("", ".", "/", "\\")):
        raise ValueError("save_directory path components must be <= 255 characters")
    if (
        ".." in path.parts
        or ".." in PureWindowsPath(raw).parts
        or ".." in raw.replace("\\", "/").split("/")
    ):
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
    approved_remote_code_fingerprint: Optional[str] = Field(
        None,
        description = "sha256 fingerprint from the remote-code scan, pinning user approval of this exact custom-code version.",
    )
    hf_token: Optional[str] = Field(
        None,
        description = "Hugging Face token used to scan/load gated checkpoints and their base models.",
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
    is_export_active: bool = Field(
        False,
        description = "True while a load / export / cleanup operation is running",
    )
    # Recovery fields: when a blocking export POST is cut off by a Cloudflare tunnel
    # timeout (524 at ~100s), the client polls this endpoint to learn the real
    # outcome of the operation that kept running on the backend.
    active_op_kind: Optional[str] = Field(
        None,
        description = "Kind of the currently running op (load_checkpoint / export_* / cleanup)",
    )
    last_op_seq: int = Field(
        0,
        description = "Monotonic counter of finished ops; client baseline to detect 'my op finished'",
    )
    last_op_kind: Optional[str] = Field(
        None,
        description = "Kind of the most recently finished op",
    )
    last_op_status: Optional[str] = Field(
        None,
        description = "Outcome of the most recently finished op: success / error / cancelled",
    )
    last_op_output_path: Optional[str] = Field(
        None,
        description = "Output path of the most recently finished op, if it produced one",
    )
    last_op_error: Optional[str] = Field(
        None,
        description = "Error message of the most recently finished op, if it failed",
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

    format_type: Literal[
        "16-bit (FP16)",
        "4-bit (FP4)",
        "FP8 (compressed-tensors)",
        "NVFP4 (compressed-tensors)",
    ] = Field(
        "16-bit (FP16)",
        description = "Export precision / format for the merged model. The compressed-tensors "
        "options run llm-compressor for vLLM (FP8 is data-free; NVFP4 calibrates).",
    )
    compressed_method: Optional[str] = Field(
        None,
        description = "Optional quantized-export alias. Either a compressed-tensors scheme "
        "(e.g. 'fp8', 'fp8_static', 'w8a8', 'w4a16', 'mxfp4', 'mxfp8', 'nvfp4' - NVIDIA only) "
        "from unsloth.save COMPRESSED_EXPORT_SCHEMES, or a portable torchao alias "
        "('torchao_fp8', 'torchao_int8') from TORCHAO_EXPORT_SCHEMES that needs no NVIDIA GPU. "
        "When set, it overrides format_type. Lets the export UI expose the full set of formats "
        "beyond the quick buttons.",
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

    quantization_method: Union[str, List[str]] = Field(
        "Q4_K_M",
        description = 'GGUF quantization method(s). A single method (e.g. "Q4_K_M") or a list '
        '(e.g. ["Q4_K_M", "Q8_0"]) to produce multiple GGUFs from one model load.',
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
    imatrix: bool = Field(
        False,
        description = "Use an importance matrix (auto-downloads the upstream unsloth GGUF "
        "imatrix). Required for the IQ low-bit quants such as iq2_xxs / iq4_xs.",
    )
    imatrix_path: Optional[str] = Field(
        None,
        description = "Path to a custom imatrix file; overrides the auto-download when set.",
    )


class ExportLoRAAdapterRequest(ExportCommonOptions):
    """Request for exporting only the LoRA adapter (not merged)."""

    gguf: bool = Field(
        False,
        description = "If True, also convert the adapter to a GGUF LoRA file "
        "(llama.cpp convert_lora_to_gguf.py), loadable with `llama-cli --lora ...`.",
    )
    gguf_outtype: Literal["q8_0", "f16", "bf16", "f32"] = Field(
        "q8_0",
        description = "GGUF LoRA output float type (only used when gguf=True). "
        "Q8_0 falls back to F16 per tensor for dims not divisible by the block size (32).",
    )
