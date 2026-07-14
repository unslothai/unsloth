# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for the Hub inventory layer (/api/hub/*).

Kept independent from upstream models/models.py so the Hub module can ship
without modifying any upstream schema."""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


ModelFormat = Literal["gguf", "safetensors", "adapter", "checkpoint", "unknown"]
ModelRuntime = Literal["llama_cpp", "transformers", "adapter", "unknown"]


class GgufVariantDetail(BaseModel):
    """A single GGUF quantization variant in a HuggingFace repo."""

    filename: str = Field(..., description = "GGUF filename (e.g., 'gemma-3-4b-it-Q4_K_M.gguf')")
    quant: str = Field(..., description = "Quantization label or internal GGUF variant key")
    display_label: Optional[str] = Field(
        None, description = "Optional user-facing label when quant is an internal key"
    )
    size_bytes: int = Field(0, description = "File size in bytes")
    download_size_bytes: int = Field(0, description = "Total bytes needed to download this variant")
    downloaded: bool = Field(
        False, description = "Whether this variant is already in the local HF cache"
    )
    update_available: bool = Field(
        False, description = "Whether a newer main GGUF blob is available on Hugging Face"
    )
    partial: bool = Field(
        False,
        description = "Whether this variant has an in-progress (.incomplete) blob in cache",
    )
    partial_transport: Optional[str] = Field(
        None,
        description = (
            'Transport recorded for the partial state ("http" or '
            '"xet"), or null if not partial / unknown. Frontend uses '
            "this to pick Resume (http) vs Redownload (xet) labels."
        ),
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


class LocalModelCapabilities(BaseModel):
    can_train: bool = False
    can_chat: bool = False
    can_delete: bool = False
    can_download: bool = False
    requires_variant: bool = False
    supports_lora: bool = False
    supports_vision: bool = False


class LocalModelInfo(BaseModel):
    """Discovered local model candidate."""

    id: str = Field(..., description = "Identifier to use for loading/training")
    inventory_id: Optional[str] = Field(
        None, description = "Stable semantic inventory row identifier"
    )
    load_id: Optional[str] = Field(
        None, description = "Identifier/path to pass to load or train APIs"
    )
    display_name: str = Field(..., description = "Display label")
    path: str = Field(..., description = "Local path where model data was discovered")
    size_bytes: int = Field(0, description = "Observed model artifact size in bytes")
    model_format: ModelFormat = Field("unknown", description = "Model file format")
    runtime: ModelRuntime = Field("unknown", description = "Expected runtime backend")
    format_variant: Optional[str] = Field(
        None, description = "Format variant label, for example a GGUF quant"
    )
    capabilities: LocalModelCapabilities = Field(
        default_factory = LocalModelCapabilities,
        description = "Declared capabilities for this inventory row",
    )
    source: Literal["models_dir", "hf_cache", "lmstudio", "ollama", "custom"] = Field(
        ...,
        description = "Discovery source",
    )
    model_id: Optional[str] = Field(
        None,
        description = "HF repo id for cached models, e.g. org/model",
    )
    base_model: Optional[str] = Field(
        None,
        description = "Base model from adapter_config.json when this is an adapter",
    )
    base_model_source: Optional[Literal["huggingface", "local", "unknown"]] = Field(
        None,
        description = "Whether the adapter base model is a HF repo id or local path",
    )
    adapter_type: Optional[str] = Field(
        None,
        description = "Adapter type from adapter_config.json, e.g. LORA",
    )
    training_method: Optional[str] = Field(
        None,
        description = "Training method hint from adapter_config.json",
    )
    updated_at: Optional[float] = Field(
        None,
        description = "Unix timestamp of latest observed update",
    )
    partial: bool = Field(
        False,
        description = "True when this hf_cache entry has incomplete blobs",
    )
    partial_transport: Optional[str] = Field(
        None,
        description = (
            'Transport recorded for the partial state ("http" or '
            '"xet"), or null if not partial / unknown.'
        ),
    )


class LocalModelListResponse(BaseModel):
    """Response schema for listing local/cached models."""

    models_dir: str = Field(..., description = "Directory scanned for custom local models")
    hf_cache_dir: Optional[str] = Field(
        None,
        description = "HF cache root that was scanned",
    )
    lmstudio_dirs: List[str] = Field(
        default_factory = list,
        description = "LM Studio model directories that were scanned",
    )
    ollama_dirs: List[str] = Field(
        default_factory = list,
        description = "Ollama model directories that were scanned",
    )
    models: List[LocalModelInfo] = Field(
        default_factory = list,
        description = "Discovered local/cached models",
    )


class CachedRepoBase(BaseModel):
    """Shared shape for a cached HF repo row surfaced under On Device."""

    repo_id: str
    size_bytes: int = 0
    cache_path: Optional[str] = None
    partial: bool = False
    partial_transport: Optional[str] = None
    inventory_id: Optional[str] = None
    load_id: Optional[str] = None
    model_format: ModelFormat = "unknown"
    runtime: ModelRuntime = "unknown"
    format_variant: Optional[str] = None
    capabilities: LocalModelCapabilities = Field(default_factory = LocalModelCapabilities)


class CachedGgufRepo(CachedRepoBase):
    model_format: ModelFormat = "gguf"


class CachedGgufResponse(BaseModel):
    cached: List[CachedGgufRepo] = Field(default_factory = list)


class CachedModelRepo(CachedRepoBase):
    quant_method: Optional[str] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    tags: Optional[List[str]] = None


class CachedModelsResponse(BaseModel):
    cached: List[CachedModelRepo] = Field(default_factory = list)


class AddScanFolderRequest(BaseModel):
    """Request body for adding a custom scan folder."""

    path: str = Field(
        ...,
        description = "Absolute or relative folder path, or a model weight file path",
    )


class ScanFolderInfo(BaseModel):
    """A registered custom model scan folder."""

    id: int = Field(..., description = "Database row ID")
    path: str = Field(..., description = "Normalized absolute path")
    created_at: str = Field(..., description = "ISO 8601 creation timestamp")


class ScanFoldersResponse(BaseModel):
    folders: List[ScanFolderInfo] = Field(default_factory = list)


class RemoveScanFolderResponse(BaseModel):
    ok: bool


class RecommendedFoldersResponse(BaseModel):
    folders: List[str] = Field(default_factory = list)


class DeleteCachedModelResponse(BaseModel):
    status: str
    repo_id: str
    variant: Optional[str] = None


class BrowseEntry(BaseModel):
    """A directory entry surfaced by the folder browser."""

    name: str = Field(..., description = "Entry name (basename, not full path)")
    has_models: bool = Field(
        False,
        description = (
            "Hint that the directory likely contains models "
            "(*.gguf, *.safetensors, config.json, or HF-style "
            "`models--*` subfolders). Used by the UI to highlight "
            "promising candidates; the scanner itself is authoritative."
        ),
    )
    hidden: bool = Field(
        False,
        description = "Name starts with a dot (e.g. `.cache`)",
    )


class BrowseFoldersResponse(BaseModel):
    """Response schema for the folder browser endpoint."""

    current: str = Field(..., description = "Absolute path of the directory just listed")
    parent: Optional[str] = Field(
        None,
        description = (
            "Parent directory of `current`, or null if `current` is the "
            "filesystem root. The frontend uses this to render an `Up` row."
        ),
    )
    entries: List[BrowseEntry] = Field(
        default_factory = list,
        description = (
            "Subdirectories of `current`. Sorted with model-bearing "
            "directories first, then alphabetically case-insensitive; "
            "hidden entries come last within each group."
        ),
    )
    suggestions: List[str] = Field(
        default_factory = list,
        description = (
            "Handy starting points (home, HF cache, already-registered "
            "scan folders). Rendered as quick-pick chips above the list."
        ),
    )
    truncated: bool = Field(
        False,
        description = (
            "True when the listing was capped because the directory had "
            "more subfolders than the server is willing to enumerate in "
            "one request. The UI should show a hint telling the user to "
            "narrow their path."
        ),
    )
    model_files_here: int = Field(
        0,
        description = (
            "Count of GGUF/safetensors files immediately inside "
            "``current``. Used by the UI to surface a hint on leaf "
            "model directories (which otherwise look `empty` because "
            "they contain only files, no subdirectories)."
        ),
    )


class ModelsFolderResponse(BaseModel):
    """The directory where downloaded models are stored (the active HF hub
    cache, honoring ``HF_HOME`` / ``HF_HUB_CACHE``)."""

    path: str = Field(
        ...,
        description = "Path to the model download directory.",
    )
