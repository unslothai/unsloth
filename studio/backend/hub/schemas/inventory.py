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
    shard_count: int = Field(0, description = "Part count for a complete canonical split GGUF")
    download_remaining_bytes: Optional[int] = Field(
        None,
        description = (
            "Bytes a resume still has to fetch: the total minus what is already on disk "
            "and reusable. Set only on a partial variant; null when not partial or when "
            "the plan cannot be resolved."
        ),
    )
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
    cleanable: bool = Field(
        False,
        description = (
            "Row exists only to offer deleting an empty leftover <quant>/ folder; the "
            "listing has no such weights, so it never proves a load would find any"
        ),
    )
    partial_transport: Optional[str] = Field(
        None,
        description = (
            'Transport recorded for the partial state ("http" or '
            '"xet"), or null if not partial / unknown.'
        ),
    )
    partial_resumable: bool = Field(
        False,
        description = (
            "Whether THIS partial can be continued byte for byte, which is what picks "
            "Resume over Continue. False for a Xet partial, and for an HTTP one no "
            "installed writer can reopen."
        ),
    )
    dependency_key: Optional[str] = Field(
        None,
        description = (
            "Opaque grouping key: variants sharing a key share one companion "
            "download footprint (text encoders, VAE, tokenizer, configs), so a "
            "footprint resolved for one of them is correct for all of them. The "
            "companion set is NOT repository-wide -- one repo can hold GGUFs of "
            "different diffusion families, and FLUX.2-klein picks its text "
            "encoder per checkpoint size -- so a client must group by this key "
            "rather than per repo. Null means unknown (no family resolved); "
            "clients should then treat the repo as a single group."
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
    resolved_locally: bool = Field(
        False,
        description = "Whether this answer came from resolving repo_id as a local path",
    )
    loadable_variants: Optional[List[str]] = Field(
        None,
        description = (
            "Quants the load resolver resolves for this identifier; None when unanswered "
            "(remote answers, or a server that predates the field)"
        ),
    )
    loadable: Optional[bool] = Field(
        None,
        description = "Whether a variantless load resolves GGUF weights; None when unanswered",
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
    active_cache: Optional[bool] = Field(
        None,
        description = "Whether this HF entry belongs to the current download cache.",
    )
    task: Optional[str] = Field(
        None,
        description = (
            "Inferred pipeline task. The task-scoped pickers filter On Device rows on it and the "
            "chat picker routes a diffusion pick by it, so a row without one is dropped from "
            "those lists."
        ),
    )
    audio_type: Optional[str] = Field(
        None,
        description = "Detected output-audio architecture or codec used by Audio runtime policy",
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
    partial_resumable: bool = Field(
        False,
        description = "Whether THIS partial can be continued byte for byte.",
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
    last_modified: Optional[float] = None
    partial: bool = False
    partial_transport: Optional[str] = None
    partial_resumable: bool = False
    inventory_id: Optional[str] = None
    load_id: Optional[str] = None
    model_format: ModelFormat = "unknown"
    runtime: ModelRuntime = "unknown"
    format_variant: Optional[str] = None
    capabilities: LocalModelCapabilities = Field(default_factory = LocalModelCapabilities)
    # The task-scoped pickers filter On Device rows on the inferred task and the chat picker routes a
    # diffusion pick by it, so a row without one is dropped from those lists.
    task: Optional[str] = None
    audio_type: Optional[str] = None


class CachedGgufRepo(CachedRepoBase):
    model_format: ModelFormat = "gguf"
    has_variant_state: bool = Field(
        False,
        description = (
            "Whether a download manifest or cancel marker exists for some quant. A sibling "
            "cancelled before any file landed changes nothing else on this row, so callers "
            "watching for on-disk change need this to notice it."
        ),
    )


class CachedGgufResponse(BaseModel):
    cached: List[CachedGgufRepo] = Field(default_factory = list)
    scan_confirmed: bool = True


class CachedModelRepo(CachedRepoBase):
    audio_type: Optional[str] = None
    quant_method: Optional[str] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    tags: Optional[List[str]] = None
    # True for a diffusion-tagged repo with NO top-level model_index.json: a single-file checkpoint
    # needing from_single_file plus a filename. Pickers must not offer it as a pipeline load unless
    # the catalog carries a curated artifact.
    single_file: bool = False
    # An sd.cpp companion mirror is never a pick on any page, but still gets a row, because these run to
    # tens of GB and the row is how they are seen and deleted.
    companion: bool = False
    # An unrecognised pipeline carries no task and no root config for can_chat, so this flag is all
    # that keeps it out of a chat picker. Declared because response_model drops undeclared keys, which
    # left the CLI and the frontend disagreeing about the same row.
    diffusers: bool = False


class CachedModelsResponse(BaseModel):
    cached: List[CachedModelRepo] = Field(default_factory = list)
    scan_confirmed: bool = True


class HiddenModelsResponse(BaseModel):
    needles: List[str] = Field(default_factory = list)
    exact_ids: List[str] = Field(default_factory = list)
    exact_paths: List[str] = Field(default_factory = list)


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
    status: str = Field(
        default = "ok",
        description = "Last scan result: ok, permission_denied, missing, or unreadable",
    )


class ScanFoldersResponse(BaseModel):
    folders: List[ScanFolderInfo] = Field(default_factory = list)


class RemoveScanFolderResponse(BaseModel):
    ok: bool


class DeleteCachedModelResponse(BaseModel):
    status: str
    repo_id: str
    variant: Optional[str] = None


class CompanionAssetInfo(BaseModel):
    """A companion base repo (text encoders, VAE, tokenizer, configs) in the cache."""

    repo_id: str
    size_bytes: int = Field(0, description = "Real on-disk blob bytes, deduped per blob")
    needed_by: List[str] = Field(
        default_factory = list,
        description = "Installed models that still need it; empty means it is reclaimable",
    )


class DeleteImpactResponse(BaseModel):
    """What a pending delete would actually do, so the confirm dialog can say it."""

    repo_id: str
    variant: Optional[str] = None
    reclaimed_bytes: int = Field(0, description = "Bytes this delete frees, from the cache scan")
    retained_companions: List[CompanionAssetInfo] = Field(
        default_factory = list,
        description = "Shared assets that stay because another installed model needs them",
    )
    freeable_companions: List[CompanionAssetInfo] = Field(
        default_factory = list,
        description = "Shared assets that become orphaned by this delete and can then be removed",
    )
    blocked_by: List[str] = Field(
        default_factory = list,
        description = "Installed models that make this delete impossible (shared-asset guard)",
    )


class OrphanCompanionInfo(BaseModel):
    repo_id: str
    size_bytes: int = 0
    cache_path: Optional[str] = None


class OrphanCompanionsResponse(BaseModel):
    companions: List[OrphanCompanionInfo] = Field(default_factory = list)
    total_bytes: int = 0


class ModelsFolderResponse(BaseModel):
    """The directory where downloaded models are stored (the active HF hub
    cache, honoring ``HF_HOME`` / ``HF_HUB_CACHE``)."""

    path: str = Field(
        ...,
        description = "Path to the model download directory.",
    )
