# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Model Management API routes
"""

import os
import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import structlog
from loggers import get_logger

# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from auth.authentication import get_current_subject

# Import backend functions
try:
    from utils.models import (
        scan_trained_loras,
        scan_exported_models,
        load_model_defaults,
        get_base_model_from_lora,
        is_vision_model,
        is_embedding_model,
        scan_checkpoints,
        list_gguf_variants,
        ModelConfig,
    )
    from utils.models.model_config import (
        _pick_best_gguf,
        _extract_quant_label,
        is_audio_input_type,
    )
    from core.inference import get_inference_backend
    from utils.paths import (
        outputs_root,
        exports_root,
        resolve_output_dir,
        resolve_export_dir,
    )
except ImportError:
    # Fallback: try to import from parent directory
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from utils.models import (
        scan_trained_loras,
        scan_exported_models,
        load_model_defaults,
        get_base_model_from_lora,
        is_vision_model,
        is_embedding_model,
        scan_checkpoints,
        list_gguf_variants,
        ModelConfig,
    )
    from utils.models.model_config import (
        _pick_best_gguf,
        _extract_quant_label,
        is_audio_input_type,
    )
    from core.inference import get_inference_backend
    from utils.paths import (
        outputs_root,
        exports_root,
        resolve_output_dir,
        resolve_export_dir,
    )

from models import (
    CheckpointInfo,
    CheckpointListResponse,
    LocalModelInfo,
    LocalModelListResponse,
    ModelCheckpoints,
    ModelDetails,
    LoRAScanResponse,
    LoRAInfo,
    ModelListResponse,
)
from models.models import GgufVariantDetail, GgufVariantsResponse, ModelType
from models.responses import (
    LoRABaseModelResponse,
    VisionCheckResponse,
    EmbeddingCheckResponse,
)

router = APIRouter()
logger = get_logger(__name__)


def derive_model_type(
    is_vision: bool, audio_type: Optional[str], is_embedding: bool = False
) -> ModelType:
    """Collapse individual capability flags into a single model modality string."""
    if is_embedding:
        return "embeddings"
    if audio_type is not None:
        return "audio"
    if is_vision:
        return "vision"
    return "text"


def _resolve_hf_cache_dir() -> Path:
    """Resolve local HF cache root used by hub downloads."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _scan_models_dir(models_dir: Path) -> List[LocalModelInfo]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []

    found: List[LocalModelInfo] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        has_model_files = (
            (child / "config.json").exists()
            or (child / "adapter_config.json").exists()
            or any(child.glob("*.safetensors"))
            or any(child.glob("*.bin"))
            or any(child.glob("*.gguf"))
        )
        if not has_model_files:
            continue
        try:
            updated_at = child.stat().st_mtime
        except OSError:
            updated_at = None
        found.append(
            LocalModelInfo(
                id = str(child),
                display_name = child.name,
                path = str(child),
                source = "models_dir",
                updated_at = updated_at,
            ),
        )
    # Also scan for standalone .gguf files directly in the models directory
    for gguf_file in models_dir.glob("*.gguf"):
        if gguf_file.is_file():
            try:
                updated_at = gguf_file.stat().st_mtime
            except OSError:
                updated_at = None
            found.append(
                LocalModelInfo(
                    id = str(gguf_file),
                    display_name = gguf_file.stem,
                    path = str(gguf_file),
                    source = "models_dir",
                    updated_at = updated_at,
                ),
            )

    return found


def _scan_hf_cache(cache_dir: Path) -> List[LocalModelInfo]:
    if not cache_dir.exists() or not cache_dir.is_dir():
        return []

    found: List[LocalModelInfo] = []
    for repo_dir in cache_dir.glob("models--*"):
        if not repo_dir.is_dir():
            continue

        repo_name = repo_dir.name[len("models--") :]
        if not repo_name:
            continue
        model_id = repo_name.replace("--", "/")

        try:
            updated_at = repo_dir.stat().st_mtime
        except OSError:
            updated_at = None

        found.append(
            LocalModelInfo(
                id = model_id,
                model_id = model_id,
                display_name = model_id.split("/")[-1],
                path = str(repo_dir),
                source = "hf_cache",
                updated_at = updated_at,
            ),
        )
    return found


@router.get("/local", response_model = LocalModelListResponse)
async def list_local_models(
    models_dir: str = Query(
        default = "./models", description = "Directory to scan for local model folders"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    List local model candidates from custom models dir and HF cache.
    """
    # Validate models_dir against an allowlist of trusted directories.
    # Only the trusted Path objects are used for filesystem access -- the
    # user-supplied string is only used for matching, never for path construction.
    hf_cache_dir = _resolve_hf_cache_dir()
    allowed_roots = [Path("./models").resolve(), hf_cache_dir]
    try:
        from utils.paths import studio_root, outputs_root

        allowed_roots.extend([studio_root(), outputs_root()])
    except Exception:
        pass

    requested = os.path.realpath(os.path.expanduser(models_dir))
    models_root = None
    for root in allowed_roots:
        root_str = os.path.realpath(str(root))
        if requested == root_str or requested.startswith(root_str + os.sep):
            models_root = root  # Use the trusted root, not the user-supplied path
            break
    if models_root is None:
        raise HTTPException(
            status_code = 403,
            detail = "Directory not allowed",
        )

    try:
        local_models = _scan_models_dir(models_root) + _scan_hf_cache(hf_cache_dir)

        deduped: dict[str, LocalModelInfo] = {}
        for model in local_models:
            if model.id not in deduped:
                deduped[model.id] = model

        models = sorted(
            deduped.values(),
            key = lambda item: (item.updated_at or 0),
            reverse = True,
        )

        return LocalModelListResponse(
            models_dir = str(models_root),
            hf_cache_dir = str(hf_cache_dir),
            models = models,
        )
    except Exception as e:
        logger.error(f"Error listing local models: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to list local models: {str(e)}",
        )


@router.get("/list")
async def list_models(
    current_subject: str = Depends(get_current_subject),
):
    """
    List available models (default models and loaded models).

    This endpoint returns the default models and any currently loaded models.
    """
    try:
        inference_backend = get_inference_backend()

        # Get default models
        default_models = inference_backend.default_models

        # Get loaded models
        loaded_models = []
        for model_name, model_data in inference_backend.models.items():
            _is_vision = model_data.get("is_vision", False)
            _audio_type = model_data.get("audio_type")
            model_info = ModelDetails(
                id = model_name,
                name = model_name.split("/")[-1] if "/" in model_name else model_name,
                is_vision = _is_vision,
                is_lora = model_data.get("is_lora", False),
                is_audio = model_data.get("is_audio", False),
                audio_type = _audio_type,
                has_audio_input = model_data.get("has_audio_input", False),
                model_type = derive_model_type(_is_vision, _audio_type),
            )
            loaded_models.append(model_info)

        # Combine default and loaded models
        all_models = []
        seen_ids = set()

        # Add default models
        for model_id in default_models:
            if model_id not in seen_ids:
                model_info = ModelDetails(
                    id = model_id,
                    name = model_id.split("/")[-1] if "/" in model_id else model_id,
                )
                all_models.append(model_info)
                seen_ids.add(model_id)

        # Add loaded models
        for model_info in loaded_models:
            if model_info.id not in seen_ids:
                all_models.append(model_info)
                seen_ids.add(model_info.id)

        return ModelListResponse(models = all_models, default_models = default_models)

    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to list models: {str(e)}")


@router.get("/config/{model_name:path}")
async def get_model_config(
    model_name: str,
    hf_token: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    """
    Get configuration for a specific model.

    This endpoint wraps the backend load_model_defaults function.
    """
    try:
        from utils.models.model_config import is_local_path

        if not is_local_path(model_name):
            model_name = model_name.lower()

        logger.info(f"Getting model config for: {model_name}")
        from utils.models.model_config import detect_audio_type

        # Load model defaults from backend
        config_dict = load_model_defaults(model_name)

        # Detect model capabilities (pass HF token for gated models)
        is_vision = is_vision_model(model_name)
        is_embedding = is_embedding_model(model_name, hf_token = hf_token)
        audio_type = detect_audio_type(model_name, hf_token = hf_token)

        # Check if it's a LoRA adapter
        is_lora = False
        base_model = None
        try:
            model_config = ModelConfig.from_identifier(model_name)
            is_lora = model_config.is_lora
            base_model = model_config.base_model if is_lora else None
        except Exception:
            pass

        logger.info(
            f"Model config result for {model_name}: is_vision={is_vision}, is_embedding={is_embedding}, audio_type={audio_type}, is_lora={is_lora}"
        )
        return ModelDetails(
            id = model_name,
            model_name = model_name,
            config = config_dict,
            is_vision = is_vision,
            is_embedding = is_embedding,
            is_lora = is_lora,
            is_audio = audio_type is not None,
            audio_type = audio_type,
            has_audio_input = is_audio_input_type(audio_type),
            model_type = derive_model_type(is_vision, audio_type, is_embedding),
            base_model = base_model,
        )

    except Exception as e:
        logger.error(f"Error getting model config: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to get model config: {str(e)}"
        )


@router.get("/loras")
async def scan_loras(
    outputs_dir: str = Query(
        default = str(outputs_root()), description = "Directory to scan for LoRA adapters"
    ),
    exports_dir: str = Query(
        default = str(exports_root()), description = "Directory to scan for exported models"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    Scan for trained LoRA adapters and exported models.

    Returns both training outputs (from outputs_dir) and exported models
    (from exports_dir) in a single list, distinguished by source field.
    """
    try:
        resolved_outputs_dir = str(resolve_output_dir(outputs_dir))
        resolved_exports_dir = str(resolve_export_dir(exports_dir))
        lora_list = []

        # Scan training outputs
        trained_loras = scan_trained_loras(outputs_dir = resolved_outputs_dir)
        for display_name, adapter_path in trained_loras:
            base_model = get_base_model_from_lora(adapter_path)
            lora_list.append(
                LoRAInfo(
                    display_name = display_name,
                    adapter_path = adapter_path,
                    base_model = base_model,
                    source = "training",
                )
            )

        # Scan exported models (merged, LoRA, base — skips GGUF)
        exported = scan_exported_models(exports_dir = resolved_exports_dir)
        for display_name, model_path, export_type, base_model in exported:
            lora_list.append(
                LoRAInfo(
                    display_name = display_name,
                    adapter_path = model_path,
                    base_model = base_model,
                    source = "exported",
                    export_type = export_type,
                )
            )

        return LoRAScanResponse(loras = lora_list, outputs_dir = resolved_outputs_dir)

    except Exception as e:
        logger.error(f"Error scanning LoRAs: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to scan LoRA adapters: {str(e)}"
        )


@router.get("/loras/{lora_path:path}/base-model", response_model = LoRABaseModelResponse)
async def get_lora_base_model(
    lora_path: str,
    current_subject: str = Depends(get_current_subject),
):
    """
    Get the base model for a LoRA adapter.

    This endpoint wraps the backend get_base_model_from_lora function.
    """
    try:
        base_model = get_base_model_from_lora(lora_path)

        if base_model is None:
            raise HTTPException(
                status_code = 404,
                detail = f"Could not determine base model for LoRA: {lora_path}",
            )

        return LoRABaseModelResponse(
            lora_path = lora_path,
            base_model = base_model,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting LoRA base model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to get base model: {str(e)}"
        )


@router.get("/check-vision/{model_name:path}", response_model = VisionCheckResponse)
async def check_vision_model(
    model_name: str,
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a model is a vision model.

    This endpoint wraps the backend is_vision_model function.
    """
    try:
        logger.info(f"Checking if vision model: {model_name}")
        is_vision = is_vision_model(model_name)

        logger.info(f"Vision check result for {model_name}: is_vision={is_vision}")
        return VisionCheckResponse(
            model_name = model_name,
            is_vision = is_vision,
        )

    except Exception as e:
        logger.error(f"Error checking vision model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to check vision model: {str(e)}"
        )


@router.get("/check-embedding/{model_name:path}", response_model = EmbeddingCheckResponse)
async def check_embedding_model(
    model_name: str,
    hf_token: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a model is an embedding model.

    This endpoint wraps the backend is_embedding_model function.
    """
    try:
        logger.info(f"Checking if embedding model: {model_name}")
        is_embedding = is_embedding_model(model_name, hf_token = hf_token)

        logger.info(
            f"Embedding check result for {model_name}: is_embedding={is_embedding}"
        )
        return EmbeddingCheckResponse(
            model_name = model_name,
            is_embedding = is_embedding,
        )

    except Exception as e:
        logger.error(f"Error checking embedding model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to check embedding model: {str(e)}"
        )


@router.get("/gguf-variants", response_model = GgufVariantsResponse)
async def get_gguf_variants(
    repo_id: str = Query(
        ..., description = "HuggingFace repo ID (e.g. 'unsloth/gemma-3-4b-it-GGUF')"
    ),
    hf_token: Optional[str] = Query(
        None, description = "HuggingFace token for private repos"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    List available GGUF quantization variants for a HuggingFace repo.

    Returns all available quantization variants (Q4_K_M, Q8_0, BF16, etc.)
    with file sizes, whether the model supports vision, and the recommended
    default variant.
    """
    try:
        variants, has_vision = list_gguf_variants(repo_id, hf_token = hf_token)

        # Determine default variant
        filenames = [v.filename for v in variants]
        best = _pick_best_gguf(filenames)
        default_variant = _extract_quant_label(best) if best else None

        # Check which variants are already downloaded in the HF cache
        # HF cache dir uses the exact case from the repo_id at download time,
        # which may differ from the canonical HF repo_id, so do a
        # case-insensitive match.
        cached_files: set = set()
        try:
            import re as _re
            from huggingface_hub import constants as hf_constants

            # Sanitize repo_id: must be "owner/name" with safe chars only
            if not _re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
                raise ValueError(f"Invalid repo_id format: {repo_id}")

            cache_dir = Path(hf_constants.HF_HUB_CACHE)
            target = f"models--{repo_id.replace('/', '--')}".lower()
            for entry in cache_dir.iterdir():
                if entry.name.lower() == target:
                    snapshots = entry / "snapshots"
                    if snapshots.is_dir():
                        for snap in snapshots.iterdir():
                            for f in snap.rglob("*.gguf"):
                                cached_files.add(f.name)
                    break
        except Exception:
            pass

        return GgufVariantsResponse(
            repo_id = repo_id,
            variants = [
                GgufVariantDetail(
                    filename = v.filename,
                    quant = v.quant,
                    size_bytes = v.size_bytes,
                    downloaded = Path(v.filename).name in cached_files,
                )
                for v in variants
            ],
            has_vision = has_vision,
            default_variant = default_variant,
        )

    except Exception as e:
        logger.error(f"Error listing GGUF variants for '{repo_id}': {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to list GGUF variants: {str(e)}",
        )


@router.get("/gguf-download-progress")
async def get_gguf_download_progress(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    expected_bytes: int = Query(0, description = "Expected total download size in bytes"),
    current_subject: str = Depends(get_current_subject),
):
    """Return download progress by checking current size of cached GGUF files."""
    import re as _re
    try:
        if not _re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
            return {"downloaded_bytes": 0, "expected_bytes": expected_bytes, "progress": 0}

        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        target = f"models--{repo_id.replace('/', '--')}".lower()
        downloaded_bytes = 0
        for entry in cache_dir.iterdir():
            if entry.name.lower() == target:
                # Sum .gguf files in snapshots + incomplete downloads in blobs
                for f in entry.rglob("*.gguf"):
                    downloaded_bytes += f.stat().st_size
                # Also check incomplete downloads (blobs without extension)
                blobs_dir = entry / "blobs"
                if blobs_dir.is_dir():
                    for f in blobs_dir.iterdir():
                        if f.is_file():
                            downloaded_bytes += f.stat().st_size
                break

        progress = min(downloaded_bytes / expected_bytes, 1.0) if expected_bytes > 0 else 0
        return {
            "downloaded_bytes": downloaded_bytes,
            "expected_bytes": expected_bytes,
            "progress": round(progress, 3),
        }
    except Exception:
        return {"downloaded_bytes": 0, "expected_bytes": expected_bytes, "progress": 0}


@router.get("/cached-gguf")
async def list_cached_gguf(
    current_subject: str = Depends(get_current_subject),
):
    """List GGUF repos that have already been downloaded to the HF cache."""
    try:
        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        cached = []
        if cache_dir.is_dir():
            for entry in sorted(cache_dir.iterdir()):
                if not entry.name.startswith("models--"):
                    continue
                # models--unsloth--Qwen3-8B-GGUF -> unsloth/Qwen3-8B-GGUF
                parts = entry.name.split("--", 1)
                if len(parts) < 2:
                    continue
                repo_id = parts[1].replace("--", "/")
                if not repo_id.lower().endswith("-gguf"):
                    continue
                # Check if there are actual .gguf files in snapshots
                snapshots = entry / "snapshots"
                if not snapshots.is_dir():
                    continue
                total_size = 0
                has_gguf = False
                for snap in snapshots.iterdir():
                    for f in snap.rglob("*.gguf"):
                        has_gguf = True
                        total_size += f.stat().st_size
                if has_gguf:
                    cached.append(
                        {
                            "repo_id": repo_id,
                            "size_bytes": total_size,
                            "cache_path": str(entry),
                        }
                    )
        return {"cached": cached}
    except Exception as e:
        logger.error(f"Error listing cached GGUF repos: {e}", exc_info = True)
        return {"cached": []}


@router.get("/checkpoints", response_model = CheckpointListResponse)
async def list_checkpoints(
    outputs_dir: str = Query(
        default = str(outputs_root()),
        description = "Directory to scan for checkpoints",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    List available checkpoints in the outputs directory.

    Scans the outputs folder for training runs and their checkpoints.
    """
    try:
        resolved_outputs_dir = str(resolve_output_dir(outputs_dir))
        raw_models = scan_checkpoints(outputs_dir = resolved_outputs_dir)

        models = [
            ModelCheckpoints(
                name = model_name,
                checkpoints = [
                    CheckpointInfo(display_name = display_name, path = path, loss = loss)
                    for display_name, path, loss in checkpoints
                ],
                base_model = metadata.get("base_model"),
                peft_type = metadata.get("peft_type"),
                lora_rank = metadata.get("lora_rank"),
            )
            for model_name, checkpoints, metadata in raw_models
        ]

        return CheckpointListResponse(
            outputs_dir = resolved_outputs_dir,
            models = models,
        )
    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to list checkpoints: {str(e)}",
        )
