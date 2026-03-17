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

import re as _re

_VALID_REPO_ID = _re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _is_valid_repo_id(repo_id: str) -> bool:
    return bool(_VALID_REPO_ID.fullmatch(repo_id))


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

        # Include active GGUF model (loaded via llama-server)
        from routes.inference import get_llama_cpp_backend

        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded and llama_backend.model_identifier:
            loaded_models.append(
                ModelDetails(
                    id = llama_backend.model_identifier,
                    name = llama_backend.model_identifier.split("/")[-1],
                    is_gguf = True,
                    is_vision = llama_backend.is_vision,
                    is_audio = getattr(llama_backend, "_is_audio", False),
                    audio_type = getattr(llama_backend, "_audio_type", None),
                )
            )

        # Combine default and loaded models
        all_models = []
        seen_ids = set()

        # Add default models
        for model_id in default_models:
            if model_id not in seen_ids:
                model_info = ModelDetails(
                    id = model_id,
                    name = model_id.split("/")[-1] if "/" in model_id else model_id,
                    is_gguf = model_id.upper().endswith("-GGUF"),
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


def _get_max_position_embeddings(config) -> Optional[int]:
    """Extract max_position_embeddings from a model config, checking text_config fallback."""
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    if hasattr(config, "text_config") and hasattr(
        config.text_config, "max_position_embeddings"
    ):
        return config.text_config.max_position_embeddings
    return None


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
        max_position_embeddings = None
        try:
            model_config = ModelConfig.from_identifier(model_name)
            is_lora = model_config.is_lora
            base_model = model_config.base_model if is_lora else None
            max_position_embeddings = _get_max_position_embeddings(model_config)
        except Exception:
            pass

        # Fallback: try AutoConfig directly if not found yet
        if max_position_embeddings is None:
            try:
                from transformers import AutoConfig as _AutoConfig

                _trust = model_name.lower().startswith("unsloth/")
                _ac = _AutoConfig.from_pretrained(
                    model_name, trust_remote_code = _trust, token = hf_token
                )
                max_position_embeddings = _get_max_position_embeddings(_ac)
            except Exception:
                pass

        logger.info(
            f"Model config result for {model_name}: is_vision={is_vision}, is_embedding={is_embedding}, audio_type={audio_type}, is_lora={is_lora}, max_position_embeddings={max_position_embeddings}"
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
            max_position_embeddings = max_position_embeddings,
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

        # Check which variants are fully downloaded in the HF cache.
        # For split GGUFs, ALL shards must be present -- sum cached bytes
        # per variant and compare against the expected total.
        # HF cache dir uses the exact case from the repo_id at download time,
        # which may differ from the canonical HF repo_id, so do a
        # case-insensitive match.
        cached_bytes_by_quant: dict[str, int] = {}
        try:
            import re as _re
            from huggingface_hub import constants as hf_constants

            # Sanitize repo_id: must be "owner/name" with safe chars only
            if not _is_valid_repo_id(repo_id):
                raise ValueError(f"Invalid repo_id format: {repo_id}")

            cache_dir = Path(hf_constants.HF_HUB_CACHE)
            target = f"models--{repo_id.replace('/', '--')}".lower()
            for entry in cache_dir.iterdir():
                if entry.name.lower() == target:
                    snapshots = entry / "snapshots"
                    if snapshots.is_dir():
                        for snap in snapshots.iterdir():
                            for f in snap.rglob("*.gguf"):
                                q = _extract_quant_label(f.name)
                                cached_bytes_by_quant[q] = (
                                    cached_bytes_by_quant.get(q, 0) + f.stat().st_size
                                )
                    break
        except Exception:
            pass

        def _is_fully_downloaded(variant) -> bool:
            cached = cached_bytes_by_quant.get(variant.quant, 0)
            if cached == 0 or variant.size_bytes == 0:
                return False
            # Allow small rounding tolerance (symlinks vs real sizes)
            return cached >= variant.size_bytes * 0.99

        return GgufVariantsResponse(
            repo_id = repo_id,
            variants = [
                GgufVariantDetail(
                    filename = v.filename,
                    quant = v.quant,
                    size_bytes = v.size_bytes,
                    downloaded = _is_fully_downloaded(v),
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
    variant: str = Query("", description = "Quantization variant (e.g. UD-TQ1_0)"),
    expected_bytes: int = Query(0, description = "Expected total download size in bytes"),
    current_subject: str = Depends(get_current_subject),
):
    """Return download progress by checking cached GGUF files for a specific variant.

    Tracks completed shard downloads in snapshots and in-progress downloads
    in the blobs directory (incomplete files).
    """
    try:
        if not _is_valid_repo_id(repo_id):
            return {
                "downloaded_bytes": 0,
                "expected_bytes": expected_bytes,
                "progress": 0,
            }

        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        target = f"models--{repo_id.replace('/', '--')}".lower()
        variant_lower = variant.lower().replace("-", "").replace("_", "")
        downloaded_bytes = 0
        in_progress_bytes = 0
        for entry in cache_dir.iterdir():
            if entry.name.lower() == target:
                # Count completed .gguf files matching this variant in snapshots
                for f in entry.rglob("*.gguf"):
                    fname = f.name.lower().replace("-", "").replace("_", "")
                    if not variant_lower or variant_lower in fname:
                        downloaded_bytes += f.stat().st_size
                # Check blobs for in-progress downloads (.incomplete files)
                blobs_dir = entry / "blobs"
                if blobs_dir.is_dir():
                    for f in blobs_dir.iterdir():
                        if f.is_file() and f.name.endswith(".incomplete"):
                            in_progress_bytes += f.stat().st_size
                break

        total_progress_bytes = downloaded_bytes + in_progress_bytes
        progress = (
            min(total_progress_bytes / expected_bytes, 0.99)
            if expected_bytes > 0
            else 0
        )
        # Only report 1.0 when all bytes are in completed files (not in-progress)
        if expected_bytes > 0 and downloaded_bytes >= expected_bytes:
            progress = 1.0
        return {
            "downloaded_bytes": total_progress_bytes,
            "expected_bytes": expected_bytes,
            "progress": round(progress, 3),
        }
    except Exception:
        return {"downloaded_bytes": 0, "expected_bytes": expected_bytes, "progress": 0}


@router.get("/download-progress")
async def get_download_progress(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    current_subject: str = Depends(get_current_subject),
):
    """Return download progress for any HuggingFace model repo.

    Checks the local HF cache for completed blobs and in-progress
    (.incomplete) downloads. Uses the HF API to determine the expected
    total size on the first call, then caches it for subsequent polls.
    """
    _empty = {"downloaded_bytes": 0, "expected_bytes": 0, "progress": 0}
    try:
        if not _is_valid_repo_id(repo_id):
            return _empty

        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        target = f"models--{repo_id.replace('/', '--')}".lower()
        completed_bytes = 0
        in_progress_bytes = 0

        for entry in cache_dir.iterdir():
            if entry.name.lower() != target:
                continue
            blobs_dir = entry / "blobs"
            if not blobs_dir.is_dir():
                break
            for f in blobs_dir.iterdir():
                if not f.is_file():
                    continue
                if f.name.endswith(".incomplete"):
                    in_progress_bytes += f.stat().st_size
                else:
                    completed_bytes += f.stat().st_size
            break

        downloaded_bytes = completed_bytes + in_progress_bytes
        if downloaded_bytes == 0:
            return _empty

        # Get expected size from HF API (cached per repo_id)
        expected_bytes = _get_repo_size_cached(repo_id)
        if expected_bytes <= 0:
            # Cannot determine total; report bytes only, no percentage
            return {
                "downloaded_bytes": downloaded_bytes,
                "expected_bytes": 0,
                "progress": 0,
            }

        # Use 95% threshold for completion (blob deduplication can make
        # completed_bytes differ slightly from expected_bytes).
        # Do NOT use "no .incomplete files" as a completion signal --
        # HF downloads files sequentially, so between files there are
        # no .incomplete files even though the download is far from done.
        if completed_bytes >= expected_bytes * 0.95:
            progress = 1.0
        else:
            progress = min(downloaded_bytes / expected_bytes, 0.99)
        return {
            "downloaded_bytes": downloaded_bytes,
            "expected_bytes": expected_bytes,
            "progress": round(progress, 3),
        }
    except Exception as e:
        logger.warning(f"Error checking download progress for {repo_id}: {e}")
        return _empty


_repo_size_cache: dict[str, int] = {}


def _get_repo_size_cached(repo_id: str) -> int:
    if repo_id in _repo_size_cache:
        return _repo_size_cache[repo_id]
    try:
        from huggingface_hub import model_info as hf_model_info

        info = hf_model_info(repo_id, token = None, files_metadata = True)
        total = sum(s.size for s in info.siblings if s.size)
        _repo_size_cache[repo_id] = total
        return total
    except Exception as e:
        logger.warning(f"Failed to get repo size for {repo_id}: {e}")
        return 0


@router.get("/cached-gguf")
async def list_cached_gguf(
    current_subject: str = Depends(get_current_subject),
):
    """List GGUF repos that have already been downloaded to the HF cache.

    Uses scan_cache_dir() for proper repo IDs, then deduplicates by
    lowercased key (HF cache dirs are lowercased but the canonical repo
    ID preserves casing).
    """
    try:
        from huggingface_hub import scan_cache_dir

        hf_cache = scan_cache_dir()
        seen_lower: dict[str, dict] = {}
        for repo_info in hf_cache.repos:
            if repo_info.repo_type != "model":
                continue
            repo_id = repo_info.repo_id
            if not repo_id.upper().endswith("-GGUF"):
                continue
            # Check for actual .gguf files and sum sizes
            total_size = 0
            has_gguf = False
            for revision in repo_info.revisions:
                for f in revision.files:
                    if f.file_name.endswith(".gguf"):
                        has_gguf = True
                        total_size += f.size_on_disk
            if not has_gguf:
                continue
            # Deduplicate: keep the entry with the most data
            key = repo_id.lower()
            existing = seen_lower.get(key)
            if existing is None or total_size > existing["size_bytes"]:
                seen_lower[key] = {
                    "repo_id": repo_id,
                    "size_bytes": total_size,
                    "cache_path": str(repo_info.repo_path),
                }
        cached = sorted(seen_lower.values(), key = lambda c: c["repo_id"])
        return {"cached": cached}
    except Exception as e:
        logger.error(f"Error listing cached GGUF repos: {e}", exc_info = True)
        return {"cached": []}


@router.get("/cached-models")
async def list_cached_models(
    current_subject: str = Depends(get_current_subject),
):
    """List non-GGUF model repos that have been downloaded to the HF cache."""
    try:
        from huggingface_hub import scan_cache_dir

        hf_cache = scan_cache_dir()
        seen_lower: dict[str, dict] = {}
        for repo_info in hf_cache.repos:
            if repo_info.repo_type != "model":
                continue
            repo_id = repo_info.repo_id
            if repo_id.upper().endswith("-GGUF"):
                continue
            total_size = sum(
                f.size_on_disk for rev in repo_info.revisions for f in rev.files
            )
            if total_size == 0:
                continue
            key = repo_id.lower()
            existing = seen_lower.get(key)
            if existing is None or total_size > existing["size_bytes"]:
                seen_lower[key] = {
                    "repo_id": repo_id,
                    "size_bytes": total_size,
                }
        cached = sorted(seen_lower.values(), key = lambda c: c["repo_id"])
        return {"cached": cached}
    except Exception as e:
        logger.error(f"Error listing cached models: {e}", exc_info = True)
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
