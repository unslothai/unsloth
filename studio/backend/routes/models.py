"""
Model Management API routes
"""
import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import logging

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
        scan_checkpoints,
        list_gguf_variants,
        ModelConfig,
    )
    from utils.models.model_config import _pick_best_gguf, _extract_quant_label
    from core.inference import get_inference_backend
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
        scan_checkpoints,
        list_gguf_variants,
        ModelConfig,
    )
    from utils.models.model_config import _pick_best_gguf, _extract_quant_label
    from core.inference import get_inference_backend

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
from models.models import GgufVariantDetail, GgufVariantsResponse
from models.responses import LoRABaseModelResponse, VisionCheckResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Configure logger
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
                id=str(child),
                display_name=child.name,
                path=str(child),
                source="models_dir",
                updated_at=updated_at,
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
                    id=str(gguf_file),
                    display_name=gguf_file.stem,
                    path=str(gguf_file),
                    source="models_dir",
                    updated_at=updated_at,
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

        repo_name = repo_dir.name[len("models--"):]
        if not repo_name:
            continue
        model_id = repo_name.replace("--", "/")

        try:
            updated_at = repo_dir.stat().st_mtime
        except OSError:
            updated_at = None

        found.append(
            LocalModelInfo(
                id=model_id,
                model_id=model_id,
                display_name=model_id.split("/")[-1],
                path=str(repo_dir),
                source="hf_cache",
                updated_at=updated_at,
            ),
        )
    return found


@router.get("/local", response_model=LocalModelListResponse)
async def list_local_models(
    models_dir: str = Query(default="./models", description="Directory to scan for local model folders"),
    current_subject: str = Depends(get_current_subject),
):
    """
    List local model candidates from custom models dir and HF cache.
    """
    try:
        models_root = Path(models_dir).expanduser().resolve()
        hf_cache_dir = _resolve_hf_cache_dir()
        local_models = _scan_models_dir(models_root) + _scan_hf_cache(hf_cache_dir)

        deduped: dict[str, LocalModelInfo] = {}
        for model in local_models:
            if model.id not in deduped:
                deduped[model.id] = model

        models = sorted(
            deduped.values(),
            key=lambda item: (item.updated_at or 0),
            reverse=True,
        )

        return LocalModelListResponse(
            models_dir=str(models_root),
            hf_cache_dir=str(hf_cache_dir),
            models=models,
        )
    except Exception as e:
        logger.error(f"Error listing local models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list local models: {str(e)}",
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
            model_info = ModelDetails(
                id=model_name,
                name=model_name.split("/")[-1] if "/" in model_name else model_name,
                is_vision=model_data.get("is_vision", False),
                is_lora=model_data.get("is_lora", False)
            )
            loaded_models.append(model_info)
        
        # Combine default and loaded models
        all_models = []
        seen_ids = set()
        
        # Add default models
        for model_id in default_models:
            if model_id not in seen_ids:
                model_info = ModelDetails(
                    id=model_id,
                    name=model_id.split("/")[-1] if "/" in model_id else model_id
                )
                all_models.append(model_info)
                seen_ids.add(model_id)
        
        # Add loaded models
        for model_info in loaded_models:
            if model_info.id not in seen_ids:
                all_models.append(model_info)
                seen_ids.add(model_info.id)
        
        return ModelListResponse(
            models=all_models,
            default_models=default_models
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/config/{model_name:path}")
async def get_model_config(
    model_name: str,
    current_subject: str = Depends(get_current_subject),
):
    """
    Get configuration for a specific model.
    
    This endpoint wraps the backend load_model_defaults function.
    """
    try:
        logger.info(f"Getting model config for: {model_name}")
        # Load model defaults from backend
        config_dict = load_model_defaults(model_name)
        
        # Check if it's a vision model
        is_vision = is_vision_model(model_name)
        
        # Check if it's a LoRA adapter
        is_lora = False
        base_model = None
        
        # Try to create ModelConfig to get more info
        try:
            model_config = ModelConfig.from_identifier(model_name)
            is_lora = model_config.is_lora
            base_model = model_config.base_model if is_lora else None
        except Exception:
            # If ModelConfig creation fails, use defaults
            pass
        
        logger.info(f"Model config result for {model_name}: is_vision={is_vision}, is_lora={is_lora}, base_model={base_model}")
        return ModelDetails(
            id=model_name,
            model_name=model_name,
            config=config_dict,
            is_vision=is_vision,
            is_lora=is_lora,
            base_model=base_model,
        )
        
    except Exception as e:
        logger.error(f"Error getting model config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model config: {str(e)}"
        )


@router.get("/loras")
async def scan_loras(
    outputs_dir: str = Query(default="./outputs", description="Directory to scan for LoRA adapters"),
    exports_dir: str = Query(default="./exports", description="Directory to scan for exported models"),
    current_subject: str = Depends(get_current_subject),
):
    """
    Scan for trained LoRA adapters and exported models.

    Returns both training outputs (from outputs_dir) and exported models
    (from exports_dir) in a single list, distinguished by source field.
    """
    try:
        lora_list = []

        # Scan training outputs
        trained_loras = scan_trained_loras(outputs_dir=outputs_dir)
        for display_name, adapter_path in trained_loras:
            base_model = get_base_model_from_lora(adapter_path)
            lora_list.append(LoRAInfo(
                display_name=display_name,
                adapter_path=adapter_path,
                base_model=base_model,
                source="training",
            ))

        # Scan exported models (merged, LoRA, base — skips GGUF)
        exported = scan_exported_models(exports_dir=exports_dir)
        for display_name, model_path, export_type, base_model in exported:
            lora_list.append(LoRAInfo(
                display_name=display_name,
                adapter_path=model_path,
                base_model=base_model,
                source="exported",
                export_type=export_type,
            ))

        return LoRAScanResponse(
            loras=lora_list,
            outputs_dir=outputs_dir
        )

    except Exception as e:
        logger.error(f"Error scanning LoRAs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan LoRA adapters: {str(e)}"
        )


@router.get("/loras/{lora_path:path}/base-model", response_model=LoRABaseModelResponse)
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
                status_code=404,
                detail=f"Could not determine base model for LoRA: {lora_path}"
            )
        
        return LoRABaseModelResponse(
            lora_path=lora_path,
            base_model=base_model,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting LoRA base model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get base model: {str(e)}"
        )


@router.get("/check-vision/{model_name:path}", response_model=VisionCheckResponse)
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
            model_name=model_name,
            is_vision=is_vision,
        )
        
    except Exception as e:
        logger.error(f"Error checking vision model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check vision model: {str(e)}"
        )

@router.get("/gguf-variants", response_model=GgufVariantsResponse)
async def get_gguf_variants(
    repo_id: str = Query(..., description="HuggingFace repo ID (e.g. 'unsloth/gemma-3-4b-it-GGUF')"),
    hf_token: Optional[str] = Query(None, description="HuggingFace token for private repos"),
    current_subject: str = Depends(get_current_subject),
):
    """
    List available GGUF quantization variants for a HuggingFace repo.

    Returns all available quantization variants (Q4_K_M, Q8_0, BF16, etc.)
    with file sizes, whether the model supports vision, and the recommended
    default variant.
    """
    try:
        variants, has_vision = list_gguf_variants(repo_id, hf_token=hf_token)

        # Determine default variant
        filenames = [v.filename for v in variants]
        best = _pick_best_gguf(filenames)
        default_variant = _extract_quant_label(best) if best else None

        return GgufVariantsResponse(
            repo_id=repo_id,
            variants=[
                GgufVariantDetail(
                    filename=v.filename,
                    quant=v.quant,
                    size_bytes=v.size_bytes,
                )
                for v in variants
            ],
            has_vision=has_vision,
            default_variant=default_variant,
        )

    except Exception as e:
        logger.error(f"Error listing GGUF variants for '{repo_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list GGUF variants: {str(e)}",
        )


@router.get("/checkpoints", response_model=CheckpointListResponse)
async def list_checkpoints(
    outputs_dir: str = Query(
        default="./outputs",
        description="Directory to scan for checkpoints",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    List available checkpoints in the outputs directory.

    Scans the outputs folder for training runs and their checkpoints.
    """
    try:
        raw_models = scan_checkpoints(outputs_dir=outputs_dir)

        models = [
            ModelCheckpoints(
                name=model_name,
                checkpoints=[
                    CheckpointInfo(display_name=display_name, path=path, loss=loss)
                    for display_name, path, loss in checkpoints
                ],
                base_model=metadata.get("base_model"),
                peft_type=metadata.get("peft_type"),
                lora_rank=metadata.get("lora_rank"),
            )
            for model_name, checkpoints, metadata in raw_models
        ]

        return CheckpointListResponse(
            outputs_dir=outputs_dir,
            models=models,
        )
    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list checkpoints: {str(e)}",
        )
