"""
Model Management API routes
"""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import backend functions
try:
    from backend.utils import search_hf_models
    from backend.model_config import (
        scan_trained_loras,
        load_model_defaults,
        get_base_model_from_lora,
        is_vision_model,
        ModelConfig,
    )
    from backend.inference import get_inference_backend
except ImportError:
    # Fallback: try to import from parent directory
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from backend.utils import search_hf_models
    from backend.model_config import (
        scan_trained_loras,
        load_model_defaults,
        get_base_model_from_lora,
        is_vision_model,
        ModelConfig,
    )
    from backend.inference import get_inference_backend

from models.models import (
    ModelSearchRequest,
    ModelSearchResponse,
    ModelInfo,
    ModelListResponse,
    ModelConfigResponse,
    LoRAScanResponse,
    LoRAInfo,
)

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


@router.post("/search")
async def search_models(request: ModelSearchRequest):
    """
    Search for models on HuggingFace Hub.
    
    This endpoint wraps the backend search_hf_models function.
    """
    try:
        # Call backend search function
        gradio_update = search_hf_models(
            search_query=request.query,
            hf_token=request.hf_token
        )
        
        # Convert Gradio update to list of model IDs
        model_list = []
        if gradio_update and hasattr(gradio_update, 'choices'):
            choices = gradio_update.choices
        elif isinstance(gradio_update, dict) and 'choices' in gradio_update:
            choices = gradio_update['choices']
        elif isinstance(gradio_update, list):
            choices = gradio_update
        else:
            choices = []
        
        # Process choices - they may be tuples (display_name, model_id) or just strings
        for choice in choices:
            if isinstance(choice, tuple) and len(choice) >= 2:
                # Format: (display_name, model_id)
                model_id = choice[1] if len(choice) > 1 else choice[0]
                display_name = choice[0]
                model_info = ModelInfo(
                    id=model_id,
                    name=display_name
                )
            elif isinstance(choice, str):
                # Just a model ID string
                model_info = ModelInfo(id=choice)
            else:
                continue
            model_list.append(model_info)
        
        return ModelSearchResponse(
            models=model_list,
            total=len(model_list)
        )
        
    except Exception as e:
        logger.error(f"Error searching models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search models: {str(e)}"
        )


@router.get("/list")
async def list_models():
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
            model_info = ModelInfo(
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
                model_info = ModelInfo(
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
async def get_model_config(model_name: str):
    """
    Get configuration for a specific model.
    
    This endpoint wraps the backend load_model_defaults function.
    """
    try:
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
        
        return ModelConfigResponse(
            model_name=model_name,
            config=config_dict,
            is_vision=is_vision,
            is_lora=is_lora,
            base_model=base_model
        )
        
    except Exception as e:
        logger.error(f"Error getting model config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model config: {str(e)}"
        )


@router.get("/loras")
async def scan_loras(
    outputs_dir: str = Query(default="./outputs", description="Directory to scan for LoRA adapters")
):
    """
    Scan for trained LoRA adapters in the outputs directory.
    
    This endpoint wraps the backend scan_trained_loras function.
    """
    try:
        # Call backend scan function
        trained_loras = scan_trained_loras(outputs_dir=outputs_dir)
        
        # Convert to LoRAInfo objects
        lora_list = []
        for display_name, adapter_path in trained_loras:
            # Get base model if available
            base_model = get_base_model_from_lora(adapter_path)
            
            lora_info = LoRAInfo(
                display_name=display_name,
                adapter_path=adapter_path,
                base_model=base_model
            )
            lora_list.append(lora_info)
        
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


@router.get("/loras/{lora_path:path}/base-model")
async def get_lora_base_model(lora_path: str):
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
        
        return {
            "lora_path": lora_path,
            "base_model": base_model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting LoRA base model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get base model: {str(e)}"
        )


@router.get("/check-vision/{model_name:path}")
async def check_vision_model(model_name: str):
    """
    Check if a model is a vision model.
    
    This endpoint wraps the backend is_vision_model function.
    """
    try:
        is_vision = is_vision_model(model_name)
        
        return {
            "model_name": model_name,
            "is_vision": is_vision
        }
        
    except Exception as e:
        logger.error(f"Error checking vision model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check vision model: {str(e)}"
        )

