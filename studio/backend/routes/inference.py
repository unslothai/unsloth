"""
Inference API routes for model loading and text generation.
"""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import logging

# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import backend functions
try:
    from core.inference import get_inference_backend
    from utils.models import ModelConfig
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.inference import get_inference_backend
    from utils.models import ModelConfig

from models.inference import (
    LoadRequest,
    UnloadRequest,
    GenerateRequest,
    LoadResponse,
    UnloadResponse,
    InferenceStatusResponse,
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


@router.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest):
    """
    Load a model for inference.
    
    The model_path should be a clean identifier from GET /models/list.
    """
    try:
        backend = get_inference_backend()
        
        # Create config using clean factory method
        config = ModelConfig.from_identifier(
            model_id=request.model_path,
            hf_token=request.hf_token,
            is_lora=request.is_lora,
        )
        
        if not config:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model identifier: {request.model_path}"
            )
        
        # Load the model
        success = backend.load_model(
            config=config,
            max_seq_length=request.max_seq_length,
            load_in_4bit=request.load_in_4bit,
            hf_token=request.hf_token,
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {config.display_name}"
            )
        
        logger.info(f"Loaded model: {config.identifier}")
        
        return LoadResponse(
            status="loaded",
            model=config.identifier,
            display_name=config.display_name,
            is_vision=config.is_vision,
            is_lora=config.is_lora,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post("/unload", response_model=UnloadResponse)
async def unload_model(request: UnloadRequest):
    """
    Unload a model from memory.
    """
    try:
        backend = get_inference_backend()
        backend.unload_model(request.model_path)
        logger.info(f"Unloaded model: {request.model_path}")
        return UnloadResponse(status="unloaded", model=request.model_path)
        
    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )


@router.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate a chat response with Server-Sent Events (SSE) streaming.
    
    For vision models, provide image_base64 with the base64-encoded image.
    """
    backend = get_inference_backend()
    
    if not backend.active_model_name:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Call POST /inference/load first."
        )
    
    # Decode image if provided (for vision models)
    image = None
    if request.image_base64:
        try:
            import base64
            from PIL import Image
            from io import BytesIO
            
            # Check if current model supports vision
            model_info = backend.models.get(backend.active_model_name, {})
            if not model_info.get("is_vision"):
                raise HTTPException(
                    status_code=400,
                    detail="Image provided but current model is text-only. Load a vision model."
                )
            
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode image: {str(e)}"
            )
    
    async def stream():
        try:
            for chunk in backend.generate_chat_response(
                messages=request.messages,
                system_prompt=request.system_prompt,
                image=image,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_new_tokens=request.max_new_tokens,
                repetition_penalty=request.repetition_penalty,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/status", response_model=InferenceStatusResponse)
async def get_status():
    """
    Get current inference backend status.
    """
    try:
        backend = get_inference_backend()
        
        is_vision = False
        if backend.active_model_name:
            model_info = backend.models.get(backend.active_model_name, {})
            is_vision = model_info.get("is_vision", False)
        
        return InferenceStatusResponse(
            active_model=backend.active_model_name,
            is_vision=is_vision,
            loading=list(getattr(backend, 'loading_models', set())),
            loaded=list(backend.models.keys()),
        )
        
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}"
        )
