import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .core.inference.llama_cpp import load_model

logger = logging.getLogger(__name__)

router = APIRouter()


class LoadModelRequest(BaseModel):
    model_path: str


@router.post("/load")
async def load_model(request: LoadModelRequest):
    model_path = request.model_path
    try:
        model = load_model(model_path)
        if model is None:
            raise HTTPException(status_code = 500, detail = "Failed to load model")
        return JSONResponse(content = {"success": True}, status_code = 200)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code = 500, detail = "Failed to load model")
