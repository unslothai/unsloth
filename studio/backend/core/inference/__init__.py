"""
Inference submodule - Inference backend for model loading and generation
"""
from .inference import InferenceBackend, get_inference_backend

__all__ = [
    'InferenceBackend',
    'get_inference_backend',
]
