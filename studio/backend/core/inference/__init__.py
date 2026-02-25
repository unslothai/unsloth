"""
Inference submodule - Inference backend for model loading and generation
"""
from .inference import InferenceBackend, get_inference_backend
from .llama_cpp import LlamaCppBackend

__all__ = [
    'InferenceBackend',
    'get_inference_backend',
    'LlamaCppBackend',
]
