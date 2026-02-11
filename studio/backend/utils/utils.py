"""
Shared backend utilities
"""
import gradio as gr
import os
import platform
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import tempfile


logger = logging.getLogger(__name__)

# ========== Device Detection & Management ==========

def get_device() -> str:
    """
    Detect the best available compute device.

    Returns:
        "cuda" on NVIDIA GPUs, "mps" on Apple Silicon, "cpu" otherwise.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
pass


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon hardware."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"
pass


def clear_gpu_cache():
    """
    Clear GPU memory cache for the current device.
    Safe to call on any platform — no-ops gracefully when the backend is unavailable.
    """
    import torch
    import gc

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
pass

@contextmanager
def without_hf_auth():
    """
    Context manager to temporarily disable HuggingFace authentication.

    Usage:
        with without_hf_auth():
            # Code that should run without cached tokens
            model_info(model_name, token=None)
    """
    # Save environment variables
    saved_env = {}
    env_vars = ['HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'HF_HOME']
    for var in env_vars:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    # Save disable flag
    saved_disable = os.environ.get('HF_HUB_DISABLE_IMPLICIT_TOKEN')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

    # Move token files temporarily
    token_files = []
    token_locations = [
        Path.home() / '.cache' / 'huggingface' / 'token',
        Path.home() / '.huggingface' / 'token'
    ]

    for token_loc in token_locations:
        if token_loc.exists():
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.close()
            shutil.move(str(token_loc), temp.name)
            token_files.append((token_loc, temp.name))

    try:
        yield
    finally:
        # Restore tokens
        for original, temp in token_files:
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(temp, str(original))
            except Exception as e:
                logger.error(f"Failed to restore token {original}: {e}")

        # Restore environment
        for var, value in saved_env.items():
            os.environ[var] = value

        if saved_disable is not None:
            os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = saved_disable
        else:
            os.environ.pop('HF_HUB_DISABLE_IMPLICIT_TOKEN', None)
pass

def format_error_message(error: Exception, model_name: str) -> str:
    """
    Format user-friendly error messages for common issues.

    Args:
        error: The exception that occurred
        model_name: Name of the model being loaded

    Returns:
        User-friendly error string
    """
    error_str = str(error).lower()
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name

    if "repository not found" in error_str or "404" in error_str:
        return f"Model '{model_short}' not found. Check the model name."

    if "401" in error_str or "unauthorized" in error_str:
        return f"Authentication failed for '{model_short}'. Please provide a valid HF token."

    if "gated" in error_str or "access to model" in error_str:
        return f"Model '{model_short}' requires authentication. Please provide a valid HF token."

    if "invalid user token" in error_str:
        return "Invalid HF token. Please check your token and try again."

    if "memory" in error_str or "cuda" in error_str or "mps" in error_str or "out of memory" in error_str:
        device = get_device()
        device_label = {"cuda": "GPU", "mps": "Apple Silicon GPU", "cpu": "system"}.get(device, "GPU")
        return f"Not enough {device_label} memory to load '{model_short}'. Try a smaller model or free memory."

    # Generic fallback
    return str(error)
pass

def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information.
    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU-only environments.
    """
    import torch

    device_type = get_device()

    # ---- CUDA path ----
    if device_type == "cuda":
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)

            total = props.total_memory
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)

            return {
                "available": True,
                "backend": "cuda",
                "device": device,
                "device_name": props.name,
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "free_gb": (total - allocated) / (1024**3),
                "utilization_pct": (allocated / total) * 100,
            }
        except Exception as e:
            logger.error(f"Error getting CUDA GPU info: {e}")
            return {"available": False, "backend": "cuda", "error": str(e)}

    # ---- MPS path (Apple Silicon) ----
    if device_type == "mps":
        try:
            allocated = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
            # MPS doesn't expose total VRAM directly — use unified memory from psutil as a proxy
            import psutil
            total = psutil.virtual_memory().total

            return {
                "available": True,
                "backend": "mps",
                "device": 0,
                "device_name": f"Apple Silicon ({platform.processor() or platform.machine()})",
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": 0,  # MPS doesn't have a separate reserved pool
                "free_gb": (total - allocated) / (1024**3),
                "utilization_pct": (allocated / total) * 100 if total else 0,
            }
        except Exception as e:
            logger.error(f"Error getting MPS GPU info: {e}")
            return {"available": False, "backend": "mps", "error": str(e)}

    # ---- CPU-only ----
    return {"available": False, "backend": "cpu"}
pass

def log_gpu_memory(context: str):
    """Log GPU memory usage with context."""
    memory_info = get_gpu_memory_info()
    if memory_info.get("available"):
        backend = memory_info.get("backend", "unknown").upper()
        device_name = memory_info.get("device_name", "")
        label = f"{backend}" + (f" ({device_name})" if device_name else "")
        logger.info(
            f"GPU Memory [{context}] {label}: "
            f"{memory_info['allocated_gb']:.2f}GB/{memory_info['total_gb']:.2f}GB "
            f"({memory_info['utilization_pct']:.1f}% used, "
            f"{memory_info['free_gb']:.2f}GB free)"
        )
    else:
        logger.info(f"GPU Memory [{context}]: No GPU available (CPU-only)")
pass

"""
Model utility functions - search, discovery, etc.
"""


def search_hf_models(search_query: str, hf_token: Optional[str] = None):
    """
    Search HuggingFace model hub.
    """
    import requests

    if not search_query or not search_query.strip():
        return gr.update(choices=[])

    # Simple debouncing: only search if query is at least 2 characters
    if len(search_query.strip()) < 2:
        return gr.update(choices=[])

    try:
        headers = {}
        if hf_token and hf_token.strip():
            headers["Authorization"] = f"Bearer {hf_token.strip()}"

        url = "https://huggingface.co/api/models"
        params = {
            "search": search_query,
            "pipeline_tag": "text-generation",
            "library": "transformers",
            "limit": 15,
            "sort": "downloads",
            "direction": -1
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            models = response.json()
            unsloth_results = []
            other_results = []

            for model in models:
                model_id = model.get("modelId", "")
                if model_id and "gguf" not in model_id.lower():
                    result = (f"{model_id}", model_id)

                    if model_id.startswith("unsloth/"):
                        unsloth_results.append(result)
                    else:
                        other_results.append(result)

            # Combine with unsloth models first
            search_results = unsloth_results + other_results
            return gr.update(choices=search_results)
        else:
            logger.warning(f"HF API returned status {response.status_code}")
            return gr.update(choices=[])

    except Exception as e:
        logger.warning(f"Model search failed: {e}")
        return gr.update(choices=[])
