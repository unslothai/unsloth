import os
import logging
from functools import cache
from typing import Dict, Callable, Generator, Optional, Type, Any, Union, Tuple

import torch

logger = logging.getLogger(__name__)

# Registry to hold all attention backends
AttentionBackendRegistry = {}

def register_attention_backend(name: str, condition_fn: Callable[[], bool] = None):
    """
    Decorator to register an attention backend
    """
    def decorator(cls):
        if condition_fn is None or condition_fn():
            AttentionBackendRegistry[name] = cls
            logger.debug(f"Registeres attention backedn: {name}")
        return cls
    return decorator

def select_attention_backend(
        backend_name: Optional[str] = None,
        config: Optional[Any] = None,
        **kwargs
) -> Tuple[str, Any]:
    """
    Select the appropriate attention backend based on the provided criteria
    """

    if backend_name is not None:
        if backend_name in AttentionBackendRegistry:
            return backend_name, AttentionBackendRegistry[backend_name]
        else:
            logger.warning(f"Given backend not available: {backend_name}")

    available_backends = list(AttentionBackendRegistry.keys())

    priority_order = os.environ.get("GLOBAL_ATTENTION_PRIORITY", "").split(",")
    if not priority_order or priority_order == [""]:
        priority_order = ["flash_attention", "flex_attention", "xformers", "sdpa"]
    
    priority_order.extend(b for b in available_backends if b not in priority_order)

    if config is not None:
        # Softcapping need for Gemma 2
        has_softcapping = hasattr(config, "attn_logit_softcapping") and config.attn_logit_softcapping > 0
        if has_softcapping and "flash_attention_softcapping" in available_backends:
            return "flash_attention_softcapping", AttentionBackendRegistry["flash_attention_softcap"]
        
        # Sliding Window Attention
        has_swa = hasattr(config, "sliding_window") and config.sliding_window not in (None, "null")
        if has_swa:
            for backend in ["flash_attention", "flex_attention"]:
                if backend in available_backends:
                    return backend, AttentionBackendRegistry[backend]
                
    for backend in priority_order:
        if backend in available_backends:
            return backend, AttentionBackendRegistry[backend]
        
    raise RuntimeError("No attention backends available.")
