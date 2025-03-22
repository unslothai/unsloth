from .selector import (
    AttentionBackendRegistry, 
    select_attention_backend,
    register_attention_backend,
)
from .layer import UnifiedAttention, create_attention_mechanism
from .backends import (
    AttentionBackend,
    FlashAttentionBackend,
    FlashAttentionSoftcapBackend,
    XFormersBackend,
    SDPABackend,
    FlexAttentionBackend,
    VanillaAttentionBackend,
    VanillaSoftcappingAttentionBackend,
)

# Register the backends
register_attention_backend("flash_attention", lambda: FlashAttentionBackend.is_available())
register_attention_backend("flash_attention_softcap", lambda: FlashAttentionSoftcapBackend.is_available())
register_attention_backend("xformers", lambda: XFormersBackend.is_available())
register_attention_backend("sdpa", lambda: SDPABackend.is_available())
register_attention_backend("flex_attention", lambda: FlexAttentionBackend.is_available())
register_attention_backend("vanilla", lambda: True)  # Always available
register_attention_backend("vanilla_softcap", lambda: True)  # Always available

__all__ = [
    # Selector functions
    "AttentionBackendRegistry",
    "select_attention_backend",
    "register_attention_backend",
    
    # Unified attention interface
    "UnifiedAttention",
    "create_attention_mechanism",
    
    # Backend classes
    "AttentionBackend",
    "FlashAttentionBackend",
    "FlashAttentionSoftcapBackend",
    "XFormersBackend",
    "SDPABackend",
    "FlexAttentionBackend",
    "VanillaAttentionBackend",
    "VanillaSoftcappingAttentionBackend",
]