import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union
import logging

from .selector import select_attention_backend
from .backends import AttentionBackend

logger = logging.getLogger(__name__)

class UnifiedAttention(nn.Module):
    """
    Unified attention layer that delegates to the appropriate attention backend.
    This provides a consistent interface for all attention mechanisms in the codebase.
    """
    
    def __init__(
        self,
        config: Optional[Any] = None, 
        backend: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the unified attention layer.
        
        Args:
            config: Model configuration object
            backend: Name of the specific backend to use (if None, auto-select)
            **kwargs: Additional arguments to pass to the backend
        """
        super().__init__()
        
        backend_name, backend_cls = select_attention_backend(backend, config, **kwargs)
        logger.debug(f"Selected attention backend: {backend_name}")
        
        self.backend = backend_cls(**kwargs)
        self.backend_name = backend_name
        
        # Store config for potential future use
        self.config = config
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform attention using the selected backend.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_kv_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_kv_heads, seq_len, head_dim]
            causal_mask: Optional causal mask
            attention_mask: Optional attention mask
            **kwargs: Additional arguments to pass to the backend
            
        Returns:
            torch.Tensor: Output tensor after attention
        """
        return self.backend.forward(
            query=query,
            key=key,
            value=value,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            **kwargs
        )
    
    @property
    def name(self) -> str:
        """Get the name of the currently used backend."""
        return self.backend_name
    
    def supports_feature(self, feature: str) -> bool:
        """Check if the current backend supports a specific feature."""
        return self.backend.supports_feature(feature)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the attention configuration to a dictionary."""
        return {
            "backend": self.backend_name,
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else None
        }

def create_attention_mechanism(
    config: Optional[Any] = None,
    backend: Optional[str] = None,
    **kwargs
) -> UnifiedAttention:
    """
    Factory function to create an attention mechanism.
    
    Args:
        config: Model configuration object
        backend: Name of the specific backend to use (if None, auto-select)
        **kwargs: Additional arguments to pass to the backend
        
    Returns:
        UnifiedAttention: An initialized attention layer
    """
    return UnifiedAttention(config, backend, **kwargs)
