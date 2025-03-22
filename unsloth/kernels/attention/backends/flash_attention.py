import torch
from typing import Optional, Any, Tuple, Dict
from .base import AttentionBackend
from ....models._utils import HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING

if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

class FlashAttentionBackend(AttentionBackend):
    """
    Flash Attention backend implementation.
    Uses Flash Attention for efficient attention computation.
    """
    
    def __init__(self, softmax_scale: float = None):
        self.softmax_scale = softmax_scale
    
    @classmethod
    def is_available(cls) -> bool:
        return HAS_FLASH_ATTENTION
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        window_size: Tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError("FlashAttentionBackend does not support attention masks. Use a different backend.")
            
        output = flash_attn_func(
            query, key, value,
            causal=True,
            window_size=window_size,
            softmax_scale=self.softmax_scale,
            dropout_p=dropout_p
        )
        
        batch_size, seq_len, num_heads, head_dim = output.shape
        output = output.reshape(batch_size, seq_len, num_heads * head_dim)
        
        return output
    
    @classmethod
    def supports_feature(cls, feature: str) -> bool:
        """
        Check if this backend supports a particular feature.
        """
        features = {
            "causal": True,
            "sliding_window": True,
            "attention_mask": False,
            "dropout": True,
            "custom_scale": True,
        }
        return features.get(feature, False)


class FlashAttentionSoftcapBackend(FlashAttentionBackend):
    """
    Flash Attention backend with softcapping support.
    Used primarily for Gemma 2 models.
    """
    
    def __init__(self, softmax_scale: float = None, softcap: float = None):
        """
        Initialize the Flash Attention Softcap backend.
        
        Args:
            softmax_scale: Optional scaling factor for the softmax operation
            softcap: Softcap value for attention logits
        """
        super().__init__(softmax_scale)
        self.softcap = softcap
    
    @classmethod
    def is_available(cls) -> bool:
        return HAS_FLASH_ATTENTION_SOFTCAPPING
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        window_size: Tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        softcap: float = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform attention using Flash Attention with softcapping.
        
        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            value: Value tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            causal_mask: Optional causal mask
            attention_mask: Optional attention mask
            window_size: Sliding window size as (backward, forward) tuple
            dropout_p: Dropout probability
            softcap: Softcap value for attention logits
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Output tensor after attention
        """
        if attention_mask is not None:
            raise ValueError("FlashAttentionSoftcapBackend does not support attention masks. Use a different backend.")
            
        softcap_value = softcap if softcap is not None else self.softcap
        
        output = flash_attn_func(
            query, key, value,
            causal=True,
            softcap=softcap_value,
            softmax_scale=self.softmax_scale,
            window_size=window_size,
            dropout_p=dropout_p
        )
        
        batch_size, seq_len, num_heads, head_dim = output.shape
        output = output.reshape(batch_size, seq_len, num_heads * head_dim)
        
        return output
    
    @classmethod
    def supports_feature(cls, feature: str) -> bool:
        """
        Check if this backend supports a particular feature.
        
        Args:
            feature: Name of the feature to check
            
        Returns:
            bool: True if the feature is supported, False otherwise
        """
        features = super().supports_feature(feature)
        if feature == "softcap":
            return True
        return features