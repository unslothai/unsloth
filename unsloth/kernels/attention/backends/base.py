from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Dict, Union
import torch

class AttentionBackend(ABC):
    """
    Base class for attention backends.
    """
    @classmethod
    def is_available(cls) -> bool:
        return True
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__