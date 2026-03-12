# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CGGR Bridge for Unsloth integration.

Provides trainer patching to enable selective backpropagation via label masking.
"""

import torch
import torch.nn.functional as F
from functools import wraps
from typing import Optional, Dict, Any, Callable
import logging
import warnings

from .router import TruncatedRouter, create_truncated_router

logger = logging.getLogger(__name__)

__all__ = ["CGGRUnslothBridge", "patch_trainer_for_cggr"]


class CGGRUnslothBridge:
    """
    Bridge class for integrating CGGR with Unsloth trainers.
    
    Patches the trainer's compute_loss method to apply label masking
    before the forward pass, enabling selective gradient computation.
    
    Example:
        >>> from unsloth.cggr import CGGRUnslothBridge
        >>> trainer = SFTTrainer(...)
        >>> CGGRUnslothBridge.patch_trainer(trainer, min_tokens_ratio=0.25)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        min_tokens_ratio: float = 0.25,
        num_router_layers: int = 2,
        warmup_steps: int = 100,
        scoring: str = "entropy",
        dynamic_threshold: bool = True,
    ):
        """
        Initialize CGGR bridge.
        
        Args:
            model: The model being trained
            min_tokens_ratio: Minimum fraction of tokens to keep gradients for (0.25 = top 25% hardest)
            num_router_layers: Number of layers for the truncated router (default: 2)
            warmup_steps: Steps before enabling CGGR (train normally first)
            scoring: Scoring strategy ('entropy', 'margin', 'loss', 'combined')
            dynamic_threshold: Whether to adjust ratio based on batch confidence
        """
        self.model = model
        self.min_tokens_ratio = min_tokens_ratio
        self.num_router_layers = num_router_layers
        self.warmup_steps = warmup_steps
        self.scoring = scoring
        self.dynamic_threshold = dynamic_threshold
        
        # Create truncated router for difficulty scoring
        self.router = create_truncated_router(model, num_layers=num_router_layers)
        
        # Training state
        self.current_step = 0
        self.total_tokens_seen = 0
        self.hard_tokens_seen = 0
        
        logger.info(
            f"Initialized CGGR Bridge: min_ratio={min_tokens_ratio}, "
            f"router_layers={num_router_layers}, warmup={warmup_steps}"
        )
    
    @torch.inference_mode()
    def compute_difficulty_scores(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute difficulty scores for each token using the truncated router.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target labels [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            difficulty_scores: Per-token difficulty [batch, seq_len]
        """
        # Get logits from truncated router (fast forward pass)
        logits = self.router(input_ids, attention_mask=attention_mask)
        
        # Compute difficulty based on scoring strategy
        if self.scoring == "entropy":
            # High entropy = uncertain = hard
            # Use log_softmax for numerical stability (single fused kernel)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            scores = -torch.sum(probs * log_probs, dim=-1)
        elif self.scoring == "margin":
            # Small margin between top-2 = hard
            # topk is efficient - only partial sort needed
            top2 = torch.topk(logits, k=2, dim=-1).values
            scores = -(top2[..., 0] - top2[..., 1])  # Negative margin (high = hard)
        elif self.scoring == "loss":
            # High loss = hard - directly compute per-token loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            scores = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.shape)
            # Pad to match original sequence length
            scores = F.pad(scores, (0, 1), value=0)
        else:  # combined - efficient fused computation
            # Compute log_softmax once (fused kernel)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            
            # Entropy from log_probs (reuse computation)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            
            # Margin from topk
            top2 = torch.topk(logits, k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            
            # Normalize and combine - use in-place operations where possible
            entropy_mean = entropy.mean()
            entropy_std = entropy.std() + 1e-10
            margin_mean = margin.mean()
            margin_std = margin.std() + 1e-10
            
            # Combined score: high entropy OR small margin = hard
            scores = (entropy - entropy_mean) / entropy_std - (margin - margin_mean) / margin_std
        
        return scores
    
    def mask_easy_tokens(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """
        Mask easy tokens in labels with -100 to skip their gradients.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target labels [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            masked_labels: Labels with easy tokens set to -100
        """
        # During warmup, don't mask anything
        if self.current_step < self.warmup_steps:
            return labels
        
        # Clone labels to avoid modifying original
        masked_labels = labels.clone()
        
        # Compute difficulty scores
        scores = self.compute_difficulty_scores(input_ids, labels, attention_mask)
        
        # Get valid (non-ignored) token mask
        valid_mask = labels != -100
        
        # Compute ratio (avoid CPU sync by keeping on GPU)
        ratio = self.min_tokens_ratio
        if self.dynamic_threshold and valid_mask.any():
            # More confident batch → keep fewer tokens
            valid_scores = scores.masked_select(valid_mask)
            score_range = valid_scores.max() - valid_scores.min() + 1e-10
            mean_normalized = (valid_scores.mean() - valid_scores.min()) / score_range
            # Lower mean score = more confident = keep fewer tokens
            confidence = 1.0 - mean_normalized
            ratio = self.min_tokens_ratio + (1.0 - self.min_tokens_ratio) * (1.0 - confidence) * 0.5
            ratio = max(self.min_tokens_ratio, min(ratio.item(), 1.0))
        
        # Vectorized masking: compute per-sequence thresholds
        batch_size, seq_len = labels.shape
        
        # Set scores of invalid tokens to -inf so they're never selected as "hard"
        scores_for_threshold = scores.clone()
        scores_for_threshold.masked_fill_(~valid_mask, float('-inf'))
        
        # Count valid tokens per sequence
        valid_counts = valid_mask.sum(dim=1)  # [batch]
        
        # Compute number to keep per sequence
        num_keep = (valid_counts.float() * ratio).long().clamp(min=1)
        
        # For each sequence, find the threshold score (k-th largest)
        # Use topk to find scores we should keep
        max_valid = valid_counts.max().item()
        if max_valid > 0:
            # Sort scores descending to find threshold
            sorted_scores, _ = scores_for_threshold.sort(dim=1, descending=True)
            
            # Get threshold for each sequence (the num_keep-th highest score)
            # Clamp indices to valid range
            threshold_indices = (num_keep - 1).clamp(min=0, max=seq_len - 1)
            thresholds = sorted_scores.gather(1, threshold_indices.unsqueeze(1)).squeeze(1)  # [batch]
            
            # Mask tokens with scores below threshold
            below_threshold = scores < thresholds.unsqueeze(1)
            mask_tokens = below_threshold & valid_mask
            masked_labels.masked_fill_(mask_tokens, -100)
        
        # Update statistics (use item() only once at end)
        total = valid_mask.sum().item()
        kept = (masked_labels != -100).sum().item()
        self.total_tokens_seen += total
        self.hard_tokens_seen += kept
        
        return masked_labels
    
    def step(self):
        """Called after each training step to update internal state."""
        self.current_step += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get CGGR statistics for logging."""
        if self.total_tokens_seen == 0:
            return {"cggr/hard_ratio": 0.0, "cggr/step": self.current_step}
        return {
            "cggr/hard_ratio": self.hard_tokens_seen / self.total_tokens_seen,
            "cggr/step": self.current_step,
            "cggr/total_tokens": self.total_tokens_seen,
        }
    
    @classmethod
    def patch_trainer(
        cls,
        trainer,
        min_tokens_ratio: float = 0.25,
        num_router_layers: int = 2,
        warmup_steps: int = 100,
        scoring: str = "entropy",
        dynamic_threshold: bool = True,
    ) -> "CGGRUnslothBridge":
        """
        Patch a trainer to use CGGR selective backpropagation.
        
        Args:
            trainer: HuggingFace/TRL trainer instance
            min_tokens_ratio: Minimum fraction of tokens to keep (0.25 = 25% hardest)
            num_router_layers: Layers for difficulty scoring router
            warmup_steps: Train normally for this many steps first
            scoring: Scoring strategy ('entropy', 'margin', 'loss', 'combined')
            dynamic_threshold: Adjust ratio based on batch confidence
            
        Returns:
            CGGRUnslothBridge instance (for accessing stats)
            
        Example:
            >>> bridge = CGGRUnslothBridge.patch_trainer(trainer)
            >>> trainer.train()
            >>> print(bridge.get_stats())
        """
        # Create bridge instance
        bridge = cls(
            model=trainer.model,
            min_tokens_ratio=min_tokens_ratio,
            num_router_layers=num_router_layers,
            warmup_steps=warmup_steps,
            scoring=scoring,
            dynamic_threshold=dynamic_threshold,
        )
        
        # Store reference on trainer
        trainer._cggr_bridge = bridge
        
        # Patch compute_loss to apply label masking
        original_compute_loss = trainer.compute_loss
        
        @wraps(original_compute_loss)
        def cggr_compute_loss(model, inputs, *args, **kwargs):
            # Apply CGGR label masking
            if "labels" in inputs and inputs["labels"] is not None:
                inputs = dict(inputs)  # Don't modify original
                inputs["labels"] = bridge.mask_easy_tokens(
                    input_ids=inputs.get("input_ids"),
                    labels=inputs["labels"],
                    attention_mask=inputs.get("attention_mask"),
                )
            
            # Call original compute_loss
            outputs = original_compute_loss(model, inputs, *args, **kwargs)
            
            # Update step counter
            bridge.step()
            
            return outputs
        
        trainer.compute_loss = cggr_compute_loss
        
        print(f"🦥 Unsloth + CGGR: Selective backpropagation enabled!")
        print(f"   → Keeping {min_tokens_ratio*100:.0f}% hardest tokens for gradient computation")
        print(f"   → Router uses {num_router_layers} layers, warmup={warmup_steps} steps")
        
        return bridge


def patch_trainer_for_cggr(
    trainer,
    min_tokens_ratio: float = 0.25,
    **kwargs,
) -> CGGRUnslothBridge:
    """
    Convenience function to patch a trainer for CGGR.
    
    Equivalent to CGGRUnslothBridge.patch_trainer().
    
    Args:
        trainer: Trainer instance to patch
        min_tokens_ratio: Fraction of tokens to keep (0.25 = 25% hardest)
        **kwargs: Additional arguments passed to CGGRUnslothBridge.patch_trainer()
        
    Returns:
        CGGRUnslothBridge instance
    """
    return CGGRUnslothBridge.patch_trainer(
        trainer,
        min_tokens_ratio=min_tokens_ratio,
        **kwargs,
    )
