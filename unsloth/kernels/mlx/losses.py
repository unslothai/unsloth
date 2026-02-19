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

"""MLX Loss Functions for pure MLX training.

This module provides loss function implementations that work directly with MLX arrays,
enabling training without PyTorch dependencies.
"""

from __future__ import annotations

from typing import Optional
import mlx.core as mx


def cross_entropy_loss(
    logits: mx.array,
    labels: mx.array,
    reduction: str = "mean",
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> mx.array:
    """Cross-entropy loss for language modeling.
    
    Args:
        logits: Logits tensor of shape (N, C) or (batch, seq_len, vocab_size)
        labels: Target labels of shape (N,) or (batch, seq_len)
        reduction: Reduction method ('mean', 'sum', or 'none')
        ignore_index: Label value to ignore (default: -100)
        label_smoothing: Label smoothing factor (default: 0.0)
        
    Returns:
        Cross-entropy loss
    """
    # Flatten inputs for processing
    original_shape = logits.shape
    
    if logits.ndim == 3:
        # (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
    elif logits.ndim == 2:
        vocab_size = logits.shape[-1]
    else:
        raise ValueError(f"Expected logits with 2 or 3 dimensions, got {logits.ndim}")
    
    # Create mask for valid labels (not ignore_index)
    valid_mask = labels != ignore_index
    
    # Clamp labels to valid range for indexing (will be masked out anyway)
    clamped_labels = mx.where(valid_mask, labels, 0)
    
    # Compute log softmax
    log_probs = mx.log_softmax(logits, axis=-1)
    
    if label_smoothing > 0.0:
        # Label smoothing: soft targets
        # loss = (1 - smoothing) * nll_loss + smoothing * uniform_loss
        n_classes = logits.shape[-1]
        
        # Negative log likelihood for true classes
        nll_loss = -log_probs[mx.arange(logits.shape[0]), clamped_labels]
        
        # Uniform loss (entropy term)
        uniform_loss = -log_probs.mean(axis=-1)
        
        # Combine with smoothing
        loss = (1 - label_smoothing) * nll_loss + label_smoothing * uniform_loss
    else:
        # Standard cross-entropy
        loss = -log_probs[mx.arange(logits.shape[0]), clamped_labels]
    
    # Apply mask
    loss = mx.where(valid_mask, loss, 0.0)
    
    # Apply reduction
    if reduction == "mean":
        # Mean over valid tokens only
        num_valid = valid_mask.sum()
        return loss.sum() / mx.maximum(num_valid, 1)
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        # Reshape back to original labels shape
        if len(original_shape) == 3:
            return loss.reshape(original_shape[0], original_shape[1])
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def cross_entropy_loss_with_z_loss(
    logits: mx.array,
    labels: mx.array,
    z_loss_weight: float = 1e-4,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> tuple[mx.array, mx.array]:
    """Cross-entropy loss with auxiliary z-loss for training stability.
    
    The z-loss penalizes large logit values, improving training stability.
    From "ST-MoE: Designing Stable and Transferable Sparse Expert Models".
    
    Args:
        logits: Logits tensor
        labels: Target labels
        z_loss_weight: Weight for z-loss component
        reduction: Reduction method
        ignore_index: Label value to ignore
        
    Returns:
        Tuple of (total_loss, cross_entropy_loss)
    """
    # Standard cross-entropy
    ce_loss = cross_entropy_loss(
        logits, labels,
        reduction=reduction,
        ignore_index=ignore_index,
    )
    
    # Z-loss: penalize large logit magnitudes
    # z_loss = (log(sum(exp(logits))) ^ 2
    log_sum_exp = mx.log(mx.sum(mx.exp(logits), axis=-1))
    z_loss = log_sum_exp ** 2
    
    # Apply reduction
    if reduction == "mean":
        z_loss = z_loss.mean()
    elif reduction == "sum":
        z_loss = z_loss.sum()
    elif reduction == "none":
        pass
    
    total_loss = ce_loss + z_loss_weight * z_loss
    return total_loss, ce_loss


def fused_cross_entropy_loss(
    logits: mx.array,
    labels: mx.array,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> mx.array:
    """Fused cross-entropy loss with gradient computation.
    
    This is a memory-efficient version that computes the loss and its gradient
    in a single fused kernel operation.
    
    Args:
        logits: Logits tensor
        labels: Target labels
        reduction: Reduction method
        ignore_index: Label value to ignore
        
    Returns:
        Cross-entropy loss
    """
    # For MLX, we use the standard implementation
    # MLX's automatic differentiation handles the backward pass efficiently
    return cross_entropy_loss(
        logits, labels,
        reduction=reduction,
        ignore_index=ignore_index,
    )


def mse_loss(
    predictions: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """Mean squared error loss.
    
    Args:
        predictions: Predicted values
        targets: Target values
        reduction: Reduction method ('mean', 'sum', or 'none')
        
    Returns:
        MSE loss
    """
    loss = (predictions - targets) ** 2
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def kl_div_loss(
    log_probs: mx.array,
    target_probs: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """Kullback-Leibler divergence loss.
    
    Args:
        log_probs: Log probabilities from model
        target_probs: Target probabilities
        reduction: Reduction method
        
    Returns:
        KL divergence loss
    """
    # KL(P || Q) = sum(P * (log P - log Q))
    # Here log_probs is log Q, target_probs is P
    loss = target_probs * (mx.log(mx.maximum(target_probs, 1e-10)) - log_probs)
    
    if reduction == "mean":
        return loss.sum(axis=-1).mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss.sum(axis=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def binary_cross_entropy(
    predictions: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """Binary cross-entropy loss.
    
    Args:
        predictions: Predicted probabilities (after sigmoid)
        targets: Binary targets (0 or 1)
        reduction: Reduction method
        
    Returns:
        Binary cross-entropy loss
    """
    # Clamp predictions for numerical stability
    epsilon = 1e-7
    predictions = mx.clip(predictions, epsilon, 1 - epsilon)
    
    loss = -(targets * mx.log(predictions) + (1 - targets) * mx.log(1 - predictions))
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def contrastive_loss(
    embeddings1: mx.array,
    embeddings2: mx.array,
    labels: mx.array,
    temperature: float = 0.07,
    reduction: str = "mean",
) -> mx.array:
    """Contrastive loss for representation learning.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        labels: Binary labels (1 for positive pairs, 0 for negative)
        temperature: Temperature scaling factor
        reduction: Reduction method
        
    Returns:
        Contrastive loss
    """
    # Normalize embeddings
    embeddings1 = embeddings1 / mx.linalg.norm(embeddings1, axis=-1, keepdims=True)
    embeddings2 = embeddings2 / mx.linalg.norm(embeddings2, axis=-1, keepdims=True)
    
    # Compute similarity matrix
    similarity = mx.matmul(embeddings1, embeddings2.T) / temperature
    
    # Contrastive loss: for positive pairs, maximize similarity
    # For negative pairs, minimize similarity
    loss = labels * (1 - similarity) + (1 - labels) * mx.maximum(0, similarity)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class LossFunction:
    """Base class for loss functions with additional options."""

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        logits: mx.array,
        labels: mx.array,
    ) -> mx.array:
        return cross_entropy_loss(
            logits,
            labels,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class LanguageModelingLoss(LossFunction):
    """Loss function for language modeling tasks.
    
    Handles causal language modeling by shifting logits and labels.
    """

    def __call__(
        self,
        logits: mx.array,
        labels: mx.array,
    ) -> mx.array:
        """Compute loss for language modeling.
        
        Shifts logits and labels for next-token prediction:
        - logits[..., :-1, :] predicts labels[..., 1:]
        
        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            labels: Target token IDs (batch, seq_len)
            
        Returns:
            Cross-entropy loss
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        
        return cross_entropy_loss(
            shift_logits,
            shift_labels,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


__all__ = [
    "cross_entropy_loss",
    "cross_entropy_loss_with_z_loss",
    "fused_cross_entropy_loss",
    "mse_loss",
    "kl_div_loss",
    "binary_cross_entropy",
    "contrastive_loss",
    "LossFunction",
    "LanguageModelingLoss",
]
