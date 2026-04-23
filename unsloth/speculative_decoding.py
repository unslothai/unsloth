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

"""Speculative decoding training module for Unsloth.

This module provides knowledge distillation training where a smaller draft model
learns to match the outputs of a larger teacher model. Useful for pre-training
efficient draft models for speculative decoding during inference.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from unsloth.trainer import UnslothTrainer

logger = logging.getLogger(__name__)

__all__ = [
    "SpeculativeDecodingConfig",
    "SpeculativeDecodingTrainer",
]


@dataclass
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding training.

    This config enables knowledge distillation training where a smaller draft model
    learns to match the outputs of a larger teacher model. Useful for pre-training
    efficient draft models for speculative decoding during inference.

    Attributes:
        distillation_temperature: Temperature for knowledge distillation. Higher values
            make the probability distributions softer (default: 4.0)
        distillation_loss_weight: Weight for the distillation loss component (default: 1.0)
        draft_model_loss_weight: Weight for the draft model's base loss (default: 0.1)
        train_draft_only: If True, only train the draft model parameters (default: True)
        teacher_model_dtype: Data type to use for teacher model (default: "float32")
    """

    distillation_temperature: float = 4.0
    distillation_loss_weight: float = 1.0
    draft_model_loss_weight: float = 0.1
    train_draft_only: bool = True
    teacher_model_dtype: str = "float32"


class SpeculativeDecodingTrainer(UnslothTrainer):
    """Trainer for speculative decoding with knowledge distillation.

    This trainer manages both a draft (smaller) and teacher (larger) model.
    The draft model is trained to match the teacher model's logits using
    knowledge distillation, enabling efficient speculative decoding at inference.

    The teacher model is frozen (no gradient updates) and used only for generating
    target logits during training.

    Args:
        draft_model: The smaller model to be trained as the draft model
        teacher_model: The larger teacher model (will be frozen during training)
        spec_config: SpeculativeDecodingConfig instance for distillation settings
        *args: Arguments passed to UnslothTrainer
        **kwargs: Keyword arguments passed to UnslothTrainer

    Example:
        >>> from unsloth import fast_mistral_loader
        >>> from unsloth.speculative_decoding import (
        ...     SpeculativeDecodingTrainer,
        ...     SpeculativeDecodingConfig,
        ... )
        >>>
        >>> # Load models
        >>> draft_model, tokenizer = fast_mistral_loader("mistral-7b", load_in_4bit=True)
        >>> teacher_model, _ = fast_mistral_loader("mistral-large", load_in_4bit=True)
        >>>
        >>> # Configure speculative decoding
        >>> spec_config = SpeculativeDecodingConfig(
        ...     distillation_temperature=4.0,
        ...     distillation_loss_weight=1.0,
        ...     draft_model_loss_weight=0.1,
        ... )
        >>>
        >>> # Create trainer
        >>> trainer = SpeculativeDecodingTrainer(
        ...     model=draft_model,
        ...     draft_model=draft_model,
        ...     teacher_model=teacher_model,
        ...     spec_config=spec_config,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ... )
        >>>
        >>> # Train
        >>> trainer.train()
        >>>
        >>> # Save models
        >>> trainer.save_speculative_models("./speculative_models")
    """

    def __init__(
        self,
        draft_model=None,
        teacher_model=None,
        spec_config: Optional[SpeculativeDecodingConfig] = None,
        *args,
        **kwargs,
    ):
        """Initialize the SpeculativeDecodingTrainer.

        Args:
            draft_model: The smaller model to train. If None, uses the model from args
            teacher_model: The larger teacher model. If None, no distillation is used
            spec_config: Configuration for speculative decoding training
            *args: Arguments passed to UnslothTrainer
            **kwargs: Keyword arguments passed to UnslothTrainer
        """
        self.draft_model = draft_model
        self.teacher_model = teacher_model
        self.spec_config = spec_config or SpeculativeDecodingConfig()

        # Freeze teacher model to prevent training
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            logger.info("Teacher model frozen - no gradients will be computed")

        super().__init__(*args, **kwargs)

    def _compute_distillation_loss(
        self, draft_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute knowledge distillation loss between draft and teacher models.

        Uses KL divergence between teacher probabilities and draft log probabilities,
        scaled by the temperature squared as per distillation literature.

        Args:
            draft_logits: Logits from the draft model, shape (batch_size, seq_len, vocab_size)
            teacher_logits: Logits from the teacher model, shape (batch_size, seq_len, vocab_size)

        Returns:
            Scalar loss tensor representing the KL divergence
        """
        temperature = self.spec_config.distillation_temperature

        # Compute softmax with temperature scaling
        draft_log_probs = F.log_softmax(draft_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # KL divergence: E[log(p_draft) - log(p_teacher)]
        kl_loss = F.kl_div(
            draft_log_probs, teacher_probs, reduction="batchmean", log_target=False
        )

        # Scale by temperature squared as per distillation literature
        # This accounts for the temperature scaling in the softmax
        kl_loss = kl_loss * (temperature ** 2)

        return kl_loss

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """Override training step to include distillation loss.

        Combines the draft model's base loss with the distillation loss from the teacher
        model's logits. The final loss is a weighted combination of both losses.

        Args:
            model: The draft model being trained
            inputs: Dictionary containing input_ids, attention_mask, labels, etc.

        Returns:
            Scalar loss tensor for backpropagation
        """
        model.train()

        # Forward pass through draft model
        draft_outputs = model(**inputs)
        draft_logits = draft_outputs.logits

        # Compute standard training loss from draft model
        if hasattr(draft_outputs, "loss") and draft_outputs.loss is not None:
            base_loss = draft_outputs.loss
        else:
            labels = inputs.get("labels")
            if labels is not None:
                base_loss = F.cross_entropy(
                    draft_logits.view(-1, draft_logits.size(-1)), labels.view(-1)
                )
            else:
                base_loss = draft_outputs.loss or torch.tensor(
                    0.0, device=model.device, dtype=model.dtype
                )

        total_loss = base_loss * self.spec_config.draft_model_loss_weight

        # Add distillation loss if teacher model is provided
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            distill_loss = self._compute_distillation_loss(
                draft_logits, teacher_logits
            )
            
            # Combine losses with their weights
            total_loss = (
                base_loss * self.spec_config.draft_model_loss_weight
                + distill_loss * self.spec_config.distillation_loss_weight
            )

            if self.state.global_step % max(1, self.args.logging_steps) == 0:
                logger.info(
                    f"Step {self.state.global_step}: "
                    f"base_loss={base_loss.item():.4f}, "
                    f"distill_loss={distill_loss.item():.4f}, "
                    f"total_loss={total_loss.item():.4f}"
                )

        return total_loss

    def get_models(self) -> Dict[str, Any]:
        """Return both draft and teacher models.

        Returns:
            Dictionary with keys 'draft_model' and 'teacher_model' containing
            the respective model instances
        """
        return {
            "draft_model": self.draft_model or self.model,
            "teacher_model": self.teacher_model,
        }

    def save_speculative_models(self, output_dir: str) -> None:
        """Save both draft and teacher models.

        Saves the draft model with all training artifacts and preserves teacher
        model metadata. The teacher model is not saved as it was frozen during training.

        Args:
            output_dir: Directory path where models will be saved
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save draft model with trainer state
        draft_dir = os.path.join(output_dir, "draft_model")
        if self.draft_model is not None:
            self.draft_model.save_pretrained(draft_dir)
        else:
            self.model.save_pretrained(draft_dir)
        logger.info(f"Draft model saved to {draft_dir}")

        # Save teacher model metadata
        if self.teacher_model is not None:
            teacher_info_file = os.path.join(output_dir, "teacher_model_info.txt")
            with open(teacher_info_file, "w") as f:
                f.write("Teacher model (frozen during training)\n")
                f.write("=" * 50 + "\n\n")
                f.write("The teacher model was used for knowledge distillation\n")
                f.write("and was not modified during training.\n\n")
                
                if hasattr(self.teacher_model, "config"):
                    config = self.teacher_model.config
                    f.write(f"Model Type: {config.model_type}\n")
                    if hasattr(config, "hidden_size"):
                        f.write(f"Hidden Size: {config.hidden_size}\n")
                    if hasattr(config, "num_hidden_layers"):
                        f.write(f"Number of Layers: {config.num_hidden_layers}\n")
                    if hasattr(config, "vocab_size"):
                        f.write(f"Vocabulary Size: {config.vocab_size}\n")
            
            logger.info(f"Teacher model info saved to {teacher_info_file}")

        # Save speculative decoding config
        config_file = os.path.join(output_dir, "speculative_config.txt")
        with open(config_file, "w") as f:
            f.write("Speculative Decoding Configuration\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Distillation Temperature: {self.spec_config.distillation_temperature}\n")
            f.write(f"Distillation Loss Weight: {self.spec_config.distillation_loss_weight}\n")
            f.write(f"Draft Model Loss Weight: {self.spec_config.draft_model_loss_weight}\n")
            f.write(f"Train Draft Only: {self.spec_config.train_draft_only}\n")
            f.write(f"Teacher Model Dtype: {self.spec_config.teacher_model_dtype}\n")
        
        logger.info(f"Speculative config saved to {config_file}")
        print(f"🦥 Unsloth: Speculative decoding models saved to {output_dir}")
