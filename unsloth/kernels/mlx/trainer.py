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

"""MLX Training Loop for pure MLX training.

This module provides a trainer implementation that works directly with MLX arrays,
enabling full training loops without PyTorch dependencies.
"""

from __future__ import annotations

from typing import Optional, Callable, Iterator, Any, Dict
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import mlx.core as mx

from .optimizers import Optimizer, AdamW, LearningRateScheduler
from .losses import cross_entropy_loss, LanguageModelingLoss


@dataclass
class TrainingConfig:
    """Configuration for MLX training."""
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    mixed_precision: bool = False
    dtype: str = "float32"  # float32, float16, bfloat16
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    
    # Checkpointing
    output_dir: str = "./mlx_checkpoints"
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_dataset: Optional[Any] = None
    compute_metrics: Optional[Callable] = None
    
    # Callbacks
    callbacks: list = field(default_factory=list)


class MLXTrainer:
    """MLX trainer for training models with automatic differentiation.
    
    This trainer provides:
    - Forward pass with mx.grad for automatic differentiation
    - Backward pass and optimizer step
    - Gradient accumulation support
    - Checkpointing (save/resume training)
    - Mixed precision training (fp16/bf16)
    - Progress logging
    
    Example:
        >>> from unsloth.kernels.mlx.trainer import MLXTrainer, TrainingConfig
        >>> from unsloth.kernels.mlx.optimizers import AdamW
        >>> 
        >>> model = create_llama_model(...)
        >>> optimizer = AdamW(learning_rate=5e-5)
        >>> config = TrainingConfig(num_epochs=3, batch_size=2)
        >>> 
        >>> trainer = MLXTrainer(model, optimizer, config)
        >>> trainer.train(train_dataset)
    """

    def __init__(
        self,
        model: Any,
        optimizer: Optimizer,
        config: TrainingConfig,
        scheduler: Optional[LearningRateScheduler] = None,
    ):
        """Initialize the MLX trainer.
        
        Args:
            model: MLX model with forward pass
            optimizer: MLX optimizer
            config: Training configuration
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision
        self.dtype = self._get_dtype(config.dtype)
        self.use_amp = config.mixed_precision
        
        # Loss function
        self.loss_fn = LanguageModelingLoss()
        
        # Metrics tracking
        self.training_loss = 0.0
        self.loss_history = []
        
        # Gradient accumulation
        self.accumulated_grads = None
        self.accumulation_count = 0
        
        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)
    
    def _get_dtype(self, dtype_str: str) -> type:
        """Convert dtype string to MLX dtype."""
        dtype_map = {
            "float32": mx.float32,
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
        }
        return dtype_map.get(dtype_str, mx.float32)
    
    def compute_loss(
        self,
        batch: dict[str, mx.array],
        train: bool = True,
    ) -> tuple[mx.array, dict[str, Any]]:
        """Compute loss for a batch.
        
        Args:
            batch: Dictionary with 'input_ids' and optionally 'labels'
            train: Whether in training mode
            
        Returns:
            Tuple of (loss, model_outputs)
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        attention_mask = batch.get("attention_mask")
        
        # Forward pass
        logits, loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        outputs = {"logits": logits}
        
        return loss, outputs
    
    def training_step(
        self,
        batch: dict[str, mx.array],
    ) -> dict[str, Any]:
        """Perform a single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with loss and other metrics
        """
        def loss_fn(params):
            """Loss function for gradient computation."""
            # Set model parameters
            original_params = self._get_model_params()
            self._set_model_params(params)
            
            # Forward pass
            loss, outputs = self.compute_loss(batch, train=True)
            
            # Restore original parameters
            self._set_model_params(original_params)
            
            return loss
        
        # Get current parameters
        params = self._get_model_params()
        
        # Compute gradients
        loss, grads = mx.value_and_grad(loss_fn)(params)
        
        # Gradient accumulation
        if self.config.gradient_accumulation_steps > 1:
            if self.accumulated_grads is None:
                self.accumulated_grads = grads
            else:
                self.accumulated_grads = {
                    k: self.accumulated_grads[k] + grads[k]
                    for k in grads.keys()
                }
            self.accumulation_count += 1
            
            # Only update when accumulation is complete
            if self.accumulation_count >= self.config.gradient_accumulation_steps:
                # Average gradients
                self.accumulated_grads = {
                    k: v / self.config.gradient_accumulation_steps
                    for k, v in self.accumulated_grads.items()
                }
                
                # Clip gradients
                if self.config.max_grad_norm > 0:
                    from .optimizers import clip_grad_norm
                    _, self.accumulated_grads = clip_grad_norm(
                        self.accumulated_grads,
                        self.config.max_grad_norm,
                    )
                
                # Update parameters
                params = self.optimizer(self.accumulated_grads, params)
                self._set_model_params(params)
                
                # Reset accumulation
                self.accumulated_grads = None
                self.accumulation_count = 0
                self.global_step += 1
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step(self.global_step)
        else:
            # Clip gradients
            if self.config.max_grad_norm > 0:
                from .optimizers import clip_grad_norm
                _, grads = clip_grad_norm(grads, self.config.max_grad_norm)
            
            # Update parameters
            params = self.optimizer(grads, params)
            self._set_model_params(params)
            
            self.global_step += 1
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(self.global_step)
        
        # Track loss
        self.training_loss += float(loss)
        self.loss_history.append(float(loss))
        
        return {
            "loss": float(loss),
            "learning_rate": self.optimizer.learning_rate,
        }
    
    def _get_model_params(self) -> dict[str, mx.array]:
        """Get model parameters as a dictionary."""
        # This should be implemented based on model structure
        # For now, assume model has a get_params method
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        else:
            # Try to extract parameters automatically
            params = {}
            for name, layer in self._get_layers(self.model):
                if hasattr(layer, "weight"):
                    params[f"{name}.weight"] = layer.weight
                if hasattr(layer, "bias") and layer.bias is not None:
                    params[f"{name}.bias"] = layer.bias
            return params
    
    def _set_model_params(self, params: dict[str, mx.array]) -> None:
        """Set model parameters from a dictionary."""
        if hasattr(self.model, "set_params"):
            self.model.set_params(params)
        else:
            # Try to set parameters automatically
            for name, layer in self._get_layers(self.model):
                weight_key = f"{name}.weight"
                bias_key = f"{name}.bias"
                if weight_key in params and hasattr(layer, "weight"):
                    layer.weight = params[weight_key]
                if bias_key in params and hasattr(layer, "bias"):
                    layer.bias = params[bias_key]
    
    def _get_layers(self, model: Any, prefix: str = ""):
        """Recursively get all layers in the model."""
        layers = []
        
        if hasattr(model, "__dict__"):
            for name, child in model.__dict__.items():
                if name.startswith("_"):
                    continue
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if it's a layer with parameters
                if hasattr(child, "weight") or hasattr(child, "bias"):
                    layers.append((full_name, child))
                
                # Recursively get child layers
                if hasattr(child, "__dict__"):
                    layers.extend(self._get_layers(child, full_name))
                elif isinstance(child, (list, tuple)):
                    for i, item in enumerate(child):
                        layers.extend(self._get_layers(item, f"{full_name}[{i}]"))
        
        return layers
    
    def train(
        self,
        train_dataset: Iterator[dict[str, mx.array]],
        num_epochs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the model.
        
        Args:
            train_dataset: Training dataset iterator
            num_epochs: Number of epochs (overrides config if provided)
            
        Returns:
            Training results dictionary
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Mixed precision: {self.use_amp}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training loop
            self.training_loss = 0.0
            num_batches = 0
            
            for batch in train_dataset:
                step_result = self.training_step(batch)
                num_batches += 1
                
                # Logging
                if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
                    avg_loss = self.training_loss / num_batches if num_batches > 0 else 0
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                    
                    print(
                        f"Step {self.global_step} | "
                        f"Loss: {step_result['loss']:.4f} | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"LR: {step_result['learning_rate']:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f}"
                    )
                
                # Checkpointing
                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Evaluation
                if (self.config.eval_dataset is not None and 
                    self.global_step > 0 and 
                    self.global_step % self.config.eval_steps == 0):
                    eval_results = self.evaluate()
                    print(f"Evaluation at step {self.global_step}: {eval_results}")
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = self.training_loss / num_batches if num_batches > 0 else 0
            
            print(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s | "
                f"Average loss: {avg_loss:.4f}"
            )
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(suffix=f"epoch-{epoch + 1}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        
        return {
            "train_loss": self.training_loss / num_batches if num_batches > 0 else 0,
            "total_steps": self.global_step,
            "total_time": total_time,
        }
    
    def evaluate(
        self,
        eval_dataset: Optional[Iterator[dict[str, mx.array]]] = None,
    ) -> dict[str, float]:
        """Evaluate the model.
        
        Args:
            eval_dataset: Evaluation dataset (uses config.eval_dataset if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_dataset = eval_dataset or self.config.eval_dataset
        if eval_dataset is None:
            return {}
        
        self.model.eval() if hasattr(self.model, "eval") else None
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in eval_dataset:
            loss, _ = self.compute_loss(batch, train=False)
            total_loss += float(loss)
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Restore training mode
        self.model.train() if hasattr(self.model, "train") else None
        
        return {"eval_loss": avg_loss}
    
    def save_checkpoint(
        self,
        output_dir: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """Save training checkpoint.
        
        Args:
            output_dir: Directory to save checkpoint
            suffix: Optional suffix for checkpoint name
        """
        output_dir = output_dir or self.config.output_dir
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = f"checkpoint-{self.global_step}"
        if suffix:
            checkpoint_name += f"-{suffix}"
        checkpoint_dir = path / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "model.safetensors"
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(checkpoint_dir)
        elif hasattr(self.model, "save"):
            self.model.save(model_path)
        else:
            # Save parameters manually
            params = self._get_model_params()
            mx.savez(str(model_path), **params)
        
        # Save optimizer state
        optimizer_path = checkpoint_dir / "optimizer.npz"
        optimizer_state = self.optimizer.state_dict()
        mx.savez(str(optimizer_path), **{
            k: mx.array(v) if isinstance(v, (int, float)) else v
            for k, v in optimizer_state.items()
        })
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "config": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            },
        }
        
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the most recent ones."""
        if self.config.save_total_limit <= 0:
            return
        
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        
        for checkpoint in checkpoints[self.config.save_total_limit:]:
            if checkpoint.is_dir():
                import shutil
                shutil.rmtree(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load model
        model_path = checkpoint_dir / "model.safetensors"
        if model_path.exists() and hasattr(self.model, "load"):
            self.model.load(model_path)
        elif model_path.exists():
            # Load parameters manually
            params = mx.load(str(model_path))
            self._set_model_params(params)
        
        # Load optimizer state
        optimizer_path = checkpoint_dir / "optimizer.npz"
        if optimizer_path.exists():
            optimizer_state = mx.load(str(optimizer_path))
            self.optimizer.load_state_dict(optimizer_state)
        
        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                training_state = json.load(f)
            
            self.global_step = training_state.get("global_step", 0)
            self.epoch = training_state.get("epoch", 0)
            self.best_metric = training_state.get("best_metric", float("inf"))
        
        print(f"Checkpoint loaded. Resuming from step {self.global_step}")
    
    def save_model(self, output_dir: str) -> None:
        """Save the final model.
        
        Args:
            output_dir: Directory to save the model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_path)
        elif hasattr(self.model, "save"):
            self.model.save(output_path / "model.safetensors")
        else:
            params = self._get_model_params()
            mx.savez(str(output_path / "model.safetensors"), **params)
        
        print(f"Model saved to {output_path}")


__all__ = [
    "MLXTrainer",
    "TrainingConfig",
]
