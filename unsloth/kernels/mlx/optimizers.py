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

"""MLX Optimizers for pure MLX training.

This module provides optimizer implementations that work directly with MLX arrays,
enabling training without PyTorch dependencies.
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, Any
import math
import mlx.core as mx
from mlx import optimizers as mx_opt


class Optimizer:
    """Base class for MLX optimizers."""

    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate
        self.state: Dict[str, Dict[str, mx.array]] = {}
        self.step_count = 0

    def __call__(self, gradients: Dict[str, mx.array], parameters: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Apply gradients to parameters.
        
        Args:
            gradients: Dictionary of parameter gradients
            parameters: Dictionary of parameters to update
            
        Returns:
            Dictionary of updated parameters
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def apply_gradients(self, grads_and_vars: list[tuple[mx.array, mx.array]]) -> list[mx.array]:
        """Apply gradients to variables (similar to TensorFlow/Keras API).
        
        Args:
            grads_and_vars: List of (gradient, variable) tuples
            
        Returns:
            List of updated variables
        """
        updated = []
        for grad, var in grads_and_vars:
            if grad is None:
                updated.append(var)
                continue
            updated.append(self.update(var, grad))
        self.step_count += 1
        return updated

    def update(self, parameter: mx.array, gradient: mx.array) -> mx.array:
        """Update a single parameter.
        
        Args:
            parameter: Parameter to update
            gradient: Gradient for the parameter
            
        Returns:
            Updated parameter
        """
        raise NotImplementedError("Subclasses must implement update")

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            "learning_rate": self.learning_rate,
            "step_count": self.step_count,
            "state": self.state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state from checkpoint."""
        self.learning_rate = state_dict.get("learning_rate", self.learning_rate)
        self.step_count = state_dict.get("step_count", 0)
        self.state = state_dict.get("state", {})

    def zero_grad(self) -> None:
        """Zero out gradients (placeholder for compatibility)."""
        pass


class AdamW(Optimizer):
    """AdamW optimizer with weight decay.
    
    Implements the AdamW algorithm from "Decoupled Weight Decay Regularization".
    
    Args:
        learning_rate: Learning rate (default: 1e-3)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._mx_optimizer = None

    def _init_mlx_optimizer(self, params: dict[str, mx.array]) -> mx_opt.Adam:
        """Initialize MLX Adam optimizer on first call."""
        if self._mx_optimizer is None:
            self._mx_optimizer = mx_opt.Adam(
                learning_rate=self.learning_rate,
                betas=(self.beta1, self.beta2),
                eps=self.epsilon,
            )
            # Store parameter keys for state management
            self._param_keys = list(params.keys())
        return self._mx_optimizer

    def __call__(
        self,
        gradients: dict[str, mx.array],
        parameters: dict[str, mx.array],
    ) -> dict[str, mx.array]:
        """Apply AdamW update to parameters."""
        optimizer = self._init_mlx_optimizer(parameters)
        
        updated_params = {}
        
        for key, param in parameters.items():
            grad = gradients.get(key)
            if grad is None:
                updated_params[key] = param
                continue

            # Apply weight decay (decoupled from gradient)
            if self.weight_decay > 0:
                param = param * (1 - self.learning_rate * self.weight_decay)

            # Get or initialize state
            if key not in self.state:
                self.state[key] = {
                    "m": mx.zeros_like(param),
                    "v": mx.zeros_like(param),
                }

            state = self.state[key]
            m = state["m"]
            v = state["v"]

            # Adam update
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)

            # Bias correction
            bias_correction1 = 1 - self.beta1 ** (self.step_count + 1)
            bias_correction2 = 1 - self.beta2 ** (self.step_count + 1)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Update parameter
            param = param - self.learning_rate * m_hat / (mx.sqrt(v_hat) + self.epsilon)

            # Store updated state
            state["m"] = m
            state["v"] = v
            updated_params[key] = param

        self.step_count += 1
        return updated_params

    def update(self, parameter: mx.array, gradient: mx.array) -> mx.array:
        """Update a single parameter with AdamW."""
        # Apply weight decay
        if self.weight_decay > 0:
            parameter = parameter * (1 - self.learning_rate * self.weight_decay)

        # Get or initialize state
        param_id = id(parameter)
        if param_id not in self.state:
            self.state[param_id] = {
                "m": mx.zeros_like(parameter),
                "v": mx.zeros_like(parameter),
            }

        state = self.state[param_id]
        m = state["m"]
        v = state["v"]

        # Adam update
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient * gradient)

        # Bias correction
        bias_correction1 = 1 - self.beta1 ** (self.step_count + 1)
        bias_correction2 = 1 - self.beta2 ** (self.step_count + 1)

        m_hat = m / bias_correction1
        v_hat = v / bias_correction2

        # Update parameter
        updated = parameter - self.learning_rate * m_hat / (mx.sqrt(v_hat) + self.epsilon)

        # Store updated state
        state["m"] = m
        state["v"] = v

        return updated


class SGD(Optimizer):
    """SGD optimizer with momentum.
    
    Args:
        learning_rate: Learning rate (default: 1e-3)
        momentum: Momentum factor (default: 0.0)
        weight_decay: Weight decay (L2 penalty) (default: 0.0)
        nesterov: Use Nesterov momentum (default: False)
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def __call__(
        self,
        gradients: dict[str, mx.array],
        parameters: dict[str, mx.array],
    ) -> dict[str, mx.array]:
        """Apply SGD update to parameters."""
        updated_params = {}

        for key, param in parameters.items():
            grad = gradients.get(key)
            if grad is None:
                updated_params[key] = param
                continue

            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Get or initialize velocity
            if key not in self.state:
                self.state[key] = {"v": mx.zeros_like(param)}

            state = self.state[key]
            v = state["v"]

            # Update velocity
            if self.momentum > 0:
                v = self.momentum * v + grad
                if self.nesterov:
                    grad = grad + self.momentum * v
                else:
                    grad = v
                state["v"] = v

            # Update parameter
            param = param - self.learning_rate * grad
            updated_params[key] = param

        self.step_count += 1
        return updated_params

    def update(self, parameter: mx.array, gradient: mx.array) -> mx.array:
        """Update a single parameter with SGD."""
        # Apply weight decay
        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * parameter

        # Get or initialize velocity
        param_id = id(parameter)
        if param_id not in self.state:
            self.state[param_id] = {"v": mx.zeros_like(parameter)}

        state = self.state[param_id]
        v = state["v"]

        # Update velocity
        if self.momentum > 0:
            v = self.momentum * v + gradient
            if self.nesterov:
                gradient = gradient + self.momentum * v
            else:
                gradient = v
            state["v"] = v

        # Update parameter
        return parameter - self.learning_rate * gradient


class LearningRateScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.learning_rate

    def step(self, step_count: int) -> float:
        """Update learning rate and return new value."""
        lr = self.get_lr(step_count)
        self.optimizer.learning_rate = lr
        return lr

    def get_lr(self, step_count: int) -> float:
        """Get the learning rate for the given step."""
        raise NotImplementedError("Subclasses must implement get_lr")


class LinearWarmupCosineDecay(LearningRateScheduler):
    """Linear warmup followed by cosine decay.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate after decay (default: 0.0)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self, step_count: int) -> float:
        """Calculate learning rate with warmup and cosine decay."""
        if step_count < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (step_count / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay


class CosineDecay(LearningRateScheduler):
    """Cosine learning rate decay.
    
    Args:
        optimizer: Optimizer to schedule
        total_steps: Total number of training steps
        min_lr: Minimum learning rate after decay (default: 0.0)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self, step_count: int) -> float:
        """Calculate learning rate with cosine decay."""
        progress = min(step_count / self.total_steps, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay


class LinearDecay(LearningRateScheduler):
    """Linear learning rate decay.
    
    Args:
        optimizer: Optimizer to schedule
        total_steps: Total number of training steps
        min_lr: Minimum learning rate after decay (default: 0.0)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self, step_count: int) -> float:
        """Calculate learning rate with linear decay."""
        progress = min(step_count / self.total_steps, 1.0)
        return self.initial_lr + (self.min_lr - self.initial_lr) * progress


class StepDecay(LearningRateScheduler):
    """Step learning rate decay (decay by factor at specific intervals).
    
    Args:
        optimizer: Optimizer to schedule
        step_size: Decay LR every step_size steps
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
    ):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step_count: int) -> float:
        """Calculate learning rate with step decay."""
        return self.initial_lr * (self.gamma ** (step_count // self.step_size))


def clip_grad_norm(
    gradients: dict[str, mx.array],
    max_norm: float,
    norm_type: float = 2.0,
) -> mx.array:
    """Clip gradient norm.
    
    Args:
        gradients: Dictionary of gradients
        max_norm: Maximum norm value
        norm_type: Type of norm (default: 2.0 for L2)
        
    Returns:
        Total norm of gradients (before clipping)
    """
    # Flatten all gradients
    all_grads = [g.reshape(-1) for g in gradients.values() if g is not None]
    
    if not all_grads:
        return mx.array(0.0)
    
    # Concatenate all gradients
    flat_grads = mx.concatenate(all_grads)
    
    # Compute norm
    if norm_type == float("inf"):
        total_norm = mx.max(mx.abs(flat_grads))
    else:
        total_norm = mx.power(mx.sum(mx.power(mx.abs(flat_grads), norm_type)), 1.0 / norm_type)
    
    # Clip
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        clipped_grads = {}
        for key, grad in gradients.items():
            if grad is not None:
                clipped_grads[key] = grad * clip_coef
        return total_norm, clipped_grads
    
    return total_norm, gradients


def clip_grad_value_(gradients: dict[str, mx.array], clip_value: float) -> dict[str, mx.array]:
    """Clip gradients to a maximum value.
    
    Args:
        gradients: Dictionary of gradients
        clip_value: Maximum absolute value
        
    Returns:
        Clipped gradients
    """
    clipped_grads = {}
    for key, grad in gradients.items():
        if grad is not None:
            clipped_grads[key] = mx.clip(grad, -clip_value, clip_value)
        else:
            clipped_grads[key] = None
    return clipped_grads


__all__ = [
    "Optimizer",
    "AdamW",
    "SGD",
    "LearningRateScheduler",
    "LinearWarmupCosineDecay",
    "CosineDecay",
    "LinearDecay",
    "StepDecay",
    "clip_grad_norm",
    "clip_grad_value_",
]
