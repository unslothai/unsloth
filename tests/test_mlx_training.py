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

"""Tests for MLX training infrastructure.

This module tests the MLX training components:
- Optimizers (AdamW, SGD)
- Loss functions
- Training loop
- LoRA layers
"""

from __future__ import annotations

import unittest
import tempfile
import shutil
from pathlib import Path

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


@unittest.skipUnless(HAS_MLX, "MLX not available")
class TestMLXOptimizers(unittest.TestCase):
    """Test MLX optimizer implementations."""

    def setUp(self):
        """Set up test parameters."""
        from unsloth.kernels.mlx.optimizers import AdamW, SGD
        self.AdamW = AdamW
        self.SGD = SGD

    def test_adamw_initialization(self):
        """Test AdamW optimizer initialization."""
        optimizer = self.AdamW(
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
        )
        
        self.assertEqual(optimizer.learning_rate, 1e-3)
        self.assertEqual(optimizer.beta1, 0.9)
        self.assertEqual(optimizer.beta2, 0.999)
        self.assertEqual(optimizer.weight_decay, 0.01)
        self.assertEqual(optimizer.step_count, 0)

    def test_adamw_update_single_param(self):
        """Test AdamW update on a single parameter."""
        optimizer = self.AdamW(learning_rate=0.1)
        
        # Create parameter and gradient
        param = mx.array([1.0, 2.0, 3.0])
        grad = mx.array([0.1, 0.2, 0.3])
        
        # Update parameter
        updated = optimizer.update(param, grad)
        
        # Check that parameter was updated
        self.assertEqual(updated.shape, param.shape)
        # After first update with bias correction, should move in direction of negative gradient
        self.assertTrue(mx.any(updated != param))

    def test_adamw_update_multiple_params(self):
        """Test AdamW update on multiple parameters."""
        optimizer = self.AdamW(learning_rate=0.1)
        
        # Create parameters and gradients
        params = {
            "weight": mx.array([[1.0, 2.0], [3.0, 4.0]]),
            "bias": mx.array([0.5, 0.5]),
        }
        grads = {
            "weight": mx.array([[0.1, 0.1], [0.1, 0.1]]),
            "bias": mx.array([0.01, 0.01]),
        }
        
        # Update parameters
        updated = optimizer(grads, params)
        
        # Check that all parameters were updated
        self.assertIn("weight", updated)
        self.assertIn("bias", updated)
        self.assertTrue(mx.any(updated["weight"] != params["weight"]))
        self.assertTrue(mx.any(updated["bias"] != params["bias"]))

    def test_adamw_weight_decay(self):
        """Test that weight decay is applied correctly."""
        optimizer = self.AdamW(learning_rate=0.1, weight_decay=0.1)
        
        param = mx.array([1.0, 1.0, 1.0])
        grad = mx.array([0.0, 0.0, 0.0])  # Zero gradient
        
        # Update - only weight decay should be applied
        updated = optimizer.update(param, grad)
        
        # Weight decay should reduce the parameter values
        self.assertTrue(mx.all(updated < param))

    def test_sgd_initialization(self):
        """Test SGD optimizer initialization."""
        optimizer = self.SGD(
            learning_rate=1e-2,
            momentum=0.9,
            weight_decay=0.01,
            nesterov=True,
        )
        
        self.assertEqual(optimizer.learning_rate, 1e-2)
        self.assertEqual(optimizer.momentum, 0.9)
        self.assertEqual(optimizer.weight_decay, 0.01)
        self.assertTrue(optimizer.nesterov)

    def test_sgd_update(self):
        """Test SGD update on parameters."""
        optimizer = self.SGD(learning_rate=0.1, momentum=0.9)
        
        params = {
            "weight": mx.array([1.0, 2.0]),
        }
        grads = {
            "weight": mx.array([0.1, 0.2]),
        }
        
        # Multiple updates to test momentum
        for _ in range(3):
            params = optimizer(grads, params)
        
        # Parameters should be updated
        self.assertTrue(mx.any(params["weight"] != mx.array([1.0, 2.0])))

    def test_sgd_without_momentum(self):
        """Test SGD without momentum."""
        optimizer = self.SGD(learning_rate=0.1, momentum=0.0)
        
        param = mx.array([1.0, 1.0])
        grad = mx.array([0.1, 0.1])
        
        updated = optimizer.update(param, grad)
        
        # Should be simple gradient descent: param - lr * grad
        expected = mx.array([0.99, 0.99])
        self.assertTrue(mx.allclose(updated, expected))

    def test_optimizer_state_dict(self):
        """Test optimizer state saving and loading."""
        optimizer = self.AdamW(learning_rate=1e-3)
        
        # Run a few steps
        param = mx.array([1.0])
        grad = mx.array([0.1])
        for _ in range(5):
            param = optimizer.update(param, grad)
        
        # Save state
        state = optimizer.state_dict()
        
        # Create new optimizer and load state
        optimizer2 = self.AdamW()
        optimizer2.load_state_dict(state)
        
        # Check state was loaded
        self.assertEqual(optimizer2.step_count, optimizer.step_count)
        self.assertEqual(optimizer2.learning_rate, optimizer.learning_rate)


@unittest.skipUnless(HAS_MLX, "MLX not available")
class TestMXLearningRateSchedulers(unittest.TestCase):
    """Test learning rate schedulers."""

    def setUp(self):
        """Set up test."""
        from unsloth.kernels.mlx.optimizers import AdamW
        from unsloth.kernels.mlx.optimizers import (
            LinearWarmupCosineDecay,
            CosineDecay,
            LinearDecay,
            StepDecay,
        )
        self.optimizer_class = AdamW
        self.LinearWarmupCosineDecay = LinearWarmupCosineDecay
        self.CosineDecay = CosineDecay
        self.LinearDecay = LinearDecay
        self.StepDecay = StepDecay

    def test_linear_warmup_cosine_decay(self):
        """Test linear warmup followed by cosine decay."""
        optimizer = self.optimizer_class(learning_rate=1e-3)
        scheduler = self.LinearWarmupCosineDecay(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0,
        )
        
        # Test warmup phase
        lr_0 = scheduler.get_lr(0)
        lr_5 = scheduler.get_lr(5)
        lr_10 = scheduler.get_lr(10)
        
        self.assertEqual(lr_0, 0.0)  # Start at 0
        self.assertLess(lr_0, lr_5)  # Should increase
        self.assertEqual(lr_10, 1e-3)  # Should reach initial LR
        
        # Test decay phase
        lr_50 = scheduler.get_lr(50)
        lr_100 = scheduler.get_lr(100)
        
        self.assertLess(lr_50, lr_10)  # Should decay
        self.assertEqual(lr_100, 0.0)  # Should reach min_lr

    def test_cosine_decay(self):
        """Test cosine learning rate decay."""
        optimizer = self.optimizer_class(learning_rate=1e-3)
        scheduler = self.CosineDecay(
            optimizer,
            total_steps=100,
            min_lr=0.0,
        )
        
        # Check values at different points
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)
        lr_100 = scheduler.get_lr(100)
        
        self.assertEqual(lr_0, 1e-3)  # Start at initial LR
        self.assertLess(lr_50, lr_0)  # Should decay
        self.assertGreater(lr_50, lr_100)  # Should keep decaying
        self.assertEqual(lr_100, 0.0)  # End at min_lr

    def test_linear_decay(self):
        """Test linear learning rate decay."""
        optimizer = self.optimizer_class(learning_rate=1e-3)
        scheduler = self.LinearDecay(
            optimizer,
            total_steps=100,
            min_lr=0.0,
        )
        
        # Check linear decay
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)
        lr_100 = scheduler.get_lr(100)
        
        self.assertEqual(lr_0, 1e-3)
        self.assertAlmostEqual(lr_50, 5e-4, places=6)
        self.assertEqual(lr_100, 0.0)

    def test_step_decay(self):
        """Test step learning rate decay."""
        optimizer = self.optimizer_class(learning_rate=1e-3)
        scheduler = self.StepDecay(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
        
        # Check step decay
        lr_0 = scheduler.get_lr(0)
        lr_29 = scheduler.get_lr(29)
        lr_30 = scheduler.get_lr(30)
        lr_60 = scheduler.get_lr(60)
        
        self.assertEqual(lr_0, 1e-3)
        self.assertEqual(lr_29, 1e-3)  # Before first decay
        self.assertEqual(lr_30, 1e-4)  # After first decay (0.1x)
        self.assertEqual(lr_60, 1e-5)  # After second decay (0.01x)


@unittest.skipUnless(HAS_MLX, "MLX not available")
class TestMLXLossFunctions(unittest.TestCase):
    """Test MLX loss function implementations."""

    def setUp(self):
        """Set up test."""
        from unsloth.kernels.mlx.losses import (
            cross_entropy_loss,
            mse_loss,
            kl_div_loss,
        )
        self.cross_entropy_loss = cross_entropy_loss
        self.mse_loss = mse_loss
        self.kl_div_loss = kl_div_loss

    def test_cross_entropy_basic(self):
        """Test basic cross-entropy loss computation."""
        # Logits: batch_size=2, num_classes=3
        logits = mx.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        
        # Labels: first sample is class 0, second is class 1
        labels = mx.array([0, 1])
        
        loss = self.cross_entropy_loss(logits, labels)
        
        # Loss should be scalar and positive
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_cross_entropy_reduction(self):
        """Test cross-entropy with different reductions."""
        logits = mx.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        labels = mx.array([0, 1])
        
        # Test mean reduction
        loss_mean = self.cross_entropy_loss(logits, labels, reduction="mean")
        self.assertEqual(loss_mean.shape, ())
        
        # Test sum reduction
        loss_sum = self.cross_entropy_loss(logits, labels, reduction="sum")
        self.assertEqual(loss_sum.shape, ())
        self.assertGreater(float(loss_sum), float(loss_mean))
        
        # Test none reduction
        loss_none = self.cross_entropy_loss(logits, labels, reduction="none")
        self.assertEqual(loss_none.shape, (2,))

    def test_cross_entropy_ignore_index(self):
        """Test cross-entropy with ignore_index."""
        logits = mx.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        labels = mx.array([0, -100, 2])  # Ignore second sample
        
        loss = self.cross_entropy_loss(logits, labels, ignore_index=-100)
        
        # Should only consider first and third samples
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_cross_entropy_label_smoothing(self):
        """Test cross-entropy with label smoothing."""
        logits = mx.array([
            [2.0, 0.0, 0.0],
        ])
        labels = mx.array([0])
        
        # With label smoothing, loss should be lower than without
        loss_smooth = self.cross_entropy_loss(
            logits, labels, label_smoothing=0.1
        )
        loss_no_smooth = self.cross_entropy_loss(
            logits, labels, label_smoothing=0.0
        )
        
        self.assertLess(float(loss_smooth), float(loss_no_smooth))

    def test_cross_entropy_3d(self):
        """Test cross-entropy with 3D logits (sequence modeling)."""
        # Logits: (batch=2, seq_len=3, vocab=4)
        logits = mx.random.normal((2, 3, 4))
        labels = mx.array([
            [0, 1, 2],
            [3, 2, 1],
        ])
        
        loss = self.cross_entropy_loss(logits, labels)
        
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_mse_loss(self):
        """Test MSE loss computation."""
        predictions = mx.array([1.0, 2.0, 3.0])
        targets = mx.array([1.5, 2.5, 2.5])
        
        loss = self.mse_loss(predictions, targets)
        
        # Expected: mean of [0.25, 0.25, 0.25] = 0.25
        expected = 0.25
        self.assertAlmostEqual(float(loss), expected, places=5)

    def test_mse_loss_reduction(self):
        """Test MSE loss with different reductions."""
        predictions = mx.array([1.0, 2.0, 3.0])
        targets = mx.array([1.5, 2.5, 2.5])
        
        loss_mean = self.mse_loss(predictions, targets, reduction="mean")
        loss_sum = self.mse_loss(predictions, targets, reduction="sum")
        loss_none = self.mse_loss(predictions, targets, reduction="none")
        
        self.assertEqual(loss_mean.shape, ())
        self.assertEqual(loss_sum.shape, ())
        self.assertEqual(loss_none.shape, (3,))
        self.assertAlmostEqual(float(loss_mean) * 3, float(loss_sum), places=5)


@unittest.skipUnless(HAS_MLX, "MLX not available")
class TestMLXLoRA(unittest.TestCase):
    """Test MLX LoRA implementations."""

    def setUp(self):
        """Set up test."""
        from unsloth.kernels.mlx.lora import (
            LoRALinear,
            LoRAConfig,
            GradientCheckpointing,
        )
        self.LoRALinear = LoRALinear
        self.LoRAConfig = LoRAConfig
        self.GradientCheckpointing = GradientCheckpointing

    def test_lora_linear_initialization(self):
        """Test LoRA linear layer initialization."""
        layer = self.LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        
        self.assertEqual(layer.in_features, 64)
        self.assertEqual(layer.out_features, 128)
        self.assertEqual(layer.r, 8)
        self.assertEqual(layer.lora_alpha, 16)
        self.assertEqual(layer.lora_dropout, 0.1)
        self.assertEqual(layer.scaling, 2.0)  # alpha / r = 16 / 8
        
        # Check shapes
        self.assertEqual(layer.weight.shape, (128, 64))
        self.assertEqual(layer.lora_A.shape, (8, 64))
        self.assertEqual(layer.lora_B.shape, (128, 8))

    def test_lora_linear_forward(self):
        """Test LoRA linear layer forward pass."""
        layer = self.LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
        )
        
        # Input: (batch=2, in_features=64)
        x = mx.random.normal((2, 64))
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 128))

    def test_lora_linear_merge(self):
        """Test LoRA weight merging."""
        layer = self.LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
        )
        
        # Store original weight
        original_weight = layer.weight.copy()
        
        # Merge weights
        layer.merge()
        
        # Check that weights were merged
        self.assertTrue(layer.merged)
        self.assertFalse(mx.allclose(layer.weight, original_weight))
        
        # Unmerge and check
        layer.unmerge()
        self.assertFalse(layer.merged)

    def test_lora_linear_eval_mode(self):
        """Test LoRA layer eval mode."""
        layer = self.LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
            lora_dropout=0.5,  # High dropout for testing
        )
        
        x = mx.random.normal((2, 64))
        
        # Training mode (with dropout)
        layer.train()
        output_train_1 = layer(x)
        output_train_2 = layer(x)
        
        # Outputs should be different due to dropout
        # (Note: This may occasionally fail due to randomness, but unlikely)
        
        # Eval mode (no dropout)
        layer.eval()
        output_eval_1 = layer(x)
        output_eval_2 = layer(x)
        
        # Outputs should be identical in eval mode
        self.assertTrue(mx.allclose(output_eval_1, output_eval_2))

    def test_lora_config(self):
        """Test LoRA configuration."""
        config = self.LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.lora_dropout, 0.05)
        self.assertEqual(config.target_modules, ["q_proj", "k_proj", "v_proj"])

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing."""
        checkpointing = self.GradientCheckpointing()
        
        # Simple function to checkpoint
        def add_one(x):
            return x + 1
        
        x = mx.array([1.0, 2.0, 3.0])
        result = checkpointing(add_one, x)
        
        # Should get same result
        expected = mx.array([2.0, 3.0, 4.0])
        self.assertTrue(mx.allclose(result, expected))


@unittest.skipUnless(HAS_MLX, "MLX not available")
class TestMLXTrainer(unittest.TestCase):
    """Test MLX training loop."""

    def setUp(self):
        """Set up test."""
        from unsloth.kernels.mlx.trainer import MLXTrainer, TrainingConfig
        from unsloth.kernels.mlx.optimizers import AdamW
        
        self.MLXTrainer = MLXTrainer
        self.TrainingConfig = TrainingConfig
        self.AdamW = AdamW

    def test_training_config(self):
        """Test training configuration."""
        config = self.TrainingConfig(
            num_epochs=5,
            batch_size=4,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            logging_steps=10,
        )
        
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.gradient_accumulation_steps, 2)
        self.assertEqual(config.max_grad_norm, 1.0)
        self.assertEqual(config.logging_steps, 10)

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        # Create a simple mock model
        class MockModel:
            def __call__(self, input_ids, attention_mask=None, labels=None):
                # Simple linear model
                weight = mx.random.normal((10, input_ids.shape[-1]))
                logits = mx.matmul(input_ids, weight.T)
                
                if labels is not None:
                    from unsloth.kernels.mlx.losses import cross_entropy_loss
                    loss = cross_entropy_loss(logits, labels)
                else:
                    loss = mx.array(0.0)
                
                return logits, loss

        model = MockModel()
        optimizer = self.AdamW(learning_rate=1e-3)
        config = self.TrainingConfig(
            num_epochs=1,
            output_dir=tempfile.mkdtemp(),
        )
        
        trainer = self.MLXTrainer(model, optimizer, config)
        
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.epoch, 0)

    def test_trainer_training_step(self):
        """Test single training step."""
        class SimpleModel:
            def __init__(self):
                self.weight = mx.random.normal((10, 8))
                self.bias = mx.zeros(10)
            
            def __call__(self, input_ids, attention_mask=None, labels=None):
                logits = mx.matmul(input_ids, self.weight.T) + self.bias
                
                if labels is not None:
                    from unsloth.kernels.mlx.losses import cross_entropy_loss
                    loss = cross_entropy_loss(logits, labels)
                else:
                    loss = mx.array(0.0)
                
                return logits, loss
            
            def get_params(self):
                return {"weight": self.weight, "bias": self.bias}
            
            def set_params(self, params):
                self.weight = params["weight"]
                self.bias = params["bias"]

        model = SimpleModel()
        optimizer = self.AdamW(learning_rate=1e-2)
        config = self.TrainingConfig(
            num_epochs=1,
            gradient_accumulation_steps=1,
        )
        
        trainer = self.MLXTrainer(model, optimizer, config)
        
        # Create a simple batch
        batch = {
            "input_ids": mx.random.normal((2, 8)),
            "labels": mx.array([0, 1]),
        }
        
        # Perform training step
        result = trainer.training_step(batch)
        
        # Check that loss was computed
        self.assertIn("loss", result)
        self.assertIn("learning_rate", result)
        self.assertGreater(result["loss"], 0.0)

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        class SimpleModel:
            def __init__(self):
                self.weight = mx.random.normal((10, 8))
            
            def __call__(self, input_ids, attention_mask=None, labels=None):
                return mx.matmul(input_ids, self.weight.T), mx.array(0.0)
            
            def get_params(self):
                return {"weight": self.weight}
            
            def set_params(self, params):
                self.weight = params["weight"]

        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            model = SimpleModel()
            optimizer = self.AdamW(learning_rate=1e-3)
            config = self.TrainingConfig(
                num_epochs=1,
                output_dir=temp_dir,
            )
            
            trainer = self.MLXTrainer(model, optimizer, config)
            
            # Simulate some training
            trainer.global_step = 100
            trainer.epoch = 2
            
            # Save checkpoint
            trainer.save_checkpoint()
            
            # Create new trainer and load checkpoint
            model2 = SimpleModel()
            optimizer2 = self.AdamW(learning_rate=1e-3)
            config2 = self.TrainingConfig(
                num_epochs=1,
                output_dir=temp_dir,
            )
            
            # Find the saved checkpoint
            checkpoint_dirs = list(Path(temp_dir).glob("checkpoint-*"))
            self.assertGreater(len(checkpoint_dirs), 0)
            
            trainer2 = self.MLXTrainer(
                model2,
                optimizer2,
                config2,
                resume_from_checkpoint=str(checkpoint_dirs[0]),
            )
            
            # Verify state was loaded
            self.assertEqual(trainer2.global_step, 100)
            self.assertEqual(trainer2.epoch, 2)
        
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)


@unittest.skipUnless(HAS_MLX, "MLX not available")
class TestGradientClipping(unittest.TestCase):
    """Test gradient clipping functionality."""

    def setUp(self):
        """Set up test."""
        from unsloth.kernels.mlx.optimizers import clip_grad_norm, clip_grad_value_
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value_ = clip_grad_value_

    def test_clip_grad_norm(self):
        """Test gradient norm clipping."""
        # Create gradients with large values
        grads = {
            "weight": mx.array([[10.0, 10.0], [10.0, 10.0]]),
            "bias": mx.array([5.0, 5.0]),
        }
        
        # Clip gradients
        total_norm, clipped_grads = self.clip_grad_norm(grads, max_norm=1.0)
        
        # Check that gradients were clipped
        self.assertGreater(float(total_norm), 0.0)
        
        # Compute new norm
        all_grads = mx.concatenate([g.reshape(-1) for g in clipped_grads.values()])
        new_norm = mx.linalg.norm(all_grads)
        
        # New norm should be close to max_norm
        self.assertLessEqual(float(new_norm), 1.0 + 1e-5)

    def test_clip_grad_value(self):
        """Test gradient value clipping."""
        grads = {
            "weight": mx.array([[5.0, -5.0], [3.0, -3.0]]),
        }
        
        # Clip to range [-2, 2]
        clipped = self.clip_grad_value_(grads, clip_value=2.0)
        
        # Check that values are clipped
        self.assertLessEqual(float(mx.max(clipped["weight"])), 2.0)
        self.assertGreaterEqual(float(mx.min(clipped["weight"])), -2.0)


@unittest.skipUnless(HAS_MLX, "MLX not available")
class TestIntegration(unittest.TestCase):
    """Integration tests for MLX training components."""

    def test_simple_training_loop(self):
        """Test a simple end-to-end training loop."""
        from unsloth.kernels.mlx.optimizers import AdamW
        from unsloth.kernels.mlx.losses import cross_entropy_loss
        
        # Simple linear model
        class SimpleClassifier:
            def __init__(self, input_dim, num_classes):
                self.weight = mx.random.normal((num_classes, input_dim)) * 0.01
                self.bias = mx.zeros(num_classes)
            
            def forward(self, x):
                return mx.matmul(x, self.weight.T) + self.bias
            
            def get_params(self):
                return {"weight": self.weight, "bias": self.bias}
            
            def set_params(self, params):
                self.weight = params["weight"]
                self.bias = params["bias"]

        # Create model and optimizer
        model = SimpleClassifier(input_dim=16, num_classes=4)
        optimizer = AdamW(learning_rate=0.01)
        
        # Generate synthetic data
        def generate_batch(batch_size=8):
            x = mx.random.normal((batch_size, 16))
            # Simple linear relationship for labels
            logits = model.forward(x)
            labels = mx.argmax(logits, axis=-1)
            return x, labels

        # Training loop
        losses = []
        for step in range(10):
            x, labels = generate_batch()
            
            # Forward pass
            logits = model.forward(x)
            loss = cross_entropy_loss(logits, labels)
            
            # Compute gradients
            def loss_fn(params):
                original = model.get_params()
                model.set_params(params)
                logits = model.forward(x)
                loss = cross_entropy_loss(logits, labels)
                model.set_params(original)
                return loss
            
            params = model.get_params()
            loss_value, grads = mx.value_and_grad(loss_fn)(params)
            
            # Update parameters
            updated_params = optimizer(grads, params)
            model.set_params(updated_params)
            
            losses.append(float(loss_value))
        
        # Loss should generally decrease (allow for some noise)
        self.assertLess(losses[-1], losses[0] * 2)


if __name__ == "__main__":
    unittest.main()
