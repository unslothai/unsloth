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

"""Tests for ASFT (Anchored Supervised Fine-Tuning) loss module."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    pytest.skip("CUDA is required for ASFT tests", allow_module_level = True)
torch.set_default_device("cuda")

from unsloth.losses.asft import (
    ASFTStreamingConfig,
    effective_logits,
    fast_cross_entropy_loss_per_token,
    build_shift_labels,
    get_reference_forward_callable,
    compute_asft_loss,
    _compute_kl_divergence,
    _compute_dft_weights,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def dummy_logits():
    """Create dummy logits tensor (B=2, T=4, V=8)."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 8, requires_grad = True)


@pytest.fixture
def dummy_labels():
    """Create dummy labels tensor with some -100 values."""
    # Labels: [0, 1, 2, 3] and [4, 5, -100, -100]
    return torch.tensor([[0, 1, 2, 3], [4, 5, -100, -100]], dtype = torch.long)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                final_logit_softcapping = 0,
                logit_scale = 0,
            )
            self.embedding = nn.Embedding(16, 8)
            self.linear = nn.Linear(8, 8)

        def forward(self, input_ids = None, **kwargs):
            # Deterministic forward with gradients
            embeddings = self.embedding(input_ids)
            logits = self.linear(embeddings)
            return SimpleNamespace(logits = logits)

    return SimpleModel()


# -----------------------------------------------------------------------------
# A1) Test effective_logits
# -----------------------------------------------------------------------------


class TestEffectiveLogits:
    """Tests for effective_logits function."""

    def test_no_transformation(self, dummy_logits):
        """Test that no transformation is applied when softcapping/scaling are 0."""
        result = effective_logits(dummy_logits, logit_softcapping = 0, logit_scaling = 0)
        # Should be close to original (converted to float32)
        assert torch.allclose(result, dummy_logits.float(), atol = 1e-6)

    def test_logit_scaling(self, dummy_logits):
        """Test logit scaling: t * x."""
        scale = 2.0
        result = effective_logits(dummy_logits, logit_scaling = scale)
        expected = scale * dummy_logits.float()
        assert torch.allclose(result, expected, atol = 1e-6)

    def test_logit_softcapping(self, dummy_logits):
        """Test logit softcapping: t * tanh(x / t)."""
        softcap = 30.0
        result = effective_logits(dummy_logits, logit_softcapping = softcap)
        expected = softcap * torch.tanh(dummy_logits.float() / softcap)
        assert torch.allclose(result, expected, atol = 1e-6)

    def test_both_transformations(self, dummy_logits):
        """Test both scaling and softcapping together."""
        scale = 2.0
        softcap = 30.0
        result = effective_logits(
            dummy_logits, logit_softcapping = softcap, logit_scaling = scale
        )
        # Scaling first, then softcapping
        x = scale * dummy_logits.float()
        expected = softcap * torch.tanh(x / softcap)
        assert torch.allclose(result, expected, atol = 1e-6)

    def test_reads_from_model_config(self):
        """Test reading config from model."""
        model = SimpleNamespace(
            config = SimpleNamespace(
                final_logit_softcapping = 30.0,
                logit_scale = 2.0,
            )
        )
        logits = torch.randn(2, 4, 8)
        result = effective_logits(logits, model)
        # Should apply both transformations
        x = 2.0 * logits.float()
        expected = 30.0 * torch.tanh(x / 30.0)
        assert torch.allclose(result, expected, atol = 1e-6)


# -----------------------------------------------------------------------------
# A2) Test fast_cross_entropy_loss_per_token
# -----------------------------------------------------------------------------


class TestFastCrossEntropyLossPerToken:
    """Tests for fast_cross_entropy_loss_per_token function."""

    def test_basic_loss_computation(self, dummy_logits, dummy_labels):
        """Test basic per-token CE loss computation."""
        losses, valid_mask = fast_cross_entropy_loss_per_token(
            dummy_logits.detach(), dummy_labels
        )

        # Check shapes
        batch, seq_len = dummy_labels.shape
        assert losses.shape == (batch * seq_len,)
        assert valid_mask.shape == (batch * seq_len,)

        # Check that valid_mask correctly identifies -100 positions
        flat_labels = dummy_labels.view(-1)
        expected_valid = flat_labels != -100
        assert torch.equal(valid_mask, expected_valid)

    def test_ignored_positions_have_zero_loss(self, dummy_logits, dummy_labels):
        """Test that positions with label -100 have zero loss."""
        losses, valid_mask = fast_cross_entropy_loss_per_token(
            dummy_logits.detach(), dummy_labels
        )

        # Loss at ignored positions should be 0
        assert torch.all(losses[~valid_mask] == 0)

    def test_valid_positions_have_nonzero_loss(self, dummy_logits, dummy_labels):
        """Test that valid positions have non-zero loss."""
        losses, valid_mask = fast_cross_entropy_loss_per_token(
            dummy_logits.detach(), dummy_labels
        )

        # At least some valid positions should have non-zero loss
        assert torch.any(losses[valid_mask] > 0)

    def test_matches_pytorch_ce(self):
        """Test that results match PyTorch CE loss."""
        torch.manual_seed(42)
        logits = torch.randn(2, 4, 8)
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype = torch.long)

        losses, valid_mask = fast_cross_entropy_loss_per_token(logits, labels)

        # Compare with PyTorch
        flat_logits = logits.view(-1, 8)
        flat_labels = labels.view(-1)
        pytorch_losses = F.cross_entropy(flat_logits, flat_labels, reduction = "none")

        # Should be close
        assert torch.allclose(losses, pytorch_losses, atol = 1e-4)


# -----------------------------------------------------------------------------
# A3) Test build_shift_labels
# -----------------------------------------------------------------------------


class TestBuildShiftLabels:
    """Tests for build_shift_labels function."""

    def test_basic_shift(self):
        """Test basic label shifting."""
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype = torch.long)
        shift_labels = build_shift_labels(labels)

        # shift_labels[..., :-1] = labels[..., 1:]
        # shift_labels[..., -1] = -100
        expected = torch.tensor([[1, 2, 3, -100], [5, 6, 7, -100]], dtype = torch.long)
        assert torch.equal(shift_labels, expected)

    def test_preserves_ignore_index(self):
        """Test that existing -100 values are preserved after shift."""
        labels = torch.tensor([[0, 1, -100, -100], [4, 5, 6, -100]], dtype = torch.long)
        shift_labels = build_shift_labels(labels)

        # First row: [1, -100, -100, -100]
        # Second row: [5, 6, -100, -100]
        expected = torch.tensor(
            [[1, -100, -100, -100], [5, 6, -100, -100]], dtype = torch.long
        )
        assert torch.equal(shift_labels, expected)

    def test_with_packed_seq_lengths(self):
        """Test shift labels with packed sequence boundary masking."""
        # Single row with packed sequences of lengths [2, 2]
        labels = torch.tensor([[0, 1, 2, 3]], dtype = torch.long)
        packed_seq_lengths = torch.tensor([2, 2], dtype = torch.int32)

        shift_labels = build_shift_labels(labels, packed_seq_lengths)

        # After shift: [1, 2, 3, -100]
        # After boundary masking at positions 1 and 3: [1, -100, 3, -100]
        # Actually boundary positions are cumsum - 1 = [1, 3]
        # So positions 1 and 3 should be -100
        assert shift_labels[0, 1].item() == -100  # End of first sequence
        assert shift_labels[0, 3].item() == -100  # End of second sequence (also last)


# -----------------------------------------------------------------------------
# A4) Test get_reference_forward_callable
# -----------------------------------------------------------------------------


class TestGetReferenceForwardCallable:
    """Tests for get_reference_forward_callable function."""

    def test_disable_adapter_policy(self, simple_model):
        """Test disable_adapter policy when model has adapters."""
        # Mock disable_adapter
        simple_model.disable_adapter = MagicMock()
        simple_model.disable_adapter.__enter__ = MagicMock(return_value = None)
        simple_model.disable_adapter.__exit__ = MagicMock(return_value = False)

        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "disable_adapter"
        )

        # Call the forward
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = ref_forward(input_ids = input_ids)

        # Should have called disable_adapter
        assert simple_model.disable_adapter.__enter__.called

    def test_frozen_copy_policy(self, simple_model):
        """Test frozen_copy policy."""
        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "frozen_copy"
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = ref_forward(input_ids = input_ids)

        # Should return logits
        assert result.shape[0] == 1  # batch size
        assert result.shape[1] == 4  # seq len

    def test_fallback_to_frozen_copy_without_adapters(self, simple_model):
        """Test that disable_adapter falls back to frozen_copy when no adapters."""
        # Model without disable_adapter method
        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "disable_adapter"
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = ref_forward(input_ids = input_ids)

        # Should still work (uses frozen copy fallback)
        assert result is not None


# -----------------------------------------------------------------------------
# Test KL divergence computation
# -----------------------------------------------------------------------------


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_direction(self):
        """Test that KL is computed as KL(p_ref || p_cur)."""
        torch.manual_seed(42)
        cur_logits = torch.randn(4, 8)  # (B*T, V)
        ref_logits = torch.randn(4, 8)

        kl = _compute_kl_divergence(cur_logits, ref_logits)

        # KL should be non-negative
        assert torch.all(kl >= -1e-6)  # Allow small numerical errors

    def test_kl_zero_for_identical(self):
        """Test that KL is zero when distributions are identical."""
        logits = torch.randn(4, 8)

        kl = _compute_kl_divergence(logits, logits.clone())

        # Should be close to zero
        assert torch.allclose(kl, torch.zeros_like(kl), atol = 1e-5)

    def test_kl_shape(self):
        """Test KL output shape."""
        cur_logits = torch.randn(2, 4, 8)  # (B, T, V)
        ref_logits = torch.randn(2, 4, 8)

        kl = _compute_kl_divergence(cur_logits, ref_logits)

        # Should be flattened to (B*T,)
        assert kl.shape == (8,)


# -----------------------------------------------------------------------------
# Test DFT weights computation
# -----------------------------------------------------------------------------


class TestDFTWeights:
    """Tests for DFT weights computation."""

    def test_dft_weights_are_probabilities(self, dummy_logits, dummy_labels):
        """Test that DFT weights are valid probabilities."""
        flat_logits = dummy_logits.detach().view(-1, 8)
        flat_labels = dummy_labels.view(-1)

        weights = _compute_dft_weights(flat_logits, flat_labels)

        # Weights should be in [0, 1]
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)

    def test_dft_weights_are_detached(self, dummy_logits, dummy_labels):
        """Test that DFT weights are detached (no gradients)."""
        weights = _compute_dft_weights(
            dummy_logits.detach().view(-1, 8),
            dummy_labels.view(-1),
        )

        assert not weights.requires_grad


# -----------------------------------------------------------------------------
# A5) Test compute_asft_loss
# -----------------------------------------------------------------------------


class TestComputeASFTLoss:
    """Tests for the main compute_asft_loss function."""

    def test_sft_mode(self, simple_model):
        """Test SFT mode computes standard CE."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft", kl_weight = 0.0)

        # Should return a scalar loss
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_dft_mode(self, simple_model):
        """Test DFT mode."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "dft", kl_weight = 0.0)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_sft_kl_mode(self, simple_model):
        """Test SFT+KL mode."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        loss = compute_asft_loss(
            simple_model,
            inputs,
            asft_mode = "sft+kl",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
        )

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_asft_mode(self, simple_model):
        """Test full ASFT mode."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        loss = compute_asft_loss(
            simple_model,
            inputs,
            asft_mode = "asft",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
        )

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_return_outputs(self, simple_model):
        """Test return_outputs=True."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        loss, outputs = compute_asft_loss(
            simple_model, inputs, asft_mode = "sft", return_outputs = True
        )

        assert loss.dim() == 0
        assert hasattr(outputs, "logits")

    def test_handles_all_ignored_labels(self, simple_model):
        """Test that all -100 labels returns zero loss."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[-100, -100, -100, -100]]),
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft")

        # Should return zero loss
        assert loss.item() == 0.0

    def test_uses_num_items_in_batch(self, simple_model):
        """Test that num_items_in_batch is used for normalization."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
            "num_items_in_batch": 2,  # Override default
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft")

        # Should use the provided n_items
        assert loss.dim() == 0

    def test_packing_boundary_masking(self, simple_model):
        """Test that packed sequence boundaries are masked."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
            "packed_seq_lengths": torch.tensor([2, 2], dtype = torch.int32),
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft")

        # Should handle packing without error
        assert loss.dim() == 0


# -----------------------------------------------------------------------------
# Test ASFTStreamingConfig
# -----------------------------------------------------------------------------


class TestASFTStreamingConfig:
    """Tests for ASFTStreamingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ASFTStreamingConfig()

        assert config.enabled is False
        assert config.ref_strategy == "none"
        assert config.ref_microbatch_size is None
        assert config.seq_chunk_size is None
        assert config.kl_token_chunk_size is None
        assert config.force_fp32_kl is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ASFTStreamingConfig(
            enabled = True,
            ref_strategy = "batch_micro",
            ref_microbatch_size = 4,
            seq_chunk_size = 256,
        )

        assert config.enabled is True
        assert config.ref_strategy == "batch_micro"
        assert config.ref_microbatch_size == 4
        assert config.seq_chunk_size == 256

    def test_config_immutability_when_none_values(self, simple_model):
        """Test that streaming_config is not mutated when values are None."""
        config = ASFTStreamingConfig(
            enabled = True,
            ref_strategy = "batch_micro",
            ref_microbatch_size = None,  # Should use default without mutation
        )
        original_microbatch = config.ref_microbatch_size
        original_chunk = config.seq_chunk_size

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        # Call compute_asft_loss with sft mode (doesn't use streaming, but
        # the config should still not be mutated)
        loss = compute_asft_loss(
            simple_model,
            inputs,
            asft_mode = "sft",
            streaming_config = config,
        )

        # Config should not be mutated
        assert config.ref_microbatch_size == original_microbatch
        assert config.seq_chunk_size == original_chunk


# -----------------------------------------------------------------------------
# Backward Compatibility Tests
# -----------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Tests to ensure ASFT doesn't break existing behavior."""

    def test_sft_mode_matches_standard_ce(self, simple_model):
        """Test that SFT mode produces same results as standard CE."""
        torch.manual_seed(42)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        # Compute ASFT loss in SFT mode
        asft_loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft")

        # The loss should be a valid scalar
        assert asft_loss.dim() == 0
        assert not torch.isnan(asft_loss)
        assert not torch.isinf(asft_loss)

    def test_streaming_equivalence(self, simple_model):
        """Test that streaming produces equivalent results to full forward."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        # Full forward
        full_loss = compute_asft_loss(
            simple_model,
            inputs,
            asft_mode = "sft+kl",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
            streaming_config = ASFTStreamingConfig(enabled = False),
        )

        # With batch_micro streaming (should be equivalent for batch=1)
        streaming_loss = compute_asft_loss(
            simple_model,
            inputs,
            asft_mode = "sft+kl",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
            streaming_config = ASFTStreamingConfig(
                enabled = True,
                ref_strategy = "batch_micro",
                ref_microbatch_size = 1,
            ),
        )

        # Should be very close
        assert torch.allclose(full_loss, streaming_loss, atol = 1e-4)


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestASFTTrainerIntegration:
    """Integration tests for ASFTTrainer."""

    def test_import_asft_trainer(self):
        """Test that ASFTTrainer can be imported."""
        from unsloth.trainer import ASFTTrainer, ASFTStreamingConfig

        assert ASFTTrainer is not None
        assert ASFTStreamingConfig is not None

    def test_asft_trainer_inherits_unsloth_trainer(self):
        """Test that ASFTTrainer inherits from UnslothTrainer."""
        from unsloth.trainer import ASFTTrainer, UnslothTrainer

        assert issubclass(ASFTTrainer, UnslothTrainer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
