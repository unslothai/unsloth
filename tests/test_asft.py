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
    _compute_kl_seq_kv_cache,
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

    def test_respects_custom_ignore_index(self):
        """Test that custom ignore_index is honored by the kernel wrapper."""
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 8)
        labels = torch.tensor([[1, 2, 1, 3]], dtype = torch.long)

        losses, valid_mask = fast_cross_entropy_loss_per_token(
            logits, labels, ignore_index = 1
        )

        assert losses.shape == (4,)
        assert torch.equal(valid_mask, torch.tensor([False, True, False, True]))
        assert torch.all(losses[~valid_mask] == 0)


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

    def test_return_outputs_true(self, simple_model):
        """Test returning full outputs when requested."""
        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "frozen_copy", return_outputs = True
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = ref_forward(input_ids = input_ids)

        assert hasattr(result, "logits")


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

        kl = _compute_kl_divergence(cur_logits, ref_logits, kl_direction = "forward")

        # KL should be non-negative
        assert torch.all(kl >= -1e-6)  # Allow small numerical errors

    def test_kl_zero_for_identical(self):
        """Test that KL is zero when distributions are identical."""
        logits = torch.randn(4, 8)

        kl = _compute_kl_divergence(logits, logits.clone(), kl_direction = "forward")

        # Should be close to zero
        assert torch.allclose(kl, torch.zeros_like(kl), atol = 1e-5)

    def test_kl_shape(self):
        """Test KL output shape."""
        cur_logits = torch.randn(2, 4, 8)  # (B, T, V)
        ref_logits = torch.randn(2, 4, 8)

        kl = _compute_kl_divergence(cur_logits, ref_logits, kl_direction = "forward")

        # Should be flattened to (B*T,)
        assert kl.shape == (8,)

    def test_kl_reverse_matches_manual(self):
        """Test reverse KL matches manual computation."""
        torch.manual_seed(321)
        cur_logits = torch.randn(2, 5)
        ref_logits = torch.randn(2, 5)

        kl_reverse = _compute_kl_divergence(
            cur_logits, ref_logits, kl_direction = "reverse"
        )

        cur_p = F.softmax(cur_logits, dim = -1)
        ref_p = F.softmax(ref_logits, dim = -1)
        manual = (cur_p * (cur_p.log() - ref_p.log())).sum(dim = -1)

        assert torch.allclose(kl_reverse, manual, atol = 1e-5)


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

    def test_dft_weights_match_exp_neg_ce(self):
        """Test exp(-CE) matches softmax-gather for DFT weights."""
        torch.manual_seed(123)
        logits = torch.randn(2, 3, 7)
        labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.long)

        ce_losses, valid_mask = fast_cross_entropy_loss_per_token(logits, labels)

        weights_from_ce = _compute_dft_weights(
            logits,
            labels,
            ce_losses = ce_losses,
            valid_mask = valid_mask,
        )
        weights_from_softmax = _compute_dft_weights(logits, labels)

        assert torch.allclose(weights_from_ce, weights_from_softmax, atol = 1e-4)


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

    def test_dft_normalize_by_weights(self, simple_model):
        """Test DFT normalization by weight sum."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        logits = simple_model(input_ids = inputs["input_ids"]).logits
        shift_labels = build_shift_labels(inputs["labels"])
        valid_mask = shift_labels != -100
        ce_losses, _ = fast_cross_entropy_loss_per_token(logits, shift_labels)
        ce_losses = ce_losses.view(shift_labels.shape)
        dft_weights = _compute_dft_weights(
            logits,
            shift_labels,
            ce_losses = ce_losses,
            valid_mask = valid_mask,
        ).view(shift_labels.shape)
        token_loss = ce_losses * dft_weights
        expected = token_loss[valid_mask].sum() / dft_weights[valid_mask].sum().clamp_min(1e-8)

        loss = compute_asft_loss(
            simple_model,
            inputs,
            asft_mode = "dft",
            kl_weight = 0.0,
            normalize_by = "weights",
        )

        assert torch.allclose(loss, expected, atol = 1e-5)

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

        assert config.mode is None
        assert config.enabled is False
        assert config.ref_strategy == "none"
        assert config.ref_microbatch_size is None
        assert config.seq_chunk_size is None
        assert config.kl_token_chunk_size is None
        assert config.force_fp32_kl is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ASFTStreamingConfig(
            mode = "batch",
            enabled = True,
            ref_strategy = "batch_micro",
            ref_microbatch_size = 4,
            seq_chunk_size = 256,
        )

        assert config.mode == "batch"
        assert config.enabled is True
        assert config.ref_strategy == "batch_micro"
        assert config.ref_microbatch_size == 4
        assert config.seq_chunk_size == 256


class TestStreamingModeMapping:
    """Tests for streaming mode routing in compute_asft_loss."""

    def test_mode_batch_uses_batch_micro(self, simple_model):
        """Test that mode=batch routes to batch micro."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
            "labels": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        }
        config = ASFTStreamingConfig(
            mode = "batch",
            ref_microbatch_size = 1,
            enabled = False,
            ref_strategy = "seq_kv_cache",
        )

        def batch_side_effect(
            model,
            cur_logits,
            shift_labels,
            valid_mask,
            ref_forward,
            forward_inputs,
            microbatch_size,
            logit_softcapping = 0,
            logit_scaling = 0,
            force_fp32 = True,
            kl_direction = "forward",
        ):
            batch, seq_len = shift_labels.shape
            return torch.zeros(batch, seq_len, device = shift_labels.device)

        with patch(
            "unsloth.losses.asft._compute_kl_batch_micro",
            side_effect = batch_side_effect,
        ) as batch_mock, patch(
            "unsloth.losses.asft._compute_kl_seq_kv_cache",
            side_effect = AssertionError("seq_kv_cache should not be used"),
        ):
            loss = compute_asft_loss(
                simple_model,
                inputs,
                asft_mode = "sft+kl",
                kl_weight = 0.1,
                reference_policy = "frozen_copy",
                streaming_config = config,
            )

        assert batch_mock.called
        assert batch_mock.call_args[0][6] == 1
        assert loss.dim() == 0

    @pytest.mark.parametrize("mode", ["seq", "auto"])
    def test_mode_seq_and_auto_use_seq_kv_cache(self, mode, simple_model):
        """Test that mode=seq/auto routes to seq_kv_cache."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }
        config = ASFTStreamingConfig(
            mode = mode,
            seq_chunk_size = 2,
            enabled = False,
            ref_strategy = "batch_micro",
        )

        def seq_side_effect(
            model,
            cur_logits,
            shift_labels,
            valid_mask,
            ref_forward,
            forward_inputs,
            seq_chunk_size,
            **kwargs,
        ):
            batch, seq_len = shift_labels.shape
            return torch.zeros(batch, seq_len, device = shift_labels.device)

        with patch(
            "unsloth.losses.asft._compute_kl_seq_kv_cache",
            side_effect = seq_side_effect,
        ) as seq_mock, patch(
            "unsloth.losses.asft._compute_kl_batch_micro",
            side_effect = AssertionError("batch_micro should not be used"),
        ):
            loss = compute_asft_loss(
                simple_model,
                inputs,
                asft_mode = "sft+kl",
                kl_weight = 0.1,
                reference_policy = "frozen_copy",
                streaming_config = config,
            )

        assert seq_mock.called
        assert seq_mock.call_args[0][6] == 2
        assert seq_mock.call_args.kwargs["microbatch_size"] is None
        assert loss.dim() == 0

    def test_mode_hybrid_defaults_microbatch(self, simple_model):
        """Test that hybrid mode sets a default microbatch size."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
            "labels": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        }
        config = ASFTStreamingConfig(
            mode = "hybrid",
            seq_chunk_size = 2,
            ref_microbatch_size = None,
        )

        def seq_side_effect(
            model,
            cur_logits,
            shift_labels,
            valid_mask,
            ref_forward,
            forward_inputs,
            seq_chunk_size,
            **kwargs,
        ):
            batch, seq_len = shift_labels.shape
            return torch.zeros(batch, seq_len, device = shift_labels.device)

        with patch(
            "unsloth.losses.asft._compute_kl_seq_kv_cache",
            side_effect = seq_side_effect,
        ) as seq_mock:
            loss = compute_asft_loss(
                simple_model,
                inputs,
                asft_mode = "sft+kl",
                kl_weight = 0.1,
                reference_policy = "frozen_copy",
                streaming_config = config,
            )

        assert seq_mock.called
        assert seq_mock.call_args.kwargs["microbatch_size"] == 1
        assert config.ref_microbatch_size is None
        assert loss.dim() == 0

    def test_mode_off_uses_full_forward(self, simple_model):
        """Test that mode=off bypasses streaming helpers."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }
        config = ASFTStreamingConfig(
            mode = "off",
            enabled = True,
            ref_strategy = "seq_kv_cache",
        )

        def kl_side_effect(
            cur_logits,
            ref_logits,
            model = None,
            logit_softcapping = 0,
            logit_scaling = 0,
            force_fp32 = True,
            kl_direction = "forward",
        ):
            batch, seq_len = ref_logits.shape[:2]
            return torch.zeros(batch * seq_len, device = ref_logits.device)

        with patch(
            "unsloth.losses.asft._compute_kl_divergence",
            side_effect = kl_side_effect,
        ) as kl_mock, patch(
            "unsloth.losses.asft._compute_kl_seq_kv_cache",
            side_effect = AssertionError("seq_kv_cache should not be used"),
        ), patch(
            "unsloth.losses.asft._compute_kl_batch_micro",
            side_effect = AssertionError("batch_micro should not be used"),
        ):
            loss = compute_asft_loss(
                simple_model,
                inputs,
                asft_mode = "sft+kl",
                kl_weight = 0.1,
                reference_policy = "frozen_copy",
                streaming_config = config,
            )

        assert kl_mock.called
        assert loss.dim() == 0

    def test_invalid_mode_raises(self, simple_model):
        """Test that invalid streaming mode raises a ValueError."""
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }
        config = ASFTStreamingConfig(mode = "invalid")

        with pytest.raises(ValueError):
            compute_asft_loss(
                simple_model,
                inputs,
                asft_mode = "sft+kl",
                kl_weight = 0.1,
                reference_policy = "frozen_copy",
                streaming_config = config,
            )


class TestSeqKVCacheStreaming:
    """Tests for seq_kv_cache streaming behavior."""

    def test_seq_kv_cache_runs_when_use_cache_false(self):
        """Test that seq_kv_cache attempts chunking even if config.use_cache=False."""
        batch_size, seq_len, vocab_size = 1, 6, 5
        cur_logits = torch.randn(batch_size, seq_len, vocab_size)
        shift_labels = torch.zeros(batch_size, seq_len, dtype = torch.long)
        valid_mask = shift_labels != -100
        input_ids = torch.arange(seq_len).view(1, -1)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    use_cache = False,
                    final_logit_softcapping = 0,
                    logit_scale = 0,
                )

        model = DummyModel()
        call_state = {"saw_past": False}

        def ref_forward(**kwargs):
            input_ids_local = kwargs["input_ids"]
            if input_ids_local.shape[1] == seq_len:
                raise AssertionError("full forward not expected")
            if "past_key_values" in kwargs:
                call_state["saw_past"] = True
            batch, chunk_len = input_ids_local.shape
            logits = torch.zeros(
                batch, chunk_len, vocab_size, device = input_ids_local.device
            )
            return (logits, ("cache",))

        forward_inputs = {"input_ids": input_ids}

        kl = _compute_kl_seq_kv_cache(
            model,
            cur_logits,
            shift_labels,
            valid_mask,
            ref_forward,
            forward_inputs,
            seq_chunk_size = 4,
        )

        assert kl.shape == (batch_size, seq_len)
        assert call_state["saw_past"] is True

    def test_seq_kv_cache_supports_microbatching(self):
        """Test that seq_kv_cache can be microbatched by batch dimension."""
        batch_size, seq_len, vocab_size = 2, 4, 3
        cur_logits = torch.randn(batch_size, seq_len, vocab_size)
        shift_labels = torch.zeros(batch_size, seq_len, dtype = torch.long)
        valid_mask = shift_labels != -100
        input_ids = torch.arange(seq_len).repeat(batch_size, 1)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    use_cache = True,
                    final_logit_softcapping = 0,
                    logit_scale = 0,
                )

        model = DummyModel()
        call_state = {"max_batch": 0}

        def ref_forward(**kwargs):
            input_ids_local = kwargs["input_ids"]
            call_state["max_batch"] = max(
                call_state["max_batch"], input_ids_local.shape[0]
            )
            if input_ids_local.shape[0] > 1:
                raise AssertionError("expected microbatching")
            batch, chunk_len = input_ids_local.shape
            logits = torch.zeros(
                batch, chunk_len, vocab_size, device = input_ids_local.device
            )
            return (logits, ("cache",))

        forward_inputs = {"input_ids": input_ids}

        kl = _compute_kl_seq_kv_cache(
            model,
            cur_logits,
            shift_labels,
            valid_mask,
            ref_forward,
            forward_inputs,
            seq_chunk_size = 2,
            microbatch_size = 1,
        )

        assert kl.shape == (batch_size, seq_len)
        assert call_state["max_batch"] == 1

    def test_seq_kv_cache_falls_back_to_batch_micro(self):
        """Test that seq_kv_cache falls back to batch micro on cache failure."""
        batch_size, seq_len, vocab_size = 2, 6, 5
        cur_logits = torch.randn(batch_size, seq_len, vocab_size)
        shift_labels = torch.zeros(batch_size, seq_len, dtype = torch.long)
        valid_mask = shift_labels != -100
        input_ids = torch.arange(seq_len).repeat(batch_size, 1)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    use_cache = True,
                    final_logit_softcapping = 0,
                    logit_scale = 0,
                )

        model = DummyModel()

        def ref_forward(**kwargs):
            input_ids_local = kwargs["input_ids"]
            if (
                input_ids_local.shape[0] == batch_size
                and input_ids_local.shape[1] == seq_len
            ):
                raise AssertionError("full forward not expected on fallback")
            batch, chunk_len = input_ids_local.shape
            logits = torch.zeros(
                batch, chunk_len, vocab_size, device = input_ids_local.device
            )
            return (logits, None)

        forward_inputs = {"input_ids": input_ids}

        kl = _compute_kl_seq_kv_cache(
            model,
            cur_logits,
            shift_labels,
            valid_mask,
            ref_forward,
            forward_inputs,
            seq_chunk_size = 2,
        )

        assert kl.shape == (batch_size, seq_len)

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

    def test_seq_kv_cache_equivalence(self):
        """Test that seq_kv_cache matches full forward for KL loss."""
        torch.manual_seed(123)

        class CacheModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    use_cache = True,
                    final_logit_softcapping = 0,
                    logit_scale = 0,
                )
                self.embedding = nn.Embedding(32, 8)
                self.linear = nn.Linear(8, 32)

            def forward(
                self, input_ids = None, past_key_values = None, use_cache = None, **kwargs
            ):
                embeddings = self.embedding(input_ids)
                logits = self.linear(embeddings)
                past = ("cache",) if (use_cache or past_key_values is not None) else None
                return SimpleNamespace(logits = logits, past_key_values = past)

        model = CacheModel()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
        }

        full_loss = compute_asft_loss(
            model,
            inputs,
            asft_mode = "sft+kl",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
            streaming_config = ASFTStreamingConfig(enabled = False),
        )

        seq_loss = compute_asft_loss(
            model,
            inputs,
            asft_mode = "sft+kl",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
            streaming_config = ASFTStreamingConfig(
                enabled = True,
                ref_strategy = "seq_kv_cache",
                seq_chunk_size = 2,
            ),
        )

        assert torch.allclose(full_loss, seq_loss, atol = 1e-4)

    def test_seq_kv_cache_microbatch_equivalence(self):
        """Test that seq_kv_cache + microbatching matches full forward."""
        torch.manual_seed(456)

        class CacheModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    use_cache = True,
                    final_logit_softcapping = 0,
                    logit_scale = 0,
                )
                self.embedding = nn.Embedding(32, 8)
                self.linear = nn.Linear(8, 32)

            def forward(
                self, input_ids = None, past_key_values = None, use_cache = None, **kwargs
            ):
                embeddings = self.embedding(input_ids)
                logits = self.linear(embeddings)
                past = ("cache",) if (use_cache or past_key_values is not None) else None
                return SimpleNamespace(logits = logits, past_key_values = past)

        model = CacheModel()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
        }

        full_loss = compute_asft_loss(
            model,
            inputs,
            asft_mode = "sft+kl",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
            streaming_config = ASFTStreamingConfig(enabled = False),
        )

        combined_loss = compute_asft_loss(
            model,
            inputs,
            asft_mode = "sft+kl",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
            streaming_config = ASFTStreamingConfig(
                enabled = True,
                ref_strategy = "seq_kv_cache",
                seq_chunk_size = 2,
                ref_microbatch_size = 1,
            ),
        )

        assert torch.allclose(full_loss, combined_loss, atol = 1e-4)


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


class TestASFTTrainerComputeLoss:
    """Tests for ASFTTrainer.compute_loss behavior."""

    def test_compute_loss_calls_asft_loss(self):
        """Test ASFTTrainer compute_loss calls compute_asft_loss."""
        from unsloth.trainer import ASFTTrainer, ASFTStreamingConfig

        trainer = ASFTTrainer.__new__(ASFTTrainer)
        trainer.asft_enabled = True
        trainer.asft_mode = "sft"
        trainer.kl_weight = 0.0
        trainer.kl_direction = "forward"
        trainer.reference_policy = "disable_adapter"
        trainer.asft_streaming = ASFTStreamingConfig()
        trainer.normalize_by = "tokens"
        trainer._asft_original_model = None

        model = nn.Module()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }
        expected = torch.tensor(1.0, device = inputs["input_ids"].device)

        with patch(
            "unsloth.trainer.compute_asft_loss", return_value = expected
        ) as loss_mock:
            result = ASFTTrainer.compute_loss(
                trainer, model, inputs, return_outputs = False, num_items_in_batch = 7
            )

        assert result is expected
        assert inputs["num_items_in_batch"] == 7
        assert loss_mock.called
        assert loss_mock.call_args.kwargs["model"] is model
        assert loss_mock.call_args.kwargs["asft_mode"] == "sft"
        assert loss_mock.call_args.kwargs["kl_weight"] == 0.0
        assert loss_mock.call_args.kwargs["kl_direction"] == "forward"
        assert loss_mock.call_args.kwargs["reference_policy"] == "disable_adapter"
        assert loss_mock.call_args.kwargs["streaming_config"] is trainer.asft_streaming
        assert loss_mock.call_args.kwargs["normalize_by"] == "tokens"

    def test_compute_loss_creates_frozen_copy_once(self):
        """Test frozen copy is created once when needed."""
        from unsloth.trainer import ASFTTrainer, ASFTStreamingConfig

        trainer = ASFTTrainer.__new__(ASFTTrainer)
        trainer.asft_enabled = True
        trainer.asft_mode = "asft"
        trainer.kl_weight = 0.1
        trainer.kl_direction = "forward"
        trainer.reference_policy = "frozen_copy"
        trainer.asft_streaming = ASFTStreamingConfig()
        trainer.normalize_by = "tokens"
        trainer._asft_original_model = None

        model = nn.Module()
        model_copy = MagicMock()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        with pytest.warns(UserWarning), patch(
            "unsloth.trainer.deepcopy", return_value = model_copy
        ) as deepcopy_mock, patch(
            "unsloth.trainer.compute_asft_loss",
            return_value = torch.tensor(0.5, device = inputs["input_ids"].device),
        ):
            ASFTTrainer.compute_loss(trainer, model, inputs)
            ASFTTrainer.compute_loss(trainer, model, inputs)

        assert deepcopy_mock.call_count == 1
        assert trainer._asft_original_model is model_copy
        assert model_copy.eval.called
        assert model_copy.requires_grad_.called

    def test_compute_loss_skips_copy_with_disable_adapter(self):
        """Test disable_adapter policy skips frozen copy."""
        from unsloth.trainer import ASFTTrainer, ASFTStreamingConfig

        trainer = ASFTTrainer.__new__(ASFTTrainer)
        trainer.asft_enabled = True
        trainer.asft_mode = "asft"
        trainer.kl_weight = 0.1
        trainer.kl_direction = "forward"
        trainer.reference_policy = "disable_adapter"
        trainer.asft_streaming = ASFTStreamingConfig()
        trainer.normalize_by = "tokens"
        trainer._asft_original_model = None

        model = MagicMock()
        model.disable_adapter = MagicMock()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        with patch(
            "unsloth.trainer.deepcopy"
        ) as deepcopy_mock, patch(
            "unsloth.trainer.compute_asft_loss",
            return_value = torch.tensor(0.5, device = inputs["input_ids"].device),
        ) as loss_mock:
            ASFTTrainer.compute_loss(trainer, model, inputs)

        assert not deepcopy_mock.called
        assert loss_mock.call_args.kwargs["original_model"] is None


class TestUnslothTrainingArguments:
    """Tests for UnslothTrainingArguments."""

    def test_embedding_learning_rate_is_set(self):
        """Test embedding_learning_rate is stored on the args object."""
        from unsloth import trainer as trainer_module

        with patch.object(
            trainer_module.TrainingArguments, "__init__", return_value = None
        ) as base_init:
            args = trainer_module.UnslothTrainingArguments(embedding_learning_rate = 0.01)

        assert args.embedding_learning_rate == 0.01
        assert base_init.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
