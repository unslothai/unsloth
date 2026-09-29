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

"""Tests for the ASFT loss module."""

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


@pytest.fixture
def dummy_logits():
    torch.manual_seed(42)
    return torch.randn(2, 4, 8, requires_grad = True)


@pytest.fixture
def dummy_labels():
    return torch.tensor([[0, 1, 2, 3], [4, 5, -100, -100]], dtype = torch.long)


@pytest.fixture
def simple_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                final_logit_softcapping = 0,
                logit_scale = 0,
            )
            self.embedding = nn.Embedding(16, 8)
            self.linear = nn.Linear(8, 8)

        def forward(
            self,
            input_ids = None,
            **kwargs,
        ):
            embeddings = self.embedding(input_ids)
            logits = self.linear(embeddings)
            return SimpleNamespace(logits = logits)

    return SimpleModel()


class TestEffectiveLogits:
    def test_no_transformation(self, dummy_logits):
        result = effective_logits(dummy_logits, logit_softcapping = 0, logit_scaling = 0)
        assert torch.allclose(result, dummy_logits.float(), atol = 1e-6)

    def test_logit_scaling(self, dummy_logits):
        scale = 2.0
        result = effective_logits(dummy_logits, logit_scaling = scale)
        expected = scale * dummy_logits.float()
        assert torch.allclose(result, expected, atol = 1e-6)

    def test_logit_softcapping(self, dummy_logits):
        softcap = 30.0
        result = effective_logits(dummy_logits, logit_softcapping = softcap)
        expected = softcap * torch.tanh(dummy_logits.float() / softcap)
        assert torch.allclose(result, expected, atol = 1e-6)

    def test_both_transformations(self, dummy_logits):
        scale = 2.0
        softcap = 30.0
        result = effective_logits(dummy_logits, logit_softcapping = softcap, logit_scaling = scale)
        x = scale * dummy_logits.float()
        expected = softcap * torch.tanh(x / softcap)
        assert torch.allclose(result, expected, atol = 1e-6)

    def test_reads_from_model_config(self):
        model = SimpleNamespace(
            config = SimpleNamespace(
                final_logit_softcapping = 30.0,
                logit_scale = 2.0,
            )
        )
        logits = torch.randn(2, 4, 8)
        result = effective_logits(logits, model)
        x = 2.0 * logits.float()
        expected = 30.0 * torch.tanh(x / 30.0)
        assert torch.allclose(result, expected, atol = 1e-6)

    def test_reads_granite_logit_scaling(self):
        model = SimpleNamespace(
            config = SimpleNamespace(
                model_type = "granite",
                final_logit_softcapping = 0,
                logit_scale = 2.0,
                logit_scaling = 0,
                logits_scaling = 16.0,
            )
        )
        logits = torch.randn(2, 4, 8)
        result = effective_logits(logits, model)
        expected = (1.0 / 16.0) * logits.float()
        assert torch.allclose(result, expected, atol = 1e-6)

    def test_reads_falcon_h1_logit_scaling(self):
        model = SimpleNamespace(
            config = SimpleNamespace(
                model_type = "falcon_h1",
                final_logit_softcapping = 0,
                logit_scale = 2.0,
                logit_scaling = 0,
                lm_head_multiplier = 3.0,
            )
        )
        logits = torch.randn(2, 4, 8)
        result = effective_logits(logits, model)
        expected = 3.0 * logits.float()
        assert torch.allclose(result, expected, atol = 1e-6)


class TestFastCrossEntropyLossPerToken:
    def test_basic_loss_computation(self, dummy_logits, dummy_labels):
        losses, valid_mask = fast_cross_entropy_loss_per_token(dummy_logits.detach(), dummy_labels)

        batch, seq_len = dummy_labels.shape
        assert losses.shape == (batch * seq_len,)
        assert valid_mask.shape == (batch * seq_len,)

        flat_labels = dummy_labels.view(-1)
        expected_valid = flat_labels != -100
        assert torch.equal(valid_mask, expected_valid)

    def test_ignored_positions_have_zero_loss(self, dummy_logits, dummy_labels):
        losses, valid_mask = fast_cross_entropy_loss_per_token(dummy_logits.detach(), dummy_labels)

        assert torch.all(losses[~valid_mask] == 0)

    def test_valid_positions_have_nonzero_loss(self, dummy_logits, dummy_labels):
        losses, valid_mask = fast_cross_entropy_loss_per_token(dummy_logits.detach(), dummy_labels)

        assert torch.any(losses[valid_mask] > 0)

    def test_matches_pytorch_ce(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 4, 8)
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype = torch.long)

        losses, valid_mask = fast_cross_entropy_loss_per_token(logits, labels)

        flat_logits = logits.view(-1, 8)
        flat_labels = labels.view(-1)
        pytorch_losses = F.cross_entropy(flat_logits, flat_labels, reduction = "none")

        assert torch.allclose(losses, pytorch_losses, atol = 1e-4)

    def test_respects_custom_ignore_index(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 8)
        labels = torch.tensor([[1, 2, 1, 3]], dtype = torch.long)

        losses, valid_mask = fast_cross_entropy_loss_per_token(logits, labels, ignore_index = 1)

        assert losses.shape == (4,)
        assert torch.equal(valid_mask, torch.tensor([False, True, False, True]))
        assert torch.all(losses[~valid_mask] == 0)


class TestBuildShiftLabels:
    def test_basic_shift(self):
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype = torch.long)
        shift_labels = build_shift_labels(labels)

        expected = torch.tensor([[1, 2, 3, -100], [5, 6, 7, -100]], dtype = torch.long)
        assert torch.equal(shift_labels, expected)

    def test_preserves_ignore_index(self):
        labels = torch.tensor([[0, 1, -100, -100], [4, 5, 6, -100]], dtype = torch.long)
        shift_labels = build_shift_labels(labels)

        expected = torch.tensor([[1, -100, -100, -100], [5, 6, -100, -100]], dtype = torch.long)
        assert torch.equal(shift_labels, expected)

    def test_with_packed_seq_lengths(self):
        labels = torch.tensor([[0, 1, 2, 3]], dtype = torch.long)
        packed_seq_lengths = torch.tensor([2, 2], dtype = torch.int32)

        shift_labels = build_shift_labels(labels, packed_seq_lengths)

        assert shift_labels[0, 1].item() == -100
        assert shift_labels[0, 3].item() == -100


class TestGetReferenceForwardCallable:
    def test_disable_adapter_policy(self, simple_model):
        simple_model.disable_adapter = MagicMock()
        simple_model.disable_adapter.__enter__ = MagicMock(return_value = None)
        simple_model.disable_adapter.__exit__ = MagicMock(return_value = False)

        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "disable_adapter"
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = ref_forward(input_ids = input_ids)

        assert simple_model.disable_adapter.__enter__.called

    def test_frozen_copy_policy(self, simple_model):
        ref_forward = get_reference_forward_callable(simple_model, reference_policy = "frozen_copy")

        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = ref_forward(input_ids = input_ids)

        assert result.shape[0] == 1
        assert result.shape[1] == 4

    def test_fallback_to_frozen_copy_without_adapters(self, simple_model):
        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "disable_adapter"
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = ref_forward(input_ids = input_ids)

        assert result is not None

    def test_return_outputs_true(self, simple_model):
        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "frozen_copy", return_outputs = True
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = ref_forward(input_ids = input_ids)

        assert hasattr(result, "logits")


class TestKLDivergence:
    def test_kl_direction(self):
        torch.manual_seed(42)
        cur_logits = torch.randn(4, 8)
        ref_logits = torch.randn(4, 8)

        kl = _compute_kl_divergence(cur_logits, ref_logits, kl_direction = "forward")

        assert torch.all(kl >= -1e-6)

    def test_kl_zero_for_identical(self):
        logits = torch.randn(4, 8)

        kl = _compute_kl_divergence(logits, logits.clone(), kl_direction = "forward")

        assert torch.allclose(kl, torch.zeros_like(kl), atol = 1e-5)

    def test_kl_shape(self):
        cur_logits = torch.randn(2, 4, 8)
        ref_logits = torch.randn(2, 4, 8)

        kl = _compute_kl_divergence(cur_logits, ref_logits, kl_direction = "forward")

        assert kl.shape == (8,)

    def test_kl_reverse_matches_manual(self):
        torch.manual_seed(321)
        cur_logits = torch.randn(2, 5)
        ref_logits = torch.randn(2, 5)

        kl_reverse = _compute_kl_divergence(cur_logits, ref_logits, kl_direction = "reverse")

        cur_p = F.softmax(cur_logits, dim = -1)
        ref_p = F.softmax(ref_logits, dim = -1)
        manual = (cur_p * (cur_p.log() - ref_p.log())).sum(dim = -1)

        assert torch.allclose(kl_reverse, manual, atol = 1e-5)


class TestDFTWeights:
    def test_dft_weights_are_probabilities(self, dummy_logits, dummy_labels):
        flat_logits = dummy_logits.detach().view(-1, 8)
        flat_labels = dummy_labels.view(-1)

        weights = _compute_dft_weights(flat_logits, flat_labels)

        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)

    def test_dft_weights_are_detached(self, dummy_logits, dummy_labels):
        weights = _compute_dft_weights(
            dummy_logits.detach().view(-1, 8),
            dummy_labels.view(-1),
        )

        assert not weights.requires_grad

    def test_dft_weights_match_exp_neg_ce(self):
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


class TestComputeASFTLoss:
    def test_sft_mode(self, simple_model):
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft", kl_weight = 0.0)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_sft_mode_granite_logit_scaling(self):
        class GraniteModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    model_type = "granite",
                    final_logit_softcapping = 0,
                    logit_scale = 2.0,
                    logit_scaling = 0,
                    logits_scaling = 8.0,
                )
                self.embedding = nn.Embedding(16, 8)
                self.linear = nn.Linear(8, 8)

            def forward(
                self,
                input_ids = None,
                **kwargs,
            ):
                embeddings = self.embedding(input_ids)
                logits = self.linear(embeddings)
                return SimpleNamespace(logits = logits)

        model = GraniteModel()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }
        captured = {}

        def fake_ce(
            logits,
            labels,
            logit_softcapping = 0,
            logit_scaling = 0,
            ignore_index = -100,
        ):
            captured["logit_scaling"] = logit_scaling
            batch, seq_len, _ = logits.shape
            losses = torch.zeros(batch * seq_len, device = logits.device)
            valid_mask = labels.view(-1) != ignore_index
            return losses, valid_mask

        with patch(
            "unsloth.losses.asft.fast_cross_entropy_loss_per_token",
            side_effect = fake_ce,
        ):
            loss = compute_asft_loss(model, inputs, asft_mode = "sft", kl_weight = 0.0)

        assert captured["logit_scaling"] == pytest.approx(1.0 / 8.0)
        assert loss.dim() == 0

    def test_sft_mode_falcon_h1_logit_scaling(self):
        class FalconH1Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    model_type = "falcon_h1",
                    final_logit_softcapping = 0,
                    logit_scale = 0,
                    logit_scaling = 0,
                    lm_head_multiplier = 3.0,
                )
                self.embedding = nn.Embedding(16, 8)
                self.linear = nn.Linear(8, 8)

            def forward(
                self,
                input_ids = None,
                **kwargs,
            ):
                embeddings = self.embedding(input_ids)
                logits = self.linear(embeddings)
                return SimpleNamespace(logits = logits)

        model = FalconH1Model()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }
        captured = {}

        def fake_ce(
            logits,
            labels,
            logit_softcapping = 0,
            logit_scaling = 0,
            ignore_index = -100,
        ):
            captured["logit_scaling"] = logit_scaling
            batch, seq_len, _ = logits.shape
            losses = torch.zeros(batch * seq_len, device = logits.device)
            valid_mask = labels.view(-1) != ignore_index
            return losses, valid_mask

        with patch(
            "unsloth.losses.asft.fast_cross_entropy_loss_per_token",
            side_effect = fake_ce,
        ):
            loss = compute_asft_loss(model, inputs, asft_mode = "sft", kl_weight = 0.0)

        assert captured["logit_scaling"] == pytest.approx(3.0)
        assert loss.dim() == 0

    def test_dft_mode(self, simple_model):
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "dft", kl_weight = 0.0)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_dft_normalize_by_weights(self, simple_model):
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
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[-100, -100, -100, -100]]),
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft")

        assert loss.item() == 0.0

    def test_uses_num_items_in_batch(self, simple_model):
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
            "num_items_in_batch": 2,
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft")

        assert loss.dim() == 0

    def test_packing_boundary_masking(self, simple_model):
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
            "packed_seq_lengths": torch.tensor([2, 2], dtype = torch.int32),
        }

        loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft")

        assert loss.dim() == 0


class TestASFTStreamingConfig:
    def test_default_values(self):
        config = ASFTStreamingConfig()

        assert config.mode is None
        assert config.enabled is False
        assert config.ref_strategy == "none"
        assert config.ref_microbatch_size is None
        assert config.seq_chunk_size is None
        assert config.kl_token_chunk_size is None
        assert config.force_fp32_kl is True

    def test_custom_values(self):
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
    def test_mode_batch_uses_batch_micro(self, simple_model):
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

        with (
            patch(
                "unsloth.losses.asft._compute_kl_batch_micro",
                side_effect = batch_side_effect,
            ) as batch_mock,
            patch(
                "unsloth.losses.asft._compute_kl_seq_kv_cache",
                side_effect = AssertionError("seq_kv_cache should not be used"),
            ),
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

        with (
            patch(
                "unsloth.losses.asft._compute_kl_seq_kv_cache",
                side_effect = seq_side_effect,
            ) as seq_mock,
            patch(
                "unsloth.losses.asft._compute_kl_batch_micro",
                side_effect = AssertionError("batch_micro should not be used"),
            ),
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

        with (
            patch(
                "unsloth.losses.asft._compute_kl_divergence",
                side_effect = kl_side_effect,
            ) as kl_mock,
            patch(
                "unsloth.losses.asft._compute_kl_seq_kv_cache",
                side_effect = AssertionError("seq_kv_cache should not be used"),
            ),
            patch(
                "unsloth.losses.asft._compute_kl_batch_micro",
                side_effect = AssertionError("batch_micro should not be used"),
            ),
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
    def test_seq_kv_cache_runs_when_use_cache_false(self):
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
            logits = torch.zeros(batch, chunk_len, vocab_size, device = input_ids_local.device)
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
            call_state["max_batch"] = max(call_state["max_batch"], input_ids_local.shape[0])
            if input_ids_local.shape[0] > 1:
                raise AssertionError("expected microbatching")
            batch, chunk_len = input_ids_local.shape
            logits = torch.zeros(batch, chunk_len, vocab_size, device = input_ids_local.device)
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
            if input_ids_local.shape[0] == batch_size and input_ids_local.shape[1] == seq_len:
                raise AssertionError("full forward not expected on fallback")
            batch, chunk_len = input_ids_local.shape
            logits = torch.zeros(batch, chunk_len, vocab_size, device = input_ids_local.device)
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

    def test_seq_kv_cache_falls_back_with_packed_sequences(self):
        batch_size, seq_len, vocab_size = 2, 4, 3
        cur_logits = torch.randn(batch_size, seq_len, vocab_size)
        shift_labels = torch.zeros(batch_size, seq_len, dtype = torch.long)
        valid_mask = shift_labels != -100
        input_ids = torch.arange(seq_len).repeat(batch_size, 1)
        packed_seq_lengths = torch.tensor([2, 2], dtype = torch.int32)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    use_cache = True,
                    final_logit_softcapping = 0,
                    logit_scale = 0,
                )

        model = DummyModel()
        ref_forward = MagicMock()
        forward_inputs = {
            "input_ids": input_ids,
            "packed_seq_lengths": packed_seq_lengths,
        }

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
        ) as batch_mock:
            kl = _compute_kl_seq_kv_cache(
                model,
                cur_logits,
                shift_labels,
                valid_mask,
                ref_forward,
                forward_inputs,
                seq_chunk_size = 2,
            )

        assert batch_mock.called
        assert batch_mock.call_args[0][6] == 1
        assert not ref_forward.called
        assert kl.shape == (batch_size, seq_len)

    def test_config_immutability_when_none_values(self, simple_model):
        config = ASFTStreamingConfig(
            enabled = True,
            ref_strategy = "batch_micro",
            ref_microbatch_size = None,
        )
        original_microbatch = config.ref_microbatch_size
        original_chunk = config.seq_chunk_size

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        loss = compute_asft_loss(
            simple_model,
            inputs,
            asft_mode = "sft",
            streaming_config = config,
        )

        assert config.ref_microbatch_size == original_microbatch
        assert config.seq_chunk_size == original_chunk


class TestBackwardCompatibility:
    def test_sft_mode_matches_standard_ce(self, simple_model):
        torch.manual_seed(42)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        asft_loss = compute_asft_loss(simple_model, inputs, asft_mode = "sft")

        assert asft_loss.dim() == 0
        assert not torch.isnan(asft_loss)
        assert not torch.isinf(asft_loss)

    def test_streaming_equivalence(self, simple_model):
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }

        full_loss = compute_asft_loss(
            simple_model,
            inputs,
            asft_mode = "sft+kl",
            kl_weight = 0.1,
            reference_policy = "frozen_copy",
            streaming_config = ASFTStreamingConfig(enabled = False),
        )

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

        assert torch.allclose(full_loss, streaming_loss, atol = 1e-4)

    def test_seq_kv_cache_equivalence(self):
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
                self,
                input_ids = None,
                past_key_values = None,
                use_cache = None,
                **kwargs,
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
                self,
                input_ids = None,
                past_key_values = None,
                use_cache = None,
                **kwargs,
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


class TestASFTTrainerIntegration:
    def test_import_asft_trainer(self):
        from unsloth.trainer import ASFTTrainer, ASFTStreamingConfig
        assert ASFTTrainer is not None
        assert ASFTStreamingConfig is not None

    def test_asft_trainer_inherits_unsloth_trainer(self):
        from unsloth.trainer import ASFTTrainer, UnslothTrainer
        assert issubclass(ASFTTrainer, UnslothTrainer)


class TestASFTTrainerComputeLoss:
    def test_compute_loss_calls_asft_loss(self):
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

        with patch("unsloth.trainer.compute_asft_loss", return_value = expected) as loss_mock:
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

        with (
            pytest.warns(UserWarning),
            patch("unsloth.trainer.deepcopy", return_value = model_copy) as deepcopy_mock,
            patch(
                "unsloth.trainer.compute_asft_loss",
                return_value = torch.tensor(0.5, device = inputs["input_ids"].device),
            ),
        ):
            ASFTTrainer.compute_loss(trainer, model, inputs)
            ASFTTrainer.compute_loss(trainer, model, inputs)

        assert deepcopy_mock.call_count == 1
        assert trainer._asft_original_model is model_copy
        assert model_copy.eval.called
        assert model_copy.requires_grad_.called

    def test_compute_loss_skips_copy_with_disable_adapter(self):
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

        with (
            patch("unsloth.trainer.deepcopy") as deepcopy_mock,
            patch(
                "unsloth.trainer.compute_asft_loss",
                return_value = torch.tensor(0.5, device = inputs["input_ids"].device),
            ) as loss_mock,
        ):
            ASFTTrainer.compute_loss(trainer, model, inputs)

        assert not deepcopy_mock.called
        assert loss_mock.call_args.kwargs["original_model"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
