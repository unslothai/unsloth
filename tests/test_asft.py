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

if not torch.cuda.is_available():
    # Fast_CrossEntropyLoss is a Triton kernel.
    pytest.skip(reason = "ASFT CE runs the Triton kernel, which needs CUDA", allow_module_level = True)
torch.set_default_device("cuda")

from unsloth.losses.asft import (
    ASFTStreamingConfig,
    fast_cross_entropy_loss_per_token,
    build_shift_labels,
    get_reference_forward_callable,
    compute_asft_loss,
    _compute_kl_divergence,
)


@pytest.fixture
def dummy_logits():
    torch.manual_seed(42)
    return torch.randn(2, 4, 8, requires_grad = True)


@pytest.fixture
def dummy_labels():
    return torch.tensor([[0, 1, 2, 3], [4, 5, -100, -100]], dtype = torch.long)


class SimpleModel(nn.Module):
    def __init__(
        self,
        vocab = 16,
        softcap = 0.0,
        scale = 1.0,
    ):
        super().__init__()
        self.config = SimpleNamespace(final_logit_softcapping = softcap or None, logit_scale = scale)
        self.embedding = nn.Embedding(vocab, 8)
        self.linear = nn.Linear(8, vocab)
        self.softcap, self.scale = softcap, scale
        self.ref_calls = []

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        if torch.is_inference_mode_enabled():
            self.ref_calls.append(input_ids.shape[0])
        logits = self.linear(self.embedding(input_ids)) * self.scale
        # Like HF / Unsloth forwards, the returned logits are already transformed.
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return SimpleNamespace(logits = logits)


@pytest.fixture
def simple_model():
    torch.manual_seed(0)
    return SimpleModel()


def _batch(
    batch = 3,
    seq = 6,
    vocab = 16,
    seed = 1,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, seq), generator = g, device = "cuda")
    labels = ids.clone()
    labels[0, :2] = -100
    return {"input_ids": ids, "labels": labels}


def _reference_loss(
    model,
    ref_model,
    inputs,
    mode,
    kl_weight,
    kl_direction = "forward",
    normalize_by = "tokens",
):
    logits = model(input_ids = inputs["input_ids"]).logits.float()
    shift = build_shift_labels(inputs["labels"])
    valid = shift != -100
    ce = F.cross_entropy(logits.transpose(1, 2), shift.clamp_min(0), reduction = "none")
    w = torch.exp(-ce.detach())
    tok = ce * w if mode in ("dft", "asft") else ce
    if mode in ("sft+kl", "asft"):
        with torch.no_grad():
            ref = ref_model(input_ids = inputs["input_ids"]).logits.float()
        p, q = (ref, logits) if kl_direction == "forward" else (logits, ref)
        kl = (F.softmax(p, -1) * (F.log_softmax(p, -1) - F.log_softmax(q, -1))).sum(-1)
        tok = tok + kl_weight * kl
    norm = w[valid].sum() if normalize_by == "weights" else valid.sum()
    return tok[valid].sum() / norm


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
        ref_forward(input_ids = torch.tensor([[1, 2, 3, 4]]))
        assert simple_model.disable_adapter.__enter__.called
        assert simple_model.training

    def test_frozen_copy_policy(self, simple_model):
        ref_forward = get_reference_forward_callable(simple_model, reference_policy = "frozen_copy")
        with torch.no_grad():
            simple_model.linear.weight.add_(1.0)
        result = ref_forward(input_ids = torch.tensor([[1, 2, 3, 4]]))
        assert result.shape[:2] == (1, 4)
        assert not torch.allclose(
            result, simple_model(input_ids = torch.tensor([[1, 2, 3, 4]])).logits
        )

    def test_fallback_to_frozen_copy_without_adapters(self, simple_model):
        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "disable_adapter"
        )
        assert ref_forward(input_ids = torch.tensor([[1, 2, 3, 4]])) is not None

    def test_return_outputs_true(self, simple_model):
        ref_forward = get_reference_forward_callable(
            simple_model, reference_policy = "frozen_copy", return_outputs = True
        )
        assert hasattr(ref_forward(input_ids = torch.tensor([[1, 2, 3, 4]])), "logits")

    def test_unknown_policy_raises(self, simple_model):
        with pytest.raises(ValueError):
            get_reference_forward_callable(simple_model, reference_policy = "nope")


class TestKLDivergence:
    def test_forward_matches_manual(self):
        torch.manual_seed(42)
        cur, ref = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        kl = _compute_kl_divergence(cur, ref, kl_direction = "forward")
        ref_p = F.softmax(ref, -1)
        manual = (ref_p * (ref_p.log() - F.log_softmax(cur, -1))).sum(-1)
        assert kl.shape == (2, 4)
        assert torch.allclose(kl, manual, atol = 1e-5)

    def test_reverse_matches_manual(self):
        torch.manual_seed(321)
        cur, ref = torch.randn(2, 5), torch.randn(2, 5)
        kl = _compute_kl_divergence(cur, ref, kl_direction = "reverse")
        cur_p = F.softmax(cur, -1)
        manual = (cur_p * (cur_p.log() - F.log_softmax(ref, -1))).sum(-1)
        assert torch.allclose(kl, manual, atol = 1e-5)

    def test_zero_for_identical(self):
        logits = torch.randn(4, 8)
        assert torch.allclose(
            _compute_kl_divergence(logits, logits.clone()), torch.zeros(4), atol = 1e-5
        )

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError):
            _compute_kl_divergence(torch.randn(2, 4), torch.randn(2, 4), kl_direction = "sideways")


class TestComputeASFTLoss:
    @pytest.mark.parametrize("mode", ["sft", "dft", "sft+kl", "asft"])
    @pytest.mark.parametrize("kl_direction", ["forward", "reverse"])
    def test_modes_match_reference_formula(self, mode, kl_direction):
        torch.manual_seed(0)
        model = SimpleModel()
        ref_model = SimpleModel()
        ref_model.load_state_dict(model.state_dict())
        with torch.no_grad():
            model.linear.weight.add_(0.3 * torch.randn_like(model.linear.weight))
        inputs = _batch()
        loss = compute_asft_loss(
            model,
            dict(inputs),
            asft_mode = mode,
            kl_weight = 0.5,
            kl_direction = kl_direction,
            reference_policy = "frozen_copy",
            original_model = ref_model,
        )
        expected = _reference_loss(model, ref_model, inputs, mode, 0.5, kl_direction)
        assert loss.requires_grad
        assert torch.allclose(loss, expected, atol = 1e-4)

    @pytest.mark.parametrize("softcap,scale", [(3.0, 1.0), (0.0, 0.25), (3.0, 4.0)])
    def test_logit_transforms_not_applied_twice(self, softcap, scale):
        # Forwards return softcapped / scaled logits; CE must be taken on them as-is.
        torch.manual_seed(0)
        model = SimpleModel(softcap = softcap, scale = scale)
        with torch.no_grad():
            model.linear.weight.mul_(20.0)
        inputs = _batch()
        loss = compute_asft_loss(model, dict(inputs), asft_mode = "sft")
        expected = _reference_loss(model, None, inputs, "sft", 0.0)
        assert torch.allclose(loss, expected, atol = 1e-4)

    def test_dft_normalize_by_weights(self, simple_model):
        inputs = _batch()
        loss = compute_asft_loss(
            simple_model, dict(inputs), asft_mode = "dft", normalize_by = "weights"
        )
        expected = _reference_loss(simple_model, None, inputs, "dft", 0.0, normalize_by = "weights")
        assert torch.allclose(loss, expected, atol = 1e-5)

    def test_normalize_by_weights_rejected_without_dft(self, simple_model):
        with pytest.raises(ValueError):
            compute_asft_loss(simple_model, _batch(), asft_mode = "sft", normalize_by = "weights")

    @pytest.mark.parametrize("mode", ["sft+kl", "asft"])
    def test_zero_kl_weight_skips_reference(self, mode, simple_model):
        with patch("unsloth.losses.asft.get_reference_forward_callable") as ref_mock:
            loss = compute_asft_loss(simple_model, _batch(), asft_mode = mode, kl_weight = 0.0)
        assert not ref_mock.called
        base = "sft" if mode == "sft+kl" else "dft"
        assert torch.allclose(loss, compute_asft_loss(simple_model, _batch(), asft_mode = base))

    def test_ddp_wrapper_unwrapped_for_reference(self, simple_model):
        wrapper = nn.DataParallel(simple_model)
        with patch(
            "unsloth.losses.asft.get_reference_forward_callable",
            wraps = get_reference_forward_callable,
        ) as ref_mock:
            compute_asft_loss(wrapper, _batch(), asft_mode = "sft+kl", kl_weight = 0.1)
        assert ref_mock.call_args.args[0] is simple_model

    def test_return_outputs(self, simple_model):
        loss, outputs = compute_asft_loss(
            simple_model, _batch(), asft_mode = "sft", return_outputs = True
        )
        assert loss.dim() == 0
        assert hasattr(outputs, "logits")

    def test_handles_all_ignored_labels(self, simple_model):
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.full((1, 4), -100)}
        assert compute_asft_loss(simple_model, inputs, asft_mode = "sft").item() == 0.0

    def test_uses_num_items_in_batch(self, simple_model):
        inputs = _batch()
        mean = compute_asft_loss(simple_model, dict(inputs), asft_mode = "sft")
        n_valid = (build_shift_labels(inputs["labels"]) != -100).sum()
        summed = compute_asft_loss(
            simple_model, dict(inputs, num_items_in_batch = 2 * n_valid), asft_mode = "sft"
        )
        assert torch.allclose(summed * 2, mean, atol = 1e-5)

    def test_packing_boundary_masking(self, simple_model):
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[1, 2, 3, 4]])}
        packed = dict(inputs, packed_seq_lengths = torch.tensor([2, 2], dtype = torch.int32))
        logits = simple_model(**inputs).logits
        ce = F.cross_entropy(logits[0, [0, 2]], torch.tensor([2, 4]))
        assert torch.allclose(
            compute_asft_loss(simple_model, packed, asft_mode = "sft"), ce, atol = 1e-5
        )

    def test_invalid_mode_raises(self, simple_model):
        with pytest.raises(ValueError):
            compute_asft_loss(simple_model, _batch(), asft_mode = "nope")


class TestStreaming:
    def test_default_config(self):
        config = ASFTStreamingConfig()
        assert config.mode == "off"
        assert config.ref_microbatch_size is None
        assert config.force_fp32_kl is True

    @pytest.mark.parametrize(
        "microbatch,expected_calls", [(None, [1, 1, 1]), (1, [1, 1, 1]), (2, [2, 1]), (8, [3])]
    )
    def test_batch_mode_matches_full(self, microbatch, expected_calls):
        torch.manual_seed(0)
        model = SimpleModel()
        with torch.no_grad():
            model.linear.weight.add_(0.3)
        ref_model = SimpleModel()
        inputs = _batch()
        full = compute_asft_loss(
            model, dict(inputs), asft_mode = "asft", kl_weight = 0.2, original_model = ref_model
        )
        assert ref_model.ref_calls == [3]
        ref_model.ref_calls.clear()
        streamed = compute_asft_loss(
            model,
            dict(inputs),
            asft_mode = "asft",
            kl_weight = 0.2,
            original_model = ref_model,
            streaming_config = ASFTStreamingConfig(mode = "batch", ref_microbatch_size = microbatch),
        )
        # batch 3 // 2 = 1 row per microbatch by default.
        assert ref_model.ref_calls == expected_calls
        assert torch.allclose(full, streamed, atol = 1e-5)

    @pytest.mark.parametrize("mode", ["auto", "seq", "hybrid", "bogus"])
    def test_removed_modes_rejected(self, mode, simple_model):
        with pytest.raises(ValueError):
            compute_asft_loss(
                simple_model,
                _batch(),
                asft_mode = "sft",
                streaming_config = ASFTStreamingConfig(mode = mode),
            )


def _stub_trainer_runtime(trainer, num_processes = 1):
    trainer.accelerator = SimpleNamespace(unwrap_model = lambda m: m, num_processes = num_processes)
    trainer.args = SimpleNamespace(average_tokens_across_devices = True)


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
        _stub_trainer_runtime(trainer)

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
        _stub_trainer_runtime(trainer)

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
        _stub_trainer_runtime(trainer)

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

    @pytest.mark.parametrize("mode", ["sft+kl", "asft"])
    def test_compute_loss_skips_frozen_copy_when_kl_disabled(self, mode):
        from unsloth.trainer import ASFTTrainer

        trainer = ASFTTrainer.__new__(ASFTTrainer)
        trainer.asft_enabled, trainer.asft_mode, trainer.kl_weight = True, mode, 0.0
        trainer.kl_direction, trainer.reference_policy = "forward", "frozen_copy"
        trainer.asft_streaming, trainer.normalize_by = ASFTStreamingConfig(), "tokens"
        trainer._asft_original_model = None
        _stub_trainer_runtime(trainer)
        with (
            patch("unsloth.trainer.deepcopy") as deepcopy_mock,
            patch("unsloth.trainer.compute_asft_loss", return_value = torch.tensor(0.5)),
        ):
            ASFTTrainer.compute_loss(trainer, nn.Module(), {"input_ids": torch.tensor([[1]])})
        assert not deepcopy_mock.called

    @pytest.mark.parametrize(
        "num_processes,num_items,expected", [(1, 5, 0.5), (4, 5, 2.0), (4, None, 0.5)]
    )
    def test_compute_loss_rescales_for_token_average_across_ranks(
        self, num_processes, num_items, expected
    ):
        from unsloth.trainer import ASFTTrainer

        trainer = ASFTTrainer.__new__(ASFTTrainer)
        trainer.asft_enabled, trainer.asft_mode, trainer.kl_weight = True, "sft", 0.0
        trainer.kl_direction, trainer.reference_policy = "forward", "disable_adapter"
        trainer.asft_streaming, trainer.normalize_by = ASFTStreamingConfig(), "tokens"
        trainer._asft_original_model = None
        _stub_trainer_runtime(trainer, num_processes)
        with patch("unsloth.trainer.compute_asft_loss", return_value = torch.tensor(0.5)):
            loss = ASFTTrainer.compute_loss(
                trainer,
                nn.Module(),
                {"input_ids": torch.tensor([[1]])},
                num_items_in_batch = num_items,
            )
        assert loss.item() == pytest.approx(expected)
