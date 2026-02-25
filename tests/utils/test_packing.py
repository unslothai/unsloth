# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from unsloth import FastLanguageModel
from unsloth.utils import attention_dispatch as attention_dispatch_utils
from unsloth.utils.packing import (
    configure_padding_free,
    configure_sample_packing,
    enable_padding_free_metadata,
    enable_sample_packing,
    mask_packed_sequence_boundaries,
)

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def _build_packed_training_setup(tmp_path, device):
    dtype = None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM",
            max_seq_length = 64,
            load_in_4bit = False,
            dtype = dtype,
        )
    except OSError as exc:  # pragma: no cover - offline CI
        pytest.skip(f"Requires access to tiny llama checkpoint: {exc}")

    model.to(device)

    dataset = Dataset.from_dict(
        {
            "text": [
                "Hello world!",
                "Short sample.",
                "This is a slightly longer packed example to test batching.",
                "Another response to include in the batch.",
            ]
        }
    )

    training_args = SFTConfig(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 1,
        dataset_text_field = "text",
        max_length = 64,
        logging_steps = 1,
        max_steps = 1,
        fp16 = device.type == "cuda" and not torch.cuda.is_bf16_supported(),
        bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(),
        dataset_num_proc = 1,
        output_dir = str(tmp_path),
        packing = True,
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = training_args,
    )

    enable_sample_packing(model, trainer)

    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))

    model_device = next(model.parameters()).device

    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(model_device)

    from unsloth.models import llama as llama_mod

    return model, batch, trainer, llama_mod


def _trim_batch_to_total_tokens(data, total_tokens):
    def _trim_tensor(t: torch.Tensor):
        if t.ndim >= 2 and t.size(1) > total_tokens:
            return t[:, :total_tokens].contiguous()
        return t

    trimmed = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            trimmed[key] = _trim_tensor(value)
        else:
            trimmed[key] = value
    return trimmed


def test_mask_packed_sequence_boundaries_marks_single_row():
    shift_labels = torch.arange(6, dtype = torch.long).view(1, 6)
    changed = mask_packed_sequence_boundaries(
        shift_labels,
        torch.tensor([2, 1, 3], dtype = torch.int32),
    )
    assert changed is True
    flat = shift_labels.view(-1)
    assert flat[1].item() == -100
    assert flat[2].item() == -100
    assert flat[5].item() == -100
    assert flat[0].item() != -100


def test_mask_packed_sequence_boundaries_across_multiple_rows():
    shift_labels = torch.arange(10, dtype = torch.long).view(2, 5)
    lengths = torch.tensor([3, 2, 4, 1], dtype = torch.int32)
    changed = mask_packed_sequence_boundaries(shift_labels, lengths)
    assert changed is True
    flat = shift_labels.view(-1)
    for idx in (2, 4, 8, 9):
        assert flat[idx].item() == -100
    assert torch.any(flat != -100)


def test_configure_sample_packing():
    config = SimpleNamespace()
    configure_sample_packing(config)

    assert config.packing is True
    assert config.padding_free is True
    assert config.remove_unused_columns is False


def test_configure_padding_free():
    config = SimpleNamespace(remove_unused_columns = True)
    configure_padding_free(config)

    assert config.padding_free is True
    assert config.remove_unused_columns is False


class _DummyChild(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_seq_length = 8


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_seq_length = 16
        self.child = _DummyChild()
        self.config = SimpleNamespace(_attn_implementation = "sdpa")
        self.generation_config = SimpleNamespace(attn_implementation = "sdpa")


class _DummyTrainer:
    def __init__(self):
        self.args = SimpleNamespace(remove_unused_columns = True)
        self.data_collator = DataCollatorForLanguageModeling(
            pad_token_id = 0,
            completion_only_loss = False,
            padding_free = True,
            return_position_ids = False,
            return_tensors = "pt",
        )


class _PaddingFreeCollator:
    def __init__(self):
        self.padding_free = True
        self.return_position_ids = False
        self.calls = 0

    def torch_call(self, examples):
        self.calls += 1
        return {
            "input_ids": torch.tensor([[0]], dtype = torch.long),
            "examples_seen": self.calls,
        }


def test_enable_sample_packing():
    model = _DummyModel()
    trainer = _DummyTrainer()

    enable_sample_packing(model, trainer)

    # model hierarchy should now allow packed overlength inputs
    assert getattr(model, "_unsloth_allow_packed_overlength") is True
    assert getattr(model.child, "_unsloth_allow_packed_overlength") is True

    collator = trainer.data_collator
    assert collator.return_position_ids is True
    assert getattr(collator, "_unsloth_packing_wrapped") is True

    examples = [
        {
            "input_ids": [0, 1, 2],
            "labels": [0, 1, 2],
            "seq_lengths": [2, 1],
        },
        {
            "input_ids": [3, 4, 5],
            "labels": [3, 4, 5],
            "seq_lengths": [3],
        },
    ]
    batch = collator.torch_call(examples)

    # packed lengths are aggregated into a single tensor
    assert "packed_seq_lengths" in batch
    assert torch.equal(
        batch["packed_seq_lengths"],
        torch.tensor([2, 1, 3], dtype = torch.int32),
    )

    assert batch["input_ids"].shape == (1, 6)
    expected_positions = torch.tensor([0, 1, 0, 0, 1, 2], dtype = torch.long)
    assert torch.equal(batch["position_ids"].view(-1)[:6], expected_positions)


def test_enable_sample_packing_trl_collator(tmp_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, _, trainer, _ = _build_packed_training_setup(tmp_path, device)

    enable_sample_packing(model, trainer)

    examples = [
        {
            "input_ids": [0, 1, 2],
            "labels": [0, 1, 2],
            "seq_lengths": [2, 1],
        },
        {
            "input_ids": [3, 4, 5],
            "labels": [3, 4, 5],
            "seq_lengths": [3],
        },
    ]

    batch = trainer.data_collator.torch_call(examples)

    assert batch["input_ids"].shape == (1, 6)
    assert torch.equal(
        batch["packed_seq_lengths"],
        torch.tensor([2, 1, 3], dtype = torch.int32),
    )

    expected_positions = torch.tensor([0, 1, 0, 0, 1, 2], dtype = torch.long)
    assert torch.equal(batch["position_ids"].view(-1)[:6], expected_positions)

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.free_memory()


def test_enable_padding_free_metadata():
    model = _DummyModel()
    trainer = SimpleNamespace(
        args = SimpleNamespace(remove_unused_columns = True),
        data_collator = _PaddingFreeCollator(),
    )

    enable_padding_free_metadata(model, trainer)

    assert getattr(model, "_unsloth_allow_packed_overlength") is True
    assert getattr(model.child, "_unsloth_allow_packed_overlength") is True

    collator = trainer.data_collator
    assert collator.return_position_ids is True
    assert getattr(collator, "_unsloth_padding_free_lengths_wrapped") is True

    examples = [
        {"input_ids": [0, 1, 2]},
        {"input_ids": [3, 4]},
    ]
    batch = collator.torch_call(examples)
    assert torch.equal(
        batch["packed_seq_lengths"],
        torch.tensor([3, 2], dtype = torch.int32),
    )
    assert trainer.args.remove_unused_columns is False


def test_packing_sdpa(tmp_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, batch, trainer, llama_mod = _build_packed_training_setup(tmp_path, device)

    assert "packed_seq_lengths" in batch
    assert "attention_mask" not in batch
    assert batch["packed_seq_lengths"].dtype == torch.int32

    total_tokens = batch["input_ids"].size(-1)
    assert int(batch["packed_seq_lengths"].sum().item()) == total_tokens

    packed_tokens = int(batch["packed_seq_lengths"].sum().item())
    assert "position_ids" in batch
    flat_positions = batch["position_ids"].reshape(-1)[:packed_tokens]
    expected_positions = torch.cat(
        [
            torch.arange(length, dtype = torch.long)
            for length in batch["packed_seq_lengths"].tolist()
        ]
    )
    assert torch.equal(flat_positions.cpu(), expected_positions)
    inputs = _trim_batch_to_total_tokens(batch, packed_tokens)

    seq_info = llama_mod.get_packed_info_from_kwargs(
        {"packed_seq_lengths": batch["packed_seq_lengths"]},
        inputs["input_ids"].device,
    )
    assert seq_info is not None

    original_mask = attention_dispatch_utils.build_sdpa_packed_attention_mask
    mask_calls = []
    captured_loss_labels = {}

    def _capture_mask(seq_info, dtype, device, *, sliding_window = None):
        mask_calls.append(tuple(seq_info[0].tolist()))
        return original_mask(
            seq_info,
            dtype = dtype,
            device = device,
            sliding_window = sliding_window,
        )

    def _capture_loss(*, logits, labels, **loss_kwargs):
        captured_loss_labels["labels"] = labels.detach().to("cpu")
        return torch.zeros((), device = logits.device, dtype = logits.dtype)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(attention_dispatch_utils, "HAS_FLASH_ATTENTION", False)
        )
        stack.enter_context(
            patch.object(attention_dispatch_utils, "HAS_XFORMERS", False)
        )
        stack.enter_context(
            patch.object(
                attention_dispatch_utils,
                "build_sdpa_packed_attention_mask",
                side_effect = _capture_mask,
            )
        )
        stack.enter_context(
            patch.object(
                llama_mod,
                "fast_cross_entropy_loss",
                side_effect = _capture_loss,
            )
        )
        with torch.no_grad():
            outputs = model(**inputs)

    assert mask_calls, "SDPA packed mask was not constructed"
    assert outputs.loss is not None
    assert "labels" in captured_loss_labels
    flat_loss_labels = captured_loss_labels["labels"].reshape(-1)
    boundaries = (
        torch.cumsum(
            batch["packed_seq_lengths"].to(device = "cpu", dtype = torch.long), dim = 0
        )
        - 1
    )
    for idx in boundaries.tolist():
        assert flat_loss_labels[idx].item() == -100
    assert torch.any(flat_loss_labels != -100)

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.free_memory()
