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
import unsloth.trainer as trainer_module
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
from datasets import Dataset, IterableDataset
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


def _patch_fake_sft_trainer():
    class FakeSFTTrainer:
        def __init__(self, *args, **kwargs):
            self.model = args[0] if len(args) >= 1 else kwargs["model"]
            self.args = args[1] if len(args) >= 2 else kwargs["args"]
            self.data_collator = args[2] if len(args) >= 3 else kwargs.get("data_collator")

    trainer_module._patch_sft_trainer_auto_packing(SimpleNamespace(SFTTrainer = FakeSFTTrainer))
    return FakeSFTTrainer


def _vlm_model():
    return SimpleNamespace(
        config = SimpleNamespace(
            architectures = ["Gemma4ForConditionalGeneration"],
            model_type = "gemma4",
            vision_config = SimpleNamespace(),
        ),
        max_seq_length = 16,
    )


def _text_model():
    return SimpleNamespace(
        config = SimpleNamespace(
            architectures = ["LlamaForCausalLM"],
            model_type = "llama",
        ),
        max_seq_length = 16,
    )


class _CharacterTokenizer:
    bos_token = None
    eos_token = None
    chat_template = None

    def __call__(self, texts, **kwargs):
        is_batched = isinstance(texts, list)
        if not is_batched:
            texts = [texts]
        input_ids = [[ord(char) for char in text] for text in texts]
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            input_ids = [ids[: kwargs["max_length"]] for ids in input_ids]
        return {"input_ids": input_ids if is_batched else input_ids[0]}


def test_vlm_text_dataset_allows_explicit_packing():
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    trainer = fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        train_dataset = Dataset.from_dict({"text": ["text-only CPT sample"]}),
    )

    assert config.packing is True
    assert config.padding_free is True
    assert trainer.model._unsloth_allow_packed_overlength is True


def test_vlm_without_processing_class_still_disables_packing():
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    fake_trainer(
        _vlm_model(),
        config,
        None,
        Dataset.from_dict({"text": ["text-only sample"]}),
    )

    assert config.packing is False
    assert config.padding_free is False


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    (
        ("t5", "T5ForConditionalGeneration"),
        ("bart", "BartForConditionalGeneration"),
        ("whisper", "WhisperForConditionalGeneration"),
        ("csm", "CsmForConditionalGeneration"),
    ),
)
def test_nonvision_conditional_generation_keeps_packing(model_type, architecture):
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    model = SimpleNamespace(
        config = SimpleNamespace(model_type = model_type, architectures = [architecture]),
        max_seq_length = 16,
    )

    trainer = fake_trainer(
        model,
        config,
        None,
        Dataset.from_dict({"text": ["text-only sample"]}),
    )

    assert config.packing is True
    assert config.padding_free is True
    assert trainer.model._unsloth_allow_packed_overlength is True


def test_vlm_vision_dataset_still_disables_packing():
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    fake_trainer(
        _vlm_model(),
        config,
        None,
        Dataset.from_dict({"images": [None], "text": ["multimodal sample"]}),
        None,
        object(),
    )

    assert config.packing is False
    assert config.padding_free is False


@pytest.mark.parametrize(
    "vision_column",
    ("pixel_values", "pixel_attention_mask", "image_grid_thw"),
)
def test_vlm_preprocessed_vision_dataset_disables_packing(vision_column):
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        train_dataset = Dataset.from_dict({"input_ids": [[1]], vision_column: [None]}),
    )

    assert config.packing is False
    assert config.padding_free is False


@pytest.mark.parametrize("dict_eval", (False, True))
def test_vlm_vision_eval_dataset_disables_packing(dict_eval):
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    eval_dataset = Dataset.from_dict({"input_ids": [[1]], "pixel_values": [None]})
    if dict_eval:
        eval_dataset = {"vision": eval_dataset}

    fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        train_dataset = Dataset.from_dict({"text": ["text-only training sample"]}),
        eval_dataset = eval_dataset,
    )

    assert config.packing is False
    assert config.padding_free is False


def test_vlm_streaming_vision_dataset_without_metadata_disables_packing():
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    dataset = IterableDataset.from_generator(
        lambda: iter([{"images": [None], "text": "multimodal sample"}])
    )
    assert dataset.column_names is None

    fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        train_dataset = dataset,
    )

    assert config.packing is False
    assert config.padding_free is False
    assert next(iter(dataset))["text"] == "multimodal sample"


@pytest.mark.parametrize("data_collator", (None, object()))
def test_stateful_stream_is_not_consumed_during_detection(data_collator):
    class StatefulDataset:
        def __init__(self):
            self.rows = iter([{"text": "first"}, {"text": "second"}])

        def __iter__(self):
            return (row for row in self.rows)

    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    dataset = StatefulDataset()

    fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        data_collator = data_collator,
        train_dataset = dataset,
    )

    assert config.packing is False
    assert config.padding_free is False
    assert next(iter(dataset))["text"] == "first"


def test_text_model_stream_without_metadata_keeps_packing():
    class StatefulDataset:
        def __init__(self):
            self.rows = iter([{"text": "first"}, {"text": "second"}])

        def __iter__(self):
            return (row for row in self.rows)

    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    dataset = StatefulDataset()

    trainer = fake_trainer(
        model = _text_model(),
        args = config,
        processing_class = object(),
        train_dataset = dataset,
    )

    assert config.packing is True
    assert config.padding_free is True
    assert trainer.model._unsloth_allow_packed_overlength is True
    assert next(iter(dataset))["text"] == "first"


def test_bfd_packing_truncates_before_packing(monkeypatch):
    args = SimpleNamespace(
        dataset_num_proc = 1,
        dataset_text_field = "text",
        max_length = 4,
        packing_strategy = "bfd",
    )
    trainer = SimpleNamespace(model = None)
    dataset = Dataset.from_dict({"prompt": ["abc"], "completion": ["defghij"]})
    prepare_globals = SFTTrainer._prepare_dataset.__globals__

    def passthrough_pack_dataset(dataset, seq_length, strategy, map_kwargs):
        return dataset

    monkeypatch.setitem(prepare_globals, "pack_dataset", passthrough_pack_dataset)
    packed = SFTTrainer._prepare_dataset(
        trainer,
        dataset,
        _CharacterTokenizer(),
        args,
        True,
        None,
        "train",
    )

    assert len(packed["input_ids"][0]) == args.max_length


def test_wrapped_strategy_without_packing_still_truncates():
    args = SimpleNamespace(
        dataset_num_proc = 1,
        dataset_text_field = "text",
        max_length = 4,
        packing_strategy = "wrapped",
    )
    trainer = SimpleNamespace(model = None)
    dataset = Dataset.from_dict({"text": ["abcdefghi"]})

    prepared = SFTTrainer._prepare_dataset(
        trainer,
        dataset,
        _CharacterTokenizer(),
        args,
        False,
        None,
        "train",
    )

    assert len(prepared["input_ids"][0]) == args.max_length


@pytest.mark.parametrize("legacy_api", (False, True))
def test_wrapped_packing_preserves_overlength_tokens(monkeypatch, legacy_api):
    args_kwargs = {
        "dataset_num_proc": 1,
        "dataset_text_field": "text",
        "max_length": 4,
    }
    if not legacy_api:
        args_kwargs["packing_strategy"] = "wrapped"
    args = SimpleNamespace(**args_kwargs)
    trainer = SimpleNamespace(model = None)
    dataset = Dataset.from_dict({"text": ["abcdefghi"]})
    prepare_globals = SFTTrainer._prepare_dataset.__globals__
    pack_dataset = prepare_globals["pack_dataset"]

    def legacy_pack_dataset(
        dataset,
        seq_length,
        map_kwargs = None,
    ):
        return pack_dataset(dataset, seq_length, "wrapped", map_kwargs)

    if legacy_api:
        monkeypatch.setitem(prepare_globals, "pack_dataset", legacy_pack_dataset)

    packed = SFTTrainer._prepare_dataset(
        trainer,
        dataset,
        _CharacterTokenizer(),
        args,
        True,
        None,
        "train",
    )

    packed_ids = packed["input_ids"]
    assert sum(len(input_ids) for input_ids in packed_ids) == 9
    assert all(len(input_ids) <= args.max_length for input_ids in packed_ids)


# Named to match the unsloth_zoo helper: sft_trainer_prepare_dataset sources it by
# name and renames "def sft_prepare_dataset" -> "def _prepare_dataset". This fixture
# deliberately omits the "All Unsloth Zoo code licensed under LGPLv3" header to emulate
# a newer, compatible Zoo whose header moved (the dependency is only lower-bounded).
def sft_prepare_dataset(
    self, dataset, processing_class, args, packing, formatting_func, dataset_text_field
):
    do_truncation = True
    # Mirror the Zoo call so the "truncation = do_truncation," injection anchor
    # survives formatting (a bare tuple assignment gets rewritten to a paren form).
    dataset = processing_class(
        dataset,
        truncation = do_truncation,
    )
    return dataset


def test_wrapped_packing_setup_survives_missing_zoo_header(monkeypatch):
    # Regression: the wrapped-packing setup used to anchor on the Zoo license comment,
    # so a header change made it a no-op while the truncation reference still landed,
    # NameError-ing every SFT dataset preparation. It must now install via the
    # signature and always precede the reference.
    import ast
    import textwrap
    import unsloth.models.rl_replacements as rlr

    monkeypatch.setitem(rlr.RL_REPLACEMENTS, "sft_prepare_dataset", sft_prepare_dataset)

    source = (
        "def _prepare_dataset(self, dataset, processing_class, args, packing, "
        "formatting_func, dataset_text_field):\n    return dataset\n"
    )
    patched = rlr.sft_trainer_prepare_dataset("_prepare_dataset", source)

    assert "_unsloth_wrapped_packing = packing" in patched
    assert "import inspect as _inspect" in patched
    assert "not _unsloth_wrapped_packing" in patched
    assert patched.index("_unsloth_wrapped_packing = packing") < patched.index(
        "truncation = do_truncation and not _unsloth_wrapped_packing"
    )
    ast.parse(textwrap.dedent(patched))


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
        collator_args = {
            "pad_token_id": 0,
            "completion_only_loss": False,
            "return_tensors": "pt",
        }
        optional_flags = [
            {"padding_free": True, "return_position_ids": False},
            {"padding_free": True},
            {},
        ]
        for extra in optional_flags:
            try:
                self.data_collator = DataCollatorForLanguageModeling(**collator_args, **extra)
                break
            except TypeError:
                continue
        # Ensure attributes exist even if the constructor rejected the flags.
        if not hasattr(self.data_collator, "padding_free"):
            self.data_collator.padding_free = True
        if not hasattr(self.data_collator, "return_position_ids"):
            self.data_collator.return_position_ids = False


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

    # model hierarchy now allows packed overlength inputs
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

    # packed lengths aggregated into one tensor
    assert "packed_seq_lengths" in batch
    assert torch.equal(batch["packed_seq_lengths"], torch.tensor([2, 1, 3], dtype = torch.int32))

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
    assert torch.equal(batch["packed_seq_lengths"], torch.tensor([2, 1, 3], dtype = torch.int32))

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
    assert torch.equal(batch["packed_seq_lengths"], torch.tensor([3, 2], dtype = torch.int32))
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
        [torch.arange(length, dtype = torch.long) for length in batch["packed_seq_lengths"].tolist()]
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

    def _capture_mask(
        seq_info,
        dtype,
        device,
        *,
        sliding_window = None,
    ):
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
        stack.enter_context(patch.object(attention_dispatch_utils, "HAS_FLASH_ATTENTION", False))
        stack.enter_context(patch.object(attention_dispatch_utils, "HAS_XFORMERS", False))
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
        torch.cumsum(batch["packed_seq_lengths"].to(device = "cpu", dtype = torch.long), dim = 0) - 1
    )
    for idx in boundaries.tolist():
        assert flat_loss_labels[idx].item() == -100
    assert torch.any(flat_loss_labels != -100)

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.free_memory()
