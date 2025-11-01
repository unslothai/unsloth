from unsloth import FastLanguageModel
from unsloth.utils import attention_dispatch as attention_dispatch_utils
from unsloth.utils.packing import configure_sample_packing, enable_sample_packing

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def _build_packed_training_setup(tmp_path, device):
    dtype = None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            max_seq_length=64,
            load_in_4bit=False,
            dtype=dtype,
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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        dataset_text_field="text",
        max_length=64,
        logging_steps=1,
        max_steps=1,
        fp16=device.type == "cuda" and not torch.cuda.is_bf16_supported(),
        bf16=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        dataset_num_proc=1,
        output_dir=str(tmp_path),
    )
    configure_sample_packing(training_args)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
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


def test_configure_sample_packing():
    config = SimpleNamespace()
    configure_sample_packing(config)

    assert config.packing is True
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
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.generation_config = SimpleNamespace(attn_implementation="sdpa")


class _DummyCollator:
    def __init__(self):
        self.padding_free = False
        self.return_position_ids = False

    def torch_call(self, examples):
        batch_size = len(examples)
        max_tokens = 4
        return {
            "input_ids": torch.zeros(batch_size, max_tokens, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, max_tokens, dtype=torch.long),
            "batch": examples,
        }


class _DummyTrainer:
    def __init__(self):
        self.args = SimpleNamespace(remove_unused_columns=True)
        self.data_collator = _DummyCollator()


def test_enable_sample_packing():
    model = _DummyModel()
    trainer = _DummyTrainer()

    enable_sample_packing(model, trainer)

    # model hierarchy should now allow packed overlength inputs
    assert getattr(model, "_unsloth_allow_packed_overlength") is True
    assert getattr(model.child, "_unsloth_allow_packed_overlength") is True

    # trainer args are updated to keep the packed metadata
    assert trainer.args.remove_unused_columns is False

    collator = trainer.data_collator
    assert collator.padding_free is True
    assert collator.return_position_ids is True
    assert getattr(collator, "_unsloth_packing_wrapped") is True

    examples = [
        {"seq_lengths": [2, 1]},
        {"seq_lengths": [3]},
    ]
    batch = collator.torch_call(examples)

    # packed lengths are aggregated into a single tensor
    assert "packed_seq_lengths" in batch
    assert torch.equal(
        batch["packed_seq_lengths"],
        torch.tensor([2, 1, 3], dtype=torch.int32),
    )

    assert "position_ids" in batch
    assert torch.equal(batch["position_ids"][0, :3], torch.tensor([0, 1, 0], dtype=torch.long))
    assert torch.equal(batch["position_ids"][1, :3], torch.tensor([0, 1, 2], dtype=torch.long))

    # attention_mask is dropped when return_position_ids is set
    assert "attention_mask" not in batch
    assert batch["batch"] == examples


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
        [torch.arange(length, dtype=torch.long) for length in batch["packed_seq_lengths"].tolist()]
    )
    assert torch.equal(flat_positions.cpu(), expected_positions)
    inputs = _trim_batch_to_total_tokens(batch, packed_tokens)

    seq_info = llama_mod.get_packed_info_from_kwargs(
        {"packed_seq_lengths": batch["packed_seq_lengths"]},
        inputs["input_ids"].shape[0] * inputs["input_ids"].shape[1],
        inputs["input_ids"].device,
    )
    assert seq_info is not None

    original_mask = attention_dispatch_utils.build_sdpa_packed_attention_mask
    mask_calls = []

    def _capture_mask(seq_info, dtype, device):
        mask_calls.append(tuple(seq_info[0].tolist()))
        return original_mask(seq_info, dtype=dtype, device=device)

    with ExitStack() as stack:
        stack.enter_context(patch.object(attention_dispatch_utils, "HAS_FLASH_ATTENTION", False))
        stack.enter_context(patch.object(attention_dispatch_utils, "HAS_XFORMERS", False))
        stack.enter_context(
            patch.object(
                attention_dispatch_utils,
                "build_sdpa_packed_attention_mask",
                side_effect=_capture_mask,
            )
        )
        with torch.no_grad():
            outputs = model(**inputs)

    assert mask_calls, "SDPA packed mask was not constructed"
    assert outputs.loss is not None

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.free_memory()
