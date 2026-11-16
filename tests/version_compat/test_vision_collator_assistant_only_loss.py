# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""UnslothVisionDataCollator builds its own labels and never read `assistant_only_loss`, so the flag silently trained
on the system and user turns (Unsloth's inline-image rows) or hit TRL's transformers<5.18 vision refusal (an `images`
column). The SFT trainer now masks with the chat template's markers, or refuses when they cannot be read."""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")

import importlib.util
import sys
from pathlib import Path

import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)
if importlib.util.find_spec("trl") is None or importlib.util.find_spec("unsloth") is None:
    pytest.skip("trl or unsloth not installed", allow_module_level = True)

# Reuse the fake-train harness: it applies the CUDA spoof at import and provides the CPU fixtures.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_trl_fake_train_cpu as _harness  # noqa: E402


@pytest.fixture(scope = "module", autouse = True)
def _cpu_only_torch():
    with pytest.MonkeyPatch.context() as mp:
        _harness._fake_cpu_gpu(mp)
        yield mp


@pytest.fixture(autouse = True)
def _require_stack(_cpu_only_torch):
    import torch
    import torch._dynamo
    import unsloth  # noqa: F401  -- patches TRL trainers to the Unsloth variants

    # `import unsloth` reinstalls the real torch.compile; keep the eager passthrough and dynamo off.
    _cpu_only_torch.setattr(torch, "compile", _harness._eager_compile)
    _cpu_only_torch.setattr(torch._dynamo.config, "disable", True)


_VLM = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"


def _load_vlm():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(_VLM)
        model = AutoModelForImageTextToText.from_pretrained(_VLM).to(torch.float32)
    except OSError as e:
        pytest.skip(f"could not fetch {_VLM} (network/hub): {str(e)[:150]}")
    if not hasattr(model, "for_training"):
        model.for_training = lambda *a, **k: model.train()
    return model, processor


def _rows(images_column = False, n = 4):
    from PIL import Image

    image = Image.new("RGB", (32, 32), (255, 0, 0))
    part = {"type": "image"} if images_column else {"type": "image", "image": image}
    messages = [
        {"role": "user", "content": [part, {"type": "text", "text": "What is in this picture?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A red square."}]},
        {"role": "user", "content": [{"type": "text", "text": "What color is it?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is red."}]},
    ]
    row = {"messages": messages, "images": [image]} if images_column else {"messages": messages}
    return [dict(row) for _ in range(n)]


def _config(tmp_path, assistant_only_loss):
    from trl import SFTConfig
    return SFTConfig(
        output_dir = str(tmp_path / "vlm_aol"),
        per_device_train_batch_size = 2,
        max_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
        max_length = None,
        assistant_only_loss = assistant_only_loss,
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
    )


def _trainer(
    tmp_path,
    assistant_only_loss,
    images_column = False,
    **collator_kwargs,
):
    from trl import SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    assert SFTTrainer.__name__ == "UnslothSFTTrainer", "SFT patch did not apply"
    model, processor = _load_vlm()
    return SFTTrainer(
        model = model,
        processing_class = processor,
        data_collator = UnslothVisionDataCollator(model, processor, **collator_kwargs),
        args = _config(tmp_path, assistant_only_loss),
        train_dataset = _rows(images_column),
    ), processor


def _kept_text(trainer, processor):
    batch = next(iter(trainer.get_train_dataloader()))
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    return processor.tokenizer.decode(ids[labels != -100])


def _assert_assistant_only(kept):
    assert "A red square." in kept and "It is red." in kept
    assert "What color is it?" not in kept and "What is in this picture?" not in kept
    assert "system" not in kept


@pytest.mark.parametrize("images_column", [False, True], ids = ["inline_images", "images_column"])
def test_assistant_only_loss_with_a_plain_vision_collator_masks_with_template_markers(
    tmp_path, images_column
):
    trainer, processor = _trainer(tmp_path, assistant_only_loss = True, images_column = images_column)
    _assert_assistant_only(_kept_text(trainer, processor))
    assert (
        trainer.args.assistant_only_loss is False
    ), "TRL would gate the flag the collator now applies"


def test_assistant_only_loss_is_refused_when_markers_cannot_be_read(tmp_path, monkeypatch):
    import unsloth_zoo.dataset_utils as dataset_utils

    def _no_markers(tokenizer):
        raise ValueError("Unsloth: No chat_template to auto-detect from")

    monkeypatch.setattr(dataset_utils, "get_chat_template_parts", _no_markers)
    with pytest.raises(ValueError, match = "could not be read from the chat template"):
        _trainer(tmp_path, assistant_only_loss = True)


def test_without_assistant_only_loss_the_collator_trains_on_every_turn(tmp_path):
    """Control: the masking is keyed on the flag, and this is the full-sequence labelling it would have hidden."""
    trainer, processor = _trainer(tmp_path, assistant_only_loss = False)
    kept = _kept_text(trainer, processor)
    assert "What color is it?" in kept and "It is red." in kept


def test_a_response_masking_collator_still_honours_assistant_only_loss(tmp_path):
    trainer, processor = _trainer(
        tmp_path,
        assistant_only_loss = True,
        images_column = True,
        train_on_responses_only = True,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    _assert_assistant_only(_kept_text(trainer, processor))
