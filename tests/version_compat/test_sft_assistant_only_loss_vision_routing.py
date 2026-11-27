# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""SFT `assistant_only_loss = True` on a vision dataset masks with Unsloth's train_on_responses_only.

TRL refuses it below transformers 5.18 ("Assistant-only loss for vision datasets requires transformers>=5.18.0")
and above it uses processor assistant masks; UnslothVisionDataCollator never read the flag and trained on every
turn. The labels must equal UnslothVisionDataCollator(train_on_responses_only = True) with the chat template's
markers, on every transformers version; runs without the flag, and text runs, are unchanged."""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")

import functools
import importlib.util
import sys
from pathlib import Path

import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)
if importlib.util.find_spec("trl") is None or importlib.util.find_spec("unsloth") is None:
    pytest.skip("trl or unsloth not installed", allow_module_level = True)


def _trl_has_assistant_only_loss():
    import dataclasses

    from trl import SFTConfig
    return any(field.name == "assistant_only_loss" for field in dataclasses.fields(SFTConfig))


# TRL added SFTConfig.assistant_only_loss in 0.20; below it (the GRPO fake-run floor lane, TRL 0.18.2) the flag
# cannot be requested, so there is nothing to route.
if not _trl_has_assistant_only_loss():
    pytest.skip("this TRL's SFTConfig has no assistant_only_loss", allow_module_level = True)

# Reuse the fake-train harness: it applies the CUDA spoof at import and provides the CPU patches.
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


VLMS = [
    "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
    "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration",
    "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
]
VLM_IDS = ["qwen2vl", "qwen3vl", "gemma3"]
TEXT = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@functools.lru_cache(maxsize = None)
def _load(repo):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    try:
        if repo == TEXT:
            tok, model = (
                AutoTokenizer.from_pretrained(repo),
                AutoModelForCausalLM.from_pretrained(repo),
            )
        else:
            tok, model = (
                AutoProcessor.from_pretrained(repo),
                AutoModelForImageTextToText.from_pretrained(repo),
            )
    except OSError as e:
        pytest.skip(f"could not fetch {repo} (network/hub): {str(e)[:150]}")
    except (KeyError, ValueError) as e:
        # An architecture this transformers predates (Qwen3-VL before 4.57).
        if "does not recognize this architecture" not in str(e) and not isinstance(e, KeyError):
            raise
        pytest.skip(f"{repo}: this transformers cannot build it ({str(e)[:120]})")
    model = model.to(torch.float32)
    if not hasattr(model, "for_training"):
        model.for_training = lambda *a, **k: model.train()
    return model, tok


def _messages(image_part):
    first = [image_part] if image_part else []
    return [
        {"role": "user", "content": first + [{"type": "text", "text": "What is in this picture?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A red square."}]},
        {"role": "user", "content": [{"type": "text", "text": "What color is it?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is red."}]},
    ]


def _vision_rows():
    from PIL import Image
    image = Image.new("RGB", (32, 32), (255, 0, 0))
    return [{"messages": _messages({"type": "image"}), "images": [image]} for _ in range(2)]


def _config(
    tmp_path,
    assistant_only_loss,
    vision = True,
):
    from trl import SFTConfig
    extra = (
        dict(
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
        )
        if vision
        else {}
    )
    return SFTConfig(
        output_dir = str(tmp_path / "aol"),
        per_device_train_batch_size = 2,
        max_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
        max_length = None,
        padding_free = False,
        dataset_num_proc = 1,
        assistant_only_loss = assistant_only_loss,
        **extra,
    )


def _reference_labels(repo, rows):
    """What Unsloth's own collator produces with the chat template's markers."""
    from unsloth.trainer import UnslothVisionDataCollator
    from unsloth_zoo.dataset_utils import get_chat_template_parts

    model, processor = _load(repo)
    instruction_part, response_part = get_chat_template_parts(processor)
    collator = UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only = True,
        instruction_part = instruction_part,
        response_part = response_part,
    )
    return collator([dict(r) for r in rows])["labels"]


def _first_batch(trainer):
    return next(iter(trainer.get_train_dataloader()))


def _kept(processor, batch):
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    return processor.tokenizer.decode(ids[labels != -100])


@pytest.mark.parametrize("repo", VLMS, ids = VLM_IDS)
def test_vision_assistant_only_loss_trains_with_unsloth_response_masks(tmp_path, repo):
    from datasets import Dataset
    from trl import SFTTrainer

    model, processor = _load(repo)
    rows = _vision_rows()
    trainer = SFTTrainer(
        model = model,
        processing_class = processor,
        args = _config(tmp_path, True),
        train_dataset = Dataset.from_list(rows),
    )
    batch = _first_batch(trainer)
    assert batch["labels"].tolist() == _reference_labels(repo, rows).tolist()
    kept = _kept(processor, batch)
    assert "A red square." in kept and "It is red." in kept
    assert "What color is it?" not in kept and "What is in this picture?" not in kept
    print(f"{repo.split('tiny-')[-1]} kept: {kept!r}")
    trainer.train()


@pytest.mark.parametrize("repo", VLMS[:1], ids = VLM_IDS[:1])
def test_plain_unsloth_vision_collator_honours_assistant_only_loss(tmp_path, repo):
    from trl import SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    model, processor = _load(repo)
    rows = _vision_rows()
    trainer = SFTTrainer(
        model = model,
        processing_class = processor,
        args = _config(tmp_path, True),
        data_collator = UnslothVisionDataCollator(model, processor),
        train_dataset = rows,
    )
    assert _first_batch(trainer)["labels"].tolist() == _reference_labels(repo, rows).tolist()


def test_a_masking_collator_is_not_masked_twice_and_args_can_be_reused(tmp_path):
    from trl import SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    repo = VLMS[0]
    model, processor = _load(repo)
    rows = _vision_rows()
    collator = UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only = True,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    masker = collator.train_on_responses_only
    args = _config(tmp_path, True)
    SFTTrainer(
        model = model,
        processing_class = processor,
        args = args,
        data_collator = collator,
        train_dataset = rows,
    )
    assert collator.train_on_responses_only is masker
    # Re-created on the same args with a fresh plain collator: still masked.
    second = SFTTrainer(
        model = model,
        processing_class = processor,
        args = args,
        data_collator = UnslothVisionDataCollator(model, processor),
        train_dataset = rows,
    )
    assert _first_batch(second)["labels"].tolist() == _reference_labels(repo, rows).tolist()


def test_no_chat_template_leaves_trl_in_charge(tmp_path, capsys):
    """Markers cannot be detected without a chat template, which TRL cannot mask either: Unsloth changes
    nothing (flag, collator) and prints how to pass the parts, so TRL fails exactly as it does on main."""
    import transformers
    from datasets import Dataset
    from packaging.version import Version
    from transformers import AutoProcessor
    from trl import SFTTrainer
    from unsloth.trainer import _route_vision_assistant_only_loss

    model, _ = _load(VLMS[0])
    processor = AutoProcessor.from_pretrained(VLMS[0])  # fresh: the template is removed below
    processor.chat_template = None
    processor.tokenizer.chat_template = None
    dataset = Dataset.from_list(_vision_rows())

    args = _config(tmp_path, True)
    assert _route_vision_assistant_only_loss(args, None, model, processor, dataset) is None
    assert args.assistant_only_loss is True
    assert not getattr(args, "_unsloth_assistant_only_loss", False)
    assert "could not auto-detect" in capsys.readouterr().out

    # End to end, the error is TRL's own: its <5.18 vision refusal, or (5.18+) its missing-template failure.
    with pytest.raises((ValueError, TypeError)) as raised:
        SFTTrainer(
            model = model,
            processing_class = processor,
            args = _config(tmp_path, True),
            train_dataset = dataset,
        )
    if Version(transformers.__version__.split("+")[0]) < Version("5.18.0.dev0"):
        assert "requires transformers>=5.18.0" in str(raised.value)
    assert "Unsloth" not in str(raised.value)


def test_vision_without_assistant_only_loss_is_unchanged(tmp_path):
    from datasets import Dataset
    from trl import SFTTrainer

    model, processor = _load(VLMS[0])
    trainer = SFTTrainer(
        model = model,
        processing_class = processor,
        args = _config(tmp_path, False),
        train_dataset = Dataset.from_list(_vision_rows()),
    )
    assert type(trainer.data_collator).__name__ == "DataCollatorForVisionLanguageModeling"
    kept = _kept(processor, _first_batch(trainer))
    assert "What color is it?" in kept and "It is red." in kept


def test_text_assistant_only_loss_keeps_only_the_replies(tmp_path):
    from datasets import Dataset
    from trl import SFTTrainer

    model, tok = _load(TEXT)
    rows = [
        {
            "messages": [
                {"role": m["role"], "content": m["content"][-1]["text"]} for m in _messages(None)
            ]
        }
    ] * 2
    trainer = SFTTrainer(
        model = model,
        processing_class = tok,
        args = _config(tmp_path, True, vision = False),
        train_dataset = Dataset.from_list(rows),
    )
    assert trainer.args.assistant_only_loss is True
    batch = _first_batch(trainer)
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    kept = tok.decode(ids[labels != -100])
    # Byte-identical to unsloth main on transformers 5.17 and 5.18 (Qwen3's template adds the empty think block).
    assert kept == "A red square.<|im_end|>\n<think>\n\n</think>\n\nIt is red.<|im_end|>\n"
