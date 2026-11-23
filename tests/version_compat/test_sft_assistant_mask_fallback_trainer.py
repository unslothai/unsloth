# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""assistant_only_loss on a chat template TRL has no training template for, through the real UnslothSFTTrainer.

TRL raises "not training-compatible" for such templates (every Unsloth-edited one). Unsloth swaps in Zoo's
train_on_responses_only masks instead. TRL 1.18 changed the call it patches to pass require_prefix_preservation=False.
"""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)
if importlib.util.find_spec("trl") is None or importlib.util.find_spec("unsloth") is None:
    pytest.skip("trl or unsloth not installed", allow_module_level = True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()

import torch  # noqa: E402

_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
_TEMPLATE = (
    "{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


def _eager_compile(
    model = None,
    *args,
    **kwargs,
):
    if callable(model):
        return model
    return lambda fn: fn


@pytest.fixture(scope = "module", autouse = True)
def _cpu_only_torch():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch, "compile", _eager_compile)
        if hasattr(torch, "accelerator"):
            mp.setattr(torch.accelerator, "is_available", lambda *a, **k: False)
        yield mp


@pytest.fixture(scope = "module")
def patched_sft(_cpu_only_torch):
    global torch
    import unsloth  # noqa: F401

    _cpu_only_torch.setattr(torch, "compile", _eager_compile)
    try:
        import torch._dynamo
        _cpu_only_torch.setattr(torch._dynamo.config, "disable", True)
    except Exception:
        pass

    import trl

    if trl.SFTTrainer.__name__ != "UnslothSFTTrainer":
        from unsloth.models.rl import _patch_trl_rl_trainers
        from unsloth.trainer import _patch_trl_trainer

        _patch_trl_rl_trainers("sft_trainer")
        _patch_trl_trainer()
    from trl.trainer import sft_trainer

    source = inspect.getsource(sft_trainer)
    if "get_training_chat_template(processing_class" not in source:
        pytest.skip("this TRL has no training-template swap")
    return trl


def _build(tmp_path):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    assert SFTTrainer.__name__ == "UnslothSFTTrainer", "SFT patch did not apply"
    try:
        tok = AutoTokenizer.from_pretrained(_MODEL)
        model = AutoModelForCausalLM.from_pretrained(_MODEL).to(torch.float32)
    except OSError as error:
        pytest.skip(f"could not fetch {_MODEL}: {str(error)[:150]}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Plain ChatML without {% generation %} markers, rendered unlike any TRL family: TRL has no training template.
    tok.chat_template = _TEMPLATE
    model.max_seq_length = 128
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"What is {i} plus {i}?"},
                {"role": "assistant", "content": f"It is {2 * i}."},
            ]
        }
        for i in range(4)
    ]
    cfg = SFTConfig(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        max_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
        assistant_only_loss = True,
    )
    return SFTTrainer(
        model = model, processing_class = tok, args = cfg, train_dataset = Dataset.from_list(rows)
    )


def test_unknown_template_trains_on_assistant_tokens_only(tmp_path, patched_sft):
    trainer = _build(tmp_path)
    assert getattr(trainer, "_unsloth_assistant_mask_fallback", False) is True
    assert trainer.chat_template is None
    row = trainer.train_dataset[0]
    mask = row.get("assistant_masks", row.get("completion_mask"))
    if mask is None and "labels" in row:
        mask = [int(label != -100) for label in row["labels"]]
    assert mask is not None, f"no assistant mask column in {sorted(row)}"
    assert 0 < sum(mask) < len(mask), "assistant_only_loss must mask the prompt and keep the answer"
