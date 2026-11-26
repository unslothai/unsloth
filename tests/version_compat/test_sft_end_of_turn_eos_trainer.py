# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL 1.19 (#8565) adds the chat template's end-of-turn id to model.generation_config.eos_token_id in
SFTTrainer.__init__. Through the real UnslothSFTTrainer: the id is added once on TRL >= 1.19 (and never on
older TRL), the saved generation_config.json carries it, Unsloth's harmony stop-token widening still composes
with the list, and zoo's GRPO pad == eos rows read the widened set the way TRL's completion_mask does.

Base-model setup on purpose: tokenizer eos = pad = <|endoftext|>, ChatML template ending turns in <|im_end|>.
"""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")

import importlib.util
import json
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
from packaging.version import Version  # noqa: E402

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
    end_of_text = tok.convert_tokens_to_ids("<|endoftext|>")
    end_of_turn = tok.convert_tokens_to_ids("<|im_end|>")
    tok.eos_token = "<|endoftext|>"
    tok.pad_token = "<|endoftext|>"
    tok.chat_template = _TEMPLATE
    # Base model: config and generation config agree with the tokenizer (else transformers' Trainer re-aligns them).
    model.config.eos_token_id = end_of_text
    model.generation_config.eos_token_id = end_of_text
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
    )
    trainer = SFTTrainer(
        model = model, processing_class = tok, args = cfg, train_dataset = Dataset.from_list(rows)
    )
    return trainer, tok, end_of_text, end_of_turn


def _eos_list(generation_config):
    eos = generation_config.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos or [])


def test_end_of_turn_joins_the_stop_tokens_on_trl_1_19(tmp_path, patched_sft):
    trainer, tok, end_of_text, end_of_turn = _build(tmp_path)
    eos = _eos_list(trainer.model.generation_config)
    if Version(patched_sft.__version__) >= Version("1.19.0"):
        assert eos == [end_of_text, end_of_turn]
    else:
        assert eos == [end_of_text]
    # Unsloth's harmony <|call|> widening takes the list as it is (no harmony tokens here: unchanged).
    from unsloth.models._utils import patch_harmony_tool_call_eos

    patch_harmony_tool_call_eos(trainer.model, tok)
    assert _eos_list(trainer.model.generation_config) == eos
    trainer.train()
    trainer.save_model(str(tmp_path / "saved"))
    saved = json.loads((tmp_path / "saved" / "generation_config.json").read_text())
    assert (
        saved["eos_token_id"]
        if isinstance(saved["eos_token_id"], list)
        else [saved["eos_token_id"]]
    ) == eos


def test_grpo_pad_is_eos_reads_the_widened_set(tmp_path, patched_sft):
    # A later GRPO run on this model: TRL >= 1.x builds eos_token_ids from the tokenizer eos plus
    # generation_config.eos_token_id, and zoo's pad == eos rows must agree with TRL's completion_mask.
    import types

    from unsloth_zoo.rl_replacements import grpo_pad_is_eos_rows

    trainer, tok, end_of_text, end_of_turn = _build(tmp_path)
    eos_ids = _eos_list(trainer.model.generation_config)
    if tok.eos_token_id not in eos_ids:
        eos_ids = [tok.eos_token_id, *eos_ids]
    grpo = types.SimpleNamespace(
        processing_class = tok, eos_token_ids = eos_ids, eos_token_id = tok.eos_token_id
    )
    pad = tok.pad_token_id
    assert pad == end_of_text
    completions = torch.tensor(
        [
            [5, 6, end_of_text, pad, pad],  # stopped on the pad-id EOS: TRL keeps it
            # TRL >= 1.19 stops on <|im_end|> and pads after it; before, <|im_end|> is a plain token
            [5, 6, end_of_turn, pad, pad],
            [5, 6, 7, 8, 9],  # ran to the length cap
        ]
    )
    rows = grpo_pad_is_eos_rows(grpo, completions)
    on_1_19 = Version(patched_sft.__version__) >= Version("1.19.0")
    assert rows.tolist() == [True, not on_1_19, False]
    assert (end_of_turn in eos_ids) is on_1_19
