# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
from datasets import Dataset  # noqa: E402


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unusable imports
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as tmod  # noqa: E402
from utils.datasets import format_and_template_dataset  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


BASE_EOS = "<|end_of_text|>"


class _BaseModelTokenizer:
    eos_token = BASE_EOS
    chat_template = None

    def apply_chat_template(self, conversation, **_kwargs):
        if not self.chat_template:
            raise ValueError(
                "Cannot use chat template functions because tokenizer.chat_template is not set"
            )
        if self.chat_template == "alpaca":
            raise ValueError("alpaca renders instruction rows, not conversations")
        return "".join(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n" for t in conversation)


def _get_chat_template(
    tokenizer,
    chat_template = "chatml",
    **_kwargs,
):
    rebuilt = _BaseModelTokenizer()
    rebuilt.chat_template = chat_template
    if chat_template == "chatml":
        rebuilt.eos_token = "<|im_end|>"
    return rebuilt


@pytest.fixture
def trainer(monkeypatch):
    unsloth = types.ModuleType("unsloth")
    unsloth.chat_templates = types.ModuleType("unsloth.chat_templates")
    unsloth.chat_templates.get_chat_template = _get_chat_template
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.chat_templates", unsloth.chat_templates)

    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    t = tmod.UnslothTrainer()
    t.model_name = "unsloth/Llama-3.2-1B"
    t.tokenizer = _BaseModelTokenizer()
    return t


def _train_with(
    trainer,
    dataset,
    tmp_path,
    monkeypatch,
    own_template = None,
):
    captured = {}
    monkeypatch.setattr(tmod, "SFTConfig", lambda **kwargs: kwargs, raising = False)
    monkeypatch.setattr(tmod, "SFTTrainer", lambda **kwargs: captured.update(kwargs), raising = False)
    monkeypatch.setattr(tmod, "resolve_output_dir", lambda p: tmp_path, raising = True)

    trainer.tokenizer.chat_template = own_template
    dataset_info = format_and_template_dataset(
        dataset,
        model_name = trainer.model_name,
        tokenizer = trainer.tokenizer,
        num_proc = 1,
    )
    assert dataset_info["success"], dataset_info["errors"]

    # load_model swaps in the checkpoint's own tokenizer after the dataset is formatted.
    trainer.model = object()
    trainer.tokenizer = _BaseModelTokenizer()
    trainer.tokenizer.chat_template = own_template

    with contextlib.suppress(AttributeError, TypeError, ValueError, KeyError):
        trainer._train_worker(
            dataset_info,
            batch_size = 1,
            gradient_accumulation_steps = 1,
            max_steps = 1,
            output_dir = str(tmp_path),
        )
    assert captured, "SFTTrainer was never built, so this asserts nothing"
    assert captured["tokenizer"] is trainer.tokenizer
    return dataset_info


CONVERSATION = Dataset.from_dict(
    {"messages": [[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]]}
)


def test_base_model_trains_and_saves_with_the_chatml_template_its_rows_used(
    trainer, tmp_path, monkeypatch
):
    dataset_info = _train_with(trainer, CONVERSATION, tmp_path, monkeypatch)

    assert dataset_info["dataset"]["text"][0].startswith("<|im_start|>user\nHi<|im_end|>")
    assert trainer.tokenizer.chat_template == "chatml"
    assert trainer.tokenizer.eos_token == "<|im_end|>"


def test_base_model_trains_and_saves_with_the_alpaca_template_its_rows_used(
    trainer, tmp_path, monkeypatch
):
    rows = Dataset.from_dict({"instruction": ["Say hi"], "input": [""], "output": ["Hello!"]})

    dataset_info = _train_with(trainer, rows, tmp_path, monkeypatch)

    assert dataset_info["dataset"]["text"][0].endswith(f"Hello!{BASE_EOS}")
    assert trainer.tokenizer.chat_template == "alpaca"
    assert trainer.tokenizer.eos_token == BASE_EOS


def test_a_model_with_its_own_template_keeps_it(trainer, tmp_path, monkeypatch):
    _train_with(trainer, CONVERSATION, tmp_path, monkeypatch, own_template = "own-template")

    assert trainer.tokenizer.chat_template == "own-template"
    assert trainer.tokenizer.eos_token == BASE_EOS
