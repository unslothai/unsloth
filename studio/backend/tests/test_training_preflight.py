# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_preflight_first_batch rejects an empty/non-integer first batch (the base-model
empty-chat-template crash) before train(). The real methods are bound onto a light
fake self so the production logic runs against controlled batches."""

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


def _stub_if_missing(name, attrs):
    """Register a stub module for a dep the CPU backend CI job does not install.

    The pytest job has studio.txt + torch + transformers but not unsloth/trl,
    which core.training.trainer imports at module scope. Stub the absent ones
    (real installs are left alone) so importing it for the two pure helper
    methods never breaks test collection. __spec__ = None keeps the trainer's
    own _ensure_real_packages namespace-shadow guard a no-op on the stub.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
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

from core.training.trainer import UnslothTrainer  # noqa: E402

_preflight = UnslothTrainer._preflight_first_batch
_renders_empty = UnslothTrainer._chat_template_renders_empty


class _FakeInnerTrainer:
    def __init__(
        self,
        *,
        batch = None,
        dataloader_error = None,
        train_dataset = None,
    ):
        self._batch = batch
        self._dataloader_error = dataloader_error
        self.train_dataset = train_dataset

    def get_train_dataloader(self):
        if self._dataloader_error is not None:
            raise self._dataloader_error
        return [self._batch]


def _fake_self(
    *,
    inner,
    model_name = "org/Some-Model",
    tokenizer = None,
):
    s = SimpleNamespace(trainer = inner, model_name = model_name, tokenizer = tokenizer)
    # Bind real methods so self._chat_template_renders_empty() resolves.
    s._preflight_first_batch = _preflight.__get__(s)
    s._chat_template_renders_empty = _renders_empty.__get__(s)
    return s


class _EmptyTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return ""


class _RealTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return "<|im_start|>user\nhi<|im_end|>"


class TestPreflightFirstBatch(unittest.TestCase):
    def test_float_input_ids_with_empty_template_suggests_instruct(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.float32)},
            train_dataset = ds,
        )
        s = _fake_self(
            inner = inner, model_name = "Qwen/Qwen2-VL-7B", tokenizer = _EmptyTemplateTokenizer()
        )
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("chat template", msg)
        self.assertIn("Qwen/Qwen2-VL-7B-Instruct", msg)
        self.assertIn("base (pretrained) model", msg)

    def test_no_instruct_hint_when_model_already_instruct(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.float32)},
            train_dataset = ds,
        )
        s = _fake_self(
            inner = inner, model_name = "org/Foo-Instruct", tokenizer = _EmptyTemplateTokenizer()
        )
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertNotIn("such as", msg)  # no Instruct suggestion for an Instruct model
        self.assertIn("instruction-tuned variant", msg)

    def test_empty_int_input_ids_generic_message(self):
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.long)},
            train_dataset = [{"text": "already tokenized path"}],
        )
        s = _fake_self(inner = inner, tokenizer = _RealTemplateTokenizer())
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("invalid token IDs", msg)
        self.assertNotIn("chat template", msg)

    def test_valid_batch_returns_none(self):
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.randint(0, 1000, (2, 34), dtype = torch.long)},
        )
        s = _fake_self(inner = inner)
        self.assertIsNone(s._preflight_first_batch())

    def test_dataloader_error_is_surfaced(self):
        inner = _FakeInnerTrainer(dataloader_error = RuntimeError("boom"))
        s = _fake_self(inner = inner, model_name = "org/M")
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("failed to build the first training batch", msg)
        self.assertIn("org/M", msg)

    def test_missing_input_ids_does_not_false_positive(self):
        inner = _FakeInnerTrainer(batch = {"pixel_values": torch.zeros((1, 3))})
        s = _fake_self(inner = inner)
        self.assertIsNone(s._preflight_first_batch())


class TestChatTemplateRendersEmpty(unittest.TestCase):
    def _self(self, *, train_dataset, tokenizer):
        inner = _FakeInnerTrainer(train_dataset = train_dataset)
        return _fake_self(inner = inner, tokenizer = tokenizer)

    def test_empty_render_detected(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        s = self._self(train_dataset = ds, tokenizer = _EmptyTemplateTokenizer())
        self.assertTrue(s._chat_template_renders_empty())

    def test_nonempty_render_not_flagged(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        s = self._self(train_dataset = ds, tokenizer = _RealTemplateTokenizer())
        self.assertFalse(s._chat_template_renders_empty())

    def test_no_messages_key_not_flagged(self):
        s = self._self(train_dataset = [{"text": "raw"}], tokenizer = _EmptyTemplateTokenizer())
        self.assertFalse(s._chat_template_renders_empty())


if __name__ == "__main__":
    unittest.main()
