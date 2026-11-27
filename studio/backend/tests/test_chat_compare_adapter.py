# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib
import sys
import threading
import types
from unittest.mock import MagicMock

import pytest

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
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

try:
    import core.inference.inference
except ImportError:
    pass

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

MODEL = "tiny-llama-lora"


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "<eos>": 1, "<pad>": 2, **{f"w{i}": i + 3 for i in range(61)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        unk_token = "[UNK]",
        eos_token = "<eos>",
        pad_token = "<pad>",
        model_input_names = ["input_ids", "attention_mask"],
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m.content + ' ' }}{% endfor %}"
    return tokenizer


def _lora_backend():
    peft = pytest.importorskip("peft")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    try:
        from core.inference.inference import InferenceBackend
    except (ImportError, RuntimeError) as exc:
        pytest.skip(reason = f"full inference backend unavailable ({type(exc).__name__}: {exc})")

    class _Lora(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.disabled = False

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)

        def disable_adapter_layers(self):
            self.disabled = True

        def enable_adapter_layers(self):
            self.disabled = False

    class _PeftLlama(peft.PeftModel):
        def __init__(self, model):
            torch.nn.Module.__init__(self)
            self.base_model = _Lora(model)
            self.ran_without_lora = []

        def generate(self, **kwargs):
            self.ran_without_lora.append(self.base_model.disabled)
            return self.base_model.model.generate(**kwargs)

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        max_position_embeddings = 64,
        bos_token_id = None,
        eos_token_id = 1,
        pad_token_id = 2,
    )
    model = _PeftLlama(LlamaForCausalLM(config).eval())
    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = MODEL
    backend._generation_lock = threading.Lock()
    backend.models = {MODEL: {"model": model, "tokenizer": _tokenizer(), "is_vision": False}}
    return backend, model


def _reply(backend, use_adapter = None):
    kwargs = dict(
        messages = [{"role": "user", "content": "w1 w2 w3"}],
        system_prompt = "",
        temperature = 0.0,
        max_new_tokens = 4,
    )
    if use_adapter is None:
        list(backend.generate_chat_response(**kwargs))
    else:
        list(backend.generate_with_adapter_control(use_adapter = use_adapter, **kwargs))


def test_a_plain_chat_after_a_base_pane_reply_runs_the_lora():
    backend, model = _lora_backend()

    _reply(backend, use_adapter = False)
    _reply(backend)

    assert model.ran_without_lora == [True, False]


def test_compare_panes_keep_their_own_adapter_state():
    backend, model = _lora_backend()

    for use_adapter in (False, True, False, True):
        _reply(backend, use_adapter = use_adapter)

    assert model.ran_without_lora == [True, False, True, False]


def test_a_failed_base_pane_reply_still_turns_the_lora_back_on():
    backend, model = _lora_backend()
    model.generate = MagicMock(side_effect = RuntimeError("boom"))

    with pytest.raises(Exception):
        _reply(backend, use_adapter = False)

    assert model.base_model.disabled is False
