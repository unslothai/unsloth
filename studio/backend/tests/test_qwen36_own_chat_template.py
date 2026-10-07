# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

try:
    from core.inference.inference import InferenceBackend  # noqa: E402
except (ImportError, RuntimeError) as exc:  # pragma: no cover
    pytest.skip(
        f"full inference backend unavailable ({type(exc).__name__}: {exc})",
        allow_module_level = True,
    )

MODEL = "unsloth/Qwen3.6-27B"

_QWEN36_STYLE_TEMPLATE = (
    "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' }}"
    "{% if m.role == 'assistant' and m.reasoning_content and preserve_thinking is defined"
    " and preserve_thinking is true %}{{ '<think>\\n' + m.reasoning_content + '\\n</think>\\n\\n' }}"
    "{% endif %}{{ m.content + '<|im_end|>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}"
    "{% else %}{{ '<think>\\n' }}{% endif %}{% endif %}"
)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "<|endoftext|>": 1, "<|im_start|>": 2, "<|im_end|>": 3}
    backend = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend,
        unk_token = "[UNK]",
        eos_token = "<|im_end|>",
        pad_token = "<|endoftext|>",
    )
    tokenizer.chat_template = _QWEN36_STYLE_TEMPLATE
    return tokenizer


def _prompt(monkeypatch, messages, **kwargs):
    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = MODEL
    backend.models = {MODEL: {"tokenizer": _tokenizer(), "is_vision": False}}
    prompts = []
    monkeypatch.setattr(
        backend,
        "generate_stream",
        lambda prompt, *a, **k: prompts.append(prompt) or iter(()),
        raising = False,
    )
    list(
        backend._generate_chat_response_inner(
            messages = messages, system_prompt = "Be brief.", **kwargs
        )
    )
    return prompts[0]


def test_thinking_off_reaches_qwen36(monkeypatch):
    prompt = _prompt(
        monkeypatch, [{"role": "user", "content": "What is 17 * 23?"}], enable_thinking = False
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_preserve_thinking_keeps_earlier_reasoning_for_qwen36(monkeypatch):
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "two plus two is four"},
        {"role": "user", "content": "And times 3?"},
    ]
    prompt = _prompt(monkeypatch, messages, enable_thinking = True, preserve_thinking = True)
    assert "<think>\ntwo plus two is four\n</think>\n\n4<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
