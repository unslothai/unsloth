# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _tokenizer():
    vocab = {"<unk>": 0, "<|return|>": 1, "<|endoftext|>": 2}
    return get_chat_template(
        PreTrainedTokenizerFast(
            tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
            eos_token = "<|return|>",
            unk_token = "<unk>",
            pad_token = "<|endoftext|>",
        ),
        "gpt-oss",
    )


def _render(messages, add_generation_prompt = False):
    return _tokenizer().apply_chat_template(
        messages, tokenize = False, add_generation_prompt = add_generation_prompt
    )


USER = {"role": "user", "content": "Hi"}
REPLY = {"role": "assistant", "content": "Hello!"}


@pytest.mark.parametrize(
    "messages",
    [
        [USER, REPLY],
        [{"role": "system", "content": "Be brief."}, USER, REPLY],
        [USER, {"role": "assistant", "content": "Hello!", "thinking": "Greet back."}],
        [
            USER,
            REPLY,
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "See you."},
        ],
    ],
)
def test_last_training_turn_ends_with_the_stop_token(messages):
    text = _render(messages)
    assert text.endswith("<|channel|>final<|message|>" + messages[-1]["content"] + "<|return|>")
    assert text.count("<|return|>") == 1


def test_earlier_turns_and_generation_prompt_keep_end():
    text = _render([USER, REPLY, {"role": "user", "content": "Bye"}])
    assert "<|channel|>final<|message|>Hello!<|end|><|start|>user" in text
    assert "<|return|>" not in text
    prompt = _render([USER, REPLY, {"role": "user", "content": "Bye"}], add_generation_prompt = True)
    assert prompt == text + "<|start|>assistant"
