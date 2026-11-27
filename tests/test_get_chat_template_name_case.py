# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""get_chat_template reads a template name case-insensitively, as it already does for the system message."""

import pytest

tokenizers = pytest.importorskip("tokenizers")
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template

MESSAGES = [{"role": "user", "content": "hi"}]


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["<s>", "</s>", "<|im_start|>", "<|im_end|>", "hi"]
    tk = Tokenizer(
        models.WordLevel({"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}, unk_token = "[UNK]")
    )
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object = tk, unk_token = "[UNK]", bos_token = "<s>", eos_token = "</s>"
    )
    tok.name_or_path = "tiny/chat-tokenizer"
    return tok


def _render(chat_template, **kwargs):
    tokenizer = get_chat_template(_tokenizer(), chat_template = chat_template, **kwargs)
    return tokenizer.apply_chat_template(MESSAGES, tokenize = False, add_generation_prompt = True)


@pytest.mark.parametrize("name", ["ChatML", "CHATML", "Llama-3.1", "Qwen-2.5", "Phi-4"])
def test_a_mixed_case_name_picks_the_same_template(name):
    assert _render(name) == _render(name.lower())


def test_a_mixed_case_name_still_takes_the_system_message():
    rendered = _render("Llama-3.1", system_message = "Answer in French.")
    assert rendered == _render("llama-3.1", system_message = "Answer in French.")
    assert "Answer in French." in rendered
