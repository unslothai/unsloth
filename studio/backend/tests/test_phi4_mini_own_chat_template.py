# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets import apply_chat_template_to_dataset
from utils.datasets.completion_masking import lookup_manual_markers
from utils.datasets.model_mappings import MODEL_TO_TEMPLATE_MAPPER

_PHI4_MINI_TEMPLATE = (
    "{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}"
)

_CONVERSATION = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "<|endoftext|>": 1, "<|user|>": 2, "<|assistant|>": 3, "<|end|>": 4}
    backend = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend,
        unk_token = "[UNK]",
        eos_token = "<|endoftext|>",
        pad_token = "<|endoftext|>",
    )
    tokenizer.chat_template = _PHI4_MINI_TEMPLATE
    return tokenizer


@pytest.mark.parametrize(
    "name",
    [
        "unsloth/Phi-4-mini-instruct",
        "microsoft/Phi-4-mini-instruct",
        "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
        "unsloth/phi-4-mini-reasoning",
        "microsoft/Phi-4-mini-reasoning",
    ],
)
def test_phi4_mini_trains_on_its_own_chat_template(name):
    assert lookup_manual_markers(name) == (None, None, None)
    tokenizer = _tokenizer()
    expected = tokenizer.apply_chat_template(_CONVERSATION, tokenize = False)
    result = apply_chat_template_to_dataset(
        {
            "dataset": Dataset.from_dict({"messages": [_CONVERSATION]}),
            "final_format": "chatml_messages",
            "chat_column": "messages",
            "is_standardized": True,
        },
        tokenizer,
        model_name = name,
        num_proc = 1,
    )
    assert result["success"], result.get("errors")
    assert result["dataset"][0]["text"] == expected


def test_phi4_keeps_the_phi4_template():
    assert MODEL_TO_TEMPLATE_MAPPER["unsloth/phi-4"] == "phi-4"
    assert MODEL_TO_TEMPLATE_MAPPER["unsloth/phi-4-reasoning"] == "phi-4"
