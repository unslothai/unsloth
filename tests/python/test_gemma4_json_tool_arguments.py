# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import (
    CHAT_TEMPLATES,
    gemma4_empty_thought_template,
    get_chat_template,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
TRAINED_CALL = 'call:get_weather{city:<|"|>Paris<|"|>}'


def _tokenizer():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    return PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<s>",
        eos_token = "</s>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )


def _messages(arguments):
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": arguments},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]


def _render(template_name, arguments):
    tokenizer = get_chat_template(_tokenizer(), template_name)
    return tokenizer.apply_chat_template(
        _messages(arguments),
        tools = TOOLS,
        tokenize = False,
        add_generation_prompt = False,
    )


@pytest.mark.parametrize("template_name", ["gemma-4", "gemma4", "gemma-4-thinking", "gemma4-thinking"])
def test_json_string_arguments_match_the_trained_dict_form(template_name):
    string_render = _render(template_name, '{"city": "Paris"}')
    dict_render = _render(template_name, {"city": "Paris"})

    assert string_render == dict_render
    assert TRAINED_CALL in string_render
    assert '{"city"' not in string_render


def test_empty_thought_template_uses_the_same_argument_form():
    tokenizer = _tokenizer()
    tokenizer.chat_template = gemma4_empty_thought_template
    rendered = tokenizer.apply_chat_template(
        _messages('{"city": "Paris"}'),
        tools = TOOLS,
        tokenize = False,
        add_generation_prompt = False,
    )
    assert TRAINED_CALL in rendered
    assert gemma4_empty_thought_template != CHAT_TEMPLATES["gemma-4-thinking"][0]


@pytest.mark.parametrize(
    "arguments",
    [
        "not-json",
        '{"a": "\\u12.4"}',
        '{"a": "\\u12GG"}',
        '{"a": "\\uZZZZ"}',
        '{"a": "Par\nis"}',
        '{"ci\ty": "Paris"}',
        "\u00a0{\"a\": 1}",
        "{\"a\": 1}\x0b",
    ],
)
def test_invalid_json_arguments_fail_instead_of_being_pasted(arguments):
    with pytest.raises(Exception, match = "Gemma 4 tool call arguments must be a JSON object"):
        _render("gemma-4", arguments)
