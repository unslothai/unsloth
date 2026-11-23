# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expected strings rendered by the official LiquidAI/LFM2-1.2B
chat_template.jinja: a "List of tools" system block, <|tool_call_start|> calls and
<|tool_response_start|> tool results, hard coded for offline runs."""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    # get_chat_template writes a scratch dir under CWD; isolate it per test and xdist worker.
    monkeypatch.chdir(tmp_path)


def _tokenizer(name):
    vocab = {"<unk>": 0, "<|startoftext|>": 1, "<|im_end|>": 2, "<pad>": 3}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<|startoftext|>",
        eos_token = "<|im_end|>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )
    return get_chat_template(tokenizer, name)


# Only lfm-2 takes the official LFM2 tool template; lfm-2.5 keeps its own render
# (LiquidAI/LFM2.5 templates do not use the LFM2 tool markers).
TEMPLATES = ["lfm-2"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

TOOL_FLOW = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    },
    {"role": "tool", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny."},
]

PLAIN = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]


@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("name", TEMPLATES)
def test_tool_flow_matches_official(name, add_generation_prompt):
    tokenizer = _tokenizer(name)
    out = tokenizer.apply_chat_template(
        TOOL_FLOW,
        tokenize = False,
        add_generation_prompt = add_generation_prompt,
        tools = TOOLS,
    )
    expected = (
        "<|startoftext|>"
        "<|im_start|>system\nList of tools: <|tool_list_start|>"
        '[{"type": "function", "function": {"name": "get_weather", "description": "Get the weather.", '
        '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, '
        '"required": ["city"]}}}]<|tool_list_end|><|im_end|>\n'
        "<|im_start|>user\nweather?<|im_end|>\n"
        "<|im_start|>assistant\n<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|><|im_end|>\n"
        "<|im_start|>tool\n<|tool_response_start|>sunny<|tool_response_end|><|im_end|>\n"
        "<|im_start|>assistant\nIt is sunny.<|im_end|>\n"
    )
    if add_generation_prompt:
        expected += "<|im_start|>assistant\n"
    assert out == expected


@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("name", TEMPLATES)
def test_plain_conversation_unchanged(name, add_generation_prompt):
    tokenizer = _tokenizer(name)
    out = tokenizer.apply_chat_template(
        PLAIN, tokenize = False, add_generation_prompt = add_generation_prompt
    )
    expected = (
        "<|startoftext|>"
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello!<|im_end|>\n"
    )
    if add_generation_prompt:
        expected += "<|im_start|>assistant\n"
    assert out == expected


def test_lfm25_render_is_unchanged():
    tokenizer = _tokenizer("lfm-2.5")
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "tool", "content": "T"},
    ]
    assert tokenizer.apply_chat_template(messages, tokenize = False) == (
        "<|startoftext|><|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello!<|im_end|>\n"
        "<|im_start|>tool\nT<|im_end|>\n"
    )
