# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The official openai/gpt-oss-20b chat_template.jinja emits "\\n\\n" after the
developer message before <|end|>, whether or not tools are present."""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    # get_chat_template writes a scratch dir under CWD; isolate it per test and xdist worker.
    monkeypatch.chdir(tmp_path)


def _tokenizer(name):
    vocab = {"<unk>": 0, "<|startoftext|>": 1, "<|return|>": 2, "<pad>": 3}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<|startoftext|>",
        eos_token = "<|return|>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )
    return get_chat_template(tokenizer, name)


# gpt-oss and gptoss share one template object.
TEMPLATES = ["gpt-oss", "gptoss"]

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

DEVELOPER_BLOCK = "<|start|>developer<|message|># Instructions\n\nBe helpful.\n\n<|end|>"


@pytest.mark.parametrize("name", TEMPLATES)
@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        [
            {"role": "developer", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
    ],
    ids = ["system", "developer"],
)
def test_developer_message_has_blank_lines_before_end(name, messages):
    tokenizer = _tokenizer(name)
    out = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
    assert DEVELOPER_BLOCK in out
    assert "Be helpful.<|end|>" not in out


@pytest.mark.parametrize("name", TEMPLATES)
def test_developer_message_with_tools_still_has_blank_lines(name):
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    tokenizer = _tokenizer(name)
    out = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = False, tools = TOOLS
    )
    # Official renders '# Instructions\n\n{dev}\n\n# Tools\n\n' when both are present.
    assert "# Instructions\n\nBe helpful.\n\n# Tools\n\n" in out


@pytest.mark.parametrize("name", TEMPLATES)
def test_plain_conversation_renders(name):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    tokenizer = _tokenizer(name)
    out = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
    assert "<|start|>developer<|message|>" not in out
    assert out.endswith("<|return|>")
