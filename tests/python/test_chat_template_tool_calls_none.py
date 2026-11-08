# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""datasets pads plain messages with `tool_calls: None`; templates must test the value."""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


TEMPLATES = ["llama-3.1", "llama-3.2", "llama-3.3", "gpt-oss"]
CALL = [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
PLAIN = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Bye"},
    {"role": "assistant", "content": "Bye!"},
]
TOOL_FLOW = [
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": None, "tool_calls": CALL},
    {"role": "tool", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny."},
]


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    # get_chat_template writes a scratch dir under CWD; isolate it per test and xdist worker.
    monkeypatch.chdir(tmp_path)


def _tokenizer(name):
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<s>",
        eos_token = "</s>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )
    return get_chat_template(tokenizer, name)


def _render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = add_generation_prompt
    )


@pytest.mark.parametrize("filler", [None, []], ids = ["none", "empty_list"])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("conversation", [PLAIN, TOOL_FLOW], ids = ["plain", "tool_flow"])
@pytest.mark.parametrize("name", TEMPLATES)
def test_blank_tool_calls_match_a_missing_key(name, conversation, add_generation_prompt, filler):
    tokenizer = _tokenizer(name)
    padded = [dict(m, tool_calls = m.get("tool_calls", filler)) for m in conversation]

    expected = _render(tokenizer, conversation, add_generation_prompt)

    assert ("get_weather" in expected) == (conversation is TOOL_FLOW)
    assert _render(tokenizer, padded, add_generation_prompt) == expected


def test_gpt_oss_drops_the_tool_call_analysis_when_a_final_answer_follows():
    conversation = [
        TOOL_FLOW[0],
        {"role": "assistant", "content": None, "thinking": "Check it.", "tool_calls": CALL},
        TOOL_FLOW[2],
        TOOL_FLOW[3],
    ]
    padded = [dict(m, tool_calls = m.get("tool_calls")) for m in conversation]
    tokenizer = _tokenizer("gpt-oss")

    expected = _render(tokenizer, conversation, False)

    assert "Check it." not in expected
    assert _render(tokenizer, padded, False) == expected
