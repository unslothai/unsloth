# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _render(name, messages):
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    tokenizer = get_chat_template(
        PreTrainedTokenizerFast(
            tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
            bos_token = "<s>",
            eos_token = "</s>",
            unk_token = "<unk>",
            pad_token = "<pad>",
        ),
        name,
    )
    return tokenizer.apply_chat_template(messages, tokenize = False)


CALL = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
ASK = {"role": "user", "content": "Weather?"}
RESULT = {"role": "tool", "content": "sunny"}
ANSWER = {"role": "assistant", "content": "It is sunny."}


def _conversations(call_turn):
    return {
        "tool_flow": [ASK, call_turn, RESULT, ANSWER],
        "ends_on_call": [ASK, call_turn],
    }


@pytest.mark.parametrize("name", ["qwen3", "qwen-3"])
@pytest.mark.parametrize("flow", ["tool_flow", "ends_on_call"])
@pytest.mark.parametrize("missing", [False, True], ids = ["null_content", "no_content_key"])
def test_tool_call_turn_without_content_renders_like_empty_content(name, flow, missing):
    call_turn = {"role": "assistant", "tool_calls": [CALL]}
    if not missing:
        call_turn["content"] = None
    empty = {"role": "assistant", "content": "", "tool_calls": [CALL]}
    text = _render(name, _conversations(call_turn)[flow])
    assert text == _render(name, _conversations(empty)[flow])
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}' in text


def test_tool_flow_output_is_unchanged():
    empty = {"role": "assistant", "content": "", "tool_calls": [CALL]}
    assert _render("qwen3", _conversations(empty)["tool_flow"]) == (
        "<|im_start|>user\nWeather?<|im_end|>\n"
        "<|im_start|>assistant\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        "<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nIt is sunny.<|im_end|>\n"
    )
