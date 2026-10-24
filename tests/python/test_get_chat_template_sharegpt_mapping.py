# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


SHAREGPT = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
SHAREGPT_ROLES = {"system": "system", "user": "human", "assistant": "gpt"}
CONVERSATION = [
    ("system", "Be brief."),
    ("user", "Hi"),
    ("assistant", "Hello!"),
    ("user", "Bye"),
]
TEMPLATES = [
    "llama-3.1",
    "qwen-2.5",
    "qwen3-instruct",
    "gemma-3",
    "gemma-4",
    "gpt-oss",
    "llama-3",
    "chatml",
    "mistral",
    "phi-3",
]


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    # get_chat_template writes a scratch dir under CWD; isolate it per test and xdist worker.
    monkeypatch.chdir(tmp_path)


def _tokenizer():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    return PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<s>",
        eos_token = "</s>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )


def _render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = add_generation_prompt
    )


@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("name", TEMPLATES)
def test_sharegpt_mapping_renders_like_role_content(name, add_generation_prompt):
    sharegpt = [{"from": SHAREGPT_ROLES[role], "value": text} for role, text in CONVERSATION]
    role_content = [{"role": role, "content": text} for role, text in CONVERSATION]

    expected = _render(get_chat_template(_tokenizer(), name), role_content, add_generation_prompt)
    mapped = get_chat_template(_tokenizer(), name, mapping = SHAREGPT)

    assert "Hi" in expected and "Hello!" in expected
    assert _render(mapped, sharegpt, add_generation_prompt) == expected
    assert _render(mapped, role_content, add_generation_prompt) == expected


def test_sharegpt_mapping_keeps_extra_message_keys():
    thinking = {"thinking": "Let me think."}
    conversation = CONVERSATION[:3]
    sharegpt = [
        dict(
            {"from": SHAREGPT_ROLES[role], "value": text},
            **(thinking if role == "assistant" else {}),
        )
        for role, text in conversation
    ]
    role_content = [
        dict({"role": role, "content": text}, **(thinking if role == "assistant" else {}))
        for role, text in conversation
    ]

    expected = _render(get_chat_template(_tokenizer(), "gpt-oss"), role_content, False)
    mapped = get_chat_template(_tokenizer(), "gpt-oss", mapping = SHAREGPT)

    assert "Let me think." in expected
    assert _render(mapped, sharegpt, False) == expected


def test_custom_template_reading_sharegpt_keys_keeps_working():
    template = (
        "{{ bos_token }}{% for message in messages %}"
        "{% if message['from'] == 'human' %}{{ 'User: ' + message['value'] + '\n' }}"
        "{% elif message['from'] == 'gpt' %}{{ 'Bot: ' + message['value'] + eos_token + '\n' }}"
        "{% endif %}{% endfor %}"
    )
    sharegpt = [{"from": SHAREGPT_ROLES[role], "value": text} for role, text in CONVERSATION]
    mapped = get_chat_template(_tokenizer(), (template, "eos_token"), mapping = SHAREGPT)

    assert _render(mapped, sharegpt, False) == "<s>User: Hi\nBot: Hello!</s>\nUser: Bye\n"
