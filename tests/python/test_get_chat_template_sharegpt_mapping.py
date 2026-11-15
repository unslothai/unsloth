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


CONTENT_ONLY = {"role": "role", "content": "text", "user": "user", "assistant": "assistant"}
ROLE_VALUE_ONLY = {"role": "role", "content": "content", "user": "human", "assistant": "gpt"}
ROLE_CONTENT = [{"role": role, "content": text} for role, text in CONVERSATION]


def _render_mapped(
    mapping,
    messages,
    name = "chatml",
):
    return _render(get_chat_template(_tokenizer(), name, mapping = mapping), messages, False)


def _render_plain(messages):
    return _render(get_chat_template(_tokenizer(), "chatml"), messages, False)


def test_content_only_mapping_keeps_message_text():
    messages = [{"role": role, "text": text} for role, text in CONVERSATION]

    assert _render_mapped(CONTENT_ONLY, messages) == _render_plain(ROLE_CONTENT)


def test_content_only_mapping_after_a_native_system_turn():
    messages = [ROLE_CONTENT[0]] + [{"role": role, "text": text} for role, text in CONVERSATION[1:]]

    assert _render_mapped(CONTENT_ONLY, messages) == _render_plain(ROLE_CONTENT)


def test_sharegpt_mapping_after_a_native_system_turn():
    messages = [ROLE_CONTENT[0]] + [
        {"from": SHAREGPT_ROLES[r], "value": t} for r, t in CONVERSATION[1:]
    ]

    assert _render_mapped(SHAREGPT, messages) == _render_plain(ROLE_CONTENT)


def test_role_value_only_mapping_renders_template_roles():
    messages = [{"role": SHAREGPT_ROLES[role], "content": text} for role, text in CONVERSATION]

    assert _render_mapped(ROLE_VALUE_ONLY, messages) == _render_plain(ROLE_CONTENT)


def test_mappings_leave_native_turns_unchanged():
    for mapping in (SHAREGPT, CONTENT_ONLY, ROLE_VALUE_ONLY):
        assert _render_mapped(mapping, ROLE_CONTENT) == _render_plain(ROLE_CONTENT)


def test_sharegpt_mapping_ignores_null_columns_from_mixed_rows():
    native = [dict(turn, **{"from": None, "value": None}) for turn in ROLE_CONTENT]
    sharegpt = [
        {"from": SHAREGPT_ROLES[role], "value": text, "role": None, "content": None}
        for role, text in CONVERSATION
    ]

    assert _render_mapped(SHAREGPT, native) == _render_plain(ROLE_CONTENT)
    assert _render_mapped(SHAREGPT, sharegpt) == _render_plain(ROLE_CONTENT)


@pytest.mark.parametrize("name", ["llama-3", "llama-3.1", "qwen3-instruct"])
def test_sharegpt_tool_call_with_null_value_renders_like_null_content(name):
    call = [{"type": "function", "function": {"name": "f", "arguments": {}}}]
    sharegpt = [{"from": "human", "value": "q"}, {"from": "gpt", "value": None, "tool_calls": call}]
    native = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": call},
    ]

    expected = _render(get_chat_template(_tokenizer(), name), native, False)

    assert _render_mapped(SHAREGPT, sharegpt, name) == expected


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


@pytest.mark.parametrize("name", ["llama-3.1", "chatml"])
def test_mapping_that_keeps_the_role_key_still_renames_role_values(name):
    mapping = {"role": "role", "content": "content", "user": "human", "assistant": "gpt"}
    renamed = [{"role": SHAREGPT_ROLES[role], "content": text} for role, text in CONVERSATION]
    role_content = [{"role": role, "content": text} for role, text in CONVERSATION]

    expected = _render(get_chat_template(_tokenizer(), name), role_content, True)
    mapped = get_chat_template(_tokenizer(), name, mapping = mapping)

    assert _render(mapped, renamed, True) == expected
    assert _render(mapped, role_content, True) == expected


@pytest.mark.parametrize("name", ["llama-3.1", "chatml"])
def test_mapping_that_only_renames_the_content_key(name):
    mapping = {"role": "role", "content": "text", "user": "user", "assistant": "assistant"}
    renamed = [{"role": role, "text": text} for role, text in CONVERSATION]
    role_content = [{"role": role, "content": text} for role, text in CONVERSATION]

    expected = _render(get_chat_template(_tokenizer(), name), role_content, True)
    mapped = get_chat_template(_tokenizer(), name, mapping = mapping)

    assert _render(mapped, renamed, True) == expected
    assert _render(mapped, role_content, True) == expected


@pytest.mark.parametrize("name", ["gpt-oss", "qwen3-instruct"])
def test_sharegpt_mapping_with_content_less_tool_call_first(name):
    call = [{"type": "function", "function": {"name": "f", "arguments": {}}}]
    sharegpt = [{"from": "gpt", "tool_calls": call}, {"from": "human", "value": "ok"}]
    role_content = [{"role": "assistant", "tool_calls": call}, {"role": "user", "content": "ok"}]

    expected = _render(get_chat_template(_tokenizer(), name), role_content, False)
    mapped = get_chat_template(_tokenizer(), name, mapping = SHAREGPT)

    assert "ok" in expected
    assert _render(mapped, sharegpt, False) == expected


@pytest.mark.parametrize("name", ["llama-3.1", "qwen3-instruct"])
def test_content_only_mapping_keeps_canonical_tool_call_first_history(name):
    call = [{"type": "function", "function": {"name": "f", "arguments": {}}}]
    canonical = [
        {"role": "assistant", "tool_calls": call},
        {"role": "tool", "content": "42"},
        {"role": "user", "content": "ok"},
    ]
    mapping = {"role": "role", "content": "text", "user": "user", "assistant": "assistant"}

    expected = _render(get_chat_template(_tokenizer(), name), canonical, False)
    mapped = get_chat_template(_tokenizer(), name, mapping = mapping)

    assert "42" in expected and "ok" in expected
    assert _render(mapped, canonical, False) == expected
